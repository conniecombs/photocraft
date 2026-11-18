//! Local ONNX model registry (`ModelStore`), SHA-256 integrity verification, local folder
//! discovery, in-memory installation, and background HTTPS downloads.

use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::MlError;
use crate::catalog::{ArtifactRole, BUILTIN_CATALOG, CatalogEntry, MattingPreprocess, ModelTask, catalog_entry};
use crate::onnx::{Model, validate_external_location};

/// Lowercase hexadecimal SHA-256 digest of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    encode_hex(&digest)
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

/// Verify `bytes` against `expected_hex` (case-insensitive 64-char SHA-256 hex).
pub fn verify_sha256(bytes: &[u8], expected_hex: &str) -> Result<(), MlError> {
    let actual = sha256_hex(bytes);
    if !actual.eq_ignore_ascii_case(expected_hex.trim()) {
        return Err(MlError::ChecksumMismatch { expected: expected_hex.to_ascii_lowercase(), actual });
    }
    Ok(())
}

/// Validate a model identifier before using it as a directory/file stem.
pub fn validate_model_id(id: &str) -> Result<&str, MlError> {
    let trimmed = id.trim();
    if trimmed.is_empty()
        || trimmed.len() > 128
        || trimmed == "."
        || trimmed == ".."
        || trimmed.contains('/')
        || trimmed.contains('\\')
        || trimmed.contains(':')
        || trimmed.contains('\0')
    {
        return Err(MlError::InvalidModel(format!("invalid model identifier `{id}`")));
    }
    Ok(trimmed)
}

/// Loaded, validated model payload cached in memory for zero-overhead repeated inference.
#[derive(Clone, Debug)]
pub enum LoadedModel {
    /// Single-graph dense vision model (MODNet, BiRefNet, Sky, or custom `.onnx`).
    Single { id: String, task: ModelTask, preprocess: MattingPreprocess, model: Arc<Model> },
    /// Split encoder + prompt/mask decoder interactive segmentation model (SAM 2.1).
    Sam2 { id: String, encoder: Arc<Model>, decoder: Arc<Model> },
}

/// Origin of an installed model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ModelSource {
    Catalog,
    Folder,
    Memory,
}

/// Current state of an asynchronous download job.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "state", content = "detail")]
pub enum DownloadState {
    Downloading,
    Completed,
    Cancelled,
    Failed(String),
}

/// Serializable snapshot of a download job for UI and `ml.list`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadSnapshot {
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub status: DownloadState,
}

/// Summary row returned by [`ModelStore::list_status`] and the `ml.list` engine command.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelStatus {
    pub id: String,
    pub name: String,
    pub summary: String,
    pub task: ModelTask,
    pub license: String,
    pub size_bytes: u64,
    pub installed: bool,
    pub source: Option<ModelSource>,
    pub download: Option<DownloadSnapshot>,
}

/// Pluggable HTTPS stream fetcher (allows deterministic offline testing of downloads, SHA-256
/// mismatches and cancellation).
pub trait DownloadTransport: Send + Sync {
    /// Stream `url` into `on_chunk`, returning `Err(MlError::Cancelled)` if `on_chunk` signals stop.
    fn fetch_stream(&self, url: &str, on_chunk: &mut dyn FnMut(&[u8]) -> Result<(), MlError>) -> Result<(), MlError>;
}

/// Default HTTPS transport backed by `ureq` with `rustls` and `User-Agent: Photocraft-dev`.
#[derive(Clone, Copy, Debug, Default)]
pub struct HttpsTransport;

impl DownloadTransport for HttpsTransport {
    #[cfg(not(target_arch = "wasm32"))]
    fn fetch_stream(&self, url: &str, on_chunk: &mut dyn FnMut(&[u8]) -> Result<(), MlError>) -> Result<(), MlError> {
        if !url.starts_with("https://") {
            return Err(MlError::Download(format!("refusing non-HTTPS model URL `{url}`")));
        }
        let config = ureq::Agent::config_builder()
            .timeout_connect(Some(std::time::Duration::from_secs(30)))
            .timeout_recv_response(Some(std::time::Duration::from_secs(60)))
            .user_agent("Photocraft-dev")
            .build();
        let agent: ureq::Agent = config.into();
        let mut response = agent.get(url).call().map_err(|e| MlError::Download(format!("HTTPS GET failed: {e}")))?;
        let mut reader = response.body_mut().as_reader();
        let mut buf = vec![0u8; 64 * 1024];
        loop {
            let n = reader.read(&mut buf).map_err(|e| MlError::Download(format!("stream read error: {e}")))?;
            if n == 0 {
                break;
            }
            on_chunk(&buf[..n])?;
        }
        Ok(())
    }

    #[cfg(target_arch = "wasm32")]
    fn fetch_stream(&self, _url: &str, _on_chunk: &mut dyn FnMut(&[u8]) -> Result<(), MlError>) -> Result<(), MlError> {
        Err(MlError::Download("direct background disk downloads require the native desktop or CLI build".into()))
    }
}

struct ActiveDownload {
    downloaded: AtomicU64,
    total: u64,
    cancel: AtomicBool,
    state: Mutex<DownloadState>,
}

impl ActiveDownload {
    fn new(total: u64) -> Arc<Self> {
        Arc::new(Self { downloaded: AtomicU64::new(0), total, cancel: AtomicBool::new(false), state: Mutex::new(DownloadState::Downloading) })
    }

    fn snapshot(&self) -> DownloadSnapshot {
        DownloadSnapshot {
            downloaded_bytes: self.downloaded.load(Ordering::Acquire),
            total_bytes: self.total,
            status: self.state.lock().unwrap_or_else(PoisonError::into_inner).clone(),
        }
    }

    fn is_running(&self) -> bool {
        matches!(*self.state.lock().unwrap_or_else(PoisonError::into_inner), DownloadState::Downloading)
    }
}

#[derive(Clone)]
struct InstalledRecord {
    id: String,
    name: String,
    summary: String,
    task: ModelTask,
    priority: u8,
    license: String,
    size_bytes: u64,
    source: ModelSource,
    disk_dir: Option<PathBuf>,
    single_file: Option<PathBuf>,
    loaded: Option<Arc<LoadedModel>>,
}

struct StoreInner {
    models_dir: Option<PathBuf>,
    installed: HashMap<String, InstalledRecord>,
    downloads: HashMap<String, Arc<ActiveDownload>>,
    transport: Arc<dyn DownloadTransport>,
}

/// Thread-safe registry of catalog, folder-discovered and in-memory ONNX models.
#[derive(Clone)]
pub struct ModelStore {
    inner: Arc<Mutex<StoreInner>>,
}

impl Default for ModelStore {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ModelStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let guard = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        f.debug_struct("ModelStore").field("models_dir", &guard.models_dir).field("installed_count", &guard.installed.len()).finish()
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn read_external_slice(base_dir: &Path, loc: &str, offset: u64, length: Option<u64>) -> Result<Vec<u8>, MlError> {
    let safe_name = validate_external_location(loc)?;
    let path = base_dir.join(safe_name);
    let mut file = std::fs::File::open(&path).map_err(|e| MlError::Io(format!("failed to open external weights `{}`: {e}", path.display())))?;
    let meta_len = file.metadata().map_err(|e| MlError::Io(e.to_string()))?.len();
    if offset > meta_len {
        return Err(MlError::InvalidModel(format!("external offset {offset} exceeds file length {meta_len}")));
    }
    let read_len_u64 = match length {
        Some(l) => {
            if offset.saturating_add(l) > meta_len {
                return Err(MlError::InvalidModel(format!("external slice {offset}+{l} exceeds file length {meta_len}")));
            }
            l
        }
        None => meta_len - offset,
    };
    let read_len = usize::try_from(read_len_u64).map_err(|_| MlError::MemoryLimitExceeded { needed: usize::MAX, limit: crate::tensor::MAX_TENSOR_BYTES })?;
    if read_len > crate::onnx::MAX_MODEL_BYTES {
        return Err(MlError::MemoryLimitExceeded { needed: read_len, limit: crate::onnx::MAX_MODEL_BYTES });
    }
    file.seek(SeekFrom::Start(offset)).map_err(|e| MlError::Io(e.to_string()))?;
    let mut buf = Vec::new();
    buf.try_reserve_exact(read_len).map_err(|_| MlError::MemoryLimitExceeded { needed: read_len, limit: crate::onnx::MAX_MODEL_BYTES })?;
    buf.resize(read_len, 0);
    file.read_exact(&mut buf).map_err(|e| MlError::Io(e.to_string()))?;
    Ok(buf)
}

#[cfg(not(target_arch = "wasm32"))]
fn verify_file_sha256(path: &Path, expected_hex: &str) -> Result<u64, MlError> {
    let mut f = std::fs::File::open(path).map_err(|e| MlError::Io(format!("cannot open `{}`: {e}", path.display())))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    let mut total = 0u64;
    loop {
        let n = f.read(&mut buf).map_err(|e| MlError::Io(e.to_string()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total += n as u64;
    }
    let actual = encode_hex(&hasher.finalize());
    if !actual.eq_ignore_ascii_case(expected_hex.trim()) {
        return Err(MlError::ChecksumMismatch { expected: expected_hex.to_ascii_lowercase(), actual });
    }
    Ok(total)
}

#[cfg(not(target_arch = "wasm32"))]
fn load_catalog_from_dir(entry: &CatalogEntry, dir: &Path) -> Result<LoadedModel, MlError> {
    for art in entry.artifacts {
        let p = dir.join(art.file_name);
        verify_file_sha256(&p, art.sha256)?;
    }
    match entry.task {
        ModelTask::Interactive => {
            let enc_art = entry
                .artifacts
                .iter()
                .find(|a| a.role == ArtifactRole::EncoderGraph)
                .ok_or_else(|| MlError::InvalidModel("missing encoder artifact".into()))?;
            let dec_art = entry
                .artifacts
                .iter()
                .find(|a| a.role == ArtifactRole::DecoderGraph)
                .ok_or_else(|| MlError::InvalidModel("missing decoder artifact".into()))?;
            let enc_bytes = std::fs::read(dir.join(enc_art.file_name)).map_err(|e| MlError::Io(e.to_string()))?;
            let dec_bytes = std::fs::read(dir.join(dec_art.file_name)).map_err(|e| MlError::Io(e.to_string()))?;
            let enc = Model::from_bytes_with_external(&enc_bytes, &|loc, off, len| read_external_slice(dir, loc, off, len))?;
            let dec = Model::from_bytes_with_external(&dec_bytes, &|loc, off, len| read_external_slice(dir, loc, off, len))?;
            Ok(LoadedModel::Sam2 { id: entry.id.to_string(), encoder: Arc::new(enc), decoder: Arc::new(dec) })
        }
        _ => {
            let prim = entry
                .artifacts
                .iter()
                .find(|a| a.role == ArtifactRole::PrimaryGraph)
                .ok_or_else(|| MlError::InvalidModel("missing primary artifact".into()))?;
            let bytes = std::fs::read(dir.join(prim.file_name)).map_err(|e| MlError::Io(e.to_string()))?;
            let model = Model::from_bytes_with_external(&bytes, &|loc, off, len| read_external_slice(dir, loc, off, len))?;
            Ok(LoadedModel::Single {
                id: entry.id.to_string(),
                task: entry.task,
                preprocess: entry.preprocess.unwrap_or(MattingPreprocess::DEFAULT_CUSTOM),
                model: Arc::new(model),
            })
        }
    }
}

fn infer_folder_task(stem: &str) -> ModelTask {
    let lower = stem.to_ascii_lowercase();
    if lower.contains("sky") {
        ModelTask::Sky
    } else if lower.contains("modnet") || lower.contains("birefnet") || lower.contains("matting") || lower.contains("subject") || lower.contains("rmbg") {
        ModelTask::Subject
    } else {
        ModelTask::Custom
    }
}

impl ModelStore {
    /// Create an empty in-memory [`ModelStore`] using [`HttpsTransport`].
    pub fn new() -> Self {
        Self::with_transport(Arc::new(HttpsTransport))
    }

    /// Create a [`ModelStore`] with a custom [`DownloadTransport`] (used by offline tests).
    pub fn with_transport(transport: Arc<dyn DownloadTransport>) -> Self {
        Self { inner: Arc::new(Mutex::new(StoreInner { models_dir: None, installed: HashMap::new(), downloads: HashMap::new(), transport })) }
    }

    /// Current configured models directory, if any.
    pub fn models_dir(&self) -> Option<PathBuf> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner).models_dir.clone()
    }

    /// Configure the on-disk models folder and scan it for installed catalog or custom `.onnx`
    /// models. In-memory models installed via [`Self::install_bytes`] are preserved.
    pub fn set_models_dir(&self, dir: Option<PathBuf>) -> Result<usize, MlError> {
        {
            let mut guard = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
            if guard.models_dir == dir {
                return Ok(guard.installed.len());
            }
            guard.models_dir = dir;
        }
        self.reload()
    }

    /// Re-scan the configured models folder (retaining any `ModelSource::Memory` entries).
    pub fn reload(&self) -> Result<usize, MlError> {
        let dir_opt = {
            let mut guard = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
            guard.installed.retain(|_, rec| rec.source == ModelSource::Memory);
            guard.models_dir.clone()
        };

        #[cfg(not(target_arch = "wasm32"))]
        if let Some(root) = dir_opt
            && root.is_dir()
        {
            let mut discovered: Vec<InstalledRecord> = Vec::new();
            for entry in BUILTIN_CATALOG {
                let sub = root.join(entry.id);
                if sub.is_dir() && entry.artifacts.iter().all(|a| sub.join(a.file_name).is_file()) {
                    discovered.push(InstalledRecord {
                        id: entry.id.to_string(),
                        name: entry.name.to_string(),
                        summary: entry.summary.to_string(),
                        task: entry.task,
                        priority: entry.priority,
                        license: entry.license.to_string(),
                        size_bytes: entry.total_bytes(),
                        source: ModelSource::Catalog,
                        disk_dir: Some(sub),
                        single_file: None,
                        loaded: None,
                    });
                }
            }

            if let Ok(rd) = std::fs::read_dir(&root) {
                for item in rd.flatten() {
                    let path = item.path();
                    if path.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("onnx"))
                        && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
                        && validate_model_id(stem).is_ok()
                        && catalog_entry(stem).is_none()
                    {
                        let size = item.metadata().map(|m| m.len()).unwrap_or(0);
                        let task = infer_folder_task(stem);
                        discovered.push(InstalledRecord {
                            id: stem.to_string(),
                            name: stem.to_string(),
                            summary: format!("Local ONNX model ({})", path.file_name().and_then(|n| n.to_str()).unwrap_or(stem)),
                            task,
                            priority: 15,
                            license: "Local".into(),
                            size_bytes: size,
                            source: ModelSource::Folder,
                            disk_dir: Some(root.clone()),
                            single_file: Some(path),
                            loaded: None,
                        });
                    }
                }
            }

            let mut guard = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
            for rec in discovered {
                guard.installed.entry(rec.id.clone()).or_insert(rec);
            }
        }

        #[cfg(target_arch = "wasm32")]
        let _ = dir_opt;

        Ok(self.inner.lock().unwrap_or_else(PoisonError::into_inner).installed.len())
    }

    /// Install a single-graph `.onnx` model directly from memory (used by MCP `ml.install`, wasm
    /// sessions and tests).
    pub fn install_bytes(&self, id: &str, task: ModelTask, preprocess: Option<MattingPreprocess>, bytes: &[u8]) -> Result<ModelStatus, MlError> {
        let clean_id = validate_model_id(id)?.to_string();
        let parsed = Model::from_bytes(bytes)?;
        let prep = preprocess.or_else(|| catalog_entry(&clean_id).and_then(|c| c.preprocess)).unwrap_or(MattingPreprocess::DEFAULT_CUSTOM);
        let loaded = Arc::new(LoadedModel::Single { id: clean_id.clone(), task, preprocess: prep, model: Arc::new(parsed) });

        let rec = InstalledRecord {
            id: clean_id.clone(),
            name: catalog_entry(&clean_id).map_or_else(|| clean_id.clone(), |c| c.name.to_string()),
            summary: format!("In-memory ONNX model ({} bytes)", bytes.len()),
            task,
            priority: 25,
            license: catalog_entry(&clean_id).map_or_else(|| "Custom".into(), |c| c.license.to_string()),
            size_bytes: bytes.len() as u64,
            source: ModelSource::Memory,
            disk_dir: None,
            single_file: None,
            loaded: Some(loaded),
        };
        let status = ModelStatus {
            id: rec.id.clone(),
            name: rec.name.clone(),
            summary: rec.summary.clone(),
            task: rec.task,
            license: rec.license.clone(),
            size_bytes: rec.size_bytes,
            installed: true,
            source: Some(ModelSource::Memory),
            download: None,
        };
        self.inner.lock().unwrap_or_else(PoisonError::into_inner).installed.insert(clean_id, rec);
        Ok(status)
    }

    /// Install a split encoder + decoder SAM-2 model directly from memory.
    pub fn install_sam2_models(&self, id: &str, encoder: Model, decoder: Model) -> Result<ModelStatus, MlError> {
        let clean_id = validate_model_id(id)?.to_string();
        let loaded = Arc::new(LoadedModel::Sam2 { id: clean_id.clone(), encoder: Arc::new(encoder), decoder: Arc::new(decoder) });
        let rec = InstalledRecord {
            id: clean_id.clone(),
            name: catalog_entry(&clean_id).map_or_else(|| clean_id.clone(), |c| c.name.to_string()),
            summary: "In-memory SAM-2 encoder + prompt/mask decoder".into(),
            task: ModelTask::Interactive,
            priority: 25,
            license: catalog_entry(&clean_id).map_or_else(|| "Apache-2.0".into(), |c| c.license.to_string()),
            size_bytes: 0,
            source: ModelSource::Memory,
            disk_dir: None,
            single_file: None,
            loaded: Some(loaded),
        };
        let status = ModelStatus {
            id: rec.id.clone(),
            name: rec.name.clone(),
            summary: rec.summary.clone(),
            task: ModelTask::Interactive,
            license: rec.license.clone(),
            size_bytes: 0,
            installed: true,
            source: Some(ModelSource::Memory),
            download: None,
        };
        self.inner.lock().unwrap_or_else(PoisonError::into_inner).installed.insert(clean_id, rec);
        Ok(status)
    }

    /// Install a `.onnx` file from disk (validating its graph and resolving any sibling
    /// `.onnx_data` weights).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn install_path(&self, path: &Path, id_override: Option<&str>, task_override: Option<ModelTask>) -> Result<ModelStatus, MlError> {
        let stem =
            id_override.or_else(|| path.file_stem().and_then(|s| s.to_str())).ok_or_else(|| MlError::InvalidModel("model path has no file stem".into()))?;
        let clean_id = validate_model_id(stem)?.to_string();
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let bytes = std::fs::read(path).map_err(|e| MlError::Io(format!("cannot read `{}`: {e}", path.display())))?;
        let parsed = Model::from_bytes_with_external(&bytes, &|loc, off, len| read_external_slice(parent, loc, off, len))?;
        let task = task_override.unwrap_or_else(|| infer_folder_task(&clean_id));
        let loaded = Arc::new(LoadedModel::Single { id: clean_id.clone(), task, preprocess: MattingPreprocess::DEFAULT_CUSTOM, model: Arc::new(parsed) });
        let rec = InstalledRecord {
            id: clean_id.clone(),
            name: clean_id.clone(),
            summary: format!("Installed from {}", path.file_name().and_then(|n| n.to_str()).unwrap_or(&clean_id)),
            task,
            priority: 20,
            license: "Local".into(),
            size_bytes: bytes.len() as u64,
            source: ModelSource::Folder,
            disk_dir: Some(parent.to_path_buf()),
            single_file: Some(path.to_path_buf()),
            loaded: Some(loaded),
        };
        let status = ModelStatus {
            id: rec.id.clone(),
            name: rec.name.clone(),
            summary: rec.summary.clone(),
            task: rec.task,
            license: rec.license.clone(),
            size_bytes: rec.size_bytes,
            installed: true,
            source: Some(ModelSource::Folder),
            download: None,
        };
        self.inner.lock().unwrap_or_else(PoisonError::into_inner).installed.insert(clean_id, rec);
        Ok(status)
    }

    /// List all built-in catalog models merged with any folder/in-memory models and active
    /// download progress.
    pub fn list_status(&self) -> Vec<ModelStatus> {
        let guard = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        let mut out = Vec::with_capacity(BUILTIN_CATALOG.len() + guard.installed.len());

        for entry in BUILTIN_CATALOG {
            let inst = guard.installed.get(entry.id);
            let dl = guard.downloads.get(entry.id).map(|d| d.snapshot());
            out.push(ModelStatus {
                id: entry.id.to_string(),
                name: entry.name.to_string(),
                summary: entry.summary.to_string(),
                task: entry.task,
                license: entry.license.to_string(),
                size_bytes: entry.total_bytes(),
                installed: inst.is_some(),
                source: inst.map(|r| r.source),
                download: dl,
            });
        }

        let mut extras: Vec<&InstalledRecord> = guard.installed.values().filter(|r| catalog_entry(&r.id).is_none()).collect();
        extras.sort_by(|a, b| a.id.cmp(&b.id));
        for rec in extras {
            out.push(ModelStatus {
                id: rec.id.clone(),
                name: rec.name.clone(),
                summary: rec.summary.clone(),
                task: rec.task,
                license: rec.license.clone(),
                size_bytes: rec.size_bytes,
                installed: true,
                source: Some(rec.source),
                download: guard.downloads.get(&rec.id).map(|d| d.snapshot()),
            });
        }
        out
    }

    /// Whether at least one model serving `task` is currently installed.
    pub fn has_task_model(&self, task: ModelTask) -> bool {
        let guard = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        guard.installed.values().any(|r| r.task == task)
    }

    /// Load (or fetch from RAM cache) the highest-priority installed model for `task`.
    pub fn load_for_task(&self, task: ModelTask) -> Result<Option<Arc<LoadedModel>>, MlError> {
        let best_id = {
            let guard = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
            guard.installed.values().filter(|r| r.task == task).max_by_key(|r| r.priority).map(|r| r.id.clone())
        };
        let Some(id) = best_id else {
            return Ok(None);
        };
        self.load_by_id(&id).map(Some)
    }

    /// Load (or fetch from RAM cache) a specific installed model by `id`.
    pub fn load_by_id(&self, id: &str) -> Result<Arc<LoadedModel>, MlError> {
        let rec = {
            let guard = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
            let r = guard.installed.get(id).ok_or_else(|| MlError::NotInstalled(id.to_string()))?;
            if let Some(loaded) = &r.loaded {
                return Ok(loaded.clone());
            }
            r.clone()
        };

        #[cfg(not(target_arch = "wasm32"))]
        let loaded = {
            if let Some(entry) = catalog_entry(&rec.id)
                && let Some(dir) = &rec.disk_dir
            {
                Arc::new(load_catalog_from_dir(entry, dir)?)
            } else if let Some(file) = &rec.single_file {
                let parent = file.parent().unwrap_or_else(|| Path::new("."));
                let bytes = std::fs::read(file).map_err(|e| MlError::Io(e.to_string()))?;
                let parsed = Model::from_bytes_with_external(&bytes, &|loc, off, len| read_external_slice(parent, loc, off, len))?;
                Arc::new(LoadedModel::Single { id: rec.id.clone(), task: rec.task, preprocess: MattingPreprocess::DEFAULT_CUSTOM, model: Arc::new(parsed) })
            } else {
                return Err(MlError::NotInstalled(id.to_string()));
            }
        };

        #[cfg(target_arch = "wasm32")]
        let loaded: Arc<LoadedModel> = {
            let _ = rec;
            return Err(MlError::NotInstalled(id.to_string()));
        };

        let mut guard = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(slot) = guard.installed.get_mut(id) {
            slot.loaded = Some(loaded.clone());
        }
        Ok(loaded)
    }

    /// Start downloading a built-in catalog model into `models_dir` on a background thread (or
    /// synchronously when `blocking` is `true`).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn start_download(&self, id: &str, blocking: bool) -> Result<DownloadSnapshot, MlError> {
        let entry = catalog_entry(id).ok_or_else(|| MlError::UnknownModel(id.to_string()))?;
        let (target_root, transport, job) = {
            let mut guard = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some(existing) = guard.downloads.get(entry.id)
                && existing.is_running()
            {
                return Ok(existing.snapshot());
            }
            let root = guard.models_dir.clone().ok_or_else(|| MlError::Io("models folder is not configured".into()))?;
            let job = ActiveDownload::new(entry.total_bytes());
            guard.downloads.insert(entry.id.to_string(), job.clone());
            (root, guard.transport.clone(), job)
        };

        let store_clone = self.clone();
        let job_worker = job.clone();
        let run_download = move || {
            let res = download_entry_to_disk(entry, &target_root, transport.as_ref(), &job_worker);
            let mut guard = store_clone.inner.lock().unwrap_or_else(PoisonError::into_inner);
            match res {
                Ok(sub_dir) => {
                    *job_worker.state.lock().unwrap_or_else(PoisonError::into_inner) = DownloadState::Completed;
                    guard.installed.insert(
                        entry.id.to_string(),
                        InstalledRecord {
                            id: entry.id.to_string(),
                            name: entry.name.to_string(),
                            summary: entry.summary.to_string(),
                            task: entry.task,
                            priority: entry.priority,
                            license: entry.license.to_string(),
                            size_bytes: entry.total_bytes(),
                            source: ModelSource::Catalog,
                            disk_dir: Some(sub_dir),
                            single_file: None,
                            loaded: None,
                        },
                    );
                }
                Err(MlError::Cancelled) => {
                    *job_worker.state.lock().unwrap_or_else(PoisonError::into_inner) = DownloadState::Cancelled;
                }
                Err(e) => {
                    *job_worker.state.lock().unwrap_or_else(PoisonError::into_inner) = DownloadState::Failed(e.to_string());
                }
            }
        };

        if blocking {
            run_download();
            let snap = job.snapshot();
            return match &snap.status {
                DownloadState::Failed(msg) => Err(MlError::Download(msg.clone())),
                DownloadState::Cancelled => Err(MlError::Cancelled),
                _ => Ok(snap),
            };
        }

        let thread_name = format!("pc_ml_dl_{}", entry.id);
        if let Err(e) = std::thread::Builder::new().name(thread_name).spawn(run_download) {
            *job.state.lock().unwrap_or_else(PoisonError::into_inner) = DownloadState::Failed(e.to_string());
            return Err(MlError::Download(format!("failed to spawn download thread: {e}")));
        }
        Ok(job.snapshot())
    }

    /// Stub on `wasm32` where native background filesystem downloads are unavailable.
    #[cfg(target_arch = "wasm32")]
    pub fn start_download(&self, id: &str, _blocking: bool) -> Result<DownloadSnapshot, MlError> {
        let _ = catalog_entry(id).ok_or_else(|| MlError::UnknownModel(id.to_string()))?;
        Err(MlError::Download("background model downloads require the native desktop or CLI build".into()))
    }

    /// Request cancellation of an in-progress download for `id`.
    pub fn cancel_download(&self, id: &str) -> bool {
        let guard = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(job) = guard.downloads.get(id)
            && job.is_running()
        {
            job.cancel.store(true, Ordering::Release);
            return true;
        }
        false
    }

    /// Remove an installed model from memory and (when managed inside `models_dir`) delete its
    /// files from disk.
    pub fn remove(&self, id: &str) -> Result<bool, MlError> {
        let clean_id = validate_model_id(id)?;
        let _ = self.cancel_download(clean_id);
        let (removed, models_root) = {
            let mut guard = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
            guard.downloads.remove(clean_id);
            (guard.installed.remove(clean_id), guard.models_dir.clone())
        };
        let Some(rec) = removed else {
            return Ok(false);
        };

        #[cfg(not(target_arch = "wasm32"))]
        if let Some(root) = models_root {
            if rec.source == ModelSource::Catalog {
                let sub = root.join(clean_id);
                if sub.starts_with(&root) && sub.is_dir() {
                    let _ = std::fs::remove_dir_all(&sub);
                }
            } else if let Some(file) = rec.single_file
                && file.starts_with(&root)
                && file.is_file()
            {
                let _ = std::fs::remove_file(&file);
            }
        }
        #[cfg(target_arch = "wasm32")]
        let _ = (rec, models_root);

        Ok(true)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn download_entry_to_disk(entry: &CatalogEntry, models_root: &Path, transport: &dyn DownloadTransport, job: &ActiveDownload) -> Result<PathBuf, MlError> {
    let model_dir = models_root.join(entry.id);
    std::fs::create_dir_all(&model_dir).map_err(|e| MlError::Io(format!("cannot create `{}`: {e}", model_dir.display())))?;

    for art in entry.artifacts {
        if job.cancel.load(Ordering::Acquire) {
            return Err(MlError::Cancelled);
        }
        let final_path = model_dir.join(art.file_name);
        if final_path.is_file() && verify_file_sha256(&final_path, art.sha256).is_ok() {
            job.downloaded.fetch_add(art.size_bytes, Ordering::AcqRel);
            continue;
        }

        let part_path = model_dir.join(format!("{}.part", art.file_name));
        let dl_res = (|| -> Result<(), MlError> {
            let mut out_file = std::fs::File::create(&part_path).map_err(|e| MlError::Io(format!("cannot create `{}`: {e}", part_path.display())))?;
            let mut hasher = Sha256::new();
            let mut written = 0u64;

            transport.fetch_stream(art.url, &mut |chunk| {
                if job.cancel.load(Ordering::Acquire) {
                    return Err(MlError::Cancelled);
                }
                written = written.saturating_add(chunk.len() as u64);
                // Guard against runaway responses exceeding declared size + generous 4 MiB margin.
                if written > art.size_bytes.saturating_add(4 << 20) {
                    return Err(MlError::Download(format!("response for `{}` exceeded expected size {}", art.file_name, art.size_bytes)));
                }
                out_file.write_all(chunk).map_err(|e| MlError::Io(e.to_string()))?;
                hasher.update(chunk);
                job.downloaded.fetch_add(chunk.len() as u64, Ordering::AcqRel);
                Ok(())
            })?;

            out_file.flush().map_err(|e| MlError::Io(e.to_string()))?;
            drop(out_file);

            let actual = encode_hex(&hasher.finalize());
            if !actual.eq_ignore_ascii_case(art.sha256) {
                return Err(MlError::ChecksumMismatch { expected: art.sha256.to_ascii_lowercase(), actual });
            }
            std::fs::rename(&part_path, &final_path).map_err(|e| MlError::Io(format!("atomic rename to `{}` failed: {e}", final_path.display())))?;
            Ok(())
        })();

        if let Err(err) = dl_res {
            let _ = std::fs::remove_file(&part_path);
            return Err(err);
        }
    }

    Ok(model_dir)
}
