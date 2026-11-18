//! `ml.*` commands and colour-managed local ONNX model execution (`photocraft-ml`).
//!
//! Local models are strictly opt-in (`Preferences › AI Integrations › Enable Local AI Models`,
//! default `false`). When disabled or when no model for the requested task is installed, every
//! selection and cutout command falls back transparently to `photocraft-algo`'s classical
//! algorithms.

use std::path::PathBuf;
use std::sync::Arc;

use craft_cms::{Builtin, Transform, TransformOptions};
use photocraft_algo::segment::Sampler;
use photocraft_algo::selection::Region;
use photocraft_doc::Document;
use photocraft_geom::Rect;
use photocraft_ml::backend::resolve_backend;
use photocraft_ml::catalog::ModelTask;
use photocraft_ml::onnx::Model;
use photocraft_ml::store::{ModelStatus, ModelStore};
use photocraft_ml::tasks::matting::{predict_region, predict_surface};
use photocraft_ml::tasks::sam2::{EmbeddingCache, Sam2Context, Sam2Prompt, segment_prompt};
use photocraft_ml::{BackendKind, MlError};
use photocraft_raster::{Interrupt, Surface};
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::prefs::{MlBackend, Preferences};
use crate::{EngineError, Result, Session};

/// Maximum base64 payload accepted by `ml.install` (~192 MiB decoded).
const MAX_MODEL_B64_CHARS: usize = (192 << 20) / 3 * 4 + 4;

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

fn ml_err(e: MlError) -> EngineError {
    EngineError::Other(e.to_string())
}

fn always(_: &Session) -> std::result::Result<(), String> {
    Ok(())
}

pub(crate) fn map_backend(pref: MlBackend) -> BackendKind {
    match pref {
        MlBackend::Auto => BackendKind::Auto,
        MlBackend::Gpu => BackendKind::Gpu,
        MlBackend::Cpu => BackendKind::Cpu,
    }
}

/// Per-session local ML state (isolated across concurrent test sessions).
#[derive(Clone, Debug, Default)]
pub struct MlRuntime {
    pub store: ModelStore,
    pub sam2_cache: EmbeddingCache,
    default_models_dir: Option<PathBuf>,
    last_synced: Option<(bool, MlBackend, Option<PathBuf>)>,
}

/// Thread-safe snapshot of a session's ML settings and model caches for background worker threads
/// (`Remove Background`, `Object Finder` hover, `Select and Mask` preview).
#[derive(Clone, Debug)]
pub struct MlSnapshot {
    pub enabled: bool,
    pub backend: BackendKind,
    pub store: ModelStore,
    pub sam2_cache: EmbeddingCache,
}

impl Default for MlSnapshot {
    fn default() -> Self {
        Self { enabled: false, backend: BackendKind::Auto, store: ModelStore::new(), sam2_cache: EmbeddingCache::default() }
    }
}

impl MlRuntime {
    /// Capture a thread-safe snapshot for synchronous or background inference.
    pub fn snapshot(&self, prefs: &Preferences) -> MlSnapshot {
        MlSnapshot {
            enabled: prefs.integrations.enable_local_models,
            backend: map_backend(prefs.integrations.ml_backend),
            store: self.store.clone(),
            sam2_cache: self.sam2_cache.clone(),
        }
    }

    /// Effective models folder from `prefs` (falling back to `default_models_dir`).
    pub fn effective_dir(&self, prefs: &Preferences) -> Option<PathBuf> {
        let custom = prefs.integrations.models_folder.trim();
        if !custom.is_empty() { Some(PathBuf::from(custom)) } else { self.default_models_dir.clone() }
    }

    /// Set the platform default `Models` folder (used when `integrations.modelsFolder` is empty).
    pub fn set_default_models_dir(&mut self, dir: Option<PathBuf>, prefs: &Preferences) {
        self.default_models_dir = dir;
        let effective = self.effective_dir(prefs);
        let _ = self.store.set_models_dir(effective);
    }
}

/// Synchronize the session's [`ModelStore`] directory and selection caches after preferences load
/// or change.
pub(crate) fn sync_prefs(s: &mut Session) {
    let integ = &s.prefs().integrations;
    let enabled = integ.enable_local_models;
    let backend = integ.ml_backend;
    let effective_dir = s.ml.effective_dir(s.prefs());
    let next_state = (enabled, backend, effective_dir.clone());

    if s.ml.last_synced.as_ref() == Some(&next_state) {
        return;
    }
    let dir_changed = s.ml.last_synced.as_ref().map(|(_, _, d)| d) != Some(&effective_dir);
    s.ml.last_synced = Some(next_state);
    if dir_changed {
        let _ = s.ml.store.set_models_dir(effective_dir);
    }
    invalidate_selection_caches(s);
}

pub(crate) fn invalidate_selection_caches(s: &mut Session) {
    s.subject_cache = crate::smartselect_cmds::SubjectCache::default();
    s.object_finder = crate::smartselect_cmds::ObjectFinderCache::default();
    s.ml.sam2_cache.clear();
}

fn is_identity_srgb(t: &Transform) -> bool {
    const N: usize = 5;
    let scale = (N - 1) as f32;
    let mut out = [0.0f32; 16];
    for r in 0..N {
        for g in 0..N {
            for b in 0..N {
                let v = [r as f32 / scale, g as f32 / scale, b as f32 / scale];
                if t.eval(&v, &mut out).is_err() || (0..3).any(|c| (out[c] - v[c]).abs() > 0.5 / 255.0) {
                    return false;
                }
            }
        }
    }
    true
}

/// Colour-managed [`Sampler`] adapter converting document composite samples to `sRGB` `[0..=1]`
/// via `craft-cms` (`AGENTS.md` Rule 2) before feeding vision neural networks.
pub struct SrgbSampler<'a> {
    inner: &'a dyn Sampler,
    to_srgb: Option<Arc<Transform>>,
}

impl<'a> SrgbSampler<'a> {
    pub fn new(inner: &'a dyn Sampler, doc: &Document) -> Self {
        let src = crate::color_cmds::composite_profile(doc);
        let dst = Builtin::Srgb.profile();
        let to_srgb = craft_cms::transform::cached(&src, dst, TransformOptions::default()).ok().filter(|t| !is_identity_srgb(t));
        Self { inner, to_srgb }
    }
}

impl Sampler for SrgbSampler<'_> {
    fn rgba(&self, r: Rect) -> Vec<[f32; 4]> {
        let mut px = self.inner.rgba(r);
        let Some(t) = &self.to_srgb else {
            return px;
        };
        let mut out = [0.0f32; 16];
        for p in &mut px {
            let rgb = [p[0].clamp(0.0, 1.0), p[1].clamp(0.0, 1.0), p[2].clamp(0.0, 1.0)];
            if t.eval(&rgb, &mut out).is_ok() {
                p[0] = out[0].clamp(0.0, 1.0);
                p[1] = out[1].clamp(0.0, 1.0);
                p[2] = out[2].clamp(0.0, 1.0);
            }
        }
        px
    }
}

/// Run local subject matting (`MODNet` / `BiRefNet-lite`) when enabled and installed, returning
/// `None` so callers fall back to classical saliency+GrabCut when disabled, uninstalled or empty.
pub fn try_select_subject(ml: &MlSnapshot, smp: &dyn Sampler, doc: &Document, area: Rect, ctl: &Interrupt<'_>) -> Option<Region> {
    if !ml.enabled {
        return None;
    }
    let loaded = ml.store.load_for_task(ModelTask::Subject).ok().flatten()?;
    let accel = resolve_backend(ml.backend);
    let srgb = SrgbSampler::new(smp, doc);
    match predict_region(&loaded, &srgb, area, accel.as_ref(), ctl) {
        Ok(reg) => reg,
        Err(MlError::Cancelled) => None,
        Err(e) => {
            log::warn!("photocraft-ml subject inference failed, falling back to classical: {e}");
            None
        }
    }
}

/// Run local subject matting directly to a tiled mask [`Surface`] at `doc.depth` (`U8`/`U16`/`F32`),
/// returning `Ok(Some(surface))` on success, `Ok(None)` when disabled/uninstalled, or
/// `Err(EngineError::Cancelled)` if `ctl` cancelled mid-inference.
pub fn try_subject_surface(ml: &MlSnapshot, smp: &dyn Sampler, doc: &Document, area: Rect, ctl: &Interrupt<'_>) -> Result<Option<Surface>> {
    if !ml.enabled {
        return Ok(None);
    }
    let Some(loaded) = ml.store.load_for_task(ModelTask::Subject).ok().flatten() else {
        return Ok(None);
    };
    let accel = resolve_backend(ml.backend);
    let srgb = SrgbSampler::new(smp, doc);
    match predict_surface(&loaded, &srgb, area, doc.depth, accel.as_ref(), ctl) {
        Ok(surf) => Ok(surf),
        Err(MlError::Cancelled) => Err(EngineError::Cancelled),
        Err(e) => {
            log::warn!("photocraft-ml background removal failed, falling back to classical: {e}");
            Ok(None)
        }
    }
}

/// Run local sky segmentation when enabled and a `sky` model is installed.
pub fn try_select_sky(ml: &MlSnapshot, smp: &dyn Sampler, doc: &Document, area: Rect, ctl: &Interrupt<'_>) -> Option<Region> {
    if !ml.enabled {
        return None;
    }
    let loaded = ml.store.load_for_task(ModelTask::Sky).ok().flatten()?;
    let accel = resolve_backend(ml.backend);
    let srgb = SrgbSampler::new(smp, doc);
    predict_region(&loaded, &srgb, area, accel.as_ref(), ctl).ok().flatten()
}

/// Run local SAM-2 interactive segmentation when enabled and an `interactive` model is installed.
pub fn try_select_object(ml: &MlSnapshot, smp: &dyn Sampler, doc: &Document, area: Rect, prompt: Sam2Prompt, ctl: &Interrupt<'_>) -> Option<Region> {
    if !ml.enabled {
        return None;
    }
    let loaded = ml.store.load_for_task(ModelTask::Interactive).ok().flatten()?;
    let accel = resolve_backend(ml.backend);
    let srgb = SrgbSampler::new(smp, doc);
    let ctx = Sam2Context { loaded: &loaded, cache: &ml.sam2_cache, accel: accel.as_ref() };
    match segment_prompt(&ctx, &srgb, area, prompt, ctl) {
        Ok(reg) => reg,
        Err(MlError::Cancelled) => None,
        Err(e) => {
            log::warn!("photocraft-ml SAM-2 inference failed, falling back to classical: {e}");
            None
        }
    }
}

fn b64_decode(cmd: &str, field: &str, raw: &str) -> Result<Vec<u8>> {
    if raw.len() > MAX_MODEL_B64_CHARS {
        return Err(bad(cmd, format!("`{field}` exceeds maximum model payload size")));
    }
    photocraft_paint::tile::b64_decode(&raw.replace('-', "+").replace('_', "/")).ok_or_else(|| bad(cmd, format!("`{field}` is not valid base64")))
}

fn serialize_status(st: &ModelStatus) -> Value {
    serde_json::to_value(st).unwrap_or(Value::Null)
}

fn cmd_list(s: &mut Session, _p: &Value) -> Result<Value> {
    let integ = &s.prefs().integrations;
    let backend_kind = map_backend(integ.ml_backend);
    let accel = resolve_backend(backend_kind);
    let folder = s.ml.effective_dir(s.prefs()).map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
    let models: Vec<Value> = s.ml.store.list_status().iter().map(serialize_status).collect();
    Ok(json!({
        "enabled": integ.enable_local_models,
        "backend": backend_kind.as_str(),
        "activeBackend": accel.name(),
        "adapter": accel.adapter_name(),
        "modelsFolder": folder,
        "models": models,
    }))
}

fn cmd_install(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "ml.install";
    let task_opt = match p.get("task").and_then(Value::as_str) {
        Some(t) => Some(ModelTask::parse(t).ok_or_else(|| bad(CMD, format!("unknown `task` `{t}` (expected subject|interactive|sky|custom)")))?),
        None => None,
    };

    let status = if let (Some(enc_v), Some(dec_v)) = (p.get("encoderData"), p.get("decoderData")) {
        let id = p.get("id").and_then(Value::as_str).ok_or_else(|| bad(CMD, "missing `id`"))?;
        let enc_str = enc_v.as_str().ok_or_else(|| bad(CMD, "`encoderData` must be a base64 string"))?;
        let dec_str = dec_v.as_str().ok_or_else(|| bad(CMD, "`decoderData` must be a base64 string"))?;
        let enc_bytes = b64_decode(CMD, "encoderData", enc_str)?;
        let dec_bytes = b64_decode(CMD, "decoderData", dec_str)?;
        let enc = Model::from_bytes(&enc_bytes).map_err(ml_err)?;
        let dec = Model::from_bytes(&dec_bytes).map_err(ml_err)?;
        s.ml.store.install_sam2_models(id, enc, dec).map_err(ml_err)?
    } else if let Some(data_v) = p.get("data") {
        let id = p.get("id").and_then(Value::as_str).ok_or_else(|| bad(CMD, "missing `id`"))?;
        let data_str = data_v.as_str().ok_or_else(|| bad(CMD, "`data` must be a base64 string"))?;
        let bytes = b64_decode(CMD, "data", data_str)?;
        let task = task_opt.unwrap_or_else(|| photocraft_ml::catalog_entry(id).map_or(ModelTask::Subject, |c| c.task));
        s.ml.store.install_bytes(id, task, None, &bytes).map_err(ml_err)?
    } else if let Some(path_v) = p.get("path") {
        let path_str = path_v.as_str().filter(|v| !v.trim().is_empty()).ok_or_else(|| bad(CMD, "`path` must be a non-empty file path"))?;
        let id_override = p.get("id").and_then(Value::as_str);
        install_from_path(&s.ml.store, path_str, id_override, task_opt)?
    } else {
        return Err(bad(CMD, "provide `data` + `id`, `encoderData` + `decoderData` + `id`, or `path`"));
    };

    invalidate_selection_caches(s);
    Ok(serialize_status(&status))
}

#[cfg(not(target_arch = "wasm32"))]
fn install_from_path(store: &ModelStore, path_str: &str, id_override: Option<&str>, task_opt: Option<ModelTask>) -> Result<ModelStatus> {
    store.install_path(std::path::Path::new(path_str), id_override, task_opt).map_err(ml_err)
}

#[cfg(target_arch = "wasm32")]
fn install_from_path(_store: &ModelStore, _path_str: &str, _id_override: Option<&str>, _task_opt: Option<ModelTask>) -> Result<ModelStatus> {
    Err(EngineError::Other("installing models from a file path is not available on the web; pass base64 `data`".into()))
}

fn cmd_download(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "ml.download";
    let id = p.get("id").and_then(Value::as_str).filter(|v| !v.trim().is_empty()).ok_or_else(|| bad(CMD, "missing `id` (see ml.list)"))?;
    let wait = p.get("wait").and_then(Value::as_bool).unwrap_or(false);
    let snap = s.ml.store.start_download(id, wait).map_err(ml_err)?;
    if wait {
        invalidate_selection_caches(s);
    }
    Ok(json!({
        "id": id,
        "download": serde_json::to_value(&snap).unwrap_or(Value::Null),
    }))
}

fn cmd_cancel_download(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "ml.cancelDownload";
    let id = p.get("id").and_then(Value::as_str).filter(|v| !v.trim().is_empty()).ok_or_else(|| bad(CMD, "missing `id`"))?;
    let cancelled = s.ml.store.cancel_download(id);
    Ok(json!({ "id": id, "cancelled": cancelled }))
}

fn cmd_remove(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "ml.remove";
    let id = p.get("id").and_then(Value::as_str).filter(|v| !v.trim().is_empty()).ok_or_else(|| bad(CMD, "missing `id`"))?;
    let removed = s.ml.store.remove(id).map_err(|e| bad(CMD, e.to_string()))?;
    if !removed {
        return Err(EngineError::Other(format!("model `{id}` is not installed")));
    }
    invalidate_selection_caches(s);
    Ok(json!({ "removed": id }))
}

fn cmd_reload(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "ml.reload";
    if let Some(path_v) = p.get("path") {
        let path_str = path_v.as_str().filter(|v| !v.trim().is_empty()).ok_or_else(|| bad(CMD, "`path` must be a folder path"))?;
        s.ml.store.set_models_dir(Some(PathBuf::from(path_str.trim()))).map_err(ml_err)?;
    } else {
        let effective = s.ml.effective_dir(s.prefs());
        s.ml.store.set_models_dir(effective).map_err(ml_err)?;
        s.ml.store.reload().map_err(ml_err)?;
    }
    invalidate_selection_caches(s);
    let count = s.ml.store.list_status().iter().filter(|m| m.installed).count();
    Ok(json!({ "installedCount": count }))
}

/// Engine command specs for local ONNX model management (`ml.*`).
pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec { id: "ml.list", label: "List Local AI Models", menu: &[], shortcut: None, params: "{}", enabled: always, run: cmd_list, journal: false },
        CommandSpec {
            id: "ml.install",
            label: "Install Local AI Model…",
            menu: &[],
            shortcut: None,
            params: r##"{"id":text,"task":"subject|interactive|sky|custom","path":text,"data":text,"encoderData":text,"decoderData":text}"##,
            enabled: always,
            run: cmd_install,
            journal: false,
        },
        CommandSpec {
            id: "ml.download",
            label: "Download Local AI Model",
            menu: &[],
            shortcut: None,
            params: r##"{"id":text,"wait":bool=false}"##,
            enabled: always,
            run: cmd_download,
            journal: false,
        },
        CommandSpec {
            id: "ml.cancelDownload",
            label: "Cancel AI Model Download",
            menu: &[],
            shortcut: None,
            params: r##"{"id":text}"##,
            enabled: always,
            run: cmd_cancel_download,
            journal: false,
        },
        CommandSpec {
            id: "ml.remove",
            label: "Remove Local AI Model",
            menu: &[],
            shortcut: None,
            params: r##"{"id":text}"##,
            enabled: always,
            run: cmd_remove,
            journal: false,
        },
        CommandSpec {
            id: "ml.reload",
            label: "Reload Local AI Models",
            menu: &[],
            shortcut: None,
            params: r##"{"path":text}"##,
            enabled: always,
            run: cmd_reload,
            journal: false,
        },
    ]
}
