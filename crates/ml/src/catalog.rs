//! Curated catalog of permissive-license ONNX vision models (`Apache-2.0` / `MIT`) with pinned
//! commit URLs, byte sizes, SHA-256 digests and preprocessing specifications.
//!
//! Every entry here satisfies PhotoCraft's SPDX license allowlist (`Apache-2.0`, `MIT`). Models
//! are never bundled in the repository or binary; users opt in and download them on demand or
//! place `.onnx` files in their configured models folder.

use serde::{Deserialize, Serialize};

/// Vision task category a model serves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ModelTask {
    /// Foreground subject alpha matting (`Select › Subject`, `Remove Background`, `Select and Mask`).
    Subject,
    /// Promptable point/box interactive object segmentation (`Object Selection`, `Select and Mask`).
    Interactive,
    /// Sky segmentation (`Select › Sky`).
    Sky,
    /// User-supplied `.onnx` model without a built-in task tag.
    Custom,
}

impl ModelTask {
    pub const fn as_str(self) -> &'static str {
        match self {
            ModelTask::Subject => "subject",
            ModelTask::Interactive => "interactive",
            ModelTask::Sky => "sky",
            ModelTask::Custom => "custom",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "subject" | "matting" | "background" => Some(ModelTask::Subject),
            "interactive" | "object" | "sam" | "sam2" => Some(ModelTask::Interactive),
            "sky" => Some(ModelTask::Sky),
            "custom" => Some(ModelTask::Custom),
            _ => None,
        }
    }
}

/// Role of a file within a single-file or multi-file (encoder + decoder + `.onnx_data`) package.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ArtifactRole {
    /// Standalone single-graph `.onnx` model.
    PrimaryGraph,
    /// Vision encoder `.onnx` graph (produces cached image embeddings).
    EncoderGraph,
    /// Prompt encoder + mask decoder `.onnx` graph.
    DecoderGraph,
    /// External tensor weights file (`.onnx_data`) referenced by a sibling `.onnx` graph.
    ExternalWeights,
}

/// One downloadable, SHA-256-verified file belonging to a [`CatalogEntry`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelArtifact {
    pub role: ArtifactRole,
    pub file_name: &'static str,
    pub url: &'static str,
    pub size_bytes: u64,
    pub sha256: &'static str,
}

/// Input normalization and aspect/resolution policy for dense matting/segmentation networks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MattingPreprocess {
    /// Scale shortest edge to `shortest_edge` (capped at `max_edge`), rounded to a multiple of
    /// `divisibility`. Used by MODNet (`shortest_edge = 512`, `divisibility = 32`).
    ShortestEdgeDivisible { shortest_edge: u32, max_edge: u32, divisibility: u32, mean: [f32; 3], std: [f32; 3] },
    /// Resize directly to `side × side`. Used by BiRefNet-lite (`1024 × 1024`).
    FixedSquare { side: u32, mean: [f32; 3], std: [f32; 3] },
}

impl MattingPreprocess {
    /// Default ImageNet-normalized `512 × 512` preprocessing for user-installed `.onnx` models.
    pub const DEFAULT_CUSTOM: Self =
        Self::ShortestEdgeDivisible { shortest_edge: 512, max_edge: 1024, divisibility: 32, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] };
}

/// Built-in catalog metadata for a verified open-weights model.
#[derive(Clone, Copy, Debug)]
pub struct CatalogEntry {
    pub id: &'static str,
    pub name: &'static str,
    pub summary: &'static str,
    pub task: ModelTask,
    /// Higher priority wins when multiple models for the same [`ModelTask`] are installed.
    pub priority: u8,
    /// SPDX license identifier (must be in PhotoCraft's allowlist).
    pub license: &'static str,
    pub upstream_repo: &'static str,
    pub preprocess: Option<MattingPreprocess>,
    pub artifacts: &'static [ModelArtifact],
}

impl CatalogEntry {
    /// Total download size in bytes across all artifacts of this model.
    pub fn total_bytes(&self) -> u64 {
        self.artifacts.iter().map(|a| a.size_bytes).sum()
    }
}

const MODNET_ARTIFACTS: &[ModelArtifact] = &[ModelArtifact {
    role: ArtifactRole::PrimaryGraph,
    file_name: "model_fp16.onnx",
    url: "https://huggingface.co/Xenova/modnet/resolve/fa2fa546052fba4c08921230a26cc69a333fca12/onnx/model_fp16.onnx",
    size_bytes: 12_984_781,
    sha256: "25f165da9bfd30830a575f1f0490f1acd995975cb349bc02f3d79332e1fe5cf6",
}];

const BIREFNET_LITE_ARTIFACTS: &[ModelArtifact] = &[ModelArtifact {
    role: ArtifactRole::PrimaryGraph,
    file_name: "model_fp16.onnx",
    url: "https://huggingface.co/onnx-community/BiRefNet_lite-ONNX/resolve/de15b22ba131738a16dff04aab8bdf8dc32e3ac1/onnx/model_fp16.onnx",
    size_bytes: 114_538_221,
    sha256: "d39b897ceb16ae654c1731f3dba0cf9b368d9cae74b5a57459b455cc8bfec402",
}];

const SAM2_TINY_ARTIFACTS: &[ModelArtifact] = &[
    ModelArtifact {
        role: ArtifactRole::EncoderGraph,
        file_name: "vision_encoder_fp16.onnx",
        url: "https://huggingface.co/onnx-community/sam2.1-hiera-tiny-ONNX/resolve/814a066640debee5a91e70aa401fb8e17e030503/onnx/vision_encoder_fp16.onnx",
        size_bytes: 314_925,
        sha256: "7773e79d589dada8e1e630cc7eb18b17b2c5b4faeb1eaab5ebf9df618ecbd50a",
    },
    ModelArtifact {
        role: ArtifactRole::ExternalWeights,
        file_name: "vision_encoder_fp16.onnx_data",
        url: "https://huggingface.co/onnx-community/sam2.1-hiera-tiny-ONNX/resolve/814a066640debee5a91e70aa401fb8e17e030503/onnx/vision_encoder_fp16.onnx_data",
        size_bytes: 67_005_504,
        sha256: "a4dd3759e9b6a476d991fb3493787992e78777e51462695e1bede71ae258103e",
    },
    ModelArtifact {
        role: ArtifactRole::DecoderGraph,
        file_name: "prompt_encoder_mask_decoder_fp16.onnx",
        url: "https://huggingface.co/onnx-community/sam2.1-hiera-tiny-ONNX/resolve/814a066640debee5a91e70aa401fb8e17e030503/onnx/prompt_encoder_mask_decoder_fp16.onnx",
        size_bytes: 229_799,
        sha256: "00ac7200451f246e8307aa1a1648113f47dc45625738406403a0e6b86c69fc55",
    },
    ModelArtifact {
        role: ArtifactRole::ExternalWeights,
        file_name: "prompt_encoder_mask_decoder_fp16.onnx_data",
        url: "https://huggingface.co/onnx-community/sam2.1-hiera-tiny-ONNX/resolve/814a066640debee5a91e70aa401fb8e17e030503/onnx/prompt_encoder_mask_decoder_fp16.onnx_data",
        size_bytes: 10_454_016,
        sha256: "027b457a53f5bd76bb401c59555448451d04b4eab6d2f6a6758406b40133ee2c",
    },
];

/// Curated built-in model catalog.
pub const BUILTIN_CATALOG: &[CatalogEntry] = &[
    CatalogEntry {
        id: "modnet-photographic",
        name: "MODNet — Fast Subject Matting",
        summary: "Lightweight alpha matting for Select Subject and Remove Background (12.4 MiB).",
        task: ModelTask::Subject,
        priority: 10,
        license: "Apache-2.0",
        upstream_repo: "ZHKKKe/MODNet",
        preprocess: Some(MattingPreprocess::ShortestEdgeDivisible {
            shortest_edge: 512,
            max_edge: 1024,
            divisibility: 32,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
        }),
        artifacts: MODNET_ARTIFACTS,
    },
    CatalogEntry {
        id: "birefnet-lite",
        name: "BiRefNet Lite — High-Detail Subject Matting",
        summary: "High-resolution boundary matting for hair, fur and fine edges (109.2 MiB).",
        task: ModelTask::Subject,
        priority: 20,
        license: "MIT",
        upstream_repo: "ZhengPeng7/BiRefNet_lite",
        preprocess: Some(MattingPreprocess::FixedSquare { side: 1024, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] }),
        artifacts: BIREFNET_LITE_ARTIFACTS,
    },
    CatalogEntry {
        id: "sam2.1-tiny",
        name: "SAM 2.1 Tiny — Interactive Object Selection",
        summary: "Click, hover and box-prompted object segmentation with cached image embeddings (74.4 MiB).",
        task: ModelTask::Interactive,
        priority: 20,
        license: "Apache-2.0",
        upstream_repo: "facebookresearch/sam2",
        preprocess: None,
        artifacts: SAM2_TINY_ARTIFACTS,
    },
];

/// Look up a built-in catalog entry by its stable ID.
pub fn catalog_entry(id: &str) -> Option<&'static CatalogEntry> {
    BUILTIN_CATALOG.iter().find(|e| e.id == id)
}
