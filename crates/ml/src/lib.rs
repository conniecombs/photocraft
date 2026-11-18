//! Pure-Rust ONNX (`ModelProto`) inference runtime, `wgpu` / `rayon` compute backends, SHA-256
//! model store, and Tier 1 vision pipelines (`MODNet`, `BiRefNet-lite`, `SAM 2.1 Tiny`) for
//! PhotoCraft.
//!
//! Zero C/C++ FFI, zero `unsafe`, bounded allocations, cooperative per-node [`Interrupt`]
//! cancellation, and full `wasm32-unknown-unknown` compatibility.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod backend;
pub mod catalog;
pub mod onnx;
pub mod ops;
pub mod store;
pub mod tasks;
pub mod tensor;

use thiserror::Error;

pub use backend::{BackendKind, ComputeAccelerator, CpuAccelerator, cpu_accelerator, resolve_backend};
pub use catalog::{BUILTIN_CATALOG, CatalogEntry, MattingPreprocess, ModelArtifact, ModelTask, catalog_entry};
pub use onnx::{Model, ModelSignature};
pub use store::{DownloadSnapshot, DownloadState, LoadedModel, ModelSource, ModelStatus, ModelStore, sha256_hex, verify_sha256};
pub use tasks::sam2::{EmbeddingCache, Sam2Context, Sam2Prompt};
pub use tensor::{DType, Tensor};

/// Errors produced when parsing, downloading, verifying or executing an ONNX model.
#[derive(Debug, Error)]
pub enum MlError {
    #[error("invalid ONNX model: {0}")]
    InvalidModel(String),

    #[error("invalid tensor shape: {0}")]
    InvalidShape(String),

    #[error("tensor type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { expected: &'static str, actual: &'static str },

    #[error("tensor allocation of {needed} bytes exceeds {limit}-byte safety limit")]
    MemoryLimitExceeded { needed: usize, limit: usize },

    #[error("unsupported ONNX operator `{op}` at node `{node}`")]
    UnsupportedOp { op: String, node: String },

    #[error("missing required graph input `{0}`")]
    MissingInput(String),

    #[error("inference cancelled")]
    Cancelled,

    #[error("compute backend error: {0}")]
    Backend(String),

    #[error("SHA-256 checksum mismatch (expected {expected}, got {actual})")]
    ChecksumMismatch { expected: String, actual: String },

    #[error("unknown catalog model `{0}`")]
    UnknownModel(String),

    #[error("model `{0}` is not installed")]
    NotInstalled(String),

    #[error("model download failed: {0}")]
    Download(String),

    #[error("model I/O error: {0}")]
    Io(String),
}
