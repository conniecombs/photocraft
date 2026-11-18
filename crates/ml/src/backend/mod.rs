//! Compute backend selection (`Auto`, `Gpu`, `Cpu`) and accelerator trait for heavy linear
//! algebra and convolution kernels.
//!
//! When compiled with the `gpu` feature on native targets, [`resolve_backend`] initializes (and
//! caches across inferences) a headless `wgpu` compute pipeline accelerator for `MatMul`/`Gemm`,
//! `Conv` and `ConvTranspose`, falling back transparently to parallel `rayon` CPU kernels whenever
//! no adapter is present, a tensor exceeds device buffer limits, or `BackendKind::Cpu` is chosen.

#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
pub mod wgpu_backend;

use std::sync::Arc;
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

/// Preferred compute backend for local ONNX inference.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BackendKind {
    /// Use `wgpu` compute shaders when available, otherwise parallel CPU (`rayon`).
    #[default]
    Auto,
    /// Prefer `wgpu` compute shaders (falls back to CPU if no compatible GPU adapter exists).
    Gpu,
    /// Always run on the parallel CPU backend.
    Cpu,
}

impl BackendKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            BackendKind::Auto => "auto",
            BackendKind::Gpu => "gpu",
            BackendKind::Cpu => "cpu",
        }
    }
}

/// Dimensions for batched matrix multiplication:
/// `A` (`[a_batch, m, k]`) × `B` (`[b_batch, k, n]`) -> `Y` (`[batch, m, n]`).
#[derive(Clone, Copy, Debug)]
pub struct MatMulSpec {
    pub batch: usize,
    pub a_batch: usize,
    pub b_batch: usize,
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

/// Dimensions and hyperparameters for 2D `Conv` / `ConvTranspose`.
#[derive(Clone, Copy, Debug)]
pub struct Conv2dSpec {
    pub batch: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub in_h: usize,
    pub in_w: usize,
    pub out_h: usize,
    pub out_w: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub pad_top: usize,
    pub pad_left: usize,
    pub dil_h: usize,
    pub dil_w: usize,
    pub groups: usize,
}

/// Pluggable hardware accelerator for dense tensor operators.
///
/// Returning `None` from any kernel signals the caller to execute the exact parallel CPU path.
pub trait ComputeAccelerator: Send + Sync {
    /// Backend identifier (`"cpu"` or `"wgpu"`).
    fn name(&self) -> &'static str;

    /// Adapter description when backed by a GPU device.
    fn adapter_name(&self) -> Option<&str> {
        None
    }

    /// Whether this accelerator dispatches compute shaders on a `wgpu` device.
    fn is_gpu(&self) -> bool {
        false
    }

    /// Optional GPU offload for batched `MatMul`.
    fn matmul_batched(&self, _spec: &MatMulSpec, _a: &[f32], _b: &[f32]) -> Option<Vec<f32>> {
        None
    }

    /// Optional GPU offload for 2D NCHW `Conv`.
    fn conv2d(&self, _spec: &Conv2dSpec, _input: &[f32], _weight: &[f32], _bias: Option<&[f32]>) -> Option<Vec<f32>> {
        None
    }

    /// Optional GPU offload for 2D NCHW `ConvTranspose`.
    fn conv_transpose2d(&self, _spec: &Conv2dSpec, _input: &[f32], _weight: &[f32], _bias: Option<&[f32]>) -> Option<Vec<f32>> {
        None
    }
}

/// Parallel CPU backend (`rayon` thread pool).
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuAccelerator;

impl ComputeAccelerator for CpuAccelerator {
    fn name(&self) -> &'static str {
        "cpu"
    }
}

static CPU_ACCEL: CpuAccelerator = CpuAccelerator;

#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
static SHARED_WGPU: OnceLock<Option<Arc<wgpu_backend::WgpuAccelerator>>> = OnceLock::new();

/// Obtain the shared `wgpu` compute accelerator if the `gpu` feature is enabled and an adapter
/// initialized cleanly.
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
pub fn shared_wgpu_accelerator() -> Option<Arc<wgpu_backend::WgpuAccelerator>> {
    SHARED_WGPU
        .get_or_init(|| match wgpu_backend::WgpuAccelerator::try_new() {
            Ok(acc) => Some(Arc::new(acc)),
            Err(e) => {
                log::warn!("photocraft-ml: wgpu compute backend unavailable, falling back to CPU: {e}");
                None
            }
        })
        .clone()
}

/// Resolve `kind` to a concrete [`ComputeAccelerator`].
pub fn resolve_backend(kind: BackendKind) -> Arc<dyn ComputeAccelerator> {
    match kind {
        BackendKind::Cpu => Arc::new(CPU_ACCEL),
        BackendKind::Auto | BackendKind::Gpu => {
            #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
            if let Some(gpu) = shared_wgpu_accelerator() {
                // In `Auto` mode, skip CPU-emulated software Vulkan/DX12 rasterizers when native
                // rayon kernels are already available; explicit `Gpu` still uses software adapters
                // (useful in headless CI tests).
                if kind == BackendKind::Gpu || !gpu.is_software_adapter() {
                    return gpu;
                }
            }
            Arc::new(CPU_ACCEL)
        }
    }
}

/// Borrow the static CPU accelerator without heap allocation.
pub fn cpu_accelerator() -> &'static dyn ComputeAccelerator {
    &CPU_ACCEL
}
