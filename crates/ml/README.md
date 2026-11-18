# photocraft-ml (Layer 4)

Pure-Rust, clean-room ONNX (`ModelProto`) inference engine, compute backends (`wgpu` WGSL compute
shaders + parallel `rayon` CPU), SHA-256 verified `ModelStore`, and Tier 1 vision pipelines for
PhotoCraft:

- **ONNX Runtime (`onnx`, `tensor`, `ops`)**: Zero `unsafe`, zero C/C++ FFI, bounded Protobuf
  decoding (`f32`, `f16`, `bf16`, `f64`, integer tensors, and split `.onnx_data` external weights),
  topological graph validation, reference-counted intermediate tensor eviction, and per-node
  cooperative `Interrupt` cancellation.
- **Backends (`backend`)**: Optional headless `wgpu` compute shaders (`MatMul`/`Gemm`, `Conv2d`,
  `ConvTranspose2d`) with device-limit checks, non-panicking device-lost/error handlers, and
  automatic fallback to cache-blocked `rayon` CPU kernels. Compiles cleanly to
  `wasm32-unknown-unknown`.
- **Model Registry (`catalog`, `store`)**: Permissive-license catalog (`MODNet` `Apache-2.0`,
  `BiRefNet-lite` `MIT`, `SAM 2.1 Hiera Tiny` `Apache-2.0`), streaming HTTPS downloads with
  SHA-256 verification and atomic `.part` rename, local folder discovery, and in-memory loading.
- **Vision Pipelines (`tasks`)**:
  - `tasks::matting`: Strip-downsampled sRGB preprocessing and tiled upsampling into both 8-bit
    `Region`s and native `8/16/32f` single-channel `Surface`s.
  - `tasks::sam2`: Promptable point/hover/box segmentation with bounded `EmbeddingCache` so
    repeated hover and click queries reuse the Hiera vision encoder's feature pyramid.
