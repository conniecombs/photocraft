//! Headless `wgpu` WGSL compute-shader backend for `MatMul`, `Conv2d` and `ConvTranspose2d`.
//!
//! Installs non-panicking `on_uncaptured_error` and `set_device_lost_callback` handlers on its
//! [`wgpu::Device`], validates every buffer against device limits before allocation, bounds GPU
//! fence waits with a timeout, and returns `None` on any limit or runtime fault so the caller
//! falls back seamlessly to the parallel CPU implementation.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use crate::MlError;
use crate::backend::{ComputeAccelerator, Conv2dSpec, MatMulSpec};

const GPU_WAIT_TIMEOUT: Duration = Duration::from_secs(15);
const INIT_TIMEOUT: Duration = Duration::from_secs(10);

const MATMUL_WGSL: &str = r#"
struct Params {
    batch: u32,
    a_batch: u32,
    b_batch: u32,
    m: u32,
    k: u32,
    n: u32,
    z_offset: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> u: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let batch_idx = gid.z + u.z_offset;
    let row = gid.y;
    let col = gid.x;
    let a_b = select(batch_idx, 0u, u.a_batch == 1u);
    let b_b = select(batch_idx, 0u, u.b_batch == 1u);
    let a_base = a_b * u.m * u.k;
    let b_base = b_b * u.k * u.n;

    var acc: f32 = 0.0;
    let num_tiles = (u.k + 15u) / 16u;
    let local_idx = lid.y * 16u + lid.x;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * 16u + lid.x;
        let b_row = t * 16u + lid.y;

        if (batch_idx < u.batch && row < u.m && a_col < u.k) {
            tile_a[local_idx] = a[a_base + row * u.k + a_col];
        } else {
            tile_a[local_idx] = 0.0;
        }

        if (batch_idx < u.batch && b_row < u.k && col < u.n) {
            tile_b[local_idx] = b[b_base + b_row * u.n + col];
        } else {
            tile_b[local_idx] = 0.0;
        }

        workgroupBarrier();

        for (var p: u32 = 0u; p < 16u; p = p + 1u) {
            acc = acc + tile_a[lid.y * 16u + p] * tile_b[p * 16u + lid.x];
        }

        workgroupBarrier();
    }

    if (batch_idx < u.batch && row < u.m && col < u.n) {
        out[(batch_idx * u.m + row) * u.n + col] = acc;
    }
}
"#;

const CONV2D_WGSL: &str = r#"
struct Params {
    batch: u32,
    in_c: u32,
    out_c: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    pt: u32,
    pl: u32,
    dh: u32,
    dw: u32,
    groups: u32,
    has_bias: u32,
    z_offset: u32,
    _p0: u32,
    _p1: u32,
}

@group(0) @binding(0) var<uniform> u: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ow = gid.x;
    let oh = gid.y;
    let boc = gid.z + u.z_offset;
    let total_boc = u.batch * u.out_c;
    if (ow >= u.out_w || oh >= u.out_h || boc >= total_boc) {
        return;
    }

    let b = boc / u.out_c;
    let oc = boc % u.out_c;
    let out_c_per_g = u.out_c / u.groups;
    let in_c_per_g = u.in_c / u.groups;
    let g = oc / out_c_per_g;
    let ic_start = g * in_c_per_g;

    var sum: f32 = 0.0;
    if (u.has_bias != 0u) {
        sum = bias[oc];
    }

    let base_ih: i32 = i32(oh * u.sh) - i32(u.pt);
    let base_iw: i32 = i32(ow * u.sw) - i32(u.pl);

    for (var ic_l: u32 = 0u; ic_l < in_c_per_g; ic_l = ic_l + 1u) {
        let in_plane = ((b * u.in_c + ic_start + ic_l) * u.in_h) * u.in_w;
        let w_plane = ((oc * in_c_per_g + ic_l) * u.kh) * u.kw;
        for (var ky: u32 = 0u; ky < u.kh; ky = ky + 1u) {
            let ih = base_ih + i32(ky * u.dh);
            if (ih >= 0 && u32(ih) < u.in_h) {
                let in_row = in_plane + u32(ih) * u.in_w;
                let w_row = w_plane + ky * u.kw;
                for (var kx: u32 = 0u; kx < u.kw; kx = kx + 1u) {
                    let iw = base_iw + i32(kx * u.dw);
                    if (iw >= 0 && u32(iw) < u.in_w) {
                        sum = sum + input[in_row + u32(iw)] * weight[w_row + kx];
                    }
                }
            }
        }
    }

    out[(boc * u.out_h + oh) * u.out_w + ow] = sum;
}
"#;

const CONV_TRANSPOSE2D_WGSL: &str = r#"
struct Params {
    batch: u32,
    in_c: u32,
    out_c: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    pt: u32,
    pl: u32,
    dh: u32,
    dw: u32,
    groups: u32,
    has_bias: u32,
    z_offset: u32,
    _p0: u32,
    _p1: u32,
}

@group(0) @binding(0) var<uniform> u: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ow = gid.x;
    let oh = gid.y;
    let boc = gid.z + u.z_offset;
    let total_boc = u.batch * u.out_c;
    if (ow >= u.out_w || oh >= u.out_h || boc >= total_boc) {
        return;
    }

    let b = boc / u.out_c;
    let oc = boc % u.out_c;
    let out_c_per_g = u.out_c / u.groups;
    let in_c_per_g = u.in_c / u.groups;
    let g = oc / out_c_per_g;
    let oc_local = oc % out_c_per_g;
    let ic_start = g * in_c_per_g;

    var sum: f32 = 0.0;
    if (u.has_bias != 0u) {
        sum = bias[oc];
    }

    for (var ky: u32 = 0u; ky < u.kh; ky = ky + 1u) {
        let h_nom: i32 = i32(oh) + i32(u.pt) - i32(ky * u.dh);
        if (h_nom >= 0 && (u32(h_nom) % u.sh == 0u)) {
            let ih = u32(h_nom) / u.sh;
            if (ih < u.in_h) {
                for (var kx: u32 = 0u; kx < u.kw; kx = kx + 1u) {
                    let w_nom: i32 = i32(ow) + i32(u.pl) - i32(kx * u.dw);
                    if (w_nom >= 0 && (u32(w_nom) % u.sw == 0u)) {
                        let iw = u32(w_nom) / u.sw;
                        if (iw < u.in_w) {
                            for (var ic_l: u32 = 0u; ic_l < in_c_per_g; ic_l = ic_l + 1u) {
                                let ic = ic_start + ic_l;
                                let x_idx = ((b * u.in_c + ic) * u.in_h + ih) * u.in_w + iw;
                                let w_idx = ((ic * out_c_per_g + oc_local) * u.kh + ky) * u.kw + kx;
                                sum = sum + input[x_idx] * weight[w_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    out[(boc * u.out_h + oh) * u.out_w + ow] = sum;
}
"#;

/// Compute-shader accelerator running on a headless [`wgpu::Device`].
pub struct WgpuAccelerator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    limits: wgpu::Limits,
    adapter_info: wgpu::AdapterInfo,
    faulted: Arc<AtomicBool>,
    exec_lock: Mutex<()>,
    matmul_layout: wgpu::BindGroupLayout,
    matmul_pipe: wgpu::ComputePipeline,
    conv_layout: wgpu::BindGroupLayout,
    conv_pipe: wgpu::ComputePipeline,
    conv_t_pipe: wgpu::ComputePipeline,
}

fn bgl_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

impl WgpuAccelerator {
    /// Request a headless high-performance (or fallback) `wgpu` adapter and compile the three
    /// WGSL compute pipelines.
    pub fn try_new() -> Result<Self, MlError> {
        let (tx, rx) = std::sync::mpsc::channel();
        let spawn_res = std::thread::Builder::new().name("pc_ml_wgpu_init".into()).spawn(move || {
            let _ = tx.send(Self::init_blocking());
        });
        if let Err(e) = spawn_res {
            return Err(MlError::Backend(format!("failed to spawn wgpu init thread: {e}")));
        }
        match rx.recv_timeout(INIT_TIMEOUT) {
            Ok(res) => res,
            Err(e) => Err(MlError::Backend(format!("wgpu compute initialization timed out or failed: {e}"))),
        }
    }

    fn init_blocking() -> Result<Self, MlError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle_from_env()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            ..Default::default()
        }))
        .or_else(|_| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::None,
                force_fallback_adapter: true,
                compatible_surface: None,
                ..Default::default()
            }))
        })
        .map_err(|e| MlError::Backend(format!("no wgpu adapter available: {e}")))?;

        let adapter_info = adapter.get_info();
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("pc_ml_compute"),
            required_features: wgpu::Features::empty(),
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .map_err(|e| MlError::Backend(format!("wgpu request_device failed: {e}")))?;

        let faulted = Arc::new(AtomicBool::new(false));
        let f_lost = faulted.clone();
        device.set_device_lost_callback(move |reason, msg| {
            log::warn!("photocraft-ml wgpu device lost ({reason:?}): {msg}");
            f_lost.store(true, Ordering::Release);
        });
        let f_err = faulted.clone();
        device.on_uncaptured_error(Arc::new(move |err| {
            log::warn!("photocraft-ml wgpu uncaptured error: {err}");
            f_err.store(true, Ordering::Release);
        }));

        let matmul_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pc_ml_matmul_bgl"),
            entries: &[
                bgl_entry(0, wgpu::BufferBindingType::Uniform),
                bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let conv_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pc_ml_conv_bgl"),
            entries: &[
                bgl_entry(0, wgpu::BufferBindingType::Uniform),
                bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(4, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });

        let matmul_pipe = Self::build_pipeline(&device, "pc_ml_matmul", &matmul_layout, MATMUL_WGSL);
        let conv_pipe = Self::build_pipeline(&device, "pc_ml_conv2d", &conv_layout, CONV2D_WGSL);
        let conv_t_pipe = Self::build_pipeline(&device, "pc_ml_conv_t2d", &conv_layout, CONV_TRANSPOSE2D_WGSL);

        if faulted.load(Ordering::Acquire) {
            return Err(MlError::Backend("wgpu compute pipeline compilation faulted".into()));
        }

        Ok(Self { device, queue, limits, adapter_info, faulted, exec_lock: Mutex::new(()), matmul_layout, matmul_pipe, conv_layout, conv_pipe, conv_t_pipe })
    }

    fn build_pipeline(device: &wgpu::Device, label: &str, bgl: &wgpu::BindGroupLayout, wgsl: &str) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(wgsl.into()) });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &[Some(bgl)], immediate_size: 0 });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    }

    /// Whether the underlying `wgpu` adapter is a software CPU rasterizer (WARP / Lavapipe).
    pub fn is_software_adapter(&self) -> bool {
        self.adapter_info.device_type == wgpu::DeviceType::Cpu
    }

    /// Whether the device remains healthy (no uncaptured errors or device loss).
    pub fn is_healthy(&self) -> bool {
        !self.faulted.load(Ordering::Acquire)
    }

    fn check_storage_bytes(&self, elems: usize) -> Option<u64> {
        let bytes = u64::try_from(elems.checked_mul(4)?).ok()?;
        if bytes == 0 || bytes > self.limits.max_storage_buffer_binding_size || bytes > self.limits.max_buffer_size {
            return None;
        }
        Some(bytes)
    }

    fn create_storage_init(&self, label: &str, data: &[f32]) -> Option<wgpu::Buffer> {
        let size = self.check_storage_bytes(data.len())?;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
        Some(buf)
    }

    fn create_uniform(&self, label: &str, words: &[u32]) -> wgpu::Buffer {
        let bytes = bytemuck::cast_slice(words);
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buf, 0, bytes);
        buf
    }

    fn readback_f32(&self, out_buf: &wgpu::Buffer, out_elems: usize) -> Option<Vec<f32>> {
        let byte_len = self.check_storage_bytes(out_elems)?;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pc_ml_staging"),
            size: byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pc_ml_readback") });
        enc.copy_buffer_to_buffer(out_buf, 0, &staging, 0, byte_len);
        self.queue.submit(Some(enc.finish()));

        let mapped_ok = Arc::new(AtomicBool::new(false));
        let mapped_done = Arc::new(AtomicBool::new(false));
        let ok_clone = mapped_ok.clone();
        let done_clone = mapped_done.clone();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |res| {
            ok_clone.store(res.is_ok(), Ordering::Release);
            done_clone.store(true, Ordering::Release);
        });

        let start = Instant::now();
        while !mapped_done.load(Ordering::Acquire) {
            if !self.is_healthy() || start.elapsed() >= GPU_WAIT_TIMEOUT {
                return None;
            }
            let poll_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(Duration::from_millis(100)) })
            }));
            if poll_res.is_err() {
                self.faulted.store(true, Ordering::Release);
                return None;
            }
        }
        if !mapped_ok.load(Ordering::Acquire) || !self.is_healthy() {
            return None;
        }

        let view = staging.slice(..).get_mapped_range().ok()?;
        let floats: &[f32] = bytemuck::try_cast_slice(&view).ok()?;
        let out = floats.to_vec();
        drop(view);
        staging.unmap();
        Some(out)
    }

    fn run_conv_common(&self, spec: &Conv2dSpec, input: &[f32], weight: &[f32], bias: Option<&[f32]>, transposed: bool) -> Option<Vec<f32>> {
        if !self.is_healthy() {
            return None;
        }
        let _guard = self.exec_lock.lock().unwrap_or_else(PoisonError::into_inner);

        let out_elems = spec.batch.checked_mul(spec.out_channels)?.checked_mul(spec.out_h)?.checked_mul(spec.out_w)?;
        let out_bytes = self.check_storage_bytes(out_elems)?;
        let max_wg = self.limits.max_compute_workgroups_per_dimension as usize;
        let wg_x = spec.out_w.div_ceil(8);
        let wg_y = spec.out_h.div_ceil(8);
        let total_boc = spec.batch.checked_mul(spec.out_channels)?;
        if wg_x == 0 || wg_y == 0 || total_boc == 0 || wg_x > max_wg || wg_y > max_wg {
            return None;
        }

        let in_buf = self.create_storage_init("pc_ml_conv_in", input)?;
        let w_buf = self.create_storage_init("pc_ml_conv_w", weight)?;
        let dummy_bias = [0.0f32];
        let b_buf = self.create_storage_init("pc_ml_conv_b", bias.filter(|b| !b.is_empty()).unwrap_or(&dummy_bias))?;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pc_ml_conv_out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let pipe = if transposed { &self.conv_t_pipe } else { &self.conv_pipe };
        let mut z_off = 0usize;
        while z_off < total_boc {
            let chunk_z = (total_boc - z_off).min(max_wg);
            let words: [u32; 20] = [
                u32::try_from(spec.batch).ok()?,
                u32::try_from(spec.in_channels).ok()?,
                u32::try_from(spec.out_channels).ok()?,
                u32::try_from(spec.in_h).ok()?,
                u32::try_from(spec.in_w).ok()?,
                u32::try_from(spec.out_h).ok()?,
                u32::try_from(spec.out_w).ok()?,
                u32::try_from(spec.kernel_h).ok()?,
                u32::try_from(spec.kernel_w).ok()?,
                u32::try_from(spec.stride_h).ok()?,
                u32::try_from(spec.stride_w).ok()?,
                u32::try_from(spec.pad_top).ok()?,
                u32::try_from(spec.pad_left).ok()?,
                u32::try_from(spec.dil_h).ok()?,
                u32::try_from(spec.dil_w).ok()?,
                u32::try_from(spec.groups).ok()?,
                u32::from(bias.is_some_and(|b| !b.is_empty())),
                u32::try_from(z_off).ok()?,
                0,
                0,
            ];
            let u_buf = self.create_uniform("pc_ml_conv_u", &words);
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pc_ml_conv_bg"),
                layout: &self.conv_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: u_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: in_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: w_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: b_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: out_buf.as_entire_binding() },
                ],
            });
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pc_ml_conv_enc") });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("pc_ml_conv_pass"), timestamp_writes: None });
                pass.set_pipeline(pipe);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(wg_x as u32, wg_y as u32, chunk_z as u32);
            }
            self.queue.submit(Some(enc.finish()));
            z_off += chunk_z;
        }

        self.readback_f32(&out_buf, out_elems)
    }
}

impl ComputeAccelerator for WgpuAccelerator {
    fn name(&self) -> &'static str {
        "wgpu"
    }

    fn adapter_name(&self) -> Option<&str> {
        Some(&self.adapter_info.name)
    }

    fn is_gpu(&self) -> bool {
        true
    }

    fn matmul_batched(&self, spec: &MatMulSpec, a: &[f32], b: &[f32]) -> Option<Vec<f32>> {
        if !self.is_healthy() {
            return None;
        }
        let _guard = self.exec_lock.lock().unwrap_or_else(PoisonError::into_inner);

        let out_elems = spec.batch.checked_mul(spec.m)?.checked_mul(spec.n)?;
        let out_bytes = self.check_storage_bytes(out_elems)?;
        let max_wg = self.limits.max_compute_workgroups_per_dimension as usize;
        let wg_x = spec.n.div_ceil(16);
        let wg_y = spec.m.div_ceil(16);
        if wg_x == 0 || wg_y == 0 || spec.batch == 0 || spec.k == 0 || wg_x > max_wg || wg_y > max_wg {
            return None;
        }

        let a_buf = self.create_storage_init("pc_ml_mm_a", a)?;
        let b_buf = self.create_storage_init("pc_ml_mm_b", b)?;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pc_ml_mm_out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let mut z_off = 0usize;
        while z_off < spec.batch {
            let chunk_z = (spec.batch - z_off).min(max_wg);
            let words: [u32; 8] = [
                u32::try_from(spec.batch).ok()?,
                u32::try_from(spec.a_batch).ok()?,
                u32::try_from(spec.b_batch).ok()?,
                u32::try_from(spec.m).ok()?,
                u32::try_from(spec.k).ok()?,
                u32::try_from(spec.n).ok()?,
                u32::try_from(z_off).ok()?,
                0,
            ];
            let u_buf = self.create_uniform("pc_ml_mm_u", &words);
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pc_ml_mm_bg"),
                layout: &self.matmul_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: u_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: a_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: b_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
                ],
            });
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pc_ml_mm_enc") });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("pc_ml_mm_pass"), timestamp_writes: None });
                pass.set_pipeline(&self.matmul_pipe);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(wg_x as u32, wg_y as u32, chunk_z as u32);
            }
            self.queue.submit(Some(enc.finish()));
            z_off += chunk_z;
        }

        self.readback_f32(&out_buf, out_elems)
    }

    fn conv2d(&self, spec: &Conv2dSpec, input: &[f32], weight: &[f32], bias: Option<&[f32]>) -> Option<Vec<f32>> {
        self.run_conv_common(spec, input, weight, bias, false)
    }

    fn conv_transpose2d(&self, spec: &Conv2dSpec, input: &[f32], weight: &[f32], bias: Option<&[f32]>) -> Option<Vec<f32>> {
        self.run_conv_common(spec, input, weight, bias, true)
    }
}
