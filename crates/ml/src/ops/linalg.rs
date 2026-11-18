//! Linear algebra ONNX operators (`MatMul`, `Gemm`, `Einsum`) with optional `wgpu` offload and
//! cache-blocked `rayon` CPU execution.

#[cfg(not(target_arch = "wasm32"))]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(not(target_arch = "wasm32"))]
use rayon::slice::ParallelSliceMut;

use crate::MlError;
use crate::backend::{ComputeAccelerator, MatMulSpec};
use crate::onnx::Node;
use crate::tensor::{DType, MAX_RANK, Tensor, broadcast_shape, broadcast_strides, checked_numel, compute_strides, unravel_offset};

fn req<'a>(inputs: &'a [Option<&'a Tensor>], idx: usize, op: &str) -> Result<&'a Tensor, MlError> {
    inputs.get(idx).and_then(|t| *t).ok_or_else(|| MlError::InvalidModel(format!("{op} missing input {idx}")))
}

/// Cache-friendly row-major `C[m, n] = A[m, k] * B[k, n]` for a single batch slice.
#[inline]
pub(crate) fn sgemm_slice(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], out: &mut [f32]) {
    out.fill(0.0);
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let out_row = &mut out[i * n..(i + 1) * n];
        for (p, &a_ip) in a_row.iter().enumerate() {
            if a_ip == 0.0 {
                continue;
            }
            let b_row = &b[p * n..(p + 1) * n];
            for (dst, &b_pj) in out_row.iter_mut().zip(b_row.iter()) {
                *dst += a_ip * b_pj;
            }
        }
    }
}

fn transpose_2d(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    dst
}

#[derive(Clone, Copy)]
struct BatchLayout<'a> {
    out_strides: &'a [usize],
    batch_rank: usize,
    a_aligned: [usize; MAX_RANK],
    b_aligned: [usize; MAX_RANK],
    m: usize,
    k: usize,
    n: usize,
}

fn compute_batch_cpu(a: &[f32], b: &[f32], batch_total: usize, layout: BatchLayout<'_>) -> Result<Vec<f32>, MlError> {
    let plane_out = layout.m.checked_mul(layout.n).ok_or_else(|| MlError::InvalidShape("MatMul plane overflow".into()))?;
    let total_out = batch_total.checked_mul(plane_out).ok_or_else(|| MlError::InvalidShape("MatMul output overflow".into()))?;
    let mut out = vec![0.0f32; total_out];
    let mk = layout.m * layout.k;
    let kn = layout.k * layout.n;

    let run_plane = |b_idx: usize, dst_plane: &mut [f32]| {
        let a_b = if layout.batch_rank == 0 { 0 } else { unravel_offset(b_idx, layout.out_strides, layout.batch_rank, &layout.a_aligned) };
        let b_b = if layout.batch_rank == 0 { 0 } else { unravel_offset(b_idx, layout.out_strides, layout.batch_rank, &layout.b_aligned) };
        let a_plane = &a[a_b * mk..(a_b + 1) * mk];
        let b_plane = &b[b_b * kn..(b_b + 1) * kn];
        sgemm_slice(layout.m, layout.k, layout.n, a_plane, b_plane, dst_plane);
    };

    #[cfg(not(target_arch = "wasm32"))]
    if batch_total > 1 && total_out.saturating_mul(layout.k) >= 32_768 {
        out.par_chunks_mut(plane_out).enumerate().for_each(|(b_idx, dst_plane)| run_plane(b_idx, dst_plane));
        return Ok(out);
    }

    #[cfg(not(target_arch = "wasm32"))]
    if batch_total == 1 && layout.m >= 16 && total_out.saturating_mul(layout.k) >= 32_768 {
        let a_plane = &a[..mk];
        let b_plane = &b[..kn];
        out.par_chunks_mut(layout.n).enumerate().for_each(|(row, out_row)| {
            let a_row = &a_plane[row * layout.k..(row + 1) * layout.k];
            for (p, &a_ip) in a_row.iter().enumerate() {
                if a_ip == 0.0 {
                    continue;
                }
                let b_row = &b_plane[p * layout.n..(p + 1) * layout.n];
                for (dst, &b_pj) in out_row.iter_mut().zip(b_row.iter()) {
                    *dst += a_ip * b_pj;
                }
            }
        });
        return Ok(out);
    }

    for (b_idx, dst_plane) in out.chunks_mut(plane_out).enumerate() {
        run_plane(b_idx, dst_plane);
    }
    Ok(out)
}

fn run_matmul(a_in: &Tensor, b_in: &Tensor, accel: &dyn ComputeAccelerator) -> Result<Tensor, MlError> {
    if a_in.rank() == 0 || b_in.rank() == 0 {
        return Err(MlError::InvalidShape("MatMul does not accept scalar (rank-0) inputs".into()));
    }
    let prepend_a = a_in.rank() == 1;
    let append_b = b_in.rank() == 1;

    let a_shape: Vec<usize> = if prepend_a { vec![1, a_in.shape()[0]] } else { a_in.shape().to_vec() };
    let b_shape: Vec<usize> = if append_b { vec![b_in.shape()[0], 1] } else { b_in.shape().to_vec() };

    let m = a_shape[a_shape.len() - 2];
    let k_a = a_shape[a_shape.len() - 1];
    let k_b = b_shape[b_shape.len() - 2];
    let n = b_shape[b_shape.len() - 1];
    if k_a != k_b {
        return Err(MlError::InvalidShape(format!("MatMul inner dimension mismatch: {k_a} vs {k_b} ({:?} @ {:?})", a_in.shape(), b_in.shape())));
    }

    let batch_a = &a_shape[..a_shape.len() - 2];
    let batch_b = &b_shape[..b_shape.len() - 2];
    let batch_out = broadcast_shape(batch_a, batch_b)?;

    let mut final_shape = batch_out.clone();
    if !prepend_a {
        final_shape.push(m);
    }
    if !append_b {
        final_shape.push(n);
    }

    let out_numel = checked_numel(&final_shape, DType::F32)?;
    if out_numel == 0 || k_a == 0 {
        return Tensor::zeros(final_shape, DType::F32);
    }

    let batch_total: usize = batch_out.iter().product();
    let a_batch_prod: usize = batch_a.iter().product();
    let b_batch_prod: usize = batch_b.iter().product();
    let a_data = a_in.to_f32_cow();
    let b_data = b_in.to_f32_cow();

    let simple_batch_broadcast = (a_batch_prod == batch_total || a_batch_prod == 1) && (b_batch_prod == batch_total || b_batch_prod == 1);
    if simple_batch_broadcast && accel.is_gpu() {
        let spec = MatMulSpec { batch: batch_total, a_batch: a_batch_prod, b_batch: b_batch_prod, m, k: k_a, n };
        if let Some(gpu_out) = accel.matmul_batched(&spec, &a_data, &b_data) {
            return Tensor::from_f32(final_shape, gpu_out);
        }
    }

    let batch_out_strides = compute_strides(&batch_out);
    let a_batch_strides = compute_strides(batch_a);
    let b_batch_strides = compute_strides(batch_b);
    let a_aligned = broadcast_strides(batch_a, &a_batch_strides, &batch_out)?;
    let b_aligned = broadcast_strides(batch_b, &b_batch_strides, &batch_out)?;

    let layout = BatchLayout { out_strides: &batch_out_strides, batch_rank: batch_out.len(), a_aligned, b_aligned, m, k: k_a, n };
    let out = compute_batch_cpu(&a_data, &b_data, batch_total, layout)?;
    Tensor::from_f32(final_shape, out)
}

fn run_gemm(node: &Node, inputs: &[Option<&Tensor>], accel: &dyn ComputeAccelerator) -> Result<Tensor, MlError> {
    let a = req(inputs, 0, "Gemm")?;
    let b = req(inputs, 1, "Gemm")?;
    if a.rank() != 2 || b.rank() != 2 {
        return Err(MlError::InvalidShape(format!("Gemm expects 2D matrices, got {:?} and {:?}", a.shape(), b.shape())));
    }
    let trans_a = node.attr_int("transA").unwrap_or(0) != 0;
    let trans_b = node.attr_int("transB").unwrap_or(0) != 0;
    let alpha = node.attr_float("alpha").unwrap_or(1.0);
    let beta = node.attr_float("beta").unwrap_or(1.0);

    let (m, k_a) = if trans_a { (a.shape()[1], a.shape()[0]) } else { (a.shape()[0], a.shape()[1]) };
    let (k_b, n) = if trans_b { (b.shape()[1], b.shape()[0]) } else { (b.shape()[0], b.shape()[1]) };
    if k_a != k_b {
        return Err(MlError::InvalidShape(format!("Gemm inner dimension mismatch: {k_a} vs {k_b}")));
    }

    let a_src = a.to_f32_cow();
    let b_src = b.to_f32_cow();
    let a_mat = if trans_a { std::borrow::Cow::Owned(transpose_2d(&a_src, a.shape()[0], a.shape()[1])) } else { a_src };
    let b_mat = if trans_b { std::borrow::Cow::Owned(transpose_2d(&b_src, b.shape()[0], b.shape()[1])) } else { b_src };

    let out_shape = vec![m, n];
    let total = checked_numel(&out_shape, DType::F32)?;
    if total == 0 {
        return Tensor::zeros(out_shape, DType::F32);
    }

    let mut out = if k_a == 0 {
        vec![0.0f32; total]
    } else if accel.is_gpu() {
        let spec = MatMulSpec { batch: 1, a_batch: 1, b_batch: 1, m, k: k_a, n };
        accel.matmul_batched(&spec, &a_mat, &b_mat).unwrap_or_else(|| {
            let mut cpu = vec![0.0f32; total];
            sgemm_slice(m, k_a, n, &a_mat, &b_mat, &mut cpu);
            cpu
        })
    } else {
        let mut cpu = vec![0.0f32; total];
        sgemm_slice(m, k_a, n, &a_mat, &b_mat, &mut cpu);
        cpu
    };

    if (alpha - 1.0).abs() > f32::EPSILON {
        for v in &mut out {
            *v *= alpha;
        }
    }

    if let Some(c) = inputs.get(2).and_then(|t| *t)
        && beta != 0.0
    {
        let c_data = c.to_f32_cow();
        let out_strides = compute_strides(&out_shape);
        let c_aligned = broadcast_strides(c.shape(), c.strides(), &out_shape)?;
        for (idx, dst) in out.iter_mut().enumerate() {
            let c_off = unravel_offset(idx, &out_strides, 2, &c_aligned);
            *dst += beta * c_data[c_off];
        }
    }

    Tensor::from_f32(out_shape, out)
}

fn run_einsum(node: &Node, inputs: &[Option<&Tensor>]) -> Result<Tensor, MlError> {
    let eq_raw = node.attr_str("equation").ok_or_else(|| MlError::InvalidModel("Einsum missing equation".into()))?;
    let eq: String = eq_raw.chars().filter(|c| !c.is_whitespace()).collect();
    let (lhs, rhs_opt) = match eq.split_once("->") {
        Some((l, r)) => (l, Some(r)),
        None => (eq.as_str(), None),
    };
    let terms: Vec<&str> = lhs.split(',').collect();
    if terms.len() != inputs.len() {
        return Err(MlError::InvalidModel(format!("Einsum equation `{eq_raw}` expects {} inputs, got {}", terms.len(), inputs.len())));
    }

    let mut dim_sizes: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
    let mut label_order: Vec<char> = Vec::new();
    let mut counts: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
    let mut term_labels: Vec<Vec<char>> = Vec::with_capacity(terms.len());

    for (ti, &term) in terms.iter().enumerate() {
        let t = req(inputs, ti, "Einsum")?;
        let chars: Vec<char> = term.chars().collect();
        if chars.len() != t.rank() {
            return Err(MlError::InvalidShape(format!("Einsum term `{term}` rank mismatch for {:?}", t.shape())));
        }
        for (&ch, &dim) in chars.iter().zip(t.shape().iter()) {
            if let Some(&prev) = dim_sizes.get(&ch) {
                if prev != dim {
                    return Err(MlError::InvalidShape(format!("Einsum subscript `{ch}` size mismatch: {prev} vs {dim}")));
                }
            } else {
                dim_sizes.insert(ch, dim);
                label_order.push(ch);
            }
            *counts.entry(ch).or_insert(0) += 1;
        }
        term_labels.push(chars);
    }

    let out_labels: Vec<char> = match rhs_opt {
        Some(r) => r.chars().collect(),
        None => {
            let mut v: Vec<char> = label_order.iter().copied().filter(|c| counts.get(c) == Some(&1)).collect();
            v.sort_unstable();
            v
        }
    };
    let sum_labels: Vec<char> = label_order.iter().copied().filter(|c| !out_labels.contains(c)).collect();
    let out_shape: Vec<usize> = out_labels.iter().map(|c| dim_sizes.get(c).copied().unwrap_or(0)).collect();
    let sum_shape: Vec<usize> = sum_labels.iter().map(|c| dim_sizes.get(c).copied().unwrap_or(0)).collect();

    let out_total = checked_numel(&out_shape, DType::F32)?;
    if out_total == 0 {
        return Tensor::zeros(out_shape, DType::F32);
    }
    let sum_total: usize = sum_shape.iter().product();
    if sum_total == 0 {
        return Tensor::zeros(out_shape, DType::F32);
    }

    let tensors_f32: Vec<std::borrow::Cow<'_, [f32]>> = (0..terms.len())
        .map(|i| inputs[i].map(|t| t.to_f32_cow()))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| MlError::InvalidModel("Einsum missing input".into()))?;
    let out_strides = compute_strides(&out_shape);
    let sum_strides = compute_strides(&sum_shape);
    let mut out = vec![0.0f32; out_total];

    let mut coord_map = [0usize; 128];
    for (out_idx, dst) in out.iter_mut().enumerate() {
        let mut rem = out_idx;
        for (ax, &ch) in out_labels.iter().enumerate() {
            let s = out_strides[ax];
            if (ch as usize) < 128 {
                coord_map[ch as usize] = rem / s;
            }
            rem %= s;
        }
        let mut acc = 0.0f32;
        for s_idx in 0..sum_total {
            let mut s_rem = s_idx;
            for (ax, &ch) in sum_labels.iter().enumerate() {
                let s = sum_strides[ax];
                if (ch as usize) < 128 {
                    coord_map[ch as usize] = s_rem / s;
                }
                s_rem %= s;
            }
            let mut prod = 1.0f32;
            for (ti, labels) in term_labels.iter().enumerate() {
                let Some(t) = inputs[ti] else { continue };
                let strides = t.strides();
                let mut off = 0usize;
                for (ax, &ch) in labels.iter().enumerate() {
                    let c = if (ch as usize) < 128 { coord_map[ch as usize] } else { 0 };
                    off += c * strides[ax];
                }
                prod *= tensors_f32[ti][off];
            }
            acc += prod;
        }
        *dst = acc;
    }

    Tensor::from_f32(out_shape, out)
}

/// Try executing a linear-algebra ONNX node (`MatMul`, `Gemm`, `Einsum`).
pub fn try_run(node: &Node, inputs: &[Option<&Tensor>], accel: &dyn ComputeAccelerator) -> Result<Option<Tensor>, MlError> {
    let op = node.op_type.as_str();
    let res = match op {
        "MatMul" => run_matmul(req(inputs, 0, op)?, req(inputs, 1, op)?, accel)?,
        "Gemm" => run_gemm(node, inputs, accel)?,
        "Einsum" => run_einsum(node, inputs)?,
        _ => return Ok(None),
    };
    Ok(Some(res))
}
