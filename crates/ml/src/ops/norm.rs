//! Normalization, `Softmax` and reduction ONNX operators (`LayerNormalization`,
//! `BatchNormalization`, `InstanceNormalization`, `GroupNormalization`, `Softmax`, `LogSoftmax`,
//! `Reduce*`, `ArgMax`, `ArgMin`).

use crate::MlError;
use crate::onnx::Node;
use crate::tensor::{DType, Tensor, broadcast_strides, checked_numel, compute_strides, resolve_axis, unravel_offset};

fn req<'a>(inputs: &'a [Option<&'a Tensor>], idx: usize, op: &str) -> Result<&'a Tensor, MlError> {
    inputs.get(idx).and_then(|t| *t).ok_or_else(|| MlError::InvalidModel(format!("{op} missing input {idx}")))
}

fn run_batch_norm(node: &Node, inputs: &[Option<&Tensor>]) -> Result<Tensor, MlError> {
    let x = req(inputs, 0, "BatchNormalization")?;
    let scale = req(inputs, 1, "BatchNormalization")?.to_f32_cow();
    let bias = req(inputs, 2, "BatchNormalization")?.to_f32_cow();
    let mean = req(inputs, 3, "BatchNormalization")?.to_f32_cow();
    let var = req(inputs, 4, "BatchNormalization")?.to_f32_cow();
    let eps = node.attr_float("epsilon").unwrap_or(1e-5);

    if x.rank() < 2 {
        return Err(MlError::InvalidShape("BatchNormalization expects rank >= 2".into()));
    }
    let (n, c) = (x.shape()[0], x.shape()[1]);
    if scale.len() != c || bias.len() != c || mean.len() != c || var.len() != c {
        return Err(MlError::InvalidShape(format!("BatchNormalization channel mismatch for C={c}")));
    }
    let spatial: usize = x.shape()[2..].iter().product();
    let src = x.to_f32_cow();
    let mut out = vec![0.0f32; src.len()];

    for ch in 0..c {
        let inv_std = 1.0 / (var[ch] + eps).max(0.0).sqrt();
        let a = scale[ch] * inv_std;
        let b = bias[ch] - mean[ch] * a;
        for batch in 0..n {
            let start = (batch * c + ch) * spatial;
            for (dst_v, &src_v) in out[start..start + spatial].iter_mut().zip(&src[start..start + spatial]) {
                *dst_v = src_v * a + b;
            }
        }
    }
    Tensor::from_f32(x.shape().to_vec(), out)
}

fn run_instance_norm(node: &Node, inputs: &[Option<&Tensor>]) -> Result<Tensor, MlError> {
    let x = req(inputs, 0, "InstanceNormalization")?;
    let scale = req(inputs, 1, "InstanceNormalization")?.to_f32_cow();
    let bias = req(inputs, 2, "InstanceNormalization")?.to_f32_cow();
    let eps = node.attr_float("epsilon").unwrap_or(1e-5);

    if x.rank() < 2 {
        return Err(MlError::InvalidShape("InstanceNormalization expects rank >= 2".into()));
    }
    let (n, c) = (x.shape()[0], x.shape()[1]);
    if scale.len() != c || bias.len() != c {
        return Err(MlError::InvalidShape(format!("InstanceNormalization channel mismatch for C={c}")));
    }
    let spatial: usize = x.shape()[2..].iter().product();
    if spatial == 0 {
        return Tensor::zeros(x.shape().to_vec(), DType::F32);
    }
    let src = x.to_f32_cow();
    let mut out = vec![0.0f32; src.len()];
    let inv_n = 1.0 / (spatial as f64);

    for batch in 0..n {
        for ch in 0..c {
            let start = (batch * c + ch) * spatial;
            let slice = &src[start..start + spatial];
            let mean = slice.iter().map(|&v| f64::from(v)).sum::<f64>() * inv_n;
            let var = slice.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() * inv_n;
            let inv_std = (1.0 / (var + f64::from(eps)).max(0.0).sqrt()) as f32;
            let a = scale[ch] * inv_std;
            let b = bias[ch] - (mean as f32) * a;
            for (dst_v, &src_v) in out[start..start + spatial].iter_mut().zip(slice) {
                *dst_v = src_v * a + b;
            }
        }
    }
    Tensor::from_f32(x.shape().to_vec(), out)
}

fn run_group_norm(node: &Node, inputs: &[Option<&Tensor>]) -> Result<Tensor, MlError> {
    let x = req(inputs, 0, "GroupNormalization")?;
    let scale = req(inputs, 1, "GroupNormalization")?.to_f32_cow();
    let bias = req(inputs, 2, "GroupNormalization")?.to_f32_cow();
    let eps = node.attr_float("epsilon").unwrap_or(1e-5);
    let groups =
        usize::try_from(node.attr_int("num_groups").unwrap_or(1)).ok().filter(|&g| g >= 1).ok_or_else(|| MlError::InvalidShape("invalid num_groups".into()))?;

    if x.rank() < 2 {
        return Err(MlError::InvalidShape("GroupNormalization expects rank >= 2".into()));
    }
    let (n, c) = (x.shape()[0], x.shape()[1]);
    if c % groups != 0 {
        return Err(MlError::InvalidShape(format!("GroupNormalization C={c} not divisible by groups={groups}")));
    }
    let c_per_g = c / groups;
    let spatial: usize = x.shape()[2..].iter().product();
    let group_elems = c_per_g * spatial;
    if group_elems == 0 {
        return Tensor::zeros(x.shape().to_vec(), DType::F32);
    }
    let src = x.to_f32_cow();
    let mut out = vec![0.0f32; src.len()];
    let per_channel_affine = scale.len() == c && bias.len() == c;

    for batch in 0..n {
        for g in 0..groups {
            let g_start = (batch * c + g * c_per_g) * spatial;
            let slice = &src[g_start..g_start + group_elems];
            let mean = slice.iter().map(|&v| f64::from(v)).sum::<f64>() / (group_elems as f64);
            let var = slice.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / (group_elems as f64);
            let inv_std = (1.0 / (var + f64::from(eps)).max(0.0).sqrt()) as f32;
            let m32 = mean as f32;
            for ci in 0..c_per_g {
                let ch = g * c_per_g + ci;
                let sc = if per_channel_affine { scale[ch] } else { *scale.get(g).unwrap_or(&1.0) };
                let bi = if per_channel_affine { bias[ch] } else { *bias.get(g).unwrap_or(&0.0) };
                let c_start = (batch * c + ch) * spatial;
                for (dst_v, &src_v) in out[c_start..c_start + spatial].iter_mut().zip(&src[c_start..c_start + spatial]) {
                    *dst_v = (src_v - m32) * inv_std * sc + bi;
                }
            }
        }
    }
    Tensor::from_f32(x.shape().to_vec(), out)
}

fn run_layer_norm(node: &Node, inputs: &[Option<&Tensor>]) -> Result<Tensor, MlError> {
    let x = req(inputs, 0, "LayerNormalization")?;
    let scale_t = req(inputs, 1, "LayerNormalization")?;
    let bias_t = inputs.get(2).and_then(|t| *t);
    let eps = node.attr_float("epsilon").unwrap_or(1e-5);
    let axis = resolve_axis(node.attr_int("axis").unwrap_or(-1), x.rank())?;

    let outer: usize = x.shape()[..axis].iter().product();
    let norm_size: usize = x.shape()[axis..].iter().product();
    if outer == 0 || norm_size == 0 {
        return Tensor::zeros(x.shape().to_vec(), DType::F32);
    }

    let src = x.to_f32_cow();
    let scale = scale_t.to_f32_cow();
    let bias = bias_t.map(Tensor::to_f32_cow);
    let mut out = vec![0.0f32; src.len()];
    let inv_n = 1.0 / (norm_size as f64);

    // Fast path when scale (and optional bias) has exact length `norm_size`.
    let direct_scale = scale.len() == norm_size;
    let direct_bias = bias.as_ref().is_none_or(|b| b.len() == norm_size);
    if direct_scale && direct_bias {
        for o in 0..outer {
            let start = o * norm_size;
            let row = &src[start..start + norm_size];
            let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() * inv_n;
            let var = row.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() * inv_n;
            let inv_std = (1.0 / (var + f64::from(eps)).max(0.0).sqrt()) as f32;
            let m32 = mean as f32;
            let dst_row = &mut out[start..start + norm_size];
            if let Some(b) = &bias {
                for i in 0..norm_size {
                    dst_row[i] = (row[i] - m32) * inv_std * scale[i] + b[i];
                }
            } else {
                for i in 0..norm_size {
                    dst_row[i] = (row[i] - m32) * inv_std * scale[i];
                }
            }
        }
        return Tensor::from_f32(x.shape().to_vec(), out);
    }

    // General broadcasted scale/bias path.
    let out_strides = compute_strides(x.shape());
    let s_aligned = broadcast_strides(scale_t.shape(), scale_t.strides(), x.shape())?;
    let b_aligned = match bias_t {
        Some(bt) => Some(broadcast_strides(bt.shape(), bt.strides(), x.shape())?),
        None => None,
    };
    let rank = x.rank();
    for o in 0..outer {
        let start = o * norm_size;
        let row = &src[start..start + norm_size];
        let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() * inv_n;
        let var = row.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() * inv_n;
        let inv_std = (1.0 / (var + f64::from(eps)).max(0.0).sqrt()) as f32;
        let m32 = mean as f32;
        for i in 0..norm_size {
            let idx = start + i;
            let sc = scale[unravel_offset(idx, &out_strides, rank, &s_aligned)];
            let bi = match (&bias, &b_aligned) {
                (Some(b), Some(ba)) => b[unravel_offset(idx, &out_strides, rank, ba)],
                _ => 0.0,
            };
            out[idx] = (src[idx] - m32) * inv_std * sc + bi;
        }
    }
    Tensor::from_f32(x.shape().to_vec(), out)
}

fn run_softmax(node: &Node, x: &Tensor, log_mode: bool) -> Result<Tensor, MlError> {
    if x.rank() == 0 {
        return Tensor::from_f32(Vec::new(), vec![if log_mode { 0.0 } else { 1.0 }]);
    }
    let axis = resolve_axis(node.attr_int("axis").unwrap_or(-1), x.rank())?;
    let dim = x.shape()[axis];
    if x.numel() == 0 || dim == 0 {
        return Tensor::zeros(x.shape().to_vec(), DType::F32);
    }
    let outer: usize = x.shape()[..axis].iter().product();
    let inner: usize = x.shape()[axis + 1..].iter().product();
    let src = x.to_f32_cow();
    let mut out = vec![0.0f32; src.len()];

    for o in 0..outer {
        for r in 0..inner {
            let mut max_v = f32::NEG_INFINITY;
            for d in 0..dim {
                max_v = max_v.max(src[(o * dim + d) * inner + r]);
            }
            let mut sum = 0.0f64;
            for d in 0..dim {
                let e = (src[(o * dim + d) * inner + r] - max_v).exp();
                out[(o * dim + d) * inner + r] = e;
                sum += f64::from(e);
            }
            if log_mode {
                let log_sum = (sum.max(1e-38).ln()) as f32;
                for d in 0..dim {
                    let pos = (o * dim + d) * inner + r;
                    out[pos] = (src[pos] - max_v) - log_sum;
                }
            } else {
                let inv = (1.0 / sum.max(1e-38)) as f32;
                for d in 0..dim {
                    out[(o * dim + d) * inner + r] *= inv;
                }
            }
        }
    }
    Tensor::from_f32(x.shape().to_vec(), out)
}

#[derive(Clone, Copy)]
enum ReduceKind {
    Mean,
    Sum,
    Max,
    Min,
    Prod,
    L1,
    L2,
    LogSumExp,
}

fn resolve_reduce_axes(node: &Node, inputs: &[Option<&Tensor>], rank: usize) -> Result<Option<Vec<bool>>, MlError> {
    let noop = node.attr_int("noop_with_empty_axes").unwrap_or(0) != 0;
    let raw_axes: Option<Vec<i64>> = inputs.get(1).and_then(|t| *t).map(|t| t.to_i64_cow().into_owned()).or_else(|| node.attr_ints("axes").map(|a| a.to_vec()));

    match raw_axes {
        Some(list) if list.is_empty() => {
            if noop {
                Ok(None)
            } else {
                Ok(Some(vec![true; rank]))
            }
        }
        Some(list) => {
            let mut flags = vec![false; rank];
            for a in list {
                flags[resolve_axis(a, rank)?] = true;
            }
            Ok(Some(flags))
        }
        None => {
            if noop {
                Ok(None)
            } else {
                Ok(Some(vec![true; rank]))
            }
        }
    }
}

fn run_reduce(node: &Node, inputs: &[Option<&Tensor>], kind: ReduceKind) -> Result<Tensor, MlError> {
    let x = req(inputs, 0, &node.op_type)?;
    let rank = x.rank();
    let Some(reduce_mask) = resolve_reduce_axes(node, inputs, rank)? else {
        return Ok(x.clone());
    };
    let keepdims = node.attr_int("keepdims").unwrap_or(1) != 0;

    let mut kept_shape = Vec::with_capacity(rank);
    let mut final_shape = Vec::with_capacity(rank);
    let mut reduced_count = 1usize;
    for (ax, &d) in x.shape().iter().enumerate() {
        if reduce_mask[ax] {
            kept_shape.push(1);
            reduced_count = reduced_count.saturating_mul(d);
            if keepdims {
                final_shape.push(1);
            }
        } else {
            kept_shape.push(d);
            final_shape.push(d);
        }
    }

    let out_numel = checked_numel(&final_shape, x.dtype())?;
    if out_numel == 0 || reduced_count == 0 {
        return Tensor::zeros(final_shape, x.dtype());
    }

    if x.dtype() == DType::I64 && matches!(kind, ReduceKind::Sum | ReduceKind::Max | ReduceKind::Min | ReduceKind::Prod) {
        let init = match kind {
            ReduceKind::Max => i64::MIN,
            ReduceKind::Min => i64::MAX,
            ReduceKind::Prod => 1,
            _ => 0,
        };
        let mut acc = vec![init; out_numel];
        let src = x.to_i64_cow();
        let kept_strides = compute_strides(&kept_shape);
        let src_strides = x.strides();
        for (i, &val) in src.iter().enumerate() {
            let mut rem = i;
            let mut dst_idx = 0usize;
            for ax in 0..rank {
                let c = rem / src_strides[ax];
                rem %= src_strides[ax];
                if !reduce_mask[ax] {
                    dst_idx += c * kept_strides[ax];
                }
            }
            acc[dst_idx] = match kind {
                ReduceKind::Max => acc[dst_idx].max(val),
                ReduceKind::Min => acc[dst_idx].min(val),
                ReduceKind::Prod => acc[dst_idx].wrapping_mul(val),
                _ => acc[dst_idx].wrapping_add(val),
            };
        }
        return Tensor::from_i64(final_shape, acc);
    }

    let init = match kind {
        ReduceKind::Max | ReduceKind::LogSumExp => f64::NEG_INFINITY,
        ReduceKind::Min => f64::INFINITY,
        ReduceKind::Prod => 1.0,
        _ => 0.0,
    };
    let mut acc = vec![init; out_numel];
    let src = x.to_f32_cow();
    let kept_strides = compute_strides(&kept_shape);
    let src_strides = x.strides();
    let map_dst = |mut rem: usize| -> usize {
        let mut dst_idx = 0usize;
        for ax in 0..rank {
            let c = rem / src_strides[ax];
            rem %= src_strides[ax];
            if !reduce_mask[ax] {
                dst_idx += c * kept_strides[ax];
            }
        }
        dst_idx
    };

    if matches!(kind, ReduceKind::LogSumExp) {
        for (i, &val) in src.iter().enumerate() {
            let d = map_dst(i);
            acc[d] = acc[d].max(f64::from(val));
        }
        let mut sums = vec![0.0f64; out_numel];
        for (i, &val) in src.iter().enumerate() {
            let d = map_dst(i);
            sums[d] += (f64::from(val) - acc[d]).exp();
        }
        let out: Vec<f32> = acc.into_iter().zip(sums).map(|(m, s)| (m + s.max(1e-38).ln()) as f32).collect();
        return Tensor::from_f32(final_shape, out);
    }

    for (i, &val) in src.iter().enumerate() {
        let d = map_dst(i);
        let v = f64::from(val);
        acc[d] = match kind {
            ReduceKind::Mean | ReduceKind::Sum => acc[d] + v,
            ReduceKind::Max => acc[d].max(v),
            ReduceKind::Min => acc[d].min(v),
            ReduceKind::Prod => acc[d] * v,
            ReduceKind::L1 => acc[d] + v.abs(),
            ReduceKind::L2 => acc[d] + v * v,
            ReduceKind::LogSumExp => acc[d],
        };
    }

    let inv = 1.0 / (reduced_count as f64);
    let out: Vec<f32> = acc
        .into_iter()
        .map(|a| match kind {
            ReduceKind::Mean => (a * inv) as f32,
            ReduceKind::L2 => a.sqrt() as f32,
            _ => a as f32,
        })
        .collect();
    Tensor::from_f32(final_shape, out)
}

fn run_arg_extrema(node: &Node, x: &Tensor, is_max: bool) -> Result<Tensor, MlError> {
    let rank = x.rank();
    if rank == 0 {
        return Tensor::from_i64(Vec::new(), vec![0]);
    }
    let axis = resolve_axis(node.attr_int("axis").unwrap_or(0), rank)?;
    let keepdims = node.attr_int("keepdims").unwrap_or(1) != 0;
    let select_last = node.attr_int("select_last_index").unwrap_or(0) != 0;
    let dim = x.shape()[axis];
    if dim == 0 {
        return Err(MlError::InvalidShape("ArgMax/ArgMin cannot reduce a zero-sized axis".into()));
    }

    let mut out_shape = Vec::with_capacity(rank);
    for (i, &d) in x.shape().iter().enumerate() {
        if i == axis {
            if keepdims {
                out_shape.push(1);
            }
        } else {
            out_shape.push(d);
        }
    }

    let outer: usize = x.shape()[..axis].iter().product();
    let inner: usize = x.shape()[axis + 1..].iter().product();
    let src = x.to_f32_cow();
    let mut out = vec![0i64; outer * inner];

    for o in 0..outer {
        for r in 0..inner {
            let mut best_v = src[o * dim * inner + r];
            let mut best_i = 0usize;
            for d in 1..dim {
                let v = src[(o * dim + d) * inner + r];
                let better = if is_max {
                    if select_last { v >= best_v } else { v > best_v }
                } else if select_last {
                    v <= best_v
                } else {
                    v < best_v
                };
                if better {
                    best_v = v;
                    best_i = d;
                }
            }
            out[o * inner + r] = best_i as i64;
        }
    }
    Tensor::from_i64(out_shape, out)
}

/// Try executing a normalization, softmax or reduction ONNX node.
pub fn try_run(node: &Node, inputs: &[Option<&Tensor>]) -> Result<Option<Tensor>, MlError> {
    let op = node.op_type.as_str();
    let res = match op {
        "BatchNormalization" => run_batch_norm(node, inputs)?,
        "InstanceNormalization" => run_instance_norm(node, inputs)?,
        "GroupNormalization" => run_group_norm(node, inputs)?,
        "LayerNormalization" => run_layer_norm(node, inputs)?,
        "Softmax" => run_softmax(node, req(inputs, 0, op)?, false)?,
        "LogSoftmax" => run_softmax(node, req(inputs, 0, op)?, true)?,
        "ReduceMean" => run_reduce(node, inputs, ReduceKind::Mean)?,
        "ReduceSum" => run_reduce(node, inputs, ReduceKind::Sum)?,
        "ReduceMax" => run_reduce(node, inputs, ReduceKind::Max)?,
        "ReduceMin" => run_reduce(node, inputs, ReduceKind::Min)?,
        "ReduceProd" => run_reduce(node, inputs, ReduceKind::Prod)?,
        "ReduceL1" => run_reduce(node, inputs, ReduceKind::L1)?,
        "ReduceL2" => run_reduce(node, inputs, ReduceKind::L2)?,
        "ReduceLogSumExp" => run_reduce(node, inputs, ReduceKind::LogSumExp)?,
        "ArgMax" => run_arg_extrema(node, req(inputs, 0, op)?, true)?,
        "ArgMin" => run_arg_extrema(node, req(inputs, 0, op)?, false)?,
        _ => return Ok(None),
    };
    Ok(Some(res))
}
