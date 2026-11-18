//! Shape, indexing, structural and routing ONNX operators (`Shape`, `Gather`, `Slice`, `Concat`,
//! `Split`, `Reshape`, `Transpose`, `Squeeze`, `Unsqueeze`, `Expand`, `Tile`, `Pad`,
//! `ConstantOfShape`, `Range`, `OneHot`, `ScatterND`, `TopK`, `DepthToSpace`, `SpaceToDepth`).

use crate::MlError;
use crate::onnx::{Attr, Node};
use crate::tensor::{DType, MAX_RANK, Tensor, TensorData, broadcast_strides, checked_numel, compute_strides, resolve_axis, unravel_offset};

fn req<'a>(inputs: &'a [Option<&'a Tensor>], idx: usize, op: &str) -> Result<&'a Tensor, MlError> {
    inputs.get(idx).and_then(|t| *t).ok_or_else(|| MlError::InvalidModel(format!("{op} missing input {idx}")))
}

fn run_constant(node: &Node) -> Result<Tensor, MlError> {
    if let Some(t) = node.attr_tensor("value") {
        return Ok(t.clone());
    }
    if let Some(f) = node.attr_float("value_float") {
        return Tensor::from_f32(Vec::new(), vec![f]);
    }
    if let Some(i) = node.attr_int("value_int") {
        return Tensor::from_i64(Vec::new(), vec![i]);
    }
    if let Some(fs) = node.attr_floats("value_floats") {
        return Tensor::from_f32(vec![fs.len()], fs.to_vec());
    }
    if let Some(is) = node.attr_ints("value_ints") {
        return Tensor::from_i64(vec![is.len()], is.to_vec());
    }
    Err(MlError::InvalidModel("Constant node has no supported value attribute".into()))
}

fn run_shape(node: &Node, x: &Tensor) -> Result<Tensor, MlError> {
    let rank = x.rank() as i64;
    let mut start = node.attr_int("start").unwrap_or(0);
    let mut end = node.attr_int("end").unwrap_or(rank);
    if start < 0 {
        start = (start + rank).max(0);
    }
    if end < 0 {
        end = (end + rank).max(0);
    }
    let s = (start.clamp(0, rank)) as usize;
    let e = (end.clamp(s as i64, rank)) as usize;
    let dims: Vec<i64> = x.shape()[s..e].iter().map(|&d| d as i64).collect();
    Tensor::from_i64(vec![dims.len()], dims)
}

fn run_reshape(node: &Node, data: &Tensor, shape_tensor: &Tensor) -> Result<Tensor, MlError> {
    let allowzero = node.attr_int("allowzero").unwrap_or(0) != 0;
    let raw = shape_tensor.to_i64_cow();
    let mut new_shape = Vec::with_capacity(raw.len());
    let mut infer_idx: Option<usize> = None;
    let mut known_prod = 1usize;

    for (i, &d) in raw.iter().enumerate() {
        if d == -1 {
            if infer_idx.is_some() {
                return Err(MlError::InvalidShape("Reshape allows at most one -1 dimension".into()));
            }
            infer_idx = Some(i);
            new_shape.push(1);
        } else if d == 0 && !allowzero {
            let copied = *data.shape().get(i).ok_or_else(|| MlError::InvalidShape(format!("Reshape 0-dim at {i} out of bounds for {:?}", data.shape())))?;
            known_prod = known_prod.checked_mul(copied).ok_or_else(|| MlError::InvalidShape("Reshape overflow".into()))?;
            new_shape.push(copied);
        } else if d >= 0 {
            let u = d as usize;
            known_prod = known_prod.checked_mul(u).ok_or_else(|| MlError::InvalidShape("Reshape overflow".into()))?;
            new_shape.push(u);
        } else {
            return Err(MlError::InvalidShape(format!("invalid Reshape dimension {d}")));
        }
    }

    if let Some(idx) = infer_idx {
        if known_prod == 0 {
            new_shape[idx] = 0;
        } else {
            if !data.numel().is_multiple_of(known_prod) {
                return Err(MlError::InvalidShape(format!("cannot infer Reshape dim for {} elements and known product {known_prod}", data.numel())));
            }
            new_shape[idx] = data.numel() / known_prod;
        }
    }
    data.clone().into_reshaped(new_shape)
}

fn run_flatten(node: &Node, x: &Tensor) -> Result<Tensor, MlError> {
    let rank = x.rank();
    let axis_raw = node.attr_int("axis").unwrap_or(1);
    let r = rank as i64;
    let axis = if axis_raw < 0 { axis_raw + r } else { axis_raw };
    if axis < 0 || axis > r {
        return Err(MlError::InvalidShape(format!("Flatten axis {axis_raw} out of range for rank {rank}")));
    }
    let ax = axis as usize;
    let d0: usize = x.shape()[..ax].iter().product();
    let d1: usize = x.shape()[ax..].iter().product();
    x.clone().into_reshaped(vec![d0, d1])
}

fn run_squeeze(node: &Node, inputs: &[Option<&Tensor>]) -> Result<Tensor, MlError> {
    let x = req(inputs, 0, "Squeeze")?;
    let axes_opt: Option<Vec<i64>> = inputs.get(1).and_then(|t| *t).map(|t| t.to_i64_cow().into_owned()).or_else(|| node.attr_ints("axes").map(|s| s.to_vec()));

    let mut new_shape = Vec::with_capacity(x.rank());
    if let Some(axes) = axes_opt {
        let mut drop = vec![false; x.rank()];
        for a in axes {
            let resolved = resolve_axis(a, x.rank())?;
            if x.shape()[resolved] != 1 {
                return Err(MlError::InvalidShape(format!("cannot Squeeze axis {resolved} of size {}", x.shape()[resolved])));
            }
            drop[resolved] = true;
        }
        for (i, &d) in x.shape().iter().enumerate() {
            if !drop[i] {
                new_shape.push(d);
            }
        }
    } else {
        for &d in x.shape() {
            if d != 1 {
                new_shape.push(d);
            }
        }
    }
    x.clone().into_reshaped(new_shape)
}

fn run_unsqueeze(node: &Node, inputs: &[Option<&Tensor>]) -> Result<Tensor, MlError> {
    let x = req(inputs, 0, "Unsqueeze")?;
    let axes: Vec<i64> = inputs
        .get(1)
        .and_then(|t| *t)
        .map(|t| t.to_i64_cow().into_owned())
        .or_else(|| node.attr_ints("axes").map(|s| s.to_vec()))
        .ok_or_else(|| MlError::InvalidModel("Unsqueeze missing axes".into()))?;

    let out_rank = x.rank().saturating_add(axes.len());
    if out_rank > MAX_RANK {
        return Err(MlError::InvalidShape(format!("Unsqueeze rank {out_rank} exceeds {MAX_RANK}")));
    }
    let mut is_new = vec![false; out_rank];
    for a in axes {
        let idx = resolve_axis(a, out_rank)?;
        if is_new[idx] {
            return Err(MlError::InvalidShape(format!("duplicate Unsqueeze axis {a}")));
        }
        is_new[idx] = true;
    }
    let mut out_shape = Vec::with_capacity(out_rank);
    let mut src_iter = x.shape().iter().copied();
    for inserted in is_new {
        if inserted {
            out_shape.push(1);
        } else if let Some(d) = src_iter.next() {
            out_shape.push(d);
        }
    }
    x.clone().into_reshaped(out_shape)
}

fn run_transpose(node: &Node, x: &Tensor) -> Result<Tensor, MlError> {
    let rank = x.rank();
    if rank <= 1 {
        return Ok(x.clone());
    }
    let perm: Vec<usize> = match node.attr_ints("perm") {
        Some(ints) => {
            if ints.len() != rank {
                return Err(MlError::InvalidShape(format!("Transpose perm {:?} does not match rank {rank}", ints)));
            }
            let mut seen = vec![false; rank];
            let mut p = Vec::with_capacity(rank);
            for &a in ints {
                let u = usize::try_from(a).ok().filter(|&u| u < rank).ok_or_else(|| MlError::InvalidShape(format!("invalid Transpose perm axis {a}")))?;
                if seen[u] {
                    return Err(MlError::InvalidShape("duplicate axis in Transpose perm".into()));
                }
                seen[u] = true;
                p.push(u);
            }
            p
        }
        None => (0..rank).rev().collect(),
    };

    let out_shape: Vec<usize> = perm.iter().map(|&p| x.shape()[p]).collect();
    if x.numel() == 0 {
        return Tensor::zeros(out_shape, x.dtype());
    }
    let out_strides = compute_strides(&out_shape);
    let src_strides = x.strides();
    let mut permuted_src_strides = [0usize; MAX_RANK];
    for (i, &p) in perm.iter().enumerate() {
        permuted_src_strides[i] = src_strides[p];
    }

    match x.data() {
        TensorData::F32(src) => {
            let mut dst = vec![0.0f32; src.len()];
            for (i, slot) in dst.iter_mut().enumerate() {
                *slot = src[unravel_offset(i, &out_strides, rank, &permuted_src_strides)];
            }
            Tensor::from_f32(out_shape, dst)
        }
        TensorData::I64(src) => {
            let mut dst = vec![0i64; src.len()];
            for (i, slot) in dst.iter_mut().enumerate() {
                *slot = src[unravel_offset(i, &out_strides, rank, &permuted_src_strides)];
            }
            Tensor::from_i64(out_shape, dst)
        }
    }
}

fn run_gather(node: &Node, data: &Tensor, indices: &Tensor) -> Result<Tensor, MlError> {
    let rank = data.rank();
    if rank == 0 {
        return Err(MlError::InvalidShape("Gather cannot index a scalar".into()));
    }
    let axis = resolve_axis(node.attr_int("axis").unwrap_or(0), rank)?;
    let axis_dim = data.shape()[axis] as i64;

    let mut out_shape = Vec::with_capacity(rank - 1 + indices.rank());
    out_shape.extend_from_slice(&data.shape()[..axis]);
    out_shape.extend_from_slice(indices.shape());
    out_shape.extend_from_slice(&data.shape()[axis + 1..]);

    let outer: usize = data.shape()[..axis].iter().product();
    let inner: usize = data.shape()[axis + 1..].iter().product();
    let idx_vals = indices.to_i64_cow();
    let total = checked_numel(&out_shape, data.dtype())?;
    if total == 0 {
        return Tensor::zeros(out_shape, data.dtype());
    }

    let resolve_idx = |raw: i64| -> Result<usize, MlError> {
        let r = if raw < 0 { raw + axis_dim } else { raw };
        if r < 0 || r >= axis_dim {
            return Err(MlError::InvalidShape(format!("Gather index {raw} out of bounds for dim {axis_dim}")));
        }
        Ok(r as usize)
    };

    match data.data() {
        TensorData::F32(src) => {
            let mut out = Vec::with_capacity(total);
            for o in 0..outer {
                let base_o = o * (axis_dim as usize) * inner;
                for &raw in idx_vals.iter() {
                    let k = resolve_idx(raw)?;
                    let start = base_o + k * inner;
                    out.extend_from_slice(&src[start..start + inner]);
                }
            }
            Tensor::from_f32(out_shape, out)
        }
        TensorData::I64(src) => {
            let mut out = Vec::with_capacity(total);
            for o in 0..outer {
                let base_o = o * (axis_dim as usize) * inner;
                for &raw in idx_vals.iter() {
                    let k = resolve_idx(raw)?;
                    let start = base_o + k * inner;
                    out.extend_from_slice(&src[start..start + inner]);
                }
            }
            Tensor::from_i64(out_shape, out)
        }
    }
}

#[derive(Clone, Copy)]
struct SliceAxis {
    start: i64,
    step: i64,
    len: usize,
}

fn compute_slice_axis(dim: usize, start_raw: i64, end_raw: i64, step: i64) -> Result<SliceAxis, MlError> {
    if step == 0 {
        return Err(MlError::InvalidShape("Slice step cannot be 0".into()));
    }
    let d = dim as i64;
    if d == 0 {
        return Ok(SliceAxis { start: 0, step, len: 0 });
    }
    let (start, end) = if step > 0 {
        let s = if start_raw < 0 { start_raw.saturating_add(d) } else { start_raw }.clamp(0, d);
        let e = if end_raw < 0 { end_raw.saturating_add(d) } else { end_raw }.clamp(0, d);
        (s, e)
    } else {
        let s = if start_raw < 0 { start_raw.saturating_add(d) } else { start_raw }.clamp(-1, d - 1);
        let e = if end_raw < -d {
            -1
        } else if end_raw < 0 {
            end_raw.saturating_add(d)
        } else {
            end_raw
        }
        .clamp(-1, d - 1);
        (s, e)
    };
    let len = if (step > 0 && end > start) || (step < 0 && start > end) {
        let span = (end - start).unsigned_abs();
        let st = step.unsigned_abs();
        usize::try_from(span.div_ceil(st)).unwrap_or(0)
    } else {
        0
    };
    Ok(SliceAxis { start, step, len })
}

fn run_slice(node: &Node, inputs: &[Option<&Tensor>]) -> Result<Tensor, MlError> {
    let x = req(inputs, 0, "Slice")?;
    let rank = x.rank();
    let starts: Vec<i64> = inputs
        .get(1)
        .and_then(|t| *t)
        .map(|t| t.to_i64_cow().into_owned())
        .or_else(|| node.attr_ints("starts").map(|s| s.to_vec()))
        .ok_or_else(|| MlError::InvalidModel("Slice missing `starts`".into()))?;
    let ends: Vec<i64> = inputs
        .get(2)
        .and_then(|t| *t)
        .map(|t| t.to_i64_cow().into_owned())
        .or_else(|| node.attr_ints("ends").map(|s| s.to_vec()))
        .ok_or_else(|| MlError::InvalidModel("Slice missing `ends`".into()))?;
    if starts.len() != ends.len() {
        return Err(MlError::InvalidShape("Slice `starts` and `ends` length mismatch".into()));
    }
    let axes: Vec<i64> = inputs
        .get(3)
        .and_then(|t| *t)
        .map(|t| t.to_i64_cow().into_owned())
        .or_else(|| node.attr_ints("axes").map(|s| s.to_vec()))
        .unwrap_or_else(|| (0..starts.len() as i64).collect());
    let steps: Vec<i64> = inputs.get(4).and_then(|t| *t).map(|t| t.to_i64_cow().into_owned()).unwrap_or_else(|| vec![1; starts.len()]);
    if axes.len() != starts.len() || steps.len() != starts.len() {
        return Err(MlError::InvalidShape("Slice `axes`/`steps` length mismatch".into()));
    }

    let mut specs: Vec<SliceAxis> = x.shape().iter().map(|&d| SliceAxis { start: 0, step: 1, len: d }).collect();
    for i in 0..starts.len() {
        let ax = resolve_axis(axes[i], rank)?;
        specs[ax] = compute_slice_axis(x.shape()[ax], starts[i], ends[i], steps[i])?;
    }

    let out_shape: Vec<usize> = specs.iter().map(|s| s.len).collect();
    let total = checked_numel(&out_shape, x.dtype())?;
    if total == 0 {
        return Tensor::zeros(out_shape, x.dtype());
    }
    let out_strides = compute_strides(&out_shape);
    let src_strides = x.strides();
    let map_idx = |mut linear: usize| -> usize {
        let mut off = 0usize;
        for ax in 0..rank {
            let coord = (linear / out_strides[ax]) as i64;
            linear %= out_strides[ax];
            let src_c = (specs[ax].start + coord * specs[ax].step) as usize;
            off += src_c * src_strides[ax];
        }
        off
    };

    match x.data() {
        TensorData::F32(src) => Tensor::from_f32(out_shape, (0..total).map(|i| src[map_idx(i)]).collect()),
        TensorData::I64(src) => Tensor::from_i64(out_shape, (0..total).map(|i| src[map_idx(i)]).collect()),
    }
}

fn run_concat(node: &Node, inputs: &[Option<&Tensor>]) -> Result<Tensor, MlError> {
    let tensors: Vec<&Tensor> = inputs.iter().flatten().copied().collect();
    let first = *tensors.first().ok_or_else(|| MlError::InvalidModel("Concat needs at least 1 input".into()))?;
    let rank = first.rank();
    let axis_attr = node.attr_int("axis").ok_or_else(|| MlError::InvalidModel("Concat missing `axis`".into()))?;
    let axis = resolve_axis(axis_attr, rank)?;

    let mut cat_dim = 0usize;
    let mut any_f32 = false;
    for t in &tensors {
        if t.rank() != rank {
            return Err(MlError::InvalidShape("Concat input rank mismatch".into()));
        }
        for (i, (&a, &b)) in first.shape().iter().zip(t.shape().iter()).enumerate() {
            if i != axis && a != b {
                return Err(MlError::InvalidShape(format!("Concat shape mismatch at dim {i}: {:?} vs {:?}", first.shape(), t.shape())));
            }
        }
        cat_dim = cat_dim.checked_add(t.shape()[axis]).ok_or_else(|| MlError::InvalidShape("Concat dimension overflow".into()))?;
        if t.dtype() == DType::F32 {
            any_f32 = true;
        }
    }

    let mut out_shape = first.shape().to_vec();
    out_shape[axis] = cat_dim;
    let target_dtype = if any_f32 { DType::F32 } else { DType::I64 };
    let total = checked_numel(&out_shape, target_dtype)?;
    if total == 0 {
        return Tensor::zeros(out_shape, target_dtype);
    }

    let outer: usize = first.shape()[..axis].iter().product();
    let inner: usize = first.shape()[axis + 1..].iter().product();

    if target_dtype == DType::F32 {
        let cows: Vec<_> = tensors.iter().map(|t| t.to_f32_cow()).collect();
        let mut out = Vec::with_capacity(total);
        for o in 0..outer {
            for (t, cow) in tensors.iter().zip(cows.iter()) {
                let chunk = t.shape()[axis] * inner;
                let start = o * chunk;
                out.extend_from_slice(&cow[start..start + chunk]);
            }
        }
        Tensor::from_f32(out_shape, out)
    } else {
        let cows: Vec<_> = tensors.iter().map(|t| t.to_i64_cow()).collect();
        let mut out = Vec::with_capacity(total);
        for o in 0..outer {
            for (t, cow) in tensors.iter().zip(cows.iter()) {
                let chunk = t.shape()[axis] * inner;
                let start = o * chunk;
                out.extend_from_slice(&cow[start..start + chunk]);
            }
        }
        Tensor::from_i64(out_shape, out)
    }
}

fn run_split(node: &Node, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, MlError> {
    let x = req(inputs, 0, "Split")?;
    let axis = resolve_axis(node.attr_int("axis").unwrap_or(0), x.rank())?;
    let dim = x.shape()[axis];
    let num_outputs = node.attr_int("num_outputs").map(|n| n.max(1) as usize).unwrap_or_else(|| node.outputs.len().max(1));

    let splits: Vec<usize> = if let Some(sp) = inputs.get(1).and_then(|t| *t) {
        sp.to_i64_cow()
            .iter()
            .map(|&v| usize::try_from(v).map_err(|_| MlError::InvalidShape(format!("negative Split size {v}"))))
            .collect::<Result<Vec<_>, _>>()?
    } else if let Some(ints) = node.attr_ints("split") {
        ints.iter().map(|&v| usize::try_from(v).map_err(|_| MlError::InvalidShape(format!("negative Split size {v}")))).collect::<Result<Vec<_>, _>>()?
    } else {
        let base = dim.div_ceil(num_outputs);
        let mut v = Vec::with_capacity(num_outputs);
        let mut rem = dim;
        for _ in 0..num_outputs {
            let take = rem.min(base);
            v.push(take);
            rem -= take;
        }
        v
    };

    let sum: usize = splits.iter().sum();
    if sum != dim {
        return Err(MlError::InvalidShape(format!("Split sizes {splits:?} sum to {sum}, expected {dim}")));
    }

    let outer: usize = x.shape()[..axis].iter().product();
    let inner: usize = x.shape()[axis + 1..].iter().product();
    let mut out_tensors = Vec::with_capacity(splits.len());
    let mut axis_offset = 0usize;

    for &s in &splits {
        let mut sh = x.shape().to_vec();
        sh[axis] = s;
        let chunk = s * inner;
        let full_row = dim * inner;
        let t = match x.data() {
            TensorData::F32(src) => {
                let mut buf = Vec::with_capacity(outer * chunk);
                for o in 0..outer {
                    let start = o * full_row + axis_offset * inner;
                    buf.extend_from_slice(&src[start..start + chunk]);
                }
                Tensor::from_f32(sh, buf)?
            }
            TensorData::I64(src) => {
                let mut buf = Vec::with_capacity(outer * chunk);
                for o in 0..outer {
                    let start = o * full_row + axis_offset * inner;
                    buf.extend_from_slice(&src[start..start + chunk]);
                }
                Tensor::from_i64(sh, buf)?
            }
        };
        axis_offset += s;
        out_tensors.push(t);
    }
    Ok(out_tensors)
}

fn run_expand(x: &Tensor, shape_tensor: &Tensor) -> Result<Tensor, MlError> {
    let target: Vec<usize> = shape_tensor
        .to_i64_cow()
        .iter()
        .map(|&d| usize::try_from(d).map_err(|_| MlError::InvalidShape(format!("negative Expand dim {d}"))))
        .collect::<Result<Vec<_>, _>>()?;
    let out_shape = crate::tensor::broadcast_shape(x.shape(), &target)?;
    let total = checked_numel(&out_shape, x.dtype())?;
    if total == 0 {
        return Tensor::zeros(out_shape, x.dtype());
    }
    if out_shape == x.shape() {
        return Ok(x.clone());
    }
    let out_strides = compute_strides(&out_shape);
    let aligned = broadcast_strides(x.shape(), x.strides(), &out_shape)?;
    let rank = out_shape.len();
    match x.data() {
        TensorData::F32(src) => Tensor::from_f32(out_shape, (0..total).map(|i| src[unravel_offset(i, &out_strides, rank, &aligned)]).collect()),
        TensorData::I64(src) => Tensor::from_i64(out_shape, (0..total).map(|i| src[unravel_offset(i, &out_strides, rank, &aligned)]).collect()),
    }
}

fn run_tile(x: &Tensor, repeats: &Tensor) -> Result<Tensor, MlError> {
    let reps: Vec<usize> = repeats
        .to_i64_cow()
        .iter()
        .map(|&r| usize::try_from(r).map_err(|_| MlError::InvalidShape(format!("negative Tile repeat {r}"))))
        .collect::<Result<Vec<_>, _>>()?;
    if reps.len() != x.rank() {
        return Err(MlError::InvalidShape(format!("Tile repeats {:?} length mismatch with rank {}", reps, x.rank())));
    }
    let out_shape: Vec<usize> = x.shape().iter().zip(reps.iter()).map(|(&d, &r)| d.saturating_mul(r)).collect();
    let total = checked_numel(&out_shape, x.dtype())?;
    if total == 0 {
        return Tensor::zeros(out_shape, x.dtype());
    }
    let out_strides = compute_strides(&out_shape);
    let src_strides = x.strides();
    let rank = out_shape.len();
    let map_idx = |mut linear: usize| -> usize {
        let mut off = 0usize;
        for ax in 0..rank {
            let coord = linear / out_strides[ax];
            linear %= out_strides[ax];
            off += (coord % x.shape()[ax]) * src_strides[ax];
        }
        off
    };
    match x.data() {
        TensorData::F32(src) => Tensor::from_f32(out_shape, (0..total).map(|i| src[map_idx(i)]).collect()),
        TensorData::I64(src) => Tensor::from_i64(out_shape, (0..total).map(|i| src[map_idx(i)]).collect()),
    }
}

fn run_pad(node: &Node, inputs: &[Option<&Tensor>]) -> Result<Tensor, MlError> {
    let x = req(inputs, 0, "Pad")?;
    let rank = x.rank();
    let mode = node.attr_str("mode").unwrap_or("constant");
    let raw_pads: Vec<i64> = inputs
        .get(1)
        .and_then(|t| *t)
        .map(|t| t.to_i64_cow().into_owned())
        .or_else(|| node.attr_ints("pads").map(|p| p.to_vec()))
        .ok_or_else(|| MlError::InvalidModel("Pad missing `pads`".into()))?;

    let mut pad_before = vec![0i64; rank];
    let mut pad_after = vec![0i64; rank];
    if let Some(axes_t) = inputs.get(3).and_then(|t| *t) {
        let axes = axes_t.to_i64_cow();
        if raw_pads.len() != axes.len() * 2 {
            return Err(MlError::InvalidShape("Pad `pads` and `axes` mismatch".into()));
        }
        for (i, &a) in axes.iter().enumerate() {
            let ax = resolve_axis(a, rank)?;
            pad_before[ax] = raw_pads[i];
            pad_after[ax] = raw_pads[axes.len() + i];
        }
    } else {
        if raw_pads.len() != rank * 2 {
            return Err(MlError::InvalidShape(format!("Pad `pads` len {} != 2 * rank {rank}", raw_pads.len())));
        }
        pad_before.copy_from_slice(&raw_pads[..rank]);
        pad_after.copy_from_slice(&raw_pads[rank..]);
    }

    let mut out_shape = Vec::with_capacity(rank);
    for ax in 0..rank {
        let d = (x.shape()[ax] as i64) + pad_before[ax] + pad_after[ax];
        let u = usize::try_from(d).map_err(|_| MlError::InvalidShape("Pad produces negative dimension".into()))?;
        out_shape.push(u);
    }
    let total = checked_numel(&out_shape, x.dtype())?;
    if total == 0 {
        return Tensor::zeros(out_shape, x.dtype());
    }
    let out_strides = compute_strides(&out_shape);
    let src_strides = x.strides();

    let resolve_coord = |mut linear: usize| -> Option<usize> {
        let mut src_off = 0usize;
        for ax in 0..rank {
            let c = (linear / out_strides[ax]) as i64 - pad_before[ax];
            linear %= out_strides[ax];
            let dim = x.shape()[ax] as i64;
            if dim <= 0 {
                return None;
            }
            let sc = match mode {
                "edge" => c.clamp(0, dim - 1) as usize,
                "reflect" => {
                    if dim == 1 {
                        0
                    } else {
                        let period = 2 * (dim - 1);
                        let m = c.rem_euclid(period);
                        (if m >= dim { period - m } else { m }) as usize
                    }
                }
                _ => {
                    if c < 0 || c >= dim {
                        return None;
                    }
                    c as usize
                }
            };
            src_off += sc * src_strides[ax];
        }
        Some(src_off)
    };

    match x.data() {
        TensorData::F32(src) => {
            let cval = inputs.get(2).and_then(|t| *t).and_then(|t| t.to_f32_cow().first().copied()).or_else(|| node.attr_float("value")).unwrap_or(0.0);
            Tensor::from_f32(out_shape, (0..total).map(|i| resolve_coord(i).map_or(cval, |o| src[o])).collect())
        }
        TensorData::I64(src) => {
            let cval = inputs.get(2).and_then(|t| *t).and_then(|t| t.to_i64_cow().first().copied()).unwrap_or(0);
            Tensor::from_i64(out_shape, (0..total).map(|i| resolve_coord(i).map_or(cval, |o| src[o])).collect())
        }
    }
}

fn run_constant_of_shape(node: &Node, shape_tensor: &Tensor) -> Result<Tensor, MlError> {
    let shape: Vec<usize> = shape_tensor
        .to_i64_cow()
        .iter()
        .map(|&d| usize::try_from(d).map_err(|_| MlError::InvalidShape(format!("negative ConstantOfShape dim {d}"))))
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(Attr::Tensor(val)) = node.attrs.get("value") {
        return match val.data() {
            TensorData::F32(v) => Tensor::full_f32(shape, v.first().copied().unwrap_or(0.0)),
            TensorData::I64(v) => Tensor::full_i64(shape, v.first().copied().unwrap_or(0)),
        };
    }
    Tensor::full_f32(shape, 0.0)
}

fn run_range(start_t: &Tensor, limit_t: &Tensor, delta_t: &Tensor) -> Result<Tensor, MlError> {
    if start_t.dtype() == DType::I64 && limit_t.dtype() == DType::I64 && delta_t.dtype() == DType::I64 {
        let start = start_t.to_i64_cow().first().copied().unwrap_or(0);
        let limit = limit_t.to_i64_cow().first().copied().unwrap_or(0);
        let delta = delta_t.to_i64_cow().first().copied().unwrap_or(1);
        if delta == 0 {
            return Err(MlError::InvalidShape("Range delta cannot be 0".into()));
        }
        let n = if (delta > 0 && limit > start) || (delta < 0 && start > limit) {
            usize::try_from((limit - start).unsigned_abs().div_ceil(delta.unsigned_abs())).unwrap_or(0)
        } else {
            0
        };
        let _ = checked_numel(&[n], DType::I64)?;
        let vals: Vec<i64> = (0..n).map(|i| start.saturating_add((i as i64).saturating_mul(delta))).collect();
        return Tensor::from_i64(vec![n], vals);
    }
    let start = start_t.to_f32_cow().first().copied().unwrap_or(0.0);
    let limit = limit_t.to_f32_cow().first().copied().unwrap_or(0.0);
    let delta = delta_t.to_f32_cow().first().copied().unwrap_or(1.0);
    if delta == 0.0 || !delta.is_finite() {
        return Err(MlError::InvalidShape("Range delta must be finite and non-zero".into()));
    }
    let raw_n = ((limit - start) / delta).ceil().max(0.0);
    let n = if raw_n.is_finite() { raw_n as usize } else { 0 };
    let _ = checked_numel(&[n], DType::F32)?;
    let vals: Vec<f32> = (0..n).map(|i| start + (i as f32) * delta).collect();
    Tensor::from_f32(vec![n], vals)
}

fn run_one_hot(node: &Node, indices: &Tensor, depth_t: &Tensor, values_t: &Tensor) -> Result<Tensor, MlError> {
    let depth_i64 = depth_t.to_i64_cow().first().copied().unwrap_or(0);
    let depth = usize::try_from(depth_i64).map_err(|_| MlError::InvalidShape(format!("negative OneHot depth {depth_i64}")))?;
    let out_rank = indices.rank() + 1;
    let axis = resolve_axis(node.attr_int("axis").unwrap_or(-1), out_rank)?;

    let mut out_shape = Vec::with_capacity(out_rank);
    out_shape.extend_from_slice(&indices.shape()[..axis]);
    out_shape.push(depth);
    out_shape.extend_from_slice(&indices.shape()[axis..]);

    let outer: usize = indices.shape()[..axis].iter().product();
    let inner: usize = indices.shape()[axis..].iter().product();
    let idx_vals = indices.to_i64_cow();

    if values_t.dtype() == DType::I64 {
        let vals = values_t.to_i64_cow();
        let (off_v, on_v) = (vals.first().copied().unwrap_or(0), vals.get(1).copied().unwrap_or(1));
        let total = checked_numel(&out_shape, DType::I64)?;
        let mut out = vec![off_v; total];
        for o in 0..outer {
            for r in 0..inner {
                let raw = idx_vals[o * inner + r];
                let resolved = if raw < 0 { raw + depth_i64 } else { raw };
                if resolved >= 0 && resolved < depth_i64 {
                    out[(o * depth + resolved as usize) * inner + r] = on_v;
                }
            }
        }
        return Tensor::from_i64(out_shape, out);
    }

    let vals = values_t.to_f32_cow();
    let (off_v, on_v) = (vals.first().copied().unwrap_or(0.0), vals.get(1).copied().unwrap_or(1.0));
    let total = checked_numel(&out_shape, DType::F32)?;
    let mut out = vec![off_v; total];
    for o in 0..outer {
        for r in 0..inner {
            let raw = idx_vals[o * inner + r];
            let resolved = if raw < 0 { raw + depth_i64 } else { raw };
            if resolved >= 0 && resolved < depth_i64 {
                out[(o * depth + resolved as usize) * inner + r] = on_v;
            }
        }
    }
    Tensor::from_f32(out_shape, out)
}

fn run_scatter_nd(node: &Node, data: &Tensor, indices: &Tensor, updates: &Tensor) -> Result<Tensor, MlError> {
    let reduction = node.attr_str("reduction").unwrap_or("none");
    if indices.rank() == 0 {
        return Err(MlError::InvalidShape("ScatterND indices must have rank >= 1".into()));
    }
    let k = *indices.shape().last().unwrap_or(&0);
    if k > data.rank() {
        return Err(MlError::InvalidShape(format!("ScatterND index tuple {k} exceeds data rank {}", data.rank())));
    }
    let num_slices: usize = indices.shape()[..indices.rank() - 1].iter().product();
    let slice_len: usize = data.shape()[k..].iter().product();
    if updates.numel() != num_slices.saturating_mul(slice_len) {
        return Err(MlError::InvalidShape("ScatterND updates size mismatch".into()));
    }
    if data.numel() == 0 || num_slices == 0 || slice_len == 0 {
        return Ok(data.clone());
    }

    let data_strides = data.strides();
    let idx_vals = indices.to_i64_cow();
    let resolve_base = |s: usize| -> Result<usize, MlError> {
        let mut base = 0usize;
        for m in 0..k {
            let dim = data.shape()[m] as i64;
            let raw = idx_vals[s * k + m];
            let r = if raw < 0 { raw + dim } else { raw };
            if r < 0 || r >= dim {
                return Err(MlError::InvalidShape(format!("ScatterND index {raw} out of bounds for dim {dim}")));
            }
            base += (r as usize) * data_strides[m];
        }
        Ok(base)
    };

    if data.dtype() == DType::I64 {
        let mut out = data.to_i64_cow().into_owned();
        let upd = updates.to_i64_cow();
        for s in 0..num_slices {
            let base = resolve_base(s)?;
            let u_base = s * slice_len;
            for j in 0..slice_len {
                let v = upd[u_base + j];
                out[base + j] = match reduction {
                    "add" => out[base + j].wrapping_add(v),
                    "mul" => out[base + j].wrapping_mul(v),
                    "max" => out[base + j].max(v),
                    "min" => out[base + j].min(v),
                    _ => v,
                };
            }
        }
        return Tensor::from_i64(data.shape().to_vec(), out);
    }

    let mut out = data.to_f32_cow().into_owned();
    let upd = updates.to_f32_cow();
    for s in 0..num_slices {
        let base = resolve_base(s)?;
        let u_base = s * slice_len;
        for j in 0..slice_len {
            let v = upd[u_base + j];
            out[base + j] = match reduction {
                "add" => out[base + j] + v,
                "mul" => out[base + j] * v,
                "max" => out[base + j].max(v),
                "min" => out[base + j].min(v),
                _ => v,
            };
        }
    }
    Tensor::from_f32(data.shape().to_vec(), out)
}

fn run_topk(node: &Node, x: &Tensor, k_tensor: &Tensor) -> Result<Vec<Tensor>, MlError> {
    let axis = resolve_axis(node.attr_int("axis").unwrap_or(-1), x.rank())?;
    let largest = node.attr_int("largest").unwrap_or(1) != 0;
    let k_raw = k_tensor.to_i64_cow().first().copied().unwrap_or(0);
    let dim = x.shape()[axis];
    let k = usize::try_from(k_raw).map_err(|_| MlError::InvalidShape("negative TopK k".into()))?.min(dim);

    let mut out_shape = x.shape().to_vec();
    out_shape[axis] = k;
    let total = checked_numel(&out_shape, DType::F32)?;
    if total == 0 {
        return Ok(vec![Tensor::zeros(out_shape.clone(), DType::F32)?, Tensor::zeros(out_shape, DType::I64)?]);
    }

    let outer: usize = x.shape()[..axis].iter().product();
    let inner: usize = x.shape()[axis + 1..].iter().product();
    let src = x.to_f32_cow();
    let mut values = vec![0.0f32; total];
    let mut indices = vec![0i64; total];
    let mut pairs: Vec<(f32, usize)> = Vec::with_capacity(dim);

    for o in 0..outer {
        for r in 0..inner {
            pairs.clear();
            for d in 0..dim {
                pairs.push((src[(o * dim + d) * inner + r], d));
            }
            pairs.sort_by(|a, b| {
                let ord = a.0.total_cmp(&b.0);
                if largest { ord.reverse().then_with(|| a.1.cmp(&b.1)) } else { ord.then_with(|| a.1.cmp(&b.1)) }
            });
            for (j, &(val, idx)) in pairs.iter().take(k).enumerate() {
                let pos = (o * k + j) * inner + r;
                values[pos] = val;
                indices[pos] = idx as i64;
            }
        }
    }
    Ok(vec![Tensor::from_f32(out_shape.clone(), values)?, Tensor::from_i64(out_shape, indices)?])
}

fn run_depth_to_space(node: &Node, x: &Tensor) -> Result<Tensor, MlError> {
    if x.rank() != 4 {
        return Err(MlError::InvalidShape("DepthToSpace expects [N, C, H, W]".into()));
    }
    let bs =
        usize::try_from(node.attr_int("blocksize").unwrap_or(1)).ok().filter(|&b| b >= 1).ok_or_else(|| MlError::InvalidShape("invalid blocksize".into()))?;
    let mode = node.attr_str("mode").unwrap_or("DCR");
    let (n, c, h, w) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
    let bs2 = bs * bs;
    if c % bs2 != 0 {
        return Err(MlError::InvalidShape("DepthToSpace C must be divisible by blocksize^2".into()));
    }
    let oc = c / bs2;
    let (oh, ow) = (h * bs, w * bs);
    let out_shape = vec![n, oc, oh, ow];
    let src = x.to_f32_cow();
    let mut dst = vec![0.0f32; checked_numel(&out_shape, DType::F32)?];

    for b in 0..n {
        for ch in 0..oc {
            for y in 0..oh {
                let (sy, by) = (y / bs, y % bs);
                for col in 0..ow {
                    let (sx, bx) = (col / bs, col % bs);
                    let sc = if mode == "CRD" { (ch * bs + by) * bs + bx } else { (by * bs + bx) * oc + ch };
                    dst[((b * oc + ch) * oh + y) * ow + col] = src[((b * c + sc) * h + sy) * w + sx];
                }
            }
        }
    }
    Tensor::from_f32(out_shape, dst)
}

/// Try executing a shape, indexing or structural ONNX node.
pub fn try_run(node: &Node, inputs: &[Option<&Tensor>]) -> Result<Option<Vec<Tensor>>, MlError> {
    let op = node.op_type.as_str();
    let single = |t: Tensor| Ok(Some(vec![t]));
    match op {
        "Constant" => single(run_constant(node)?),
        "Identity" | "Dropout" => single(req(inputs, 0, op)?.clone()),
        "Shape" => single(run_shape(node, req(inputs, 0, op)?)?),
        "Size" => single(Tensor::from_i64(Vec::new(), vec![req(inputs, 0, op)?.numel() as i64])?),
        "Reshape" => single(run_reshape(node, req(inputs, 0, op)?, req(inputs, 1, op)?)?),
        "Flatten" => single(run_flatten(node, req(inputs, 0, op)?)?),
        "Squeeze" => single(run_squeeze(node, inputs)?),
        "Unsqueeze" => single(run_unsqueeze(node, inputs)?),
        "Transpose" => single(run_transpose(node, req(inputs, 0, op)?)?),
        "Gather" => single(run_gather(node, req(inputs, 0, op)?, req(inputs, 1, op)?)?),
        "Slice" => single(run_slice(node, inputs)?),
        "Concat" => single(run_concat(node, inputs)?),
        "Split" => Ok(Some(run_split(node, inputs)?)),
        "Expand" => single(run_expand(req(inputs, 0, op)?, req(inputs, 1, op)?)?),
        "Tile" => single(run_tile(req(inputs, 0, op)?, req(inputs, 1, op)?)?),
        "Pad" => single(run_pad(node, inputs)?),
        "ConstantOfShape" => single(run_constant_of_shape(node, req(inputs, 0, op)?)?),
        "Range" => single(run_range(req(inputs, 0, op)?, req(inputs, 1, op)?, req(inputs, 2, op)?)?),
        "OneHot" => single(run_one_hot(node, req(inputs, 0, op)?, req(inputs, 1, op)?, req(inputs, 2, op)?)?),
        "ScatterND" => single(run_scatter_nd(node, req(inputs, 0, op)?, req(inputs, 1, op)?, req(inputs, 2, op)?)?),
        "TopK" => Ok(Some(run_topk(node, req(inputs, 0, op)?, req(inputs, 1, op)?)?)),
        "DepthToSpace" => single(run_depth_to_space(node, req(inputs, 0, op)?)?),
        _ => Ok(None),
    }
}
