//! Elementwise math, activations, comparisons, logic, `Cast`, `Clip` and `Where`.

use crate::MlError;
use crate::onnx::{ElemType, Node};
use crate::tensor::{DType, Tensor, broadcast_shape, broadcast_strides, compute_strides, unravel_offset};

/// Approximation of the Gauss error function `erf(x)` accurate within `1.5e-7` (Abramowitz & Stegun 7.1.26).
#[inline]
pub fn erf_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    if ax >= 4.0 {
        return sign;
    }
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = (((((1.0614054 * t - 1.4531521) * t) + 1.4214138) * t - 0.28449672) * t + 0.2548296) * t;
    sign * (1.0 - poly * (-ax * ax).exp())
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn unary_f32(inp: &Tensor, f: impl Fn(f32) -> f32) -> Result<Tensor, MlError> {
    let src = inp.to_f32_cow();
    let out: Vec<f32> = src.iter().map(|&x| f(x)).collect();
    Tensor::from_f32(inp.shape().to_vec(), out)
}

fn unary_i64(inp: &Tensor, f: impl Fn(i64) -> i64) -> Result<Tensor, MlError> {
    let src = inp.to_i64_cow();
    let out: Vec<i64> = src.iter().map(|&x| f(x)).collect();
    Tensor::from_i64(inp.shape().to_vec(), out)
}

fn binary_f32(a: &Tensor, b: &Tensor, op: impl Fn(f32, f32) -> f32) -> Result<Tensor, MlError> {
    let out_shape = broadcast_shape(a.shape(), b.shape())?;
    let sa = a.to_f32_cow();
    let sb = b.to_f32_cow();
    if a.shape() == b.shape() {
        let out: Vec<f32> = sa.iter().zip(sb.iter()).map(|(&x, &y)| op(x, y)).collect();
        return Tensor::from_f32(out_shape, out);
    }
    if sb.len() == 1 && out_shape == a.shape() {
        let y = sb[0];
        return Tensor::from_f32(out_shape, sa.iter().map(|&x| op(x, y)).collect());
    }
    if sa.len() == 1 && out_shape == b.shape() {
        let x = sa[0];
        return Tensor::from_f32(out_shape, sb.iter().map(|&y| op(x, y)).collect());
    }
    let mut res = Tensor::zeros(out_shape.clone(), DType::F32)?;
    let n = res.numel();
    if n == 0 {
        return Ok(res);
    }
    let out_strides = compute_strides(&out_shape);
    let aa = broadcast_strides(a.shape(), a.strides(), &out_shape)?;
    let ab = broadcast_strides(b.shape(), b.strides(), &out_shape)?;
    let rank = out_shape.len();
    let dst = res.as_f32_mut()?;
    for (i, slot) in dst.iter_mut().enumerate() {
        let ia = unravel_offset(i, &out_strides, rank, &aa);
        let ib = unravel_offset(i, &out_strides, rank, &ab);
        *slot = op(sa[ia], sb[ib]);
    }
    Ok(res)
}

fn binary_i64(a: &Tensor, b: &Tensor, op: impl Fn(i64, i64) -> i64) -> Result<Tensor, MlError> {
    let out_shape = broadcast_shape(a.shape(), b.shape())?;
    let sa = a.to_i64_cow();
    let sb = b.to_i64_cow();
    if a.shape() == b.shape() {
        let out: Vec<i64> = sa.iter().zip(sb.iter()).map(|(&x, &y)| op(x, y)).collect();
        return Tensor::from_i64(out_shape, out);
    }
    let mut out = Vec::new();
    let n = crate::tensor::checked_numel(&out_shape, DType::I64)?;
    if n == 0 {
        return Tensor::from_i64(out_shape, out);
    }
    out.resize(n, 0);
    let out_strides = compute_strides(&out_shape);
    let aa = broadcast_strides(a.shape(), a.strides(), &out_shape)?;
    let ab = broadcast_strides(b.shape(), b.strides(), &out_shape)?;
    let rank = out_shape.len();
    for (i, slot) in out.iter_mut().enumerate() {
        let ia = unravel_offset(i, &out_strides, rank, &aa);
        let ib = unravel_offset(i, &out_strides, rank, &ab);
        *slot = op(sa[ia], sb[ib]);
    }
    Tensor::from_i64(out_shape, out)
}

fn compare_op(a: &Tensor, b: &Tensor, cmp_f: impl Fn(f32, f32) -> bool, cmp_i: impl Fn(i64, i64) -> bool) -> Result<Tensor, MlError> {
    if a.dtype() == DType::I64 && b.dtype() == DType::I64 {
        return binary_i64(a, b, |x, y| i64::from(cmp_i(x, y)));
    }
    let out_shape = broadcast_shape(a.shape(), b.shape())?;
    let sa = a.to_f32_cow();
    let sb = b.to_f32_cow();
    let n = crate::tensor::checked_numel(&out_shape, DType::I64)?;
    if n == 0 {
        return Tensor::from_i64(out_shape, Vec::new());
    }
    let mut out = vec![0i64; n];
    let out_strides = compute_strides(&out_shape);
    let aa = broadcast_strides(a.shape(), a.strides(), &out_shape)?;
    let ab = broadcast_strides(b.shape(), b.strides(), &out_shape)?;
    let rank = out_shape.len();
    for (i, slot) in out.iter_mut().enumerate() {
        let ia = unravel_offset(i, &out_strides, rank, &aa);
        let ib = unravel_offset(i, &out_strides, rank, &ab);
        *slot = i64::from(cmp_f(sa[ia], sb[ib]));
    }
    Tensor::from_i64(out_shape, out)
}

fn run_where(cond: &Tensor, x: &Tensor, y: &Tensor) -> Result<Tensor, MlError> {
    let xy_shape = broadcast_shape(x.shape(), y.shape())?;
    let out_shape = broadcast_shape(cond.shape(), &xy_shape)?;
    let sc = cond.to_i64_cow();
    let out_strides = compute_strides(&out_shape);
    let ac = broadcast_strides(cond.shape(), cond.strides(), &out_shape)?;
    let ax = broadcast_strides(x.shape(), x.strides(), &out_shape)?;
    let ay = broadcast_strides(y.shape(), y.strides(), &out_shape)?;
    let rank = out_shape.len();

    if x.dtype() == DType::I64 && y.dtype() == DType::I64 {
        let sx = x.to_i64_cow();
        let sy = y.to_i64_cow();
        let n = crate::tensor::checked_numel(&out_shape, DType::I64)?;
        let mut out = vec![0i64; n];
        for (i, slot) in out.iter_mut().enumerate() {
            let c = sc[unravel_offset(i, &out_strides, rank, &ac)];
            *slot = if c != 0 { sx[unravel_offset(i, &out_strides, rank, &ax)] } else { sy[unravel_offset(i, &out_strides, rank, &ay)] };
        }
        return Tensor::from_i64(out_shape, out);
    }
    let sx = x.to_f32_cow();
    let sy = y.to_f32_cow();
    let n = crate::tensor::checked_numel(&out_shape, DType::F32)?;
    let mut out = vec![0.0f32; n];
    for (i, slot) in out.iter_mut().enumerate() {
        let c = sc[unravel_offset(i, &out_strides, rank, &ac)];
        *slot = if c != 0 { sx[unravel_offset(i, &out_strides, rank, &ax)] } else { sy[unravel_offset(i, &out_strides, rank, &ay)] };
    }
    Tensor::from_f32(out_shape, out)
}

fn run_clip(node: &Node, inputs: &[Option<&Tensor>]) -> Result<Tensor, MlError> {
    let x = inputs.first().and_then(|t| *t).ok_or_else(|| MlError::InvalidModel("Clip missing input".into()))?;
    if x.dtype() == DType::I64 {
        let lo = inputs.get(1).and_then(|t| *t).and_then(|t| t.to_i64_cow().first().copied()).unwrap_or(i64::MIN);
        let hi = inputs.get(2).and_then(|t| *t).and_then(|t| t.to_i64_cow().first().copied()).unwrap_or(i64::MAX);
        return unary_i64(x, |v| v.clamp(lo, hi.max(lo)));
    }
    let lo = inputs.get(1).and_then(|t| *t).and_then(|t| t.to_f32_cow().first().copied()).or_else(|| node.attr_float("min")).unwrap_or(f32::NEG_INFINITY);
    let hi = inputs.get(2).and_then(|t| *t).and_then(|t| t.to_f32_cow().first().copied()).or_else(|| node.attr_float("max")).unwrap_or(f32::INFINITY);
    unary_f32(x, |v| v.clamp(lo, hi.max(lo)))
}

fn variadic_f32(inputs: &[Option<&Tensor>], op: impl Fn(f32, f32) -> f32) -> Result<Tensor, MlError> {
    let mut iter = inputs.iter().flatten().copied();
    let first = iter.next().ok_or_else(|| MlError::InvalidModel("variadic op needs at least 1 input".into()))?;
    let mut acc = first.cast(DType::F32)?;
    for next in iter {
        acc = binary_f32(&acc, next, &op)?;
    }
    Ok(acc)
}

fn req<'a>(inputs: &'a [Option<&'a Tensor>], idx: usize, op: &str) -> Result<&'a Tensor, MlError> {
    inputs.get(idx).and_then(|t| *t).ok_or_else(|| MlError::InvalidModel(format!("{op} missing input {idx}")))
}

/// Execute an elementwise math, activation, comparison, logical or `Cast`/`Clip`/`Where` node.
/// Returns `Ok(None)` when `node.op_type` is not handled here.
pub fn try_run(node: &Node, inputs: &[Option<&Tensor>]) -> Result<Option<Tensor>, MlError> {
    let op = node.op_type.as_str();
    let res = match op {
        "Add" => {
            let (a, b) = (req(inputs, 0, op)?, req(inputs, 1, op)?);
            if a.dtype() == DType::I64 && b.dtype() == DType::I64 { binary_i64(a, b, i64::wrapping_add)? } else { binary_f32(a, b, |x, y| x + y)? }
        }
        "Sub" => {
            let (a, b) = (req(inputs, 0, op)?, req(inputs, 1, op)?);
            if a.dtype() == DType::I64 && b.dtype() == DType::I64 { binary_i64(a, b, i64::wrapping_sub)? } else { binary_f32(a, b, |x, y| x - y)? }
        }
        "Mul" => {
            let (a, b) = (req(inputs, 0, op)?, req(inputs, 1, op)?);
            if a.dtype() == DType::I64 && b.dtype() == DType::I64 { binary_i64(a, b, i64::wrapping_mul)? } else { binary_f32(a, b, |x, y| x * y)? }
        }
        "Div" => {
            let (a, b) = (req(inputs, 0, op)?, req(inputs, 1, op)?);
            if a.dtype() == DType::I64 && b.dtype() == DType::I64 {
                binary_i64(a, b, |x, y| if y == 0 { 0 } else { x.wrapping_div(y) })?
            } else {
                binary_f32(a, b, |x, y| x / y)?
            }
        }
        "Mod" => {
            let (a, b) = (req(inputs, 0, op)?, req(inputs, 1, op)?);
            if a.dtype() == DType::I64 && b.dtype() == DType::I64 {
                binary_i64(a, b, |x, y| if y == 0 { 0 } else { x.wrapping_rem(y) })?
            } else {
                binary_f32(a, b, |x, y| x % y)?
            }
        }
        "Pow" => binary_f32(req(inputs, 0, op)?, req(inputs, 1, op)?, f32::powf)?,
        "Min" => variadic_f32(inputs, f32::min)?,
        "Max" => variadic_f32(inputs, f32::max)?,
        "Sum" => variadic_f32(inputs, |x, y| x + y)?,
        "Mean" => {
            let count = inputs.iter().flatten().count().max(1) as f32;
            let sum = variadic_f32(inputs, |x, y| x + y)?;
            unary_f32(&sum, |x| x / count)?
        }
        "PRelu" => binary_f32(req(inputs, 0, op)?, req(inputs, 1, op)?, |x, s| if x >= 0.0 { x } else { s * x })?,
        "Cast" => {
            let x = req(inputs, 0, op)?;
            let to = node.attr_int("to").ok_or_else(|| MlError::InvalidModel("Cast missing `to` attribute".into()))?;
            let elem = ElemType::from_onnx(to as i32)?;
            x.cast(elem.runtime_dtype())?
        }
        "Clip" => run_clip(node, inputs)?,
        "Where" => run_where(req(inputs, 0, op)?, req(inputs, 1, op)?, req(inputs, 2, op)?)?,
        "Equal" => compare_op(req(inputs, 0, op)?, req(inputs, 1, op)?, |x, y| x == y, |x, y| x == y)?,
        "Greater" => compare_op(req(inputs, 0, op)?, req(inputs, 1, op)?, |x, y| x > y, |x, y| x > y)?,
        "GreaterOrEqual" => compare_op(req(inputs, 0, op)?, req(inputs, 1, op)?, |x, y| x >= y, |x, y| x >= y)?,
        "Less" => compare_op(req(inputs, 0, op)?, req(inputs, 1, op)?, |x, y| x < y, |x, y| x < y)?,
        "LessOrEqual" => compare_op(req(inputs, 0, op)?, req(inputs, 1, op)?, |x, y| x <= y, |x, y| x <= y)?,
        "Not" => unary_i64(req(inputs, 0, op)?, |x| i64::from(x == 0))?,
        "And" => binary_i64(req(inputs, 0, op)?, req(inputs, 1, op)?, |x, y| i64::from(x != 0 && y != 0))?,
        "Or" => binary_i64(req(inputs, 0, op)?, req(inputs, 1, op)?, |x, y| i64::from(x != 0 || y != 0))?,
        "Xor" => binary_i64(req(inputs, 0, op)?, req(inputs, 1, op)?, |x, y| i64::from((x != 0) ^ (y != 0)))?,
        _ => return try_run_unary(node, inputs),
    };
    Ok(Some(res))
}

fn try_run_unary(node: &Node, inputs: &[Option<&Tensor>]) -> Result<Option<Tensor>, MlError> {
    let op = node.op_type.as_str();
    let x_fn = || req(inputs, 0, op);
    let res = match op {
        "Neg" => {
            let x = x_fn()?;
            if x.dtype() == DType::I64 { unary_i64(x, i64::wrapping_neg)? } else { unary_f32(x, |v| -v)? }
        }
        "Abs" => {
            let x = x_fn()?;
            if x.dtype() == DType::I64 { unary_i64(x, i64::wrapping_abs)? } else { unary_f32(x, f32::abs)? }
        }
        "Sign" => unary_f32(x_fn()?, |v| {
            if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                0.0
            }
        })?,
        "Sqrt" => unary_f32(x_fn()?, f32::sqrt)?,
        "Exp" => unary_f32(x_fn()?, f32::exp)?,
        "Log" => unary_f32(x_fn()?, f32::ln)?,
        "Floor" => unary_f32(x_fn()?, f32::floor)?,
        "Ceil" => unary_f32(x_fn()?, f32::ceil)?,
        "Round" => unary_f32(x_fn()?, f32::round_ties_even)?,
        "Reciprocal" => unary_f32(x_fn()?, |v| 1.0 / v)?,
        "Erf" => unary_f32(x_fn()?, erf_f32)?,
        "Sin" => unary_f32(x_fn()?, f32::sin)?,
        "Cos" => unary_f32(x_fn()?, f32::cos)?,
        "Tan" => unary_f32(x_fn()?, f32::tan)?,
        "Asin" => unary_f32(x_fn()?, f32::asin)?,
        "Acos" => unary_f32(x_fn()?, f32::acos)?,
        "Atan" => unary_f32(x_fn()?, f32::atan)?,
        "Sinh" => unary_f32(x_fn()?, f32::sinh)?,
        "Cosh" => unary_f32(x_fn()?, f32::cosh)?,
        "Tanh" => unary_f32(x_fn()?, f32::tanh)?,
        "Relu" => unary_f32(x_fn()?, |v| v.max(0.0))?,
        "LeakyRelu" => {
            let alpha = node.attr_float("alpha").unwrap_or(0.01);
            unary_f32(x_fn()?, |v| if v >= 0.0 { v } else { alpha * v })?
        }
        "ThresholdedRelu" => {
            let alpha = node.attr_float("alpha").unwrap_or(1.0);
            unary_f32(x_fn()?, |v| if v > alpha { v } else { 0.0 })?
        }
        "Sigmoid" => unary_f32(x_fn()?, sigmoid)?,
        "HardSigmoid" => {
            let alpha = node.attr_float("alpha").unwrap_or(0.2);
            let beta = node.attr_float("beta").unwrap_or(0.5);
            unary_f32(x_fn()?, |v| (alpha * v + beta).clamp(0.0, 1.0))?
        }
        "HardSwish" => unary_f32(x_fn()?, |v| v * (v / 6.0 + 0.5).clamp(0.0, 1.0))?,
        "Swish" | "Silu" => unary_f32(x_fn()?, |v| v * sigmoid(v))?,
        "Gelu" => {
            let approx_tanh = node.attr_str("approximate") == Some("tanh");
            if approx_tanh {
                const K: f32 = 0.797_884_6; // sqrt(2 / pi)
                unary_f32(x_fn()?, |v| 0.5 * v * (1.0 + (K * (v + 0.044_715 * v * v * v)).tanh()))?
            } else {
                const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
                unary_f32(x_fn()?, |v| 0.5 * v * (1.0 + erf_f32(v * INV_SQRT2)))?
            }
        }
        "Elu" => {
            let alpha = node.attr_float("alpha").unwrap_or(1.0);
            unary_f32(x_fn()?, |v| if v >= 0.0 { v } else { alpha * (v.exp() - 1.0) })?
        }
        "Selu" => {
            let alpha = node.attr_float("alpha").unwrap_or(1.673_263_2);
            let gamma = node.attr_float("gamma").unwrap_or(1.050_701);
            unary_f32(x_fn()?, |v| gamma * if v > 0.0 { v } else { alpha * v.exp() - alpha })?
        }
        "Softplus" => unary_f32(x_fn()?, |v| if v > 20.0 { v } else { (1.0 + v.exp()).ln() })?,
        "Softsign" => unary_f32(x_fn()?, |v| v / (1.0 + v.abs()))?,
        "Mish" => unary_f32(x_fn()?, |v| {
            let sp = if v > 20.0 { v } else { (1.0 + v.exp()).ln() };
            v * sp.tanh()
        })?,
        "IsNaN" => {
            let src = x_fn()?.to_f32_cow();
            Tensor::from_i64(x_fn()?.shape().to_vec(), src.iter().map(|v| i64::from(v.is_nan())).collect())?
        }
        _ => return Ok(None),
    };
    Ok(Some(res))
}
