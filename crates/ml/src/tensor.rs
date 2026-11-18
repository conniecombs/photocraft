//! N-dimensional dense tensors (`f32` and `i64`) with checked shapes, strides and ONNX
//! multidirectional broadcasting.
//!
//! Zero-sized dimensions (such as SAM-2's `[1, 0, 4]` empty box prompt) are valid and carry
//! zero elements. Every allocation checks [`MAX_TENSOR_BYTES`] and integer overflow up front so
//! hostile model metadata or dynamic shape inputs return an error rather than panicking or
//! aborting on out-of-memory.

use crate::MlError;

/// Maximum bytes a single tensor may allocate (512 MiB).
pub const MAX_TENSOR_BYTES: usize = 512 << 20;

/// Maximum rank supported for N-dimensional tensor indexing.
pub const MAX_RANK: usize = 8;

/// Element data type kept at runtime.
///
/// Floating-point initializers stored as `f16`, `bf16` or `f64` convert to `F32` when loaded;
/// integer/index/boolean tensors keep exact 64-bit values in `I64` so dynamic shape pipelines
/// (`Shape -> Gather -> Concat -> Reshape`) and prompt labels (`input_labels`) never lose precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    I64,
}

impl DType {
    /// Bytes per element in memory.
    pub const fn byte_width(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::I64 => 8,
        }
    }
}

/// Backing storage of a [`Tensor`].
#[derive(Clone, Debug, PartialEq)]
pub enum TensorData {
    F32(Vec<f32>),
    I64(Vec<i64>),
}

impl TensorData {
    pub fn dtype(&self) -> DType {
        match self {
            TensorData::F32(_) => DType::F32,
            TensorData::I64(_) => DType::I64,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            TensorData::F32(v) => v.len(),
            TensorData::I64(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// An N-dimensional row-major tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    shape: Vec<usize>,
    strides: Vec<usize>,
    data: TensorData,
}

/// Checked element count for `shape`, enforcing [`MAX_RANK`] and [`MAX_TENSOR_BYTES`].
pub fn checked_numel(shape: &[usize], dtype: DType) -> Result<usize, MlError> {
    if shape.len() > MAX_RANK {
        return Err(MlError::InvalidShape(format!("tensor rank {} exceeds limit {MAX_RANK}", shape.len())));
    }
    let mut total = 1usize;
    for &d in shape {
        total = total.checked_mul(d).ok_or_else(|| MlError::InvalidShape(format!("shape {shape:?} overflows element count")))?;
    }
    let bytes = total.checked_mul(dtype.byte_width()).ok_or(MlError::MemoryLimitExceeded { needed: usize::MAX, limit: MAX_TENSOR_BYTES })?;
    if bytes > MAX_TENSOR_BYTES {
        return Err(MlError::MemoryLimitExceeded { needed: bytes, limit: MAX_TENSOR_BYTES });
    }
    Ok(total)
}

/// Row-major contiguous strides for `shape`.
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let rank = shape.len();
    if rank == 0 {
        return Vec::new();
    }
    let mut strides = vec![1usize; rank];
    let mut acc = 1usize;
    for axis in (0..rank).rev() {
        strides[axis] = acc;
        acc = acc.saturating_mul(shape[axis].max(1));
    }
    strides
}

impl Tensor {
    /// Construct an `f32` tensor after checking `shape` matches `values.len()`.
    pub fn from_f32(shape: Vec<usize>, values: Vec<f32>) -> Result<Self, MlError> {
        let expected = checked_numel(&shape, DType::F32)?;
        if values.len() != expected {
            return Err(MlError::InvalidShape(format!("shape {shape:?} expects {expected} elements, got {}", values.len())));
        }
        let strides = compute_strides(&shape);
        Ok(Self { shape, strides, data: TensorData::F32(values) })
    }

    /// Construct an `i64` tensor after checking `shape` matches `values.len()`.
    pub fn from_i64(shape: Vec<usize>, values: Vec<i64>) -> Result<Self, MlError> {
        let expected = checked_numel(&shape, DType::I64)?;
        if values.len() != expected {
            return Err(MlError::InvalidShape(format!("shape {shape:?} expects {expected} elements, got {}", values.len())));
        }
        let strides = compute_strides(&shape);
        Ok(Self { shape, strides, data: TensorData::I64(values) })
    }

    /// Construct a zero-filled tensor of `shape` and `dtype`.
    pub fn zeros(shape: Vec<usize>, dtype: DType) -> Result<Self, MlError> {
        let n = checked_numel(&shape, dtype)?;
        let mut v_f32 = Vec::new();
        let mut v_i64 = Vec::new();
        let data = match dtype {
            DType::F32 => {
                v_f32.try_reserve_exact(n).map_err(|_| MlError::MemoryLimitExceeded { needed: n * 4, limit: MAX_TENSOR_BYTES })?;
                v_f32.resize(n, 0.0);
                TensorData::F32(v_f32)
            }
            DType::I64 => {
                v_i64.try_reserve_exact(n).map_err(|_| MlError::MemoryLimitExceeded { needed: n * 8, limit: MAX_TENSOR_BYTES })?;
                v_i64.resize(n, 0);
                TensorData::I64(v_i64)
            }
        };
        let strides = compute_strides(&shape);
        Ok(Self { shape, strides, data })
    }

    /// Construct a constant-filled `f32` tensor.
    pub fn full_f32(shape: Vec<usize>, fill: f32) -> Result<Self, MlError> {
        let mut t = Self::zeros(shape, DType::F32)?;
        if let TensorData::F32(v) = &mut t.data {
            v.fill(fill);
        }
        Ok(t)
    }

    /// Construct a constant-filled `i64` tensor.
    pub fn full_i64(shape: Vec<usize>, fill: i64) -> Result<Self, MlError> {
        let mut t = Self::zeros(shape, DType::I64)?;
        if let TensorData::I64(v) = &mut t.data {
            v.fill(fill);
        }
        Ok(t)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn dtype(&self) -> DType {
        self.data.dtype()
    }

    pub fn data(&self) -> &TensorData {
        &self.data
    }

    /// Borrow as `f32` slice, returning an error if `I64`.
    pub fn as_f32(&self) -> Result<&[f32], MlError> {
        match &self.data {
            TensorData::F32(v) => Ok(v),
            TensorData::I64(_) => Err(MlError::TypeMismatch { expected: "f32", actual: "i64" }),
        }
    }

    /// Borrow mutably as `f32` slice.
    pub fn as_f32_mut(&mut self) -> Result<&mut [f32], MlError> {
        match &mut self.data {
            TensorData::F32(v) => Ok(v),
            TensorData::I64(_) => Err(MlError::TypeMismatch { expected: "f32", actual: "i64" }),
        }
    }

    /// Borrow as `i64` slice, returning an error if `F32`.
    pub fn as_i64(&self) -> Result<&[i64], MlError> {
        match &self.data {
            TensorData::I64(v) => Ok(v),
            TensorData::F32(_) => Err(MlError::TypeMismatch { expected: "i64", actual: "f32" }),
        }
    }

    /// Convert any tensor's elements to `f32` (borrowing when already `F32`).
    pub fn to_f32_cow(&self) -> std::borrow::Cow<'_, [f32]> {
        match &self.data {
            TensorData::F32(v) => std::borrow::Cow::Borrowed(v),
            TensorData::I64(v) => std::borrow::Cow::Owned(v.iter().map(|&x| x as f32).collect()),
        }
    }

    /// Convert any tensor's elements to `i64` (borrowing when already `I64`).
    pub fn to_i64_cow(&self) -> std::borrow::Cow<'_, [i64]> {
        match &self.data {
            TensorData::I64(v) => std::borrow::Cow::Borrowed(v),
            TensorData::F32(v) => std::borrow::Cow::Owned(v.iter().map(|&x| if x.is_finite() { x as i64 } else { 0 }).collect()),
        }
    }

    /// Cast to `target` data type.
    pub fn cast(&self, target: DType) -> Result<Self, MlError> {
        if self.dtype() == target {
            return Ok(self.clone());
        }
        match target {
            DType::F32 => Self::from_f32(self.shape.clone(), self.to_f32_cow().into_owned()),
            DType::I64 => Self::from_i64(self.shape.clone(), self.to_i64_cow().into_owned()),
        }
    }

    /// Reshape without copying when element counts match.
    pub fn into_reshaped(mut self, new_shape: Vec<usize>) -> Result<Self, MlError> {
        let expected = checked_numel(&new_shape, self.dtype())?;
        if expected != self.numel() {
            return Err(MlError::InvalidShape(format!(
                "cannot reshape {:?} ({} elements) into {:?} ({expected} elements)",
                self.shape,
                self.numel(),
                new_shape
            )));
        }
        self.strides = compute_strides(&new_shape);
        self.shape = new_shape;
        Ok(self)
    }

    /// Consume into owned `Vec<f32>`.
    pub fn into_f32(self) -> Result<Vec<f32>, MlError> {
        match self.data {
            TensorData::F32(v) => Ok(v),
            TensorData::I64(v) => Ok(v.into_iter().map(|x| x as f32).collect()),
        }
    }
}

/// Resolve a potentially negative ONNX axis in `[-rank, rank - 1]` to `0..rank`.
pub fn resolve_axis(axis: i64, rank: usize) -> Result<usize, MlError> {
    let r = i64::try_from(rank).unwrap_or(0);
    let resolved = if axis < 0 { axis.saturating_add(r) } else { axis };
    if resolved < 0 || resolved >= r {
        return Err(MlError::InvalidShape(format!("axis {axis} out of bounds for rank {rank}")));
    }
    Ok(resolved as usize)
}

/// Compute the ONNX multidirectional broadcast shape of `a` and `b`.
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, MlError> {
    let out_rank = a.len().max(b.len());
    if out_rank > MAX_RANK {
        return Err(MlError::InvalidShape(format!("broadcast rank {out_rank} exceeds {MAX_RANK}")));
    }
    let mut out = vec![1usize; out_rank];
    for (i, dst_dim) in out.iter_mut().enumerate() {
        let da = a.len().checked_sub(out_rank - i).and_then(|idx| a.get(idx)).copied().unwrap_or(1);
        let db = b.len().checked_sub(out_rank - i).and_then(|idx| b.get(idx)).copied().unwrap_or(1);
        *dst_dim = if da == db {
            da
        } else if da == 1 {
            db
        } else if db == 1 {
            da
        } else {
            return Err(MlError::InvalidShape(format!("cannot broadcast shapes {a:?} and {b:?}")));
        };
    }
    Ok(out)
}

/// Compute broadcast strides aligned to `out_shape` (`0` where a dimension of size `1` broadcasts).
pub fn broadcast_strides(src_shape: &[usize], src_strides: &[usize], out_shape: &[usize]) -> Result<[usize; MAX_RANK], MlError> {
    if src_shape.len() > out_shape.len() || out_shape.len() > MAX_RANK {
        return Err(MlError::InvalidShape(format!("cannot align shape {src_shape:?} to {out_shape:?}")));
    }
    let mut aligned = [0usize; MAX_RANK];
    let offset = out_shape.len() - src_shape.len();
    for (i, (&sd, &ss)) in src_shape.iter().zip(src_strides.iter()).enumerate() {
        let od = out_shape[offset + i];
        if sd == od {
            aligned[offset + i] = if od <= 1 { 0 } else { ss };
        } else if sd == 1 {
            aligned[offset + i] = 0;
        } else {
            return Err(MlError::InvalidShape(format!("cannot broadcast dimension {sd} into {od}")));
        }
    }
    Ok(aligned)
}

/// Unravel a flat row-major index `linear` in `shape` (with `strides`) and compute the flat
/// offset using `aligned_strides`.
#[inline]
pub fn unravel_offset(mut linear: usize, out_strides: &[usize], rank: usize, aligned: &[usize; MAX_RANK]) -> usize {
    let mut off = 0usize;
    for axis in 0..rank {
        let s = out_strides[axis];
        let coord = linear / s;
        linear %= s;
        off = off.saturating_add(coord.saturating_mul(aligned[axis]));
    }
    off
}
