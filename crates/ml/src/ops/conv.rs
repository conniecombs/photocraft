//! Spatial vision ONNX operators (`Conv`, `ConvTranspose`, `MaxPool`, `AveragePool`,
//! `GlobalAveragePool`, `GlobalMaxPool`, `Resize`, `Upsample`, `GridSample`).

#[cfg(not(target_arch = "wasm32"))]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(not(target_arch = "wasm32"))]
use rayon::slice::ParallelSliceMut;

use crate::MlError;
use crate::backend::{ComputeAccelerator, Conv2dSpec};
use crate::onnx::Node;
use crate::ops::linalg::sgemm_slice;
use crate::tensor::{DType, Tensor, checked_numel};

fn req<'a>(inputs: &'a [Option<&'a Tensor>], idx: usize, op: &str) -> Result<&'a Tensor, MlError> {
    inputs.get(idx).and_then(|t| *t).ok_or_else(|| MlError::InvalidModel(format!("{op} missing input {idx}")))
}

fn non_empty<'a>(inputs: &'a [Option<&'a Tensor>], idx: usize) -> Option<&'a Tensor> {
    inputs.get(idx).and_then(|t| *t).filter(|t| t.numel() > 0)
}

#[derive(Clone, Copy, Debug)]
struct Spatial2dParams {
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    pt: usize,
    pl: usize,
    pb: usize,
    pr: usize,
    dh: usize,
    dw: usize,
}

fn parse_pair(vals: Option<&[i64]>, default: usize, is_1d: bool, name: &str) -> Result<(usize, usize), MlError> {
    let Some(s) = vals else {
        return Ok((if is_1d { 1 } else { default }, default));
    };
    let check = |v: i64| usize::try_from(v).map_err(|_| MlError::InvalidShape(format!("negative {name} {v}")));
    if is_1d {
        if s.is_empty() {
            return Ok((1, default));
        }
        Ok((1, check(s[0])?))
    } else if s.len() >= 2 {
        Ok((check(s[0])?, check(s[1])?))
    } else if s.len() == 1 {
        let v = check(s[0])?;
        Ok((v, v))
    } else {
        Ok((default, default))
    }
}

fn parse_pads_2d(
    node: &Node,
    in_hw: (usize, usize),
    eff_k: (usize, usize),
    stride: (usize, usize),
    is_1d: bool,
) -> Result<(usize, usize, usize, usize), MlError> {
    let auto_pad = node.attr_str("auto_pad").unwrap_or("NOTSET");
    match auto_pad {
        "VALID" => Ok((0, 0, 0, 0)),
        "SAME_UPPER" | "SAME_LOWER" => {
            let calc = |in_d: usize, eff_kd: usize, sd: usize| -> (usize, usize) {
                let out_d = in_d.div_ceil(sd);
                let needed = (out_d.saturating_sub(1)).saturating_mul(sd).saturating_add(eff_kd);
                let total_pad = needed.saturating_sub(in_d);
                if auto_pad == "SAME_UPPER" { (total_pad / 2, total_pad - total_pad / 2) } else { (total_pad - total_pad / 2, total_pad / 2) }
            };
            let (pt, pb) = if is_1d { (0, 0) } else { calc(in_hw.0, eff_k.0, stride.0) };
            let (pl, pr) = calc(in_hw.1, eff_k.1, stride.1);
            Ok((pt, pl, pb, pr))
        }
        _ => {
            let Some(pads) = node.attr_ints("pads") else {
                return Ok((0, 0, 0, 0));
            };
            let check = |v: i64| usize::try_from(v).map_err(|_| MlError::InvalidShape(format!("negative pad {v}")));
            if is_1d && pads.len() >= 2 {
                Ok((0, check(pads[0])?, 0, check(pads[1])?))
            } else if pads.len() >= 4 {
                Ok((check(pads[0])?, check(pads[1])?, check(pads[2])?, check(pads[3])?))
            } else {
                Ok((0, 0, 0, 0))
            }
        }
    }
}

fn parse_spatial_2d(node: &Node, in_hw: (usize, usize), k_hw: (usize, usize), is_1d: bool) -> Result<Spatial2dParams, MlError> {
    let (sh, sw) = parse_pair(node.attr_ints("strides"), 1, is_1d, "stride")?;
    let (dh, dw) = parse_pair(node.attr_ints("dilations"), 1, is_1d, "dilation")?;
    if sh == 0 || sw == 0 || dh == 0 || dw == 0 || k_hw.0 == 0 || k_hw.1 == 0 {
        return Err(MlError::InvalidShape("zero stride, dilation or kernel size".into()));
    }
    let eff_kh = dh.saturating_mul(k_hw.0 - 1) + 1;
    let eff_kw = dw.saturating_mul(k_hw.1 - 1) + 1;
    let (pt, pl, pb, pr) = parse_pads_2d(node, in_hw, (eff_kh, eff_kw), (sh, sw), is_1d)?;
    Ok(Spatial2dParams { kh: k_hw.0, kw: k_hw.1, sh, sw, pt, pl, pb, pr, dh, dw })
}

fn conv_out_dim(in_d: usize, p_before: usize, p_after: usize, dil: usize, k: usize, stride: usize, ceil_mode: bool) -> Result<usize, MlError> {
    let eff_k = dil.checked_mul(k - 1).and_then(|v| v.checked_add(1)).ok_or_else(|| MlError::InvalidShape("kernel overflow".into()))?;
    let padded = in_d.checked_add(p_before).and_then(|v| v.checked_add(p_after)).ok_or_else(|| MlError::InvalidShape("padded size overflow".into()))?;
    if padded < eff_k {
        return Err(MlError::InvalidShape(format!("padded input {padded} smaller than effective kernel {eff_k}")));
    }
    let diff = padded - eff_k;
    Ok(if ceil_mode { diff.div_ceil(stride) + 1 } else { diff / stride + 1 })
}

fn conv2d_depthwise_cpu(spec: &Conv2dSpec, x: &[f32], w: &[f32], bias: Option<&[f32]>) -> Vec<f32> {
    let out_plane = spec.out_h * spec.out_w;
    let in_plane = spec.in_h * spec.in_w;
    let k_plane = spec.kernel_h * spec.kernel_w;
    let mut out = vec![0.0f32; spec.batch * spec.out_channels * out_plane];

    let compute_channel = |boc: usize, dst: &mut [f32]| {
        let b = boc / spec.out_channels;
        let ch = boc % spec.out_channels;
        let b_val = bias.and_then(|bv| bv.get(ch).copied()).unwrap_or(0.0);
        let in_slice = &x[(b * spec.in_channels + ch) * in_plane..(b * spec.in_channels + ch + 1) * in_plane];
        let w_slice = &w[ch * k_plane..(ch + 1) * k_plane];

        for oh in 0..spec.out_h {
            let ih_base = (oh * spec.stride_h) as isize - spec.pad_top as isize;
            let dst_row = &mut dst[oh * spec.out_w..(oh + 1) * spec.out_w];
            dst_row.fill(b_val);
            for ky in 0..spec.kernel_h {
                let ih = ih_base + (ky * spec.dil_h) as isize;
                if ih < 0 || (ih as usize) >= spec.in_h {
                    continue;
                }
                let in_row = &in_slice[(ih as usize) * spec.in_w..((ih as usize) + 1) * spec.in_w];
                for kx in 0..spec.kernel_w {
                    let wv = w_slice[ky * spec.kernel_w + kx];
                    let kx_off = (kx * spec.dil_w) as isize - spec.pad_left as isize;
                    for (ow, dst_v) in dst_row.iter_mut().enumerate() {
                        let iw = (ow * spec.stride_w) as isize + kx_off;
                        if iw >= 0 && (iw as usize) < spec.in_w {
                            *dst_v += in_row[iw as usize] * wv;
                        }
                    }
                }
            }
        }
    };

    #[cfg(not(target_arch = "wasm32"))]
    if spec.batch * spec.out_channels > 1 && out.len() >= 4096 {
        out.par_chunks_mut(out_plane).enumerate().for_each(|(boc, dst)| compute_channel(boc, dst));
        return out;
    }

    for (boc, dst) in out.chunks_mut(out_plane).enumerate() {
        compute_channel(boc, dst);
    }
    out
}

fn conv2d_im2col_group(spec: &Conv2dSpec, in_group: &[f32], w_group: &[f32], bias_group: Option<&[f32]>, out_group: &mut [f32]) {
    let in_c_g = spec.in_channels / spec.groups;
    let out_c_g = spec.out_channels / spec.groups;
    let out_spatial = spec.out_h * spec.out_w;
    let in_spatial = spec.in_h * spec.in_w;

    // Fast path for 1x1 stride-1 pad-0 pointwise convolution: zero im2col copy!
    if spec.kernel_h == 1 && spec.kernel_w == 1 && spec.stride_h == 1 && spec.stride_w == 1 && spec.pad_top == 0 && spec.pad_left == 0 {
        sgemm_slice(out_c_g, in_c_g, out_spatial, w_group, in_group, out_group);
        if let Some(bg) = bias_group {
            for (oc, &bv) in bg.iter().enumerate().take(out_c_g) {
                for v in &mut out_group[oc * out_spatial..(oc + 1) * out_spatial] {
                    *v += bv;
                }
            }
        }
        return;
    }

    const COL_TILE: usize = 1024;
    let k_col = in_c_g * spec.kernel_h * spec.kernel_w;
    let mut col_buf = vec![0.0f32; k_col * out_spatial.min(COL_TILE)];
    let mut tile_out = vec![0.0f32; out_c_g * out_spatial.min(COL_TILE)];

    let mut sp_start = 0usize;
    while sp_start < out_spatial {
        let tile_len = (out_spatial - sp_start).min(COL_TILE);
        let col = &mut col_buf[..k_col * tile_len];
        col.fill(0.0);

        for ic in 0..in_c_g {
            let in_plane = &in_group[ic * in_spatial..(ic + 1) * in_spatial];
            for ky in 0..spec.kernel_h {
                let ky_dil = (ky * spec.dil_h) as isize - spec.pad_top as isize;
                for kx in 0..spec.kernel_w {
                    let kx_dil = (kx * spec.dil_w) as isize - spec.pad_left as isize;
                    let row_idx = (ic * spec.kernel_h + ky) * spec.kernel_w + kx;
                    let col_row = &mut col[row_idx * tile_len..(row_idx + 1) * tile_len];
                    for (t, dst_col) in col_row.iter_mut().enumerate() {
                        let sp = sp_start + t;
                        let oh = sp / spec.out_w;
                        let ow = sp % spec.out_w;
                        let ih = (oh * spec.stride_h) as isize + ky_dil;
                        let iw = (ow * spec.stride_w) as isize + kx_dil;
                        if ih >= 0 && (ih as usize) < spec.in_h && iw >= 0 && (iw as usize) < spec.in_w {
                            *dst_col = in_plane[(ih as usize) * spec.in_w + (iw as usize)];
                        }
                    }
                }
            }
        }

        let tout = &mut tile_out[..out_c_g * tile_len];
        sgemm_slice(out_c_g, k_col, tile_len, w_group, col, tout);

        for oc in 0..out_c_g {
            let bv = bias_group.and_then(|bg| bg.get(oc).copied()).unwrap_or(0.0);
            let dst = &mut out_group[oc * out_spatial + sp_start..oc * out_spatial + sp_start + tile_len];
            let src = &tout[oc * tile_len..(oc + 1) * tile_len];
            for (d, &s) in dst.iter_mut().zip(src.iter()) {
                *d = s + bv;
            }
        }
        sp_start += tile_len;
    }
}

fn run_conv(node: &Node, inputs: &[Option<&Tensor>], accel: &dyn ComputeAccelerator) -> Result<Tensor, MlError> {
    let x = req(inputs, 0, "Conv")?;
    let w = req(inputs, 1, "Conv")?;
    let bias_t = inputs.get(2).and_then(|t| *t);
    if x.rank() != w.rank() || (x.rank() != 3 && x.rank() != 4) {
        return Err(MlError::InvalidShape(format!("Conv supports 3D (NCL) and 4D (NCHW), got {:?} and {:?}", x.shape(), w.shape())));
    }
    let is_1d = x.rank() == 3;
    let batch = x.shape()[0];
    let in_c = x.shape()[1];
    let (in_h, in_w) = if is_1d { (1, x.shape()[2]) } else { (x.shape()[2], x.shape()[3]) };
    let out_c = w.shape()[0];
    let w_ic = w.shape()[1];
    let (kh, kw) = if is_1d { (1, w.shape()[2]) } else { (w.shape()[2], w.shape()[3]) };

    let groups =
        usize::try_from(node.attr_int("group").unwrap_or(1)).ok().filter(|&g| g >= 1).ok_or_else(|| MlError::InvalidShape("Conv invalid group".into()))?;
    if in_c != w_ic.saturating_mul(groups) || out_c % groups != 0 {
        return Err(MlError::InvalidShape(format!("Conv channel mismatch: X C={in_c}, W={:?}, groups={groups}", w.shape())));
    }
    if let Some(b) = bias_t
        && b.numel() != out_c
    {
        return Err(MlError::InvalidShape(format!("Conv bias length {} != out_channels {out_c}", b.numel())));
    }

    let sp = parse_spatial_2d(node, (in_h, in_w), (kh, kw), is_1d)?;
    let out_h = if is_1d { 1 } else { conv_out_dim(in_h, sp.pt, sp.pb, sp.dh, sp.kh, sp.sh, false)? };
    let out_w = conv_out_dim(in_w, sp.pl, sp.pr, sp.dw, sp.kw, sp.sw, false)?;
    let out_shape = if is_1d { vec![batch, out_c, out_w] } else { vec![batch, out_c, out_h, out_w] };
    let total = checked_numel(&out_shape, DType::F32)?;
    if total == 0 {
        return Tensor::zeros(out_shape, DType::F32);
    }

    let spec = Conv2dSpec {
        batch,
        in_channels: in_c,
        out_channels: out_c,
        in_h,
        in_w,
        out_h,
        out_w,
        kernel_h: sp.kh,
        kernel_w: sp.kw,
        stride_h: sp.sh,
        stride_w: sp.sw,
        pad_top: sp.pt,
        pad_left: sp.pl,
        dil_h: sp.dh,
        dil_w: sp.dw,
        groups,
    };

    let x_data = x.to_f32_cow();
    let w_data = w.to_f32_cow();
    let b_cow = bias_t.map(|b| b.to_f32_cow());
    let b_slice = b_cow.as_deref();

    if accel.is_gpu()
        && let Some(gpu_out) = accel.conv2d(&spec, &x_data, &w_data, b_slice)
    {
        return Tensor::from_f32(out_shape, gpu_out);
    }

    if groups == in_c && out_c == in_c {
        return Tensor::from_f32(out_shape, conv2d_depthwise_cpu(&spec, &x_data, &w_data, b_slice));
    }

    let in_c_g = in_c / groups;
    let out_c_g = out_c / groups;
    let in_g_elems = in_c_g * in_h * in_w;
    let out_g_elems = out_c_g * out_h * out_w;
    let w_g_elems = out_c_g * in_c_g * sp.kh * sp.kw;
    let mut out = vec![0.0f32; total];

    let run_bg = |bg_idx: usize, dst_group: &mut [f32]| {
        let b = bg_idx / groups;
        let g = bg_idx % groups;
        let in_g = &x_data[(b * groups + g) * in_g_elems..(b * groups + g + 1) * in_g_elems];
        let w_g = &w_data[g * w_g_elems..(g + 1) * w_g_elems];
        let b_g = b_slice.map(|bs| &bs[g * out_c_g..(g + 1) * out_c_g]);
        conv2d_im2col_group(&spec, in_g, w_g, b_g, dst_group);
    };

    #[cfg(not(target_arch = "wasm32"))]
    if batch * groups > 1 && total >= 8192 {
        out.par_chunks_mut(out_g_elems).enumerate().for_each(|(bg_idx, dst)| run_bg(bg_idx, dst));
        return Tensor::from_f32(out_shape, out);
    }

    for (bg_idx, dst) in out.chunks_mut(out_g_elems).enumerate() {
        run_bg(bg_idx, dst);
    }
    Tensor::from_f32(out_shape, out)
}

fn run_conv_transpose(node: &Node, inputs: &[Option<&Tensor>], accel: &dyn ComputeAccelerator) -> Result<Tensor, MlError> {
    let x = req(inputs, 0, "ConvTranspose")?;
    let w = req(inputs, 1, "ConvTranspose")?;
    let bias_t = inputs.get(2).and_then(|t| *t);
    if x.rank() != w.rank() || (x.rank() != 3 && x.rank() != 4) {
        return Err(MlError::InvalidShape(format!("ConvTranspose supports 3D and 4D, got {:?}", x.shape())));
    }
    let is_1d = x.rank() == 3;
    let batch = x.shape()[0];
    let in_c = x.shape()[1];
    let (in_h, in_w) = if is_1d { (1, x.shape()[2]) } else { (x.shape()[2], x.shape()[3]) };
    let groups = usize::try_from(node.attr_int("group").unwrap_or(1))
        .ok()
        .filter(|&g| g >= 1)
        .ok_or_else(|| MlError::InvalidShape("ConvTranspose invalid group".into()))?;
    if w.shape()[0] != in_c || in_c % groups != 0 {
        return Err(MlError::InvalidShape(format!("ConvTranspose W={:?} incompatible with C_in={in_c}, groups={groups}", w.shape())));
    }
    let out_c_g = w.shape()[1];
    let out_c = out_c_g.checked_mul(groups).ok_or_else(|| MlError::InvalidShape("ConvTranspose channel overflow".into()))?;
    let (kh, kw) = if is_1d { (1, w.shape()[2]) } else { (w.shape()[2], w.shape()[3]) };
    if let Some(b) = bias_t
        && b.numel() != out_c
    {
        return Err(MlError::InvalidShape(format!("ConvTranspose bias length {} != {out_c}", b.numel())));
    }

    let sp = parse_spatial_2d(node, (in_h, in_w), (kh, kw), is_1d)?;
    let (op_h, op_w) = parse_pair(node.attr_ints("output_padding"), 0, is_1d, "output_padding")?;
    let calc_dim = |in_d: usize, s: usize, d: usize, k: usize, pb: usize, pa: usize, op: usize| -> Result<usize, MlError> {
        let base = in_d
            .saturating_sub(1)
            .checked_mul(s)
            .and_then(|v| v.checked_add(op))
            .and_then(|v| v.checked_add(d * (k - 1) + 1))
            .ok_or_else(|| MlError::InvalidShape("ConvTranspose dim overflow".into()))?;
        base.checked_sub(pb + pa).ok_or_else(|| MlError::InvalidShape("ConvTranspose pads exceed output size".into()))
    };
    let mut out_h = if is_1d { 1 } else { calc_dim(in_h, sp.sh, sp.dh, sp.kh, sp.pt, sp.pb, op_h)? };
    let mut out_w = calc_dim(in_w, sp.sw, sp.dw, sp.kw, sp.pl, sp.pr, op_w)?;
    if let Some(os) = node.attr_ints("output_shape") {
        if is_1d && !os.is_empty() {
            out_w = usize::try_from(os[os.len() - 1]).map_err(|_| MlError::InvalidShape("negative output_shape".into()))?;
        } else if os.len() >= 2 {
            out_h = usize::try_from(os[os.len() - 2]).map_err(|_| MlError::InvalidShape("negative output_shape".into()))?;
            out_w = usize::try_from(os[os.len() - 1]).map_err(|_| MlError::InvalidShape("negative output_shape".into()))?;
        }
    }

    let out_shape = if is_1d { vec![batch, out_c, out_w] } else { vec![batch, out_c, out_h, out_w] };
    let total = checked_numel(&out_shape, DType::F32)?;
    if total == 0 {
        return Tensor::zeros(out_shape, DType::F32);
    }

    let spec = Conv2dSpec {
        batch,
        in_channels: in_c,
        out_channels: out_c,
        in_h,
        in_w,
        out_h,
        out_w,
        kernel_h: sp.kh,
        kernel_w: sp.kw,
        stride_h: sp.sh,
        stride_w: sp.sw,
        pad_top: sp.pt,
        pad_left: sp.pl,
        dil_h: sp.dh,
        dil_w: sp.dw,
        groups,
    };

    let x_data = x.to_f32_cow();
    let w_data = w.to_f32_cow();
    let b_cow = bias_t.map(|b| b.to_f32_cow());
    let b_slice = b_cow.as_deref();

    if accel.is_gpu()
        && let Some(gpu_out) = accel.conv_transpose2d(&spec, &x_data, &w_data, b_slice)
    {
        return Tensor::from_f32(out_shape, gpu_out);
    }

    let in_c_g = in_c / groups;
    let out_plane = out_h * out_w;
    let mut out = vec![0.0f32; total];

    let compute_oc = |boc: usize, dst: &mut [f32]| {
        let b = boc / out_c;
        let oc = boc % out_c;
        let g = oc / out_c_g;
        let oc_local = oc % out_c_g;
        let ic_start = g * in_c_g;
        let b_val = b_slice.and_then(|bs| bs.get(oc).copied()).unwrap_or(0.0);
        dst.fill(b_val);

        for ic_l in 0..in_c_g {
            let ic = ic_start + ic_l;
            let x_plane = &x_data[(b * in_c + ic) * in_h * in_w..(b * in_c + ic + 1) * in_h * in_w];
            let w_plane = &w_data[(ic * out_c_g + oc_local) * sp.kh * sp.kw..(ic * out_c_g + oc_local + 1) * sp.kh * sp.kw];
            for ih in 0..in_h {
                for ky in 0..sp.kh {
                    let oh_s = (ih * sp.sh + ky * sp.dh) as isize - sp.pt as isize;
                    if oh_s < 0 || (oh_s as usize) >= out_h {
                        continue;
                    }
                    let dst_row = &mut dst[(oh_s as usize) * out_w..((oh_s as usize) + 1) * out_w];
                    let x_row = &x_plane[ih * in_w..(ih + 1) * in_w];
                    for kx in 0..sp.kw {
                        let wv = w_plane[ky * sp.kw + kx];
                        let kx_off = (kx * sp.dw) as isize - sp.pl as isize;
                        for (iw, &xv) in x_row.iter().enumerate() {
                            let ow_s = (iw * sp.sw) as isize + kx_off;
                            if ow_s >= 0 && (ow_s as usize) < out_w {
                                dst_row[ow_s as usize] += xv * wv;
                            }
                        }
                    }
                }
            }
        }
    };

    #[cfg(not(target_arch = "wasm32"))]
    if batch * out_c > 1 && total >= 4096 {
        out.par_chunks_mut(out_plane).enumerate().for_each(|(boc, dst)| compute_oc(boc, dst));
        return Tensor::from_f32(out_shape, out);
    }

    for (boc, dst) in out.chunks_mut(out_plane).enumerate() {
        compute_oc(boc, dst);
    }
    Tensor::from_f32(out_shape, out)
}

fn run_pool(node: &Node, x: &Tensor, is_max: bool) -> Result<Tensor, MlError> {
    if x.rank() != 3 && x.rank() != 4 {
        return Err(MlError::InvalidShape(format!("Pool expects 3D or 4D tensor, got {:?}", x.shape())));
    }
    let is_1d = x.rank() == 3;
    let (batch, c) = (x.shape()[0], x.shape()[1]);
    let (in_h, in_w) = if is_1d { (1, x.shape()[2]) } else { (x.shape()[2], x.shape()[3]) };
    let ks = node.attr_ints("kernel_shape").ok_or_else(|| MlError::InvalidModel("Pool missing kernel_shape".into()))?;
    let (kh, kw) = parse_pair(Some(ks), 1, is_1d, "kernel_shape")?;
    let sp = parse_spatial_2d(node, (in_h, in_w), (kh, kw), is_1d)?;
    let ceil_mode = node.attr_int("ceil_mode").unwrap_or(0) != 0;
    let include_pad = node.attr_int("count_include_pad").unwrap_or(0) != 0;

    let out_h = if is_1d { 1 } else { conv_out_dim(in_h, sp.pt, sp.pb, sp.dh, sp.kh, sp.sh, ceil_mode)? };
    let out_w = conv_out_dim(in_w, sp.pl, sp.pr, sp.dw, sp.kw, sp.sw, ceil_mode)?;
    let out_shape = if is_1d { vec![batch, c, out_w] } else { vec![batch, c, out_h, out_w] };
    let total = checked_numel(&out_shape, DType::F32)?;
    if total == 0 {
        return Tensor::zeros(out_shape, DType::F32);
    }

    let src = x.to_f32_cow();
    let mut out = vec![0.0f32; total];
    let in_plane = in_h * in_w;
    let out_plane = out_h * out_w;
    let full_area = (sp.kh * sp.kw) as f32;

    for bc in 0..batch * c {
        let plane = &src[bc * in_plane..(bc + 1) * in_plane];
        let dst = &mut out[bc * out_plane..(bc + 1) * out_plane];
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut acc = if is_max { f32::NEG_INFINITY } else { 0.0 };
                let mut valid = 0usize;
                for ky in 0..sp.kh {
                    let ih = (oh * sp.sh + ky * sp.dh) as isize - sp.pt as isize;
                    if ih < 0 || (ih as usize) >= in_h {
                        continue;
                    }
                    for kx in 0..sp.kw {
                        let iw = (ow * sp.sw + kx * sp.dw) as isize - sp.pl as isize;
                        if iw < 0 || (iw as usize) >= in_w {
                            continue;
                        }
                        let v = plane[(ih as usize) * in_w + (iw as usize)];
                        if is_max {
                            if v > acc {
                                acc = v;
                            }
                        } else {
                            acc += v;
                        }
                        valid += 1;
                    }
                }
                dst[oh * out_w + ow] = if is_max {
                    if valid == 0 { 0.0 } else { acc }
                } else {
                    let denom = if include_pad { full_area } else { (valid.max(1)) as f32 };
                    acc / denom
                };
            }
        }
    }
    Tensor::from_f32(out_shape, out)
}

fn run_global_pool(x: &Tensor, is_max: bool) -> Result<Tensor, MlError> {
    if x.rank() < 3 {
        return Err(MlError::InvalidShape(format!("GlobalPool expects rank >= 3, got {:?}", x.shape())));
    }
    let (batch, c) = (x.shape()[0], x.shape()[1]);
    let spatial: usize = x.shape()[2..].iter().product();
    let mut out_shape = vec![1usize; x.rank()];
    out_shape[0] = batch;
    out_shape[1] = c;
    if batch == 0 || c == 0 || spatial == 0 {
        return Tensor::zeros(out_shape, DType::F32);
    }
    let src = x.to_f32_cow();
    let mut out = vec![0.0f32; batch * c];
    let inv = 1.0 / (spatial as f64);
    for (bc, dst) in out.iter_mut().enumerate() {
        let slice = &src[bc * spatial..(bc + 1) * spatial];
        *dst = if is_max { slice.iter().copied().fold(f32::NEG_INFINITY, f32::max) } else { (slice.iter().map(|&v| f64::from(v)).sum::<f64>() * inv) as f32 };
    }
    Tensor::from_f32(out_shape, out)
}

#[inline]
fn map_resize_coord(coord_out: usize, len_in: usize, len_out: usize, scale: f32, coord_mode: &str) -> f32 {
    let x = coord_out as f32;
    match coord_mode {
        "align_corners" => {
            if len_out <= 1 {
                0.0
            } else {
                x * ((len_in.saturating_sub(1)) as f32) / ((len_out - 1) as f32)
            }
        }
        "asymmetric" => x / scale.max(1e-12),
        "pytorch_half_pixel" => {
            if len_out > 1 {
                (x + 0.5) / scale.max(1e-12) - 0.5
            } else {
                0.0
            }
        }
        "half_pixel_symmetric" => {
            let s = scale.max(1e-12);
            let adj = (len_out as f32) / (s * (len_in.max(1) as f32));
            let center = (len_in as f32) * 0.5;
            center * (1.0 - adj) + (x + 0.5) / s - 0.5
        }
        // "half_pixel" and default
        _ => (x + 0.5) / scale.max(1e-12) - 0.5,
    }
}

#[inline]
fn round_nearest(pos: f32, nearest_mode: &str, len_in: usize) -> usize {
    let idx = match nearest_mode {
        "floor" => pos.floor(),
        "ceil" => pos.ceil(),
        "round_prefer_ceil" => (pos + 0.5).floor(),
        // "round_prefer_floor" default: ties (.5) round down
        _ => {
            let f = pos.floor();
            if pos - f > 0.5 { f + 1.0 } else { f }
        }
    };
    (idx.clamp(0.0, (len_in.saturating_sub(1)) as f32)) as usize
}

#[inline]
fn cubic_weights(t: f32, a: f32) -> [f32; 4] {
    let d0 = 1.0 + t;
    let d1 = t;
    let d2 = 1.0 - t;
    let d3 = 2.0 - t;
    [
        a * d0 * d0 * d0 - 5.0 * a * d0 * d0 + 8.0 * a * d0 - 4.0 * a,
        (a + 2.0) * d1 * d1 * d1 - (a + 3.0) * d1 * d1 + 1.0,
        (a + 2.0) * d2 * d2 * d2 - (a + 3.0) * d2 * d2 + 1.0,
        a * d3 * d3 * d3 - 5.0 * a * d3 * d3 + 8.0 * a * d3 - 4.0 * a,
    ]
}

fn resolve_resize_shape(node: &Node, inputs: &[Option<&Tensor>], in_shape: &[usize]) -> Result<(Vec<usize>, Vec<f32>), MlError> {
    let rank = in_shape.len();
    let is_upsample = node.op_type == "Upsample";

    let sizes_t = if is_upsample { None } else { non_empty(inputs, 3) };
    let scales_t =
        if is_upsample { non_empty(inputs, 1) } else { non_empty(inputs, 2).or_else(|| if inputs.len() == 2 { non_empty(inputs, 1) } else { None }) };

    if let Some(sz) = sizes_t {
        let raw = sz.to_i64_cow();
        if raw.len() != rank {
            return Err(MlError::InvalidShape(format!("Resize sizes length {} != rank {rank}", raw.len())));
        }
        let mut out_shape = Vec::with_capacity(rank);
        let mut scales = Vec::with_capacity(rank);
        for (&d_out_i, &d_in) in raw.iter().zip(in_shape.iter()) {
            let d_out = usize::try_from(d_out_i).map_err(|_| MlError::InvalidShape(format!("negative Resize size {d_out_i}")))?;
            out_shape.push(d_out);
            scales.push(if d_in == 0 { 1.0 } else { (d_out as f32) / (d_in as f32) });
        }
        return Ok((out_shape, scales));
    }

    let scales: Vec<f32> = if let Some(sc) = scales_t {
        sc.to_f32_cow().into_owned()
    } else if let Some(attr_sc) = node.attr_floats("scales") {
        attr_sc.to_vec()
    } else {
        return Err(MlError::InvalidModel("Resize requires non-empty `scales` or `sizes`".into()));
    };

    if scales.len() != rank {
        return Err(MlError::InvalidShape(format!("Resize scales length {} != rank {rank}", scales.len())));
    }
    let mut out_shape = Vec::with_capacity(rank);
    for (&d_in, &sc) in in_shape.iter().zip(scales.iter()) {
        if !sc.is_finite() || sc < 0.0 {
            return Err(MlError::InvalidShape(format!("invalid Resize scale {sc}")));
        }
        out_shape.push(((d_in as f32) * sc).floor() as usize);
    }
    Ok((out_shape, scales))
}

fn run_resize(node: &Node, inputs: &[Option<&Tensor>]) -> Result<Tensor, MlError> {
    let x = req(inputs, 0, &node.op_type)?;
    if x.rank() != 3 && x.rank() != 4 {
        return Err(MlError::InvalidShape(format!("Resize supports 3D (NCL) and 4D (NCHW), got {:?}", x.shape())));
    }
    let (out_shape, scales) = resolve_resize_shape(node, inputs, x.shape())?;
    if out_shape == x.shape() {
        return Ok(x.clone());
    }
    let total = checked_numel(&out_shape, x.dtype())?;
    if total == 0 || x.numel() == 0 {
        return Tensor::zeros(out_shape, x.dtype());
    }

    let is_1d = x.rank() == 3;
    if out_shape[0] != x.shape()[0] || out_shape[1] != x.shape()[1] {
        return Err(MlError::InvalidShape(format!("Resize N/C scaling ({:?} -> {:?}) is not supported", x.shape(), out_shape)));
    }

    let mode = node.attr_str("mode").unwrap_or("nearest");
    let coord_mode = node.attr_str("coordinate_transformation_mode").unwrap_or(if node.op_type == "Upsample" { "asymmetric" } else { "half_pixel" });
    let nearest_mode = node.attr_str("nearest_mode").unwrap_or("round_prefer_floor");
    let cubic_a = node.attr_float("cubic_coeff_a").unwrap_or(-0.75);

    let (batch, c) = (x.shape()[0], x.shape()[1]);
    let (in_h, in_w) = if is_1d { (1, x.shape()[2]) } else { (x.shape()[2], x.shape()[3]) };
    let (out_h, out_w) = if is_1d { (1, out_shape[2]) } else { (out_shape[2], out_shape[3]) };
    let (scale_h, scale_w) = if is_1d { (1.0, scales[2]) } else { (scales[2], scales[3]) };

    let y_coords: Vec<f32> = (0..out_h).map(|oh| if is_1d { 0.0 } else { map_resize_coord(oh, in_h, out_h, scale_h, coord_mode) }).collect();
    let x_coords: Vec<f32> = (0..out_w).map(|ow| map_resize_coord(ow, in_w, out_w, scale_w, coord_mode)).collect();

    let src = x.to_f32_cow();
    let mut out = vec![0.0f32; total];
    let in_plane = in_h * in_w;
    let out_plane = out_h * out_w;

    let resize_plane = |bc: usize, dst: &mut [f32]| {
        let plane = &src[bc * in_plane..(bc + 1) * in_plane];
        match mode {
            "linear" | "bilinear" => {
                for (oh, &fy) in y_coords.iter().enumerate() {
                    let yc = fy.clamp(0.0, (in_h - 1) as f32);
                    let y0 = yc.floor() as usize;
                    let y1 = (y0 + 1).min(in_h - 1);
                    let wy = yc - (y0 as f32);
                    let row0 = &plane[y0 * in_w..(y0 + 1) * in_w];
                    let row1 = &plane[y1 * in_w..(y1 + 1) * in_w];
                    let dst_row = &mut dst[oh * out_w..(oh + 1) * out_w];
                    for (ow, &fx) in x_coords.iter().enumerate() {
                        let xc = fx.clamp(0.0, (in_w - 1) as f32);
                        let x0 = xc.floor() as usize;
                        let x1 = (x0 + 1).min(in_w - 1);
                        let wx = xc - (x0 as f32);
                        let top = row0[x0] * (1.0 - wx) + row0[x1] * wx;
                        let bot = row1[x0] * (1.0 - wx) + row1[x1] * wx;
                        dst_row[ow] = top * (1.0 - wy) + bot * wy;
                    }
                }
            }
            "cubic" => {
                for (oh, &fy) in y_coords.iter().enumerate() {
                    let iy = fy.floor() as isize;
                    let wy = cubic_weights(fy - (iy as f32), cubic_a);
                    let dst_row = &mut dst[oh * out_w..(oh + 1) * out_w];
                    for (ow, &fx) in x_coords.iter().enumerate() {
                        let ix = fx.floor() as isize;
                        let wx = cubic_weights(fx - (ix as f32), cubic_a);
                        let mut acc = 0.0f32;
                        for (j, &wy_j) in wy.iter().enumerate() {
                            let sy = (iy + (j as isize) - 1).clamp(0, (in_h - 1) as isize) as usize;
                            let row = &plane[sy * in_w..(sy + 1) * in_w];
                            let mut row_acc = 0.0f32;
                            for (i, &wx_i) in wx.iter().enumerate() {
                                let sx = (ix + (i as isize) - 1).clamp(0, (in_w - 1) as isize) as usize;
                                row_acc += row[sx] * wx_i;
                            }
                            acc += row_acc * wy_j;
                        }
                        dst_row[ow] = acc;
                    }
                }
            }
            _ => {
                let yi: Vec<usize> = y_coords.iter().map(|&fy| round_nearest(fy, nearest_mode, in_h)).collect();
                let xi: Vec<usize> = x_coords.iter().map(|&fx| round_nearest(fx, nearest_mode, in_w)).collect();
                for (oh, &sy) in yi.iter().enumerate() {
                    let row = &plane[sy * in_w..(sy + 1) * in_w];
                    let dst_row = &mut dst[oh * out_w..(oh + 1) * out_w];
                    for (ow, &sx) in xi.iter().enumerate() {
                        dst_row[ow] = row[sx];
                    }
                }
            }
        }
    };

    #[cfg(not(target_arch = "wasm32"))]
    if batch * c > 1 && total >= 4096 {
        out.par_chunks_mut(out_plane).enumerate().for_each(|(bc, dst)| resize_plane(bc, dst));
    } else {
        for (bc, dst) in out.chunks_mut(out_plane).enumerate() {
            resize_plane(bc, dst);
        }
    }
    #[cfg(target_arch = "wasm32")]
    for (bc, dst) in out.chunks_mut(out_plane).enumerate() {
        resize_plane(bc, dst);
    }

    let res = Tensor::from_f32(out_shape, out)?;
    res.cast(x.dtype())
}

fn sample_padded_2d(plane: &[f32], h: usize, w: usize, iy: isize, ix: isize, padding_mode: &str) -> f32 {
    let map_idx = |mut idx: isize, len: usize| -> Option<usize> {
        let l = len as isize;
        match padding_mode {
            "border" => Some(idx.clamp(0, l - 1) as usize),
            "reflection" => {
                if l <= 1 {
                    return Some(0);
                }
                let period = 2 * (l - 1);
                idx = idx.rem_euclid(period);
                if idx >= l {
                    idx = period - idx;
                }
                Some(idx as usize)
            }
            _ => {
                if idx >= 0 && idx < l {
                    Some(idx as usize)
                } else {
                    None
                }
            }
        }
    };
    let Some(sy) = map_idx(iy, h) else { return 0.0 };
    let Some(sx) = map_idx(ix, w) else { return 0.0 };
    plane[sy * w + sx]
}

fn run_grid_sample(node: &Node, x: &Tensor, grid: &Tensor) -> Result<Tensor, MlError> {
    if x.rank() != 4 || grid.rank() != 4 || grid.shape()[3] != 2 || x.shape()[0] != grid.shape()[0] {
        return Err(MlError::InvalidShape(format!("GridSample expects X=[N,C,H,W] and grid=[N,Ho,Wo,2], got {:?} and {:?}", x.shape(), grid.shape())));
    }
    let (n, c, ih, iw) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
    let (oh, ow) = (grid.shape()[1], grid.shape()[2]);
    let out_shape = vec![n, c, oh, ow];
    let total = checked_numel(&out_shape, DType::F32)?;
    if total == 0 || ih == 0 || iw == 0 {
        return Tensor::zeros(out_shape, DType::F32);
    }

    let align_corners = node.attr_int("align_corners").unwrap_or(0) != 0;
    let mode = node.attr_str("mode").unwrap_or("bilinear");
    let padding_mode = node.attr_str("padding_mode").unwrap_or("zeros");
    let unnorm = |g: f32, len: usize| -> f32 { if align_corners { ((g + 1.0) * 0.5) * ((len - 1) as f32) } else { ((g + 1.0) * (len as f32) - 1.0) * 0.5 } };

    let x_src = x.to_f32_cow();
    let g_src = grid.to_f32_cow();
    let mut out = vec![0.0f32; total];

    for b in 0..n {
        for y in 0..oh {
            for col in 0..ow {
                let g_base = ((b * oh + y) * ow + col) * 2;
                let fx = unnorm(g_src[g_base], iw);
                let fy = unnorm(g_src[g_base + 1], ih);
                for ch in 0..c {
                    let plane = &x_src[(b * c + ch) * ih * iw..(b * c + ch + 1) * ih * iw];
                    let val = if mode == "nearest" {
                        sample_padded_2d(plane, ih, iw, fy.round_ties_even() as isize, fx.round_ties_even() as isize, padding_mode)
                    } else {
                        let x0 = fx.floor() as isize;
                        let y0 = fy.floor() as isize;
                        let wx = fx - (x0 as f32);
                        let wy = fy - (y0 as f32);
                        let v00 = sample_padded_2d(plane, ih, iw, y0, x0, padding_mode);
                        let v01 = sample_padded_2d(plane, ih, iw, y0, x0 + 1, padding_mode);
                        let v10 = sample_padded_2d(plane, ih, iw, y0 + 1, x0, padding_mode);
                        let v11 = sample_padded_2d(plane, ih, iw, y0 + 1, x0 + 1, padding_mode);
                        (v00 * (1.0 - wx) + v01 * wx) * (1.0 - wy) + (v10 * (1.0 - wx) + v11 * wx) * wy
                    };
                    out[((b * c + ch) * oh + y) * ow + col] = val;
                }
            }
        }
    }
    Tensor::from_f32(out_shape, out)
}

/// Try executing a spatial vision ONNX node (`Conv`, `ConvTranspose`, `Pool`, `Resize`, `GridSample`).
pub fn try_run(node: &Node, inputs: &[Option<&Tensor>], accel: &dyn ComputeAccelerator) -> Result<Option<Vec<Tensor>>, MlError> {
    let op = node.op_type.as_str();
    let single = |t: Tensor| Ok(Some(vec![t]));
    match op {
        "Conv" => single(run_conv(node, inputs, accel)?),
        "ConvTranspose" => single(run_conv_transpose(node, inputs, accel)?),
        "MaxPool" => single(run_pool(node, req(inputs, 0, op)?, true)?),
        "AveragePool" => single(run_pool(node, req(inputs, 0, op)?, false)?),
        "GlobalAveragePool" => single(run_global_pool(req(inputs, 0, op)?, false)?),
        "GlobalMaxPool" => single(run_global_pool(req(inputs, 0, op)?, true)?),
        "Resize" | "Upsample" => single(run_resize(node, inputs)?),
        "GridSample" => single(run_grid_sample(node, req(inputs, 0, op)?, req(inputs, 1, op)?)?),
        _ => Ok(None),
    }
}
