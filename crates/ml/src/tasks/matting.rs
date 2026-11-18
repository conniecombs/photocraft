//! Dense foreground/sky alpha matting pipeline (`MODNet`, `BiRefNet-lite` and custom single-graph
//! `.onnx` segmentation models).

use std::collections::HashMap;

use photocraft_algo::segment::Sampler;
use photocraft_algo::selection::Region;
use photocraft_color::SampleType;
use photocraft_geom::Rect;
use photocraft_raster::{Interrupt, Surface};

use crate::MlError;
use crate::backend::ComputeAccelerator;
use crate::catalog::MattingPreprocess;
use crate::onnx::{Dim, Model};
use crate::ops::execute_graph;
use crate::store::LoadedModel;
use crate::tasks::{alpha_grid_to_region, alpha_grid_to_surface, resize_normalize_nchw};
use crate::tensor::Tensor;

fn resolve_input_geometry(model: &Model, preprocess: MattingPreprocess, area_w: usize, area_h: usize) -> (usize, usize, [f32; 3], [f32; 3]) {
    let (mean, std, default_w, default_h) = match preprocess {
        MattingPreprocess::ShortestEdgeDivisible { shortest_edge, max_edge, divisibility, mean, std } => {
            let div = (divisibility.max(1)) as usize;
            let short = area_w.min(area_h).max(1) as f32;
            let mut scale = (shortest_edge.max(1) as f32) / short;
            let long = area_w.max(area_h) as f32;
            if long * scale > (max_edge.max(shortest_edge) as f32) {
                scale = (max_edge.max(shortest_edge) as f32) / long;
            }
            let round_div = |v: f32| -> usize { (((v / (div as f32)).round() as usize).max(1)) * div };
            (mean, std, round_div((area_w as f32) * scale), round_div((area_h as f32) * scale))
        }
        MattingPreprocess::FixedSquare { side, mean, std } => {
            let s = (side.max(1)) as usize;
            (mean, std, s, s)
        }
    };

    if let Some(inp) = model.graph.inputs.first()
        && inp.dims.len() == 4
        && let (Dim::Fixed(fh), Dim::Fixed(fw)) = (&inp.dims[2], &inp.dims[3])
        && *fh > 0
        && *fw > 0
    {
        return (*fw, *fh, mean, std);
    }
    (default_w, default_h, mean, std)
}

/// Walk backwards through shape/resize pass-through ops to determine whether `out_name` already
/// comes from a `Sigmoid`/`HardSigmoid`/`Softmax`/`Clip` probability head (vs raw logits).
fn output_is_probability(model: &Model, out_name: &str) -> bool {
    let mut cur = out_name;
    for _ in 0..16 {
        let Some(producer) = model.graph.nodes.iter().rev().find(|n| n.outputs.iter().any(|o| o == cur)) else {
            return false;
        };
        match producer.op_type.as_str() {
            "Sigmoid" | "HardSigmoid" | "Softmax" | "Clip" => return true,
            "Resize" | "Upsample" | "Identity" | "Reshape" | "Squeeze" | "Unsqueeze" | "Transpose" | "Cast" => {
                let Some(next) = producer.inputs.first().filter(|s| !s.is_empty()) else {
                    return false;
                };
                cur = next.as_str();
            }
            _ => return false,
        }
    }
    false
}

/// Run a dense matting/segmentation model over `area`, returning `(matte, matte_w, matte_h)` with
/// probabilities in `0.0..=1.0`.
pub fn predict_matte_grid(
    model: &Model,
    preprocess: MattingPreprocess,
    sampler: &dyn Sampler,
    area: Rect,
    accel: &dyn ComputeAccelerator,
    ctl: &Interrupt<'_>,
) -> Result<(Vec<f32>, usize, usize), MlError> {
    let (aw, ah) = (area.width() as usize, area.height() as usize);
    if aw == 0 || ah == 0 {
        return Ok((Vec::new(), 0, 0));
    }
    if ctl.cancelled() {
        return Err(MlError::Cancelled);
    }

    let (net_w, net_h, mean, std) = resolve_input_geometry(model, preprocess, aw, ah);
    let step = (aw.min(ah) / (net_w.max(net_h).saturating_mul(2)).max(1)).max(1);
    let coarse = sampler.rgb_scaled(area, step);
    if ctl.cancelled() {
        return Err(MlError::Cancelled);
    }

    let nchw = resize_normalize_nchw(&coarse, net_w, net_h, mean, std);
    let input_tensor = Tensor::from_f32(vec![1, 3, net_h, net_w], nchw)?;
    let input_name = model.graph.inputs.first().map_or_else(|| "input".to_string(), |v| v.name.clone());

    let mut runtime_inputs = HashMap::with_capacity(1);
    runtime_inputs.insert(input_name, input_tensor);

    let mut outputs = execute_graph(model, &runtime_inputs, accel, ctl)?;
    let out_info = model.graph.outputs.last().ok_or_else(|| MlError::InvalidModel("matting model has no graph outputs".into()))?;
    let out_tensor = outputs.remove(&out_info.name).ok_or_else(|| MlError::InvalidModel(format!("missing output `{}`", out_info.name)))?;

    let (mh, mw) = match out_tensor.shape() {
        [1, 1, h, w] => (*h, *w),
        [1, h, w] => (*h, *w),
        [h, w] => (*h, *w),
        [1, c, h, w] if *c >= 2 => {
            // Multi-class segmentation output (e.g. [1, 2, H, W] background vs foreground/sky):
            // convert foreground class 1 via two-class softmax/channel slice.
            let raw = out_tensor.to_f32_cow();
            let plane = (*h) * (*w);
            let mut fg = vec![0.0f32; plane];
            for i in 0..plane {
                let bg_v = raw[i];
                let fg_v = raw[plane + i];
                let m = bg_v.max(fg_v);
                let e0 = (bg_v - m).exp();
                let e1 = (fg_v - m).exp();
                fg[i] = (e1 / (e0 + e1)).clamp(0.0, 1.0);
            }
            return Ok((fg, *w, *h));
        }
        other => return Err(MlError::InvalidShape(format!("unexpected matting output shape {other:?}"))),
    };

    let is_prob = output_is_probability(model, &out_info.name);
    let mut probs = out_tensor.into_f32()?;
    for v in &mut probs {
        if !v.is_finite() {
            *v = 0.0;
        } else if is_prob {
            *v = v.clamp(0.0, 1.0);
        } else {
            *v = (1.0 / (1.0 + (-*v).exp())).clamp(0.0, 1.0);
        }
    }
    Ok((probs, mw, mh))
}

/// Run `loaded` over `area` and upsample the predicted alpha matte into an 8-bit [`Region`].
pub fn predict_region(
    loaded: &LoadedModel,
    sampler: &dyn Sampler,
    area: Rect,
    accel: &dyn ComputeAccelerator,
    ctl: &Interrupt<'_>,
) -> Result<Option<Region>, MlError> {
    let LoadedModel::Single { preprocess, model, .. } = loaded else {
        return Err(MlError::InvalidModel("expected single-graph matting model".into()));
    };
    let (grid, gw, gh) = predict_matte_grid(model, *preprocess, sampler, area, accel, ctl)?;
    Ok(alpha_grid_to_region(&grid, gw, gh, area))
}

/// Run `loaded` over `area` and upsample the continuous float `[0.0..=1.0]` matte directly into a
/// tiled mask [`Surface`] at `depth` (`U8`, `U16` or `F32`), preserving full 16-bit and 32-bit
/// float alpha transitions.
pub fn predict_surface(
    loaded: &LoadedModel,
    sampler: &dyn Sampler,
    area: Rect,
    depth: SampleType,
    accel: &dyn ComputeAccelerator,
    ctl: &Interrupt<'_>,
) -> Result<Option<Surface>, MlError> {
    let LoadedModel::Single { preprocess, model, .. } = loaded else {
        return Err(MlError::InvalidModel("expected single-graph matting model".into()));
    };
    let (grid, gw, gh) = predict_matte_grid(model, *preprocess, sampler, area, accel, ctl)?;
    Ok(alpha_grid_to_surface(&grid, gw, gh, area, depth))
}
