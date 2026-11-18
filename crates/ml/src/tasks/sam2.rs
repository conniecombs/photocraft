//! Promptable interactive object segmentation (`SAM 2.1 Hiera Tiny`) with thread-safe image
//! embedding caching (`EmbeddingCache`) for instant hover, click and box queries.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use photocraft_algo::segment::Sampler;
use photocraft_algo::selection::Region;
use photocraft_geom::Rect;
use photocraft_raster::Interrupt;
use sha2::{Digest, Sha256};

use crate::MlError;
use crate::backend::ComputeAccelerator;
use crate::onnx::{Dim, Model};
use crate::ops::execute_graph;
use crate::store::LoadedModel;
use crate::tasks::{alpha_grid_to_region, resize_normalize_nchw, sample_bilinear_grid};
use crate::tensor::Tensor;

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Interactive user prompt in document pixel coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sam2Prompt {
    /// Single foreground click or Object Finder hover at `(x, y)`.
    Point { x: i32, y: i32 },
    /// Bounding-box drag marquee.
    Box(Rect),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CacheKey {
    model_id: String,
    area: [i32; 4],
    digest: [u8; 32],
}

/// Cached vision-encoder feature pyramid for one `(model, area, image)` tuple.
#[derive(Clone, Debug)]
pub struct CachedEmbedding {
    pub enc_w: usize,
    pub enc_h: usize,
    pub features: HashMap<String, Tensor>,
}

struct CacheInner {
    capacity: usize,
    entries: VecDeque<(CacheKey, Arc<CachedEmbedding>)>,
}

/// Bounded thread-safe LRU cache of SAM-2 vision-encoder embeddings.
#[derive(Clone)]
pub struct EmbeddingCache {
    inner: Arc<Mutex<CacheInner>>,
    hits: Arc<AtomicU64>,
    misses: Arc<AtomicU64>,
}

impl Default for EmbeddingCache {
    fn default() -> Self {
        Self::new(4)
    }
}

impl std::fmt::Debug for EmbeddingCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingCache").field("hits", &self.hits()).field("misses", &self.misses()).finish()
    }
}

impl EmbeddingCache {
    /// Create a cache retaining up to `capacity` image embeddings (`>= 1`).
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(CacheInner { capacity: capacity.max(1), entries: VecDeque::new() })),
            hits: Arc::new(AtomicU64::new(0)),
            misses: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    pub fn clear(&self) {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner).entries.clear();
    }

    fn get(&self, key: &CacheKey) -> Option<Arc<CachedEmbedding>> {
        let mut guard = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        let pos = guard.entries.iter().position(|(k, _)| k == key)?;
        let pair = guard.entries.remove(pos)?;
        let val = pair.1.clone();
        guard.entries.push_back(pair);
        self.hits.fetch_add(1, Ordering::Relaxed);
        Some(val)
    }

    fn insert(&self, key: CacheKey, val: Arc<CachedEmbedding>) {
        let mut guard = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(pos) = guard.entries.iter().position(|(k, _)| k == &key) {
            guard.entries.remove(pos);
        }
        while guard.entries.len() >= guard.capacity {
            guard.entries.pop_front();
        }
        guard.entries.push_back((key, val));
    }
}

fn encoder_input_size(encoder: &Model) -> (usize, usize) {
    if let Some(inp) = encoder.graph.inputs.first()
        && inp.dims.len() == 4
        && let (Dim::Fixed(fh), Dim::Fixed(fw)) = (&inp.dims[2], &inp.dims[3])
        && *fh > 0
        && *fw > 0
    {
        return (*fw, *fh);
    }
    (1024, 1024)
}

fn hash_f32_slice(data: &[f32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytemuck::cast_slice(data));
    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

fn compute_or_get_embedding(
    model_id: &str,
    encoder: &Model,
    sampler: &dyn Sampler,
    area: Rect,
    cache: &EmbeddingCache,
    accel: &dyn ComputeAccelerator,
    ctl: &Interrupt<'_>,
) -> Result<Arc<CachedEmbedding>, MlError> {
    let (aw, ah) = (area.width() as usize, area.height() as usize);
    let (enc_w, enc_h) = encoder_input_size(encoder);
    let step = (aw.min(ah) / (enc_w.max(enc_h).saturating_mul(2)).max(1)).max(1);
    let coarse = sampler.rgb_scaled(area, step);
    if ctl.cancelled() {
        return Err(MlError::Cancelled);
    }

    let nchw = resize_normalize_nchw(&coarse, enc_w, enc_h, IMAGENET_MEAN, IMAGENET_STD);
    let digest = hash_f32_slice(&nchw);
    let key = CacheKey { model_id: model_id.to_string(), area: [area.x0, area.y0, area.x1, area.y1], digest };

    if let Some(hit) = cache.get(&key) {
        return Ok(hit);
    }
    cache.misses.fetch_add(1, Ordering::Relaxed);

    let in_name = encoder.graph.inputs.first().map_or_else(|| "pixel_values".to_string(), |v| v.name.clone());
    let mut inputs = HashMap::with_capacity(1);
    inputs.insert(in_name, Tensor::from_f32(vec![1, 3, enc_h, enc_w], nchw)?);

    let features = execute_graph(encoder, &inputs, accel, ctl)?;
    let emb = Arc::new(CachedEmbedding { enc_w, enc_h, features });
    cache.insert(key, emb.clone());
    Ok(emb)
}

fn build_decoder_inputs(decoder: &Model, emb: &CachedEmbedding, area: Rect, prompt: Sam2Prompt) -> Result<Option<HashMap<String, Tensor>>, MlError> {
    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    for vi in &decoder.graph.inputs {
        if let Some(feat) = emb.features.get(&vi.name) {
            inputs.insert(vi.name.clone(), feat.clone());
        }
    }

    let aw = (area.width().max(1)) as f32;
    let ah = (area.height().max(1)) as f32;
    let sx = (emb.enc_w as f32) / aw;
    let sy = (emb.enc_h as f32) / ah;

    let has_boxes_in = decoder.graph.inputs.iter().any(|v| v.name == "input_boxes");
    let points_vi = decoder.graph.inputs.iter().find(|v| v.name == "input_points" || v.name == "point_coords");
    let labels_vi = decoder.graph.inputs.iter().find(|v| v.name == "input_labels" || v.name == "point_labels");
    let points_rank = points_vi.map_or(4, |v| if v.dims.is_empty() { 4 } else { v.dims.len() });

    let (pts_flat, lbls_flat, box_opt): (Vec<f32>, Vec<i64>, Option<[f32; 4]>) = match prompt {
        Sam2Prompt::Point { x, y } => {
            if !area.contains(x, y) {
                return Ok(None);
            }
            let px = ((x - area.x0) as f32 + 0.5) * sx;
            let py = ((y - area.y0) as f32 + 0.5) * sy;
            (vec![px, py], vec![1], None)
        }
        Sam2Prompt::Box(r) => {
            let clipped = r.intersect(&area);
            if clipped.is_empty() {
                return Ok(None);
            }
            let bx0 = ((clipped.x0 - area.x0) as f32) * sx;
            let by0 = ((clipped.y0 - area.y0) as f32) * sy;
            let bx1 = ((clipped.x1 - area.x0) as f32) * sx;
            let by1 = ((clipped.y1 - area.y0) as f32) * sy;
            if has_boxes_in {
                // Pass SAM-2's official padding point (label = -1) alongside the 1-box tensor.
                (vec![0.0, 0.0], vec![-1], Some([bx0, by0, bx1, by1]))
            } else {
                // Box encoded directly as top-left (2) + bottom-right (3) corner points.
                (vec![bx0, by0, bx1, by1], vec![2, 3], None)
            }
        }
    };

    let num_pts = lbls_flat.len();
    if let Some(pvi) = points_vi {
        let p_shape = if points_rank == 3 { vec![1, num_pts, 2] } else { vec![1, 1, num_pts, 2] };
        inputs.insert(pvi.name.clone(), Tensor::from_f32(p_shape, pts_flat)?);
    }
    if let Some(lvi) = labels_vi {
        let l_shape = if points_rank == 3 { vec![1, num_pts] } else { vec![1, 1, num_pts] };
        inputs.insert(lvi.name.clone(), Tensor::from_i64(l_shape, lbls_flat)?);
    }
    if has_boxes_in {
        let b_tensor = match box_opt {
            Some(b) => Tensor::from_f32(vec![1, 1, 4], b.to_vec())?,
            None => Tensor::from_f32(vec![1, 0, 4], Vec::new())?,
        };
        inputs.insert("input_boxes".to_string(), b_tensor);
    }

    // Optional legacy SAM-1 / auxiliary mask inputs if declared by a custom export.
    for vi in &decoder.graph.inputs {
        if inputs.contains_key(&vi.name) || decoder.graph.initializers.contains_key(&vi.name) {
            continue;
        }
        match vi.name.as_str() {
            "has_mask_input" => {
                inputs.insert(vi.name.clone(), Tensor::from_f32(vec![1], vec![0.0])?);
            }
            "mask_input" => {
                inputs.insert(vi.name.clone(), Tensor::zeros(vec![1, 1, 256, 256], crate::tensor::DType::F32)?);
            }
            "orig_im_size" => {
                inputs.insert(vi.name.clone(), Tensor::from_i64(vec![2], vec![i64::from(area.height()), i64::from(area.width())])?);
            }
            _ => {}
        }
    }

    Ok(Some(inputs))
}

/// Upsample a coarse `[gh, gw]` signed-logit field (`0.0` = contour boundary) into an anti-aliased
/// tight-cropped [`Region`] over `area`.
fn logits_grid_to_region(logits: &[f32], gw: usize, gh: usize, area: Rect) -> Option<Region> {
    let (aw, ah) = (area.width() as usize, area.height() as usize);
    if aw == 0 || ah == 0 || gw == 0 || gh == 0 || logits.len() < gw * gh {
        return None;
    }
    let sx = (gw as f32) / (aw as f32);
    let sy = (gh as f32) / (ah as f32);

    let (mut gx0, mut gy0, mut gx1, mut gy1) = (gw, gh, 0usize, 0usize);
    for gy in 0..gh {
        for gx in 0..gw {
            if logits[gy * gw + gx] > -4.0 {
                gx0 = gx0.min(gx);
                gy0 = gy0.min(gy);
                gx1 = gx1.max(gx + 1);
                gy1 = gy1.max(gy + 1);
            }
        }
    }
    if gx0 >= gx1 || gy0 >= gy1 {
        return None;
    }

    let px0 = (((gx0.saturating_sub(2)) as f32 / sx).floor() as usize).min(aw);
    let py0 = (((gy0.saturating_sub(2)) as f32 / sy).floor() as usize).min(ah);
    let px1 = ((((gx1 + 2).min(gw)) as f32 / sx).ceil() as usize).min(aw);
    let py1 = ((((gy1 + 2).min(gh)) as f32 / sy).ceil() as usize).min(ah);
    if px0 >= px1 || py0 >= py1 {
        return None;
    }

    let cw = px1 - px0;
    let ch = py1 - py0;
    let mut raw = vec![0u8; cw * ch];
    let (mut tx0, mut ty0, mut tx1, mut ty1) = (cw, ch, 0usize, 0usize);

    for ly in 0..ch {
        let fy = ((py0 + ly) as f32 + 0.5) * sy - 0.5;
        let row = &mut raw[ly * cw..(ly + 1) * cw];
        for (lx, dst) in row.iter_mut().enumerate() {
            let fx = ((px0 + lx) as f32 + 0.5) * sx - 0.5;
            let logit = sample_bilinear_grid(logits, gw, gh, fx, fy);
            let u = if !logit.is_finite() || logit <= -3.5 {
                0u8
            } else if logit >= 3.5 {
                255u8
            } else {
                let prob = 1.0 / (1.0 + (-logit * 2.0).exp());
                (prob * 255.0 + 0.5) as u8
            };
            if u > 0 {
                *dst = u;
                tx0 = tx0.min(lx);
                ty0 = ty0.min(ly);
                tx1 = tx1.max(lx + 1);
                ty1 = ty1.max(ly + 1);
            }
        }
    }
    if tx0 >= tx1 || ty0 >= ty1 {
        return None;
    }

    let rw = tx1 - tx0;
    let rh = ty1 - ty0;
    let mut tight = Vec::with_capacity(rw * rh);
    for y in ty0..ty1 {
        tight.extend_from_slice(&raw[y * cw + tx0..y * cw + tx1]);
    }
    let bbox = Rect::new(area.x0 + (px0 + tx0) as i32, area.y0 + (py0 + ty0) as i32, area.x0 + (px0 + tx1) as i32, area.y0 + (py0 + ty1) as i32);
    Some(Region { bbox, mask: tight })
}

/// Context bundling the model, embedding cache and compute backend for interactive SAM-2 queries.
pub struct Sam2Context<'a> {
    pub loaded: &'a LoadedModel,
    pub cache: &'a EmbeddingCache,
    pub accel: &'a dyn ComputeAccelerator,
}

/// Run SAM-2 interactive segmentation for `prompt` (point or box) inside `area`.
pub fn segment_prompt(ctx: &Sam2Context<'_>, sampler: &dyn Sampler, area: Rect, prompt: Sam2Prompt, ctl: &Interrupt<'_>) -> Result<Option<Region>, MlError> {
    let LoadedModel::Sam2 { id, encoder, decoder } = ctx.loaded else {
        return Err(MlError::InvalidModel("expected split SAM-2 encoder+decoder model".into()));
    };
    if area.is_empty() {
        return Ok(None);
    }

    let emb = compute_or_get_embedding(id, encoder, sampler, area, ctx.cache, ctx.accel, ctl)?;
    let Some(dec_inputs) = build_decoder_inputs(decoder, &emb, area, prompt)? else {
        return Ok(None);
    };
    let outputs = execute_graph(decoder, &dec_inputs, ctx.accel, ctl)?;

    if let Some(obj_score) = outputs.get("object_score_logits")
        && let Some(&score) = obj_score.to_f32_cow().first()
        && score < -2.5
    {
        return Ok(None);
    }

    let masks_t = outputs
        .get("pred_masks")
        .or_else(|| outputs.get("masks"))
        .or_else(|| decoder.graph.outputs.iter().find_map(|o| outputs.get(&o.name).filter(|t| t.rank() >= 3)))
        .ok_or_else(|| MlError::InvalidModel("SAM-2 decoder produced no mask tensor".into()))?;

    let iou_t = outputs.get("iou_scores").or_else(|| outputs.get("iou_predictions"));
    let shape = masks_t.shape();
    if shape.len() < 3 {
        return Err(MlError::InvalidShape(format!("unexpected SAM-2 mask shape {shape:?}")));
    }
    let mh = shape[shape.len() - 2];
    let mw = shape[shape.len() - 1];
    let num_masks: usize = shape[..shape.len() - 2].iter().product();
    if num_masks == 0 || mh == 0 || mw == 0 {
        return Ok(None);
    }

    let best_idx = if let Some(iou) = iou_t {
        let scores = iou.to_f32_cow();
        scores.iter().take(num_masks).enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(idx, _)| idx)
    } else {
        0
    };

    let mask_plane = mh * mw;
    let all_masks = masks_t.to_f32_cow();
    let chosen = &all_masks[best_idx * mask_plane..(best_idx + 1) * mask_plane];

    let ends_with_sigmoid = decoder.graph.nodes.last().is_some_and(|n| n.op_type == "Sigmoid");
    if ends_with_sigmoid { Ok(alpha_grid_to_region(chosen, mw, mh, area)) } else { Ok(logits_grid_to_region(chosen, mw, mh, area)) }
}
