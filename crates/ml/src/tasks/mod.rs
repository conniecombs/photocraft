//! High-level vision task pipelines (subject/sky alpha matting and SAM-2 interactive object
//! segmentation) operating on PhotoCraft [`Sampler`](photocraft_algo::segment::Sampler)s,
//! [`Region`](photocraft_algo::selection::Region)s and tiled [`Surface`](photocraft_raster::Surface)s.

pub mod matting;
pub mod sam2;

use photocraft_algo::segment::RgbImage;
use photocraft_algo::selection::Region;
use photocraft_color::{PixelFormat, SampleType, write_sample};
use photocraft_geom::{Rect, TILE_SIZE};
use photocraft_raster::Surface;

/// Bilinearly resize an [`RgbImage`] into a normalized planar NCHW `[1, 3, dst_h, dst_w]` `f32`
/// buffer using `mean` and `std`.
pub(crate) fn resize_normalize_nchw(src: &RgbImage, dst_w: usize, dst_h: usize, mean: [f32; 3], std: [f32; 3]) -> Vec<f32> {
    let plane = dst_w * dst_h;
    let mut out = vec![0.0f32; 3 * plane];
    if src.w == 0 || src.h == 0 || plane == 0 {
        return out;
    }
    let inv_std = [1.0 / std[0].max(1e-6), 1.0 / std[1].max(1e-6), 1.0 / std[2].max(1e-6)];
    let scale_y = (src.h as f32) / (dst_h as f32);
    let scale_x = (src.w as f32) / (dst_w as f32);

    for dy in 0..dst_h {
        let fy = ((dy as f32 + 0.5) * scale_y - 0.5).clamp(0.0, (src.h - 1) as f32);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(src.h - 1);
        let wy = fy - (y0 as f32);
        for dx in 0..dst_w {
            let fx = ((dx as f32 + 0.5) * scale_x - 0.5).clamp(0.0, (src.w - 1) as f32);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(src.w - 1);
            let wx = fx - (x0 as f32);

            let p00 = src.px[y0 * src.w + x0];
            let p01 = src.px[y0 * src.w + x1];
            let p10 = src.px[y1 * src.w + x0];
            let p11 = src.px[y1 * src.w + x1];
            let idx = dy * dst_w + dx;

            for c in 0..3 {
                let top = p00[c] * (1.0 - wx) + p01[c] * wx;
                let bot = p10[c] * (1.0 - wx) + p11[c] * wx;
                let val = top * (1.0 - wy) + bot * wy;
                out[c * plane + idx] = (val - mean[c]) * inv_std[c];
            }
        }
    }
    out
}

#[inline]
pub(crate) fn sample_bilinear_grid(grid: &[f32], gw: usize, gh: usize, fx: f32, fy: f32) -> f32 {
    let xc = fx.clamp(0.0, (gw.saturating_sub(1)) as f32);
    let yc = fy.clamp(0.0, (gh.saturating_sub(1)) as f32);
    let x0 = xc.floor() as usize;
    let y0 = yc.floor() as usize;
    let x1 = (x0 + 1).min(gw - 1);
    let y1 = (y0 + 1).min(gh - 1);
    let wx = xc - (x0 as f32);
    let wy = yc - (y0 as f32);
    let top = grid[y0 * gw + x0] * (1.0 - wx) + grid[y0 * gw + x1] * wx;
    let bot = grid[y1 * gw + x0] * (1.0 - wx) + grid[y1 * gw + x1] * wx;
    top * (1.0 - wy) + bot * wy
}

/// Upsample a coarse `[gh, gw]` probability matte (`0.0..=1.0`) over `area` into a tight-cropped
/// 8-bit [`Region`], returning `None` if no pixel rounds to non-zero coverage.
pub fn alpha_grid_to_region(alpha: &[f32], gw: usize, gh: usize, area: Rect) -> Option<Region> {
    let (aw, ah) = (area.width() as usize, area.height() as usize);
    if aw == 0 || ah == 0 || gw == 0 || gh == 0 || alpha.len() < gw * gh {
        return None;
    }
    let sx = (gw as f32) / (aw as f32);
    let sy = (gh as f32) / (ah as f32);

    // First find tight bounding box on the coarse grid (expanded by 2 coarse texels for bilinear
    // support) so empty borders around a small subject are never scanned at full resolution.
    let (mut gx0, mut gy0, mut gx1, mut gy1) = (gw, gh, 0usize, 0usize);
    for gy in 0..gh {
        for gx in 0..gw {
            if alpha[gy * gw + gx] >= (0.5 / 255.0) {
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
            let v = sample_bilinear_grid(alpha, gw, gh, fx, fy).clamp(0.0, 1.0);
            let u = (v * 255.0 + 0.5) as u8;
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

/// Upsample a coarse `[gh, gw]` float matte (`0.0..=1.0`) directly into a tiled single-channel
/// mask [`Surface`] at `depth` (`U8`, `U16` or `F32`), skipping empty tiles and sharing a single
/// copy-on-write solid tile across fully opaque interior tiles.
pub fn alpha_grid_to_surface(alpha: &[f32], gw: usize, gh: usize, area: Rect, depth: SampleType) -> Option<Surface> {
    let (aw, ah) = (area.width() as usize, area.height() as usize);
    if aw == 0 || ah == 0 || gw == 0 || gh == 0 || alpha.len() < gw * gh {
        return None;
    }
    let fmt = PixelFormat::GRAY8.with_sample(depth);
    let mut surface = Surface::new(fmt);
    let solid_one = surface.solid_tile(&[1.0]);
    let sx = (gw as f32) / (aw as f32);
    let sy = (gh as f32) / (ah as f32);
    let eps = match depth {
        SampleType::U8 => 0.5 / 255.0,
        SampleType::U16 => 0.5 / 65535.0,
        SampleType::F32 => 1e-6,
    };

    let mut any_non_zero = false;
    for tc in area.tiles() {
        let tr = tc.rect().intersect(&area);
        if tr.is_empty() {
            continue;
        }
        let gx0 = ((((tr.x0 - area.x0) as f32) * sx).floor() as isize - 2).clamp(0, (gw - 1) as isize) as usize;
        let gy0 = ((((tr.y0 - area.y0) as f32) * sy).floor() as isize - 2).clamp(0, (gh - 1) as isize) as usize;
        let gx1 = (((((tr.x1 - area.x0) as f32) * sx).ceil() as usize) + 2).min(gw);
        let gy1 = (((((tr.y1 - area.y0) as f32) * sy).ceil() as usize) + 2).min(gh);

        let mut min_v = 1.0f32;
        let mut max_v = 0.0f32;
        for gy in gy0..gy1 {
            for &v in &alpha[gy * gw + gx0..gy * gw + gx1] {
                let c = v.clamp(0.0, 1.0);
                min_v = min_v.min(c);
                max_v = max_v.max(c);
            }
        }
        if max_v < eps {
            continue;
        }
        if min_v > 1.0 - eps && tr == tc.rect() {
            surface.put_tiles([(tc, solid_one.clone())]);
            any_non_zero = true;
            continue;
        }

        let mut tile_has_fg = false;
        {
            let mut bytes = surface.tile_mut(tc).bytes_mut();
            for y in tr.y0..tr.y1 {
                let fy = ((y - area.y0) as f32 + 0.5) * sy - 0.5;
                let ty_off = ((y - tc.ty * TILE_SIZE) as usize) * (TILE_SIZE as usize);
                for x in tr.x0..tr.x1 {
                    let fx = ((x - area.x0) as f32 + 0.5) * sx - 0.5;
                    let v = sample_bilinear_grid(alpha, gw, gh, fx, fy).clamp(0.0, 1.0);
                    if v >= eps {
                        tile_has_fg = true;
                        let tx_off = (x - tc.tx * TILE_SIZE) as usize;
                        write_sample(&mut bytes, depth, ty_off + tx_off, v);
                    }
                }
            }
        }
        if tile_has_fg {
            any_non_zero = true;
        }
    }

    if !any_non_zero {
        return None;
    }
    surface.prune();
    Some(surface)
}
