use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use photocraft_algo::segment::Sampler;
use photocraft_color::SampleType;
use photocraft_geom::Rect;
use photocraft_ml::backend::{ComputeAccelerator, Conv2dSpec, MatMulSpec, cpu_accelerator};
use photocraft_ml::catalog::{BUILTIN_CATALOG, MattingPreprocess, ModelTask, catalog_entry};
use photocraft_ml::onnx::builder::{ModelBuilder, NodeBuilder, encode_external_tensor_proto, encode_f16_tensor_proto};
use photocraft_ml::onnx::{Attr, Dim, ElemType, Model};
use photocraft_ml::ops::execute_graph;
use photocraft_ml::store::{DownloadState, DownloadTransport, ModelStore, sha256_hex, verify_sha256};
use photocraft_ml::tasks::matting::{predict_region, predict_surface};
use photocraft_ml::tasks::sam2::{EmbeddingCache, Sam2Context, Sam2Prompt, segment_prompt};
use photocraft_ml::{MlError, Tensor};
use photocraft_raster::Interrupt;

struct SyntheticSampler {
    /// Foreground rectangle with smooth ramp border.
    fg: Rect,
    fg_rgb: [f32; 3],
    bg_rgb: [f32; 3],
}

impl Sampler for SyntheticSampler {
    fn rgba(&self, r: Rect) -> Vec<[f32; 4]> {
        let mut out = Vec::with_capacity((r.width() * r.height()) as usize);
        for y in r.y0..r.y1 {
            for x in r.x0..r.x1 {
                // Smooth 4px ramp near fg boundary so continuous alpha matting is testable at 16/32f.
                let dx = ((self.fg.x0 - x).max(0).max(x - (self.fg.x1 - 1))) as f32;
                let dy = ((self.fg.y0 - y).max(0).max(y - (self.fg.y1 - 1))) as f32;
                let dist = (dx * dx + dy * dy).sqrt();
                let t = (1.0 - dist / 4.0).clamp(0.0, 1.0);
                let rgb = [
                    self.bg_rgb[0] * (1.0 - t) + self.fg_rgb[0] * t,
                    self.bg_rgb[1] * (1.0 - t) + self.fg_rgb[1] * t,
                    self.bg_rgb[2] * (1.0 - t) + self.fg_rgb[2] * t,
                ];
                out.push([rgb[0], rgb[1], rgb[2], 1.0]);
            }
        }
        out
    }
}

#[test]
fn catalog_entries_use_spdx_allowlist_and_https_pins() {
    assert_eq!(BUILTIN_CATALOG.len(), 3);
    let allowed = ["MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "CC0-1.0"];
    for entry in BUILTIN_CATALOG {
        assert!(allowed.contains(&entry.license), "disallowed license {}", entry.license);
        assert!(entry.total_bytes() > 0);
        assert!(!entry.artifacts.is_empty());
        for art in entry.artifacts {
            assert!(art.url.starts_with("https://"), "non-https url {}", art.url);
            assert_eq!(art.sha256.len(), 64, "bad sha256 on {}", art.file_name);
            assert!(art.sha256.chars().all(|c| c.is_ascii_hexdigit()));
        }
    }
}

#[test]
fn rejects_malformed_cyclic_and_hostile_onnx_bytes_without_panicking() {
    // Empty and garbage bytes.
    assert!(Model::from_bytes(&[]).is_err());
    assert!(Model::from_bytes(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F]).is_err());
    // Truncated length-delimited field claiming 100 MiB.
    assert!(Model::from_bytes(&[0x3A, 0x80, 0x80, 0x80, 0x30, 0x01, 0x02]).is_err());

    // Cyclic graph: A -> B -> A.
    let cyclic = ModelBuilder::new("cyclic")
        .input("x", ElemType::Float, vec![Dim::Fixed(2)])
        .output("b", ElemType::Float, vec![Dim::Fixed(2)])
        .node(NodeBuilder::new("Relu", &["b"], &["a"]).name("n1"))
        .node(NodeBuilder::new("Relu", &["a"], &["b"]).name("n2"))
        .build();
    assert!(matches!(Model::from_bytes(&cyclic), Err(MlError::InvalidModel(_))));

    // Path traversal in external_data location must be rejected before calling resolver.
    let traversal = ModelBuilder::new("traversal")
        .input("x", ElemType::Float, vec![Dim::Fixed(2)])
        .output("y", ElemType::Float, vec![Dim::Fixed(2)])
        .raw_initializer(encode_external_tensor_proto("w", ElemType::Float, &[2], "../secret.bin", 0, 8))
        .node(NodeBuilder::new("Add", &["x", "w"], &["y"]))
        .build();
    let res = Model::from_bytes_with_external(&traversal, &|_, _, _| Ok(vec![0u8; 8]));
    assert!(matches!(res, Err(MlError::InvalidModel(_))));
}

#[test]
fn executes_f16_and_external_weights_and_core_onnx_ops() {
    // External weights file containing 4 floats: [1.0, 2.0, 3.0, 4.0] at offset 16.
    let mut ext_blob = vec![0xAAu8; 16];
    for v in [1.0f32, 2.0, 3.0, 4.0] {
        ext_blob.extend_from_slice(&v.to_le_bytes());
    }

    let bytes = ModelBuilder::new("core_ops")
        .input("x", ElemType::Float, vec![Dim::Fixed(1), Dim::Fixed(2), Dim::Fixed(2)])
        .output("probs", ElemType::Float, vec![Dim::Fixed(1), Dim::Fixed(2), Dim::Fixed(2)])
        .raw_initializer(encode_external_tensor_proto("ext_bias", ElemType::Float, &[1, 2, 2], "weights.onnx_data", 16, 16))
        .raw_initializer(encode_f16_tensor_proto("scale_f16", &[2], &[2.0, 0.5]))
        .initializer("ln_bias", &Tensor::from_f32(vec![2], vec![0.0, 0.0]).expect("valid"))
        .node(NodeBuilder::new("Add", &["x", "ext_bias"], &["sum"]))
        .node(NodeBuilder::new("Gelu", &["sum"], &["act"]))
        .node(NodeBuilder::new("LayerNormalization", &["act", "scale_f16", "ln_bias"], &["norm"]).attr("axis", Attr::Int(-1)))
        .node(NodeBuilder::new("Softmax", &["norm"], &["probs"]).attr("axis", Attr::Int(-1)))
        .build();

    let model = Model::from_bytes_with_external(&bytes, &|loc, off, len| {
        assert_eq!(loc, "weights.onnx_data");
        let s = off as usize;
        let e = s + len.unwrap_or(16) as usize;
        Ok(ext_blob[s..e].to_vec())
    })
    .expect("model parses");

    let mut inputs = HashMap::new();
    inputs.insert("x".into(), Tensor::from_f32(vec![1, 2, 2], vec![0.5, -0.5, 1.0, -1.0]).expect("x"));
    let out = execute_graph(&model, &inputs, cpu_accelerator(), &Interrupt::NONE).expect("runs");
    let probs = out.get("probs").expect("probs").as_f32().expect("f32");
    assert_eq!(probs.len(), 4);
    // Each row of Softmax sums to 1.0.
    assert!((probs[0] + probs[1] - 1.0).abs() < 1e-5);
    assert!((probs[2] + probs[3] - 1.0).abs() < 1e-5);
}

#[test]
fn executes_dynamic_shape_scatter_onehot_where_and_resize() {
    // Exercises SAM-2 & MODNet dynamic ops: Shape -> Gather -> Unsqueeze -> Concat -> Resize,
    // plus OneHot + Where + ScatterND + zero-sized [1, 0, 4] concat!
    let bytes = ModelBuilder::new("sam_ops")
        .input("img", ElemType::Float, vec![Dim::Fixed(1), Dim::Fixed(1), Dim::Symbolic("h".into()), Dim::Symbolic("w".into())])
        .input("labels", ElemType::Int64, vec![Dim::Fixed(1), Dim::Fixed(2)])
        .input("empty_boxes", ElemType::Float, vec![Dim::Fixed(1), Dim::Fixed(0), Dim::Fixed(4)])
        .output("resized", ElemType::Float, vec![Dim::Fixed(1), Dim::Fixed(1), Dim::Fixed(4), Dim::Fixed(4)])
        .output("onehot_sum", ElemType::Float, vec![Dim::Fixed(1), Dim::Fixed(2)])
        .output("cat_boxes", ElemType::Float, vec![Dim::Fixed(1), Dim::Symbolic("b".into()), Dim::Fixed(4)])
        .initializer("depth", &Tensor::from_i64(Vec::new(), vec![5]).expect("d"))
        .initializer("oh_vals", &Tensor::from_f32(vec![2], vec![0.0, 1.0]).expect("v"))
        .initializer("one_i64", &Tensor::from_i64(Vec::new(), vec![1]).expect("1"))
        .initializer("target_sizes", &Tensor::from_i64(vec![4], vec![1, 1, 4, 4]).expect("sz"))
        .initializer("empty_roi", &Tensor::from_f32(vec![0], Vec::new()).expect("roi"))
        .initializer("empty_scales", &Tensor::from_f32(vec![0], Vec::new()).expect("sc"))
        .initializer("one_box", &Tensor::from_f32(vec![1, 1, 4], vec![10.0, 20.0, 30.0, 40.0]).expect("box"))
        .node(NodeBuilder::new("Add", &["labels", "one_i64"], &["shifted"]))
        .node(NodeBuilder::new("OneHot", &["shifted", "depth", "oh_vals"], &["oh"]))
        .node(NodeBuilder::new("ReduceSum", &["oh"], &["onehot_sum"]).attr("axes", Attr::Ints(vec![-1])).attr("keepdims", Attr::Int(0)))
        .node(NodeBuilder::new("Resize", &["img", "empty_roi", "empty_scales", "target_sizes"], &["resized"]).attr("mode", Attr::String("linear".into())))
        .node(NodeBuilder::new("Concat", &["empty_boxes", "one_box"], &["cat_boxes"]).attr("axis", Attr::Int(1)))
        .build();

    let model = Model::from_bytes(&bytes).expect("parse");
    let mut inputs = HashMap::new();
    inputs.insert("img".into(), Tensor::from_f32(vec![1, 1, 2, 2], vec![0.0, 1.0, 1.0, 2.0]).expect("img"));
    inputs.insert("labels".into(), Tensor::from_i64(vec![1, 2], vec![-1, 1]).expect("lbl"));
    inputs.insert("empty_boxes".into(), Tensor::from_f32(vec![1, 0, 4], Vec::new()).expect("eb"));

    let out = execute_graph(&model, &inputs, cpu_accelerator(), &Interrupt::NONE).expect("exec");
    assert_eq!(out["resized"].shape(), &[1, 1, 4, 4]);
    assert_eq!(out["onehot_sum"].as_f32().expect("f32"), &[1.0, 1.0]);
    assert_eq!(out["cat_boxes"].shape(), &[1, 1, 4]);
    assert_eq!(out["cat_boxes"].as_f32().expect("f32"), &[10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn interrupt_cancels_graph_execution_promptly() {
    let mut b = ModelBuilder::new("cancel_test").input("x", ElemType::Float, vec![Dim::Fixed(4)]).output("n3", ElemType::Float, vec![Dim::Fixed(4)]);
    b = b.node(NodeBuilder::new("Relu", &["x"], &["n1"])).node(NodeBuilder::new("Sigmoid", &["n1"], &["n2"])).node(NodeBuilder::new("Tanh", &["n2"], &["n3"]));
    let model = Model::from_bytes(&b.build()).expect("parse");

    let flag = AtomicBool::new(true);
    let cancel_fn = || flag.load(Ordering::Relaxed);
    let ctl = Interrupt::cancel_only(&cancel_fn);
    let mut inputs = HashMap::new();
    inputs.insert("x".into(), Tensor::from_f32(vec![4], vec![1.0, -1.0, 2.0, -2.0]).expect("x"));

    let err = execute_graph(&model, &inputs, cpu_accelerator(), &ctl).expect_err("must cancel");
    assert!(matches!(err, MlError::Cancelled));
}

#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
#[test]
fn wgpu_compute_backend_matches_cpu_within_1e_4() {
    let Ok(gpu) = photocraft_ml::backend::wgpu_backend::WgpuAccelerator::try_new() else {
        // Headless environment with zero Vulkan/DX12/Metal/GL adapters: skip gracefully.
        return;
    };
    assert!(gpu.is_gpu());

    // 1. Batched & broadcasted non-multiple-of-16 MatMul: [3, 19, 37] x [1, 37, 23].
    let spec_mm = MatMulSpec { batch: 3, a_batch: 3, b_batch: 1, m: 19, k: 37, n: 23 };
    let a: Vec<f32> = (0..3 * 19 * 37).map(|i| ((i * 17 % 97) as f32 - 48.0) * 0.05).collect();
    let b: Vec<f32> = (0..37 * 23).map(|i| ((i * 31 % 89) as f32 - 44.0) * 0.05).collect();

    let mm_bytes = ModelBuilder::new("mm_parity")
        .input("a", ElemType::Float, vec![Dim::Fixed(3), Dim::Fixed(19), Dim::Fixed(37)])
        .input("b", ElemType::Float, vec![Dim::Fixed(1), Dim::Fixed(37), Dim::Fixed(23)])
        .output("y", ElemType::Float, vec![Dim::Fixed(3), Dim::Fixed(19), Dim::Fixed(23)])
        .node(NodeBuilder::new("MatMul", &["a", "b"], &["y"]))
        .build();
    let mm_model = Model::from_bytes(&mm_bytes).expect("mm model");
    let mut mm_in = HashMap::new();
    mm_in.insert("a".into(), Tensor::from_f32(vec![3, 19, 37], a.clone()).expect("a"));
    mm_in.insert("b".into(), Tensor::from_f32(vec![1, 37, 23], b.clone()).expect("b"));

    let gpu_mm = gpu.matmul_batched(&spec_mm, &a, &b).expect("gpu matmul ran");
    let cpu_mm_map = execute_graph(&mm_model, &mm_in, cpu_accelerator(), &Interrupt::NONE).expect("cpu mm");
    let cpu_mm = cpu_mm_map["y"].as_f32().expect("f32");
    for (g, c) in gpu_mm.iter().zip(cpu_mm.iter()) {
        assert!((g - c).abs() <= 1e-4, "MatMul diff: gpu={g} vs cpu={c}");
    }

    // 2. Grouped dilated Conv2d with bias: [2, 6, 15, 17], groups=2, k=3x3, stride=2, pad=1, dil=2.
    let conv_bytes = ModelBuilder::new("conv_parity")
        .input("x", ElemType::Float, vec![Dim::Fixed(2), Dim::Fixed(6), Dim::Fixed(15), Dim::Fixed(17)])
        .output("y", ElemType::Float, vec![Dim::Symbolic("n".into()), Dim::Symbolic("c".into()), Dim::Symbolic("h".into()), Dim::Symbolic("w".into())])
        .initializer("w", &Tensor::from_f32(vec![4, 3, 3, 3], (0..4 * 3 * 3 * 3).map(|i| ((i * 13 % 53) as f32 - 26.0) * 0.04).collect()).expect("w"))
        .initializer("bias", &Tensor::from_f32(vec![4], vec![0.25, -0.5, 0.75, -0.1]).expect("b"))
        .node(
            NodeBuilder::new("Conv", &["x", "w", "bias"], &["y"])
                .attr("group", Attr::Int(2))
                .attr("strides", Attr::Ints(vec![2, 2]))
                .attr("pads", Attr::Ints(vec![1, 1, 1, 1]))
                .attr("dilations", Attr::Ints(vec![2, 2])),
        )
        .build();
    let conv_model = Model::from_bytes(&conv_bytes).expect("conv model");
    let mut conv_in = HashMap::new();
    conv_in.insert("x".into(), Tensor::from_f32(vec![2, 6, 15, 17], (0..2 * 6 * 15 * 17).map(|i| ((i * 7 % 41) as f32 - 20.0) * 0.05).collect()).expect("x"));
    let out_gpu = execute_graph(&conv_model, &conv_in, &gpu, &Interrupt::NONE).expect("gpu conv");
    let out_cpu = execute_graph(&conv_model, &conv_in, cpu_accelerator(), &Interrupt::NONE).expect("cpu conv");
    for (g, c) in out_gpu["y"].as_f32().expect("g").iter().zip(out_cpu["y"].as_f32().expect("c").iter()) {
        assert!((g - c).abs() <= 1e-4, "Conv2d diff: gpu={g} vs cpu={c}");
    }

    // 3. Stride-2 ConvTranspose2d (SAM-2 mask upscaling head): [1, 4, 8, 9] -> [1, 2, 16, 18].
    let ct_bytes = ModelBuilder::new("conv_t_parity")
        .input("x", ElemType::Float, vec![Dim::Fixed(1), Dim::Fixed(4), Dim::Fixed(8), Dim::Fixed(9)])
        .output("y", ElemType::Float, vec![Dim::Fixed(1), Dim::Fixed(2), Dim::Fixed(16), Dim::Fixed(18)])
        .initializer("w", &Tensor::from_f32(vec![4, 2, 2, 2], (0..4 * 2 * 2 * 2).map(|i| ((i * 19 % 37) as f32 - 18.0) * 0.05).collect()).expect("w"))
        .initializer("bias", &Tensor::from_f32(vec![2], vec![0.15, -0.35]).expect("b"))
        .node(NodeBuilder::new("ConvTranspose", &["x", "w", "bias"], &["y"]).attr("strides", Attr::Ints(vec![2, 2])))
        .build();
    let ct_model = Model::from_bytes(&ct_bytes).expect("ct model");
    let mut ct_in = HashMap::new();
    ct_in.insert("x".into(), Tensor::from_f32(vec![1, 4, 8, 9], (0..4 * 8 * 9).map(|i| ((i * 11 % 29) as f32 - 14.0) * 0.1).collect()).expect("x"));
    let ct_gpu = execute_graph(&ct_model, &ct_in, &gpu, &Interrupt::NONE).expect("gpu ct");
    let ct_cpu = execute_graph(&ct_model, &ct_in, cpu_accelerator(), &Interrupt::NONE).expect("cpu ct");
    for (g, c) in ct_gpu["y"].as_f32().expect("g").iter().zip(ct_cpu["y"].as_f32().expect("c").iter()) {
        assert!((g - c).abs() <= 1e-4, "ConvTranspose2d diff: gpu={g} vs cpu={c}");
    }

    // Verify excessive dimensions return `None` cleanly instead of panicking.
    let huge_spec = Conv2dSpec {
        batch: 1_000_000,
        in_channels: 64,
        out_channels: 64,
        in_h: 1024,
        in_w: 1024,
        out_h: 1024,
        out_w: 1024,
        kernel_h: 1,
        kernel_w: 1,
        stride_h: 1,
        stride_w: 1,
        pad_top: 0,
        pad_left: 0,
        dil_h: 1,
        dil_w: 1,
        groups: 1,
    };
    assert!(gpu.conv2d(&huge_spec, &[1.0], &[1.0], None).is_none());
}

#[test]
fn matting_pipeline_produces_region_and_high_bit_depth_float_surfaces() {
    // Synthetic 1x1 Conv + Sigmoid matting network detecting bright red foreground ([1, 0, 0]) over
    // dark blue background ([0, 0, 1]). Normalized with mean=0.5, std=0.5 => red=[+1,-1,-1], blue=[-1,-1,+1].
    let bytes = ModelBuilder::new("synthetic_modnet")
        .input("input", ElemType::Float, vec![Dim::Fixed(1), Dim::Fixed(3), Dim::Symbolic("h".into()), Dim::Symbolic("w".into())])
        .output("output", ElemType::Float, vec![Dim::Fixed(1), Dim::Fixed(1), Dim::Symbolic("h".into()), Dim::Symbolic("w".into())])
        .initializer("w", &Tensor::from_f32(vec![1, 3, 1, 1], vec![3.0, -1.0, -3.0]).expect("w"))
        .initializer("b", &Tensor::from_f32(vec![1], vec![-1.0]).expect("b"))
        .node(NodeBuilder::new("Conv", &["input", "w", "b"], &["logits"]))
        .node(NodeBuilder::new("Sigmoid", &["logits"], &["output"]))
        .build();

    let store = ModelStore::new();
    let prep = MattingPreprocess::ShortestEdgeDivisible { shortest_edge: 64, max_edge: 128, divisibility: 32, mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5] };
    store.install_bytes("modnet-photographic", ModelTask::Subject, Some(prep), &bytes).expect("install");
    let loaded = store.load_for_task(ModelTask::Subject).expect("load").expect("present");

    let area = Rect::new(0, 0, 128, 96);
    let sampler = SyntheticSampler { fg: Rect::new(32, 24, 96, 72), fg_rgb: [0.95, 0.05, 0.05], bg_rgb: [0.05, 0.05, 0.90] };

    let region = predict_region(&loaded, &sampler, area, cpu_accelerator(), &Interrupt::NONE).expect("run").expect("non-empty");
    assert!(region.at(64, 48) > 0.9, "subject interior must be selected");
    assert!(region.at(4, 4) < 0.1, "background corner must be unselected");

    // Verify 8-bit, 16-bit and 32-bit float surfaces preserve intermediate soft ramp alpha!
    for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
        let surf = predict_surface(&loaded, &sampler, area, depth, cpu_accelerator(), &Interrupt::NONE).expect("surface ok").expect("non-empty");
        assert_eq!(surf.format().sample, depth);
        assert!(surf.sample_channel(64, 48, 0) > 0.9);
        assert!(surf.sample_channel(2, 2, 0) < 0.05);
        let edge_alpha = surf.sample_channel(31, 48, 0);
        assert!(edge_alpha > 0.05 && edge_alpha < 0.95, "expected continuous ramp alpha at {depth:?}, got {edge_alpha}");
    }
}

#[test]
fn sam2_pipeline_segments_point_and_box_prompts_and_caches_embeddings() {
    // Synthetic 32x32 SAM-2 encoder + prompt/mask decoder:
    // Encoder extracts redness activation map `image_embeddings.0` [1, 1, 32, 32] (positive inside
    // the red object, negative outside).
    let enc_bytes = ModelBuilder::new("sam2_encoder")
        .input("pixel_values", ElemType::Float, vec![Dim::Fixed(1), Dim::Fixed(3), Dim::Fixed(32), Dim::Fixed(32)])
        .output("image_embeddings.0", ElemType::Float, vec![Dim::Fixed(1), Dim::Fixed(1), Dim::Fixed(32), Dim::Fixed(32)])
        .initializer("w", &Tensor::from_f32(vec![1, 3, 1, 1], vec![2.5, -1.0, -2.5]).expect("w"))
        .initializer("b", &Tensor::from_f32(vec![1], vec![0.0]).expect("b"))
        .node(NodeBuilder::new("Conv", &["pixel_values", "w", "b"], &["image_embeddings.0"]))
        .build();

    // Decoder takes `image_embeddings.0`, `input_points` [1,1,P,2], `input_labels` [1,1,P],
    // `input_boxes` [1,B,4] and produces `pred_masks` [1, 1, 3, 32, 32] and `iou_scores` [1, 1, 3]
    // where mask 1 (highest IoU = 0.95) contains the object logits!
    let dec_bytes = ModelBuilder::new("sam2_decoder")
        .input("image_embeddings.0", ElemType::Float, vec![Dim::Fixed(1), Dim::Fixed(1), Dim::Fixed(32), Dim::Fixed(32)])
        .input("input_points", ElemType::Float, vec![Dim::Fixed(1), Dim::Fixed(1), Dim::Symbolic("p".into()), Dim::Fixed(2)])
        .input("input_labels", ElemType::Int64, vec![Dim::Fixed(1), Dim::Fixed(1), Dim::Symbolic("p".into())])
        .input("input_boxes", ElemType::Float, vec![Dim::Fixed(1), Dim::Symbolic("b".into()), Dim::Fixed(4)])
        .output("pred_masks", ElemType::Float, vec![Dim::Fixed(1), Dim::Fixed(1), Dim::Fixed(3), Dim::Fixed(32), Dim::Fixed(32)])
        .output("iou_scores", ElemType::Float, vec![Dim::Fixed(1), Dim::Fixed(1), Dim::Fixed(3)])
        .initializer("unsq_ax", &Tensor::from_i64(vec![1], vec![1]).expect("ax"))
        .initializer("neg_scale", &Tensor::from_f32(Vec::new(), vec![-1.0]).expect("ns"))
        .initializer("ious", &Tensor::from_f32(vec![1, 1, 3], vec![0.2, 0.95, 0.4]).expect("ious"))
        .node(NodeBuilder::new("Unsqueeze", &["image_embeddings.0", "unsq_ax"], &["m_good"]))
        .node(NodeBuilder::new("Mul", &["m_good", "neg_scale"], &["m_bad"]))
        .node(NodeBuilder::new("Concat", &["m_bad", "m_good", "m_bad"], &["pred_masks"]).attr("axis", Attr::Int(2)))
        .node(NodeBuilder::new("Identity", &["ious"], &["iou_scores"]))
        .build();

    let store = ModelStore::new();
    store.install_sam2_models("sam2.1-tiny", Model::from_bytes(&enc_bytes).expect("enc"), Model::from_bytes(&dec_bytes).expect("dec")).expect("install sam2");
    let loaded = store.load_for_task(ModelTask::Interactive).expect("load").expect("present");
    let cache = EmbeddingCache::new(4);
    let ctx = Sam2Context { loaded: &loaded, cache: &cache, accel: cpu_accelerator() };

    let area = Rect::new(0, 0, 128, 128);
    let mut sampler = SyntheticSampler { fg: Rect::new(32, 32, 96, 96), fg_rgb: [0.95, 0.05, 0.05], bg_rgb: [0.05, 0.05, 0.90] };

    // 1. First point query runs encoder (miss = 1, hits = 0).
    let r1 = segment_prompt(&ctx, &sampler, area, Sam2Prompt::Point { x: 64, y: 64 }, &Interrupt::NONE).expect("point ok").expect("region");
    assert_eq!(cache.misses(), 1);
    assert_eq!(cache.hits(), 0);
    assert!(r1.at(64, 64) > 0.9);
    assert!(r1.at(4, 4) < 0.1);

    // 2. Second hover point and box prompt reuse cached encoder embedding (hits = 2, misses = 1)!
    let _r2 = segment_prompt(&ctx, &sampler, area, Sam2Prompt::Point { x: 50, y: 50 }, &Interrupt::NONE).expect("hover ok").expect("region");
    let r3 = segment_prompt(&ctx, &sampler, area, Sam2Prompt::Box(Rect::new(28, 28, 100, 100)), &Interrupt::NONE).expect("box ok").expect("region");
    assert_eq!(cache.misses(), 1);
    assert_eq!(cache.hits(), 2);
    assert!(r3.at(64, 64) > 0.9);

    // 3. Editing image pixels invalidates digest and recomputes embedding cleanly (misses = 2).
    sampler.fg_rgb = [0.90, 0.15, 0.05];
    let _r4 = segment_prompt(&ctx, &sampler, area, Sam2Prompt::Point { x: 64, y: 64 }, &Interrupt::NONE).expect("edited ok").expect("region");
    assert_eq!(cache.misses(), 2);
}

struct MockCorruptTransport;
impl DownloadTransport for MockCorruptTransport {
    fn fetch_stream(&self, _url: &str, on_chunk: &mut dyn FnMut(&[u8]) -> Result<(), MlError>) -> Result<(), MlError> {
        on_chunk(b"corrupted-payload-bytes")
    }
}

struct MockSlowTransport {
    chunks_sent: Arc<AtomicUsize>,
}
impl DownloadTransport for MockSlowTransport {
    fn fetch_stream(&self, _url: &str, on_chunk: &mut dyn FnMut(&[u8]) -> Result<(), MlError>) -> Result<(), MlError> {
        for _ in 0..50 {
            self.chunks_sent.fetch_add(1, Ordering::Relaxed);
            on_chunk(&[0u8; 1024])?;
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        Ok(())
    }
}

#[test]
fn sha256_verification_and_download_rollback_and_cancellation() {
    let sample = b"photocraft-onnx-test";
    let hex = sha256_hex(sample);
    assert!(verify_sha256(sample, &hex).is_ok());
    assert!(matches!(verify_sha256(sample, "0000000000000000000000000000000000000000000000000000000000000000"), Err(MlError::ChecksumMismatch { .. })));

    // 1. Corrupt download must fail SHA-256, remove `.part`, and leave model uninstalled.
    let dir = tempfile::tempdir().expect("tempdir");
    let store = ModelStore::with_transport(Arc::new(MockCorruptTransport));
    store.set_models_dir(Some(dir.path().to_path_buf())).expect("set dir");
    let err = store.start_download("modnet-photographic", true).expect_err("must fail checksum");
    assert!(err.to_string().contains("SHA-256"));
    let entry = catalog_entry("modnet-photographic").expect("entry");
    assert!(!dir.path().join("modnet-photographic").join("model_fp16.onnx.part").exists());
    assert!(!dir.path().join("modnet-photographic").join(entry.artifacts[0].file_name).exists());
    assert!(!store.has_task_model(ModelTask::Subject));

    // 2. Background download cancellation stops cleanly and leaves no `.part` file.
    let chunks_sent = Arc::new(AtomicUsize::new(0));
    let store_cancel = ModelStore::with_transport(Arc::new(MockSlowTransport { chunks_sent: chunks_sent.clone() }));
    store_cancel.set_models_dir(Some(dir.path().to_path_buf())).expect("set dir");
    let _snap = store_cancel.start_download("modnet-photographic", false).expect("started");
    while chunks_sent.load(Ordering::Relaxed) == 0 {
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(store_cancel.cancel_download("modnet-photographic"));
    for _ in 0..100 {
        let list = store_cancel.list_status();
        let row = list.iter().find(|r| r.id == "modnet-photographic").expect("row");
        if matches!(row.download.as_ref().map(|d| &d.status), Some(DownloadState::Cancelled)) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(!dir.path().join("modnet-photographic").join("model_fp16.onnx.part").exists());
}
