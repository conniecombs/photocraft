use base64::Engine as _;
use photocraft_color::{ColorMode, SampleType};
use photocraft_engine::Session;
use photocraft_geom::Rect;
use photocraft_ml::Tensor;
use photocraft_ml::onnx::builder::{ModelBuilder, NodeBuilder};
use photocraft_ml::onnx::{Attr, Dim, ElemType};
use serde_json::json;

fn b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Synthetic matting model that predicts 1.0 wherever R > B (foreground red square) and 0.0
/// wherever B > R (background blue).
fn synthetic_subject_onnx() -> Vec<u8> {
    ModelBuilder::new("engine_subject_test")
        .input("input", ElemType::Float, vec![Dim::Fixed(1), Dim::Fixed(3), Dim::Symbolic("h".into()), Dim::Symbolic("w".into())])
        .output("output", ElemType::Float, vec![Dim::Fixed(1), Dim::Fixed(1), Dim::Symbolic("h".into()), Dim::Symbolic("w".into())])
        .initializer("w", &Tensor::from_f32(vec![1, 3, 1, 1], vec![4.0, -0.5, -4.0]).expect("w"))
        .initializer("b", &Tensor::from_f32(vec![1], vec![-0.5]).expect("b"))
        .node(NodeBuilder::new("Conv", &["input", "w", "b"], &["logits"]))
        .node(NodeBuilder::new("Sigmoid", &["logits"], &["output"]))
        .build()
}

fn synthetic_sam2_pair() -> (Vec<u8>, Vec<u8>) {
    let enc = ModelBuilder::new("engine_sam2_enc")
        .input("pixel_values", ElemType::Float, vec![Dim::Fixed(1), Dim::Fixed(3), Dim::Fixed(32), Dim::Fixed(32)])
        .output("image_embeddings.0", ElemType::Float, vec![Dim::Fixed(1), Dim::Fixed(1), Dim::Fixed(32), Dim::Fixed(32)])
        .initializer("w", &Tensor::from_f32(vec![1, 3, 1, 1], vec![3.0, -0.5, -3.0]).expect("w"))
        .initializer("b", &Tensor::from_f32(vec![1], vec![0.0]).expect("b"))
        .node(NodeBuilder::new("Conv", &["pixel_values", "w", "b"], &["image_embeddings.0"]))
        .build();

    let dec = ModelBuilder::new("engine_sam2_dec")
        .input("image_embeddings.0", ElemType::Float, vec![Dim::Fixed(1), Dim::Fixed(1), Dim::Fixed(32), Dim::Fixed(32)])
        .input("input_points", ElemType::Float, vec![Dim::Fixed(1), Dim::Fixed(1), Dim::Symbolic("p".into()), Dim::Fixed(2)])
        .input("input_labels", ElemType::Int64, vec![Dim::Fixed(1), Dim::Fixed(1), Dim::Symbolic("p".into())])
        .input("input_boxes", ElemType::Float, vec![Dim::Fixed(1), Dim::Symbolic("b".into()), Dim::Fixed(4)])
        .output("pred_masks", ElemType::Float, vec![Dim::Fixed(1), Dim::Fixed(1), Dim::Fixed(3), Dim::Fixed(32), Dim::Fixed(32)])
        .output("iou_scores", ElemType::Float, vec![Dim::Fixed(1), Dim::Fixed(1), Dim::Fixed(3)])
        .initializer("unsq_ax", &Tensor::from_i64(vec![1], vec![1]).expect("ax"))
        .initializer("neg", &Tensor::from_f32(Vec::new(), vec![-1.0]).expect("neg"))
        .initializer("ious", &Tensor::from_f32(vec![1, 1, 3], vec![0.1, 0.99, 0.2]).expect("ious"))
        .node(NodeBuilder::new("Unsqueeze", &["image_embeddings.0", "unsq_ax"], &["m1"]))
        .node(NodeBuilder::new("Mul", &["m1", "neg"], &["m0"]))
        .node(NodeBuilder::new("Concat", &["m0", "m1", "m0"], &["pred_masks"]).attr("axis", Attr::Int(2)))
        .node(NodeBuilder::new("Identity", &["ious"], &["iou_scores"]))
        .build();

    (enc, dec)
}

#[test]
fn ml_commands_and_preferences_round_trip_and_reject_bad_params() {
    let mut s = Session::new();
    let list = s.execute("ml.list", json!({})).expect("ml.list");
    assert_eq!(list["enabled"], false);
    assert_eq!(list["backend"], "auto");
    assert_eq!(list["models"].as_array().expect("models").len(), 3);

    // Preferences validation for integrations.enableLocalModels and integrations.mlBackend.
    s.execute("prefs.set", json!({ "values": { "integrations.enableLocalModels": true, "integrations.mlBackend": "cpu" } })).expect("prefs.set");
    assert!(s.execute("prefs.set", json!({ "values": { "integrations.mlBackend": "quantum" } })).is_err());

    // Hostile/invalid ml.* calls return Err cleanly, never panicking.
    for (cmd, bad_p) in [
        ("ml.install", json!({})),
        ("ml.install", json!({"id": "x", "data": "%%%not-base64%%%"})),
        ("ml.install", json!({"id": "../escape", "data": b64(&synthetic_subject_onnx())})),
        ("ml.download", json!({"id": "nonexistent-model"})),
        ("ml.cancelDownload", json!({})),
        ("ml.remove", json!({"id": "not-installed"})),
    ] {
        assert!(s.execute(cmd, bad_p).is_err(), "{cmd} should reject bad params");
    }

    // Install and remove via base64.
    let inst = s.execute("ml.install", json!({ "id": "custom-subject", "task": "subject", "data": b64(&synthetic_subject_onnx()) })).expect("install");
    assert_eq!(inst["installed"], true);

    let list_after = s.execute("ml.list", json!({})).expect("ml.list");
    assert_eq!(list_after["enabled"], true);
    assert_eq!(list_after["backend"], "cpu");
    assert_eq!(list_after["models"].as_array().expect("models").len(), 4);

    s.execute("ml.remove", json!({ "id": "custom-subject" })).expect("remove");
    let list_removed = s.execute("ml.list", json!({})).expect("ml.list");
    assert_eq!(list_removed["models"].as_array().expect("models").len(), 3);
}

#[test]
fn select_subject_remove_background_and_sky_work_across_bit_depths_and_colour_spaces() {
    for (mode, depth) in [("rgb", 8), ("rgb", 16), ("rgb", 32), ("cmyk", 8)] {
        let mut s = Session::new();
        s.execute("file.new", json!({ "width": 96, "height": 96, "mode": mode, "depth": depth })).expect("file.new");

        // Paint blue background and red central subject [24..72, 24..72] in the layer's native format.
        s.edit("paint", |doc, active| {
            let lid = active.expect("active");
            let surf = doc.layer_mut(lid).expect("layer").surface_mut().expect("surf");
            let fmt = surf.format();
            let blue = photocraft_raster::from_rgba(&fmt, [0.05, 0.05, 0.90, 1.0]);
            let red = photocraft_raster::from_rgba(&fmt, [0.95, 0.05, 0.05, 1.0]);
            surf.fill_rect(Rect::new(0, 0, 96, 96), &blue);
            surf.fill_rect(Rect::new(24, 24, 72, 72), &red);
            Ok(())
        })
        .expect("paint");

        // Enable local models and install synthetic subject & sky models.
        s.execute("prefs.set", json!({ "values": { "integrations.enableLocalModels": true, "integrations.mlBackend": "cpu" } })).expect("enable");
        let onnx = b64(&synthetic_subject_onnx());
        s.execute("ml.install", json!({ "id": "modnet-photographic", "task": "subject", "data": onnx })).expect("install subject");
        s.execute("ml.install", json!({ "id": "sky-seg", "task": "sky", "data": onnx })).expect("install sky");

        // 1. Select › Subject.
        let subj = s.execute("select.subject", json!({})).expect("select.subject");
        assert_eq!(subj["selected"], true);
        let sel = s.active().expect("doc").doc.selection.as_ref().expect("selection");
        assert!(sel.sample_channel(48, 48, 0) > 0.85, "subject centre selected in {mode}/{depth}");
        assert!(sel.sample_channel(4, 4, 0) < 0.15, "corner unselected in {mode}/{depth}");
        s.execute("select.deselect", json!({})).expect("deselect");

        // 2. Select › Sky.
        let sky = s.execute("select.sky", json!({})).expect("select.sky");
        assert_eq!(sky["selected"], true);
        s.execute("select.deselect", json!({})).expect("deselect");

        // 3. Remove Background preserves document bit depth (U8, U16, F32) and supports Undo!
        if mode == "rgb" {
            let rb = s.execute("layer.removeBackground", json!({})).expect("removeBackground");
            assert!(rb.get("bounds").is_some());
            let d = s.active().expect("doc");
            assert_eq!(d.doc.mode, ColorMode::Rgb);
            let lid = d.active_layer.expect("lid");
            let mask = d.doc.layer(lid).expect("layer").mask.as_ref().expect("mask");
            let expected_sample = match depth {
                8 => SampleType::U8,
                16 => SampleType::U16,
                _ => SampleType::F32,
            };
            assert_eq!(mask.surface.format().sample, expected_sample);
            assert!(mask.surface.sample_channel(48, 48, 0) > 0.85);
            assert!(mask.surface.sample_channel(4, 4, 0) < 0.15);

            // Single-step undo restores unmasked layer.
            s.execute("edit.undo", json!({})).expect("undo");
            let d_undo = s.active().expect("doc");
            assert!(d_undo.doc.layer(lid).expect("layer").mask.is_none());
        }
    }
}

#[test]
fn object_selection_and_select_and_mask_use_sam2_with_embedding_cache() {
    let mut s = Session::new();
    s.execute("file.new", json!({ "width": 128, "height": 128 })).expect("new");
    s.edit("paint", |doc, active| {
        let surf = doc.layer_mut(active.expect("lid")).expect("layer").surface_mut().expect("surf");
        surf.fill_rect(Rect::new(0, 0, 128, 128), &[0.05, 0.10, 0.85, 1.0]);
        surf.fill_rect(Rect::new(32, 32, 96, 96), &[0.95, 0.10, 0.05, 1.0]);
        Ok(())
    })
    .expect("paint");

    let (enc, dec) = synthetic_sam2_pair();
    s.execute("prefs.set", json!({ "values": { "integrations.enableLocalModels": true, "integrations.mlBackend": "cpu" } })).expect("enable");
    s.execute("ml.install", json!({ "id": "sam2.1-tiny", "encoderData": b64(&enc), "decoderData": b64(&dec) })).expect("install sam2");

    // 1. Point mode (`select.object` with `"point"`).
    let pt_res = s.execute("select.object", json!({ "point": [64, 64] })).expect("point");
    assert_eq!(pt_res["selected"], true);
    assert_eq!(s.ml.sam2_cache.misses(), 1);
    assert_eq!(s.ml.sam2_cache.hits(), 0);

    // 2. Box mode (`select.object` with `"rect"`) reuses cached vision embedding!
    let box_res = s.execute("select.object", json!({ "rect": [24, 24, 80, 80] })).expect("rect");
    assert_eq!(box_res["selected"], true);
    assert_eq!(s.ml.sam2_cache.misses(), 1);
    assert_eq!(s.ml.sam2_cache.hits(), 1);

    // 3. Select and Mask (`select.refineEdge`) with `clear` + `object` edit also reuses cached embedding!
    let sm_res = s
        .execute(
            "select.refineEdge",
            json!({
                "edits": [
                    { "tool": "clear" },
                    { "tool": "object", "point": [60, 60], "mode": "add", "sampleAllLayers": true }
                ],
                "output": "selection",
            }),
        )
        .expect("refineEdge");
    assert_eq!(sm_res["selected"], true);
    assert!(s.ml.sam2_cache.hits() >= 2);
}
