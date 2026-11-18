//! Topological ONNX graph execution with per-node [`Interrupt`] cancellation/progress and
//! reference-counted intermediate tensor eviction.

pub mod conv;
pub mod linalg;
pub mod math;
pub mod norm;
pub mod shape;

use std::collections::HashMap;

use photocraft_raster::Interrupt;

use crate::MlError;
use crate::backend::ComputeAccelerator;
use crate::onnx::{Model, Node};
use crate::tensor::Tensor;

fn dispatch_node(node: &Node, inputs: &[Option<&Tensor>], accel: &dyn ComputeAccelerator) -> Result<Vec<Tensor>, MlError> {
    if let Some(outs) = shape::try_run(node, inputs)? {
        return Ok(outs);
    }
    if let Some(out) = math::try_run(node, inputs)? {
        return Ok(vec![out]);
    }
    if let Some(out) = norm::try_run(node, inputs)? {
        return Ok(vec![out]);
    }
    if let Some(out) = linalg::try_run(node, inputs, accel)? {
        return Ok(vec![out]);
    }
    if let Some(outs) = conv::try_run(node, inputs, accel)? {
        return Ok(outs);
    }
    Err(MlError::UnsupportedOp { op: node.op_type.clone(), node: node.name.clone() })
}

/// Execute `model` on `runtime_inputs` using `accel`, checking `ctl` before every node and freeing
/// intermediate activations as soon as their last consumer finishes.
pub fn execute_graph(
    model: &Model,
    runtime_inputs: &HashMap<String, Tensor>,
    accel: &dyn ComputeAccelerator,
    ctl: &Interrupt<'_>,
) -> Result<HashMap<String, Tensor>, MlError> {
    let graph = &model.graph;
    for req_in in &graph.inputs {
        if !runtime_inputs.contains_key(&req_in.name) && !graph.initializers.contains_key(&req_in.name) {
            return Err(MlError::MissingInput(req_in.name.clone()));
        }
    }

    // Count downstream consumers for every intermediate value so dead activations drop immediately.
    let mut remaining_uses: HashMap<&str, usize> = HashMap::new();
    for node in &graph.nodes {
        for inp in &node.inputs {
            if !inp.is_empty() {
                *remaining_uses.entry(inp.as_str()).or_insert(0) += 1;
            }
        }
    }
    for out in &graph.outputs {
        *remaining_uses.entry(out.name.as_str()).or_insert(0) += 1;
    }

    let mut intermediates: HashMap<String, Tensor> = HashMap::new();
    let total_nodes = graph.nodes.len().max(1);

    for (step_idx, node) in graph.nodes.iter().enumerate() {
        if ctl.cancelled() {
            return Err(MlError::Cancelled);
        }
        if step_idx % 8 == 0 {
            ctl.progress((step_idx as f32) / (total_nodes as f32));
        }

        let outs = {
            let mut resolved: Vec<Option<&Tensor>> = Vec::with_capacity(node.inputs.len());
            for inp in &node.inputs {
                if inp.is_empty() {
                    resolved.push(None);
                } else if let Some(t) = intermediates.get(inp.as_str()) {
                    resolved.push(Some(t));
                } else if let Some(t) = runtime_inputs.get(inp.as_str()) {
                    resolved.push(Some(t));
                } else if let Some(t) = graph.initializers.get(inp.as_str()) {
                    resolved.push(Some(t));
                } else {
                    return Err(MlError::InvalidModel(format!("node `{}` ({}) references missing tensor `{inp}`", node.name, node.op_type)));
                }
            }
            dispatch_node(node, &resolved, accel)?
        };

        for inp in &node.inputs {
            if inp.is_empty() {
                continue;
            }
            if let Some(rem) = remaining_uses.get_mut(inp.as_str()) {
                *rem = rem.saturating_sub(1);
                if *rem == 0 {
                    intermediates.remove(inp.as_str());
                }
            }
        }

        for (out_name, tensor) in node.outputs.iter().zip(outs) {
            if out_name.is_empty() {
                continue;
            }
            if remaining_uses.get(out_name.as_str()).copied().unwrap_or(0) > 0 {
                intermediates.insert(out_name.clone(), tensor);
            }
        }
    }

    if ctl.cancelled() {
        return Err(MlError::Cancelled);
    }
    ctl.progress(1.0);

    let mut results = HashMap::with_capacity(graph.outputs.len());
    for out in &graph.outputs {
        let name = out.name.as_str();
        if let Some(t) = intermediates.remove(name) {
            results.insert(out.name.clone(), t);
        } else if let Some(t) = runtime_inputs.get(name) {
            results.insert(out.name.clone(), t.clone());
        } else if let Some(t) = graph.initializers.get(name) {
            results.insert(out.name.clone(), t.clone());
        } else {
            return Err(MlError::InvalidModel(format!("graph output `{name}` was never produced")));
        }
    }
    Ok(results)
}
