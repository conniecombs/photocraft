//! Clean-room ONNX (`ModelProto`) representation, signature metadata and topological validation.
//!
//! Supports both self-contained `.onnx` files and split external-data models (`.onnx` +
//! `.onnx_data`, such as `sam2.1-hiera-tiny-ONNX`), resolving external weight segments only
//! through a caller-supplied byte resolver that rejects path traversal.

pub mod builder;
pub mod proto;

use std::collections::{HashMap, HashSet};

use crate::MlError;
use crate::tensor::{DType, Tensor};

/// Maximum model file size accepted by [`Model::from_bytes`] (1 GiB).
pub const MAX_MODEL_BYTES: usize = 1 << 30;

/// Maximum nodes allowed in a single ONNX graph.
pub const MAX_GRAPH_NODES: usize = 100_000;

/// ONNX `TensorProto.DataType` codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElemType {
    Float,
    Uint8,
    Int8,
    Uint16,
    Int16,
    Int32,
    Int64,
    String,
    Bool,
    Float16,
    Double,
    Uint32,
    Uint64,
    BFloat16,
}

impl ElemType {
    pub fn from_onnx(code: i32) -> Result<Self, MlError> {
        match code {
            1 => Ok(ElemType::Float),
            2 => Ok(ElemType::Uint8),
            3 => Ok(ElemType::Int8),
            4 => Ok(ElemType::Uint16),
            5 => Ok(ElemType::Int16),
            6 => Ok(ElemType::Int32),
            7 => Ok(ElemType::Int64),
            8 => Ok(ElemType::String),
            9 => Ok(ElemType::Bool),
            10 => Ok(ElemType::Float16),
            11 => Ok(ElemType::Double),
            12 => Ok(ElemType::Uint32),
            13 => Ok(ElemType::Uint64),
            16 => Ok(ElemType::BFloat16),
            other => Err(MlError::InvalidModel(format!("unsupported ONNX tensor element type {other}"))),
        }
    }

    pub const fn to_onnx(self) -> i32 {
        match self {
            ElemType::Float => 1,
            ElemType::Uint8 => 2,
            ElemType::Int8 => 3,
            ElemType::Uint16 => 4,
            ElemType::Int16 => 5,
            ElemType::Int32 => 6,
            ElemType::Int64 => 7,
            ElemType::String => 8,
            ElemType::Bool => 9,
            ElemType::Float16 => 10,
            ElemType::Double => 11,
            ElemType::Uint32 => 12,
            ElemType::Uint64 => 13,
            ElemType::BFloat16 => 16,
        }
    }

    /// Runtime [`DType`] used when executing operations on this ONNX element type.
    pub const fn runtime_dtype(self) -> DType {
        match self {
            ElemType::Float | ElemType::Float16 | ElemType::BFloat16 | ElemType::Double => DType::F32,
            ElemType::Uint8
            | ElemType::Int8
            | ElemType::Uint16
            | ElemType::Int16
            | ElemType::Int32
            | ElemType::Int64
            | ElemType::Bool
            | ElemType::Uint32
            | ElemType::Uint64
            | ElemType::String => DType::I64,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            ElemType::Float => "float32",
            ElemType::Uint8 => "uint8",
            ElemType::Int8 => "int8",
            ElemType::Uint16 => "uint16",
            ElemType::Int16 => "int16",
            ElemType::Int32 => "int32",
            ElemType::Int64 => "int64",
            ElemType::String => "string",
            ElemType::Bool => "bool",
            ElemType::Float16 => "float16",
            ElemType::Double => "float64",
            ElemType::Uint32 => "uint32",
            ElemType::Uint64 => "uint64",
            ElemType::BFloat16 => "bfloat16",
        }
    }
}

/// Symbolic or fixed dimension in a [`ValueInfo`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Dim {
    Fixed(usize),
    Symbolic(String),
}

/// Declared graph input or output descriptor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValueInfo {
    pub name: String,
    pub elem_type: Option<ElemType>,
    pub dims: Vec<Dim>,
}

/// Node attribute value.
#[derive(Clone, Debug, PartialEq)]
pub enum Attr {
    Float(f32),
    Int(i64),
    String(String),
    Tensor(Tensor),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Strings(Vec<String>),
}

/// One computation node in an ONNX [`Graph`].
#[derive(Clone, Debug)]
pub struct Node {
    pub name: String,
    pub op_type: String,
    pub domain: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: HashMap<String, Attr>,
}

impl Node {
    pub fn attr_int(&self, key: &str) -> Option<i64> {
        match self.attrs.get(key) {
            Some(Attr::Int(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn attr_float(&self, key: &str) -> Option<f32> {
        match self.attrs.get(key) {
            Some(Attr::Float(v)) => Some(*v),
            Some(Attr::Int(v)) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn attr_str(&self, key: &str) -> Option<&str> {
        match self.attrs.get(key) {
            Some(Attr::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn attr_ints(&self, key: &str) -> Option<&[i64]> {
        match self.attrs.get(key) {
            Some(Attr::Ints(v)) => Some(v.as_slice()),
            _ => None,
        }
    }

    pub fn attr_floats(&self, key: &str) -> Option<&[f32]> {
        match self.attrs.get(key) {
            Some(Attr::Floats(v)) => Some(v.as_slice()),
            _ => None,
        }
    }

    pub fn attr_tensor(&self, key: &str) -> Option<&Tensor> {
        match self.attrs.get(key) {
            Some(Attr::Tensor(t)) => Some(t),
            _ => None,
        }
    }
}

/// Parsed ONNX computation graph, topologically validated and ready to run.
#[derive(Clone, Debug)]
pub struct Graph {
    pub name: String,
    pub nodes: Vec<Node>,
    pub initializers: HashMap<String, Tensor>,
    /// Real runtime inputs (declared graph inputs excluding initializers).
    pub inputs: Vec<ValueInfo>,
    pub outputs: Vec<ValueInfo>,
}

/// Summary of a model's inputs, outputs and operator set for inspection and validation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelSignature {
    pub ir_version: i64,
    pub opset_version: i64,
    pub producer: String,
    pub inputs: Vec<ValueInfo>,
    pub outputs: Vec<ValueInfo>,
    pub node_count: usize,
    pub initializer_count: usize,
}

/// Resolver callback for ONNX external tensor data (`location`, `offset`, `length`).
pub type ExternalResolver<'a> = &'a dyn Fn(&str, u64, Option<u64>) -> Result<Vec<u8>, MlError>;

/// Parsed, validated ONNX model.
#[derive(Clone, Debug)]
pub struct Model {
    pub ir_version: i64,
    pub opset_version: i64,
    pub producer_name: String,
    pub metadata: HashMap<String, String>,
    pub graph: Graph,
}

/// Validate an ONNX `external_data` `location` string: must be a plain single-component file name
/// without separators, drive letters or parent traversal.
pub fn validate_external_location(location: &str) -> Result<&str, MlError> {
    let trimmed = location.trim();
    if trimmed.is_empty()
        || trimmed == "."
        || trimmed == ".."
        || trimmed.contains('/')
        || trimmed.contains('\\')
        || trimmed.contains(':')
        || trimmed.contains('\0')
    {
        return Err(MlError::InvalidModel(format!("rejected unsafe external_data location `{location}`")));
    }
    Ok(trimmed)
}

impl Model {
    /// Parse and validate a self-contained `.onnx` byte buffer.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, MlError> {
        Self::from_bytes_with_external(bytes, &|loc, _, _| {
            Err(MlError::InvalidModel(format!("model references external tensor data `{loc}`, which was not provided")))
        })
    }

    /// Parse and validate an `.onnx` buffer, resolving any external weight segments through `ext`.
    pub fn from_bytes_with_external(bytes: &[u8], ext: ExternalResolver<'_>) -> Result<Self, MlError> {
        if bytes.is_empty() {
            return Err(MlError::InvalidModel("empty ONNX buffer".into()));
        }
        if bytes.len() > MAX_MODEL_BYTES {
            return Err(MlError::MemoryLimitExceeded { needed: bytes.len(), limit: MAX_MODEL_BYTES });
        }
        let mut model = proto::parse_model(bytes, ext)?;
        model.validate_and_order()?;
        Ok(model)
    }

    /// Lightweight metadata summary of this model.
    pub fn signature(&self) -> ModelSignature {
        ModelSignature {
            ir_version: self.ir_version,
            opset_version: self.opset_version,
            producer: self.producer_name.clone(),
            inputs: self.graph.inputs.clone(),
            outputs: self.graph.outputs.clone(),
            node_count: self.graph.nodes.len(),
            initializer_count: self.graph.initializers.len(),
        }
    }

    /// Ensure the graph has outputs, stays within node limits, and sort nodes topologically if
    /// the exporter emitted them out of dependency order, rejecting cycles or missing values.
    fn validate_and_order(&mut self) -> Result<(), MlError> {
        if self.graph.nodes.len() > MAX_GRAPH_NODES {
            return Err(MlError::InvalidModel(format!("ONNX graph has {} nodes, exceeding {MAX_GRAPH_NODES}", self.graph.nodes.len())));
        }
        if self.graph.outputs.is_empty() {
            return Err(MlError::InvalidModel("ONNX graph declares no outputs".into()));
        }
        let mut available: HashSet<String> = self.graph.initializers.keys().cloned().collect();
        for inp in &self.graph.inputs {
            available.insert(inp.name.clone());
        }
        // Fast check: most exporters already emit nodes in topological order.
        if is_topological(&self.graph.nodes, &available) {
            return check_outputs_reachable(&self.graph, &available);
        }
        self.graph.nodes = topological_sort(&self.graph.nodes, &mut available)?;
        check_outputs_reachable(&self.graph, &available)
    }
}

fn is_topological(nodes: &[Node], initial: &HashSet<String>) -> bool {
    let mut ready = initial.clone();
    for node in nodes {
        if !node.inputs.iter().all(|inp| inp.is_empty() || ready.contains(inp)) {
            return false;
        }
        for out in &node.outputs {
            if !out.is_empty() {
                ready.insert(out.clone());
            }
        }
    }
    true
}

fn topological_sort(nodes: &[Node], available: &mut HashSet<String>) -> Result<Vec<Node>, MlError> {
    let mut remaining: Vec<Option<Node>> = nodes.iter().cloned().map(Some).collect();
    let mut ordered = Vec::with_capacity(nodes.len());
    while ordered.len() < nodes.len() {
        let mut progressed = false;
        for slot in &mut remaining {
            let can_run = match slot.as_ref() {
                Some(node) => node.inputs.iter().all(|inp| inp.is_empty() || available.contains(inp)),
                None => false,
            };
            if can_run && let Some(node) = slot.take() {
                for out in &node.outputs {
                    if !out.is_empty() {
                        available.insert(out.clone());
                    }
                }
                ordered.push(node);
                progressed = true;
            }
        }
        if !progressed {
            let stuck = remaining.iter().flatten().next().map(|n| format!("{} ({})", n.op_type, n.name)).unwrap_or_else(|| "unknown".into());
            return Err(MlError::InvalidModel(format!("ONNX graph has a cycle or unresolved input at node {stuck}")));
        }
    }
    Ok(ordered)
}

fn check_outputs_reachable(graph: &Graph, initial: &HashSet<String>) -> Result<(), MlError> {
    let mut produced = initial.clone();
    for node in &graph.nodes {
        for out in &node.outputs {
            if !out.is_empty() {
                produced.insert(out.clone());
            }
        }
    }
    for out in &graph.outputs {
        if !produced.contains(&out.name) {
            return Err(MlError::InvalidModel(format!("ONNX graph output `{}` is never produced", out.name)));
        }
    }
    Ok(())
}
