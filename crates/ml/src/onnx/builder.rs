//! Spec-compliant ONNX `ModelProto` binary builder for unit tests and synthetic fixtures.

use crate::onnx::{Attr, Dim, ElemType, ValueInfo};
use crate::tensor::{Tensor, TensorData};

fn push_varint(out: &mut Vec<u8>, mut val: u64) {
    loop {
        let mut b = (val & 0x7F) as u8;
        val >>= 7;
        if val != 0 {
            b |= 0x80;
        }
        out.push(b);
        if val == 0 {
            break;
        }
    }
}

fn push_tag(out: &mut Vec<u8>, field_num: u32, wire_type: u8) {
    push_varint(out, (u64::from(field_num) << 3) | u64::from(wire_type));
}

fn push_bytes(out: &mut Vec<u8>, field_num: u32, bytes: &[u8]) {
    push_tag(out, field_num, 2);
    push_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

fn push_str(out: &mut Vec<u8>, field_num: u32, s: &str) {
    push_bytes(out, field_num, s.as_bytes());
}

fn push_varint_field(out: &mut Vec<u8>, field_num: u32, val: u64) {
    push_tag(out, field_num, 0);
    push_varint(out, val);
}

fn push_f32_field(out: &mut Vec<u8>, field_num: u32, val: f32) {
    push_tag(out, field_num, 5);
    out.extend_from_slice(&val.to_le_bytes());
}

/// Encode a runtime [`Tensor`] into an ONNX `TensorProto` message.
pub fn encode_tensor_proto(name: &str, tensor: &Tensor) -> Vec<u8> {
    let mut out = Vec::new();
    for &d in tensor.shape() {
        push_varint_field(&mut out, 1, d as u64);
    }
    match tensor.data() {
        TensorData::F32(v) => {
            push_varint_field(&mut out, 2, ElemType::Float.to_onnx() as u64);
            push_str(&mut out, 8, name);
            let mut raw = Vec::with_capacity(v.len() * 4);
            for &x in v {
                raw.extend_from_slice(&x.to_le_bytes());
            }
            push_bytes(&mut out, 9, &raw);
        }
        TensorData::I64(v) => {
            push_varint_field(&mut out, 2, ElemType::Int64.to_onnx() as u64);
            push_str(&mut out, 8, name);
            let mut raw = Vec::with_capacity(v.len() * 8);
            for &x in v {
                raw.extend_from_slice(&x.to_le_bytes());
            }
            push_bytes(&mut out, 9, &raw);
        }
    }
    out
}

/// Encode an `f16` initializer `TensorProto` (`raw_data` in IEEE 754 binary16).
pub fn encode_f16_tensor_proto(name: &str, shape: &[usize], values: &[f32]) -> Vec<u8> {
    let mut out = Vec::new();
    for &d in shape {
        push_varint_field(&mut out, 1, d as u64);
    }
    push_varint_field(&mut out, 2, ElemType::Float16.to_onnx() as u64);
    push_str(&mut out, 8, name);
    let mut raw = Vec::with_capacity(values.len() * 2);
    for &x in values {
        raw.extend_from_slice(&half::f16::from_f32(x).to_le_bytes());
    }
    push_bytes(&mut out, 9, &raw);
    out
}

/// Encode a `TensorProto` referencing external weights (`data_location = EXTERNAL`).
pub fn encode_external_tensor_proto(name: &str, elem: ElemType, shape: &[usize], location: &str, offset: u64, length: u64) -> Vec<u8> {
    let mut out = Vec::new();
    for &d in shape {
        push_varint_field(&mut out, 1, d as u64);
    }
    push_varint_field(&mut out, 2, elem.to_onnx() as u64);
    push_str(&mut out, 8, name);
    for (k, v) in [("location", location.to_owned()), ("offset", offset.to_string()), ("length", length.to_string())] {
        let mut entry = Vec::new();
        push_str(&mut entry, 1, k);
        push_str(&mut entry, 2, &v);
        push_bytes(&mut out, 13, &entry);
    }
    push_varint_field(&mut out, 14, 1);
    out
}

fn encode_value_info(vi: &ValueInfo) -> Vec<u8> {
    let mut shape_bytes = Vec::new();
    for d in &vi.dims {
        let mut dim_bytes = Vec::new();
        match d {
            Dim::Fixed(n) => push_varint_field(&mut dim_bytes, 1, *n as u64),
            Dim::Symbolic(s) => push_str(&mut dim_bytes, 2, s),
        }
        push_bytes(&mut shape_bytes, 1, &dim_bytes);
    }
    let mut tensor_type = Vec::new();
    if let Some(et) = vi.elem_type {
        push_varint_field(&mut tensor_type, 1, et.to_onnx() as u64);
    }
    push_bytes(&mut tensor_type, 2, &shape_bytes);

    let mut type_proto = Vec::new();
    push_bytes(&mut type_proto, 1, &tensor_type);

    let mut out = Vec::new();
    push_str(&mut out, 1, &vi.name);
    push_bytes(&mut out, 2, &type_proto);
    out
}

fn encode_attr(name: &str, attr: &Attr) -> Vec<u8> {
    let mut out = Vec::new();
    push_str(&mut out, 1, name);
    match attr {
        Attr::Float(f) => {
            push_f32_field(&mut out, 2, *f);
            push_varint_field(&mut out, 20, 1);
        }
        Attr::Int(i) => {
            push_varint_field(&mut out, 3, *i as u64);
            push_varint_field(&mut out, 20, 2);
        }
        Attr::String(s) => {
            push_str(&mut out, 4, s);
            push_varint_field(&mut out, 20, 3);
        }
        Attr::Tensor(t) => {
            let tp = encode_tensor_proto(name, t);
            push_bytes(&mut out, 5, &tp);
            push_varint_field(&mut out, 20, 4);
        }
        Attr::Floats(fs) => {
            for &f in fs {
                push_f32_field(&mut out, 7, f);
            }
            push_varint_field(&mut out, 20, 6);
        }
        Attr::Ints(is) => {
            for &i in is {
                push_varint_field(&mut out, 8, i as u64);
            }
            push_varint_field(&mut out, 20, 7);
        }
        Attr::Strings(ss) => {
            for s in ss {
                push_str(&mut out, 9, s);
            }
            push_varint_field(&mut out, 20, 8);
        }
    }
    out
}

/// Fluent builder for a single ONNX `NodeProto`.
pub struct NodeBuilder {
    name: String,
    op_type: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    attrs: Vec<(String, Attr)>,
}

impl NodeBuilder {
    pub fn new(op_type: &str, inputs: &[&str], outputs: &[&str]) -> Self {
        Self {
            name: String::new(),
            op_type: op_type.to_owned(),
            inputs: inputs.iter().map(|s| (*s).to_owned()).collect(),
            outputs: outputs.iter().map(|s| (*s).to_owned()).collect(),
            attrs: Vec::new(),
        }
    }

    pub fn name(mut self, name: &str) -> Self {
        self.name = name.to_owned();
        self
    }

    pub fn attr(mut self, key: &str, val: Attr) -> Self {
        self.attrs.push((key.to_owned(), val));
        self
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for inp in &self.inputs {
            push_str(&mut out, 1, inp);
        }
        for o in &self.outputs {
            push_str(&mut out, 2, o);
        }
        if !self.name.is_empty() {
            push_str(&mut out, 3, &self.name);
        }
        push_str(&mut out, 4, &self.op_type);
        for (k, v) in &self.attrs {
            let ab = encode_attr(k, v);
            push_bytes(&mut out, 5, &ab);
        }
        out
    }
}

/// Fluent builder for an ONNX `ModelProto` byte vector.
pub struct ModelBuilder {
    name: String,
    opset: i64,
    inputs: Vec<ValueInfo>,
    outputs: Vec<ValueInfo>,
    initializers: Vec<Vec<u8>>,
    nodes: Vec<Vec<u8>>,
}

impl ModelBuilder {
    pub fn new(name: &str) -> Self {
        Self { name: name.to_owned(), opset: 18, inputs: Vec::new(), outputs: Vec::new(), initializers: Vec::new(), nodes: Vec::new() }
    }

    pub fn opset(mut self, opset: i64) -> Self {
        self.opset = opset;
        self
    }

    pub fn input(mut self, name: &str, elem: ElemType, dims: Vec<Dim>) -> Self {
        self.inputs.push(ValueInfo { name: name.to_owned(), elem_type: Some(elem), dims });
        self
    }

    pub fn output(mut self, name: &str, elem: ElemType, dims: Vec<Dim>) -> Self {
        self.outputs.push(ValueInfo { name: name.to_owned(), elem_type: Some(elem), dims });
        self
    }

    pub fn initializer(mut self, name: &str, tensor: &Tensor) -> Self {
        self.initializers.push(encode_tensor_proto(name, tensor));
        self
    }

    pub fn raw_initializer(mut self, encoded_tensor_proto: Vec<u8>) -> Self {
        self.initializers.push(encoded_tensor_proto);
        self
    }

    pub fn node(mut self, node: NodeBuilder) -> Self {
        self.nodes.push(node.encode());
        self
    }

    pub fn build(self) -> Vec<u8> {
        let mut graph = Vec::new();
        for n in &self.nodes {
            push_bytes(&mut graph, 1, n);
        }
        push_str(&mut graph, 2, &self.name);
        for init in &self.initializers {
            push_bytes(&mut graph, 5, init);
        }
        for inp in &self.inputs {
            push_bytes(&mut graph, 11, &encode_value_info(inp));
        }
        for o in &self.outputs {
            push_bytes(&mut graph, 12, &encode_value_info(o));
        }

        let mut opset_proto = Vec::new();
        push_str(&mut opset_proto, 1, "");
        push_varint_field(&mut opset_proto, 2, self.opset as u64);

        let mut model = Vec::new();
        push_varint_field(&mut model, 1, 9); // IR version 9
        push_str(&mut model, 2, "photocraft-ml");
        push_bytes(&mut model, 7, &graph);
        push_bytes(&mut model, 8, &opset_proto);
        model
    }
}
