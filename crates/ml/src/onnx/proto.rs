//! Safe, depth- and size-bounded Protocol Buffers wire-format reader for ONNX (`onnx.proto3`).

use std::collections::{HashMap, HashSet};

use crate::MlError;
use crate::onnx::{Attr, Dim, ElemType, ExternalResolver, Graph, MAX_GRAPH_NODES, Model, Node, ValueInfo, validate_external_location};
use crate::tensor::{DType, MAX_TENSOR_BYTES, Tensor, checked_numel};

/// Maximum protobuf message nesting depth accepted before rejecting input as hostile.
const MAX_PROTO_DEPTH: usize = 16;

#[derive(Clone, Copy)]
struct WireField<'a> {
    number: u32,
    payload: WirePayload<'a>,
}

#[derive(Clone, Copy)]
enum WirePayload<'a> {
    Varint(u64),
    Fixed64([u8; 8]),
    Bytes(&'a [u8]),
    Fixed32([u8; 4]),
}

struct ProtoReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> ProtoReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn read_varint(&mut self) -> Result<u64, MlError> {
        let mut acc = 0u64;
        let mut shift = 0u32;
        loop {
            let &b = self.buf.get(self.pos).ok_or_else(|| MlError::InvalidModel("truncated protobuf varint".into()))?;
            self.pos += 1;
            if shift >= 64 && b > 1 {
                return Err(MlError::InvalidModel("protobuf varint overflow".into()));
            }
            acc |= u64::from(b & 0x7F) << shift;
            if b & 0x80 == 0 {
                return Ok(acc);
            }
            shift += 7;
            if shift > 63 {
                return Err(MlError::InvalidModel("protobuf varint too long".into()));
            }
        }
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8], MlError> {
        let end = self.pos.checked_add(len).ok_or_else(|| MlError::InvalidModel("protobuf length overflow".into()))?;
        let slice = self.buf.get(self.pos..end).ok_or_else(|| MlError::InvalidModel("truncated protobuf length-delimited field".into()))?;
        self.pos = end;
        Ok(slice)
    }

    fn next_field(&mut self) -> Result<Option<WireField<'a>>, MlError> {
        if self.is_eof() {
            return Ok(None);
        }
        let tag = self.read_varint()?;
        let number = u32::try_from(tag >> 3).map_err(|_| MlError::InvalidModel("protobuf field number overflow".into()))?;
        if number == 0 {
            return Err(MlError::InvalidModel("invalid protobuf field number 0".into()));
        }
        let wire_type = (tag & 0x7) as u8;
        let payload = match wire_type {
            0 => WirePayload::Varint(self.read_varint()?),
            1 => {
                let s = self.read_exact(8)?;
                let mut arr = [0u8; 8];
                arr.copy_from_slice(s);
                WirePayload::Fixed64(arr)
            }
            2 => {
                let len = usize::try_from(self.read_varint()?).map_err(|_| MlError::InvalidModel("protobuf slice length overflow".into()))?;
                WirePayload::Bytes(self.read_exact(len)?)
            }
            5 => {
                let s = self.read_exact(4)?;
                let mut arr = [0u8; 4];
                arr.copy_from_slice(s);
                WirePayload::Fixed32(arr)
            }
            other => return Err(MlError::InvalidModel(format!("unsupported protobuf wire type {other}"))),
        };
        Ok(Some(WireField { number, payload }))
    }
}

fn as_utf8(payload: WirePayload<'_>, ctx: &str) -> Result<String, MlError> {
    match payload {
        WirePayload::Bytes(b) => std::str::from_utf8(b).map(str::to_owned).map_err(|e| MlError::InvalidModel(format!("invalid UTF-8 in {ctx}: {e}"))),
        _ => Err(MlError::InvalidModel(format!("expected bytes in {ctx}"))),
    }
}

fn as_bytes<'a>(payload: WirePayload<'a>, ctx: &str) -> Result<&'a [u8], MlError> {
    match payload {
        WirePayload::Bytes(b) => Ok(b),
        _ => Err(MlError::InvalidModel(format!("expected bytes in {ctx}"))),
    }
}

fn as_varint(payload: WirePayload<'_>, ctx: &str) -> Result<u64, MlError> {
    match payload {
        WirePayload::Varint(v) => Ok(v),
        _ => Err(MlError::InvalidModel(format!("expected varint in {ctx}"))),
    }
}

fn as_f32(payload: WirePayload<'_>, ctx: &str) -> Result<f32, MlError> {
    match payload {
        WirePayload::Fixed32(b) => Ok(f32::from_le_bytes(b)),
        _ => Err(MlError::InvalidModel(format!("expected float32 in {ctx}"))),
    }
}

fn push_varints_i64(payload: WirePayload<'_>, dst: &mut Vec<i64>, max_items: usize) -> Result<(), MlError> {
    match payload {
        WirePayload::Varint(v) => {
            if dst.len() >= max_items {
                return Err(MlError::MemoryLimitExceeded { needed: dst.len() + 1, limit: max_items });
            }
            dst.push(v as i64);
            Ok(())
        }
        WirePayload::Bytes(b) => {
            let mut r = ProtoReader::new(b);
            while !r.is_eof() {
                if dst.len() >= max_items {
                    return Err(MlError::MemoryLimitExceeded { needed: dst.len() + 1, limit: max_items });
                }
                dst.push(r.read_varint()? as i64);
            }
            Ok(())
        }
        _ => Err(MlError::InvalidModel("expected int64/int32 varint or packed slice".into())),
    }
}

fn push_floats(payload: WirePayload<'_>, dst: &mut Vec<f32>, max_items: usize) -> Result<(), MlError> {
    match payload {
        WirePayload::Fixed32(b) => {
            if dst.len() >= max_items {
                return Err(MlError::MemoryLimitExceeded { needed: dst.len() + 1, limit: max_items });
            }
            dst.push(f32::from_le_bytes(b));
            Ok(())
        }
        WirePayload::Bytes(b) => {
            if b.len() % 4 != 0 {
                return Err(MlError::InvalidModel("packed float slice byte length is not a multiple of 4".into()));
            }
            let add = b.len() / 4;
            if dst.len().saturating_add(add) > max_items {
                return Err(MlError::MemoryLimitExceeded { needed: dst.len().saturating_add(add), limit: max_items });
            }
            dst.reserve(add);
            for chunk in b.as_chunks::<4>().0 {
                dst.push(f32::from_le_bytes(*chunk));
            }
            Ok(())
        }
        _ => Err(MlError::InvalidModel("expected float32 or packed float slice".into())),
    }
}

fn push_doubles(payload: WirePayload<'_>, dst: &mut Vec<f64>, max_items: usize) -> Result<(), MlError> {
    match payload {
        WirePayload::Fixed64(b) => {
            if dst.len() >= max_items {
                return Err(MlError::MemoryLimitExceeded { needed: dst.len() + 1, limit: max_items });
            }
            dst.push(f64::from_le_bytes(b));
            Ok(())
        }
        WirePayload::Bytes(b) => {
            if b.len() % 8 != 0 {
                return Err(MlError::InvalidModel("packed double slice byte length is not a multiple of 8".into()));
            }
            let add = b.len() / 8;
            if dst.len().saturating_add(add) > max_items {
                return Err(MlError::MemoryLimitExceeded { needed: dst.len().saturating_add(add), limit: max_items });
            }
            dst.reserve(add);
            for chunk in b.as_chunks::<8>().0 {
                dst.push(f64::from_le_bytes(*chunk));
            }
            Ok(())
        }
        _ => Err(MlError::InvalidModel("expected float64 or packed double slice".into())),
    }
}

/// Parse an ONNX `ModelProto` from `bytes`.
pub fn parse_model(bytes: &[u8], ext: ExternalResolver<'_>) -> Result<Model, MlError> {
    let mut r = ProtoReader::new(bytes);
    let mut ir_version = 7i64;
    let mut opset_version = 13i64;
    let mut producer_name = String::new();
    let mut metadata = HashMap::new();
    let mut graph: Option<Graph> = None;

    while let Some(f) = r.next_field()? {
        match f.number {
            1 => ir_version = as_varint(f.payload, "ModelProto.ir_version")? as i64,
            2 => producer_name = as_utf8(f.payload, "ModelProto.producer_name")?,
            7 => graph = Some(parse_graph(as_bytes(f.payload, "ModelProto.graph")?, ext, 1)?),
            8 => {
                let (domain, ver) = parse_opset(as_bytes(f.payload, "ModelProto.opset_import")?)?;
                if domain.is_empty() || domain == "ai.onnx" {
                    opset_version = ver;
                }
            }
            14 => {
                let (k, v) = parse_string_entry(as_bytes(f.payload, "ModelProto.metadata_props")?)?;
                metadata.insert(k, v);
            }
            _ => {}
        }
    }
    let graph = graph.ok_or_else(|| MlError::InvalidModel("ONNX ModelProto has no graph".into()))?;
    Ok(Model { ir_version, opset_version, producer_name, metadata, graph })
}

fn parse_opset(bytes: &[u8]) -> Result<(String, i64), MlError> {
    let mut r = ProtoReader::new(bytes);
    let mut domain = String::new();
    let mut version = 0i64;
    while let Some(f) = r.next_field()? {
        match f.number {
            1 => domain = as_utf8(f.payload, "OperatorSetIdProto.domain")?,
            2 => version = as_varint(f.payload, "OperatorSetIdProto.version")? as i64,
            _ => {}
        }
    }
    Ok((domain, version))
}

fn parse_string_entry(bytes: &[u8]) -> Result<(String, String), MlError> {
    let mut r = ProtoReader::new(bytes);
    let mut k = String::new();
    let mut v = String::new();
    while let Some(f) = r.next_field()? {
        match f.number {
            1 => k = as_utf8(f.payload, "StringStringEntryProto.key")?,
            2 => v = as_utf8(f.payload, "StringStringEntryProto.value")?,
            _ => {}
        }
    }
    Ok((k, v))
}

fn parse_graph(bytes: &[u8], ext: ExternalResolver<'_>, depth: usize) -> Result<Graph, MlError> {
    if depth > MAX_PROTO_DEPTH {
        return Err(MlError::InvalidModel("ONNX protobuf nesting exceeds depth limit".into()));
    }
    let mut r = ProtoReader::new(bytes);
    let mut name = String::new();
    let mut nodes = Vec::new();
    let mut initializers = HashMap::new();
    let mut raw_inputs = Vec::new();
    let mut outputs = Vec::new();

    while let Some(f) = r.next_field()? {
        match f.number {
            1 => {
                if nodes.len() >= MAX_GRAPH_NODES {
                    return Err(MlError::InvalidModel(format!("ONNX graph exceeds {MAX_GRAPH_NODES} nodes")));
                }
                nodes.push(parse_node(as_bytes(f.payload, "GraphProto.node")?, ext, depth + 1)?);
            }
            2 => name = as_utf8(f.payload, "GraphProto.name")?,
            5 => {
                let (t_name, tensor) = parse_tensor(as_bytes(f.payload, "GraphProto.initializer")?, ext)?;
                initializers.insert(t_name, tensor);
            }
            11 => raw_inputs.push(parse_value_info(as_bytes(f.payload, "GraphProto.input")?)?),
            12 => outputs.push(parse_value_info(as_bytes(f.payload, "GraphProto.output")?)?),
            _ => {}
        }
    }
    let init_names: HashSet<&str> = initializers.keys().map(String::as_str).collect();
    let inputs = raw_inputs.into_iter().filter(|vi| !init_names.contains(vi.name.as_str())).collect();
    Ok(Graph { name, nodes, initializers, inputs, outputs })
}

fn parse_node(bytes: &[u8], ext: ExternalResolver<'_>, depth: usize) -> Result<Node, MlError> {
    if depth > MAX_PROTO_DEPTH {
        return Err(MlError::InvalidModel("ONNX protobuf nesting exceeds depth limit".into()));
    }
    let mut r = ProtoReader::new(bytes);
    let mut name = String::new();
    let mut op_type = String::new();
    let mut domain = String::new();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut attrs = HashMap::new();

    while let Some(f) = r.next_field()? {
        match f.number {
            1 => inputs.push(as_utf8(f.payload, "NodeProto.input")?),
            2 => outputs.push(as_utf8(f.payload, "NodeProto.output")?),
            3 => name = as_utf8(f.payload, "NodeProto.name")?,
            4 => op_type = as_utf8(f.payload, "NodeProto.op_type")?,
            5 => {
                if let Some((k, v)) = parse_attr(as_bytes(f.payload, "NodeProto.attribute")?, ext, depth + 1)? {
                    attrs.insert(k, v);
                }
            }
            7 => domain = as_utf8(f.payload, "NodeProto.domain")?,
            _ => {}
        }
    }
    if op_type.is_empty() {
        return Err(MlError::InvalidModel("ONNX NodeProto missing op_type".into()));
    }
    Ok(Node { name, op_type, domain, inputs, outputs, attrs })
}

fn parse_attr(bytes: &[u8], ext: ExternalResolver<'_>, depth: usize) -> Result<Option<(String, Attr)>, MlError> {
    if depth > MAX_PROTO_DEPTH {
        return Err(MlError::InvalidModel("ONNX protobuf nesting exceeds depth limit".into()));
    }
    let mut r = ProtoReader::new(bytes);
    let mut name = String::new();
    let mut attr_type = 0i64;
    let mut f_val = 0.0f32;
    let mut i_val = 0i64;
    let mut s_val: Option<String> = None;
    let mut t_val: Option<Tensor> = None;
    let mut floats = Vec::new();
    let mut ints = Vec::new();
    let mut strings = Vec::new();

    while let Some(f) = r.next_field()? {
        match f.number {
            1 => name = as_utf8(f.payload, "AttributeProto.name")?,
            2 => f_val = as_f32(f.payload, "AttributeProto.f")?,
            3 => i_val = as_varint(f.payload, "AttributeProto.i")? as i64,
            4 => s_val = Some(String::from_utf8_lossy(as_bytes(f.payload, "AttributeProto.s")?).into_owned()),
            5 => t_val = Some(parse_tensor(as_bytes(f.payload, "AttributeProto.t")?, ext)?.1),
            7 => push_floats(f.payload, &mut floats, 1 << 20)?,
            8 => push_varints_i64(f.payload, &mut ints, 1 << 20)?,
            9 => strings.push(String::from_utf8_lossy(as_bytes(f.payload, "AttributeProto.strings")?).into_owned()),
            20 => attr_type = as_varint(f.payload, "AttributeProto.type")? as i64,
            _ => {}
        }
    }
    let attr = match attr_type {
        1 => Attr::Float(f_val),
        2 => Attr::Int(i_val),
        3 => Attr::String(s_val.unwrap_or_default()),
        4 => Attr::Tensor(t_val.ok_or_else(|| MlError::InvalidModel(format!("attribute `{name}` has type TENSOR but no tensor payload")))?),
        6 => Attr::Floats(floats),
        7 => Attr::Ints(ints),
        8 => Attr::Strings(strings),
        _ => return Ok(None),
    };
    Ok(Some((name, attr)))
}

#[derive(Default)]
struct RawTensorFields<'a> {
    dims: Vec<i64>,
    data_type: i32,
    name: String,
    raw_data: Option<&'a [u8]>,
    float_data: Vec<f32>,
    int32_data: Vec<i64>,
    int64_data: Vec<i64>,
    double_data: Vec<f64>,
    uint64_data: Vec<i64>,
    data_location: i64,
    ext_entries: HashMap<String, String>,
}

fn parse_tensor(bytes: &[u8], ext: ExternalResolver<'_>) -> Result<(String, Tensor), MlError> {
    let mut r = ProtoReader::new(bytes);
    let mut tf = RawTensorFields::default();
    let max_elems = MAX_TENSOR_BYTES / 2;

    while let Some(f) = r.next_field()? {
        match f.number {
            1 => push_varints_i64(f.payload, &mut tf.dims, 16)?,
            2 => tf.data_type = as_varint(f.payload, "TensorProto.data_type")? as i32,
            4 => push_floats(f.payload, &mut tf.float_data, max_elems)?,
            5 => push_varints_i64(f.payload, &mut tf.int32_data, max_elems)?,
            7 => push_varints_i64(f.payload, &mut tf.int64_data, max_elems)?,
            8 => tf.name = as_utf8(f.payload, "TensorProto.name")?,
            9 => tf.raw_data = Some(as_bytes(f.payload, "TensorProto.raw_data")?),
            10 => push_doubles(f.payload, &mut tf.double_data, max_elems)?,
            11 => push_varints_i64(f.payload, &mut tf.uint64_data, max_elems)?,
            13 => {
                let (k, v) = parse_string_entry(as_bytes(f.payload, "TensorProto.external_data")?)?;
                tf.ext_entries.insert(k, v);
            }
            14 => tf.data_location = as_varint(f.payload, "TensorProto.data_location")? as i64,
            _ => {}
        }
    }

    let shape = tf
        .dims
        .iter()
        .map(|&d| usize::try_from(d).map_err(|_| MlError::InvalidShape(format!("negative tensor dimension {d} in `{}`", tf.name))))
        .collect::<Result<Vec<_>, _>>()?;
    let elem = ElemType::from_onnx(tf.data_type)?;
    let numel = checked_numel(&shape, elem.runtime_dtype())?;

    let owned_external: Option<Vec<u8>> = if tf.data_location == 1 { Some(load_external_bytes(&tf.ext_entries, ext)?) } else { None };
    let raw_bytes = owned_external.as_deref().or(tf.raw_data);

    let tensor = decode_tensor_elements(elem, shape, numel, raw_bytes, &tf)?;
    Ok((tf.name, tensor))
}

fn load_external_bytes(entries: &HashMap<String, String>, ext: ExternalResolver<'_>) -> Result<Vec<u8>, MlError> {
    let loc = entries.get("location").ok_or_else(|| MlError::InvalidModel("external_data missing `location`".into()))?;
    let clean_loc = validate_external_location(loc)?;
    let offset = match entries.get("offset") {
        Some(s) => s.parse::<u64>().map_err(|_| MlError::InvalidModel(format!("invalid external_data offset `{s}`")))?,
        None => 0,
    };
    let length = match entries.get("length") {
        Some(s) => {
            let len = s.parse::<u64>().map_err(|_| MlError::InvalidModel(format!("invalid external_data length `{s}`")))?;
            if len > MAX_TENSOR_BYTES as u64 {
                return Err(MlError::MemoryLimitExceeded { needed: usize::try_from(len).unwrap_or(usize::MAX), limit: MAX_TENSOR_BYTES });
            }
            Some(len)
        }
        None => None,
    };
    ext(clean_loc, offset, length)
}

fn decode_tensor_elements(elem: ElemType, shape: Vec<usize>, numel: usize, raw: Option<&[u8]>, tf: &RawTensorFields<'_>) -> Result<Tensor, MlError> {
    if let Some(bytes) = raw {
        return decode_raw_bytes(elem, shape, numel, bytes);
    }
    match elem {
        ElemType::Float => Tensor::from_f32(shape, tf.float_data.clone()),
        ElemType::Float16 => {
            let v: Vec<f32> = tf.int32_data.iter().map(|&bits| half::f16::from_bits(bits as u16).to_f32()).collect();
            Tensor::from_f32(shape, v)
        }
        ElemType::BFloat16 => {
            let v: Vec<f32> = tf.int32_data.iter().map(|&bits| half::bf16::from_bits(bits as u16).to_f32()).collect();
            Tensor::from_f32(shape, v)
        }
        ElemType::Double => {
            let v: Vec<f32> = tf.double_data.iter().map(|&x| x as f32).collect();
            Tensor::from_f32(shape, v)
        }
        ElemType::Int64 => Tensor::from_i64(shape, tf.int64_data.clone()),
        ElemType::Uint64 => Tensor::from_i64(shape, tf.uint64_data.clone()),
        ElemType::Int32 | ElemType::Int16 | ElemType::Int8 | ElemType::Uint16 | ElemType::Uint8 | ElemType::Bool => {
            let v: Vec<i64> = tf.int32_data.iter().map(|&x| i64::from(x as i32)).collect();
            Tensor::from_i64(shape, v)
        }
        ElemType::Uint32 => {
            let v: Vec<i64> = tf.uint64_data.iter().map(|&x| i64::from(x as u32)).collect();
            Tensor::from_i64(shape, v)
        }
        ElemType::String => Tensor::zeros(shape, DType::I64),
    }
}

fn decode_raw_bytes(elem: ElemType, shape: Vec<usize>, numel: usize, bytes: &[u8]) -> Result<Tensor, MlError> {
    let check_len = |elem_size: usize| -> Result<(), MlError> {
        let expected = numel.checked_mul(elem_size).ok_or_else(|| MlError::InvalidModel("tensor byte size overflow".into()))?;
        if bytes.len() != expected {
            return Err(MlError::InvalidModel(format!("raw_data length {} does not match expected {expected} for {:?}", bytes.len(), elem)));
        }
        Ok(())
    };
    match elem {
        ElemType::Float => {
            check_len(4)?;
            let v = bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
            Tensor::from_f32(shape, v)
        }
        ElemType::Float16 => {
            check_len(2)?;
            let v = bytes.as_chunks::<2>().0.iter().map(|c| half::f16::from_le_bytes(*c).to_f32()).collect();
            Tensor::from_f32(shape, v)
        }
        ElemType::BFloat16 => {
            check_len(2)?;
            let v = bytes.as_chunks::<2>().0.iter().map(|c| half::bf16::from_le_bytes(*c).to_f32()).collect();
            Tensor::from_f32(shape, v)
        }
        ElemType::Double => {
            check_len(8)?;
            let v = bytes.as_chunks::<8>().0.iter().map(|c| f64::from_le_bytes(*c) as f32).collect();
            Tensor::from_f32(shape, v)
        }
        ElemType::Int64 | ElemType::Uint64 => {
            check_len(8)?;
            let v = bytes.as_chunks::<8>().0.iter().map(|c| i64::from_le_bytes(*c)).collect();
            Tensor::from_i64(shape, v)
        }
        ElemType::Int32 => {
            check_len(4)?;
            let v = bytes.as_chunks::<4>().0.iter().map(|c| i64::from(i32::from_le_bytes(*c))).collect();
            Tensor::from_i64(shape, v)
        }
        ElemType::Uint32 => {
            check_len(4)?;
            let v = bytes.as_chunks::<4>().0.iter().map(|c| i64::from(u32::from_le_bytes(*c))).collect();
            Tensor::from_i64(shape, v)
        }
        ElemType::Int16 => {
            check_len(2)?;
            let v = bytes.as_chunks::<2>().0.iter().map(|c| i64::from(i16::from_le_bytes(*c))).collect();
            Tensor::from_i64(shape, v)
        }
        ElemType::Uint16 => {
            check_len(2)?;
            let v = bytes.as_chunks::<2>().0.iter().map(|c| i64::from(u16::from_le_bytes(*c))).collect();
            Tensor::from_i64(shape, v)
        }
        ElemType::Int8 => {
            check_len(1)?;
            let v = bytes.iter().map(|&b| i64::from(b as i8)).collect();
            Tensor::from_i64(shape, v)
        }
        ElemType::Uint8 | ElemType::Bool => {
            check_len(1)?;
            let v = bytes.iter().map(|&b| i64::from(b)).collect();
            Tensor::from_i64(shape, v)
        }
        ElemType::String => Tensor::zeros(shape, DType::I64),
    }
}

fn parse_value_info(bytes: &[u8]) -> Result<ValueInfo, MlError> {
    let mut r = ProtoReader::new(bytes);
    let mut name = String::new();
    let mut elem_type = None;
    let mut dims = Vec::new();

    while let Some(f) = r.next_field()? {
        match f.number {
            1 => name = as_utf8(f.payload, "ValueInfoProto.name")?,
            2 => {
                let (et, d) = parse_type_proto(as_bytes(f.payload, "ValueInfoProto.type")?)?;
                elem_type = et;
                dims = d;
            }
            _ => {}
        }
    }
    Ok(ValueInfo { name, elem_type, dims })
}

fn parse_type_proto(bytes: &[u8]) -> Result<(Option<ElemType>, Vec<Dim>), MlError> {
    let mut r = ProtoReader::new(bytes);
    let mut elem_type = None;
    let mut dims = Vec::new();
    while let Some(f) = r.next_field()? {
        if f.number == 1 {
            let mut tr = ProtoReader::new(as_bytes(f.payload, "TypeProto.tensor_type")?);
            while let Some(tf) = tr.next_field()? {
                match tf.number {
                    1 => elem_type = ElemType::from_onnx(as_varint(tf.payload, "TypeProto.Tensor.elem_type")? as i32).ok(),
                    2 => dims = parse_shape_proto(as_bytes(tf.payload, "TypeProto.Tensor.shape")?)?,
                    _ => {}
                }
            }
        }
    }
    Ok((elem_type, dims))
}

fn parse_shape_proto(bytes: &[u8]) -> Result<Vec<Dim>, MlError> {
    let mut r = ProtoReader::new(bytes);
    let mut dims = Vec::new();
    while let Some(f) = r.next_field()? {
        if f.number == 1 {
            let mut dr = ProtoReader::new(as_bytes(f.payload, "TensorShapeProto.dim")?);
            let mut dim = Dim::Symbolic(String::new());
            while let Some(df) = dr.next_field()? {
                match df.number {
                    1 => {
                        let v = as_varint(df.payload, "Dimension.dim_value")? as i64;
                        if let Ok(u) = usize::try_from(v) {
                            dim = Dim::Fixed(u);
                        }
                    }
                    2 => dim = Dim::Symbolic(as_utf8(df.payload, "Dimension.dim_param")?),
                    _ => {}
                }
            }
            dims.push(dim);
        }
    }
    Ok(dims)
}
