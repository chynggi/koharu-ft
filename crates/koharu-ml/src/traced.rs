//! The weights of a TorchScript archive, under the names its `state_dict()`
//! gives them, for a model only published as a trace. The C API here runs no
//! TorchScript, so a port of the traced network takes its weights from here.
//!
//! An archive is a zip holding `data.pkl`, the pickled module tree, and each
//! tensor's storage as `data/<key>`. The pickle is read by the few opcodes
//! `torch.jit.save` writes for weights; no pickle crate reads PyTorch's
//! persistent storage references, so this does.
//! https://github.com/pytorch/pytorch/blob/v2.5.1/torch/csrc/jit/serialization/pickler.cpp

use std::{collections::HashMap, io::Read, path::Path};

use anyhow::{Context as _, Result, bail, ensure};
use safetensors::{Dtype, tensor::TensorView};

/// One value the pickle built, or a container still being filled; containers
/// are shared through the memo, so values live in an arena by index.
#[derive(Clone, Debug)]
enum Value {
    Mark,
    None,
    Bool,
    Int(i64),
    Text(String),
    Global(String),
    Tuple(Vec<usize>),
    Dict(Vec<(usize, usize)>),
    Object { state: Option<usize> },
    Storage { dtype: Dtype, key: String },
    Tensor(Tensor),
}

#[derive(Clone, Debug)]
struct Tensor {
    dtype: Dtype,
    key: String,
    offset: usize,
    shape: Vec<usize>,
    stride: Vec<usize>,
}

/// Every tensor of the archive at `path`, by its `state_dict()` name, saved
/// as safetensors to `output`, which appears only once whole.
pub(crate) fn convert(path: &Path, output: &Path) -> Result<()> {
    let mut archive = zip::ZipArchive::new(fs_err::File::open(path)?)?;
    let pickle = archive
        .file_names()
        .filter_map(|name| name.ok())
        .find(|name| name.ends_with("/data.pkl") && name.matches('/').count() == 1)
        .context("the archive has no data.pkl")?
        .into_owned();
    let folder = pickle.trim_end_matches("data.pkl").to_owned();
    let mut bytes = Vec::new();
    archive.by_name(&pickle)?.read_to_end(&mut bytes)?;
    let (values, root) = unpickle(&bytes)?;

    let mut tensors = Vec::new();
    collect(&values, root, "", &mut tensors)?;
    let mut storages = HashMap::<String, Vec<u8>>::new();
    let mut data = Vec::with_capacity(tensors.len());
    for (name, tensor) in &tensors {
        if !storages.contains_key(&tensor.key) {
            let mut storage = Vec::new();
            archive
                .by_name(&format!("{folder}data/{}", tensor.key))?
                .read_to_end(&mut storage)?;
            storages.insert(tensor.key.clone(), storage);
        }
        let element = tensor.dtype.bitsize() / 8;
        let count: usize = tensor.shape.iter().product();
        // Every weight of a traced module is laid out contiguously.
        let mut expected = 1;
        for (size, stride) in tensor.shape.iter().zip(&tensor.stride).rev() {
            ensure!(
                *size == 1 || *stride == expected,
                "`{name}` isn't contiguous"
            );
            expected *= size;
        }
        let start = tensor.offset * element;
        let storage = &storages[&tensor.key];
        let bytes = storage
            .get(start..start + count * element)
            .with_context(|| format!("`{name}` runs past its storage"))?;
        data.push((
            name.clone(),
            tensor.dtype,
            tensor.shape.clone(),
            bytes.to_vec(),
        ));
    }
    let views = data
        .iter()
        .map(|(name, dtype, shape, bytes)| {
            Ok((name.clone(), TensorView::new(*dtype, shape.clone(), bytes)?))
        })
        .collect::<Result<Vec<_>>>()?;
    let partial = output.with_extension("partial");
    safetensors::serialize_to_file(views, None, &partial)?;
    fs_err::rename(&partial, output)?;
    Ok(())
}

/// The tensors under the object at `at`, its attributes in pickled order.
fn collect(
    values: &[Value],
    at: usize,
    prefix: &str,
    tensors: &mut Vec<(String, Tensor)>,
) -> Result<()> {
    let Value::Object { state: Some(state) } = &values[at] else {
        return Ok(());
    };
    let Value::Dict(items) = &values[*state] else {
        bail!("a module's state isn't a dict");
    };
    for (key, value) in items {
        let Value::Text(key) = &values[*key] else {
            bail!("a module's attribute isn't named");
        };
        let name = if prefix.is_empty() {
            key.clone()
        } else {
            format!("{prefix}.{key}")
        };
        match &values[*value] {
            Value::Object { .. } => collect(values, *value, &name, tensors)?,
            Value::Tensor(tensor) => tensors.push((name, tensor.clone())),
            _ => {}
        }
    }
    Ok(())
}

/// The arena of values `bytes` builds and the index of its result.
fn unpickle(bytes: &[u8]) -> Result<(Vec<Value>, usize)> {
    let mut reader = Reader { bytes, at: 0 };
    let mut values = Vec::new();
    let mut stack = Vec::<usize>::new();
    let mut memo = HashMap::<u32, usize>::new();
    let push = |values: &mut Vec<Value>, stack: &mut Vec<usize>, value| {
        values.push(value);
        stack.push(values.len() - 1);
    };
    let pop = |stack: &mut Vec<usize>| stack.pop().context("the pickle's stack is empty");
    let to_mark = |values: &[Value], stack: &mut Vec<usize>| -> Result<Vec<usize>> {
        let mark = stack
            .iter()
            .rposition(|index| matches!(values[*index], Value::Mark))
            .context("the pickle has no mark")?;
        let items = stack.split_off(mark + 1);
        stack.pop();
        Ok(items)
    };
    loop {
        let opcode = reader.take(1)?[0];
        match opcode {
            // PROTO
            0x80 => {
                reader.take(1)?;
            }
            // GLOBAL
            b'c' => {
                let module = reader.line()?;
                let name = reader.line()?;
                push(
                    &mut values,
                    &mut stack,
                    Value::Global(format!("{module}.{name}")),
                );
            }
            // BINPUT, LONG_BINPUT
            b'q' | b'r' => {
                let index = if opcode == b'q' {
                    u32::from(reader.take(1)?[0])
                } else {
                    u32::from_le_bytes(reader.take(4)?.try_into()?)
                };
                memo.insert(index, *stack.last().context("nothing to memoize")?);
            }
            // BINGET, LONG_BINGET
            b'h' | b'j' => {
                let index = if opcode == b'h' {
                    u32::from(reader.take(1)?[0])
                } else {
                    u32::from_le_bytes(reader.take(4)?.try_into()?)
                };
                stack.push(*memo.get(&index).context("the pickle reads an unset memo")?);
            }
            // EMPTY_TUPLE, EMPTY_DICT
            b')' => push(&mut values, &mut stack, Value::Tuple(Vec::new())),
            b'}' => push(&mut values, &mut stack, Value::Dict(Vec::new())),
            // MARK
            b'(' => push(&mut values, &mut stack, Value::Mark),
            // TUPLE
            b't' => {
                let items = to_mark(&values, &mut stack)?;
                push(&mut values, &mut stack, Value::Tuple(items));
            }
            // BINUNICODE
            b'X' => {
                let length = u32::from_le_bytes(reader.take(4)?.try_into()?) as usize;
                let text = std::str::from_utf8(reader.take(length)?)?.to_owned();
                push(&mut values, &mut stack, Value::Text(text));
            }
            // NEWTRUE, NEWFALSE, NONE
            0x88 | 0x89 => push(&mut values, &mut stack, Value::Bool),
            b'N' => push(&mut values, &mut stack, Value::None),
            // BININT1, BININT2, BININT
            b'K' => {
                let value = i64::from(reader.take(1)?[0]);
                push(&mut values, &mut stack, Value::Int(value));
            }
            b'M' => {
                let value = i64::from(u16::from_le_bytes(reader.take(2)?.try_into()?));
                push(&mut values, &mut stack, Value::Int(value));
            }
            b'J' => {
                let value = i64::from(i32::from_le_bytes(reader.take(4)?.try_into()?));
                push(&mut values, &mut stack, Value::Int(value));
            }
            // NEWOBJ: a module of the class; its arguments are always empty.
            0x81 => {
                pop(&mut stack)?;
                pop(&mut stack)?;
                push(&mut values, &mut stack, Value::Object { state: None });
            }
            // SETITEMS
            b'u' => {
                let items = to_mark(&values, &mut stack)?;
                let dict = *stack.last().context("no dict to set items in")?;
                let Value::Dict(entries) = &mut values[dict] else {
                    bail!("the pickle sets items in a non-dict");
                };
                entries.extend(items.chunks_exact(2).map(|pair| (pair[0], pair[1])));
            }
            // BUILD
            b'b' => {
                let state = pop(&mut stack)?;
                let object = *stack.last().context("no object to build")?;
                let Value::Object { state: slot } = &mut values[object] else {
                    bail!("the pickle builds a non-object");
                };
                *slot = Some(state);
            }
            // BINPERSID: ('storage', class, key, location, numel)
            b'Q' => {
                let id = pop(&mut stack)?;
                let Value::Tuple(parts) = &values[id] else {
                    bail!("a persistent id isn't a tuple");
                };
                let (Some(Value::Global(class)), Some(Value::Text(key))) = (
                    parts.get(1).map(|i| &values[*i]),
                    parts.get(2).map(|i| &values[*i]),
                ) else {
                    bail!("a persistent id isn't a storage");
                };
                let dtype = match class.as_str() {
                    "torch.FloatStorage" => Dtype::F32,
                    "torch.HalfStorage" => Dtype::F16,
                    "torch.LongStorage" => Dtype::I64,
                    other => bail!("{other} isn't a storage this reads"),
                };
                let key = key.clone();
                push(&mut values, &mut stack, Value::Storage { dtype, key });
            }
            // REDUCE
            b'R' => {
                let arguments = pop(&mut stack)?;
                let callable = pop(&mut stack)?;
                let Value::Global(callable) = &values[callable] else {
                    bail!("the pickle calls a non-global");
                };
                let value = match callable.as_str() {
                    "collections.OrderedDict" => Value::Dict(Vec::new()),
                    "torch._utils._rebuild_tensor_v2" => {
                        let Value::Tuple(arguments) = &values[arguments] else {
                            bail!("a tensor's arguments aren't a tuple");
                        };
                        let int = |index: usize| match &values[index] {
                            Value::Int(value) => usize::try_from(*value).ok(),
                            _ => None,
                        };
                        let ints = |index: usize| match &values[index] {
                            Value::Tuple(items) => items.iter().map(|i| int(*i)).collect(),
                            _ => None,
                        };
                        let Value::Storage { dtype, key } = &values[arguments[0]] else {
                            bail!("a tensor isn't over a storage");
                        };
                        Value::Tensor(Tensor {
                            dtype: *dtype,
                            key: key.clone(),
                            offset: int(arguments[1]).context("a tensor's offset")?,
                            shape: ints(arguments[2]).context("a tensor's shape")?,
                            stride: ints(arguments[3]).context("a tensor's stride")?,
                        })
                    }
                    other => bail!("the pickle calls {other}, which this doesn't"),
                };
                push(&mut values, &mut stack, value);
            }
            // STOP
            b'.' => {
                let root = pop(&mut stack)?;
                return Ok((values, root));
            }
            other => bail!("pickle opcode {other:#04x} isn't one a trace's weights use"),
        }
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let taken = self
            .bytes
            .get(self.at..self.at + count)
            .context("the pickle ends early")?;
        self.at += count;
        Ok(taken)
    }

    /// Text up to the next newline, which is skipped.
    fn line(&mut self) -> Result<String> {
        let end = self.bytes[self.at..]
            .iter()
            .position(|&byte| byte == b'\n')
            .context("a global's name isn't ended")?;
        let text = std::str::from_utf8(self.take(end)?)?.to_owned();
        self.take(1)?;
        Ok(text)
    }
}
