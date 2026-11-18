//! Inference-only Manga OCR CPT-flex network.
//!
//! Port of `MangaOcrNet`: a ViT-B/16 encoder and a 2-layer post-LN BERT
//! decoder with cross-attention whose output layer is tied to the input
//! embedding. Parameter paths follow the checkpoint (`enc.*`, `dec.*`).
//! https://github.com/muscgab/manga-ocr-CPT/blob/2f6ceefbb3f43a9591f05480b4b17cd54453cc81/manga_ocr_cpt/model.py

use std::path::Path;

use anyhow::Result;
use koharu_torch::{Device, Kind, Tensor};

use crate::nn::{self, Module};

use super::{
    config::{
        DECODER_LAYERS, DECODER_POSITIONS, ENCODER_LAYERS, END_TOKEN, HEAD_DIM, HIDDEN_SIZE,
        INTERMEDIATE_SIZE, LAYER_NORM_EPS, MAX_NEW_TOKENS, MAX_PATCHES, NUM_HEADS, OUTPUT_TOKENS,
        PATCH_FEATURES, POSITION_GRID, START_TOKEN,
    },
    processor::ImageInput,
};

#[derive(Debug)]
pub(super) struct Model {
    vs: nn::VarStore,
    encoder: Encoder,
    decoder: Decoder,
}

impl Model {
    pub(super) fn new(device: Device, kind: Kind) -> Result<Self> {
        let vs = nn::VarStore::new(device, kind);
        let root = vs.root();
        let encoder = Encoder::new(&(&root / "enc"))?;
        let decoder = Decoder::new(&(&root / "dec"))?;
        Ok(Self {
            vs,
            encoder,
            decoder,
        })
    }

    pub(super) fn load(&mut self, path: impl AsRef<Path>) -> Result<()> {
        self.vs.load(path)?;
        Ok(())
    }

    /// Upstream `greedy` for one crop. Returns the ids before the end token.
    pub(super) fn generate(&self, input: &ImageInput) -> Result<Vec<i64>> {
        let kind = self.vs.kind();
        let device = input.patches.device();
        let (memory, memory_mask) =
            self.encoder
                .forward(&input.patches.to_kind(kind)?, &input.valid, input.grid)?;
        let cross = self.decoder.cross_key_values(&memory)?;
        let mut cache: Vec<Option<(Tensor, Tensor)>> = (0..DECODER_LAYERS).map(|_| None).collect();
        let mut token = START_TOKEN;
        let mut tokens = Vec::new();
        for position in 0..MAX_NEW_TOKENS {
            token = self
                .decoder
                .step(token, position, &mut cache, &cross, &memory_mask, device)?;
            if token == END_TOKEN {
                break;
            }
            tokens.push(token);
        }
        Ok(tokens)
    }
}

fn layer_norm(path: nn::Path<'_>) -> koharu_torch::Result<nn::LayerNorm> {
    nn::layer_norm(
        path,
        vec![HIDDEN_SIZE],
        nn::LayerNormConfig {
            eps: LAYER_NORM_EPS,
            ..Default::default()
        },
    )
}

fn linear(path: nn::Path<'_>, input: i64, output: i64) -> koharu_torch::Result<nn::Linear> {
    nn::linear(path, input, output, Default::default())
}

/// `(batch, sequence, hidden)` -> `(batch, heads, sequence, head_dim)`.
fn heads(x: &Tensor) -> koharu_torch::Result<Tensor> {
    let size = x.size();
    x.view([size[0], size[1], NUM_HEADS, HEAD_DIM])?
        .transpose_int(1, 2)
}

fn attend(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    mask: Option<&Tensor>,
) -> koharu_torch::Result<Tensor> {
    let attended = Tensor::scaled_dot_product_attention(
        query,
        key,
        value,
        mask,
        0.0,
        false,
        Some((HEAD_DIM as f64).powf(-0.5)),
        false,
    )?;
    let size = attended.size();
    attended
        .transpose_int(1, 2)?
        .reshape([size[0], size[2], HIDDEN_SIZE])
}

fn additive_mask(keys: &Tensor, kind: Kind) -> koharu_torch::Result<Tensor> {
    keys.logical_not()?
        .to_kind(Kind::Float)?
        .mul_scalar(f32::MIN as f64)?
        .to_kind(kind)?
        .view([1, 1, 1, -1])
}

#[derive(Debug)]
struct Encoder {
    patch: nn::Linear,
    cls: Tensor,
    pos: Tensor,
    layers: Vec<EncoderLayer>,
    norm: nn::LayerNorm,
}

impl Encoder {
    fn new(path: &nn::Path<'_>) -> Result<Self> {
        Ok(Self {
            patch: linear(path / "patch", PATCH_FEATURES, HIDDEN_SIZE)?,
            cls: path.var("cls", &[HIDDEN_SIZE])?,
            pos: path.var("pos", &[MAX_PATCHES + 1, HIDDEN_SIZE])?,
            layers: (0..ENCODER_LAYERS)
                .map(|index| EncoderLayer::new(&(path / index)))
                .collect::<Result<Vec<_>>>()?,
            norm: layer_norm(path / "ln")?,
        })
    }

    /// Upstream `pe`: the 14x14 learned table bicubically resized to the patch
    /// grid, zero on padded patches.
    fn position_embeddings(&self, [grid_h, grid_w]: [i64; 2]) -> koharu_torch::Result<Tensor> {
        let device = self.pos.device();
        let table = self
            .pos
            .narrow(0, 1, MAX_PATCHES)?
            .to_kind(Kind::Float)?
            .reshape([POSITION_GRID, POSITION_GRID, HIDDEN_SIZE])?
            .permute([2, 0, 1])?
            .unsqueeze(0)?;
        let table = if [grid_h, grid_w] == [POSITION_GRID, POSITION_GRID] {
            table
        } else {
            table.upsample_bicubic2d([grid_h, grid_w], false, None::<f64>, None::<f64>)?
        };
        let rows = table
            .squeeze_dim(0)?
            .flatten_using_ints(1, 2)?
            .transpose_int(0, 1)?;
        let padding = Tensor::zeros(
            [MAX_PATCHES - grid_h * grid_w, HIDDEN_SIZE],
            (Kind::Float, device),
        )?;
        Tensor::cat(&[rows, padding], 0)?.to_kind(self.pos.kind())
    }

    fn forward(
        &self,
        patches: &Tensor,
        valid: &Tensor,
        grid: [i64; 2],
    ) -> Result<(Tensor, Tensor)> {
        let patches = self
            .patch
            .forward(patches)?
            .add_tensor(&self.position_embeddings(grid)?.unsqueeze(0)?, 1)?;
        let cls = self
            .cls
            .add_tensor(&self.pos.select_int(0, 0)?, 1)?
            .view([1, 1, HIDDEN_SIZE])?;
        let mut hidden = Tensor::cat(&[cls, patches], 1)?;
        let keys = Tensor::cat(
            &[
                Tensor::ones([1, 1], (Kind::Bool, valid.device()))?,
                valid.shallow_clone(),
            ],
            1,
        )?;
        let mask = additive_mask(&keys, hidden.kind())?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &mask)?;
        }
        Ok((self.norm.forward(&hidden)?, mask))
    }
}

/// Pre-LN ViT layer.
#[derive(Debug)]
struct EncoderLayer {
    ln1: nn::LayerNorm,
    q: nn::Linear,
    k: nn::Linear,
    v: nn::Linear,
    o: nn::Linear,
    ln2: nn::LayerNorm,
    f1: nn::Linear,
    f2: nn::Linear,
}

impl EncoderLayer {
    fn new(path: &nn::Path<'_>) -> Result<Self> {
        Ok(Self {
            ln1: layer_norm(path / "ln1")?,
            q: linear(path / "q", HIDDEN_SIZE, HIDDEN_SIZE)?,
            k: linear(path / "k", HIDDEN_SIZE, HIDDEN_SIZE)?,
            v: linear(path / "v", HIDDEN_SIZE, HIDDEN_SIZE)?,
            o: linear(path / "o", HIDDEN_SIZE, HIDDEN_SIZE)?,
            ln2: layer_norm(path / "ln2")?,
            f1: linear(path / "f1", HIDDEN_SIZE, INTERMEDIATE_SIZE)?,
            f2: linear(path / "f2", INTERMEDIATE_SIZE, HIDDEN_SIZE)?,
        })
    }

    fn forward(&self, hidden: &Tensor, mask: &Tensor) -> koharu_torch::Result<Tensor> {
        let normed = self.ln1.forward(hidden)?;
        let attended = attend(
            &heads(&self.q.forward(&normed)?)?,
            &heads(&self.k.forward(&normed)?)?,
            &heads(&self.v.forward(&normed)?)?,
            Some(mask),
        )?;
        let hidden = hidden.add_tensor(&self.o.forward(&attended)?, 1)?;
        let ffn = self
            .f2
            .forward(&self.f1.forward(&self.ln2.forward(&hidden)?)?.gelu("none")?)?;
        hidden.add_tensor(&ffn, 1)
    }
}

#[derive(Debug)]
struct Decoder {
    word: Tensor,
    pos: Tensor,
    type0: Tensor,
    norm: nn::LayerNorm,
    layers: Vec<DecoderLayer>,
    head: nn::Linear,
    head_norm: nn::LayerNorm,
    out_bias: Tensor,
}

impl Decoder {
    fn new(path: &nn::Path<'_>) -> Result<Self> {
        Ok(Self {
            word: path.var("word", &[OUTPUT_TOKENS, HIDDEN_SIZE])?,
            pos: path.var("pos", &[DECODER_POSITIONS, HIDDEN_SIZE])?,
            type0: path.var("type0", &[HIDDEN_SIZE])?,
            norm: layer_norm(path / "ln")?,
            layers: (0..DECODER_LAYERS)
                .map(|index| DecoderLayer::new(&(path / index)))
                .collect::<Result<Vec<_>>>()?,
            head: linear(path / "head", HIDDEN_SIZE, HIDDEN_SIZE)?,
            head_norm: layer_norm(path / "head_ln")?,
            out_bias: path.var("out_bias", &[OUTPUT_TOKENS])?,
        })
    }

    /// Upstream `cross_kv`: each layer's cross-attention keys and values,
    /// computed once per crop.
    fn cross_key_values(&self, memory: &Tensor) -> Result<Vec<(Tensor, Tensor)>> {
        self.layers
            .iter()
            .map(|layer| {
                Ok((
                    heads(&layer.ck.forward(memory)?)?,
                    heads(&layer.cv.forward(memory)?)?,
                ))
            })
            .collect()
    }

    fn step(
        &self,
        token: i64,
        position: i64,
        cache: &mut [Option<(Tensor, Tensor)>],
        cross: &[(Tensor, Tensor)],
        memory_mask: &Tensor,
        device: Device,
    ) -> Result<i64> {
        let id = Tensor::from_slice(&[token])?.to(device)?;
        let embedding = self
            .word
            .index_select(0, &id)?
            .add_tensor(&self.pos.select_int(0, position)?, 1)?
            .add_tensor(&self.type0, 1)?;
        let mut x = self.norm.forward(&embedding.unsqueeze(0)?)?;
        for ((layer, cache), (key, value)) in self.layers.iter().zip(cache.iter_mut()).zip(cross) {
            x = layer.forward(&x, cache, key, value, memory_mask)?;
        }
        let z = self
            .head_norm
            .forward(&self.head.forward(&x)?.gelu("none")?)?;
        let logits = z
            .matmul(&self.word.transpose_int(0, 1)?)?
            .add_tensor(&self.out_bias, 1)?;
        Ok(Vec::<i64>::try_from(logits.argmax(-1, false)?.view([-1])?)?[0])
    }
}

/// Post-LN BERT layer with cross-attention.
#[derive(Debug)]
struct DecoderLayer {
    q: nn::Linear,
    k: nn::Linear,
    v: nn::Linear,
    o: nn::Linear,
    ln1: nn::LayerNorm,
    cq: nn::Linear,
    ck: nn::Linear,
    cv: nn::Linear,
    co: nn::Linear,
    ln2: nn::LayerNorm,
    f1: nn::Linear,
    f2: nn::Linear,
    ln3: nn::LayerNorm,
}

impl DecoderLayer {
    fn new(path: &nn::Path<'_>) -> Result<Self> {
        Ok(Self {
            q: linear(path / "q", HIDDEN_SIZE, HIDDEN_SIZE)?,
            k: linear(path / "k", HIDDEN_SIZE, HIDDEN_SIZE)?,
            v: linear(path / "v", HIDDEN_SIZE, HIDDEN_SIZE)?,
            o: linear(path / "o", HIDDEN_SIZE, HIDDEN_SIZE)?,
            ln1: layer_norm(path / "ln1")?,
            cq: linear(path / "cq", HIDDEN_SIZE, HIDDEN_SIZE)?,
            ck: linear(path / "ck", HIDDEN_SIZE, HIDDEN_SIZE)?,
            cv: linear(path / "cv", HIDDEN_SIZE, HIDDEN_SIZE)?,
            co: linear(path / "co", HIDDEN_SIZE, HIDDEN_SIZE)?,
            ln2: layer_norm(path / "ln2")?,
            f1: linear(path / "f1", HIDDEN_SIZE, INTERMEDIATE_SIZE)?,
            f2: linear(path / "f2", INTERMEDIATE_SIZE, HIDDEN_SIZE)?,
            ln3: layer_norm(path / "ln3")?,
        })
    }

    /// One token. The self-attention cache holds only past and current
    /// positions, so it needs no causal mask.
    fn forward(
        &self,
        x: &Tensor,
        cache: &mut Option<(Tensor, Tensor)>,
        cross_key: &Tensor,
        cross_value: &Tensor,
        memory_mask: &Tensor,
    ) -> koharu_torch::Result<Tensor> {
        let key = heads(&self.k.forward(x)?)?;
        let value = heads(&self.v.forward(x)?)?;
        let (key, value) = match cache.take() {
            Some((past_key, past_value)) => (
                Tensor::cat(&[past_key, key], 2)?,
                Tensor::cat(&[past_value, value], 2)?,
            ),
            None => (key, value),
        };
        let attended = attend(&heads(&self.q.forward(x)?)?, &key, &value, None)?;
        *cache = Some((key, value));
        let x = self
            .ln1
            .forward(&self.o.forward(&attended)?.add_tensor(x, 1)?)?;
        let attended = attend(
            &heads(&self.cq.forward(&x)?)?,
            cross_key,
            cross_value,
            Some(memory_mask),
        )?;
        let x = self
            .ln2
            .forward(&self.co.forward(&attended)?.add_tensor(&x, 1)?)?;
        self.ln3.forward(
            &self
                .f2
                .forward(&self.f1.forward(&x)?.gelu("none")?)?
                .add_tensor(&x, 1)?,
        )
    }
}
