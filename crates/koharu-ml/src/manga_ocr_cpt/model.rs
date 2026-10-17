//! Inference-only Manga OCR CPT-flex network.
//!
//! Port of `MangaOcrNet`: a ViT-B/16 encoder and a 2-layer post-LN BERT
//! decoder with cross-attention whose output layer is tied to the input
//! embedding. Parameter paths follow the checkpoint (`enc.*`, `dec.*`).
//! https://github.com/muscgab/manga-ocr-CPT/blob/2f6ceefbb3f43a9591f05480b4b17cd54453cc81/manga_ocr_cpt/model.py

use std::path::Path;

use anyhow::Result;
use koharu_torch::{
    Device, Kind, Tensor,
    nn::{self, Module},
};

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
    pub(super) fn new(device: Device) -> Self {
        let mut vs = nn::VarStore::new(device);
        crate::backend::set_precision(&mut vs);
        let root = vs.root();
        let encoder = Encoder::new(&(&root / "enc"));
        let decoder = Decoder::new(&(&root / "dec"));
        vs.freeze();
        Self {
            vs,
            encoder,
            decoder,
        }
    }

    pub(super) fn load(&mut self, path: impl AsRef<Path>) -> Result<()> {
        self.vs.load(path)?;
        // Deliberate divergence: upstream keeps the decoder in fp32 on CUDA
        // and only runs encoder matmuls in fp16. Koharu applies its shared
        // precision policy to the whole store; the equivalence gate runs on
        // the CPU in fp32.
        crate::backend::set_precision(&mut self.vs);
        Ok(())
    }

    /// Upstream `greedy` for one crop. Returns the ids before the end token.
    pub(super) fn generate(&self, input: &ImageInput) -> Result<Vec<i64>> {
        let kind = self.vs.kind();
        let device = input.patches.device();
        let (memory, memory_mask) =
            self.encoder
                .forward(&input.patches.to_kind(kind), &input.valid, input.grid);
        let cross = self.decoder.cross_key_values(&memory);
        let mut cache: Vec<Option<(Tensor, Tensor)>> = (0..DECODER_LAYERS).map(|_| None).collect();
        let mut token = START_TOKEN;
        let mut tokens = Vec::new();
        for position in 0..MAX_NEW_TOKENS {
            token = self
                .decoder
                .step(token, position, &mut cache, &cross, &memory_mask, device);
            if token == END_TOKEN {
                break;
            }
            tokens.push(token);
        }
        Ok(tokens)
    }
}

fn layer_norm(path: nn::Path<'_>) -> nn::LayerNorm {
    nn::layer_norm(
        path,
        vec![HIDDEN_SIZE],
        nn::LayerNormConfig {
            eps: LAYER_NORM_EPS,
            ..Default::default()
        },
    )
}

fn linear(path: nn::Path<'_>, input: i64, output: i64) -> nn::Linear {
    nn::linear(path, input, output, Default::default())
}

/// `(batch, sequence, hidden)` -> `(batch, heads, sequence, head_dim)`.
fn heads(x: &Tensor) -> Tensor {
    let size = x.size();
    x.view([size[0], size[1], NUM_HEADS, HEAD_DIM])
        .transpose(1, 2)
}

fn attend(query: &Tensor, key: &Tensor, value: &Tensor, mask: Option<&Tensor>) -> Tensor {
    let attended = Tensor::scaled_dot_product_attention(
        query,
        key,
        value,
        mask,
        0.0,
        false,
        (HEAD_DIM as f64).powf(-0.5),
        false,
    );
    let size = attended.size();
    attended
        .transpose(1, 2)
        .contiguous()
        .reshape([size[0], size[2], HIDDEN_SIZE])
}

/// Additive SDPA bias over keys: `false` entries become forbidden.
fn additive_mask(keys: &Tensor, kind: Kind) -> Tensor {
    (keys.logical_not().to_kind(Kind::Float) * f32::MIN as f64)
        .to_kind(kind)
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
    fn new(path: &nn::Path<'_>) -> Self {
        Self {
            patch: linear(path / "patch", PATCH_FEATURES, HIDDEN_SIZE),
            cls: path.var("cls", &[HIDDEN_SIZE], nn::Init::Const(0.0)),
            pos: path.var("pos", &[MAX_PATCHES + 1, HIDDEN_SIZE], nn::Init::Const(0.0)),
            layers: (0..ENCODER_LAYERS)
                .map(|index| EncoderLayer::new(&(path / index)))
                .collect(),
            norm: layer_norm(path / "ln"),
        }
    }

    /// Upstream `pe`: the 14x14 learned table bicubically resized to the patch
    /// grid, zero on padded patches.
    fn position_embeddings(&self, [grid_h, grid_w]: [i64; 2]) -> Tensor {
        let device = self.pos.device();
        let table = self
            .pos
            .narrow(0, 1, MAX_PATCHES)
            .to_kind(Kind::Float)
            .reshape([POSITION_GRID, POSITION_GRID, HIDDEN_SIZE])
            .permute([2, 0, 1])
            .unsqueeze(0);
        let table = if [grid_h, grid_w] == [POSITION_GRID, POSITION_GRID] {
            table
        } else {
            table.upsample_bicubic2d([grid_h, grid_w], false, None::<f64>, None::<f64>)
        };
        let rows = table.squeeze_dim(0).flatten(1, 2).transpose(0, 1);
        let padding = Tensor::zeros(
            [MAX_PATCHES - grid_h * grid_w, HIDDEN_SIZE],
            (Kind::Float, device),
        );
        Tensor::cat(&[rows, padding], 0).to_kind(self.pos.kind())
    }

    /// Returns the memory `(1, 197, hidden)` and its additive key mask.
    fn forward(&self, patches: &Tensor, valid: &Tensor, grid: [i64; 2]) -> (Tensor, Tensor) {
        let device = patches.device();
        let patches = self.patch.forward(patches) + self.position_embeddings(grid).unsqueeze(0);
        let cls = (&self.cls + self.pos.select(0, 0)).view([1, 1, HIDDEN_SIZE]);
        let mut hidden = Tensor::cat(&[cls, patches], 1);
        let keys = Tensor::cat(
            &[
                Tensor::ones([1, 1], (Kind::Bool, device)),
                valid.shallow_clone(),
            ],
            1,
        );
        let mask = additive_mask(&keys, hidden.kind());
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &mask);
        }
        (self.norm.forward(&hidden), mask)
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
    fn new(path: &nn::Path<'_>) -> Self {
        Self {
            ln1: layer_norm(path / "ln1"),
            q: linear(path / "q", HIDDEN_SIZE, HIDDEN_SIZE),
            k: linear(path / "k", HIDDEN_SIZE, HIDDEN_SIZE),
            v: linear(path / "v", HIDDEN_SIZE, HIDDEN_SIZE),
            o: linear(path / "o", HIDDEN_SIZE, HIDDEN_SIZE),
            ln2: layer_norm(path / "ln2"),
            f1: linear(path / "f1", HIDDEN_SIZE, INTERMEDIATE_SIZE),
            f2: linear(path / "f2", INTERMEDIATE_SIZE, HIDDEN_SIZE),
        }
    }

    fn forward(&self, hidden: &Tensor, mask: &Tensor) -> Tensor {
        let normed = self.ln1.forward(hidden);
        let attended = attend(
            &heads(&self.q.forward(&normed)),
            &heads(&self.k.forward(&normed)),
            &heads(&self.v.forward(&normed)),
            Some(mask),
        );
        let hidden = hidden + self.o.forward(&attended);
        // Upstream `F.gelu` is the exact erf form.
        let ffn = self
            .f2
            .forward(&self.f1.forward(&self.ln2.forward(&hidden)).gelu("none"));
        &hidden + ffn
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
    fn new(path: &nn::Path<'_>) -> Self {
        Self {
            word: path.var("word", &[OUTPUT_TOKENS, HIDDEN_SIZE], nn::Init::Const(0.0)),
            pos: path.var(
                "pos",
                &[DECODER_POSITIONS, HIDDEN_SIZE],
                nn::Init::Const(0.0),
            ),
            type0: path.var("type0", &[HIDDEN_SIZE], nn::Init::Const(0.0)),
            norm: layer_norm(path / "ln"),
            layers: (0..DECODER_LAYERS)
                .map(|index| DecoderLayer::new(&(path / index)))
                .collect(),
            head: linear(path / "head", HIDDEN_SIZE, HIDDEN_SIZE),
            head_norm: layer_norm(path / "head_ln"),
            out_bias: path.var("out_bias", &[OUTPUT_TOKENS], nn::Init::Const(0.0)),
        }
    }

    /// Upstream `cross_kv`: each layer's cross-attention keys and values,
    /// computed once per crop.
    fn cross_key_values(&self, memory: &Tensor) -> Vec<(Tensor, Tensor)> {
        self.layers
            .iter()
            .map(|layer| {
                (
                    heads(&layer.ck.forward(memory)),
                    heads(&layer.cv.forward(memory)),
                )
            })
            .collect()
    }

    /// Upstream `step`: feeds `token` at `position` and returns the argmax id.
    fn step(
        &self,
        token: i64,
        position: i64,
        cache: &mut [Option<(Tensor, Tensor)>],
        cross: &[(Tensor, Tensor)],
        memory_mask: &Tensor,
        device: Device,
    ) -> i64 {
        let id = Tensor::from_slice(&[token]).to_device(device);
        let embedding = self.word.index_select(0, &id) + self.pos.select(0, position) + &self.type0;
        let mut x = self.norm.forward(&embedding.unsqueeze(0));
        for ((layer, cache), (key, value)) in self.layers.iter().zip(cache.iter_mut()).zip(cross) {
            x = layer.forward(&x, cache, key, value, memory_mask);
        }
        let z = self.head_norm.forward(&self.head.forward(&x).gelu("none"));
        // The output layer is tied to the input embedding.
        let logits = z.matmul(&self.word.transpose(0, 1)) + &self.out_bias;
        logits.argmax(-1, false).int64_value(&[0, 0])
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
    fn new(path: &nn::Path<'_>) -> Self {
        Self {
            q: linear(path / "q", HIDDEN_SIZE, HIDDEN_SIZE),
            k: linear(path / "k", HIDDEN_SIZE, HIDDEN_SIZE),
            v: linear(path / "v", HIDDEN_SIZE, HIDDEN_SIZE),
            o: linear(path / "o", HIDDEN_SIZE, HIDDEN_SIZE),
            ln1: layer_norm(path / "ln1"),
            cq: linear(path / "cq", HIDDEN_SIZE, HIDDEN_SIZE),
            ck: linear(path / "ck", HIDDEN_SIZE, HIDDEN_SIZE),
            cv: linear(path / "cv", HIDDEN_SIZE, HIDDEN_SIZE),
            co: linear(path / "co", HIDDEN_SIZE, HIDDEN_SIZE),
            ln2: layer_norm(path / "ln2"),
            f1: linear(path / "f1", HIDDEN_SIZE, INTERMEDIATE_SIZE),
            f2: linear(path / "f2", INTERMEDIATE_SIZE, HIDDEN_SIZE),
            ln3: layer_norm(path / "ln3"),
        }
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
    ) -> Tensor {
        let key = heads(&self.k.forward(x));
        let value = heads(&self.v.forward(x));
        let (key, value) = match cache.take() {
            Some((past_key, past_value)) => (
                Tensor::cat(&[past_key, key], 2),
                Tensor::cat(&[past_value, value], 2),
            ),
            None => (key, value),
        };
        let attended = attend(&heads(&self.q.forward(x)), &key, &value, None);
        *cache = Some((key, value));
        let x = self.ln1.forward(&(self.o.forward(&attended) + x));
        let attended = attend(
            &heads(&self.cq.forward(&x)),
            cross_key,
            cross_value,
            Some(memory_mask),
        );
        let x = self.ln2.forward(&(self.co.forward(&attended) + &x));
        self.ln3
            .forward(&(self.f2.forward(&self.f1.forward(&x).gelu("none")) + &x))
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use koharu_torch::Device;

    use super::Model;

    #[tokio::test]
    #[ignore = "requires the dynamically loaded LibTorch runtime"]
    async fn model_tree_matches_checkpoint_tensor_shapes() -> Result<()> {
        crate::init().await?;
        let model = Model::new(Device::Cpu);
        let variables = model.vs.variables();
        // The pinned checkpoint holds 260 tensors.
        assert_eq!(variables.len(), 260);
        for (name, shape) in [
            ("enc.patch.weight", vec![768, 768]),
            ("enc.cls", vec![768]),
            ("enc.pos", vec![197, 768]),
            ("enc.11.f1.weight", vec![3072, 768]),
            ("enc.ln.bias", vec![768]),
            ("dec.word", vec![6822, 768]),
            ("dec.pos", vec![512, 768]),
            ("dec.type0", vec![768]),
            ("dec.1.ck.weight", vec![768, 768]),
            ("dec.1.ln3.weight", vec![768]),
            ("dec.head.weight", vec![768, 768]),
            ("dec.head_ln.bias", vec![768]),
            ("dec.out_bias", vec![6822]),
        ] {
            assert_eq!(variables[name].size(), shape, "{name}");
        }
        Ok(())
    }
}
