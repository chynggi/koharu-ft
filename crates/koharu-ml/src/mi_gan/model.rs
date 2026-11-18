//! Inference-only port of MI-GAN's generator, at 512 pixels a side.
//!
//! Original implementation:
//! https://github.com/Picsart-AI-Research/MI-GAN/blob/2b793c5ece43f4253e32d4afc257120a5deed6f5/lib/model_zoo/migan_inference.py

use std::path::Path;

use anyhow::Result;
use koharu_torch::{Device, Kind, Tensor};

use crate::nn::{self, Module};

/// The side the generator takes; `Upsample2d`'s `filter_const` and the noise
/// are made at it.
pub(super) const RESOLUTION: i64 = 512;
/// `ch_base` and `ch_max`.
const CHANNEL_BASE: i64 = 32768;
const CHANNEL_MAX: i64 = 512;

fn channels(resolution: i64) -> i64 {
    (CHANNEL_BASE / resolution).min(CHANNEL_MAX)
}

/// `lrelu_agc(alpha=0.2, gain='sqrt_2', clamp=256)`, every block's activation.
fn activation(x: &Tensor) -> koharu_torch::Result<Tensor> {
    x.leaky_relu(0.2)?
        .mul_scalar(std::f64::consts::SQRT_2)?
        .clamp(Some((-256.0).into()), Some(256.0.into()))
}

#[derive(Debug)]
pub(super) struct Model {
    vs: nn::VarStore,
    generator: Generator,
}

impl Model {
    pub(super) fn new(device: Device) -> Result<Self> {
        // IOPaint runs it in float32; the network is small enough not to need
        // less.
        let vs = nn::VarStore::new(device, Kind::Float);
        let generator = Generator::new(&vs.root())?;
        Ok(Self { vs, generator })
    }

    pub(super) fn load(&mut self, path: impl AsRef<Path>) -> Result<()> {
        self.vs.load(path)?;
        Ok(())
    }

    /// `Generator.forward`: the four-channel `0.5 - mask` and erased image to
    /// the image, both in [-1, 1].
    pub(super) fn forward(&self, input: &Tensor) -> koharu_torch::Result<Tensor> {
        let (x, features) = self.generator.encoder.forward(input)?;
        self.generator.synthesis.forward(&x, &features)
    }
}

#[derive(Debug)]
struct Generator {
    synthesis: Synthesis,
    encoder: Encoder,
}

impl Generator {
    fn new(path: &nn::Path<'_>) -> Result<Self> {
        Ok(Self {
            synthesis: Synthesis::new(&(path / "synthesis"))?,
            encoder: Encoder::new(&(path / "encoder"))?,
        })
    }
}

#[derive(Debug)]
struct Downsample2d {
    filter: nn::Conv2D,
}

impl Downsample2d {
    fn new(path: &nn::Path<'_>, channels: i64) -> Result<Self> {
        Ok(Self {
            filter: nn::conv2d(
                &(path / "filter"),
                channels,
                channels,
                4,
                nn::ConvConfig {
                    stride: 2,
                    padding: 1,
                    groups: channels,
                    bias: false,
                    ..Default::default()
                },
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> koharu_torch::Result<Tensor> {
        self.filter.forward(x)
    }
}

#[derive(Debug)]
struct Upsample2d {
    filter_const: Tensor,
    filter: nn::Conv2D,
}

impl Upsample2d {
    fn new(path: &nn::Path<'_>, channels: i64, resolution: i64) -> Result<Self> {
        Ok(Self {
            filter_const: path.var("filter_const", &[1, 1, resolution, resolution])?,
            filter: nn::conv2d(
                &(path / "filter"),
                channels,
                channels,
                4,
                nn::ConvConfig {
                    groups: channels,
                    bias: false,
                    ..Default::default()
                },
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> koharu_torch::Result<Tensor> {
        let size = x.size();
        let x = x
            .upsample_nearest2d([size[2] * 2, size[3] * 2], None, None)?
            .mul_tensor(&self.filter_const)?
            .constant_pad_nd([2, 1, 2, 1], 0.0)?;
        self.filter.forward(&x)
    }
}

#[derive(Debug)]
struct SeparableConv2d {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    downsample: Option<Downsample2d>,
    upsample: Option<Upsample2d>,
    /// `noise_const` and `noise_strength`.
    noise: Option<(Tensor, Tensor)>,
}

#[derive(Default)]
struct Separable {
    resolution: i64,
    use_noise: bool,
    down: bool,
    up: bool,
}

impl SeparableConv2d {
    /// Kernel 3, with bias and the activation, as every block makes it.
    fn new(
        path: &nn::Path<'_>,
        in_channels: i64,
        out_channels: i64,
        options: Separable,
    ) -> Result<Self> {
        Ok(Self {
            conv1: nn::conv2d(
                &(path / "conv1"),
                in_channels,
                in_channels,
                3,
                nn::ConvConfig {
                    padding: 1,
                    groups: in_channels,
                    ..Default::default()
                },
            )?,
            conv2: nn::conv2d(
                &(path / "conv2"),
                in_channels,
                out_channels,
                1,
                nn::ConvConfig {
                    bias: false,
                    ..Default::default()
                },
            )?,
            downsample: options
                .down
                .then(|| Downsample2d::new(&(path / "downsample"), in_channels))
                .transpose()?,
            upsample: options
                .up
                .then(|| Upsample2d::new(&(path / "upsample"), out_channels, options.resolution))
                .transpose()?,
            noise: options
                .use_noise
                .then(|| {
                    anyhow::Ok((
                        path.var("noise_const", &[options.resolution, options.resolution])?,
                        path.var("noise_strength", &[])?,
                    ))
                })
                .transpose()?,
        })
    }

    fn forward(&self, x: &Tensor) -> koharu_torch::Result<Tensor> {
        let mut x = activation(&self.conv1.forward(x)?)?;
        if let Some(downsample) = &self.downsample {
            x = downsample.forward(&x)?;
        }
        x = self.conv2.forward(&x)?;
        if let Some(upsample) = &self.upsample {
            x = upsample.forward(&x)?;
        }
        if let Some((noise_const, noise_strength)) = &self.noise {
            x = x.add_tensor(&noise_const.mul_tensor(noise_strength)?, 1)?;
        }
        activation(&x)
    }
}

#[derive(Debug)]
struct EncoderBlock {
    fromrgb: Option<nn::Conv2D>,
    conv1: SeparableConv2d,
    conv2: SeparableConv2d,
}

impl EncoderBlock {
    fn new(
        path: &nn::Path<'_>,
        in_channels: i64,
        out_channels: i64,
        rgb: Option<i64>,
        down: bool,
    ) -> Result<Self> {
        Ok(Self {
            fromrgb: rgb
                .map(|rgb| nn::conv2d(&(path / "fromrgb"), rgb, in_channels, 1, Default::default()))
                .transpose()?,
            conv1: SeparableConv2d::new(
                &(path / "conv1"),
                in_channels,
                in_channels,
                Separable::default(),
            )?,
            conv2: SeparableConv2d::new(
                &(path / "conv2"),
                in_channels,
                out_channels,
                Separable {
                    down,
                    ..Separable::default()
                },
            )?,
        })
    }

    /// The block's output and the feature the synthesis adds back.
    fn forward(&self, x: Option<Tensor>, image: &Tensor) -> koharu_torch::Result<(Tensor, Tensor)> {
        let x = match (&self.fromrgb, x) {
            (Some(fromrgb), x) => {
                let y = activation(&fromrgb.forward(image)?)?;
                match x {
                    Some(x) => x.add_tensor(&y, 1)?,
                    None => y,
                }
            }
            (None, Some(x)) => x,
            (None, None) => unreachable!("the first block reads the image"),
        };
        let feature = self.conv1.forward(&x)?;
        Ok((self.conv2.forward(&feature)?, feature))
    }
}

#[derive(Debug)]
struct Encoder {
    /// From 512 down to 8, each halving the side.
    blocks: Vec<(i64, EncoderBlock)>,
    b4: EncoderBlock,
}

impl Encoder {
    fn new(path: &nn::Path<'_>) -> Result<Self> {
        let mut blocks = Vec::new();
        let mut resolution = RESOLUTION;
        while resolution > 4 {
            let rgb = (resolution == RESOLUTION).then_some(4);
            let block = EncoderBlock::new(
                &(path / format!("b{resolution}")),
                channels(resolution),
                channels(resolution / 2),
                rgb,
                true,
            )?;
            blocks.push((resolution, block));
            resolution /= 2;
        }
        let b4 = EncoderBlock::new(&(path / "b4"), channels(4), channels(4), None, false)?;
        Ok(Self { blocks, b4 })
    }

    /// The encoding and each block's feature, by side, 4 last.
    fn forward(&self, image: &Tensor) -> koharu_torch::Result<(Tensor, Vec<Tensor>)> {
        let mut x = None;
        let mut features = Vec::with_capacity(self.blocks.len() + 1);
        for (_, block) in &self.blocks {
            let (next, feature) = block.forward(x, image)?;
            x = Some(next);
            features.push(feature);
        }
        let (x, feature) = self.b4.forward(x, image)?;
        features.push(feature);
        Ok((x, features))
    }
}

#[derive(Debug)]
struct SynthesisBlockFirst {
    conv1: SeparableConv2d,
    conv2: SeparableConv2d,
    torgb: nn::Conv2D,
}

#[derive(Debug)]
struct SynthesisBlock {
    conv1: SeparableConv2d,
    conv2: SeparableConv2d,
    torgb: nn::Conv2D,
    upsample: Upsample2d,
}

#[derive(Debug)]
struct Synthesis {
    b4: SynthesisBlockFirst,
    /// From 8 up to 512.
    blocks: Vec<SynthesisBlock>,
}

impl Synthesis {
    fn new(path: &nn::Path<'_>) -> Result<Self> {
        let channels4 = channels(4);
        let b4 = path / "b4";
        let b4 = SynthesisBlockFirst {
            conv1: SeparableConv2d::new(
                &(&b4 / "conv1"),
                channels4,
                channels4,
                Separable::default(),
            )?,
            conv2: SeparableConv2d::new(
                &(&b4 / "conv2"),
                channels4,
                channels4,
                Separable::default(),
            )?,
            torgb: nn::conv2d(&(&b4 / "torgb"), channels4, 3, 1, Default::default())?,
        };
        let mut blocks = Vec::new();
        let mut resolution = 8;
        while resolution <= RESOLUTION {
            let path = path / format!("b{resolution}");
            let (in_channels, out_channels) = (channels(resolution / 2), channels(resolution));
            let noisy = |up| Separable {
                resolution,
                use_noise: true,
                up,
                ..Separable::default()
            };
            blocks.push(SynthesisBlock {
                conv1: SeparableConv2d::new(
                    &(&path / "conv1"),
                    in_channels,
                    out_channels,
                    noisy(true),
                )?,
                conv2: SeparableConv2d::new(
                    &(&path / "conv2"),
                    out_channels,
                    out_channels,
                    noisy(false),
                )?,
                torgb: nn::conv2d(&(&path / "torgb"), out_channels, 3, 1, Default::default())?,
                upsample: Upsample2d::new(&(&path / "upsample"), 3, resolution)?,
            });
            resolution *= 2;
        }
        Ok(Self { b4, blocks })
    }

    /// `features` as the encoder gives them, 512 first and 4 last.
    fn forward(&self, x: &Tensor, features: &[Tensor]) -> koharu_torch::Result<Tensor> {
        let mut features = features.iter().rev();
        let mut next = || features.next().expect("a feature for each block");
        let x = self.b4.conv1.forward(x)?.add_tensor(next(), 1)?;
        let mut x = self.b4.conv2.forward(&x)?;
        let mut image = self.b4.torgb.forward(&x)?;
        for block in &self.blocks {
            let y = block.conv1.forward(&x)?.add_tensor(next(), 1)?;
            x = block.conv2.forward(&y)?;
            image = block
                .upsample
                .forward(&image)?
                .add_tensor(&block.torgb.forward(&x)?, 1)?;
        }
        Ok(image)
    }
}
