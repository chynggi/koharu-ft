//! Inference-only port of the inpaintor IOPaint runs as
//! `manga_inpaintor.jit`: Seamless Manga Inpainting's ScreenVAE encoder,
//! semantic inpainting generator, and manga inpainting generator, chained as
//! the trace's `JITWrapper` chains them.
//!
//! Original implementation:
//! https://github.com/msxie92/MangaInpainting/tree/02b749406e9a88dcea1b98599ce33d6b075a432e/src
//!
//! The trace is the authority where it and the repository differ: its
//! ScreenVAE keeps only the encoder, the generators' spectral norms are its
//! evaluation form (`weight_orig` over `u · W v`, no power iteration), and
//! the attention returns only its output.

use anyhow::Result;
use koharu_torch::{Kind, Tensor};

use crate::nn::{self, Module};

/// `LayerNormWrapper`: layer norm over channels and pixels, without affine
/// parameters.
fn layer_norm(x: &Tensor) -> koharu_torch::Result<Tensor> {
    let size = x.size();
    Tensor::layer_norm(x, &size[1..], None, None, 1e-5, true)
}

/// `nn.UpsamplingBilinear2d(scale_factor=2)` and ScreenVAE's bilinear
/// `nn.Upsample`, both aligning corners.
fn double(x: &Tensor) -> koharu_torch::Result<Tensor> {
    let size = x.size();
    x.upsample_bilinear2d([size[2] * 2, size[3] * 2], true, None, None)
}

/// A convolution under `nn.utils.spectral_norm`, as evaluation runs it.
#[derive(Debug)]
struct Spectral {
    weight_orig: Tensor,
    weight_u: Tensor,
    weight_v: Tensor,
    bias: Tensor,
    /// A transposed convolution's weight is normalized over its output
    /// channels, its second dimension.
    transposed: bool,
}

impl Spectral {
    fn new(
        path: &nn::Path<'_>,
        in_channels: i64,
        out_channels: i64,
        kernel: i64,
        transposed: bool,
    ) -> Result<Self> {
        let shape = if transposed {
            [in_channels, out_channels, kernel, kernel]
        } else {
            [out_channels, in_channels, kernel, kernel]
        };
        Ok(Self {
            weight_orig: path.var("weight_orig", &shape)?,
            weight_u: path.var("weight_u", &[out_channels])?,
            weight_v: path.var("weight_v", &[in_channels * kernel * kernel])?,
            bias: path.var("bias", &[out_channels])?,
            transposed,
        })
    }

    fn weight(&self) -> koharu_torch::Result<Tensor> {
        let matrix = if self.transposed {
            self.weight_orig.permute([1, 0, 2, 3])?
        } else {
            self.weight_orig.alias()?
        };
        let rows = matrix.size()[0];
        let matrix = matrix.reshape([rows, -1])?;
        let sigma = self.weight_u.dot(&matrix.mv(&self.weight_v)?)?;
        self.weight_orig.div_tensor(&sigma)
    }

    /// A convolution with `padding` zeros, or a transposed one of stride 2.
    fn forward(&self, x: &Tensor, stride: i64, padding: i64) -> koharu_torch::Result<Tensor> {
        let weight = self.weight()?;
        if self.transposed {
            Tensor::conv_transpose2d_input(
                x,
                &weight,
                Some(&self.bias),
                [stride, stride],
                [padding, padding],
                [0, 0],
                1,
                [1, 1],
            )
        } else {
            Tensor::conv2d(
                x,
                &weight,
                Some(&self.bias),
                [stride, stride],
                [padding, padding],
                [1, 1],
                1,
            )
        }
    }
}

fn conv(
    path: &nn::Path<'_>,
    in_channels: i64,
    out_channels: i64,
    kernel: i64,
    config: nn::ConvConfig,
) -> Result<nn::Conv2D> {
    Ok(nn::conv2d(path, in_channels, out_channels, kernel, config)?)
}

/// ScreenVAE's `ResnetBlock`: replicate-padded convolutions with layer norm.
#[derive(Debug)]
struct ScreenBlock {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
}

/// ScreenVAE's encoder, `ResnetGenerator(2, 8, 24, layer norm, 6 blocks)`:
/// the screentone the page and its lines suggest.
#[derive(Debug)]
struct ScreenEncoder {
    stem: nn::Conv2D,
    downs: Vec<nn::Conv2D>,
    blocks: Vec<ScreenBlock>,
    /// Each 1x1 after the upsampling and the 3x3 after it.
    ups: Vec<(nn::Conv2D, nn::Conv2D)>,
    head: nn::Conv2D,
}

impl ScreenEncoder {
    fn new(path: &nn::Path<'_>) -> Result<Self> {
        let path = path / "model";
        let ngf = 24;
        let stem = conv(&(&path / 1), 2, ngf, 7, Default::default())?;
        let mut index = 4;
        let mut downs = Vec::new();
        for level in 0..3 {
            let channels = ngf << level;
            let config = nn::ConvConfig {
                stride: 2,
                ..Default::default()
            };
            downs.push(conv(
                &(&path / (index + 1)),
                channels,
                channels * 2,
                3,
                config,
            )?);
            index += 4;
        }
        let width = ngf * 8;
        let mut blocks = Vec::new();
        for _ in 0..6 {
            let block = &path / index / "conv_block";
            blocks.push(ScreenBlock {
                conv1: conv(&(&block / 1), width, width, 3, Default::default())?,
                conv2: conv(&(&block / 5), width, width, 3, Default::default())?,
            });
            index += 1;
        }
        let mut ups = Vec::new();
        for level in 0..3 {
            let channels = width >> level;
            ups.push((
                conv(
                    &(&path / (index + 1)),
                    channels,
                    channels / 2,
                    1,
                    Default::default(),
                )?,
                conv(
                    &(&path / (index + 5)),
                    channels / 2,
                    channels / 2,
                    3,
                    Default::default(),
                )?,
            ));
            index += 8;
        }
        let head = conv(&(&path / (index + 1)), ngf, 8, 7, Default::default())?;
        Ok(Self {
            stem,
            downs,
            blocks,
            ups,
            head,
        })
    }

    fn forward(&self, x: &Tensor) -> koharu_torch::Result<Tensor> {
        let mut x = layer_norm(&self.stem.forward(&x.replication_pad2d([3, 3, 3, 3])?)?)?.relu()?;
        for down in &self.downs {
            x = layer_norm(&down.forward(&x.replication_pad2d([1, 1, 1, 1])?)?)?.relu()?;
        }
        for block in &self.blocks {
            let y =
                layer_norm(&block.conv1.forward(&x.replication_pad2d([1, 1, 1, 1])?)?)?.relu()?;
            let y = layer_norm(&block.conv2.forward(&y.replication_pad2d([1, 1, 1, 1])?)?)?;
            x = x.add_tensor(&y, 1)?;
        }
        for (pointwise, conv) in &self.ups {
            x = layer_norm(&pointwise.forward(&double(&x)?)?)?.relu()?;
            x = layer_norm(&conv.forward(&x.replication_pad2d([1, 1, 1, 1])?)?)?.relu()?;
        }
        self.head.forward(&x.replication_pad2d([3, 3, 3, 3])?)
    }
}

/// The semantic generator's `ResnetBlock(256, 2, use_instance=True)`.
#[derive(Debug)]
struct SemanticBlock {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
}

/// `SemanticInpaintGenerator(7, 4)`: the masked screentone and lines filled.
#[derive(Debug)]
struct SemanticGenerator {
    encoder1: Spectral,
    encoder2: Spectral,
    encoder3: Spectral,
    encoder4: Spectral,
    middle: Vec<SemanticBlock>,
    decoder_scr4: Spectral,
    decoder_scr3: Spectral,
    decoder_scr2: Spectral,
    decoder_scr1: nn::Conv2D,
    decoder_l4: Spectral,
    decoder_l3: Spectral,
    decoder_l2: Spectral,
    decoder_l1: nn::Conv2D,
}

impl SemanticGenerator {
    fn new(path: &nn::Path<'_>) -> Result<Self> {
        let spectral = |name: &str, index, from, to, kernel, transposed| {
            Spectral::new(&(path / name / index), from, to, kernel, transposed)
        };
        let dilated = nn::ConvConfig {
            dilation: 2,
            ..Default::default()
        };
        let middle = (0..6)
            .map(|index| {
                let block = path / "middle" / index / "conv_block";
                Ok(SemanticBlock {
                    conv1: conv(&(&block / 1), 256, 256, 3, dilated)?,
                    conv2: conv(&(&block / 5), 256, 256, 3, Default::default())?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            encoder1: spectral("encoder1", 1, 7, 64, 7, false)?,
            encoder2: spectral("encoder2", 0, 64, 128, 4, false)?,
            encoder3: spectral("encoder3", 0, 128, 256, 4, false)?,
            encoder4: spectral("encoder4", 0, 256, 256, 4, false)?,
            middle,
            decoder_scr4: spectral("decoder_scr4", 1, 256, 256, 3, false)?,
            decoder_scr3: spectral("decoder_scr3", 1, 256, 128, 3, false)?,
            decoder_scr2: spectral("decoder_scr2", 1, 128, 64, 3, false)?,
            decoder_scr1: conv(&(path / "decoder_scr1" / 1), 64, 4, 7, Default::default())?,
            decoder_l4: spectral("decoder_l4", 0, 256, 256, 4, true)?,
            decoder_l3: spectral("decoder_l3", 0, 256, 128, 4, true)?,
            decoder_l2: spectral("decoder_l2", 0, 128, 64, 4, true)?,
            decoder_l1: conv(&(path / "decoder_l1" / 1), 64, 1, 7, Default::default())?,
        })
    }

    /// The screentone and the lines, each in [-1, 1].
    fn forward(&self, input: &Tensor) -> koharu_torch::Result<(Tensor, Tensor)> {
        let down = |layer: &Spectral, x: &Tensor| -> koharu_torch::Result<Tensor> {
            layer_norm(&layer.forward(x, 2, 1)?)?.relu()
        };
        let x1 = layer_norm(&self.encoder1.forward(
            &input.reflection_pad2d([3, 3, 3, 3])?,
            1,
            0,
        )?)?
        .relu()?;
        let x2 = down(&self.encoder2, &x1)?;
        let x3 = down(&self.encoder3, &x2)?;
        let x4 = down(&self.encoder4, &x3)?;
        let mut x = x4.alias()?;
        for block in &self.middle {
            let y =
                layer_norm(&block.conv1.forward(&x.reflection_pad2d([2, 2, 2, 2])?)?)?.relu()?;
            let y = layer_norm(&block.conv2.forward(&y.reflection_pad2d([1, 1, 1, 1])?)?)?;
            x = x.add_tensor(&y, 1)?;
        }

        let up = |layer: &Spectral, x: &Tensor| layer.forward(&double(x)?, 1, 1)?.relu();
        let screen = up(&self.decoder_scr4, &x)?;
        let screen = up(&self.decoder_scr3, &screen)?;
        let screen = up(&self.decoder_scr2, &screen)?;
        let screen = self
            .decoder_scr1
            .forward(&screen.reflection_pad2d([3, 3, 3, 3])?)?
            .clamp(Some((-1.0).into()), Some(1.0.into()))?;

        let up = |layer: &Spectral, x: &Tensor| layer_norm(&layer.forward(x, 2, 1)?)?.relu();
        let line = up(&self.decoder_l4, &x.add_tensor(&x4, 1)?)?;
        let line = up(&self.decoder_l3, &line.add_tensor(&x3, 1)?)?;
        let line = up(&self.decoder_l2, &line.add_tensor(&x2, 1)?)?;
        let line = self
            .decoder_l1
            .forward(&line.add_tensor(&x1, 1)?.reflection_pad2d([3, 3, 3, 3])?)?
            .clamp(Some((-1.0).into()), Some(1.0.into()))?;
        Ok((screen, line))
    }
}

/// `Conv2dBlock` as `gen_conv` and `gen_deconv` make it: replicate padding,
/// then the convolution, then the activation.
#[derive(Debug)]
struct Block {
    pad: i64,
    conv: Layer,
    activation: Activation,
}

#[derive(Debug)]
enum Layer {
    Conv(nn::Conv2D),
    Deconv(nn::ConvTranspose2D),
}

#[derive(Clone, Copy, Debug)]
enum Activation {
    Elu,
    Relu,
    None,
}

impl Block {
    #[allow(clippy::too_many_arguments)]
    fn conv(
        path: &nn::Path<'_>,
        from: i64,
        to: i64,
        kernel: i64,
        stride: i64,
        pad: i64,
        dilation: i64,
        activation: Activation,
    ) -> Result<Self> {
        let config = nn::ConvConfig {
            stride,
            dilation,
            ..Default::default()
        };
        Ok(Self {
            pad,
            conv: Layer::Conv(conv(&(path / "conv"), from, to, kernel, config)?),
            activation,
        })
    }

    /// `gen_deconv(from, to, 3, 2, 1)`: kernel 3, stride 2, padding and output
    /// padding 1, which doubles the size.
    fn deconv(path: &nn::Path<'_>, from: i64, to: i64) -> Result<Self> {
        let config = nn::ConvTransposeConfig {
            stride: 2,
            padding: 1,
            output_padding: 1,
            ..Default::default()
        };
        Ok(Self {
            pad: 0,
            conv: Layer::Deconv(nn::conv_transpose2d(&(path / "conv"), from, to, 3, config)?),
            activation: Activation::Elu,
        })
    }

    fn forward(&self, x: &Tensor) -> koharu_torch::Result<Tensor> {
        let x = if self.pad > 0 {
            x.replication_pad2d([self.pad; 4])?
        } else {
            x.alias()?
        };
        let x = match &self.conv {
            Layer::Conv(conv) => conv.forward(&x)?,
            Layer::Deconv(deconv) => deconv.forward(&x)?,
        };
        match self.activation {
            Activation::Elu => x.elu(1.0, 1.0, 1.0),
            Activation::Relu => x.relu(),
            Activation::None => Ok(x),
        }
    }
}

/// `ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3,
/// softmax_scale=10, fuse=True)`, of a feature map on itself.
#[derive(Debug)]
struct ContextualAttention {
    fuse_weight: Tensor,
}

/// `same_padding`: zeros so a window of `kernel` at `stride` covers every
/// pixel, the odd one after.
fn same_padding(x: &Tensor, kernel: i64, stride: i64) -> koharu_torch::Result<Tensor> {
    let size = x.size();
    let pad = |length: i64| {
        let out = (length + stride - 1) / stride;
        ((out - 1) * stride + kernel - length).max(0)
    };
    let (rows, cols) = (pad(size[2]), pad(size[3]));
    x.constant_pad_nd([cols / 2, cols - cols / 2, rows / 2, rows - rows / 2], 0.0)
}

/// `extract_image_patches` with `same` padding, as `[L, C, k, k]`.
fn patches(x: &Tensor, kernel: i64, stride: i64) -> koharu_torch::Result<Tensor> {
    let channels = x.size()[1];
    same_padding(x, kernel, stride)?
        .im2col([kernel, kernel], [1, 1], [0, 0], [stride, stride])?
        .view([1, channels, kernel, kernel, -1])?
        .permute([0, 4, 1, 2, 3])?
        .squeeze_dim(0)
}

impl ContextualAttention {
    const RATE: i64 = 2;
    const SCALE: f64 = 10.0;

    fn forward(&self, x: &Tensor, masks: &Tensor) -> koharu_torch::Result<Tensor> {
        let kernel = 2 * Self::RATE;
        let raw = patches(x, kernel, Self::RATE)?;
        let half = |x: &Tensor, scale: f64| {
            let size = x.size();
            x.upsample_nearest2d(
                [
                    (size[2] as f64 * scale) as i64,
                    (size[3] as f64 * scale) as i64,
                ],
                scale,
                scale,
            )
        };
        let f = half(x, 1.0 / Self::RATE as f64)?;
        let [height, width] = [f.size()[2], f.size()[3]];
        let w = patches(&f, 3, 1)?;
        let mask = half(masks, 1.0 / (4 * Self::RATE) as f64)?;
        // Patches with any masked pixel can't be borrowed from.
        let mm = patches(&mask, 3, 1)?
            .mean_dim([1, 2, 3], true, None::<Kind>)?
            .eq_scalar(0.0)?
            .to_kind(Kind::Float)?
            .permute([1, 0, 2, 3])?;

        let norm = w
            .pow_tensor_scalar(2)?
            .add_scalar(1e-4, 1)?
            .sum_dim_intlist([1, 2, 3], true, None::<Kind>)?
            .sqrt()?;
        let normed = w.div_tensor(&norm)?;
        let y = Tensor::conv2d(
            &same_padding(&f, 3, 1)?,
            &normed,
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            1,
        )?;

        let area = height * width;
        let fuse = |y: &Tensor| {
            Tensor::conv2d(
                &same_padding(y, 3, 1)?,
                &self.fuse_weight,
                None,
                [1, 1],
                [0, 0],
                [1, 1],
                1,
            )
        };
        let y = fuse(&y.view([1, 1, area, area])?)?
            .contiguous(koharu_torch::MemoryFormat::Contiguous)?
            .view([1, height, width, height, width])?
            .permute([0, 2, 1, 4, 3])?
            .contiguous(koharu_torch::MemoryFormat::Contiguous)?
            .view([1, 1, area, area])?;
        let y = fuse(&y)?
            .contiguous(koharu_torch::MemoryFormat::Contiguous)?
            .view([1, width, height, width, height])?
            .permute([0, 2, 1, 4, 3])?
            .contiguous(koharu_torch::MemoryFormat::Contiguous)?
            .view([1, area, height, width])?;
        let y = y
            .mul_tensor(&mm)?
            .mul_scalar(Self::SCALE)?
            .softmax_int(1, None::<Kind>)?
            .mul_tensor(&mm)?;
        Tensor::conv_transpose2d_input(
            &y,
            &raw,
            None,
            [Self::RATE, Self::RATE],
            [1, 1],
            [0, 0],
            1,
            [1, 1],
        )?
        .div_scalar(4.0)
    }
}

/// `MangaInpaintGenerator(6, 32, 1)`: the page filled from the masked page,
/// the screentone, and the lines.
#[derive(Debug)]
struct MangaGenerator {
    /// `conv1` to `conv10_atrous`.
    hallucination: Vec<Block>,
    /// `pmconv1` to `pmconv6`.
    before_attention: Vec<Block>,
    attention: ContextualAttention,
    /// `pmconv9` and `pmconv10`.
    after_attention: Vec<Block>,
    /// `allconv11` to `allconv17`.
    merge: Vec<Block>,
}

impl MangaGenerator {
    fn new(path: &nn::Path<'_>) -> Result<Self> {
        use Activation::{Elu, None, Relu};
        let c = 32;
        let conv = |name: &str, from, to, kernel, stride, pad, dilation, activation| {
            Block::conv(
                &(path / name),
                from,
                to,
                kernel,
                stride,
                pad,
                dilation,
                activation,
            )
        };
        Ok(Self {
            hallucination: vec![
                conv("conv1", 8, c, 5, 1, 2, 1, Elu)?,
                conv("conv2_downsample", c, c, 3, 2, 1, 1, Elu)?,
                conv("conv3", c, c * 2, 3, 1, 1, 1, Elu)?,
                conv("conv4_downsample", c * 2, c * 2, 3, 2, 1, 1, Elu)?,
                conv("conv5", c * 2, c * 4, 3, 1, 1, 1, Elu)?,
                conv("conv6_downsample", c * 4, c * 4, 3, 1, 1, 1, Elu)?,
                conv("conv7_atrous", c * 4, c * 4, 3, 1, 2, 2, Elu)?,
                conv("conv8_atrous", c * 4, c * 4, 3, 1, 4, 4, Elu)?,
                conv("conv9_atrous", c * 4, c * 4, 3, 1, 8, 8, Elu)?,
                conv("conv10_atrous", c * 4, c * 4, 3, 1, 16, 16, Elu)?,
            ],
            before_attention: vec![
                conv("pmconv1", 8, c, 5, 1, 2, 1, Elu)?,
                conv("pmconv2_downsample", c, c, 3, 2, 1, 1, Elu)?,
                conv("pmconv3", c, c * 2, 3, 1, 1, 1, Elu)?,
                conv("pmconv4_downsample", c * 2, c * 4, 3, 2, 1, 1, Elu)?,
                conv("pmconv5", c * 4, c * 4, 3, 1, 1, 1, Elu)?,
                conv("pmconv6", c * 4, c * 4, 3, 1, 1, 1, Relu)?,
            ],
            attention: ContextualAttention {
                fuse_weight: (path / "contextul_attention").var("fuse_weight", &[1, 1, 3, 3])?,
            },
            after_attention: vec![
                conv("pmconv9", c * 4, c * 4, 3, 1, 1, 1, Elu)?,
                conv("pmconv10", c * 4, c * 4, 3, 1, 1, 1, Elu)?,
            ],
            merge: vec![
                conv("allconv11", c * 8, c * 4, 3, 1, 1, 1, Elu)?,
                conv("allconv12", c * 4, c * 4, 3, 1, 1, 1, Elu)?,
                Block::deconv(&(path / "allconv13"), c * 4, c * 2)?,
                conv("allconv14", c * 2, c * 2, 3, 1, 1, 1, Elu)?,
                Block::deconv(&(path / "allconv15"), c * 2, c)?,
                conv("allconv16", c, c / 2, 3, 1, 1, 1, Elu)?,
                conv("allconv17", c / 2, 1, 5, 1, 2, 1, None)?,
            ],
        })
    }

    fn forward(&self, xin: &Tensor, ones: &Tensor, masks: &Tensor) -> koharu_torch::Result<Tensor> {
        let run = |blocks: &[Block], x: &Tensor| -> koharu_torch::Result<Tensor> {
            let mut x = x.alias()?;
            for block in blocks {
                x = block.forward(&x)?;
            }
            Ok(x)
        };
        let input = Tensor::cat(&[xin.alias()?, ones.alias()?, masks.alias()?], 1)?;
        let hallucinated = run(&self.hallucination, &input)?;
        let attended = self
            .attention
            .forward(&run(&self.before_attention, &input)?, masks)?;
        let attended = run(&self.after_attention, &attended)?;
        run(&self.merge, &Tensor::cat(&[hallucinated, attended], 1)?)?
            .clamp(Some((-1.0).into()), Some(1.0.into()))
    }
}

/// The trace's `JITWrapper`.
#[derive(Debug)]
pub(super) struct Inpaintor {
    screen: ScreenEncoder,
    semantic: SemanticGenerator,
    manga: MangaGenerator,
}

impl Inpaintor {
    pub(super) fn new(path: &nn::Path<'_>) -> Result<Self> {
        Ok(Self {
            screen: ScreenEncoder::new(&(path / "svae_model" / "enc"))?,
            semantic: SemanticGenerator::new(
                &(path / "semantic_inpaint_model" / "generator" / "module"),
            )?,
            manga: MangaGenerator::new(&(path / "manga_inpaint_model" / "generator"))?,
        })
    }

    /// The gray page, its lines, the mask, and noise, each `[1, 1, H, W]`
    /// with pixels in [-1, 1] and the mask 0 or 1, to the filled page.
    pub(super) fn forward(
        &self,
        images: &Tensor,
        lines: &Tensor,
        masks: &Tensor,
        noise: &Tensor,
    ) -> koharu_torch::Result<Tensor> {
        let kept = masks.ones_like(None, None)?.sub_tensor(masks, 1)?;
        // Masked pixels white, in the page and in its lines.
        let x = images.mul_tensor(&kept)?.add_tensor(masks, 1)?;
        let line = lines.mul_tensor(&kept)?.add_tensor(masks, 1)?;

        // `ScreenVAE.forward(line, x, rep=True)`.
        let sign = line.sign()?;
        let x = x
            .add_tensor(&sign.ones_like(None, None)?.sub_tensor(&sign, 1)?, 1)?
            .clamp(Some((-1.0).into()), Some(1.0.into()))?;
        let screen = self
            .screen
            .forward(&Tensor::cat(&[x, sign], 1)?)?
            .narrow(1, 0, 4)?;

        let (screen, line) = self.semantic.forward(&Tensor::cat(
            &[screen, line, masks.alias()?, noise.alias()?],
            1,
        )?)?;
        let hints = Tensor::cat(&[screen, line], 1)?;
        let masked = images.mul_tensor(&kept)?;
        self.manga.forward(
            &Tensor::cat(&[masked, hints], 1)?,
            &masks.ones_like(None, None)?,
            masks,
        )
    }
}
