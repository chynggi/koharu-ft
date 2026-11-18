//! Inference-only port of the line extractor IOPaint runs as `erika.jit`:
//! MangaLineExtraction's `res_skip`.
//!
//! Original implementation:
//! https://github.com/ljsabc/MangaLineExtraction_PyTorch/blob/6ec136d5332180b65476e62c2558f2873d5d936a/model_torch.py

use anyhow::Result;
use koharu_torch::Tensor;

use crate::nn::{self, Module, ModuleT};

/// `_bn_relu_conv`, with `_u_bn_relu_conv`'s nearest upsampling after it.
#[derive(Debug)]
struct BnReluConv {
    norm: nn::BatchNorm,
    conv: nn::Conv2D,
    up: bool,
}

impl BnReluConv {
    fn new(
        path: &nn::Path<'_>,
        in_filters: i64,
        filters: i64,
        kernel: i64,
        stride: i64,
        up: bool,
    ) -> Result<Self> {
        let path = path / "model";
        Ok(Self {
            norm: nn::batch_norm2d(
                &(&path / 0),
                in_filters,
                nn::BatchNormConfig {
                    eps: 1e-3,
                    ..Default::default()
                },
            )?,
            conv: nn::conv2d(
                &(&path / 2),
                in_filters,
                filters,
                kernel,
                nn::ConvConfig {
                    stride,
                    padding: kernel / 2,
                    ..Default::default()
                },
            )?,
            up,
        })
    }

    fn forward(&self, x: &Tensor) -> koharu_torch::Result<Tensor> {
        let x = self.norm.forward_t(x, false)?.leaky_relu(0.2)?;
        let x = self.conv.forward(&x)?;
        if self.up { double(&x) } else { Ok(x) }
    }
}

/// `nn.Upsample(scale_factor=2, mode='nearest')`.
fn double(x: &Tensor) -> koharu_torch::Result<Tensor> {
    let size = x.size();
    x.upsample_nearest2d([size[2] * 2, size[3] * 2], None, None)
}

/// `basic_block` and `_u_basic_block`; the shortcut convolves only when the
/// block changes the channels or the size.
#[derive(Debug)]
struct Block {
    conv1: BnReluConv,
    residual: BnReluConv,
    shortcut: Option<nn::Conv2D>,
    up: bool,
}

impl Block {
    fn new(
        path: &nn::Path<'_>,
        in_filters: i64,
        filters: i64,
        stride: i64,
        up: bool,
    ) -> Result<Self> {
        let projects = in_filters != filters || stride != 1;
        Ok(Self {
            conv1: BnReluConv::new(&(path / "conv1"), in_filters, filters, 3, stride, up)?,
            residual: BnReluConv::new(&(path / "residual"), filters, filters, 3, 1, false)?,
            shortcut: projects
                .then(|| {
                    nn::conv2d(
                        &(path / "shortcut" / "model" / 0),
                        in_filters,
                        filters,
                        1,
                        nn::ConvConfig {
                            stride,
                            ..Default::default()
                        },
                    )
                })
                .transpose()?,
            up,
        })
    }

    fn forward(&self, x: &Tensor) -> koharu_torch::Result<Tensor> {
        let y = self.residual.forward(&self.conv1.forward(x)?)?;
        let x = match &self.shortcut {
            Some(conv) if self.up => double(&conv.forward(x)?)?,
            Some(conv) => conv.forward(x)?,
            None => x.alias()?,
        };
        x.add_tensor(&y, 1)
    }
}

/// `_residual_block` (halving the size in its last block unless it's the
/// first) or `_upsampling_residual_block` (doubling it in its first).
#[derive(Debug)]
struct Stage(Vec<Block>);

impl Stage {
    fn down(
        path: &nn::Path<'_>,
        in_filters: i64,
        filters: i64,
        repetitions: usize,
        first: bool,
    ) -> Result<Self> {
        let path = path / "model";
        (0..repetitions)
            .map(|index| {
                let stride = if index == repetitions - 1 && !first {
                    2
                } else {
                    1
                };
                let input = if index == 0 { in_filters } else { filters };
                Block::new(&(&path / index), input, filters, stride, false)
            })
            .collect::<Result<_>>()
            .map(Self)
    }

    fn up(path: &nn::Path<'_>, in_filters: i64, filters: i64, repetitions: usize) -> Result<Self> {
        let path = path / "model";
        (0..repetitions)
            .map(|index| {
                let input = if index == 0 { in_filters } else { filters };
                Block::new(&(&path / index), input, filters, 1, index == 0)
            })
            .collect::<Result<_>>()
            .map(Self)
    }

    fn forward(&self, x: &Tensor) -> koharu_torch::Result<Tensor> {
        let mut x = x.alias()?;
        for block in &self.0 {
            x = block.forward(&x)?;
        }
        Ok(x)
    }
}

/// `res_skip`: the gray page, 0 to 255, to its lines on the same scale.
#[derive(Debug)]
pub(super) struct Erika {
    block0: Stage,
    block1: Stage,
    block2: Stage,
    block3: Stage,
    block4: Stage,
    block5: Stage,
    block6: Stage,
    block7: Stage,
    block8: Stage,
    block9: Stage,
    conv15: BnReluConv,
}

impl Erika {
    pub(super) fn new(path: &nn::Path<'_>) -> Result<Self> {
        Ok(Self {
            block0: Stage::down(&(path / "block0"), 1, 24, 2, true)?,
            block1: Stage::down(&(path / "block1"), 24, 48, 3, false)?,
            block2: Stage::down(&(path / "block2"), 48, 96, 5, false)?,
            block3: Stage::down(&(path / "block3"), 96, 192, 7, false)?,
            block4: Stage::down(&(path / "block4"), 192, 384, 12, false)?,
            block5: Stage::up(&(path / "block5"), 384, 192, 7)?,
            block6: Stage::up(&(path / "block6"), 192, 96, 5)?,
            block7: Stage::up(&(path / "block7"), 96, 48, 3)?,
            block8: Stage::up(&(path / "block8"), 48, 24, 2)?,
            block9: Stage::down(&(path / "block9"), 24, 16, 2, true)?,
            conv15: BnReluConv::new(&(path / "conv15"), 16, 1, 1, 1, false)?,
        })
    }

    /// `res1` to `res4` are `_shortcut`s of equal channels: plain sums.
    pub(super) fn forward(&self, x: &Tensor) -> koharu_torch::Result<Tensor> {
        let x0 = self.block0.forward(x)?;
        let x1 = self.block1.forward(&x0)?;
        let x2 = self.block2.forward(&x1)?;
        let x3 = self.block3.forward(&x2)?;
        let x4 = self.block4.forward(&x3)?;
        let res1 = x3.add_tensor(&self.block5.forward(&x4)?, 1)?;
        let res2 = x2.add_tensor(&self.block6.forward(&res1)?, 1)?;
        let res3 = x1.add_tensor(&self.block7.forward(&res2)?, 1)?;
        let res4 = x0.add_tensor(&self.block8.forward(&res3)?, 1)?;
        self.conv15.forward(&self.block9.forward(&res4)?)
    }
}
