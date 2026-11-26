//! MI-GAN, run as IOPaint runs it: Picsart's 512-pixel Places2 generator,
//! given a crop around each masked region.
//!
//! Upstream:
//! https://github.com/Picsart-AI-Research/MI-GAN/blob/2b793c5ece43f4253e32d4afc257120a5deed6f5/lib/model_zoo/migan_inference.py
//! https://github.com/Sanster/IOPaint/blob/61a759fb3f332bacdce8b2813f4837495c9b86e0/iopaint/model/mi_gan.py
//!
//! IOPaint loads a TorchScript trace of the generator, which libtorch's C API
//! here can't run. Its weights are Picsart's under Picsart's names; this
//! loads the same weights from a safetensors conversion made for MLX, which
//! keeps the names but stores each four-dimensional tensor channels-last.
//! They're turned back to PyTorch's layout once, beside the file. The trace
//! inserts an upsample's zeros with a transposed convolution where Picsart's
//! code multiplies by `filter_const`; both give the same values.
//!
//! Divergence: IOPaint resizes a crop with OpenCV's bicubic, whose kernel
//! differs a little from the Catmull-Rom one here.

mod model;

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail, ensure};
use fast_image_resize::{FilterType, ResizeAlg};
use image::{GrayImage, RgbImage, imageops};
use koharu_runtime::{Library, Repository, Turn};
use koharu_torch::{Device, Kind, Tensor};
use safetensors::{SafeTensors, tensor::TensorView};

use self::model::{Model, RESOLUTION};

const REPOSITORY: Repository = Repository::model(
    "mlx-community/MI-GAN-512-places2-fp16",
    "19644256716fc26ed723fbd9c0797d9c27f66c83",
);

/// `hd_strategy_crop_margin`, which `MIGAN.__call__` sets.
const MARGIN: u32 = 128;
const BICUBIC: ResizeAlg = ResizeAlg::Convolution(FilterType::CatmullRom);

#[derive(Debug)]
pub struct MiGan {
    device: Device,
    model: Model,
}

impl MiGan {
    pub async fn load(device: &crate::Device) -> Result<Self> {
        let (device, _) = crate::torch::device(device);
        let [weights] = REPOSITORY
            .resolve(["model.safetensors"])
            .await
            .context("failed to resolve MI-GAN's files")?;
        let weights = channels_first(&weights)?;
        let _turn = Turn::take(Library::Torch);
        let mut model = Model::new(device)?;
        model.load(weights)?;
        Ok(Self { device, model })
    }

    /// `image` repainted where `mask` is white, `MIGAN.__call__`: a 512-pixel
    /// image whole, and any other a crop around each masked region at a
    /// time, shrunk to fit 512 pixels.
    pub fn inference(&self, image: &RgbImage, mask: &GrayImage) -> Result<RgbImage> {
        ensure!(
            image.dimensions() == mask.dimensions(),
            "image and mask dimensions differ"
        );
        let side = RESOLUTION as u32;
        crate::torch::inference(|| {
            if image.dimensions() == (side, side) {
                return self.pad_forward(image, mask);
            }
            let (width, height) = image.dimensions();
            let mut result = image.clone();
            for found in crate::lama::boxes_from_mask(mask) {
                let [left, top, right, bottom] =
                    crate::lama::crop_box(width, height, found, MARGIN);
                let (crop_width, crop_height) = (right - left, bottom - top);
                let crop = imageops::crop_imm(image, left, top, crop_width, crop_height).to_image();
                let crop_mask =
                    imageops::crop_imm(mask, left, top, crop_width, crop_height).to_image();
                // `resize_max_size`.
                let longest = crop_width.max(crop_height);
                let painted = if longest > side {
                    let ratio = f64::from(side) / f64::from(longest);
                    let fit = |length: u32| (f64::from(length) * ratio + 0.5) as u32;
                    let (fit_width, fit_height) = (fit(crop_width), fit(crop_height));
                    let painted = self.pad_forward(
                        &crate::image::resize(&crop, fit_width, fit_height, BICUBIC)?,
                        &crate::image::resize(&crop_mask, fit_width, fit_height, BICUBIC)?,
                    )?;
                    crate::image::resize(&painted, crop_width, crop_height, BICUBIC)?
                } else {
                    self.pad_forward(&crop, &crop_mask)?
                };
                // Only the masked pixels of the crop come from the model.
                let mut pasted = crop;
                for ((pixel, new), held) in pasted
                    .pixels_mut()
                    .zip(painted.pixels())
                    .zip(crop_mask.pixels())
                {
                    if held.0[0] >= 127 {
                        *pixel = *new;
                    }
                }
                imageops::replace(&mut result, &pasted, left.into(), top.into());
            }
            Ok(result)
        })
    }

    /// `_pad_forward` and `MIGAN.forward`: the image padded symmetrically to
    /// 512 pixels square, through the generator with the mask over 120, cropped
    /// back, and blended over the image by the mask.
    fn pad_forward(&self, image: &RgbImage, mask: &GrayImage) -> Result<RgbImage> {
        let (width, height) = image.dimensions();
        let side = RESOLUTION as u32;
        ensure!(
            width <= side && height <= side,
            "MI-GAN takes at most {side} pixels a side"
        );
        let index = |at: u32, length: u32| crate::image::symmetric_index(at, length);
        let padded = RgbImage::from_fn(side, side, |x, y| {
            *image.get_pixel(index(x, width), index(y, height))
        });
        let padded_mask = GrayImage::from_fn(side, side, |x, y| {
            *mask.get_pixel(index(x, width), index(y, height))
        });
        let side = i64::from(side);
        let picture = Tensor::from_slice(padded.as_raw())?
            .view([side, side, 3])?
            .permute([2, 0, 1])?
            .unsqueeze(0)?
            .to(self.device)?
            .to_kind(Kind::Float)?
            .div_scalar(127.5)?
            .sub_scalar(1.0, 1)?;
        let held = Tensor::from_slice(padded_mask.as_raw())?
            .view([1, 1, side, side])?
            .to(self.device)?
            .gt_scalar(120.0)?
            .to_kind(Kind::Float)?;
        let kept = held.ones_like(None, None)?.sub_tensor(&held, 1)?;
        let input = Tensor::cat(
            &[held.neg()?.add_scalar(0.5, 1)?, picture.mul_tensor(&kept)?],
            1,
        )?;
        let output = self
            .model
            .forward(&input)?
            .mul_scalar(127.5)?
            .add_scalar(127.5, 1)?
            .round()?
            .clamp(Some(0.0.into()), Some(255.0.into()))?
            .to_kind(Kind::Uint8)?
            .squeeze_dim(0)?
            .permute([1, 2, 0])?
            .contiguous(koharu_torch::MemoryFormat::Contiguous)?
            .to(Device::Cpu)?
            .view([-1])?;
        let generated = RgbImage::from_raw(side as u32, side as u32, Vec::<u8>::try_from(&output)?)
            .context("MI-GAN's output isn't an image")?;
        // `sd_keep_unmasked_area`.
        let mut result = image.clone();
        for (x, y, pixel) in result.enumerate_pixels_mut() {
            let alpha = f32::from(mask.get_pixel(x, y).0[0]) / 255.0;
            let new = generated.get_pixel(x, y);
            for (old, new) in pixel.0.iter_mut().zip(new.0) {
                *old = (f32::from(new) * alpha + f32::from(*old) * (1.0 - alpha)) as u8;
            }
        }
        Ok(result)
    }
}

/// `weights` with every four-dimensional tensor from channels-last back to
/// PyTorch's channels-first, made once beside it.
fn channels_first(weights: &Path) -> Result<PathBuf> {
    let folder = weights
        .parent()
        .context("the weights are in a folder")?
        .join("koharu");
    let converted = folder.join("model.safetensors");
    if converted.is_file() {
        return Ok(converted);
    }
    fs_err::create_dir_all(&folder)?;
    let bytes = fs_err::read(weights)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let mut permuted = Vec::new();
    for (name, view) in tensors.tensors() {
        let data = match view.shape() {
            &[outer, height, width, inner] => {
                let element = view.dtype().bitsize() / 8;
                let data = view.data();
                let mut out = vec![0; data.len()];
                for o in 0..outer {
                    for h in 0..height {
                        for w in 0..width {
                            for i in 0..inner {
                                let from = (((o * height + h) * width + w) * inner + i) * element;
                                let to = (((o * inner + i) * height + h) * width + w) * element;
                                out[to..to + element].copy_from_slice(&data[from..from + element]);
                            }
                        }
                    }
                }
                (vec![outer, inner, height, width], out)
            }
            shape if shape.len() <= 2 => (shape.to_vec(), view.data().to_vec()),
            shape => bail!("`{name}` is {shape:?}, which MI-GAN has none of"),
        };
        permuted.push((name, view.dtype(), data));
    }
    let views = permuted
        .iter()
        .map(|(name, dtype, (shape, data))| {
            Ok((name.clone(), TensorView::new(*dtype, shape.clone(), data)?))
        })
        .collect::<Result<Vec<_>>>()?;
    let partial = converted.with_extension("partial");
    safetensors::serialize_to_file(views, None, &partial)?;
    fs_err::rename(&partial, &converted)?;
    Ok(converted)
}
