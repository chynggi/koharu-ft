//! PowerPaint v1's object removal through stable-diffusion.cpp: Stable
//! Diffusion 1.5 inpainting fine-tuned with learned task prompts, of which
//! removal asks for `P_ctxt` and steers away from `P_obj`.
//!
//! Weights, diffusers layout:
//! https://huggingface.co/Sanster/PowerPaint-V1-stable-diffusion-inpainting/tree/4104236baefdb2a1dd0f32812d8edbbfc4efd164
//! Task prompts, settings, and padding, as IOPaint runs it:
//! https://github.com/Sanster/IOPaint/blob/61a759fb3f332bacdce8b2813f4837495c9b86e0/iopaint/model/power_paint/power_paint.py
//! https://github.com/Sanster/IOPaint/blob/61a759fb3f332bacdce8b2813f4837495c9b86e0/iopaint/model/power_paint/powerpaint_tokenizer.py
//! https://github.com/Sanster/IOPaint/blob/61a759fb3f332bacdce8b2813f4837495c9b86e0/iopaint/schema.py
//!
//! Divergences: IOPaint samples with UniPC, which stable-diffusion.cpp
//! doesn't have, so this samples with DPM-Solver++(2M), the multistep solver
//! it does. Removal's prompts are the same on both of PowerPaint's prompt
//! branches and weigh 1 to 0, so one prompt is the blend.

mod tokens;

use std::sync::Mutex;

use anyhow::{Context as _, Result, anyhow, ensure};
use image::{GrayImage, RgbImage, imageops};
use koharu_diffusion::{Canceller, Context, ContextParams, ImageParams, SampleMethod, Scheduler};
use koharu_runtime::Repository;

const REPOSITORY: Repository = Repository::model(
    "Sanster/PowerPaint-V1-stable-diffusion-inpainting",
    "4104236baefdb2a1dd0f32812d8edbbfc4efd164",
);

/// `pad_mod` and `min_size`: the image is padded to at least 512 a side.
/// IOPaint pads to a multiple of 8; stable-diffusion.cpp rounds a Stable
/// Diffusion 1 size to a multiple of 64 itself, so this pads to that.
const MODULO: u32 = 64;
const MINIMUM: u32 = 512;
/// IOPaint's `InpaintRequest` defaults.
const STEPS: u32 = 50;
const GUIDANCE: f32 = 7.5;
const STRENGTH: f32 = 1.0;
const SEED: i64 = 42;

#[derive(Debug)]
pub struct PowerPaint {
    context: Mutex<Context>,
    canceller: Canceller,
}

impl PowerPaint {
    pub fn canceller(&self) -> Canceller {
        self.canceller.clone()
    }

    pub async fn load(device: &crate::Device) -> Result<Self> {
        let [unet, vae, text_encoder, added] = REPOSITORY
            .resolve([
                "unet/diffusion_pytorch_model.fp16.safetensors",
                "vae/diffusion_pytorch_model.fp16.safetensors",
                "text_encoder/model.fp16.safetensors",
                "tokenizer/added_tokens.json",
            ])
            .await
            .context("failed to resolve PowerPaint's files")?;
        let tokens = tokens::split(&text_encoder, &added)?;
        let embeddings: Vec<(&str, &std::path::Path)> = tokens
            .embeddings
            .iter()
            .map(|(name, path)| (name.as_str(), path.as_path()))
            .collect();
        let context = Context::new(&ContextParams {
            model: Some(&tokens.text_encoder),
            diffusion_model: Some(&unet),
            vae: Some(&vae),
            embeddings: &embeddings,
            device: Some(&device.name),
            mmap: true,
            ..ContextParams::default()
        })
        .context("failed to load PowerPaint")?;
        Ok(Self {
            canceller: context.canceller(),
            context: Mutex::new(context),
        })
    }

    /// `image` with what `mask` covers removed, after `prompt`, which may be
    /// blank; only masked pixels change.
    pub fn inference(&self, image: &RgbImage, mask: &GrayImage, prompt: &str) -> Result<RgbImage> {
        ensure!(
            image.dimensions() == mask.dimensions(),
            "image and mask dimensions differ"
        );
        if mask.as_raw().iter().all(|&value| value == 0) {
            return Ok(image.clone());
        }
        let (width, height) = image.dimensions();
        let padded = |side: u32| side.div_ceil(MODULO).max(MINIMUM / MODULO) * MODULO;
        let (padded_width, padded_height) = (padded(width), padded(height));
        let init = symmetric(image, padded_width, padded_height);
        let held = symmetric(mask, padded_width, padded_height);
        let prompt = format!("{prompt} P_ctxt");
        let generated = self
            .context
            .lock()
            .map_err(|_| anyhow!("PowerPaint's context lock was poisoned"))?
            .generate_image(&ImageParams {
                prompt: prompt.trim_start(),
                negative_prompt: "P_obj",
                width: Some(padded_width),
                height: Some(padded_height),
                init: Some(&init),
                mask: Some(&held),
                cfg: Some(GUIDANCE),
                scheduler: Some(Scheduler::Discrete),
                sample_method: Some(SampleMethod::DpmPp2m),
                steps: Some(STEPS),
                strength: Some(STRENGTH),
                seed: Some(SEED),
                count: Some(1),
                ..ImageParams::default()
            })
            .context("PowerPaint inference failed")?
            .into_iter()
            .next()
            .context("PowerPaint returned no image")?
            .into_rgb8();
        ensure!(
            generated.dimensions() == (padded_width, padded_height),
            "PowerPaint returned an image of another size"
        );
        // `sd_keep_unmasked_area`: only what the mask covers comes back.
        let generated = imageops::crop_imm(&generated, 0, 0, width, height).to_image();
        let mut out = image.clone();
        for ((pixel, new), held) in out.pixels_mut().zip(generated.pixels()).zip(mask.pixels()) {
            if held.0[0] > 0 {
                *pixel = *new;
            }
        }
        Ok(out)
    }
}

/// `image` grown to `width` by `height` past its right and bottom edges by
/// mirroring them, numpy's `symmetric` padding, which IOPaint's
/// `pad_img_to_modulo` uses.
fn symmetric<P: image::Pixel>(
    image: &image::ImageBuffer<P, Vec<P::Subpixel>>,
    width: u32,
    height: u32,
) -> image::ImageBuffer<P, Vec<P::Subpixel>> {
    let (own_width, own_height) = image.dimensions();
    image::ImageBuffer::from_fn(width, height, |x, y| {
        *image.get_pixel(
            crate::image::symmetric_index(x, own_width),
            crate::image::symmetric_index(y, own_height),
        )
    })
}
