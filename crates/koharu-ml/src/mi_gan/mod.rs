//! MI-GAN inference through the TorchScript archive IOPaint distributes.
//!
//! Original preprocessing and inference:
//! https://github.com/Sanster/IOPaint/blob/61a759fb3f332bacdce8b2813f4837495c9b86e0/iopaint/model/mi_gan.py

mod processor;

use anyhow::{Context, Result};
use image::{DynamicImage, GrayImage, RgbImage};
use koharu_torch::Device;

use crate::{
    backend::TryIntoDevice, lama::InpaintRequest, source::ComponentSource, torchscript::TorchScript,
};

use self::processor::Processor;

// Fetched straight from the upstream release. Not mirrored.
remote_repository! {
    WEIGHTS = "https://github.com/Sanster/models/releases/download/migan/migan_traced.pt"
        @ "fde1e5f7c6b6a48082f8eff36b9117e64b8c14ea4d1a76af508e29d357b28cbd",
}

#[derive(Debug)]
pub struct MiGan {
    model: TorchScript,
    processor: Processor,
}

impl MiGan {
    pub async fn load(device: crate::Device, source: &ComponentSource) -> Result<Self> {
        let device: Device = device.try_into_device()?;
        let path = source
            .resolve(WEIGHTS.into())
            .await
            .context("failed to resolve MI-GAN weights")?;
        Ok(Self {
            model: TorchScript::load(&path, device)?,
            processor: Processor::new(device),
        })
    }

    pub fn inference(
        &self,
        image: &DynamicImage,
        mask: &GrayImage,
        config: &InpaintRequest,
    ) -> Result<RgbImage> {
        koharu_torch::no_grad(|| self.processor.call(&self.model, image, mask, config))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::source::ComponentSource;

    #[tokio::test]
    #[cfg_attr(
        windows,
        ignore = "Windows resolves LibTorch only once Runtime::initialize preloads it"
    )]
    async fn a_non_archive_file_fails_with_the_loader_context() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("garbage.pt");
        std::fs::write(&path, b"neither safetensors nor a torchscript archive").unwrap();
        let source = ComponentSource::LocalFile(path);

        let error = MiGan::load(crate::Device::cpu(), &source)
            .await
            .unwrap_err();
        assert!(
            format!("{error:#}").contains("failed to load TorchScript archive"),
            "{error:#}"
        );
    }

    /// Requires a real checkpoint, so it is excluded from the default run.
    /// Pass `KOHARU_MIGAN_TS` with the path to `migan_traced.pt` to run it.
    #[tokio::test]
    #[ignore = "requires a local migan_traced.pt"]
    async fn a_torchscript_archive_inpaints_through_the_padded_square() {
        let path = std::path::PathBuf::from(std::env::var("KOHARU_MIGAN_TS").unwrap());
        let model = MiGan::load(crate::Device::cpu(), &ComponentSource::LocalFile(path))
            .await
            .unwrap();

        // A non-square image below 512 in both sides exercises the
        // `pad_to_square` assumption: the model must accept exactly 512x512
        // and the output must come back at the original size.
        let image = DynamicImage::ImageRgb8(RgbImage::from_fn(300, 200, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        }));
        let mut mask = GrayImage::new(300, 200);
        for y in 50..150 {
            for x in 100..200 {
                mask.put_pixel(x, y, image::Luma([255]));
            }
        }

        let output = model
            .inference(&image, &mask, &InpaintRequest::default())
            .unwrap();

        assert_eq!(output.dimensions(), (300, 200));
        let original = image.to_rgb8();
        // Pixels outside the mask survive the crop, resize, and paste back.
        assert_eq!(output.get_pixel(0, 0), original.get_pixel(0, 0));
        assert_eq!(output.get_pixel(299, 199), original.get_pixel(299, 199));
    }
}
