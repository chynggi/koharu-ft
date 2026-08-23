//! The Manga inpainter: a two-model TorchScript pipeline specialized for
//! manga line art. A line extractor (erika) recovers the line work from the
//! grayscale page, and the inpaintor fills the masked region using it.
//!
//! Original pipeline:
//! https://github.com/Sanster/IOPaint/blob/61a759fb3f332bacdce8b2813f4837495c9b86e0/iopaint/model/manga.py

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
    INPAINTOR_WEIGHTS =
        "https://github.com/Sanster/models/releases/download/manga/manga_inpaintor.jit"
            @ "dc1622dc9d96387b18f6a6391ce1a8ed10be05d99e6c01e3ca2ddb8c9a3c592b",
    LINE_WEIGHTS = "https://github.com/Sanster/models/releases/download/manga/erika.jit"
        @ "33f214b1c4010c2f7d0fc82d4e9dde0185c8fd754419b6332946d34038b1c173",
}

/// The two checkpoints the pipeline is assembled from.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MangaSource {
    pub inpaintor: ComponentSource,
    pub line: ComponentSource,
}

impl MangaSource {
    pub fn validate(&self) -> Result<()> {
        self.inpaintor
            .validate()
            .context("Manga inpaintor weights")?;
        self.line.validate().context("Manga line model weights")?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct MangaInpaintor {
    inpaintor: TorchScript,
    line: TorchScript,
    processor: Processor,
}

impl MangaInpaintor {
    pub async fn load(device: crate::Device, source: &MangaSource) -> Result<Self> {
        let device: Device = device.try_into_device()?;
        let inpaintor_path = source
            .inpaintor
            .resolve(INPAINTOR_WEIGHTS.into())
            .await
            .context("failed to resolve the Manga inpaintor weights")?;
        let line_path = source
            .line
            .resolve(LINE_WEIGHTS.into())
            .await
            .context("failed to resolve the Manga line model weights")?;
        Ok(Self {
            inpaintor: TorchScript::load(&inpaintor_path, device)?,
            line: TorchScript::load(&line_path, device)?,
            processor: Processor::new(device),
        })
    }

    pub fn inference(
        &self,
        image: &DynamicImage,
        mask: &GrayImage,
        config: &InpaintRequest,
    ) -> Result<RgbImage> {
        koharu_torch::no_grad(|| {
            self.processor
                .call(&self.inpaintor, &self.line, image, mask, config)
        })
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
        let path = directory.path().join("garbage.jit");
        std::fs::write(&path, b"neither safetensors nor a torchscript archive").unwrap();
        let source = MangaSource {
            inpaintor: ComponentSource::LocalFile(path.clone()),
            line: ComponentSource::LocalFile(path),
        };

        let error = MangaInpaintor::load(crate::Device::cpu(), &source)
            .await
            .unwrap_err();
        assert!(
            format!("{error:#}").contains("failed to load TorchScript archive"),
            "{error:#}"
        );
    }

    /// Requires both real checkpoints, so it is excluded from the default run.
    /// Pass `KOHARU_MANGA_INPAINTOR_TS` and `KOHARU_MANGA_LINE_TS` with the
    /// paths to `manga_inpaintor.jit` and `erika.jit` to run it.
    #[tokio::test]
    #[ignore = "requires local manga_inpaintor.jit and erika.jit"]
    async fn a_real_checkpoint_inpaints_and_the_seed_is_reproducible() {
        let source = MangaSource {
            inpaintor: ComponentSource::LocalFile(std::path::PathBuf::from(
                std::env::var("KOHARU_MANGA_INPAINTOR_TS").unwrap(),
            )),
            line: ComponentSource::LocalFile(std::path::PathBuf::from(
                std::env::var("KOHARU_MANGA_LINE_TS").unwrap(),
            )),
        };
        let model = MangaInpaintor::load(crate::Device::cpu(), &source)
            .await
            .unwrap();

        // A size that is not a multiple of `pad_mod` exercises the modulo-16
        // padding path.
        let image = DynamicImage::ImageRgb8(RgbImage::from_fn(300, 200, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        }));
        let mut mask = GrayImage::new(300, 200);
        for y in 50..150 {
            for x in 100..200 {
                mask.put_pixel(x, y, image::Luma([255]));
            }
        }

        let first = model
            .inference(&image, &mask, &InpaintRequest::default())
            .unwrap();
        let second = model
            .inference(&image, &mask, &InpaintRequest::default())
            .unwrap();

        assert_eq!(first.dimensions(), (300, 200));
        // `manga.py` reseeds every forward call, so two runs on the same
        // input must be byte-identical.
        assert_eq!(first.as_raw(), second.as_raw());
        // Pixels outside the mask survive the crop and paste back.
        let original = image.to_rgb8();
        assert_eq!(first.get_pixel(0, 0), original.get_pixel(0, 0));
        assert_eq!(first.get_pixel(299, 199), original.get_pixel(299, 199));
    }
}
