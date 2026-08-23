//! LaMa inference with IOPaint-compatible orchestration.

mod backend;
mod config;
mod model;
mod processor;

use anyhow::{Context, Result};
use image::{DynamicImage, GrayImage, RgbImage};
use koharu_torch::Device;

use crate::{backend::TryIntoDevice, source::ComponentSource, torchscript::TorchScript};

pub use self::config::{HDStrategy, InpaintRequest, WeightsFormat};
use self::{
    backend::Backend, config::FFCResNetGeneratorConfig, model::Model, processor::InpaintModel,
};

crate::model_repository!("mayocream/lama-manga" @ "f91c85b26913b3e83f9877867b4c336da3675238" {
    WEIGHTS = "lama-manga.safetensors"
});

// The TorchScript default is fetched straight from the upstream release. The
// manga fine-tune fits this pipeline's subject better than `big-lama.pt`, so it
// is the default; point the source at a URL to use the original.
crate::remote_repository! {
    TORCHSCRIPT_WEIGHTS =
        "https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt"
        @ "9213532a6e9990afcd0c9f3f31da82cc4c8c1ec86a13641e3ec37648d5e75f8b",
}

#[derive(Debug)]
pub struct LaMa {
    backend: Backend,
    processor: InpaintModel,
}

impl LaMa {
    pub async fn load(
        device: crate::Device,
        source: &ComponentSource,
        format: WeightsFormat,
    ) -> Result<Self> {
        let device: Device = device.try_into_device()?;
        let backend = match format {
            WeightsFormat::SafeTensors => {
                let path = source
                    .resolve(WEIGHTS.into())
                    .await
                    .context("failed to resolve LaMa safetensors weights")?;
                let mut model = Model::new(&FFCResNetGeneratorConfig::default(), device);
                model
                    .load(&path)
                    .context("failed to load LaMa safetensors")?;
                Backend::SafeTensors(Box::new(model))
            }
            WeightsFormat::TorchScript => {
                let path = source
                    .resolve(TORCHSCRIPT_WEIGHTS.into())
                    .await
                    .context("failed to resolve LaMa TorchScript weights")?;
                Backend::TorchScript(TorchScript::load(&path, device)?)
            }
        };
        Ok(Self {
            backend,
            processor: InpaintModel::new(device),
        })
    }

    pub fn inference(
        &self,
        image: &DynamicImage,
        mask: &GrayImage,
        config: &InpaintRequest,
    ) -> Result<RgbImage> {
        koharu_torch::no_grad(|| self.processor.call(&self.backend, image, mask, config))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::source::ComponentSource;

    #[test]
    fn safetensors_is_the_default_weights_format() {
        assert_eq!(WeightsFormat::default(), WeightsFormat::SafeTensors);
    }

    #[tokio::test]
    #[cfg_attr(
        windows,
        ignore = "Windows resolves LibTorch only once Runtime::initialize preloads it"
    )]
    async fn the_format_picks_which_loader_reads_the_file() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("garbage.pt");
        std::fs::write(&path, b"neither safetensors nor a torchscript archive").unwrap();
        let source = ComponentSource::LocalFile(path);

        let safetensors = LaMa::load(crate::Device::cpu(), &source, WeightsFormat::SafeTensors)
            .await
            .unwrap_err();
        assert!(
            format!("{safetensors:#}").contains("failed to load LaMa safetensors"),
            "{safetensors:#}"
        );

        let torchscript = LaMa::load(crate::Device::cpu(), &source, WeightsFormat::TorchScript)
            .await
            .unwrap_err();
        assert!(
            format!("{torchscript:#}").contains("failed to load TorchScript archive"),
            "{torchscript:#}"
        );
    }

    /// Requires a real checkpoint, so it is excluded from the default run.
    /// Pass `KOHARU_BIG_LAMA_TS` with the path to a LaMa TorchScript archive.
    #[tokio::test]
    #[ignore = "requires a local LaMa TorchScript archive"]
    async fn a_torchscript_archive_inpaints_through_the_same_processor() {
        let path = std::path::PathBuf::from(std::env::var("KOHARU_BIG_LAMA_TS").unwrap());
        let model = LaMa::load(
            crate::Device::cpu(),
            &ComponentSource::LocalFile(path),
            WeightsFormat::TorchScript,
        )
        .await
        .unwrap();

        let image = DynamicImage::ImageRgb8(RgbImage::new(64, 64));
        let mut mask = GrayImage::new(64, 64);
        for pixel in mask.pixels_mut() {
            *pixel = image::Luma([255]);
        }

        let output = model
            .inference(&image, &mask, &InpaintRequest::default())
            .unwrap();

        assert_eq!(output.dimensions(), (64, 64));
    }
}
