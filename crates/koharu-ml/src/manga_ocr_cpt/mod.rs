mod config;
mod model;
mod processor;

use anyhow::{Context, Result, ensure};
use image::DynamicImage;
use koharu_torch::Device;

use crate::backend::TryIntoDevice;

use self::{
    config::{CHARACTERS, ModelConfig},
    model::Model,
    processor::{ImageProcessor, Vocabulary},
};

crate::model_repository!("muscgab/manga-ocr-base-CPT-flex" @ "9cc15742f8dc37cb5222a6569839bc774dfc0392" {
    CONFIG = "config.json",
    WEIGHTS = "model.safetensors",
    VOCABULARY = "nar_vocabulary_6817.json",
});

/// Japanese manga text-box OCR: mangaOCR's ViT-B/16 encoder and 2-layer BERT
/// decoder after continued pre-training, with an aspect-preserving input of at
/// most 196 patches and a 6,817-character vocabulary.
///
/// The weights are CC BY-NC-SA 4.0 (non-commercial). Koharu downloads them on
/// first use and does not redistribute them.
#[derive(Debug)]
pub struct MangaOcrCptFlex {
    model: Model,
    processor: ImageProcessor,
    vocabulary: Vocabulary,
}

impl MangaOcrCptFlex {
    pub async fn load(device: crate::Device) -> Result<Self> {
        let device: Device = device.try_into_device()?;
        let (config_path, weights_path, vocabulary_path) = tokio::try_join!(
            async {
                CONFIG
                    .resolve()
                    .await
                    .context("failed to resolve Manga OCR CPT-flex config")
            },
            async {
                WEIGHTS
                    .resolve()
                    .await
                    .context("failed to resolve Manga OCR CPT-flex weights")
            },
            async {
                VOCABULARY
                    .resolve()
                    .await
                    .context("failed to resolve Manga OCR CPT-flex vocabulary")
            },
        )?;

        ModelConfig::from_file(&config_path)
            .with_context(|| format!("failed to read {}", config_path.display()))?
            .validate()?;
        let vocabulary = Vocabulary::from_file(&vocabulary_path)
            .with_context(|| format!("failed to read {}", vocabulary_path.display()))?;
        ensure!(
            vocabulary.len() as i64 == CHARACTERS,
            "Manga OCR CPT-flex vocabulary has {} characters but the decoder expects {CHARACTERS}",
            vocabulary.len()
        );

        let mut model = Model::new(device);
        model
            .load(&weights_path)
            .with_context(|| format!("failed to load {}", weights_path.display()))?;

        Ok(Self {
            model,
            processor: ImageProcessor::new(device),
            vocabulary,
        })
    }

    pub fn inference(&self, image: &DynamicImage) -> Result<String> {
        koharu_torch::no_grad(|| {
            let input = self.processor.preprocess(image)?;
            let ids = self.model.generate(&input)?;
            Ok(self.vocabulary.decode(&ids))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::MangaOcrCptFlex;

    #[tokio::test]
    #[ignore = "downloads the checkpoint and requires the LibTorch runtime"]
    async fn checkpoint_matches_upstream_character_sequence() -> anyhow::Result<()> {
        crate::init().await?;
        let model = MangaOcrCptFlex::load(crate::Device::cpu()).await?;
        let input =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benches/fixtures/ocr/title.png");
        let text = model.inference(&image::open(input)?)?;
        // Upstream plain path (CPU, fp32) on the identical crop produced ids
        // [1767, 4323, 1601, 1238, 653, 377, 363, 5] before the end token.
        assert_eq!(text, "対策委員会です!");
        Ok(())
    }
}
