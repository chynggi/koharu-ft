mod config;
mod model;
mod processor;

use anyhow::{Result, ensure};
use config::{CHARACTERS, ModelConfig};
use image::DynamicImage;
use koharu_runtime::{Library, Repository, Turn};
use model::Model;
use processor::{ImageProcessor, Vocabulary};

const REPOSITORY: Repository = Repository::model(
    "muscgab/manga-ocr-base-CPT-flex",
    "9cc15742f8dc37cb5222a6569839bc774dfc0392",
);

pub struct MangaOcrCptFlex {
    model: Model,
    processor: ImageProcessor,
    vocabulary: Vocabulary,
}

impl MangaOcrCptFlex {
    pub async fn load(device: &crate::Device) -> Result<Self> {
        let (device, kind) = crate::torch::device(device);
        let [config, weights, vocabulary] = REPOSITORY
            .resolve([
                "config.json",
                "model.safetensors",
                "nar_vocabulary_6817.json",
            ])
            .await?;
        ModelConfig::from_file(config)?.validate()?;
        let vocabulary = Vocabulary::from_file(vocabulary)?;
        ensure!(
            vocabulary.len() as i64 == CHARACTERS,
            "unexpected CPT-flex vocabulary size"
        );
        let _turn = Turn::take(Library::Torch);
        let mut model = Model::new(device, kind)?;
        model.load(weights)?;
        Ok(Self {
            model,
            processor: ImageProcessor::new(device),
            vocabulary,
        })
    }

    pub fn inference(&self, images: &[DynamicImage]) -> Result<Vec<String>> {
        crate::torch::inference(|| {
            images
                .iter()
                .map(|image| {
                    let input = self.processor.preprocess(image)?;
                    let ids = self.model.generate(&input)?;
                    Ok(self.vocabulary.decode(&ids))
                })
                .collect()
        })
    }
}
