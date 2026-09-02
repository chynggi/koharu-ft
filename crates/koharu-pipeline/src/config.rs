use anyhow::{Result, bail};
use koharu_translator::{GenerationConfig, Language};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use specta::Type;

use crate::stages::{
    Flux2KleinConfig, KoharuLayoutRFDetrSeg2XLConfig, LaMaConfig, MangaInpaintorConfig,
    MiGanConfig, PowerPaintConfig, RoremMixedConfig,
};

#[derive(Clone, Debug, PartialEq, Type)]
pub struct PipelineConfig {
    pub detection: DetectionModel,
    pub ocr: OcrModel,
    pub translation: TranslationConfig,
    pub inpainting: InpaintingModel,
    /// Settings for every model are kept independently of the active model.
    /// The active stage fields above only select which profile is used.
    pub processor: ProcessorConfig,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
struct PipelineFile {
    detection: ModelSelection,
    ocr: ModelSelection,
    translation: TranslationConfig,
    inpainting: ModelSelection,
    #[serde(default)]
    processor: ProcessorConfig,
}

impl Default for PipelineFile {
    fn default() -> Self {
        Self {
            detection: ModelSelection {
                model: "koharu-layout-rfdetr-seg-2xl".to_owned(),
            },
            ocr: ModelSelection {
                model: "paddleocr-vl-1.6".to_owned(),
            },
            translation: TranslationConfig::default(),
            inpainting: ModelSelection {
                model: "lama".to_owned(),
            },
            processor: ProcessorConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
struct ModelSelection {
    model: String,
}

impl Serialize for PipelineConfig {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let detection = match &self.detection {
            DetectionModel::KoharuLayoutRFDetrSeg2XL(_) => "koharu-layout-rfdetr-seg-2xl",
        };
        let ocr = match &self.ocr {
            OcrModel::PaddleOcrVl1_6 => "paddleocr-vl-1.6",
            OcrModel::MangaOcr => "manga-ocr",
            OcrModel::BaberuOcr => "baberu-ocr",
        };
        let inpainting = match &self.inpainting {
            InpaintingModel::LaMa(_) => "lama",
            InpaintingModel::MiGan(_) => "mi-gan",
            InpaintingModel::MangaInpaintor(_) => "manga-inpaintor",
            InpaintingModel::AotInpainting {} => "aot-inpainting",
            InpaintingModel::Flux2Klein(_) => "flux2-klein",
            InpaintingModel::RoremMixed(_) => "rorem-mixed",
            InpaintingModel::PowerPaint(_) => "powerpaint",
        };
        let mut processor = self.processor.clone();
        let DetectionModel::KoharuLayoutRFDetrSeg2XL(config) = &self.detection;
        processor
            .koharu_layout_rfdetr_seg_2xl
            .get_or_insert_with(|| config.clone());
        match &self.inpainting {
            InpaintingModel::LaMa(config) => {
                processor.lama.get_or_insert_with(|| config.clone());
            }
            InpaintingModel::MiGan(config) => {
                processor.mi_gan.get_or_insert_with(|| config.clone());
            }
            InpaintingModel::MangaInpaintor(config) => {
                processor
                    .manga_inpaintor
                    .get_or_insert_with(|| config.clone());
            }
            InpaintingModel::Flux2Klein(config) => {
                processor.flux2_klein.get_or_insert_with(|| config.clone());
            }
            InpaintingModel::RoremMixed(config) => {
                processor.rorem_mixed.get_or_insert_with(|| config.clone());
            }
            InpaintingModel::PowerPaint(config) => {
                processor.powerpaint.get_or_insert_with(|| config.clone());
            }
            InpaintingModel::AotInpainting {} => {}
        }
        PipelineFile {
            detection: ModelSelection {
                model: detection.to_owned(),
            },
            ocr: ModelSelection {
                model: ocr.to_owned(),
            },
            translation: self.translation.clone(),
            inpainting: ModelSelection {
                model: inpainting.to_owned(),
            },
            processor,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PipelineConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let file = PipelineFile::deserialize(deserializer)?;
        let detection = match file.detection.model.as_str() {
            "koharu-layout-rfdetr-seg-2xl" => DetectionModel::KoharuLayoutRFDetrSeg2XL(
                file.processor
                    .koharu_layout_rfdetr_seg_2xl
                    .clone()
                    .unwrap_or_default(),
            ),
            model => {
                return Err(serde::de::Error::custom(format!(
                    "unsupported detection model {model}"
                )));
            }
        };
        let ocr = match file.ocr.model.as_str() {
            "paddleocr-vl-1.6" => OcrModel::PaddleOcrVl1_6,
            "manga-ocr" => OcrModel::MangaOcr,
            "baberu-ocr" => OcrModel::BaberuOcr,
            model => {
                return Err(serde::de::Error::custom(format!(
                    "unsupported OCR model {model}"
                )));
            }
        };
        let inpainting = match file.inpainting.model.as_str() {
            "lama" => InpaintingModel::LaMa(file.processor.lama.clone().unwrap_or_default()),
            "mi-gan" => InpaintingModel::MiGan(file.processor.mi_gan.clone().unwrap_or_default()),
            "manga-inpaintor" => InpaintingModel::MangaInpaintor(
                file.processor.manga_inpaintor.clone().unwrap_or_default(),
            ),
            "aot-inpainting" => InpaintingModel::AotInpainting {},
            "flux2-klein" => {
                InpaintingModel::Flux2Klein(file.processor.flux2_klein.clone().unwrap_or_default())
            }
            "rorem-mixed" => {
                InpaintingModel::RoremMixed(file.processor.rorem_mixed.clone().unwrap_or_default())
            }
            "powerpaint" => {
                InpaintingModel::PowerPaint(file.processor.powerpaint.clone().unwrap_or_default())
            }
            model => {
                return Err(serde::de::Error::custom(format!(
                    "unsupported inpainting model {model}"
                )));
            }
        };
        Ok(Self {
            detection,
            ocr,
            translation: file.translation,
            inpainting,
            processor: file.processor,
        })
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            detection: DetectionModel::KoharuLayoutRFDetrSeg2XL(
                KoharuLayoutRFDetrSeg2XLConfig::default(),
            ),
            ocr: OcrModel::PaddleOcrVl1_6,
            translation: TranslationConfig::default(),
            inpainting: InpaintingModel::LaMa(LaMaConfig::default()),
            processor: ProcessorConfig::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
pub struct TranslationConfig {
    pub model: koharu_translator::ModelSelection,
    pub generation: GenerationConfig,
    #[specta(type = String)]
    pub target_language: Language,
    pub instructions: Option<String>,
}

impl Default for TranslationConfig {
    fn default() -> Self {
        Self {
            model: koharu_translator::ModelSelection::default(),
            generation: GenerationConfig::default(),
            target_language: Language::English,
            instructions: None,
        }
    }
}

impl PipelineConfig {
    pub fn load() -> anyhow::Result<koharu_config::Config<Self>> {
        koharu_config::load("pipeline")
    }

    pub fn detection(&self) -> Result<DetectionModel> {
        match &self.detection {
            DetectionModel::KoharuLayoutRFDetrSeg2XL(config) => {
                Ok(DetectionModel::KoharuLayoutRFDetrSeg2XL(
                    self.processor
                        .koharu_layout_rfdetr_seg_2xl
                        .clone()
                        .unwrap_or_else(|| config.clone()),
                ))
            }
        }
    }

    pub fn inpainting(&self) -> Result<InpaintingModel> {
        match &self.inpainting {
            InpaintingModel::LaMa(config) => Ok(InpaintingModel::LaMa(
                self.processor
                    .lama
                    .clone()
                    .unwrap_or_else(|| config.clone()),
            )),
            InpaintingModel::MiGan(config) => Ok(InpaintingModel::MiGan(
                self.processor
                    .mi_gan
                    .clone()
                    .unwrap_or_else(|| config.clone()),
            )),
            InpaintingModel::MangaInpaintor(config) => Ok(InpaintingModel::MangaInpaintor(
                self.processor
                    .manga_inpaintor
                    .clone()
                    .unwrap_or_else(|| config.clone()),
            )),
            InpaintingModel::AotInpainting {} => Ok(InpaintingModel::AotInpainting {}),
            InpaintingModel::Flux2Klein(config) => Ok(InpaintingModel::Flux2Klein(
                self.processor
                    .flux2_klein
                    .clone()
                    .unwrap_or_else(|| config.clone()),
            )),
            InpaintingModel::RoremMixed(config) => Ok(InpaintingModel::RoremMixed(
                self.processor
                    .rorem_mixed
                    .clone()
                    .unwrap_or_else(|| config.clone()),
            )),
            InpaintingModel::PowerPaint(config) => Ok(InpaintingModel::PowerPaint(
                self.processor
                    .powerpaint
                    .clone()
                    .unwrap_or_else(|| config.clone()),
            )),
        }
    }

    pub fn validate(&self) -> Result<()> {
        let _ = self.detection()?;
        let _ = self.inpainting()?;
        if !matches!(
            self.ocr,
            OcrModel::PaddleOcrVl1_6 | OcrModel::MangaOcr | OcrModel::BaberuOcr
        ) {
            bail!("unsupported OCR model")
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct ProcessorConfig {
    #[serde(rename = "koharu-layout-rfdetr-seg-2xl")]
    pub koharu_layout_rfdetr_seg_2xl: Option<KoharuLayoutRFDetrSeg2XLConfig>,
    #[serde(rename = "lama")]
    pub lama: Option<LaMaConfig>,
    #[serde(rename = "mi-gan")]
    pub mi_gan: Option<MiGanConfig>,
    #[serde(rename = "manga-inpaintor")]
    pub manga_inpaintor: Option<MangaInpaintorConfig>,
    #[serde(rename = "flux2-klein")]
    pub flux2_klein: Option<Flux2KleinConfig>,
    #[serde(rename = "rorem-mixed")]
    pub rorem_mixed: Option<RoremMixedConfig>,
    #[serde(rename = "powerpaint")]
    pub powerpaint: Option<PowerPaintConfig>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "model")]
pub enum DetectionModel {
    #[serde(rename = "koharu-layout-rfdetr-seg-2xl")]
    KoharuLayoutRFDetrSeg2XL(KoharuLayoutRFDetrSeg2XLConfig),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "model")]
pub enum OcrModel {
    #[serde(rename = "paddleocr-vl-1.6")]
    PaddleOcrVl1_6,
    #[serde(rename = "manga-ocr")]
    MangaOcr,
    #[serde(rename = "baberu-ocr")]
    BaberuOcr,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "model")]
pub enum InpaintingModel {
    #[serde(rename = "lama")]
    LaMa(LaMaConfig),
    #[serde(rename = "mi-gan")]
    MiGan(MiGanConfig),
    #[serde(rename = "manga-inpaintor")]
    MangaInpaintor(MangaInpaintorConfig),
    #[serde(rename = "aot-inpainting")]
    AotInpainting {},
    #[serde(rename = "flux2-klein")]
    Flux2Klein(Flux2KleinConfig),
    #[serde(rename = "rorem-mixed")]
    RoremMixed(RoremMixedConfig),
    #[serde(rename = "powerpaint")]
    PowerPaint(PowerPaintConfig),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stages::WeightsFormatConfig;

    #[test]
    fn defaults_select_one_processor_for_each_phase() {
        let config = PipelineConfig::default();

        assert!(matches!(
            config.detection,
            DetectionModel::KoharuLayoutRFDetrSeg2XL(_)
        ));
        assert!(matches!(config.ocr, OcrModel::PaddleOcrVl1_6));
        assert!(matches!(config.inpainting, InpaintingModel::LaMa(_)));
    }

    #[test]
    fn parses_phase_keyed_processor_configuration() {
        let config: PipelineConfig = toml::from_str(
            r#"
                [detection]
                model = "koharu-layout-rfdetr-seg-2xl"

                [ocr]
                model = "baberu-ocr"

                [inpainting]
                model = "rorem-mixed"

                [processor."rorem-mixed"]
                prompt = "Remove the lettering."
                negative_prompt = "letters, words"
            "#,
        )
        .unwrap();

        assert!(matches!(
            config.detection,
            DetectionModel::KoharuLayoutRFDetrSeg2XL(_)
        ));
        assert!(matches!(config.ocr, OcrModel::BaberuOcr));
        assert!(matches!(
            config.inpainting(),
            Ok(InpaintingModel::RoremMixed(config))
                if config.prompt == "Remove the lettering."
                    && config.negative_prompt == "letters, words"
        ));
    }

    #[test]
    fn parses_powerpaint_paths_and_survives_a_round_trip() {
        let config: PipelineConfig = toml::from_str(
            r#"
                [inpainting]
                model = "powerpaint"

                [processor."powerpaint"]
                model_path = "/models/powerpaint-v1.gguf"
                embeddings_dir = "/models/embeddings"
                steps = 24
            "#,
        )
        .unwrap();

        assert!(matches!(
            config.inpainting(),
            Ok(InpaintingModel::PowerPaint(settings))
                if settings.model_path == std::path::Path::new("/models/powerpaint-v1.gguf")
                    && settings.embeddings_dir == std::path::Path::new("/models/embeddings")
                    && settings.steps == 24
        ));

        // Serializing backfills the detection slot, so only the PowerPaint
        // halves are compared rather than the whole config.
        let reparsed: PipelineConfig = toml::from_str(&toml::to_string(&config).unwrap()).unwrap();
        assert_eq!(reparsed.inpainting, config.inpainting);
        assert_eq!(reparsed.processor.powerpaint, config.processor.powerpaint);
    }

    /// `InpaintingModel` is internally tagged on `model`, so a variant config
    /// carrying a field of that name would overwrite the tag and make the
    /// variant unreadable. PowerPaint is the only one that names a path, so it
    /// is the one that has to keep calling it `model_path`.
    #[test]
    fn a_variant_config_never_shadows_the_model_tag() {
        let serialized = toml::Value::try_from(InpaintingModel::PowerPaint(PowerPaintConfig {
            model_path: "/models/powerpaint-v1.gguf".into(),
            embeddings_dir: "/models/embeddings".into(),
            ..PowerPaintConfig::default()
        }))
        .unwrap();

        assert_eq!(serialized["model"].as_str(), Some("powerpaint"));
        assert_eq!(
            serialized["model_path"].as_str(),
            Some("/models/powerpaint-v1.gguf")
        );
    }

    #[test]
    fn missing_slots_use_defaults() {
        let config = toml::from_str::<PipelineConfig>("").unwrap();

        assert_eq!(config, PipelineConfig::default());
    }

    #[test]
    fn ignores_unknown_model_configuration_fields() {
        let config = toml::from_str::<PipelineConfig>(
            r#"
                [detection]
                model = "koharu-layout-rfdetr-seg-2xl"
                legacy_threshold = 0.5

                [ocr]
                model = "paddleocr-vl-1.6"
                legacy_language = "ja"

                [inpainting]
                model = "lama"
                legacy_resolution = 1024
            "#,
        )
        .unwrap();

        assert!(matches!(
            config.detection,
            DetectionModel::KoharuLayoutRFDetrSeg2XL(_)
        ));
        assert!(matches!(config.ocr, OcrModel::PaddleOcrVl1_6));
        assert!(matches!(config.inpainting(), Ok(InpaintingModel::LaMa(_))));
    }

    #[test]
    fn parses_detection_and_generative_inpainting_options() {
        let config = toml::from_str::<PipelineConfig>(
            r#"
                [detection]
                model = "koharu-layout-rfdetr-seg-2xl"

                [inpainting]
                model = "flux2-klein"

                [processor."koharu-layout-rfdetr-seg-2xl"]
                text_threshold = 0.25
                bubble_threshold = 0.45
                panel_threshold = 0.55

                [processor."flux2-klein"]
                prompt = "Reconstruct the illustration without text."
            "#,
        )
        .unwrap();

        assert!(matches!(
            config.detection().unwrap(),
            DetectionModel::KoharuLayoutRFDetrSeg2XL(config)
                if config.text_threshold == Some(0.25)
                    && config.bubble_threshold == Some(0.45)
                    && config.panel_threshold == Some(0.55)
        ));
        assert!(matches!(
            config.inpainting().unwrap(),
            InpaintingModel::Flux2Klein(config)
                if config.prompt == "Reconstruct the illustration without text."
        ));
    }

    #[test]
    fn keeps_profiles_separate_from_active_stage_selection() {
        let config = toml::from_str::<PipelineConfig>(
            r#"
                [detection]
                model = "koharu-layout-rfdetr-seg-2xl"

                [inpainting]
                model = "flux2-klein"

                [processor."flux2-klein"]
                prompt = "saved prompt"
            "#,
        )
        .unwrap();

        let InpaintingModel::Flux2Klein(config) = config.inpainting().unwrap() else {
            panic!("expected FLUX profile")
        };
        assert_eq!(config.prompt, "saved prompt");
    }

    #[test]
    fn serializes_model_profiles_under_processor() {
        let config = PipelineConfig {
            detection: DetectionModel::KoharuLayoutRFDetrSeg2XL(KoharuLayoutRFDetrSeg2XLConfig {
                text_threshold: Some(0.25),
                ..Default::default()
            }),
            ocr: OcrModel::PaddleOcrVl1_6,
            translation: TranslationConfig::default(),
            inpainting: InpaintingModel::Flux2Klein(Flux2KleinConfig {
                prompt: "Keep the line art.".to_owned(),
                ..Default::default()
            }),
            processor: ProcessorConfig::default(),
        };
        let document = toml::to_string(&config).unwrap();
        assert!(document.contains("[detection]\nmodel = \"koharu-layout-rfdetr-seg-2xl\""));
        assert!(document.contains("[processor.koharu-layout-rfdetr-seg-2xl]"));
        assert!(document.contains("[processor.flux2-klein]"));
        assert!(document.contains("[translation]"));
        assert!(!document.contains("prompt = \"Keep the line art.\"\n[inpainting]"));

        let restored = toml::from_str::<PipelineConfig>(&document).unwrap();
        assert!(matches!(
            restored.inpainting().unwrap(),
            InpaintingModel::Flux2Klein(config) if config.prompt == "Keep the line art."
        ));
    }

    #[test]
    fn an_existing_file_without_a_lama_section_keeps_the_builtin_checkpoint() {
        let config: PipelineConfig = toml::from_str(
            r#"
            [detection]
            model = "koharu-layout-rfdetr-seg-2xl"
            [ocr]
            model = "paddleocr-vl-1.6"
            [inpainting]
            model = "lama"
            "#,
        )
        .unwrap();

        let InpaintingModel::LaMa(lama) = config.inpainting().unwrap() else {
            panic!("expected LaMa");
        };
        assert_eq!(lama.source, LaMaConfig::default().source);
        assert_eq!(lama.format, WeightsFormatConfig::SafeTensors);
    }

    #[test]
    fn a_torchscript_lama_selection_round_trips() {
        let config = PipelineConfig {
            inpainting: InpaintingModel::LaMa(LaMaConfig {
                format: WeightsFormatConfig::TorchScript,
                ..LaMaConfig::default()
            }),
            ..PipelineConfig::default()
        };

        let text = toml::to_string(&config).unwrap();
        let parsed: PipelineConfig = toml::from_str(&text).unwrap();

        assert!(matches!(
            parsed.inpainting(),
            Ok(InpaintingModel::LaMa(config))
                if config.format == WeightsFormatConfig::TorchScript
        ));
    }

    #[test]
    fn a_mi_gan_selection_round_trips() {
        let config = PipelineConfig {
            inpainting: InpaintingModel::MiGan(MiGanConfig::default()),
            ..PipelineConfig::default()
        };

        let text = toml::to_string(&config).unwrap();
        let parsed: PipelineConfig = toml::from_str(&text).unwrap();

        assert!(matches!(parsed.inpainting(), Ok(InpaintingModel::MiGan(_))));
    }

    #[test]
    fn a_manga_inpaintor_selection_round_trips() {
        let config = PipelineConfig {
            inpainting: InpaintingModel::MangaInpaintor(MangaInpaintorConfig::default()),
            ..PipelineConfig::default()
        };

        let text = toml::to_string(&config).unwrap();
        let parsed: PipelineConfig = toml::from_str(&text).unwrap();

        assert!(matches!(
            parsed.inpainting(),
            Ok(InpaintingModel::MangaInpaintor(_))
        ));
    }
}
