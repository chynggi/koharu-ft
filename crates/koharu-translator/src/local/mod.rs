use std::sync::Arc;

use anyhow::Context;
use koharu_ml::llm::{ChatMessage, ChatTemplateOptions, Input, Llm, media_marker};

mod catalog;
mod registry;

use catalog::LocalModelDescriptor;
pub(crate) use catalog::{DEFAULT_MODEL, DEFAULT_QUANTIZATION};
pub use registry::{
    ContextMode, CustomModel, FlashAttentionMode, GpuLayers, KvCacheChoice, LlmRuntimeConfig,
    LocalConfig,
};
use registry::LocalModel;

use crate::{
    Device, Error, GenerationConfig, Model, ModelSelection, Provider, Quantization, Result,
    TranslationRequest, prompt,
};

/// The inputs that decide whether a loaded model can be reused. Everything else
/// in [`LocalConfig`] is applied per inference and needs no reload.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LoadSignature {
    gpu_layers: u32,
    custom: Option<CustomModel>,
}

impl LoadSignature {
    pub(crate) fn new(config: &LocalConfig, model: Option<&str>) -> Self {
        let model = model.unwrap_or_default();
        Self {
            gpu_layers: config.runtime_for(model).load_options(None).gpu_layers,
            custom: config
                .models
                .iter()
                .find(|entry| entry.id == model)
                .cloned(),
        }
    }
}

#[derive(Debug)]
pub struct LocalTranslator {
    model: LocalModel,
    llm: Arc<Llm>,
}

impl LocalTranslator {
    pub async fn load(
        device: Device,
        selection: &ModelSelection,
        config: &LocalConfig,
    ) -> Result<Self> {
        let id = selection
            .model
            .as_deref()
            .context("local translation requires a selected model")?;
        let model = LocalModel::find(config, id)?;
        let resolved = model.resolve(selection).await?;
        let options = config
            .runtime_for(id)
            .load_options(resolved.projector.clone());
        let llm = Llm::load_with_options(device, resolved.model, options)
            .await
            .context("failed to load local translation model")?;
        if llm.capabilities().vision != model.has_projector() {
            return Err(anyhow::anyhow!(
                "local translator '{id}' vision capability does not match its projector"
            )
            .into());
        }
        Ok(Self {
            model,
            llm: Arc::new(llm),
        })
    }

    pub(crate) async fn translate(
        &self,
        request: TranslationRequest,
        generation: GenerationConfig,
        runtime: &LlmRuntimeConfig,
    ) -> Result<Vec<String>> {
        let expected = request.segments.len();
        if expected == 0 {
            return Ok(Vec::new());
        }
        if !self
            .model
            .target_languages()
            .contains(request.target_language)
        {
            return Err(Error::UnsupportedLanguage {
                provider: "local",
                language: request.target_language,
            });
        }

        let image = request.image.clone();
        let prompt = self.render_prompt(
            &request,
            generation.reasoning.unwrap_or(false) && self.model.supports_reasoning(),
        )?;
        let schema = prompt::output_schema(expected);
        let llm = Arc::clone(&self.llm);
        let generation = self.model.generation().options(generation, runtime);
        let output = tokio::task::spawn_blocking(move || {
            let input = image.as_deref().map_or_else(
                || Input::new(&prompt),
                |image| Input::new(&prompt).with_image(image),
            );
            llm.inference_with_json_schema(&input, &generation, &schema)
        })
        .await
        .context("local translation task panicked")??;
        let segments = prompt::translations("local", &output.text, &request.segments)?;
        Ok(segments)
    }

    fn render_prompt(&self, request: &TranslationRequest, reasoning: bool) -> Result<String> {
        let (system, payload) = prompt::prompts(request)?;
        let payload = if request.image.is_some() {
            format!("{}\n{payload}", media_marker())
        } else {
            payload
        };
        Ok(self
            .llm
            .render_chat_prompt_with_options(
                &[ChatMessage::system(system), ChatMessage::user(payload)],
                ChatTemplateOptions {
                    add_generation_prompt: true,
                    enable_thinking: reasoning,
                },
            )
            .context("failed to render local translation prompt")?)
    }
}

pub(crate) fn models(config: &LocalConfig) -> Vec<Model> {
    catalog::MODELS
        .iter()
        .map(builtin_model)
        .chain(config.models.iter().map(custom_model))
        .collect()
}

fn builtin_model(descriptor: &LocalModelDescriptor) -> Model {
    Model {
        provider: Provider::Local,
        model: Some(descriptor.id.to_owned()),
        name: descriptor.name.to_owned(),
        quantizations: descriptor
            .quantizations
            .iter()
            .map(|quantization| Quantization {
                id: quantization.id.to_owned(),
                name: quantization.name.to_owned(),
            })
            .collect(),
        vision: descriptor.projector.is_some(),
        reasoning: descriptor.reasoning,
    }
}

/// A registered file is a single quantization, so it advertises none: the
/// quantization picker is meaningless for it.
fn custom_model(model: &CustomModel) -> Model {
    Model {
        provider: Provider::Local,
        model: Some(model.id.clone()),
        name: model.name.clone(),
        quantizations: Vec::new(),
        vision: model.projector.is_some(),
        reasoning: false,
    }
}

pub(crate) fn supports_vision(selection: &ModelSelection, config: &LocalConfig) -> bool {
    selection.model.as_deref().is_some_and(|model| {
        LocalModel::find(config, model).is_ok_and(|model| model.has_projector())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_models_join_the_catalog() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.gguf");
        std::fs::write(&path, b"GGUF").unwrap();
        let config = LocalConfig {
            models: vec![CustomModel {
                id: "custom-qwen-manga".to_owned(),
                name: "Qwen Manga Translator".to_owned(),
                path,
                projector: None,
            }],
            ..Default::default()
        };

        let models = models(&config);

        assert_eq!(models.len(), catalog::MODELS.len() + 1);
        let custom = models.last().unwrap();
        assert_eq!(custom.model.as_deref(), Some("custom-qwen-manga"));
        assert_eq!(custom.name, "Qwen Manga Translator");
        assert!(custom.quantizations.is_empty());
        assert!(!custom.vision);
    }

    #[test]
    fn a_custom_projector_advertises_vision() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.gguf");
        let projector = directory.path().join("mmproj-F16.gguf");
        std::fs::write(&path, b"GGUF").unwrap();
        std::fs::write(&projector, b"GGUF").unwrap();
        let config = LocalConfig {
            models: vec![CustomModel {
                id: "custom-vision".to_owned(),
                name: "Custom Vision".to_owned(),
                path,
                projector: Some(projector),
            }],
            ..Default::default()
        };
        let selection = ModelSelection {
            provider: Provider::Local,
            model: Some("custom-vision".to_owned()),
            quantization: None,
            vision: true,
            reasoning: false,
        };

        assert!(supports_vision(&selection, &config));
    }

    #[test]
    fn the_load_signature_tracks_only_load_time_settings() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.gguf");
        std::fs::write(&path, b"GGUF").unwrap();
        let model = CustomModel {
            id: "custom".to_owned(),
            name: "Custom".to_owned(),
            path,
            projector: None,
        };
        let config = LocalConfig {
            models: vec![model.clone()],
            ..Default::default()
        };

        let baseline = LoadSignature::new(&config, Some("custom"));

        // Context settings are applied per inference, so they must not force a reload.
        let mut context_only = config.clone();
        context_only.runtime.context = Some(ContextMode::Fixed { size: 8192 });
        assert_eq!(LoadSignature::new(&context_only, Some("custom")), baseline);

        // Offload is a load-time decision.
        let mut offload = config.clone();
        offload.runtime.gpu_layers = Some(GpuLayers::Custom { layers: 10 });
        assert_ne!(LoadSignature::new(&offload, Some("custom")), baseline);

        // So is repointing the model at another file.
        let mut repointed = config;
        repointed.models[0].path = directory.path().join("other.gguf");
        assert_ne!(LoadSignature::new(&repointed, Some("custom")), baseline);
    }
}
