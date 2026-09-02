//! Translation through local and hosted providers.

mod backend;
mod error;
mod json;
mod language;
mod local;
mod model;
mod prompt;
mod provider;
mod remote;

use std::sync::Arc;

use koharu_ml::Device;

use error::{Error, Result};
use local::LocalTranslator;

pub use backend::{TranslationContext, TranslationRequest};
pub use language::Language;
pub use local::{
    ContextMode, CustomModel, FlashAttentionMode, GpuLayers, KvCacheChoice, LlmRuntimeConfig,
    LocalConfig,
};
use local::LoadSignature;
pub use model::{GenerationConfig, Model, ModelSelection, Quantization};
pub(crate) use model::{ModelGeneration, QuantizationDefinition, display_name};
pub use provider::{Provider, ProviderConfig, ProvidersConfig};

#[derive(Clone)]
pub struct Translator {
    providers: koharu_config::Config<ProvidersConfig>,
    local: Arc<tokio::sync::Mutex<Option<LoadedLocal>>>,
    client: reqwest::Client,
    device: Device,
}

struct LoadedLocal {
    model: Option<String>,
    quantization: Option<String>,
    load: LoadSignature,
    translator: Arc<LocalTranslator>,
}

impl LoadedLocal {
    fn matches(&self, selection: &ModelSelection, load: &LoadSignature) -> bool {
        self.model == selection.model
            && self.quantization == selection.quantization
            && &self.load == load
    }
}

impl Translator {
    pub fn from_config(
        device: Device,
        providers: koharu_config::Config<ProvidersConfig>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            providers,
            local: Arc::new(tokio::sync::Mutex::new(None)),
            client: koharu_runtime::http_client()?,
            device,
        })
    }

    #[must_use]
    pub fn model(selection: &ModelSelection) -> &'static str {
        selection.provider.into()
    }

    #[must_use]
    pub fn supports_vision(&self, selection: &ModelSelection, generation: &GenerationConfig) -> bool {
        generation.vision.unwrap_or(false)
            && (selection.provider != Provider::Local
                || self.local_config().is_ok_and(|config| {
                    local::supports_vision(selection, &config)
                }))
    }

    #[must_use]
    pub fn loaded(&self, selection: &ModelSelection) -> bool {
        if selection.provider != Provider::Local {
            return true;
        }
        let Ok(config) = self.local_config() else {
            return true;
        };
        let load = LoadSignature::new(&config, selection.model.as_deref());
        self.local
            .try_lock()
            .map(|loaded| {
                loaded
                    .as_ref()
                    .is_some_and(|loaded| loaded.matches(selection, &load))
            })
            .unwrap_or(true)
    }

    fn local_config(&self) -> anyhow::Result<LocalConfig> {
        Ok(self.providers.read()?.local.clone())
    }

    pub fn unload(&self) -> bool {
        self.local
            .try_lock()
            .map(|mut loaded| loaded.take().is_some())
            .unwrap_or(false)
    }

    #[tracing::instrument(skip_all)]
    pub async fn load_model(&self, selection: &ModelSelection) -> anyhow::Result<()> {
        if selection.provider == Provider::Local {
            self.local(selection).await?;
        }
        Ok(())
    }

    #[tracing::instrument(
        target = "koharu_metrics",
        name = "model_run",
        skip_all,
        fields(
            stage = "translation",
            provider = %selection.provider,
            model = selection.model.as_deref().unwrap_or("provider_default"),
            target_language = request.target_language.tag(),
            outcome = tracing::field::Empty,
        ),
    )]
    pub async fn translate(
        &self,
        selection: &ModelSelection,
        generation: GenerationConfig,
        mut request: TranslationRequest,
    ) -> anyhow::Result<(&'static str, Vec<String>)> {
        let _metric = tracing::info_span!(
            target: "koharu_metrics",
            "translation_request",
            provider = %selection.provider,
            model = selection.model.as_deref().unwrap_or("provider_default"),
            target_language = request.target_language.tag(),
        );
        let provider = selection.provider;
        let provider_id: &'static str = provider.into();
        if request.segments.is_empty() {
            tracing::Span::current().record("outcome", "skipped");
            return Ok((provider_id, request.segments));
        }

        let generation = generation.for_model(selection);

        if self.supports_vision(selection, &generation) {
            request.prepare_image()?;
        } else {
            request.remove_image();
        }

        let expected = request.segments.len();
        let translated = if provider == Provider::Local {
            let config = self.local_config()?;
            let runtime = config.runtime_for(selection.model.as_deref().unwrap_or_default());
            self.local(selection)
                .await?
                .translate(request, generation, &runtime)
                .await?
        } else {
            let providers = self.providers.read()?.clone();
            remote::translate(&self.client, &providers, selection, &generation, &request).await?
        };
        if translated.len() != expected {
            return Err(Error::SegmentCount {
                provider: provider_id,
                expected,
                actual: translated.len(),
            }
            .into());
        }
        tracing::Span::current().record("outcome", "completed");
        Ok((provider_id, translated))
    }

    #[tracing::instrument(skip_all)]
    pub async fn models() -> anyhow::Result<Vec<Model>> {
        let providers = ProvidersConfig::load()?;
        let providers = providers.read()?.clone();
        let client = koharu_runtime::http_client()?;
        let mut models = local::models(&providers.local);
        models.extend(remote::models(&client, &providers).await);
        Ok(models)
    }

    async fn local(&self, selection: &ModelSelection) -> Result<Arc<LocalTranslator>> {
        let config = self.local_config()?;
        let load = LoadSignature::new(&config, selection.model.as_deref());
        let mut loaded = self.local.lock().await;
        if loaded
            .as_ref()
            .is_none_or(|loaded| !loaded.matches(selection, &load))
        {
            *loaded = Some(LoadedLocal {
                model: selection.model.clone(),
                quantization: selection.quantization.clone(),
                load,
                translator: Arc::new(
                    LocalTranslator::load(self.device.clone(), selection, &config).await?,
                ),
            });
        }
        Ok(Arc::clone(
            &loaded
                .as_ref()
                .expect("local translator was loaded")
                .translator,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn local_selection(model: &str) -> ModelSelection {
        ModelSelection {
            provider: Provider::Local,
            model: Some(model.to_owned()),
            quantization: None,
            vision: true,
            reasoning: true,
        }
    }

    fn translator(local: LocalConfig) -> Translator {
        Translator::from_config(
            Device::cpu(),
            koharu_config::Config::memory(ProvidersConfig {
                local,
                ..ProvidersConfig::default()
            }),
        )
        .unwrap()
    }

    #[test]
    fn local_vision_requires_capability_and_generation_setting() {
        let translator = translator(LocalConfig::default());

        assert!(translator.supports_vision(
            &local_selection("gemma4-e2b-it"),
            &GenerationConfig {
                vision: Some(true),
                ..GenerationConfig::default()
            }
        ));
        assert!(!translator.supports_vision(
            &local_selection("gemma4-e2b-it"),
            &GenerationConfig {
                vision: Some(false),
                ..GenerationConfig::default()
            }
        ));
        assert!(!translator.supports_vision(
            &local_selection("lfm2.5-1.2b-instruct"),
            &GenerationConfig {
                vision: Some(true),
                ..GenerationConfig::default()
            }
        ));
    }

    #[test]
    fn a_registered_model_without_a_projector_has_no_vision() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.gguf");
        std::fs::write(&path, b"GGUF").unwrap();
        let translator = translator(LocalConfig {
            models: vec![CustomModel {
                id: "custom".to_owned(),
                name: "Custom".to_owned(),
                path,
                projector: None,
            }],
            ..LocalConfig::default()
        });

        assert!(!translator.supports_vision(
            &local_selection("custom"),
            &GenerationConfig {
                vision: Some(true),
                ..GenerationConfig::default()
            }
        ));
    }
}
