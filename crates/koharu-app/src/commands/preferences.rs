use std::fmt;

use anyhow::Result;
use koharu_pipeline::PipelineConfig;
use koharu_renderer::TypesettingConfig;
use koharu_translator::{Language, Model, Provider, ProviderConfig, ProvidersConfig};
use serde::{Deserialize, Serialize};
use specta::Type;

use super::Error;

#[derive(Clone, Debug, Serialize, Type)]
pub struct Preferences {
    pub pipeline: PipelineConfig,
    pub providers: ProviderPreferences,
    pub typesetting: TypesettingConfig,
    pub languages: Vec<LanguageChoice>,
}

impl Preferences {
    pub fn load() -> Result<Self> {
        let pipeline = PipelineConfig::load()?;
        let providers = ProvidersConfig::load()?;
        let typesetting = TypesettingConfig::load()?;
        let pipeline = pipeline.read()?;
        let providers = providers.read()?;
        let typesetting = typesetting.read()?;
        Ok(Self {
            pipeline: pipeline.clone(),
            providers: ProviderPreferences::from_config(&providers)?,
            typesetting: typesetting.clone(),
            languages: Language::ALL
                .iter()
                .map(|language| LanguageChoice {
                    tag: language.tag().to_owned(),
                    name: language.to_string(),
                })
                .collect(),
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Type)]
pub struct ProviderPreferences {
    pub entries: Vec<ProviderPreference>,
}

#[derive(Clone, Debug, Deserialize, Serialize, Type)]
pub struct ProviderPreference {
    pub name: String,
    pub config: ProviderConfig,
    pub credential: Option<CredentialInput>,
}

impl ProviderPreferences {
    fn from_config(config: &ProvidersConfig) -> Result<Self> {
        let entries = config
            .entries()
            .into_iter()
            .map(|config| {
                let provider = config.provider();
                let credential = provider
                    .secret_key()
                    .map(CredentialInput::load)
                    .transpose()?;
                Ok(ProviderPreference {
                    name: provider.name().to_owned(),
                    config,
                    credential,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { entries })
    }

    fn into_config(self) -> Result<ProvidersConfig> {
        let mut configs = Vec::with_capacity(self.entries.len());
        let mut credentials = Vec::with_capacity(self.entries.len().saturating_sub(1));
        for entry in self.entries {
            let provider = entry.config.provider();
            match entry.credential {
                None if provider == Provider::Local => {}
                Some(credential) if provider != Provider::Local => {
                    credentials.push((provider.secret_key().unwrap(), credential));
                }
                None => anyhow::bail!("missing credential input for {provider}"),
                Some(_) => anyhow::bail!("local translation does not accept credentials"),
            }
            configs.push(entry.config);
        }
        let config = ProvidersConfig::from_entries(configs)?;
        // Registered GGUF files and llama.cpp overrides are rejected here rather
        // than when a translation run tries to load them.
        config.local.validate()?;
        for (key, credential) in credentials {
            credential.save(key)?;
        }
        Ok(config)
    }
}

#[derive(Clone, Default, Deserialize, Serialize, Type)]
pub struct CredentialInput {
    pub configured: bool,
    pub editable: bool,
    pub environment_variable: Option<String>,
    pub value: Option<String>,
    pub clear: bool,
}

impl fmt::Debug for CredentialInput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CredentialInput")
            .field("configured", &self.configured)
            .field("editable", &self.editable)
            .field("environment_variable", &self.environment_variable)
            .field("value", &self.value.as_ref().map(|_| "[REDACTED]"))
            .field("clear", &self.clear)
            .finish()
    }
}

impl CredentialInput {
    fn load(key: koharu_secrets::SecretKey<'_>) -> Result<Self> {
        Ok(Self {
            configured: koharu_secrets::get(key)?.is_some(),
            editable: !koharu_secrets::is_read_only(),
            environment_variable: koharu_secrets::is_read_only()
                .then(|| key.environment_variable().map(str::to_owned))
                .flatten(),
            value: None,
            clear: false,
        })
    }

    fn save(self, key: koharu_secrets::SecretKey<'_>) -> Result<()> {
        if koharu_secrets::is_read_only() {
            if self.clear || self.value.is_some() {
                anyhow::bail!(
                    "{} is managed by the environment and cannot be changed from Koharu",
                    key.environment_variable().unwrap_or("this credential")
                );
            }
            return Ok(());
        }
        if self.clear {
            koharu_secrets::delete(key)?;
        } else if let Some(value) = self.value {
            if value.trim().is_empty() {
                koharu_secrets::delete(key)?;
            } else {
                koharu_secrets::set(key, &value.into())?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct LanguageChoice {
    pub tag: String,
    pub name: String,
}

#[tracing::instrument(
    target = "koharu_metrics",
    name = "preferences_saved",
    skip_all,
    fields(setting = "application")
)]
#[tauri::command]
#[specta::specta]
pub async fn save_preferences(
    mut pipeline: PipelineConfig,
    providers: ProviderPreferences,
    typesetting: TypesettingConfig,
) -> std::result::Result<Preferences, Error> {
    remember_pipeline_profiles(&mut pipeline);
    let providers = providers.into_config()?;
    let pipeline_config = PipelineConfig::load()?;
    let providers_config = ProvidersConfig::load()?;
    let typesetting_config = TypesettingConfig::load()?;
    {
        let mut current = pipeline_config.write()?;
        *current = pipeline;
        current.save()?;
    }
    {
        let mut current = providers_config.write()?;
        *current = providers;
        current.save()?;
    }
    {
        let mut current = typesetting_config.write()?;
        *current = typesetting;
        current.save()?;
    }
    let preferences = Preferences::load()?;
    tracing::info!(
        target: "koharu_metrics",
        metric = "preference_changed",
        setting = "application",
    );
    Ok(preferences)
}

fn remember_pipeline_profiles(config: &mut PipelineConfig) {
    let koharu_pipeline::DetectionModel::KoharuLayoutRFDetrSeg2XL(settings) = &config.detection;
    config.processor.koharu_layout_rfdetr_seg_2xl = Some(settings.clone());
    if let koharu_pipeline::InpaintingModel::Flux2Klein(settings) = &config.inpainting {
        config.processor.flux2_klein = Some(settings.clone());
    }
    if let koharu_pipeline::InpaintingModel::RoremMixed(settings) = &config.inpainting {
        config.processor.rorem_mixed = Some(settings.clone());
    }
    if let koharu_pipeline::InpaintingModel::PowerPaint(settings) = &config.inpainting {
        config.processor.powerpaint = Some(settings.clone());
    }
}

#[tauri::command]
#[specta::specta]
pub async fn get_preferences() -> std::result::Result<Preferences, Error> {
    Ok(Preferences::load()?)
}

#[tauri::command]
#[specta::specta]
pub async fn get_translation_models() -> std::result::Result<Vec<Model>, Error> {
    Ok(koharu_translator::Translator::models().await?)
}
