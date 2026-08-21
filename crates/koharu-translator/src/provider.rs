use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use specta::Type;
use strum::{Display, EnumIter, EnumString, IntoStaticStr, VariantArray};

use crate::{
    local::LocalConfig,
    remote::{
        AtlasCloudConfig, CaiyunConfig, ClaudeConfig, DeepLConfig, DeepSeekConfig, GeminiConfig,
        GoogleCloudConfig, GrokConfig, LmStudioConfig, MiniMaxConfig, OpenAiCompatibleConfig,
        OpenAiConfig, OpenRouterConfig,
    },
};

macro_rules! define_providers {
    ($(
        $variant:ident {
            id: $id:literal,
            name: $name:literal,
            field: $field:ident,
            config: $config:ty,
        }
    )+) => {
        #[derive(
            Clone,
            Copy,
            Debug,
            Display,
            EnumIter,
            EnumString,
            Eq,
            Hash,
            IntoStaticStr,
            PartialEq,
            Serialize,
            Deserialize,
            Type,
            VariantArray,
        )]
        pub enum Provider {
            $(
                #[serde(rename = $id)]
                #[strum(serialize = $id)]
                $variant,
            )+
        }

        impl Provider {
            #[must_use]
            pub const fn name(self) -> &'static str {
                match self {
                    $(Self::$variant => $name,)+
                }
            }
        }

        #[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
        #[serde(tag = "provider", content = "settings")]
        pub enum ProviderConfig {
            $(
                #[serde(rename = $id)]
                $variant($config),
            )+
        }

        impl ProviderConfig {
            #[must_use]
            pub const fn provider(&self) -> Provider {
                match self {
                    $(Self::$variant(_) => Provider::$variant,)+
                }
            }
        }

        #[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Type)]
        #[serde(default)]
        pub struct ProvidersConfig {
            $(
                #[serde(rename = $id)]
                pub $field: $config,
            )+
        }

        impl ProvidersConfig {
            #[must_use]
            pub fn entries(&self) -> Vec<ProviderConfig> {
                vec![$(ProviderConfig::$variant(self.$field.clone()),)+]
            }

            pub fn from_entries(entries: impl IntoIterator<Item = ProviderConfig>) -> Result<Self> {
                let mut config = Self::default();
                let mut seen = std::collections::HashSet::new();
                for entry in entries {
                    let provider = entry.provider();
                    if !seen.insert(provider) {
                        bail!("duplicate provider configuration for {provider}");
                    }
                    match entry {
                        $(ProviderConfig::$variant(settings) => config.$field = settings,)+
                    }
                }
                if seen.len() != Provider::VARIANTS.len() {
                    let missing = Provider::VARIANTS
                        .iter()
                        .filter(|provider| !seen.contains(provider))
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(", ");
                    bail!("missing provider configurations: {missing}");
                }
                Ok(config)
            }
        }
    };
}

define_providers! {
    Local {
        id: "local",
        name: "Local",
        field: local,
        config: LocalConfig,
    }
    AtlasCloud {
        id: "atlas-cloud",
        name: "Atlas Cloud",
        field: atlas_cloud,
        config: AtlasCloudConfig,
    }
    OpenAi {
        id: "openai",
        name: "OpenAI",
        field: openai,
        config: OpenAiConfig,
    }
    Gemini {
        id: "gemini",
        name: "Gemini",
        field: gemini,
        config: GeminiConfig,
    }
    Claude {
        id: "claude",
        name: "Claude",
        field: claude,
        config: ClaudeConfig,
    }
    Grok {
        id: "grok",
        name: "Grok",
        field: grok,
        config: GrokConfig,
    }
    MiniMax {
        id: "minimax",
        name: "MiniMax",
        field: minimax,
        config: MiniMaxConfig,
    }
    DeepSeek {
        id: "deepseek",
        name: "DeepSeek",
        field: deepseek,
        config: DeepSeekConfig,
    }
    OpenAiCompatible {
        id: "openai-compatible",
        name: "OpenAI-compatible",
        field: openai_compatible,
        config: OpenAiCompatibleConfig,
    }
    OpenRouter {
        id: "openrouter",
        name: "OpenRouter",
        field: openrouter,
        config: OpenRouterConfig,
    }
    LmStudio {
        id: "lm-studio",
        name: "LM Studio",
        field: lm_studio,
        config: LmStudioConfig,
    }
    DeepL {
        id: "deepl",
        name: "DeepL",
        field: deepl,
        config: DeepLConfig,
    }
    GoogleCloudTranslation {
        id: "google-cloud-translation",
        name: "Google Cloud Translation",
        field: google_cloud_translation,
        config: GoogleCloudConfig,
    }
    Caiyun {
        id: "caiyun",
        name: "Caiyun",
        field: caiyun,
        config: CaiyunConfig,
    }
}

impl Provider {
    #[must_use]
    pub const fn secret_key(self) -> Option<koharu_secrets::SecretKey<'static>> {
        let (name, variable) = match self {
            Self::Local => return None,
            Self::AtlasCloud => ("atlas-cloud", "ATLASCLOUD_API_KEY"),
            Self::OpenAi => ("openai", "OPENAI_API_KEY"),
            Self::Gemini => ("gemini", "GEMINI_API_KEY"),
            Self::Claude => ("claude", "ANTHROPIC_API_KEY"),
            Self::Grok => ("grok", "XAI_API_KEY"),
            Self::MiniMax => ("minimax", "MINIMAX_API_KEY"),
            Self::DeepSeek => ("deepseek", "DEEPSEEK_API_KEY"),
            Self::OpenAiCompatible => ("openai-compatible", "OPENAI_COMPATIBLE_API_KEY"),
            Self::OpenRouter => ("openrouter", "OPENROUTER_API_KEY"),
            Self::LmStudio => ("lm-studio", "LM_STUDIO_API_TOKEN"),
            Self::DeepL => ("deepl", "DEEPL_API_KEY"),
            Self::GoogleCloudTranslation => ("google-cloud-translation", "GOOGLE_CLOUD_API_KEY"),
            Self::Caiyun => ("caiyun", "CAIYUN_API_KEY"),
        };
        Some(koharu_secrets::SecretKey::environment(name, variable))
    }

    #[must_use]
    pub const fn credential_required(self) -> bool {
        !matches!(self, Self::Local | Self::OpenAiCompatible | Self::LmStudio)
    }
}

impl ProvidersConfig {
    pub fn load() -> anyhow::Result<koharu_config::Config<Self>> {
        koharu_config::load("providers")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_credentials_use_canonical_environment_variables() {
        let expected = [
            (Provider::AtlasCloud, "ATLASCLOUD_API_KEY", true),
            (Provider::OpenAi, "OPENAI_API_KEY", true),
            (Provider::Gemini, "GEMINI_API_KEY", true),
            (Provider::Claude, "ANTHROPIC_API_KEY", true),
            (Provider::Grok, "XAI_API_KEY", true),
            (Provider::MiniMax, "MINIMAX_API_KEY", true),
            (Provider::DeepSeek, "DEEPSEEK_API_KEY", true),
            (
                Provider::OpenAiCompatible,
                "OPENAI_COMPATIBLE_API_KEY",
                false,
            ),
            (Provider::OpenRouter, "OPENROUTER_API_KEY", true),
            (Provider::LmStudio, "LM_STUDIO_API_TOKEN", false),
            (Provider::DeepL, "DEEPL_API_KEY", true),
            (
                Provider::GoogleCloudTranslation,
                "GOOGLE_CLOUD_API_KEY",
                true,
            ),
            (Provider::Caiyun, "CAIYUN_API_KEY", true),
        ];
        assert!(Provider::Local.secret_key().is_none());
        assert!(!Provider::Local.credential_required());
        for (provider, variable, required) in expected {
            assert_eq!(
                provider.secret_key().unwrap().environment_variable(),
                Some(variable)
            );
            assert_eq!(provider.credential_required(), required);
        }
    }
}
