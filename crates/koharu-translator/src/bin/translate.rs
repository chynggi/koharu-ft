#[cfg(not(target_os = "linux"))]
use std::env;

use anyhow::{Context, Result};
use clap::Parser;
use koharu_config::Config;
#[cfg(not(target_os = "linux"))]
use koharu_secrets::{ExposeSecret, SecretString};
use koharu_translator::{
    GenerationConfig, Language, ModelSelection, Provider, ProvidersConfig, TranslationRequest,
    Translator,
};
use url::Url;

#[derive(Debug, Parser)]
#[command(about = "Run Koharu's configured translator")]
struct Args {
    #[arg(long)]
    provider: String,

    #[arg(long)]
    model: Option<String>,

    #[arg(long)]
    api_key: Option<String>,

    #[arg(long)]
    base_url: Option<Url>,

    #[arg(long)]
    target: Language,

    #[arg(long)]
    instructions: Option<String>,

    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    json: bool,

    #[arg(required = true, num_args = 1..)]
    segments: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let provider = args
        .provider
        .parse::<Provider>()
        .with_context(|| format!("unknown translation provider '{}'", args.provider))?;
    if provider == Provider::Local {
        koharu_ml::init().await?;
    }
    prepare_secret(&args, provider)?;

    let mut providers = ProvidersConfig::default();
    if let Some(base_url) = args.base_url.clone() {
        match provider {
            Provider::OpenAiCompatible => providers.openai_compatible.base_url = Some(base_url),
            Provider::LmStudio => providers.lm_studio.base_url = Some(base_url),
            Provider::DeepL => providers.deepl.base_url = Some(base_url),
            _ => anyhow::bail!("--base-url is not supported by this provider"),
        }
    }
    let model = args.model.clone().or_else(|| {
        (provider == Provider::Local).then(|| {
            ModelSelection::default()
                .model
                .expect("local translation has a default model")
        })
    });
    if model.is_none()
        && !matches!(
            provider,
            Provider::DeepL | Provider::GoogleCloudTranslation | Provider::Caiyun
        )
    {
        anyhow::bail!("--model is required for {provider}");
    }
    let translator =
        Translator::from_config(koharu_ml::device(args.cpu), Config::memory(providers))?;
    let selection = ModelSelection {
        provider,
        model,
        quantization: None,
        vision: true,
        reasoning: true,
    };
    let mut request = TranslationRequest::new(args.segments, args.target);
    if let Some(instructions) = args.instructions {
        request = request.with_instructions(instructions);
    }
    let (_, translated) = translator
        .translate(&selection, GenerationConfig::default(), request)
        .await?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&translated)?);
    } else {
        for (index, segment) in translated.iter().enumerate() {
            println!("[{}] {segment}", index + 1);
        }
    }
    Ok(())
}

fn prepare_secret(args: &Args, provider: Provider) -> Result<()> {
    let Some(key) = provider.secret_key() else {
        return Ok(());
    };
    let variable = key
        .environment_variable()
        .expect("translation provider credentials define an environment variable");

    #[cfg(target_os = "linux")]
    {
        if args
            .api_key
            .as_deref()
            .is_some_and(|key| !key.trim().is_empty())
        {
            anyhow::bail!(
                "--api-key is unavailable on Linux; set {variable} before starting Koharu"
            );
        }
        if provider.credential_required() && koharu_secrets::get(key)?.is_none() {
            anyhow::bail!("{variable} is required");
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        let value = args
            .api_key
            .clone()
            .filter(|key| !key.trim().is_empty())
            .or_else(|| env::var(variable).ok().filter(|key| !key.trim().is_empty()));
        if let Some(value) = value {
            koharu_secrets::set(key, &SecretString::from(value))?;
        } else if provider.credential_required()
            && koharu_secrets::get(key)?.is_none_or(|value| value.expose_secret().trim().is_empty())
        {
            anyhow::bail!("--api-key, {variable}, or a stored API key is required");
        }
    }

    Ok(())
}
