use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use koharu_ml::flux2_klein::{
    ComponentSource, DEFAULT_MAX_PIXELS, Flux2Klein, Flux2KleinInpaint, Flux2KleinInpaintOptions,
    Flux2KleinOptions, Flux2KleinSource,
};

#[derive(Debug, Parser)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    input: Option<PathBuf>,

    #[arg(short, long, value_name = "FILE", requires = "input")]
    mask: Option<PathBuf>,

    #[arg(short, long, value_name = "FILE")]
    output: PathBuf,

    #[arg(short, long, value_name = "TEXT")]
    prompt: String,

    #[arg(long, value_name = "FILE")]
    reference: Option<PathBuf>,

    #[arg(long, default_value_t = 4)]
    steps: usize,

    #[arg(long, default_value_t = 0.8)]
    strength: f64,

    #[arg(long, default_value_t = -1)]
    seed: i64,

    #[arg(long)]
    padding_mask_crop: Option<u32>,

    #[arg(long, default_value_t = DEFAULT_MAX_PIXELS)]
    max_pixels: u32,

    /// Replaces the pinned transformer with a checkpoint already on disk.
    #[arg(long, value_name = "FILE")]
    transformer: Option<PathBuf>,

    /// Replaces the pinned text encoder with a checkpoint already on disk.
    #[arg(long, value_name = "FILE")]
    text_encoder: Option<PathBuf>,

    /// Replaces the pinned VAE with a checkpoint already on disk.
    #[arg(long, value_name = "FILE")]
    vae: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    cpu: bool,
}

fn component(path: Option<&PathBuf>) -> Result<ComponentSource> {
    match path {
        None => Ok(ComponentSource::Builtin),
        Some(path) => Ok(ComponentSource::LocalFile(std::path::absolute(path)?)),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    let input = cli.input.as_ref().map(image::open).transpose()?;
    let mask = cli.mask.as_ref().map(image::open).transpose()?;
    let reference = cli.reference.as_ref().map(image::open).transpose()?;

    koharu_ml::init().await?;

    let source = Flux2KleinSource {
        transformer: component(cli.transformer.as_ref())?,
        text_encoder: component(cli.text_encoder.as_ref())?,
        vae: component(cli.vae.as_ref())?,
    };
    source.validate()?;

    let output = if let Some(mask) = mask.as_ref() {
        let input = input
            .as_ref()
            .context("--input is required when --mask is provided")?;
        let pipeline = Flux2KleinInpaint::load(koharu_ml::device(cli.cpu), &source).await?;
        let options = Flux2KleinInpaintOptions {
            num_inference_steps: cli.steps,
            strength: cli.strength,
            seed: cli.seed,
            padding_mask_crop: cli.padding_mask_crop,
            max_pixels: cli.max_pixels,
        };
        pipeline.inference(&cli.prompt, input, reference.as_ref(), mask, &options)?
    } else {
        let pipeline = Flux2Klein::load(koharu_ml::device(cli.cpu), &source).await?;
        let options = Flux2KleinOptions {
            num_inference_steps: i32::try_from(cli.steps)?,
            seed: cli.seed,
            max_pixels: cli.max_pixels,
            ..Flux2KleinOptions::default()
        };
        let references = input.into_iter().chain(reference).collect::<Vec<_>>();
        let generated = pipeline
            .inference(&references, &cli.prompt, &options)?
            .into_iter()
            .next()
            .context("FLUX.2 Klein returned no image")?;
        image::DynamicImage::ImageRgb8(generated)
    };
    output.save(cli.output)?;

    Ok(())
}
