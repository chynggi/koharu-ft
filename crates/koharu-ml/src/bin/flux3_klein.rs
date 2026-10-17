use std::{path::PathBuf, time::Instant};

use anyhow::{Context, Result};
use clap::Parser;
use koharu_ml::flux3_klein::{
    ComponentSource, DEFAULT_MAX_PIXELS, DEFAULT_REQUEST, Flux3KleinInpaint,
    Flux3KleinInpaintOptions, Flux3KleinSource, Flux3PromptUpsampler,
};

#[derive(Debug, Parser)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    input: PathBuf,

    #[arg(short, long, value_name = "FILE")]
    mask: PathBuf,

    #[arg(
        short,
        long,
        value_name = "FILE",
        required_unless_present = "caption_only"
    )]
    output: Option<PathBuf>,

    #[arg(long, value_name = "TEXT", default_value = DEFAULT_REQUEST)]
    request: String,

    /// Prints the upsampler caption and exits (gate G2).
    #[arg(long, conflicts_with = "no_upsampler")]
    caption_only: bool,

    /// Sends the request as the caption. Comparison only; the pipeline always
    /// runs the upsampler.
    #[arg(long)]
    no_upsampler: bool,

    #[arg(long, default_value_t = 16)]
    steps: usize,

    #[arg(long, default_value_t = 0.8)]
    strength: f64,

    #[arg(long, default_value_t = -1)]
    seed: i64,

    #[arg(long)]
    padding_mask_crop: Option<u32>,

    #[arg(long, default_value_t = DEFAULT_MAX_PIXELS)]
    max_pixels: u32,

    #[arg(long, value_name = "FILE")]
    transformer: Option<PathBuf>,

    #[arg(long, value_name = "FILE")]
    text_encoder: Option<PathBuf>,

    #[arg(long, value_name = "FILE")]
    vae: Option<PathBuf>,

    #[arg(long, value_name = "FILE")]
    upsampler: Option<PathBuf>,

    #[arg(long, value_name = "FILE")]
    upsampler_mmproj: Option<PathBuf>,

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
    let input = image::open(&cli.input)?;
    let mask = image::open(&cli.mask)?;

    koharu_ml::init().await?;

    let source = Flux3KleinSource {
        transformer: component(cli.transformer.as_ref())?,
        text_encoder: component(cli.text_encoder.as_ref())?,
        vae: component(cli.vae.as_ref())?,
        upsampler: component(cli.upsampler.as_ref())?,
        upsampler_mmproj: component(cli.upsampler_mmproj.as_ref())?,
    };
    source.validate()?;
    let device = koharu_ml::device(cli.cpu);

    let caption = if cli.no_upsampler {
        cli.request.clone()
    } else {
        let upsampler = Flux3PromptUpsampler::load(device.clone(), &source).await?;
        let started = Instant::now();
        let caption = upsampler
            .caption(vec![input.clone()], cli.request.clone())
            .await?
            .into_iter()
            .next()
            .context("the upsampler returned no caption")?;
        eprintln!(
            "phase=caption seconds={:.2} (load + generate + release)",
            started.elapsed().as_secs_f64()
        );
        caption
    };
    println!("{caption}");
    if cli.caption_only {
        return Ok(());
    }

    let started = Instant::now();
    let pipeline = Flux3KleinInpaint::load(device, &source).await?;
    eprintln!("phase=load seconds={:.2}", started.elapsed().as_secs_f64());
    let options = Flux3KleinInpaintOptions {
        num_inference_steps: cli.steps,
        strength: cli.strength,
        seed: cli.seed,
        padding_mask_crop: cli.padding_mask_crop,
        max_pixels: cli.max_pixels,
    };
    // The first call warms the backend; only the second one is measured.
    pipeline.inference(&caption, &input, &mask, &options)?;
    let started = Instant::now();
    let output = pipeline.inference(&caption, &input, &mask, &options)?;
    eprintln!(
        "phase=inpaint seconds={:.2}",
        started.elapsed().as_secs_f64()
    );
    output.save(cli.output.context("--output is required")?)?;
    Ok(())
}
