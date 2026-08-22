use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use koharu_ml::lama::{InpaintRequest, LaMa};

#[derive(Debug, Parser)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    input: PathBuf,

    #[arg(short, long, value_name = "FILE")]
    mask: PathBuf,

    #[arg(short, long, value_name = "FILE")]
    output: PathBuf,

    #[arg(long, default_value_t = false)]
    cpu: bool,

    /// Read the weights as a TorchScript archive. The default is safetensors.
    #[arg(long)]
    torchscript: bool,

    /// Path to the weights file. The pinned repository is used when omitted.
    #[arg(long)]
    weights: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    let image = image::open(&cli.input)?;
    let mask = image::open(&cli.mask)?.to_luma8();

    koharu_ml::init().await?;

    let source = match cli.weights {
        Some(path) => koharu_ml::source::ComponentSource::LocalFile(path),
        None => koharu_ml::source::ComponentSource::Builtin,
    };
    let format = if cli.torchscript {
        koharu_ml::lama::WeightsFormat::TorchScript
    } else {
        koharu_ml::lama::WeightsFormat::SafeTensors
    };
    let model = LaMa::load(koharu_ml::device(cli.cpu), &source, format).await?;
    let inpainted = model.inference(&image, &mask, &InpaintRequest::default())?;
    inpainted.save(cli.output)?;

    Ok(())
}
