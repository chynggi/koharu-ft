use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use koharu_ml::lama::InpaintRequest;
use koharu_ml::mi_gan::MiGan;

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

    /// Path to the weights file. The pinned release is used when omitted.
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
    let model = MiGan::load(koharu_ml::device(cli.cpu), &source).await?;
    let inpainted = model.inference(&image, &mask, &InpaintRequest::default())?;
    inpainted.save(cli.output)?;

    Ok(())
}
