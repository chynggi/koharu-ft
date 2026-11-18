use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use koharu_ml::mi_gan::MiGan;

#[derive(Debug, Parser)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    input: PathBuf,

    #[arg(short, long, value_name = "FILE")]
    mask: PathBuf,

    #[arg(short, long, value_name = "FILE")]
    output: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    let image = image::open(cli.input)?.to_rgb8();
    let mask = image::open(cli.mask)?.to_luma8();
    let device = koharu_runtime::device().await?;
    let model = MiGan::load(device).await?;
    model.inference(&image, &mask)?.save(cli.output)?;
    Ok(())
}
