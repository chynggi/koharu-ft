use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use koharu_ml::manga_inpainting::MangaInpainting;
use koharu_runtime::{Backend, Device};

#[derive(Debug, Parser)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    input: PathBuf,

    #[arg(short, long, value_name = "FILE")]
    mask: PathBuf,

    #[arg(short, long, value_name = "FILE")]
    output: PathBuf,

    /// Runs on the CPU, whose noise matches PyTorch's CPU generator.
    #[arg(long)]
    cpu: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    let image = image::open(cli.input)?.to_rgb8();
    let mask = image::open(cli.mask)?.to_luma8();
    let gpu = koharu_runtime::device().await?;
    let cpu = Device {
        name: "CPU".to_owned(),
        index: 0,
        description: "CPU".to_owned(),
        backend: Backend::Cpu,
        integrated: true,
    };
    let device = if cli.cpu { &cpu } else { gpu };
    let model = MangaInpainting::load(device).await?;
    model.inference(&image, &mask)?.save(cli.output)?;
    Ok(())
}
