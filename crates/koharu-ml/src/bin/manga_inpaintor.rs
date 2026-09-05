use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use koharu_ml::lama::InpaintRequest;
use koharu_ml::manga_inpaintor::{MangaInpaintor, MangaSource};

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

    /// Path to the inpaintor weights. The pinned release is used when omitted.
    #[arg(long)]
    inpaintor_weights: Option<PathBuf>,

    /// Path to the line model weights. The pinned release is used when omitted.
    #[arg(long)]
    line_weights: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    let image = image::open(&cli.input)?;
    let mask = image::open(&cli.mask)?.to_luma8();

    koharu_ml::init().await?;

    let source = MangaSource {
        inpaintor: match cli.inpaintor_weights {
            Some(path) => koharu_ml::source::ComponentSource::LocalFile(path),
            None => koharu_ml::source::ComponentSource::Builtin,
        },
        line: match cli.line_weights {
            Some(path) => koharu_ml::source::ComponentSource::LocalFile(path),
            None => koharu_ml::source::ComponentSource::Builtin,
        },
    };
    let model = MangaInpaintor::load(koharu_ml::device(cli.cpu), &source).await?;
    let inpainted = model.inference(&image, &mask, &InpaintRequest::default())?;
    inpainted.save(cli.output)?;

    Ok(())
}
