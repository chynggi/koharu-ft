use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use koharu_ml::paddle_ocr_vl::{PaddleOCRVL, PaddleOCRVLAdapter, PaddleOCRVLTask};
use koharu_ml::source::ComponentSource;

#[derive(Debug, Parser)]
#[command(about = "Run PaddleOCR-VL-1.6 element recognition")]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    input: PathBuf,

    #[arg(short, long, value_enum, default_value_t = Task::Ocr)]
    task: Task,

    #[arg(long, default_value_t = false)]
    cpu: bool,

    /// Apply the pinned manga LoRA adapter and its fine-tuned vision tower.
    #[arg(long, default_value_t = false)]
    adapter: bool,

    /// Path to the adapter weights. Implies --adapter.
    #[arg(long, value_name = "FILE")]
    adapter_weights: Option<PathBuf>,

    /// Path to the peft adapter config. Implies --adapter.
    #[arg(long, value_name = "FILE")]
    adapter_config: Option<PathBuf>,

    /// Path to the fine-tuned vision tower. Implies --adapter.
    #[arg(long, value_name = "FILE")]
    adapter_tower: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Task {
    Ocr,
    Table,
    Formula,
    Chart,
    Spotting,
    Seal,
}

impl From<Task> for PaddleOCRVLTask {
    fn from(value: Task) -> Self {
        match value {
            Task::Ocr => Self::Ocr,
            Task::Table => Self::Table,
            Task::Formula => Self::Formula,
            Task::Chart => Self::Chart,
            Task::Spotting => Self::Spotting,
            Task::Seal => Self::Seal,
        }
    }
}

/// The pinned adapter, with whatever files the user pointed at instead.
fn adapter(cli: &Cli) -> Option<PaddleOCRVLAdapter> {
    if !cli.adapter
        && cli.adapter_weights.is_none()
        && cli.adapter_config.is_none()
        && cli.adapter_tower.is_none()
    {
        return None;
    }
    let source = |path: &Option<PathBuf>| match path {
        Some(path) => ComponentSource::LocalFile(path.clone()),
        None => ComponentSource::Builtin,
    };
    Some(PaddleOCRVLAdapter {
        config: source(&cli.adapter_config),
        weights: source(&cli.adapter_weights),
        tower: source(&cli.adapter_tower),
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let filter = tracing_subscriber::EnvFilter::builder()
        .with_default_directive(tracing::Level::INFO.into())
        .from_env_lossy();
    tracing_subscriber::fmt().with_env_filter(filter).init();
    let cli = Cli::parse();
    let image = image::open(&cli.input)?;
    koharu_ml::init().await?;
    let device = koharu_ml::device(cli.cpu);
    let model = PaddleOCRVL::load(device, adapter(&cli).as_ref()).await?;
    let result = model.inference(&image, cli.task.into())?;
    println!("{}", result.text);
    Ok(())
}
