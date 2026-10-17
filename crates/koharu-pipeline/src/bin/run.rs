use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context as _, Result};
use clap::{Parser, ValueEnum};
use koharu_config::Config;
use koharu_pipeline::{
    Committer, DetectionModel, Flux2KleinConfig, Flux3KleinConfig, InpaintingModel,
    KoharuLayoutRFDetrSeg2XLConfig, LaMaConfig, MangaInpaintorConfig, MiGanConfig, OcrModel,
    Operation, PaddleOcrVl1_6MangaLoraConfig, Pipeline, PipelineConfig, PowerPaintConfig, Progress,
    Request, RoremMixedConfig, Scope, StageOutput, TranslationConfig,
};
use koharu_rasterizer::{RasterOptions, Rasterizer};
use koharu_renderer::Renderer;
use koharu_scene::{AssetInput, AssetMetadata, AssetRole, At, PageDraft, Session};
use koharu_translator::{GenerationConfig, Language, ModelSelection, Provider, ProvidersConfig};

#[derive(Debug, Parser)]
#[command(version, about = "Run Koharu's complete in-process pipeline")]
struct Arguments {
    /// Repeat to process several pages in one run. A single input renders to
    /// `--output` as a file; several inputs render one PNG per page into
    /// `--output`, which is then a directory.
    #[arg(short, long, value_name = "INPUT", num_args = 1.., required = true)]
    input: Vec<PathBuf>,

    #[arg(short, long, value_name = "OUTPUT")]
    output: PathBuf,

    #[arg(long, value_enum, default_value = "koharu-layout-rfdetr-seg-2xl")]
    detection: DetectionChoice,

    #[arg(long, value_enum, default_value = "paddleocr-vl-1.6")]
    ocr: OcrChoice,

    #[arg(long, value_enum, default_value = "lama")]
    inpainting: InpaintingChoice,

    #[arg(long, default_value = "en-US")]
    target_language: Language,

    #[arg(long)]
    translation_instructions: Option<String>,

    #[arg(long, default_value = "gemma4-12b-it")]
    llm: String,

    #[arg(long)]
    cpu: bool,
}

struct SessionCommitter<'a>(&'a mut Session);

#[async_trait::async_trait]
impl Committer for SessionCommitter<'_> {
    async fn commit(&mut self, output: StageOutput) -> Result<koharu_scene::Snapshot> {
        Ok(self.0.commit(output.patch).await?.snapshot)
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DetectionChoice {
    #[value(name = "koharu-layout-rfdetr-seg-2xl")]
    KoharuLayoutRFDetrSeg2XL,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OcrChoice {
    #[value(name = "paddleocr-vl-1.6")]
    PaddleOcrVl1_6,
    #[value(name = "paddleocr-vl-1.6-manga-lora")]
    PaddleOcrVl1_6MangaLora,
    #[value(name = "manga-ocr")]
    MangaOcr,
    #[value(name = "baberu-ocr")]
    BaberuOcr,
    #[value(name = "hayai-ocr")]
    HayaiOcr,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum InpaintingChoice {
    #[value(name = "lama")]
    LaMa,
    #[value(name = "mi-gan")]
    MiGan,
    #[value(name = "manga-inpaintor")]
    MangaInpaintor,
    #[value(name = "aot-inpainting")]
    AotInpainting,
    #[value(name = "flux2-klein")]
    Flux2Klein,
    #[value(name = "flux3-klein")]
    Flux3Klein,
    #[value(name = "rorem-mixed")]
    RoremMixed,
    #[value(name = "powerpaint")]
    PowerPaint,
}

impl Arguments {
    fn pipeline_config(&self) -> PipelineConfig {
        PipelineConfig {
            detection: match self.detection {
                DetectionChoice::KoharuLayoutRFDetrSeg2XL => {
                    DetectionModel::KoharuLayoutRFDetrSeg2XL(
                        KoharuLayoutRFDetrSeg2XLConfig::default(),
                    )
                }
            },
            ocr: match self.ocr {
                OcrChoice::PaddleOcrVl1_6 => OcrModel::PaddleOcrVl1_6,
                OcrChoice::PaddleOcrVl1_6MangaLora => {
                    OcrModel::PaddleOcrVl1_6MangaLora(PaddleOcrVl1_6MangaLoraConfig::default())
                }
                OcrChoice::MangaOcr => OcrModel::MangaOcr,
                OcrChoice::BaberuOcr => OcrModel::BaberuOcr,
                OcrChoice::HayaiOcr => OcrModel::HayaiOcr,
            },
            translation: TranslationConfig {
                model: ModelSelection {
                    provider: Provider::Local,
                    model: Some(self.llm.clone()),
                    quantization: None,
                    vision: true,
                    reasoning: true,
                },
                generation: GenerationConfig::default(),
                target_language: self.target_language,
                instructions: self.translation_instructions.clone(),
                combine_pages: false,
                max_pages_per_request: 5,
            },
            inpainting: match self.inpainting {
                InpaintingChoice::LaMa => InpaintingModel::LaMa(LaMaConfig::default()),
                InpaintingChoice::MiGan => InpaintingModel::MiGan(MiGanConfig::default()),
                InpaintingChoice::MangaInpaintor => {
                    InpaintingModel::MangaInpaintor(MangaInpaintorConfig::default())
                }
                InpaintingChoice::AotInpainting => InpaintingModel::AotInpainting {},
                InpaintingChoice::Flux2Klein => {
                    InpaintingModel::Flux2Klein(Flux2KleinConfig::default())
                }
                InpaintingChoice::Flux3Klein => {
                    InpaintingModel::Flux3Klein(Flux3KleinConfig::default())
                }
                InpaintingChoice::RoremMixed => {
                    InpaintingModel::RoremMixed(RoremMixedConfig::default())
                }
                InpaintingChoice::PowerPaint => {
                    InpaintingModel::PowerPaint(PowerPaintConfig::default())
                }
            },
            processor: Default::default(),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let arguments = Arguments::parse();
    init_tracing();
    initialize_with_retry().await;

    let mut session = Session::memory().await?;
    let mut pages = Vec::with_capacity(arguments.input.len());
    for input in &arguments.input {
        let source = fs::read(input)
            .with_context(|| format!("failed to read {}", input.display()))?;
        let decoded = image::load_from_memory(&source).context("failed to decode input image")?;
        let name = input
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("input");
        let mut page = None;
        let patch = session.snapshot().patch(|edit| {
            let id = edit.add_page(
                PageDraft::new(
                    name,
                    f64::from(decoded.width()),
                    f64::from(decoded.height()),
                ),
                At::End,
            )?;
            edit.set_asset(
                id,
                &AssetRole::new("source")?,
                AssetInput::new(
                    Arc::<[u8]>::from(source),
                    image_media_type(input),
                    AssetMetadata {
                        width: Some(decoded.width()),
                        height: Some(decoded.height()),
                        attributes: BTreeMap::new(),
                    },
                ),
            )?;
            page = Some(id);
            Ok(())
        })?;
        session.commit(patch).await?;
        pages.push(page.expect("page ID is assigned by the edit"));
    }

    let device = koharu_ml::device(arguments.cpu);
    let pipeline = Pipeline::from_config(
        Config::memory(arguments.pipeline_config()),
        Config::memory(ProvidersConfig::default()),
        device,
    )?;
    let snapshot = session.snapshot();
    let mut committer = SessionCommitter(&mut session);
    let report = pipeline
        .execute(
            snapshot,
            Request {
                operation: Operation::Full,
                scope: Scope::Pages(pages.clone()),
                progress: Some(Arc::new(|event| {
                    if let Progress::Finished { stage, elapsed, .. } = event {
                        eprintln!("{stage} finished in {:.2}s", elapsed.as_secs_f64());
                    }
                })),
                ..Request::default()
            },
            &mut committer,
        )
        .await?;
    eprintln!("pipeline finished in {:.2}s", report.elapsed.as_secs_f64());

    let renderer = Renderer::new()?;
    let rasterizer = Rasterizer::new()?;
    let single = pages.len() == 1;
    if !single {
        fs::create_dir_all(&arguments.output)
            .with_context(|| format!("failed to create {}", arguments.output.display()))?;
    }
    for (index, page) in pages.iter().enumerate() {
        let render_started = Instant::now();
        let snapshot = session.snapshot();
        let frame = renderer.render(&snapshot, *page).await?;
        let raster = rasterizer.rasterize(&frame.raster_frame()?, RasterOptions::default())?;
        let target = if single {
            arguments.output.clone()
        } else {
            let stem = arguments.input[index]
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("page");
            arguments.output.join(format!("{stem}.png"))
        };
        raster
            .image
            .save(&target)
            .with_context(|| format!("failed to write {}", target.display()))?;
        eprintln!(
            "rendered {} in {:.2}s",
            target.display(),
            render_started.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

/// The gate reads `koharu_metrics` events (`model_unload`) out of this binary's
/// stderr, so it needs the same subscriber shape the desktop app installs.
fn init_tracing() {
    use tracing_subscriber::{Layer as _, layer::SubscriberExt as _};
    tracing::subscriber::set_global_default(
        tracing_subscriber::registry().with(
            tracing_subscriber::fmt::layer()
                .with_writer(std::io::stderr)
                .with_filter(
                    tracing_subscriber::filter::EnvFilter::builder()
                        .with_default_directive(tracing::Level::INFO.into())
                        .from_env_lossy(),
                ),
        ),
    )
    .expect("failed to set the global tracing subscriber");
}

async fn initialize_with_retry() {
    let mut delay = Duration::from_secs(1);
    let mut attempt = 0_u64;
    loop {
        attempt += 1;
        match koharu_ml::init().await {
            Ok(()) => return,
            Err(error) => {
                let jitter = Duration::from_millis((attempt.wrapping_mul(137)) % 251);
                let wait = delay + jitter;
                eprintln!(
                    "runtime initialization attempt {attempt} failed: {error}; retrying in {:.1}s",
                    wait.as_secs_f64()
                );
                tokio::time::sleep(wait).await;
                delay = delay.saturating_mul(2).min(Duration::from_secs(30));
            }
        }
    }
}

fn image_media_type(path: &std::path::Path) -> &'static str {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        _ => "image/png",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_flags_select_models() {
        let arguments = Arguments::try_parse_from([
            "run",
            "--input",
            "input.png",
            "--output",
            "output.png",
            "--ocr",
            "manga-ocr",
            "--inpainting",
            "flux2-klein",
        ])
        .unwrap();
        assert!(matches!(
            arguments.pipeline_config().ocr,
            OcrModel::MangaOcr
        ));
        assert!(matches!(
            arguments.pipeline_config().inpainting,
            InpaintingModel::Flux2Klein(_)
        ));
    }

    #[test]
    fn several_inputs_are_accepted_for_one_output_directory() {
        let arguments = Arguments::try_parse_from([
            "run",
            "--input",
            "one.jpg",
            "--input",
            "two.jpg",
            "--output",
            "rendered",
        ])
        .unwrap();
        assert_eq!(
            arguments.input,
            [PathBuf::from("one.jpg"), PathBuf::from("two.jpg")]
        );
    }
}
