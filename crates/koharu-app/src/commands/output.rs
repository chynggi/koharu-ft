use anyhow::{Context as _, Result};
use futures::{StreamExt as _, TryStreamExt as _, stream};
use image::{
    ExtendedColorType, ImageEncoder as _,
    codecs::png::{CompressionType, FilterType, PngEncoder},
};
use koharu_pipeline::StopToken;
use koharu_psd::{PsdExportOptions, export_page};
use koharu_rasterizer::{Raster, RasterOptions, Rasterizer};
use koharu_renderer::{Frame, Renderer};
use koharu_scene::{AssetRole, EntityId, Snapshot};
use serde::Deserialize;
use specta::Type;
use std::sync::Arc;
use tauri::{Cef, State, WebviewWindow, ipc::IpcResponse};

use super::{Error, project::CurrentProject};
use koharu_desktop::Desktop;

const THUMBNAIL_EDGE: u32 = 128;

#[derive(Type)]
#[specta(transparent)]
pub struct ThumbnailBytes(#[specta(type = Vec<u8>)] Vec<u8>);

impl IpcResponse for ThumbnailBytes {
    fn body(self) -> tauri::Result<tauri::ipc::InvokeResponseBody> {
        Ok(self.0.into())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum ExportFormat {
    Png,
    Psd,
}

impl ExportFormat {
    #[must_use]
    pub fn extension(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Psd => "psd",
        }
    }

    /// 형식별 하위 폴더 이름. 확장자와 같지만 뜻이 달라 따로 둔다.
    #[must_use]
    pub fn subfolder(self) -> &'static str {
        self.extension()
    }
}

/// 한 번의 내보내기 실행이 받는 선택지.
#[derive(Clone, Debug, Deserialize, Type)]
pub struct ExportOptions {
    /// 최소 하나. 둘을 함께 주면 페이지마다 두 파일이 나온다.
    pub formats: Vec<ExportFormat>,
    /// `crate::commands::naming::Template`이 파싱하는 패턴.
    pub pattern: String,
    /// 형식이 둘일 때 `png/`, `psd/`로 나눌지.
    pub subfolders: bool,
}

impl ExportOptions {
    /// 대상 폴더 아래에서 이 형식이 쓰일 경로.
    fn directory(&self, root: &std::path::Path, format: ExportFormat) -> std::path::PathBuf {
        if self.subfolders {
            root.join(format.subfolder())
        } else {
            root.to_owned()
        }
    }
}

#[tauri::command]
#[specta::specta]
pub async fn export_pages(
    window: WebviewWindow<Cef>,
    pages: Vec<EntityId>,
    options: ExportOptions,
    project: State<'_, CurrentProject>,
    desktop: State<'_, Desktop>,
) -> std::result::Result<(), Error> {
    let Some(directory) = rfd::AsyncFileDialog::new()
        .set_parent(&window)
        .pick_folder()
        .await
        .map(|directory| directory.path().to_owned())
    else {
        return Ok(());
    };
    export_pages_to(
        directory,
        pages,
        options,
        Arc::new(|_, _, _| {}),
        StopToken::default(),
        project,
        desktop,
    )
    .await
}

/// [`export_pages`]의 코어. 네이티브 대화상자 대신 명시적 출력 폴더를 받는다.
///
/// `pages`가 비어 있으면 프로젝트의 모든 페이지가 대상이다. 진행률의 분모는
/// `pages × formats`지만 파일 이름의 번호는 형식과 무관한 페이지 순번이다 —
/// 두 값은 서로 다른 것을 센다.
pub async fn export_pages_to(
    directory: std::path::PathBuf,
    pages: Vec<EntityId>,
    options: ExportOptions,
    progress: Arc<dyn Fn(usize, usize, EntityId) + Send + Sync>,
    stop: StopToken,
    project: State<'_, CurrentProject>,
    desktop: State<'_, Desktop>,
) -> std::result::Result<(), Error> {
    if options.formats.is_empty() {
        return Err(anyhow::anyhow!("no export format was selected").into());
    }
    let template = crate::commands::naming::Template::parse(&options.pattern)?;
    let snapshot = {
        let project = project.project.lock().await;
        let project = project.as_ref().context("no project is open")?;
        project.snapshot()
    };
    let pages = if pages.is_empty() {
        snapshot.pages().map(|page| page.id()).collect()
    } else {
        pages
    };
    if pages.is_empty() {
        return Err(anyhow::anyhow!("there are no pages to export").into());
    }
    let page_count = pages.len();

    // 이름은 형식과 무관하게 페이지마다 한 번 정해진다. 충돌 해소도 여기서
    // 끝나야 PNG와 PSD가 같은 줄기를 공유한다.
    let mut names = crate::commands::naming::Names::default();
    let jobs = pages
        .into_iter()
        .enumerate()
        .map(|(index, page_id)| {
            let page = snapshot.page(page_id)?.page()?;
            let stem = template.render(index + 1, &page.label)?;
            Ok::<_, anyhow::Error>((page_id, names.unique(stem)))
        })
        .collect::<Result<Vec<_>>>()?;

    for format in &options.formats {
        let target = options.directory(&directory, *format);
        tokio::fs::create_dir_all(&target)
            .await
            .with_context(|| format!("failed to create {}", target.display()))?;
    }

    let total = page_count.saturating_mul(options.formats.len());
    let completed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let renderer = desktop.renderer();
    let rasterizer = desktop.rasterizer().await?;
    let units: Vec<_> = jobs
        .into_iter()
        .flat_map(|(page_id, stem)| {
            options
                .formats
                .iter()
                .map(move |format| (page_id, stem.clone(), *format))
                .collect::<Vec<_>>()
        })
        .collect();

    stream::iter(units)
        .map(|(page_id, stem, format)| {
            let renderer = renderer.clone();
            let rasterizer = Arc::clone(&rasterizer);
            let snapshot = snapshot.clone();
            let target = options.directory(&directory, format);
            let stop = stop.clone();
            let progress = Arc::clone(&progress);
            let completed = Arc::clone(&completed);
            async move {
                // 취소는 협조적이다. 이미 시작된 최대 4건은 마저 끝난다.
                if stop.stopped() {
                    return Ok::<_, anyhow::Error>(());
                }
                let frame = renderer.render(&snapshot, page_id).await?;
                match format {
                    ExportFormat::Png => {
                        let image =
                            rasterize(Arc::clone(&rasterizer), &frame, RasterOptions::default())
                                .await?
                                .image;
                        let path = target.join(format!("{stem}.png"));
                        tokio::task::spawn_blocking(move || -> Result<()> {
                            let file = std::fs::File::create(path)?;
                            PngEncoder::new_with_quality(
                                file,
                                CompressionType::Best,
                                FilterType::Adaptive,
                            )
                            .write_image(
                                image.as_raw(),
                                image.width(),
                                image.height(),
                                ExtendedColorType::Rgba8,
                            )?;
                            Ok(())
                        })
                        .await
                        .context("PNG export worker stopped unexpectedly")??;
                    }
                    ExportFormat::Psd => {
                        let bytes = export_page(
                            Arc::clone(&rasterizer),
                            &snapshot,
                            &frame,
                            &PsdExportOptions::default(),
                        )
                        .await?;
                        tokio::fs::write(target.join(format!("{stem}.psd")), bytes).await?;
                    }
                }
                let done = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                progress(done, total, page_id);
                Ok(())
            }
        })
        .buffer_unordered(4)
        .try_collect::<Vec<_>>()
        .await?;

    tracing::info!(
        target: "koharu_metrics",
        metric = "export",
        export_formats = ?options.formats,
        page_count,
    );
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub async fn get_thumbnail(
    page: EntityId,
    project: State<'_, CurrentProject>,
) -> std::result::Result<ThumbnailBytes, Error> {
    let snapshot = project
        .project
        .lock()
        .await
        .as_ref()
        .context("no project is open")?
        .snapshot();
    snapshot.page(page)?;
    let blob = snapshot
        .asset(page, &AssetRole::new("source")?)?
        .with_context(|| format!("page {page} has no source image"))?
        .blob;
    let bytes = snapshot.read_blob(blob).await?;
    let bytes = tokio::task::spawn_blocking(move || -> Result<Vec<u8>> {
        let image = image::load_from_memory(&bytes).context("failed to decode source image")?;
        if image.width() == 0 || image.height() == 0 {
            return Err(anyhow::anyhow!("source image is empty"));
        }
        let image = image.thumbnail(THUMBNAIL_EDGE, THUMBNAIL_EDGE).to_rgba8();
        let encoder = webp::Encoder::from_rgba(image.as_raw(), image.width(), image.height());
        Ok(encoder.encode(80.0).to_vec())
    })
    .await
    .context("thumbnail worker stopped unexpectedly")??;
    Ok(ThumbnailBytes(bytes))
}

pub async fn rendered_preview(
    renderer: &Renderer,
    rasterizer: Arc<Rasterizer>,
    snapshot: &Snapshot,
    page: EntityId,
) -> Result<Vec<u8>> {
    snapshot.page(page)?;
    let frame = renderer.render(snapshot, page).await?;
    let image = rasterize(rasterizer, &frame, RasterOptions::default())
        .await?
        .image;
    tokio::task::spawn_blocking(move || {
        let image = image::DynamicImage::ImageRgba8(image)
            .resize(1024, 1024, image::imageops::FilterType::Lanczos3)
            .to_rgba8();
        let encoder = webp::Encoder::from_rgba(image.as_raw(), image.width(), image.height());
        Ok::<_, anyhow::Error>(encoder.encode(85.0).to_vec())
    })
    .await
    .context("preview encode worker stopped unexpectedly")?
}

async fn rasterize(
    rasterizer: Arc<Rasterizer>,
    frame: &Frame,
    options: RasterOptions,
) -> Result<Raster> {
    let frame = frame.raster_frame()?;
    tokio::task::spawn_blocking(move || rasterizer.rasterize(&frame, options))
        .await
        .context("rasterizer worker stopped unexpectedly")?
        .map_err(Into::into)
}
