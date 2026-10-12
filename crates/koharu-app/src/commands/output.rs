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
use std::{collections::HashSet, io::Write as _, sync::Arc};
use tauri::{AppHandle, Manager as _, State, WebviewWindow, ipc::IpcResponse};
use tauri_runtime_cef::CefRuntime;

use super::processing::{Job, JobChannel, JobId, JobKind, JobState, Processing};
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
    /// 페이지 전체를 하나의 CBZ(ZIP) 아카이브로 묶는다. 항목은 PNG이고,
    /// 이름은 나머지 형식과 같은 `pattern` 줄기를 쓴다.
    Cbz,
}

impl ExportFormat {
    /// 페이지마다 파일 하나를 쓰는 형식인가.
    ///
    /// `Cbz`만 페이지 전체를 하나의 아카이브로 묶으므로, 형식 목록을 페이지별
    /// 작업으로 펼치기 전에 이 구분이 필요하다.
    #[must_use]
    fn writes_page_files(self) -> bool {
        !matches!(self, Self::Cbz)
    }

    /// 형식별 하위 폴더 이름. 페이지마다 파일을 쓰는 형식만 하위 폴더를 갖는다 —
    /// 아카이브는 한 폴더에 하나만 놓이므로 나눌 것이 없다.
    #[must_use]
    fn subfolder(self) -> Option<&'static str> {
        match self {
            Self::Png => Some("png"),
            Self::Psd => Some("psd"),
            Self::Cbz => None,
        }
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
        match (self.subfolders, format.subfolder()) {
            (true, Some(name)) => root.join(name),
            _ => root.to_owned(),
        }
    }
}

/// 네이티브 폴더 선택창을 띄우고 내보내기 Job을 시작한다.
///
/// 선택창을 띄우기 **전에** 다른 작업이 도는지 확인한다. 거절할 것이면
/// 사용자를 폴더 선택으로 붙잡아두기 전에 거절해야 한다.
#[tracing::instrument(
    target = "koharu_metrics",
    name = "export",
    skip_all,
    fields(origin = "user", export_formats = ?options.formats),
)]
#[tauri::command]
#[specta::specta]
pub async fn export_pages(
    window: WebviewWindow<CefRuntime>,
    pages: Vec<EntityId>,
    options: ExportOptions,
) -> std::result::Result<Option<JobId>, Error> {
    let handle = window.app_handle().clone();
    if !handle.state::<Processing>().stops.lock().is_empty() {
        return Err(anyhow::anyhow!("another process is already running").into());
    }
    let Some(directory) = rfd::AsyncFileDialog::new()
        .set_parent(&window)
        .pick_folder()
        .await
        .map(|directory| directory.path().to_owned())
    else {
        return Ok(None);
    };
    Ok(Some(start_export(handle, directory, pages, options).await?))
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
    let (project_name, snapshot) = {
        let project = project.project.lock().await;
        let project = project.as_ref().context("no project is open")?;
        (project.name.clone(), project.snapshot())
    };
    // 빈 목록은 "전체 페이지"라는 뜻이다. 선택 목록이 오면 프로젝트 순서로 다시
    // 세우고, 이 프로젝트에 없는 id는 버린다.
    let pages = resolve_export_pages(
        snapshot.pages().map(|page| page.id()).collect(),
        (!pages.is_empty()).then_some(pages),
    )?;
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

    // 아카이브는 페이지별 파일이 아니라 페이지 전체를 담는 한 파일이므로,
    // 페이지 단위로 펼칠 수 있는 형식과 그렇지 않은 형식을 나눈다.
    let page_formats: Vec<ExportFormat> = options
        .formats
        .iter()
        .copied()
        .filter(|format| format.writes_page_files())
        .collect();
    let archive = options
        .formats
        .iter()
        .any(|format| !format.writes_page_files());

    for format in &page_formats {
        let target = options.directory(&directory, *format);
        tokio::fs::create_dir_all(&target)
            .await
            .with_context(|| format!("failed to create {}", target.display()))?;
    }

    // 진행률의 분모는 페이지별 파일 수에 아카이브 항목 수를 더한 값이다.
    let total = page_count
        .saturating_mul(page_formats.len())
        .saturating_add(if archive { page_count } else { 0 });
    let completed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let renderer = desktop.renderer();
    let rasterizer = desktop.rasterizer().await?;
    let units: Vec<_> = jobs
        .iter()
        .flat_map(|(page_id, stem)| {
            page_formats
                .iter()
                .map(move |format| (*page_id, stem.clone(), *format))
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
                    // `units`는 `writes_page_files()`를 통과한 형식만 담는다.
                    // 아카이브는 페이지 순서를 지켜야 해서 별도 패스가 쓴다.
                    ExportFormat::Cbz => unreachable!("the container format has its own pass"),
                }
                let done = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                progress(done, total, page_id);
                tracing::info!(
                    target: "koharu_metrics",
                    metric = "page_exported",
                    format = ?format,
                );
                Ok(())
            }
        })
        .buffer_unordered(4)
        .try_collect::<Vec<_>>()
        .await?;

    if archive {
        let destination = directory.join(archive_file_name(&project_name));
        let counter = Arc::clone(&completed);
        write_cbz(
            destination,
            &jobs,
            &snapshot,
            &renderer,
            Arc::clone(&rasterizer),
            &stop,
            |page| {
                let done = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                progress(done, total, page);
            },
        )
        .await?;
    }

    tracing::info!(
        target: "koharu_metrics",
        metric = "export",
        export_formats = ?options.formats,
        page_count,
    );
    Ok(())
}

fn resolve_export_pages(
    project_pages: Vec<EntityId>,
    selected_pages: Option<Vec<EntityId>>,
) -> Result<Vec<EntityId>> {
    let is_selection = selected_pages.is_some();
    let pages = match selected_pages {
        Some(selected_pages) => {
            let selected = selected_pages.into_iter().collect::<HashSet<_>>();
            project_pages
                .into_iter()
                .filter(|page| selected.contains(page))
                .collect()
        }
        None => project_pages,
    };
    if pages.is_empty() {
        let message = if is_selection {
            "none of the selected pages belong to this project"
        } else {
            "there are no pages to export"
        };
        return Err(anyhow::anyhow!(message));
    }
    Ok(pages)
}

/// 아카이브 파일 이름. 프로젝트 이름을 파일 이름 규칙으로 다듬어 쓴다.
///
/// 이름이 통째로 걸러지면(프로젝트 이름이 비었거나 금지 문자뿐이면) 고정
/// 이름으로 떨어진다. 확장자는 언제나 하나뿐이므로 붙일 자리가 하나다.
fn archive_file_name(project_name: &str) -> String {
    let stem = crate::commands::naming::sanitize_label(project_name);
    if stem.is_empty() {
        "export.cbz".to_owned()
    } else {
        format!("{stem}.cbz")
    }
}

/// 페이지들을 하나의 CBZ(ZIP)로 쓴다.
///
/// 페이지 순서가 곧 아카이브 항목 순서라서 병렬 스트림에 섞지 않는다. 그 대가로
/// PNG와 CBZ를 함께 고르면 같은 페이지를 두 번 렌더링한다 — 두 출력의 픽셀은
/// 같지만 하나는 파일로, 하나는 아카이브 항목으로 가므로 공유할 지점이 없다.
///
/// 중단되더라도 중앙 디렉터리는 쓴다. 잘린 ZIP보다는 그때까지 담긴 페이지를
/// 읽을 수 있는 ZIP이 낫고, 부분 결과를 남기는 것은 페이지 파일과 같은 규칙이다.
async fn write_cbz(
    destination: std::path::PathBuf,
    jobs: &[(EntityId, String)],
    snapshot: &Snapshot,
    renderer: &Renderer,
    rasterizer: Arc<Rasterizer>,
    stop: &StopToken,
    report: impl Fn(EntityId),
) -> Result<()> {
    let file = std::fs::File::create(&destination)
        .with_context(|| format!("failed to create {}", destination.display()))?;
    let mut archive = ::zip::ZipWriter::new(std::io::BufWriter::new(file));
    // 항목이 곧 PNG라서 이미 압축돼 있다. 다시 압축하는 것은 시간만 쓴다.
    let options = ::zip::write::SimpleFileOptions::default()
        .compression_method(::zip::CompressionMethod::Stored);
    for (page_id, stem) in jobs {
        if stop.stopped() {
            break;
        }
        let frame = renderer.render(snapshot, *page_id).await?;
        let image = rasterize(Arc::clone(&rasterizer), &frame, RasterOptions::default())
            .await?
            .image;
        let bytes = tokio::task::spawn_blocking(move || -> Result<Vec<u8>> {
            let mut bytes = Vec::new();
            PngEncoder::new_with_quality(&mut bytes, CompressionType::Best, FilterType::Adaptive)
                .write_image(
                    image.as_raw(),
                    image.width(),
                    image.height(),
                    ExtendedColorType::Rgba8,
                )?;
            Ok(bytes)
        })
        .await
        .context("archive worker stopped unexpectedly")??;
        // 항목 이름은 페이지 파일과 같은 줄기를 쓴다. 아카이브 안은 언제나 PNG다.
        archive.start_file(format!("{stem}.png"), options)?;
        archive.write_all(&bytes)?;
        report(*page_id);
        tracing::info!(
            target: "koharu_metrics",
            metric = "page_exported",
            format = ?ExportFormat::Cbz,
        );
    }
    archive.finish()?;
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

/// 내보내기를 백그라운드 Job으로 시작하고 그 id를 즉시 돌려준다.
///
/// `Processing.stops`의 단일 작업 제약을 그대로 쓴다. 내보내기와 파이프라인은
/// 둘 다 GPU를 많이 쓰므로 동시에 돌 이유가 없고, 이 제약 덕분에 스테이징
/// 디렉터리도 한 번에 하나만 존재한다.
pub async fn start_export(
    handle: AppHandle<CefRuntime>,
    directory: std::path::PathBuf,
    pages: Vec<EntityId>,
    options: ExportOptions,
) -> std::result::Result<JobId, Error> {
    // 패턴은 여기서 한 번 검증한다. 대화상자가 이미 막지만 HTTP API는
    // 신뢰 경계이고, 경로 이탈을 클라이언트 검증에만 맡길 수 없다.
    crate::commands::naming::Template::parse(&options.pattern)?;

    let id = JobId::new();
    let stop = StopToken::default();
    {
        let processing = handle.state::<Processing>();
        let mut stops = processing.stops.lock();
        if !stops.is_empty() {
            return Err(anyhow::anyhow!("another process is already running").into());
        }
        stops.insert(id, stop.clone());
    }
    let job = Job {
        id,
        kind: JobKind::Export,
        state: JobState::Running,
        completed: 0,
        total: 0,
        page: None,
        stage: None,
        model: None,
        error: None,
    };
    handle
        .state::<Processing>()
        .jobs
        .lock()
        .insert(id, job.clone());
    handle.state::<JobChannel>().publish(job);

    let task_handle = handle.clone();
    let task_stop = stop.clone();
    drop(tokio::spawn(async move {
        let progress_handle = task_handle.clone();
        let progress: Arc<dyn Fn(usize, usize, EntityId) + Send + Sync> =
            Arc::new(move |completed, total, page| {
                let job = {
                    let processing = progress_handle.state::<Processing>();
                    let mut jobs = processing.jobs.lock();
                    jobs.get_mut(&id).map(|job| {
                        job.completed = completed;
                        job.total = total;
                        job.page = Some(page);
                        job.clone()
                    })
                };
                if let Some(job) = job {
                    progress_handle.state::<JobChannel>().publish(job);
                }
            });
        let result = export_pages_to(
            directory,
            pages,
            options,
            progress,
            task_stop.clone(),
            task_handle.state::<CurrentProject>(),
            task_handle.state::<Desktop>(),
        )
        .await;
        let (stopped, error) = match result {
            Ok(()) => (task_stop.stopped(), None),
            Err(error) => {
                tracing::error!(%error, "export failed");
                (false, Some(format!("{error}")))
            }
        };
        task_handle.state::<Processing>().stops.lock().remove(&id);
        let job = task_handle
            .state::<Processing>()
            .jobs
            .lock()
            .remove(&id)
            .map(|mut job| {
                job.state = if stopped {
                    JobState::Stopped
                } else if error.is_some() {
                    JobState::Failed
                } else {
                    JobState::Finished
                };
                job.error = error;
                job
            });
        if let Some(job) = job {
            task_handle.state::<JobChannel>().publish(job);
        }
    }));
    Ok(id)
}

#[cfg(test)]
mod tests {
    use super::{archive_file_name, resolve_export_pages};
    use koharu_scene::EntityId;

    #[test]
    fn selected_pages_keep_project_order_and_ignore_duplicate_ids() {
        let project_pages = vec![EntityId::new(), EntityId::new(), EntityId::new()];
        let requested = vec![project_pages[2], project_pages[0], project_pages[2]];

        assert_eq!(
            resolve_export_pages(project_pages.clone(), Some(requested)).unwrap(),
            vec![project_pages[0], project_pages[2]],
        );
    }

    #[test]
    fn selected_pages_must_belong_to_the_project() {
        let project_page = EntityId::new();

        assert!(resolve_export_pages(vec![project_page], Some(vec![EntityId::new()])).is_err());
    }

    #[test]
    fn an_empty_selection_means_every_page() {
        let project_pages = vec![EntityId::new(), EntityId::new()];

        assert_eq!(
            resolve_export_pages(project_pages.clone(), None).unwrap(),
            project_pages,
        );
    }

    #[test]
    fn the_archive_keeps_the_project_name_but_not_its_separators() {
        assert_eq!(archive_file_name("My Manga"), "My Manga.cbz");
        assert_eq!(archive_file_name("vol/01"), "vol_01.cbz");
        assert_eq!(archive_file_name("   "), "export.cbz");
    }
}
