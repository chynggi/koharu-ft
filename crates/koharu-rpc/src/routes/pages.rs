//! Page listing, selection, ordering, import, and export.
//!
//! Import and export each come in two shapes, and which one a caller wants is
//! decided by **where the files have to live**, not by preference:
//!
//! - `/pages/import` and `/pages/export` take server-side paths. That is right
//!   for the desktop window, where the server and the user are the same machine
//!   and the bytes are already on disk.
//! - `/pages/import/upload` and `/pages/export/download` move the bytes over
//!   HTTP. That is the only thing that can work for a remote browser: the source
//!   images live on the user's machine and the rendered output has to get back
//!   there, so no path either side could name would refer to the same file.

use std::io::Write as _;
use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::Context as _;
use axum::Json;
use axum::body::Bytes;
use axum::extract::{ConnectInfo, DefaultBodyLimit, Multipart, Path, State};
use axum::http::header;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::Router;
use tokio::io::AsyncWriteExt as _;
use koharu_scene::{AssetInput, AssetMetadata, AssetRole, At, EntityId, PageDraft};
use serde::Deserialize;
use tauri::Manager as _;

use koharu_app::commands::canvas::CanvasChannel;
use koharu_app::commands::editing;
use koharu_app::commands::import;
use koharu_app::commands::lifecycle::{self, PageImportSource, PageSelection};
use koharu_app::commands::output::{self, ExportOptions};
use koharu_app::commands::processing::{JobId, Processing};
use koharu_app::commands::project::{CurrentProject, Page, PageSummary};

use crate::AppState;
use crate::error::{ApiResult, ipc_bytes};

/// 브라우저용 ZIP이 만들어지는 임시 디렉터리.
///
/// 다운로드가 2단계가 되면서 임시 디렉터리가 요청보다 오래 살아야 한다.
/// 단일 작업 제약 덕에 동시에 하나뿐이므로 맵이 아니라 슬롯 하나면 된다.
/// 새 내보내기가 시작되면 이전 것이 교체되며 `TempDir`의 Drop이 지운다.
///
/// `parking_lot`이 아니라 `std::sync::Mutex`인 것은 `koharu-rpc`가
/// `parking_lot`에 의존하지 않기 때문이다. 잠금 구간에 `await`가 없으므로
/// 표준 뮤텍스로 충분하다.
#[derive(Default)]
pub struct ExportStaging(std::sync::Mutex<Option<(JobId, tempfile::TempDir)>>);

impl ExportStaging {
    fn put(&self, job: JobId, directory: tempfile::TempDir) {
        *self.0.lock().expect("the export staging lock is never poisoned") = Some((job, directory));
    }

    /// 이 job의 스테이징을 꺼내 소유권을 넘긴다. 호출자가 놓으면 지워진다.
    fn take(&self, job: JobId) -> Option<tempfile::TempDir> {
        let mut slot = self.0.lock().expect("the export staging lock is never poisoned");
        match slot.take() {
            Some((held, directory)) if held == job => Some(directory),
            other => {
                *slot = other;
                None
            }
        }
    }
}

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/pages", get(get_pages))
        .route("/page", get(get_page))
        .route("/page/select", post(select_page))
        .route("/pages/import", post(import_pages))
        // The body limit is lifted rather than raised: a manga import can be a
        // multi-gigabyte CBZ, and there is no honest number between "a few
        // pages" and "a whole series". Memory stays bounded regardless because
        // every field is streamed to disk a chunk at a time; the resource this
        // spends is temp-directory space, and reaching the route at all already
        // requires the API token on any non-loopback bind.
        .route(
            "/pages/import/upload",
            post(import_upload).layer(DefaultBodyLimit::disable()),
        )
        .route("/pages/import/dialog", post(import_dialog))
        .route("/pages/export", post(export_pages))
        .route("/pages/export/dialog", post(export_dialog))
        .route("/pages/export/download", post(export_download))
        .route("/pages/export/download/{job}", get(export_download_archive))
        .route("/pages/{id}/thumbnail", get(get_thumbnail))
        .route("/page/rename", post(rename_page))
        .route("/pages/delete", post(delete_pages))
        .route("/page/move", post(move_page))
}

async fn get_pages(State(app): State<AppState>) -> ApiResult<Json<Vec<PageSummary>>> {
    Ok(Json(
        lifecycle::get_pages(app.state::<CurrentProject>()).await?,
    ))
}

async fn get_page(State(app): State<AppState>) -> ApiResult<Json<Option<Page>>> {
    Ok(Json(
        lifecycle::get_page(app.state::<CurrentProject>()).await?,
    ))
}

#[derive(Deserialize)]
struct SelectPageRequest {
    page: EntityId,
}

async fn select_page(
    State(app): State<AppState>,
    Json(request): Json<SelectPageRequest>,
) -> ApiResult<Json<PageSelection>> {
    Ok(Json(
        lifecycle::select_page(
            app.state::<koharu_desktop::Desktop>(),
            request.page,
            app.state::<CurrentProject>(),
            app.state::<CanvasChannel>(),
        )
        .await?,
    ))
}

#[derive(Deserialize)]
struct ImportPagesRequest {
    files: Vec<String>,
}

async fn import_pages(
    State(app): State<AppState>,
    Json(request): Json<ImportPagesRequest>,
) -> ApiResult<Json<Vec<PageSummary>>> {
    reject_while_processing(&app)?;
    let files: Vec<PathBuf> = request
        .files
        .into_iter()
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .collect();
    if files.is_empty() {
        return Err(anyhow::anyhow!("none of the provided files exist").into());
    }
    commit_imported_pages(&app, decode_paths(files).await?).await
}

/// Import uploaded bytes by first writing them to a temporary directory and
/// then running the ordinary path-based import over them.
///
/// The detour through disk is deliberate. `import::import` dispatches on the
/// file extension and hands ZIP/CBZ, RAR and PDF sources to extractors that
/// each take a `&Path`. A bytes-based dispatch would therefore be a *second*
/// copy of that four-way branch, free to drift from the first. Writing the
/// upload out costs one pass over bytes that just crossed a network, and buys
/// a single import implementation for every format.
async fn import_upload(
    State(app): State<AppState>,
    mut multipart: Multipart,
) -> ApiResult<Json<Vec<PageSummary>>> {
    reject_while_processing(&app)?;
    let staging = tempfile::tempdir().context("failed to create a staging directory")?;
    let mut files = Vec::new();
    while let Some(mut field) = multipart
        .next_field()
        .await
        .context("failed to read the next uploaded file")?
    {
        // Take only the final component: the name is caller-controlled and is
        // about to become a path, so anything that could climb out of the
        // staging directory has to be dropped here rather than sanitized.
        let Some(name) = field
            .file_name()
            .map(std::path::Path::new)
            .and_then(std::path::Path::file_name)
            .map(|name| name.to_string_lossy().into_owned())
        else {
            continue;
        };
        let path = staging.path().join(&name);
        let mut file = tokio::fs::File::create(&path)
            .await
            .with_context(|| format!("failed to stage the uploaded file {name}"))?;
        // Streamed rather than `field.bytes()`: a single CBZ can be larger than
        // it is reasonable to hold in memory, and the whole point of staging is
        // that it never has to be.
        while let Some(chunk) = field
            .chunk()
            .await
            .with_context(|| format!("failed to read the uploaded file {name}"))?
        {
            file.write_all(&chunk)
                .await
                .with_context(|| format!("failed to stage the uploaded file {name}"))?;
        }
        file.flush()
            .await
            .with_context(|| format!("failed to stage the uploaded file {name}"))?;
        files.push(path);
    }
    if files.is_empty() {
        return Err(anyhow::anyhow!("the upload contained no files").into());
    }
    let pages = decode_paths(files).await?;
    // Held until here so the staging directory outlives the decode; dropping it
    // removes the files.
    drop(staging);
    commit_imported_pages(&app, pages).await
}

#[derive(Deserialize)]
struct ImportDialogRequest {
    source: PageImportSource,
}

/// Import through the desktop window's native file dialog.
///
/// Loopback only, and the guard is not a formality: the dialog opens in the
/// server process, so for a remote caller it would appear on a machine they
/// cannot see — and in the container that display is a headless Xvfb, which
/// means the request would simply never return. Browsers use
/// `/pages/import/upload` instead.
async fn import_dialog(
    State(app): State<AppState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    Json(request): Json<ImportDialogRequest>,
) -> ApiResult<Json<Vec<PageSummary>>> {
    let window = require_local_window(&app, peer)?;
    lifecycle::import_pages(
        request.source,
        window,
        app.state::<koharu_desktop::Desktop>(),
        app.state::<CurrentProject>(),
        app.state::<Processing>(),
        app.state::<CanvasChannel>(),
    )
    .await?;
    Ok(Json(
        lifecycle::get_pages(app.state::<CurrentProject>()).await?,
    ))
}

#[derive(Deserialize)]
struct ExportDialogRequest {
    pages: Vec<EntityId>,
    options: ExportOptions,
}

/// 네이티브 폴더 선택은 `koharu-app`에 있다. `rfd`가 그 크레이트의 의존성이고
/// `koharu-rpc`의 것이 아니므로, 선택창을 여기로 옮기면 의존성이 하나 늘어난다.
/// 단일 작업 가드도 선택창 바로 옆에 있는 편이 낫다.
async fn export_dialog(
    State(app): State<AppState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    Json(request): Json<ExportDialogRequest>,
) -> ApiResult<Json<Option<JobId>>> {
    let window = require_local_window(&app, peer)?;
    Ok(Json(
        output::export_pages(window, request.pages, request.options).await?,
    ))
}

fn require_local_window(
    app: &AppState,
    peer: SocketAddr,
) -> ApiResult<tauri::WebviewWindow<tauri::Cef>> {
    if !peer.ip().is_loopback() {
        return Err(anyhow::anyhow!(
            "a native dialog can only be opened for a caller on this machine"
        )
        .into());
    }
    Ok(app
        .get_webview_window("main")
        .context("there is no desktop window to open a dialog in")?)
}

fn reject_while_processing(app: &AppState) -> ApiResult<()> {
    if !app.state::<Processing>().stops.lock().is_empty() {
        return Err(anyhow::anyhow!("pages cannot be imported while processing is running").into());
    }
    Ok(())
}

async fn decode_paths(files: Vec<PathBuf>) -> ApiResult<Vec<import::Page>> {
    Ok(tokio::task::spawn_blocking(move || import::import(files))
        .await
        .context("page import worker stopped unexpectedly")??)
}

async fn commit_imported_pages(
    app: &AppState,
    pages: Vec<import::Page>,
) -> ApiResult<Json<Vec<PageSummary>>> {
    let app = app.clone();
    let desktop = app.state::<koharu_desktop::Desktop>();
    let canvas_channel = app.state::<CanvasChannel>();
    let (commit, page) = {
        let current = app.state::<CurrentProject>();
        let mut project = current.project.lock().await;
        let project = project
            .as_mut()
            .context("no project is open")?;
        let source = AssetRole::new("source")?;
        let patch = project.snapshot().patch(|edit| {
            for imported in pages {
                let page = edit.add_page(
                    PageDraft::new(
                        imported.name,
                        f64::from(imported.width),
                        f64::from(imported.height),
                    ),
                    At::End,
                )?;
                edit.set_asset(
                    page,
                    &source,
                    AssetInput::new(
                        imported.bytes,
                        imported.format.to_mime_type(),
                        AssetMetadata {
                            width: Some(imported.width),
                            height: Some(imported.height),
                            attributes: Default::default(),
                        },
                    ),
                )?;
            }
            Ok(())
        })?;
        let commit = project.session.commit(patch).await?;
        project.record(vec![commit.revision]);
        project.reconcile_page();
        (commit, project.active_page())
    };
    desktop.synchronize(&commit.snapshot, page, &commit).await?;
    canvas_channel.publish(desktop.canvas_state());
    Ok(Json(lifecycle::get_pages(app.state::<CurrentProject>()).await?))
}

#[derive(Deserialize)]
struct RenamePageRequest {
    page: EntityId,
    label: String,
}

async fn rename_page(
    State(app): State<AppState>,
    Json(request): Json<RenamePageRequest>,
) -> ApiResult<()> {
    editing::rename_page(
        request.page,
        request.label,
        app.state::<koharu_desktop::Desktop>(),
        app.state::<CurrentProject>(),
        app.state::<CanvasChannel>(),
    )
    .await?;
    Ok(())
}

#[derive(Deserialize)]
struct DeletePagesRequest {
    pages: Vec<EntityId>,
}

async fn delete_pages(
    State(app): State<AppState>,
    Json(request): Json<DeletePagesRequest>,
) -> ApiResult<()> {
    editing::delete_pages(
        request.pages,
        app.state::<koharu_desktop::Desktop>(),
        app.state::<CurrentProject>(),
        app.state::<CanvasChannel>(),
    )
    .await?;
    Ok(())
}

#[derive(Deserialize)]
struct ExportPagesRequest {
    pages: Vec<EntityId>,
    options: ExportOptions,
    directory: String,
}

async fn export_pages(
    State(app): State<AppState>,
    Json(request): Json<ExportPagesRequest>,
) -> ApiResult<Json<JobId>> {
    let directory = PathBuf::from(request.directory);
    if !directory.is_dir() {
        return Err(anyhow::anyhow!("the export directory does not exist").into());
    }
    Ok(Json(
        output::start_export(app.clone(), directory, request.pages, request.options).await?,
    ))
}

#[derive(Deserialize)]
struct ExportDownloadRequest {
    pages: Vec<EntityId>,
    options: ExportOptions,
}

/// 브라우저용 내보내기를 임시 디렉터리에 시작한다.
///
/// 요청 하나로 렌더링까지 끝내면 응답이 마지막에야 나가 진행률을 보낼 길이
/// 없다. 그래서 여기서는 Job만 시작하고, 클라이언트가 job이 끝나는 것을 SSE로
/// 본 뒤 `GET /pages/export/download/{job}`으로 ZIP을 받는다.
async fn export_download(
    State(app): State<AppState>,
    Json(request): Json<ExportDownloadRequest>,
) -> ApiResult<Json<JobId>> {
    let staging = tempfile::tempdir().context("failed to create a staging directory")?;
    let job = output::start_export(
        app.clone(),
        staging.path().to_owned(),
        request.pages,
        request.options,
    )
    .await?;
    app.state::<ExportStaging>().put(job, staging);
    Ok(Json(job))
}

/// 끝난 내보내기의 스테이징을 ZIP으로 넘기고 지운다.
async fn export_download_archive(
    State(app): State<AppState>,
    Path(job): Path<JobId>,
) -> ApiResult<axum::response::Response> {
    // 없는 스테이징은 404다. `ApiError`로 흘려보내면 전부 500이 되는데,
    // 취소되었거나 실패한 내보내기를 받으러 온 것과 서버가 고장난 것을
    // 클라이언트가 구분할 수 없게 된다.
    let Some(staging) = app.state::<ExportStaging>().take(job) else {
        return Ok((
            axum::http::StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": "there is no finished export waiting for this job",
            })),
        )
            .into_response());
    };
    let root = staging.path().to_owned();
    let archive = tokio::task::spawn_blocking(move || archive_directory(&root))
        .await
        .context("export archive worker stopped unexpectedly")??;
    drop(staging);
    Ok((
        [
            (header::CONTENT_TYPE, "application/zip"),
            (
                header::CONTENT_DISPOSITION,
                "attachment; filename=\"koharu-export.zip\"",
            ),
        ],
        Bytes::from(archive),
    )
        .into_response())
}

/// `export_pages_to`가 방금 쓴 파일들을 ZIP으로 묶는다.
///
/// 형식별 하위 폴더를 쓰면 출력이 한 겹 더 깊어지므로 재귀해야 한다. 엔트리
/// 이름은 `root` 기준 상대 경로라 압축을 풀면 폴더 구조가 그대로 살아난다.
fn archive_directory(root: &std::path::Path) -> anyhow::Result<Vec<u8>> {
    let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    // Deflate earns little on PNG and a lot on PSD. One method for both keeps
    // this to a single path; the cost next to rendering the pages is noise.
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    let mut entries = Vec::new();
    collect_files(root, &mut entries)?;
    if entries.is_empty() {
        anyhow::bail!("the export produced no files");
    }
    entries.sort();
    for path in entries {
        let name = path
            .strip_prefix(root)
            .context("an export file escaped the staging directory")?
            .to_string_lossy()
            // ZIP은 언제나 '/'를 쓴다. Windows의 '\'를 그대로 두면 풀 때
            // 폴더가 아니라 이름에 역슬래시가 든 파일이 된다.
            .replace('\\', "/");
        let bytes = std::fs::read(&path)
            .with_context(|| format!("failed to read the rendered page {name}"))?;
        writer
            .start_file(&name, options)
            .with_context(|| format!("failed to add {name} to the export archive"))?;
        writer
            .write_all(&bytes)
            .with_context(|| format!("failed to write {name} into the export archive"))?;
    }
    Ok(writer
        .finish()
        .context("failed to finalize the export archive")?
        .into_inner())
}

/// `directory` 아래의 모든 파일 경로를 깊이 우선으로 모은다.
fn collect_files(
    directory: &std::path::Path,
    out: &mut Vec<std::path::PathBuf>,
) -> anyhow::Result<()> {
    for entry in std::fs::read_dir(directory).context("failed to list the rendered export")? {
        let path = entry.context("failed to list the rendered export")?.path();
        if path.is_dir() {
            collect_files(&path, out)?;
        } else if path.is_file() {
            out.push(path);
        }
    }
    Ok(())
}

async fn get_thumbnail(
    State(app): State<AppState>,
    Path(page): Path<EntityId>,
) -> ApiResult<Vec<u8>> {
    ipc_bytes(output::get_thumbnail(page, app.state::<CurrentProject>()).await?)
}

#[derive(Deserialize)]
struct MovePageRequest {
    page: EntityId,
    index: u32,
}

async fn move_page(
    State(app): State<AppState>,
    Json(request): Json<MovePageRequest>,
) -> ApiResult<()> {
    editing::move_page(
        request.page,
        request.index,
        app.state::<koharu_desktop::Desktop>(),
        app.state::<CurrentProject>(),
        app.state::<CanvasChannel>(),
    )
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::archive_directory;

    #[test]
    fn a_nested_export_keeps_its_subfolders_in_the_archive() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join("png")).unwrap();
        std::fs::create_dir(root.path().join("psd")).unwrap();
        std::fs::write(root.path().join("png/0001_a.png"), b"png").unwrap();
        std::fs::write(root.path().join("psd/0001_a.psd"), b"psd").unwrap();

        let archive = archive_directory(root.path()).unwrap();
        let mut zip = zip::ZipArchive::new(std::io::Cursor::new(archive)).unwrap();
        let mut names: Vec<_> = (0..zip.len())
            .map(|index| zip.by_index(index).unwrap().name().to_owned())
            .collect();
        names.sort();
        assert_eq!(names, vec!["png/0001_a.png", "psd/0001_a.psd"]);
    }

    #[test]
    fn a_flat_export_is_unchanged() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("0002_b.png"), b"b").unwrap();
        std::fs::write(root.path().join("0001_a.png"), b"a").unwrap();

        let archive = archive_directory(root.path()).unwrap();
        let mut zip = zip::ZipArchive::new(std::io::Cursor::new(archive)).unwrap();
        assert_eq!(zip.len(), 2);
        assert_eq!(zip.by_index(0).unwrap().name(), "0001_a.png");
    }

    #[test]
    fn an_empty_export_is_still_an_error() {
        let root = tempfile::tempdir().unwrap();
        assert!(archive_directory(root.path()).is_err());
    }
}
