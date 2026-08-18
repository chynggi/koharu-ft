//! Page listing, selection, ordering, and import. Thin delegation to the
//! desktop commands; import accepts explicit file paths instead of a dialog.

use std::path::PathBuf;

use anyhow::Context as _;
use axum::Json;
use axum::extract::{Path, State};
use axum::routing::{get, post};
use axum::Router;
use koharu_scene::{AssetInput, AssetMetadata, AssetRole, At, EntityId, PageDraft};
use serde::Deserialize;
use tauri::Manager as _;

use koharu_app::commands::canvas::CanvasChannel;
use koharu_app::commands::editing;
use koharu_app::commands::import;
use koharu_app::commands::lifecycle::{self, PageSelection};
use koharu_app::commands::output::{self, ExportFormat};
use koharu_app::commands::processing::Processing;
use koharu_app::commands::project::{CurrentProject, Page, PageSummary};

use crate::AppState;
use crate::error::{ApiResult, ipc_bytes};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/pages", get(get_pages))
        .route("/page", get(get_page))
        .route("/page/select", post(select_page))
        .route("/pages/import", post(import_pages))
        .route("/pages/export", post(export_pages))
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
    if !app.state::<Processing>().stops.lock().is_empty() {
        return Err(anyhow::anyhow!("pages cannot be imported while processing is running").into());
    }
    let files: Vec<PathBuf> = request
        .files
        .into_iter()
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .collect();
    if files.is_empty() {
        return Err(anyhow::anyhow!("none of the provided files exist").into());
    }
    let pages = tokio::task::spawn_blocking(move || import::import(files))
        .await
        .context("page import worker stopped unexpectedly")??;

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
    format: ExportFormat,
    directory: String,
}

async fn export_pages(
    State(app): State<AppState>,
    Json(request): Json<ExportPagesRequest>,
) -> ApiResult<()> {
    let directory = PathBuf::from(request.directory);
    if !directory.is_dir() {
        return Err(anyhow::anyhow!("the export directory does not exist").into());
    }
    output::export_pages_to(
        directory,
        request.pages,
        request.format,
        app.state::<CurrentProject>(),
        app.state::<koharu_desktop::Desktop>(),
    )
    .await?;
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
