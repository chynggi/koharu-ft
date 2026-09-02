//! Project library lifecycle. Thin delegation to the desktop commands.

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use tauri::Manager as _;

use koharu_app::commands::lifecycle;
use koharu_app::commands::project::{CurrentProject, ProjectInfo, ProjectLibrary, ProjectSummary};

use crate::AppState;
use crate::error::ApiResult;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/projects", get(list_projects).post(create_project))
        .route("/projects/{name}/open", post(open_project))
        .route("/projects/{name}", axum::routing::delete(delete_project))
        .route("/project", get(get_project))
        .route("/project/close", post(close_project))
}

async fn list_projects(State(app): State<AppState>) -> ApiResult<Json<Vec<ProjectSummary>>> {
    Ok(Json(
        lifecycle::list_projects(app.state::<ProjectLibrary>()).await?,
    ))
}

#[derive(Deserialize)]
struct CreateProjectRequest {
    name: String,
}

async fn create_project(
    State(app): State<AppState>,
    Json(request): Json<CreateProjectRequest>,
) -> ApiResult<StatusCode> {
    lifecycle::create_project(request.name, app.clone()).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn open_project(
    State(app): State<AppState>,
    Path(name): Path<String>,
) -> ApiResult<StatusCode> {
    lifecycle::open_project(name, app.clone()).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn delete_project(
    State(app): State<AppState>,
    Path(name): Path<String>,
) -> ApiResult<StatusCode> {
    lifecycle::delete_project(name, app.clone()).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn get_project(State(app): State<AppState>) -> ApiResult<Json<Option<ProjectInfo>>> {
    Ok(Json(
        lifecycle::get_project(app.state::<CurrentProject>()).await?,
    ))
}

async fn close_project(State(app): State<AppState>) -> ApiResult<StatusCode> {
    lifecycle::close_project(app.clone()).await?;
    Ok(StatusCode::NO_CONTENT)
}
