//! Pipeline processing jobs.

use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use tauri::Manager as _;

use koharu_app::commands::processing::{Job, JobId, JobChannel, Processing};
use koharu_app::commands::project::CurrentProject;
use koharu_app::commands::processing;
use koharu_pipeline::{Operation, Scope};

use crate::AppState;
use crate::error::ApiResult;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/process", post(process))
        .route("/process/stop", post(stop_job))
        .route("/jobs", get(list_jobs))
}

#[derive(Deserialize)]
struct ProcessRequest {
    scope: Scope,
    operation: Operation,
}

async fn process(
    State(app): State<AppState>,
    Json(request): Json<ProcessRequest>,
) -> ApiResult<Json<JobId>> {
    Ok(Json(
        processing::process(
            app.clone(),
            request.scope,
            request.operation,
            app.state::<CurrentProject>(),
            app.state::<Processing>(),
            app.state::<JobChannel>(),
        )
        .await?,
    ))
}

#[derive(Deserialize)]
struct StopJobRequest {
    job: JobId,
}

async fn stop_job(
    State(app): State<AppState>,
    Json(request): Json<StopJobRequest>,
) -> ApiResult<()> {
    processing::stop_job(request.job, app.state::<Processing>()).await?;
    Ok(())
}

async fn list_jobs(State(app): State<AppState>) -> ApiResult<Json<Vec<Job>>> {
    let jobs = app
        .state::<Processing>()
        .jobs
        .lock()
        .values()
        .cloned()
        .collect();
    Ok(Json(jobs))
}
