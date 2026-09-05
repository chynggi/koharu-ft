//! Font catalog.

use axum::extract::{Path, State};
use axum::routing::get;
use axum::{Json, Router};
use tauri::Manager as _;

use koharu_app::commands::fonts::{self, FontFamily};
use koharu_desktop::Desktop;

use crate::AppState;
use crate::error::{ApiResult, ipc_bytes};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/fonts", get(get_fonts))
        .route("/fonts/{family_name}/preview", get(get_font_preview))
}

async fn get_fonts(State(app): State<AppState>) -> ApiResult<Json<Vec<FontFamily>>> {
    Ok(Json(fonts::get_fonts(app.state::<Desktop>()).await?))
}

async fn get_font_preview(
    State(app): State<AppState>,
    Path(family_name): Path<String>,
) -> ApiResult<Vec<u8>> {
    ipc_bytes(fonts::get_font_preview(family_name, app.state::<Desktop>()).await?)
}
