//! Preferences (pipeline, providers, typesetting) via the desktop commands.

use axum::extract::State;
use axum::routing::get;
use axum::{Json, Router};

use koharu_app::commands::preferences::{self, Preferences, ProviderPreferences};
use koharu_pipeline::PipelineConfig;
use koharu_renderer::TypesettingConfig;
use koharu_translator::Model;

use crate::AppState;
use crate::error::ApiResult;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/config", get(get_config).put(put_config))
        .route("/translation-models", get(get_translation_models))
}

async fn get_config(_state: State<AppState>) -> ApiResult<Json<Preferences>> {
    Ok(Json(preferences::get_preferences().await?))
}

#[derive(serde::Deserialize)]
struct SaveConfigRequest {
    pipeline: PipelineConfig,
    providers: ProviderPreferences,
    typesetting: TypesettingConfig,
}

async fn put_config(
    _state: State<AppState>,
    Json(request): Json<SaveConfigRequest>,
) -> ApiResult<Json<Preferences>> {
    Ok(Json(
        preferences::save_preferences(
            request.pipeline,
            request.providers,
            request.typesetting,
        )
        .await?,
    ))
}

async fn get_translation_models(_state: State<AppState>) -> ApiResult<Json<Vec<Model>>> {
    Ok(Json(preferences::get_translation_models().await?))
}
