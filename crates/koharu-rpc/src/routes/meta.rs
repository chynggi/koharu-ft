//! `GET /meta` — server metadata.

use axum::extract::State;
use axum::routing::get;
use axum::{Json, Router};
use serde_json::{Value, json};

use crate::AppState;
use crate::error::ApiResult;

pub fn router() -> Router<AppState> {
    Router::new().route("/meta", get(get_meta))
}

async fn get_meta(_state: State<AppState>) -> ApiResult<Json<Value>> {
    Ok(Json(json!({
        "name": "koharu",
        "version": env!("CARGO_PKG_VERSION"),
    })))
}
