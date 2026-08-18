use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::Router;
use tower_http::cors::CorsLayer;
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;

use crate::AppState;
use crate::routes;

async fn require_token(State(token): State<Arc<str>>, req: Request, next: Next) -> Response {
    let bearer = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "));
    let query_token = req.uri().query().and_then(|query| {
        query.split('&').find_map(|pair| {
            let (key, value) = pair.split_once('=')?;
            (key == "token").then_some(value)
        })
    });

    if bearer == Some(&*token) || query_token == Some(&*token) {
        next.run(req).await
    } else {
        (StatusCode::UNAUTHORIZED, "missing or invalid API token").into_response()
    }
}

pub fn router(app: AppState, static_dir: Option<PathBuf>, api_token: Option<String>) -> Router {
    let mut api = routes::router();
    if let Some(token) = api_token {
        api = api.layer(middleware::from_fn_with_state(
            Arc::<str>::from(token),
            require_token,
        ));
    }
    let mut router = Router::new().nest("/api/v1", api);

    if let Some(dir) = static_dir {
        if dir.is_dir() {
            let index = dir.join("index.html");
            let serve_dir = ServeDir::new(&dir).not_found_service(ServeFile::new(index));
            router = router.fallback_service(serve_dir);
        } else {
            tracing::warn!(
                path = %dir.display(),
                "static frontend directory does not exist; serving the API only"
            );
        }
    }

    router
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(app)
}

