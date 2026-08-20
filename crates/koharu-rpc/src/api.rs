use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
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
    if let Some(token) = api_token.clone() {
        api = api.layer(middleware::from_fn_with_state(
            Arc::<str>::from(token),
            require_token,
        ));
    }
    let mut router = Router::new().nest("/api/v1", api);

    if let Some(dir) = static_dir {
        if dir.is_dir() {
            let index = dir.join("index.html");
            let serve_dir = ServeDir::new(&dir).not_found_service(ServeFile::new(index.clone()));
            let token = Arc::<str>::from(api_token.unwrap_or_default());
            let index_route = move || {
                let index = index.clone();
                let token = token.clone();
                async move { index_with_token(token, index).await }
            };
            router = router.route("/", get(index_route)).fallback_service(serve_dir);
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

async fn index_with_token(token: Arc<str>, index: PathBuf) -> Response {
    let body = match tokio::fs::read_to_string(&index).await {
        Ok(body) => body,
        Err(error) => {
            tracing::error!(%error, path = %index.display(), "failed to read index.html");
            return StatusCode::INTERNAL_SERVER_ERROR.into_response();
        }
    };

    let injected = if token.is_empty() {
        body
    } else {
        body.replacen(
            "<head>",
            &format!(
                "<head><script>window.__KOHARU_API_TOKEN__ = {token:?};</script>",
                token = token,
            ),
            1,
        )
    };

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/html; charset=utf-8")
        .body(Body::from(injected))
        .unwrap_or_else(|error| {
            tracing::error!(%error, "failed to build index.html response");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        })
}

