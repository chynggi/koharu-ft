use std::path::PathBuf;

use axum::Router;
use tower_http::cors::CorsLayer;
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;

use crate::AppState;
use crate::routes;

pub fn router(app: AppState, static_dir: Option<PathBuf>) -> Router {
    let mut router = Router::new().nest("/api/v1", routes::router());

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

