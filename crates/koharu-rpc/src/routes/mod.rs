use axum::Router;

use crate::AppState;

pub mod agent;
pub mod canvas;
pub mod config;
pub mod events;
pub mod fonts;
pub mod layers;
pub mod llm;
pub mod meta;
pub mod operations;
pub mod pages;
pub mod projects;

pub fn router() -> Router<AppState> {
    Router::new()
        .merge(meta::router())
        .merge(config::router())
        .merge(llm::router())
        .merge(projects::router())
        .merge(pages::router())
        .merge(operations::router())
        .merge(fonts::router())
        .merge(canvas::router())
        .merge(layers::router())
        .merge(agent::router())
        .merge(events::router())
}
