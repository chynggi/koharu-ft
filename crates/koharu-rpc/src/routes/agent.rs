//! Codex agent integration: plain JSON routes for status/logout/config/cancel,
//! plus SSE routes that bridge the two streaming commands (`login_agent`,
//! `run_agent`) which normally push events through a Tauri IPC `Channel`.

use std::convert::Infallible;

use axum::extract::{Path, State};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use futures::Stream;
use koharu_agent::{Config, RunId};
use serde::Deserialize;
use tauri::Manager as _;
use tauri::ipc::{Channel, InvokeResponseBody};
use tokio_stream::wrappers::UnboundedReceiverStream;

use koharu_app::commands::agent::{self, AgentState, AgentStatus};

use crate::AppState;
use crate::error::ApiResult;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/agent/status", get(get_status))
        .route("/agent/logout", post(logout))
        .route("/agent/config", put(save_config))
        .route("/agent/run/{run_id}/cancel", post(cancel))
        .route("/agent/login", post(login))
        .route("/agent/run", post(run))
}

async fn get_status(State(app): State<AppState>) -> ApiResult<Json<AgentStatus>> {
    Ok(Json(agent::get_agent_status(app.state::<AgentState>()).await?))
}

async fn logout(State(app): State<AppState>) -> ApiResult<Json<AgentStatus>> {
    Ok(Json(agent::logout_agent(app.state::<AgentState>()).await?))
}

async fn save_config(
    State(app): State<AppState>,
    Json(config): Json<Config>,
) -> ApiResult<Json<Config>> {
    Ok(Json(
        agent::save_agent_config(config, app.state::<AgentState>()).await?,
    ))
}

async fn cancel(State(app): State<AppState>, Path(run): Path<RunId>) -> ApiResult<()> {
    agent::cancel_agent(run, app.state::<AgentState>()).await?;
    Ok(())
}

/// `tauri::ipc::Channel::new` only needs a plain message callback, not a live
/// webview, so we bridge it straight into an mpsc sender and stream the
/// forwarded JSON bodies back out as SSE — no Tauri IPC context required.
fn bridge_channel<T>() -> (
    Channel<T>,
    UnboundedReceiverStream<Result<SseEvent, Infallible>>,
)
where
    T: tauri::ipc::IpcResponse,
{
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let channel = Channel::new(move |body| {
        let _ = tx.send(Ok(to_sse_event(body)));
        Ok(())
    });
    (channel, UnboundedReceiverStream::new(rx))
}

fn to_sse_event(body: InvokeResponseBody) -> SseEvent {
    let data = match body {
        InvokeResponseBody::Json(json) => json,
        InvokeResponseBody::Raw(bytes) => serde_json::to_string(&bytes).unwrap_or_default(),
    };
    SseEvent::default().data(data)
}

async fn login(
    State(app): State<AppState>,
) -> Sse<impl Stream<Item = Result<SseEvent, Infallible>>> {
    let (channel, stream) = bridge_channel();
    tokio::spawn(async move {
        if let Err(error) = agent::login_agent(app.state::<AgentState>(), channel).await {
            tracing::error!(%error, "agent login failed");
        }
    });
    Sse::new(stream).keep_alive(KeepAlive::default())
}

#[derive(Deserialize)]
struct RunAgentRequest {
    prompt: String,
}

async fn run(
    State(app): State<AppState>,
    Json(request): Json<RunAgentRequest>,
) -> ApiResult<Sse<impl Stream<Item = Result<SseEvent, Infallible>>>> {
    let (channel, stream) = bridge_channel();
    agent::run_agent(
        request.prompt,
        channel,
        app.clone(),
        app.state::<AgentState>(),
    )
    .await?;
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}
