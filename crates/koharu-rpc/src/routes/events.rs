//! `GET /events` — a single SSE stream multiplexing the five startup-subscription
//! channels (canvas, job, download, resource, project) as `{"type":..,"data":..}`.

use axum::Router;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::routing::get;
use futures::Stream;
use serde::Serialize;
use serde_json::json;
use tauri::Manager as _;
use tokio_stream::StreamExt as _;
use tokio_stream::wrappers::UnboundedReceiverStream;

use koharu_app::commands::canvas::CanvasChannel;
use koharu_app::commands::lifecycle::{Download, DownloadState, ModelResources, ProjectChannel};
use koharu_app::commands::processing::JobChannel;

use crate::AppState;

pub fn router() -> Router<AppState> {
    Router::new().route("/events", get(events))
}

async fn events(
    State(app): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();

    let mut canvas_rx = app.state::<CanvasChannel>().broadcast.subscribe();
    let mut job_rx = app.state::<JobChannel>().broadcast.subscribe();
    let mut project_rx = app.state::<ProjectChannel>().broadcast.subscribe();
    let mut download_rx = koharu_runtime::downloads::subscribe();
    let mut resource_rx = app.state::<koharu_pipeline::Pipeline>().subscribe_resources();

    tokio::spawn(async move {
        loop {
            let sent = tokio::select! {
                result = canvas_rx.recv() => forward(&tx, "canvas", result),
                result = job_rx.recv() => forward(&tx, "job", result),
                result = project_rx.recv() => forward(&tx, "project", result),
                result = download_rx.recv() => match result {
                    Ok(event) => send(&tx, "download", &map_download(event)),
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => true,
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => false,
                },
                changed = resource_rx.changed() => match changed {
                    Ok(()) => {
                        let snapshot = resource_rx.borrow_and_update().clone();
                        send(&tx, "resource", &ModelResources::from(snapshot))
                    }
                    Err(_) => false,
                },
            };
            if !sent {
                break;
            }
        }
    });

    Sse::new(UnboundedReceiverStream::new(rx).map(Ok)).keep_alive(KeepAlive::default())
}

/// Forwards a broadcast receive result, skipping `Lagged` and stopping on `Closed`.
fn forward<T: Serialize>(
    tx: &tokio::sync::mpsc::UnboundedSender<Event>,
    kind: &str,
    result: Result<T, tokio::sync::broadcast::error::RecvError>,
) -> bool {
    match result {
        Ok(value) => send(tx, kind, &value),
        Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => true,
        Err(tokio::sync::broadcast::error::RecvError::Closed) => false,
    }
}

fn send<T: Serialize>(tx: &tokio::sync::mpsc::UnboundedSender<Event>, kind: &str, data: &T) -> bool {
    let Ok(event) = Event::default().json_data(json!({ "type": kind, "data": data })) else {
        return true;
    };
    tx.send(event).is_ok()
}

fn map_download(event: koharu_runtime::downloads::Event) -> Download {
    match event {
        koharu_runtime::downloads::Event::Started { id, name } => Download {
            id,
            state: DownloadState::Running,
            name: Some(name),
            completed: 0,
            total: 0,
            error: None,
        },
        koharu_runtime::downloads::Event::Progress {
            id,
            name,
            completed,
            total,
        } => Download {
            id,
            state: DownloadState::Running,
            name: Some(name),
            completed,
            total,
            error: None,
        },
        koharu_runtime::downloads::Event::Finished { id } => Download {
            id,
            state: DownloadState::Finished,
            name: None,
            completed: 0,
            total: 0,
            error: None,
        },
        koharu_runtime::downloads::Event::Failed { id, name, error } => Download {
            id,
            state: DownloadState::Failed,
            name: Some(name),
            completed: 0,
            total: 0,
            error: Some(error),
        },
    }
}
