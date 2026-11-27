//! Koharu's processing server: the desktop app sends it a run's pages and
//! request, it runs them with its own models, and it answers with the run's
//! events and edits, which the app replays on its project. One run at a time,
//! as the models share one GPU; a run whose app hangs up stops.
//!
//! The wire is `koharu_pipeline::remote`'s.

use std::sync::Arc;

use axum::{
    Router,
    body::{Body, Bytes},
    extract::{DefaultBodyLimit, State},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use koharu_pipeline::{
    Event, Pipeline,
    remote::{self, Message, RemoteRun},
};
use koharu_scene::Session;
use parking_lot::Mutex;
use tokio::sync::mpsc;

/// The server's routes, running on `pipeline`; `token`, when there's one,
/// is what every request must bring as its bearer token.
pub fn router(pipeline: Pipeline, token: Option<String>) -> Router {
    let server = Arc::new(Server {
        pipeline,
        token,
        busy: Arc::default(),
    });
    Router::new()
        .route("/v1/health", get(|| async { "ok" }))
        .route("/v1/runs", post(run))
        // A run's pages are as large as their scans.
        .layer(DefaultBodyLimit::disable())
        .with_state(server)
}

struct Server {
    pipeline: Pipeline,
    token: Option<String>,
    /// Held while a run goes on.
    busy: Arc<tokio::sync::Mutex<()>>,
}

async fn run(State(server): State<Arc<Server>>, headers: HeaderMap, body: Bytes) -> Response {
    if let Some(token) = &server.token {
        let bearer = headers
            .get(header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("Bearer "));
        if bearer != Some(token.as_str()) {
            return (StatusCode::UNAUTHORIZED, "a token is needed").into_response();
        }
    }
    let Ok(held) = Arc::clone(&server.busy).try_lock_owned() else {
        return (StatusCode::CONFLICT, "another run is going on").into_response();
    };
    let started = tokio::task::spawn_blocking(move || prepare(&body)).await;
    let (folder, session, request) = match started {
        Ok(Ok(started)) => started,
        Ok(Err(error)) => {
            return (StatusCode::BAD_REQUEST, format!("{error:#}")).into_response();
        }
        Err(error) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()).into_response();
        }
    };
    let session = Arc::new(Mutex::new(session));
    let mut events = server.pipeline.events();
    let id = match server.pipeline.run(Arc::clone(&session), request) {
        Ok(id) => id,
        Err(error) => return (StatusCode::BAD_REQUEST, error.to_string()).into_response(),
    };
    let (frames, answer) = mpsc::channel::<Result<Vec<u8>, std::io::Error>>(16);
    tokio::spawn(async move {
        // Kept until the run ends: the lock, and the project's folder.
        let _held = held;
        let _folder = folder;
        let project = session.lock().id();
        loop {
            let event = tokio::select! {
                event = events.recv() => event,
                () = frames.closed() => {
                    tracing::info!("the app hung up; stopping the run");
                    server.pipeline.stop();
                    break;
                }
            };
            let event = match event {
                Ok(event) => event,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            };
            let (Event::Queued { run, .. }
            | Event::Started { run, .. }
            | Event::Finished { run, .. }
            | Event::Failed { run, .. }
            | Event::Ended { run, .. }) = &event;
            if *run != id {
                continue;
            }
            // The edits made so far go before what's told of them.
            let ended = matches!(event, Event::Ended { .. });
            let mut messages: Vec<Message> = session
                .lock()
                .recorded()
                .into_iter()
                .map(|applied| Message::Applied { applied })
                .collect();
            messages.push(Message::Event { event });
            for message in messages {
                let frame = remote::frame(&message).map_err(std::io::Error::other);
                if frames.send(frame).await.is_err() {
                    server.pipeline.stop();
                    break;
                }
            }
            if ended {
                break;
            }
        }
        server.pipeline.close(project);
    });
    Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(answer)).into_response()
}

/// A run's pages in a new project in its own folder, recording from then on,
/// and its request to run there.
fn prepare(body: &[u8]) -> anyhow::Result<(tempfile::TempDir, Session, koharu_pipeline::Request)> {
    let RemoteRun {
        meta,
        pages,
        mut request,
    } = remote::decode(body)?;
    let folder = tempfile::tempdir()?;
    let mut session = Session::create(folder.path().join("run.khr"), meta)?;
    for page in pages {
        session.replay(page)?;
    }
    session.record();
    // A server runs what it's sent itself.
    if let Some(config) = &mut request.config {
        config.remote = None;
    }
    Ok((folder, session, request))
}
