//! Runs on a processing server: the pages a run works on go to the server
//! as snapshots, it runs the request there with its own models, and its
//! events and edits come back. Each edit replays on the open project, so a
//! remote run writes what a local one would, one undo step per stage and
//! page, under the same ids. Its events carry the local run's id.
//!
//! The wire is HTTP: a request's body and each answer frame are JSON
//! compressed with zstd, each frame after its length as four little-endian
//! bytes.

use std::sync::Arc;

use anyhow::{Context as _, bail};
use futures::StreamExt as _;
use koharu_scene::{Applied, PageId, Session};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tokio_util::sync::CancellationToken;

use crate::{Event, Outcome, Request, RunId, Stage, event::Events};

/// The secret holding the token a server asks for.
pub const TOKEN: &str = "koharu-server";

/// Where a server runs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, specta::Type)]
pub struct RemoteConfig {
    /// Its address, as `http://host:port`.
    pub url: String,
}

/// What a run sends: its pages as snapshots, in project order, and the
/// request, its settings included.
#[derive(Debug, Serialize, Deserialize)]
pub struct RemoteRun {
    pub pages: Vec<Applied>,
    pub request: Request,
}

/// What a server answers with, in order.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Message {
    Event { event: Event },
    Applied { applied: Applied },
}

/// `value` as a request body or frame contents.
pub fn encode<T: Serialize>(value: &T) -> anyhow::Result<Vec<u8>> {
    Ok(zstd::encode_all(serde_json::to_vec(value)?.as_slice(), 3)?)
}

pub fn decode<T: DeserializeOwned>(bytes: &[u8]) -> anyhow::Result<T> {
    Ok(serde_json::from_slice(&zstd::decode_all(bytes)?)?)
}

/// `message` as a frame of an answer.
pub fn frame(message: &Message) -> anyhow::Result<Vec<u8>> {
    let body = encode(message)?;
    let mut frame = u32::try_from(body.len())?.to_le_bytes().to_vec();
    frame.extend(body);
    Ok(frame)
}

/// Runs `request` on `pages` of `session` at `remote` as the local run `id`,
/// telling `events`, until it ends or `cancel` stops it. Every end, a failure
/// to reach the server too, is told as the run's `Ended`.
pub(crate) async fn run(
    remote: RemoteConfig,
    session: Arc<Mutex<Session>>,
    pages: Vec<PageId>,
    request: Request,
    id: RunId,
    events: Events,
    cancel: CancellationToken,
) {
    // Where a failure is told as being, which `Outcome::Failed` asks for.
    let mut at = (
        pages.first().copied(),
        request.stages.first().copied().unwrap_or(Stage::Detect),
    );
    let ended = tokio::select! {
        ended = exchange(&remote, &session, &pages, request, id, &events, &mut at) => ended,
        () = cancel.cancelled() => Ok(Outcome::Stopped),
    };
    let outcome = ended.unwrap_or_else(|error| match at {
        (Some(page), stage) => Outcome::Failed {
            page,
            stage,
            reason: format!("{error:#}"),
        },
        // A run with no page fails when it's asked for, not here.
        (None, _) => Outcome::Stopped,
    });
    events.send(Event::Ended { run: id, outcome });
}

async fn exchange(
    remote: &RemoteConfig,
    session: &Arc<Mutex<Session>>,
    pages: &[PageId],
    request: Request,
    id: RunId,
    events: &Events,
    at: &mut (Option<PageId>, Stage),
) -> anyhow::Result<Outcome> {
    let (token, body) = {
        let session = Arc::clone(session);
        let pages = pages.to_vec();
        tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
            let token = koharu_config::secrets::get(TOKEN)?;
            let session = session.lock();
            let pages = pages
                .iter()
                .enumerate()
                .map(|(index, page)| session.snapshot(*page, index))
                .collect::<koharu_scene::Result<Vec<_>>>()?;
            Ok((token, encode(&RemoteRun { pages, request })?))
        })
        .await??
    };
    let url = format!("{}/v1/runs", remote.url.trim_end_matches('/'));
    let mut asked = reqwest::Client::new().post(&url).body(body);
    if let Some(token) = token {
        use koharu_config::secrets::ExposeSecret as _;
        asked = asked.bearer_auth(token.expose_secret());
    }
    let answer = asked
        .send()
        .await
        .with_context(|| format!("couldn't reach the processing server at {}", remote.url))?;
    if !answer.status().is_success() {
        let status = answer.status();
        let text = answer.text().await.unwrap_or_default();
        bail!("the processing server answered {status}: {text}");
    }
    let mut stream = answer.bytes_stream();
    let mut buffer = Vec::new();
    loop {
        // Every whole frame the buffer holds, in order.
        while let Some(length) = buffer
            .first_chunk::<4>()
            .map(|length| u32::from_le_bytes(*length) as usize)
            .filter(|length| buffer.len() >= 4 + length)
        {
            let message: Message = decode(&buffer[4..4 + length])?;
            buffer.drain(..4 + length);
            match message {
                Message::Applied { applied } => {
                    let session = Arc::clone(session);
                    tokio::task::spawn_blocking(move || session.lock().replay(applied))
                        .await?
                        .context("the project changed under the run")?;
                }
                // Told when it was asked for, with the steps it takes here.
                Message::Event {
                    event: Event::Queued { .. },
                } => {}
                Message::Event {
                    event: Event::Ended { outcome, .. },
                } => return Ok(outcome),
                Message::Event {
                    event: Event::Started { page, stage, .. },
                } => {
                    *at = (Some(page), stage);
                    events.send(Event::Started {
                        run: id,
                        page,
                        stage,
                    });
                }
                Message::Event {
                    event: Event::Finished { page, stage, .. },
                } => events.send(Event::Finished {
                    run: id,
                    page,
                    stage,
                }),
                Message::Event {
                    event:
                        Event::Failed {
                            page,
                            stage,
                            reason,
                            ..
                        },
                } => events.send(Event::Failed {
                    run: id,
                    page,
                    stage,
                    reason,
                }),
            }
        }
        match stream.next().await {
            Some(chunk) => buffer.extend_from_slice(&chunk?),
            None => bail!("the processing server closed the run before it ended"),
        }
    }
}
