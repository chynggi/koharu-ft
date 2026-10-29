use std::{fmt, sync::Arc};

use anyhow::{Context as _, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use specta::Type;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::{
    AgentModel, Backend, Codex, Config, Control, Host, ToolCall,
    backend::{Delta, Item, Prompt},
    chat::{self, Chat},
};

const INSTRUCTIONS: &str = r#"You are Koharu Agent, operating a manga translation project inside Koharu.
A koharu_project_context block on every user turn summarizes every page and details the active page. Call inspect_page before working on another page.
Page images are intentionally omitted from that context. Call view_page only when visual inspection is needed, and only for relevant pages.
Use the provided tools whenever the user asks to inspect or change the project. Never claim a change succeeded unless its tool result says it succeeded.
All project changes are revisioned and reversible. Do not ask for permission. Do not produce a plan or expose internal steps; continue using tools until the request is complete or cannot be completed.
Do not invent entity identifiers. Preserve artwork and existing authored content unless the user asks to change them."#;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize, Type)]
#[serde(transparent)]
pub struct RunId(Uuid);

impl RunId {
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for RunId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for RunId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Clone, Debug, Deserialize, Serialize, Type)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Clone, Debug, Serialize, Type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    Started {
        run: RunId,
    },
    TextDelta {
        run: RunId,
        delta: String,
    },
    ReasoningDelta {
        run: RunId,
        delta: String,
    },
    ToolStarted {
        run: RunId,
        call_id: String,
        name: String,
    },
    ToolFinished {
        run: RunId,
        call_id: String,
        name: String,
        changed: bool,
        output: String,
    },
    Completed {
        run: RunId,
        message: String,
    },
    Failed {
        run: RunId,
        message: String,
    },
    Cancelled {
        run: RunId,
    },
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct RunResult {
    pub run: RunId,
    pub message: String,
}

pub struct Agent<H> {
    codex: Codex,
    client: Client,
    host: Arc<H>,
    config: koharu_config::Config<Config>,
    history: Mutex<History>,
    serial: Mutex<()>,
}

/// The conversation belongs to the backend that produced it. A run on another
/// backend starts fresh instead of replaying foreign items.
#[derive(Default)]
struct History {
    backend: Backend,
    items: Vec<Item>,
}

impl<H> Agent<H>
where
    H: Host,
{
    pub fn new(codex: Codex, host: H) -> Result<Self> {
        Ok(Self {
            codex,
            client: koharu_runtime::http_client()?,
            host: Arc::new(host),
            config: Config::load()?,
            history: Mutex::new(History::default()),
            serial: Mutex::new(()),
        })
    }

    pub fn codex(&self) -> &Codex {
        &self.codex
    }

    pub fn config(&self) -> Result<Config> {
        Ok(self.config.read()?.clone())
    }

    pub async fn models(&self) -> Result<Vec<AgentModel>> {
        match self.config()?.backend {
            Backend::Codex => self.codex.models().await,
            Backend::Provider(provider) => chat::models(provider).await,
        }
    }

    pub fn save_config(&self, config: Config) -> Result<Config> {
        let mut current = self.config.write()?;
        *current = config;
        let saved = current.clone();
        current.save()?;
        Ok(saved)
    }

    pub async fn clear(&self) {
        self.history.lock().await.items.clear();
    }

    #[tracing::instrument(skip_all)]
    pub async fn run<F>(
        &self,
        run: RunId,
        prompt: String,
        control: Control,
        mut publish: F,
    ) -> Result<RunResult>
    where
        F: FnMut(Event) + Send,
    {
        let _serial = self.serial.lock().await;
        publish(Event::Started { run });
        let result = self.run_inner(run, prompt, &control, &mut publish).await;
        match &result {
            Ok(result) => publish(Event::Completed {
                run,
                message: result.message.clone(),
            }),
            Err(_) if control.is_cancelled() => publish(Event::Cancelled { run }),
            Err(error) => publish(Event::Failed {
                run,
                message: format!("{error:#}"),
            }),
        }
        result
    }

    async fn run_inner<F>(
        &self,
        run: RunId,
        prompt: String,
        control: &Control,
        publish: &mut F,
    ) -> Result<RunResult>
    where
        F: FnMut(Event),
    {
        control.ensure_running()?;
        let context = self.host.context().await?;
        let config = self.config()?;
        let (chat, model, reasoning) = match config.backend {
            Backend::Codex => {
                let models = self.codex.models().await?;
                let model = select_model(&models, config.model.as_deref())?;
                let reasoning =
                    if model.reasoning.is_empty() || model.reasoning.contains(&config.reasoning) {
                        config.reasoning
                    } else {
                        model.reasoning[0]
                    };
                (None, model.id.clone(), reasoning)
            }
            Backend::Provider(provider) => {
                let chat = Chat::connect(self.client.clone(), provider)?;
                let model = match config.model {
                    Some(model) => model,
                    None => select_model(&chat::models(provider).await?, None)?
                        .id
                        .clone(),
                };
                (Some(chat), model, config.reasoning)
            }
        };
        let clean_user = Item::User(prompt);
        let context_message = Item::User(format!(
            "<koharu_project_context>\n{}\n</koharu_project_context>",
            serde_json::to_string(&context)?
        ));
        let base = {
            let history = self.history.lock().await;
            if history.backend == config.backend {
                history.items.clone()
            } else {
                Vec::new()
            }
        };
        let mut input = base.clone();
        input.push(context_message);
        input.push(clean_user.clone());
        let mut persisted = base;
        persisted.push(clean_user);
        let tools = self.host.tools();
        let session = run.to_string();

        loop {
            control.ensure_running()?;
            let prompt = Prompt {
                model: &model,
                instructions: INSTRUCTIONS,
                items: &input,
                tools: &tools,
                reasoning,
                session: &session,
            };
            let mut delta = |event| match event {
                Delta::Text(delta) => publish(Event::TextDelta { run, delta }),
                Delta::Reasoning(delta) => publish(Event::ReasoningDelta { run, delta }),
            };
            let turn = match &chat {
                None => self.codex.respond(&prompt, control, &mut delta).await?,
                Some(chat) => chat.respond(&prompt, control, &mut delta).await?,
            };
            let assistant = Item::Assistant {
                text: turn.text.clone(),
                calls: turn.calls.clone(),
                native: turn.native,
            };
            input.push(assistant.clone());
            persisted.push(assistant);

            if turn.calls.is_empty() {
                *self.history.lock().await = History {
                    backend: config.backend,
                    items: persisted,
                };
                return Ok(RunResult {
                    run,
                    message: turn.text,
                });
            }

            for call in turn.calls {
                control.ensure_running()?;
                publish(Event::ToolStarted {
                    run,
                    call_id: call.call_id.clone(),
                    name: call.name.clone(),
                });
                let invocation = self
                    .host
                    .invoke(
                        ToolCall {
                            call_id: call.call_id.clone(),
                            name: call.name.clone(),
                            arguments: call.arguments,
                        },
                        control,
                    )
                    .await;
                let (output, changed, images) = match invocation {
                    Ok(invocation) => (
                        json!({ "ok": true, "value": invocation.value }),
                        invocation.changed,
                        invocation.images,
                    ),
                    Err(error) => (
                        json!({ "ok": false, "error": format!("{error:#}") }),
                        false,
                        Vec::new(),
                    ),
                };
                publish(Event::ToolFinished {
                    run,
                    call_id: call.call_id.clone(),
                    name: call.name,
                    changed,
                    output: output.to_string(),
                });
                persisted.push(Item::ToolResult {
                    call_id: call.call_id.clone(),
                    output: output.clone(),
                    images: Vec::new(),
                });
                input.push(Item::ToolResult {
                    call_id: call.call_id,
                    output,
                    images,
                });
            }
        }
    }
}

fn select_model<'a>(models: &'a [AgentModel], selected: Option<&str>) -> Result<&'a AgentModel> {
    match selected {
        Some(selected) => models
            .iter()
            .find(|model| model.id == selected)
            .with_context(|| format!("configured agent model {selected} is not available")),
        None => models
            .first()
            .context("the agent backend returned no available model"),
    }
}
