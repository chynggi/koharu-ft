//! Types shared by the agent loop and its backends. The loop owns a
//! provider-neutral transcript; each backend converts it to its own wire format.

use serde::Serialize;
use serde_json::Value;
use specta::Type;

use crate::{Reasoning, Tool, ToolCall, ToolImage};

pub(crate) const MAX_STREAM_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Type)]
pub struct AgentModel {
    pub id: String,
    pub name: String,
    pub reasoning: Vec<Reasoning>,
}

#[derive(Clone, Debug)]
pub(crate) enum Item {
    User(String),
    /// `native` holds the backend's own output items, such as Codex reasoning
    /// with encrypted content. History is discarded when the backend changes,
    /// so only the backend that produced them replays them.
    Assistant {
        text: String,
        calls: Vec<ToolCall>,
        native: Vec<Value>,
    },
    ToolResult {
        call_id: String,
        output: Value,
        images: Vec<ToolImage>,
    },
}

pub(crate) struct Prompt<'a> {
    pub(crate) model: &'a str,
    pub(crate) instructions: &'a str,
    pub(crate) items: &'a [Item],
    pub(crate) tools: &'a [Tool],
    pub(crate) reasoning: Reasoning,
    pub(crate) session: &'a str,
}

pub(crate) enum Delta {
    Text(String),
    Reasoning(String),
}

pub(crate) struct Turn {
    pub(crate) text: String,
    pub(crate) calls: Vec<ToolCall>,
    pub(crate) native: Vec<Value>,
}
