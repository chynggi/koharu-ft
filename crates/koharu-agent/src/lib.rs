//! Koharu's in-process agent, driven by Codex or a Chat Completions provider.

mod agent;
mod backend;
mod chat;
mod codex;
mod config;
mod control;
mod tool;

pub use agent::{Agent, Event, Message, Role, RunId, RunResult};
pub use backend::AgentModel;
pub use codex::{Account, Codex, LoginEvent};
pub use config::{Backend, Config, Reasoning};
pub use control::Control;
pub use tool::{Host, Invocation, Tool, ToolCall, ToolImage};
