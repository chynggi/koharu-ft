use std::collections::BTreeMap;

use anyhow::{Context as _, Result, anyhow, bail};
use eventsource_stream::Eventsource as _;
use futures::StreamExt as _;
use reqwest::header::CONTENT_TYPE;
use serde_json::Value;
use uuid::Uuid;

use crate::{
    Control, ToolCall,
    backend::{Delta, MAX_STREAM_BYTES, Turn},
};

pub(super) async fn read<F>(
    provider: &str,
    response: reqwest::Response,
    control: &Control,
    mut publish: F,
) -> Result<Turn>
where
    F: FnMut(Delta),
{
    let streamed = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.contains("text/event-stream"));
    if !streamed {
        return read_completion(provider, response, control, publish).await;
    }

    let mut stream = response.bytes_stream().eventsource();
    let mut bytes = 0_usize;
    let mut turn = Accumulator::default();
    let mut completed = false;
    loop {
        let event = tokio::select! {
            event = stream.next() => event,
            () = control.cancelled() => {
                control.ensure_running()?;
                unreachable!("cancelled control must fail ensure_running")
            }
        };
        let Some(event) = event else {
            break;
        };
        let event = event.map_err(|error| anyhow!("invalid {provider} event stream: {error}"))?;
        bytes = bytes.saturating_add(event.data.len());
        if bytes > MAX_STREAM_BYTES {
            bail!("{provider} response stream exceeded {MAX_STREAM_BYTES} bytes");
        }
        if event.data.trim().is_empty() {
            continue;
        }
        if event.data == "[DONE]" {
            completed = true;
            break;
        }
        let value: Value = serde_json::from_str(&event.data)
            .with_context(|| format!("invalid JSON in {provider} response stream"))?;
        fail_on_error(provider, &value)?;
        let Some(choice) = value.pointer("/choices/0") else {
            continue;
        };
        if let Some(delta) = choice.get("delta") {
            turn.apply(delta, &mut publish);
        }
        if choice
            .get("finish_reason")
            .is_some_and(|reason| !reason.is_null())
        {
            completed = true;
        }
    }
    if !completed {
        bail!("{provider} response stream ended before completion");
    }
    turn.finish(provider)
}

/// Some servers ignore `stream: true` and answer with one completion object.
async fn read_completion<F>(
    provider: &str,
    response: reqwest::Response,
    control: &Control,
    mut publish: F,
) -> Result<Turn>
where
    F: FnMut(Delta),
{
    let bytes = tokio::select! {
        bytes = response.bytes() => bytes?,
        () = control.cancelled() => {
            control.ensure_running()?;
            unreachable!("cancelled control must fail ensure_running")
        }
    };
    if bytes.len() > MAX_STREAM_BYTES {
        bail!("{provider} response exceeded {MAX_STREAM_BYTES} bytes");
    }
    let value: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("invalid JSON in {provider} response"))?;
    fail_on_error(provider, &value)?;
    let message = value
        .pointer("/choices/0/message")
        .with_context(|| format!("{provider} returned no choices"))?;
    let mut turn = Accumulator::default();
    turn.apply(message, &mut publish);
    turn.finish(provider)
}

fn fail_on_error(provider: &str, value: &Value) -> Result<()> {
    if let Some(error) = value.get("error") {
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .map_or_else(|| error.to_string(), ToOwned::to_owned);
        bail!("{provider} returned an error: {message}");
    }
    Ok(())
}

#[derive(Default)]
struct Accumulator {
    text: String,
    calls: BTreeMap<usize, PartialCall>,
}

#[derive(Default)]
struct PartialCall {
    id: String,
    name: String,
    arguments: String,
}

impl Accumulator {
    /// Applies a streamed `delta` or a complete `message`. Both share one shape;
    /// a streamed tool call arrives as fragments keyed by `index`.
    fn apply<F>(&mut self, delta: &Value, publish: &mut F)
    where
        F: FnMut(Delta),
    {
        if let Some(text) = delta.get("content").and_then(Value::as_str)
            && !text.is_empty()
        {
            self.text.push_str(text);
            publish(Delta::Text(text.to_owned()));
        }
        // llama.cpp and DeepSeek use `reasoning_content`; Ollama and OpenRouter
        // use `reasoning`.
        for key in ["reasoning_content", "reasoning"] {
            if let Some(text) = delta.get(key).and_then(Value::as_str)
                && !text.is_empty()
            {
                publish(Delta::Reasoning(text.to_owned()));
            }
        }
        let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) else {
            return;
        };
        for (position, call) in calls.iter().enumerate() {
            let index = call
                .get("index")
                .and_then(Value::as_u64)
                .and_then(|index| usize::try_from(index).ok())
                .unwrap_or(position);
            let partial = self.calls.entry(index).or_default();
            if let Some(id) = call.get("id").and_then(Value::as_str)
                && !id.is_empty()
            {
                id.clone_into(&mut partial.id);
            }
            let Some(function) = call.get("function") else {
                continue;
            };
            if let Some(name) = function.get("name").and_then(Value::as_str)
                && !name.is_empty()
            {
                name.clone_into(&mut partial.name);
            }
            match function.get("arguments") {
                Some(Value::String(fragment)) => partial.arguments.push_str(fragment),
                // Ollama sends complete arguments as an object.
                Some(arguments @ Value::Object(_)) => partial.arguments = arguments.to_string(),
                _ => {}
            }
        }
    }

    fn finish(self, provider: &str) -> Result<Turn> {
        let calls = self
            .calls
            .into_values()
            .map(|call| {
                if call.name.is_empty() {
                    bail!("{provider} returned a tool call without a name");
                }
                Ok(ToolCall {
                    call_id: if call.id.is_empty() {
                        format!("call_{}", Uuid::new_v4().simple())
                    } else {
                        call.id
                    },
                    name: call.name,
                    arguments: if call.arguments.trim().is_empty() {
                        "{}".to_owned()
                    } else {
                        call.arguments
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Turn {
            text: self.text,
            calls,
            native: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn replay(chunks: &[Value]) -> (Turn, String) {
        let mut turn = Accumulator::default();
        let mut reasoning = String::new();
        for chunk in chunks {
            turn.apply(&chunk["choices"][0]["delta"], &mut |delta| {
                if let Delta::Reasoning(text) = delta {
                    reasoning.push_str(&text);
                }
            });
        }
        (turn.finish("test").unwrap(), reasoning)
    }

    #[test]
    fn llama_cpp_fragments_join_by_index() {
        let (turn, reasoning) = replay(&[
            json!({ "choices": [{ "delta": { "reasoning_content": "look" } }] }),
            json!({ "choices": [{ "delta": { "tool_calls": [{ "index": 0, "id": "c0", "type": "function", "function": { "name": "rename_page", "arguments": "" } }] } }] }),
            json!({ "choices": [{ "delta": { "tool_calls": [{ "index": 0, "function": { "arguments": "{\"page\":" } }] } }] }),
            json!({ "choices": [{ "delta": { "tool_calls": [{ "index": 0, "function": { "arguments": "\"p1\"}" } }] } }] }),
            json!({ "choices": [{ "delta": { "tool_calls": [{ "index": 1, "id": "c1", "function": { "name": "inspect_project", "arguments": "" } }] } }] }),
            json!({ "choices": [{ "delta": {}, "finish_reason": "tool_calls" }] }),
        ]);

        assert_eq!(reasoning, "look");
        assert_eq!(turn.calls.len(), 2);
        assert_eq!(turn.calls[0].call_id, "c0");
        assert_eq!(turn.calls[0].name, "rename_page");
        assert_eq!(turn.calls[0].arguments, r#"{"page":"p1"}"#);
        assert_eq!(turn.calls[1].arguments, "{}");
    }

    #[test]
    fn ollama_object_arguments_without_ids() {
        let (turn, _) = replay(&[
            json!({ "choices": [{ "delta": { "content": "On it. " } }] }),
            json!({ "choices": [{ "delta": { "tool_calls": [{ "function": { "name": "view_page", "arguments": { "page": "p1" } } }] } }] }),
        ]);

        assert_eq!(turn.text, "On it. ");
        assert_eq!(turn.calls[0].name, "view_page");
        assert!(turn.calls[0].call_id.starts_with("call_"));
        assert_eq!(
            serde_json::from_str::<Value>(&turn.calls[0].arguments).unwrap(),
            json!({ "page": "p1" })
        );
    }

    #[test]
    fn nameless_tool_call_fails_closed() {
        let mut turn = Accumulator::default();
        turn.apply(
            &json!({ "tool_calls": [{ "index": 0, "function": { "arguments": "{}" } }] }),
            &mut |_| {},
        );
        assert!(turn.finish("test").is_err());
    }
}
