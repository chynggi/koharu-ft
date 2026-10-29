// https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create

mod stream;

use anyhow::{Result, bail};
use koharu_secrets::ExposeSecret as _;
use koharu_translator::{ChatEndpoint, Provider, ProvidersConfig, Translator};
use reqwest::Client;
use serde_json::{Value, json};

use crate::{
    AgentModel, Control, Tool, ToolImage,
    backend::{Delta, Item, Prompt, Turn},
};

pub(crate) struct Chat {
    client: Client,
    provider: Provider,
    endpoint: ChatEndpoint,
}

impl Chat {
    pub(crate) fn connect(client: Client, provider: Provider) -> Result<Self> {
        let providers = ProvidersConfig::load()?;
        let endpoint = koharu_translator::chat_endpoint(&*providers.read()?, provider)?;
        Ok(Self {
            client,
            provider,
            endpoint,
        })
    }

    pub(crate) async fn respond<F>(
        &self,
        prompt: &Prompt<'_>,
        control: &Control,
        publish: F,
    ) -> Result<Turn>
    where
        F: FnMut(Delta),
    {
        control.ensure_running()?;
        let request = self
            .client
            .post(&self.endpoint.url)
            .header("accept", "text/event-stream")
            .json(&request(prompt));
        let request = match &self.endpoint.api_key {
            Some(api_key) => request.bearer_auth(api_key.expose_secret()),
            None => request,
        };
        let response = tokio::select! {
            response = request.send() => response?,
            () = control.cancelled() => {
                control.ensure_running()?;
                unreachable!("cancelled control must fail ensure_running")
            }
        };
        let name = self.provider.name();
        if !response.status().is_success() {
            let status = response.status();
            let mut body = response.text().await.unwrap_or_default();
            body.truncate(16 * 1024);
            bail!("{name} returned {status}: {body}");
        }
        stream::read(name, response, control, publish).await
    }
}

pub(crate) async fn models(provider: Provider) -> Result<Vec<AgentModel>> {
    Ok(Translator::models()
        .await?
        .into_iter()
        .filter(|model| model.provider == provider)
        .filter_map(|model| {
            Some(AgentModel {
                id: model.model?,
                name: model.name,
                reasoning: Vec::new(),
            })
        })
        .collect())
}

// Chat Completions has no portable reasoning-effort control across hosted and
// local servers, so `prompt.reasoning` applies to Codex only.
fn request(prompt: &Prompt<'_>) -> Value {
    json!({
        "model": prompt.model,
        "messages": messages(prompt.instructions, prompt.items),
        "tools": prompt.tools.iter().map(tool).collect::<Vec<_>>(),
        "tool_choice": "auto",
        "stream": true,
    })
}

fn tool(tool: &Tool) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    })
}

fn messages(instructions: &str, items: &[Item]) -> Vec<Value> {
    let mut messages = vec![json!({ "role": "system", "content": instructions })];
    // Tool messages must directly follow the assistant message that called
    // them, and most servers reject images inside tool content. Images from a
    // run of tool results therefore follow as one user message.
    let mut images: Vec<&ToolImage> = Vec::new();
    for item in items {
        if !matches!(item, Item::ToolResult { .. }) {
            flush_images(&mut messages, &mut images);
        }
        match item {
            Item::User(text) => messages.push(json!({ "role": "user", "content": text })),
            Item::Assistant { text, calls, .. } => {
                let mut message = json!({
                    "role": "assistant",
                    "content": if text.is_empty() && !calls.is_empty() {
                        Value::Null
                    } else {
                        Value::String(text.clone())
                    },
                });
                if !calls.is_empty() {
                    message["tool_calls"] = calls
                        .iter()
                        .map(|call| {
                            json!({
                                "id": call.call_id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": call.arguments,
                                },
                            })
                        })
                        .collect();
                }
                messages.push(message);
            }
            Item::ToolResult {
                call_id,
                output,
                images: results,
            } => {
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": output.to_string(),
                }));
                images.extend(results);
            }
        }
    }
    flush_images(&mut messages, &mut images);
    messages
}

fn flush_images(messages: &mut Vec<Value>, images: &mut Vec<&ToolImage>) {
    if images.is_empty() {
        return;
    }
    let content = images
        .drain(..)
        .flat_map(|image| {
            [
                json!({ "type": "text", "text": image.label }),
                json!({ "type": "image_url", "image_url": { "url": image.data_url } }),
            ]
        })
        .collect::<Vec<_>>();
    messages.push(json!({ "role": "user", "content": content }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Reasoning, ToolCall};

    fn call(id: &str) -> ToolCall {
        ToolCall {
            call_id: id.to_owned(),
            name: "view_page".to_owned(),
            arguments: "{}".to_owned(),
        }
    }

    fn image(label: &str) -> ToolImage {
        ToolImage {
            label: label.to_owned(),
            data_url: "data:image/png;base64,AA".to_owned(),
        }
    }

    #[test]
    fn transcript_maps_to_chat_messages() {
        let items = [
            Item::User("prompt".to_owned()),
            Item::Assistant {
                text: String::new(),
                calls: vec![call("a"), call("b")],
                native: Vec::new(),
            },
            Item::ToolResult {
                call_id: "a".to_owned(),
                output: json!({ "ok": true }),
                images: vec![image("Page 1")],
            },
            Item::ToolResult {
                call_id: "b".to_owned(),
                output: json!({ "ok": true }),
                images: vec![image("Page 2")],
            },
            Item::Assistant {
                text: "done".to_owned(),
                calls: Vec::new(),
                native: Vec::new(),
            },
        ];
        let tools = [Tool::new("view_page", "View.", json!({ "type": "object" }))];
        let value = request(&Prompt {
            model: "qwen3",
            instructions: "system",
            items: &items,
            tools: &tools,
            reasoning: Reasoning::Medium,
            session: "session",
        });
        let messages = value["messages"].as_array().unwrap();
        let roles = messages
            .iter()
            .map(|message| message["role"].as_str().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(
            roles,
            [
                "system",
                "user",
                "assistant",
                "tool",
                "tool",
                "user",
                "assistant"
            ]
        );
        assert_eq!(messages[2]["content"], Value::Null);
        assert_eq!(messages[2]["tool_calls"][1]["id"], "b");
        assert_eq!(messages[3]["tool_call_id"], "a");
        assert_eq!(messages[5]["content"].as_array().unwrap().len(), 4);
        assert_eq!(messages[5]["content"][3]["type"], "image_url");
        assert_eq!(messages[6]["content"], "done");
        assert_eq!(value["tools"][0]["function"]["name"], "view_page");
        assert!(value.get("reasoning_effort").is_none());
    }
}
