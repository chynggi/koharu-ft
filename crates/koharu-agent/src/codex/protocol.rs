use serde::Serialize;
use serde_json::{Value, json};

use crate::{
    Tool, ToolImage,
    backend::{Item, Prompt},
};

#[derive(Debug, Serialize)]
pub(super) struct Request {
    pub(super) model: String,
    pub(super) instructions: String,
    pub(super) input: Vec<Value>,
    pub(super) tools: Vec<FunctionTool>,
    pub(super) tool_choice: &'static str,
    pub(super) parallel_tool_calls: bool,
    pub(super) reasoning: ReasoningOptions,
    pub(super) text: TextOptions,
    pub(super) include: [&'static str; 1],
    pub(super) stream: bool,
    pub(super) store: bool,
    pub(super) prompt_cache_key: String,
}

#[derive(Debug, Serialize)]
pub(super) struct FunctionTool {
    #[serde(rename = "type")]
    kind: &'static str,
    name: String,
    description: String,
    parameters: Value,
    strict: bool,
}

impl From<&Tool> for FunctionTool {
    fn from(tool: &Tool) -> Self {
        Self {
            kind: "function",
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: tool.parameters.clone(),
            strict: false,
        }
    }
}

#[derive(Debug, Serialize)]
pub(super) struct ReasoningOptions {
    effort: &'static str,
    summary: &'static str,
}

#[derive(Debug, Serialize)]
pub(super) struct TextOptions {
    verbosity: &'static str,
}

impl Request {
    pub(super) fn new(prompt: &Prompt<'_>) -> Result<Self, serde_json::Error> {
        Ok(Self {
            model: prompt.model.to_owned(),
            instructions: prompt.instructions.to_owned(),
            input: input(prompt.items)?,
            tools: prompt.tools.iter().map(FunctionTool::from).collect(),
            tool_choice: "auto",
            parallel_tool_calls: false,
            reasoning: ReasoningOptions {
                effort: prompt.reasoning.as_str(),
                summary: "auto",
            },
            text: TextOptions { verbosity: "low" },
            include: ["reasoning.encrypted_content"],
            stream: true,
            store: false,
            prompt_cache_key: prompt.session.to_owned(),
        })
    }
}

fn input(items: &[Item]) -> Result<Vec<Value>, serde_json::Error> {
    let mut input = Vec::with_capacity(items.len());
    for item in items {
        match item {
            Item::User(text) => input.push(message("user", text)),
            Item::Assistant { native, .. } => input.extend(native.iter().cloned()),
            Item::ToolResult {
                call_id,
                output,
                images,
            } => input.push(function_output(call_id, output, images)?),
        }
    }
    Ok(input)
}

fn message(role: &str, text: &str) -> Value {
    json!({
        "type": "message",
        "role": role,
        "content": [{
            "type": "input_text",
            "text": text,
        }],
    })
}

fn function_output(
    call_id: &str,
    output: &Value,
    images: &[ToolImage],
) -> Result<Value, serde_json::Error> {
    let output = if images.is_empty() {
        Value::String(serde_json::to_string(output)?)
    } else {
        let mut content = vec![json!({
            "type": "input_text",
            "text": serde_json::to_string(output)?,
        })];
        for image in images {
            content.push(json!({
                "type": "input_text",
                "text": image.label,
            }));
            content.push(json!({
                "type": "input_image",
                "image_url": image.data_url,
                "detail": "high",
            }));
        }
        Value::Array(content)
    };
    Ok(json!({
        "type": "function_call_output",
        "call_id": call_id,
        "output": output,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Reasoning, ToolCall};

    #[test]
    fn transcript_maps_to_responses_items() {
        let native = json!({ "type": "function_call", "call_id": "c1", "name": "inspect_project", "arguments": "{}" });
        let items = [
            Item::User("context".to_owned()),
            Item::Assistant {
                text: String::new(),
                calls: vec![ToolCall {
                    call_id: "c1".to_owned(),
                    name: "inspect_project".to_owned(),
                    arguments: "{}".to_owned(),
                }],
                native: vec![native.clone()],
            },
            Item::ToolResult {
                call_id: "c1".to_owned(),
                output: json!({ "ok": true }),
                images: vec![ToolImage {
                    label: "Page 1".to_owned(),
                    data_url: "data:image/png;base64,AA".to_owned(),
                }],
            },
        ];
        let tools = [Tool::new(
            "inspect_project",
            "Inspect.",
            json!({ "type": "object" }),
        )];
        let request = Request::new(&Prompt {
            model: "codex",
            instructions: "instructions",
            items: &items,
            tools: &tools,
            reasoning: Reasoning::High,
            session: "session",
        })
        .unwrap();
        let value = serde_json::to_value(&request).unwrap();

        assert_eq!(value["input"][0], message("user", "context"));
        assert_eq!(value["input"][1], native);
        assert_eq!(value["input"][2]["type"], "function_call_output");
        assert_eq!(value["input"][2]["output"][0]["text"], r#"{"ok":true}"#);
        assert_eq!(value["input"][2]["output"][2]["type"], "input_image");
        assert_eq!(
            value["tools"][0],
            json!({
                "type": "function",
                "name": "inspect_project",
                "description": "Inspect.",
                "parameters": { "type": "object" },
                "strict": false,
            })
        );
        assert_eq!(value["reasoning"]["effort"], "high");
        assert_eq!(value["prompt_cache_key"], "session");
    }
}
