//! Local LLM runtime diagnostics and GGUF registration helpers.
//!
//! Registering a model is an edit to `providers.local.models`, so it travels
//! through the existing `save_preferences` command. Only the parts that need
//! native access — a file dialog and the runtime's own view of the device —
//! live here.

use std::path::PathBuf;

use koharu_ml::{Backend, DeviceType};
use serde::Serialize;
use specta::Type;
use tauri::{Cef, WebviewWindow};

use super::Error;

/// What the current build and device actually support.
///
/// Anything llama.cpp only validates while creating a context is reported in
/// `deferred` rather than guessed at: the UI keeps those controls enabled and
/// surfaces llama.cpp's own error if a value is rejected.
#[derive(Clone, Debug, Serialize, Type)]
pub struct LlmCapabilities {
    /// Human-readable accelerator description, or `CPU`.
    pub device: String,
    pub backend: String,
    /// Whether layers can be offloaded at all. `false` means the GPU Layers
    /// control has no effect.
    pub gpu_offload: bool,
    /// Total device memory in bytes, when the driver reports it.
    #[specta(type = Option<f64>)]
    pub total_memory: Option<u64>,
    /// Settings that are only validated when a context is created.
    pub deferred: Vec<DeferredCapability>,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct DeferredCapability {
    /// Field name in the runtime settings, e.g. `flash_attention`.
    pub setting: String,
    pub reason: String,
}

#[tauri::command]
#[specta::specta]
pub async fn get_llm_capabilities() -> std::result::Result<LlmCapabilities, Error> {
    let device = koharu_ml::device(false);
    let accelerated = device.backend != Backend::Cpu && device.device_type != DeviceType::Cpu;
    // Mirrors the condition koharu-ml applies when building model params: an
    // accelerator is only used when the loaded llama.cpp build can offload to it.
    let gpu_offload = accelerated
        && koharu_ml::llama_backend().is_some_and(koharu_ml::llama::llama_backend::LlamaBackend::supports_gpu_offload);

    let mut deferred = vec![
        DeferredCapability {
            setting: "flash_attention".to_owned(),
            reason: "llama.cpp selects flash attention per backend and model when set to Auto, \
                     and only reports incompatibility while creating a context."
                .to_owned(),
        },
        DeferredCapability {
            setting: "kv_cache_type".to_owned(),
            reason: "A quantized KV cache generally needs flash attention. llama.cpp rejects an \
                     unsupported combination when it creates a context."
                .to_owned(),
        },
    ];
    if !gpu_offload {
        deferred.push(DeferredCapability {
            setting: "gpu_layers".to_owned(),
            reason: if accelerated {
                "The active llama.cpp build does not advertise GPU offload.".to_owned()
            } else {
                "No accelerator is selected, so every layer runs on the CPU.".to_owned()
            },
        });
    }

    Ok(LlmCapabilities {
        device: device.description.clone(),
        backend: device.backend.to_string(),
        gpu_offload,
        total_memory: (device.memory_total > 0).then_some(device.memory_total as u64),
        deferred,
    })
}

/// Opens a native picker for a `.gguf` file. Returns `None` when dismissed.
#[tauri::command]
#[specta::specta]
pub async fn pick_gguf_file(
    window: WebviewWindow<Cef>,
) -> std::result::Result<Option<PathBuf>, Error> {
    Ok(rfd::AsyncFileDialog::new()
        .add_filter("GGUF model", &["gguf"])
        .set_parent(&window)
        .pick_file()
        .await
        .map(|file| file.path().to_owned()))
}
