# Koharu 0.70.3 — Architecture Map (fork baseline)

Baseline commit: `db439495` (`chore(release): 0.70.3`). The brief targets 0.70.2; the
working tree is 0.70.3 and nothing below differs between the two for the areas we touch.

Everything in this document was read out of the tree, not inferred. File and line
references are the anchors Phase 1 will modify.

---

## 1. Crate topology

```
koharu (bin)
└── koharu-app ──────── Tauri commands + specta binding generation
    ├── koharu-pipeline ─ stage scheduling, model residency, resource telemetry
    │   ├── koharu-translator ─ translation providers (local + 11 remote)
    │   │   └── koharu-ml ── model implementations
    │   │       ├── koharu-llama ──── safe llama.cpp wrapper
    │   │       │   └── koharu-llama-sys (bindgen, dynamic loading)
    │   │       ├── koharu-diffusion ─ safe stable-diffusion.cpp wrapper
    │   │       ├── koharu-torch
    │   │       └── koharu-runtime ── device discovery, HF download, package install
    │   └── koharu-scene / koharu-storage ─ ECS document + on-disk project
    ├── koharu-renderer / koharu-rasterizer / koharu-psd ─ typesetting + export
    ├── koharu-config ── live TOML config (~/.koharu/config.toml)
    ├── koharu-secrets ─ OS keychain
    └── koharu-agent ─── Codex agent host
```

Frontend: Next.js app in `packages/koharu`, shadcn-style primitives in `packages/ui`,
generated Tauri bindings in `packages/bridge/src/protocol.ts`.

---

## 2. Configuration: how settings are stored and reloaded

`crates/koharu-config/src/lib.rs`

- One process-wide TOML file: `~/.koharu/config.toml` (`path()`, line 374).
- `load::<T>("section")` returns a live `Config<T>` handle — `RwLock` value + revision
  counter + `tokio::sync::watch` change channel. Section state is deduplicated per
  process and type-checked (loading one section as two types is an error, line 300).
- **Deep merge with defaults on read** (`load_value` → `merge`, line 437). Unknown keys
  in the file survive; missing keys fall back to `T::default()`. This is why adding new
  fields to a config struct is automatically backward compatible.
- One special rule at line 440: when a table's `provider` or `model` **tag changes**,
  the whole subtree is replaced instead of merged. This exists so switching provider
  doesn't leave stale provider-specific keys. Any new tagged config we add inherits
  this behaviour, and it constrains how we name discriminants.
- `save()` re-reads the document, replaces only our section, writes atomically
  (`atomicwrites`). Concurrent section saves are safe (test at line 622).

Registered sections today: `pipeline`, `providers`, `typesetting`, plus agent config.

**Consequence for the fork:** new configuration should be a *new top-level section*
(e.g. `[llm]`, `[models]`, `[vram]`) rather than new fields inside `[pipeline]`.
New sections are invisible to upstream code, so upstream merges never conflict on them.

---

## 3. Pipeline configuration and stage selection

`crates/koharu-pipeline/src/config.rs`

`PipelineConfig` has a **hand-written `Serialize`/`Deserialize` pair** (lines 54–156)
that maps between two shapes:

| in-memory                                   | on disk                                  |
| ------------------------------------------- | ---------------------------------------- |
| `detection: DetectionModel` (enum + config) | `[detection] model = "..."`              |
| `ocr: OcrModel`                             | `[ocr] model = "..."`                    |
| `inpainting: InpaintingModel`               | `[inpainting] model = "..."`             |
| `translation: TranslationConfig`            | `[translation]`                          |
| `processor: ProcessorConfig`                | `[processor."<model-id>"]` per-model bag |

The `processor` table is the existing **"keep every model's settings even when it is
not the active model"** mechanism (comment at config.rs:14). `PipelineConfig::inpainting()`
(line 210) resolves the active enum by preferring the stored profile over the inline one.
`remember_pipeline_profiles` in `crates/koharu-app/src/commands/preferences.rs:181`
writes the active model's settings back into `processor` on every save.

**This is already a model-profile system.** The fork's "Model Profiles" requirement
extends it rather than replacing it.

Unknown model IDs are a hard deserialization error (`unsupported inpainting model {model}`,
line 143). Adding a model means touching both match arms plus `validate()`.

### Stage execution

- `Stage` = `Detection | Ocr | Translation | Inpainting` (`stage.rs`), fixed 4-variant enum.
- Dependency graph is a hardcoded `prerequisite()` function (`scheduler.rs:145`):
  `Detection → {Ocr, Inpainting}`, `Ocr → Translation`.
- `Scheduler` (`scheduler.rs`) runs a sliding window over pages; one job per stage at a
  time, different stages may run concurrently.
- `AcceleratorGate` (`accelerator.rs:23`) **serialises all accelerator work to one lane**
  — a `Semaphore(1)` — because heterogeneous CUDA pairs measured 2.5–4× slower together.
  On CPU there is no gate.
- `Execution` (`execution.rs`) drives the scheduler, commits patches to the scene,
  and rebases each stage output onto the latest snapshot.

### Model residency (current, real behaviour)

- Each stage processor owns a `ModelCell<M>` (`model_cell.rs`) = `Mutex<Option<M>>`.
  `ensure()` loads once; `unload()` drops it.
- **Models stay resident indefinitely.** There is no policy, no budget, no timer.
- The only eviction path is OOM recovery: `StageRunner::run_with_recovery`
  (`stage_runner.rs:50`) detects an out-of-memory error *by string matching*
  (`is_out_of_memory`, line 143 — `"out of memory"`, `"cuda_error_out_of_memory"`,
  `"not enough memory"`), then `AcceleratorGate::recover` unloads every *other* stage's
  model and retries **once**.
- `Pipeline::from_config` (`pipeline.rs:44`) subscribes to config changes and rebuilds
  the whole `StageRunner` on any edit — which drops all `ModelCell`s, i.e. **any settings
  change unloads every model.**

### Resource telemetry (current)

`resources.rs` + `resources/vram.rs`

- 100 ms sampling loop; host memory/CPU via `sysinfo`, VRAM via platform providers
  (`resources/windows.rs`, `resources/linux.rs`).
- Reports **per-device totals only**: `memory_budget_bytes`, `memory_used_bytes`,
  `utilization_percent`. Values are `Option` and become `None` when unavailable
  (`vram::unavailable`), so "unknown" is already modelled honestly.
- There is **no per-component attribution** (weights vs KV cache vs compute buffers).
  Producing that requires either llama.cpp/sd.cpp instrumentation or estimation — the
  brief's `measured` / `estimated` distinction is therefore mandatory, not optional.
- Surfaced to the UI as `ModelResources` (`commands/lifecycle.rs:99`) over a Tauri
  `Channel`, rendered by `components/editor/ResourceMonitor.tsx`.

---

## 4. Local LLM path — from UI to llama.cpp

This is the exact chain the brief's §4 settings must travel.

```
TranslationPreferences.tsx / GenerationPreferences.tsx
  → PipelineConfig.translation : TranslationConfig
      { model: ModelSelection, generation: GenerationConfig, target_language, instructions }
  → koharu_pipeline::stages::translation::Processor
  → koharu_translator::Translator::translate(selection, generation, request)
  → LocalTranslator (local/mod.rs)
      descriptor = catalog::MODELS.find(id == selection.model)      ← closed catalog
      resolved   = descriptor.resolve(selection)                     ← HF download
      Llm::load_with_options(device, path, LoadOptions { mtmd, ..default() })
      generation = descriptor.generation.options(user_generation)    ← merge
      llm.inference_with_json_schema(input, &generation, &schema)
  → koharu_ml::llm::Llm  → llm/model.rs::Model::inference
      context_values(prepared, options)                              ← dynamic n_ctx
      LlamaContextParams
  → koharu-llama → llama.cpp
```

### 4.1 The dynamic-context calculation (must be preserved)

`crates/koharu-ml/src/llm/model.rs:620` — `context_values`:

```rust
required = max(
    prompt_positions + max_tokens + 1,
    prompt_tokens    + max_tokens + 1,
    batch_tokens     + 1,
);
n_ctx = options.n_ctx.unwrap_or(required);          // explicit n_ctx < required → error
n_batch  = options.n_batch.unwrap_or(batch_tokens); // must be >= batch_tokens
n_ubatch = if non_causal { n_batch }
           else { options.n_ubatch.unwrap_or(min(n_batch, 512)) };  // must be <= n_batch
```

`prompt_positions` differs from `prompt_tokens` only for multimodal prompts
(`prepare_prompt`, line 397: MTMD image chunks occupy positions without being text
tokens). A fresh `LlamaContext` is created **per inference call** (`model.rs:248`) —
the context is not cached, only the model weights are.

Exactly matches the brief §3. Tests at `model.rs:783–817` lock this behaviour in.

### 4.2 What is already plumbed vs. what is not

`GenerationOptions` (`llm/mod.rs:246`) **already carries**:
`max_tokens, temperature, top_k, top_p, min_p, seed, repeat_penalty, repeat_last_n,
frequency_penalty, presence_penalty, add_special, n_ctx, n_batch, n_ubatch,
n_threads, n_threads_batch`.

`LoadOptions` (`llm/mod.rs:132`) **already carries**:
`gpu_layers (default 1000 = "all"), load_mode, eos_token_id, mtmd`.

`koharu-llama` `LlamaContextParams` **already exposes** (`context/params/get_set.rs`):
`with_type_k` / `with_type_v` taking `KvCacheType` (F32/F16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/
Q8_1/Q2_K…Q8_K/IQ*/BF16/TQ*/MXFP4 — `context/params.rs:159`), and
`with_flash_attention_policy(llama_flash_attn_type)`.

`GenerationConfig` (`koharu-translator/src/model.rs:9`) — the type the **UI actually
edits** — carries only:
`temperature, top_k, top_p, min_p, max_tokens, repeat_penalty, frequency_penalty,
presence_penalty, thinking`.

> **The gap is a single narrow layer.** `ModelGeneration::options()`
> (`koharu-translator/src/model.rs:108`) builds `GenerationOptions` and simply never
> sets `n_ctx`, `n_batch`, `n_ubatch`, `n_threads`, `n_threads_batch`. `LocalTranslator::load`
> (`local/mod.rs:37`) builds `LoadOptions` and never sets `gpu_layers` or `load_mode`.
> KV-cache type and flash attention are exposed by `koharu-llama` but never reach
> `LlamaContextParams` in `llm/model.rs::context_config` (line 593).
>
> Phase 1's LLM work is therefore *additive plumbing*, not a redesign.

### 4.3 The closed model catalog

`crates/koharu-translator/src/local/catalog.rs`

- `static MODELS: &[LocalModelDescriptor]` — 26 entries (asserted at line 972).
- Each entry: `id`, `name`, `quantizations: &[QuantizationDefinition{id,name,filename}]`,
  `generation: ModelGeneration`, `repository` (HF repo), `projector` (mmproj filename),
  `target_languages`.
- Resolution (`resolve`, line 927): `HuggingFaceFile::latest(repository, filename).resolve()`
  — downloads into the runtime store, joined with the projector download.
- `LocalTranslator::load` (`local/mod.rs:31`) **errors on any id not in `MODELS`**:
  `unknown local translator '{model}'`.
- `models()` (line 113) is what `get_translation_models` returns to the UI.
- Defaults: `DEFAULT_MODEL = "gemma4-12b-it"`, `DEFAULT_QUANTIZATION = "Q4_K_XL"`.

`LocalConfig` (`catalog.rs:35`) is **an empty struct** — `pub struct LocalConfig {}` —
serialized as `[providers.local]`. It is the natural, already-wired home for the fork's
custom-model registry and LLM runtime settings.

### 4.4 Provider registry

`crates/koharu-translator/src/provider.rs` — a `define_providers!` macro generates
`Provider` (enum), `ProviderConfig` (`#[serde(tag="provider", content="settings")]`),
and `ProvidersConfig` (a struct with one field per provider) from one declarative list.
`ProvidersConfig::from_entries` **requires every variant to be present** (line 101).
Adding a provider is a single macro entry; the UI iterates `entries()`.

### 4.5 Prompt construction

`crates/koharu-translator/src/prompt.rs` — the system prompt is a hardcoded `indoc!`
string in `translation_system_prompt` (line 79) with conditional paragraphs appended for
context / image / user instructions. The user message is JSON:
`{source_language, target_language, context, segments:[{id,text}]}`.
Output is constrained by a generated JSON Schema (`output_schema`, line 55) requiring
exactly one string per input id, enforced again in `translations()` (line 29).

That schema-per-request contract is what makes segment counts reliable; the Prompt
Studio (§22) must preserve it.

---

## 5. FLUX.2 Klein — what is actually hardcoded

> This section describes the **0.70.3 baseline**. The checkpoint sources, the
> inference settings and the working area were made configurable in
> `PHASE-2-NOTES.md`; text CFG, scheduler, sample method and the residency
> switch are still as described below.

`crates/koharu-ml/src/flux2_klein/mod.rs:23–31`

```rust
model_repository!("unsloth/FLUX.2-klein-4B-GGUF" @ "0084d1df…" {
    TRANSFORMER_WEIGHTS = "flux-2-klein-4b-Q4_K_M.gguf" });
model_repository!("black-forest-labs/FLUX.2-small-decoder" @ "a3efc24f…" {
    VAE_WEIGHTS = "full_encoder_small_decoder.safetensors" });
model_repository!("unsloth/Qwen3-4B-GGUF" @ "22c9fc8a…" {
    TEXT_ENCODER_WEIGHTS = "Qwen3-4B-Q4_K_M.gguf" });
```

`model_repository!` (`koharu-ml/src/lib.rs:6`) expands to `const … HuggingFaceFile`,
i.e. these are **compile-time constants pinned to specific commits**. Making them
configurable means introducing a runtime path resolution step — a real change to
`Flux2Klein{,Inpaint}::load`, which currently takes only a `Device`.

Also hardcoded:

| what | where | value |
| --- | --- | --- |
| max processing area | `mod.rs:80`, `:179`, `:230` | `1024 * 1024` (1 MP), three separate sites |
| text CFG | `mod.rs:109`, `:270` | `1.0` |
| scheduler / method | `mod.rs:113`, `:273` | `Flux2` / `Euler` |
| `Flux2KleinInpaintOptions` in the pipeline | `stages/inpainting.rs:237` | `::default()` — steps 4, strength 0.8, seed −1, no mask crop |
| VRAM-based residency switch | `flux2_klein/model.rs:55` | `memory_free >= 20 GiB` → keep params on GPU, else `params_backend = "*=cpu"` |

`Flux2KleinConfig` (`stages/inpainting.rs:34`) currently exposes **only `prompt`**.
Everything in `Flux2KleinInpaintOptions` (`processor.rs:31`) is reachable but unused.

> **On Klein 9B (brief §13):** the loader (`flux2_klein/model.rs:53`) builds a
> `ContextParams` with `vae_format: VaeFormat::Flux2` and hands three file paths to
> stable-diffusion.cpp, which infers architecture from GGUF metadata. Whether a 9B
> transformer works is a stable-diffusion.cpp question (does the pinned build support
> that parameter count / does the small decoder match its latent width), **not** a
> path-substitution question. This must be verified against the pinned
> stable-diffusion.cpp release before any 9B option ships. Deferred out of Phase 1.

---

## 6. Model acquisition and storage

`crates/koharu-runtime`

- `Store::root()` (`store.rs:50`) = `<os cache dir>/koharu/packages`, overridable once
  per process.
- `HuggingFaceFile` (`source/hugging_face.rs`) — `pinned(repo, sha, file)` or
  `latest(repo, file)`. `latest` resolves the repo HEAD **once per process** and caches
  it, so a multi-file model can't be assembled from mixed revisions (doc comment line 44).
  Files land at `…/hugging-face/models/<owner>--<name>/snapshots/<sha>/<file>`.
  Path traversal is rejected (line 116).
- Downloads go through `downloads::Transfer` (`downloads.rs`) — multi-part, resumable,
  and it **already broadcasts progress globally** via `downloads::subscribe()`
  (`Event::Started|Progress|Finished|Failed`). The app forwards these to the UI on the
  `Download` channel. A HF-browser download UI can reuse this for free.
- There is **no repository *listing* API client** — only single-file resolve. Browsing
  `.gguf` files in a repo (brief §6) needs a new call to
  `https://huggingface.co/api/models/{repo}` (the response is already partially modelled
  as `struct Repository { sha }` at line 160; the `siblings` array is what we need).
- Native runtime binaries (llama.cpp b10430, stable-diffusion.cpp, CUDA/ROCm) are
  downloaded per hardware variant (`runtime/packages/llama.rs`). The selected variant
  is what determines which llama.cpp features are actually available at run time —
  this is the hook for capability detection.

### Device discovery

`crates/koharu-runtime/src/hardware/mod.rs` — probes CUDA / HIP / Vulkan / Metal once
(`OnceLock`), builds `Vec<Device>` with `memory_total` / `memory_free` /
`compute_capability`, ranks candidates, and selects **exactly one**.
`Device` (`device.rs:47`) carries `index`, `name`, `backend`, `device_type`, memory.

**The whole app runs on one device.** `koharu_ml::device(false)` is called once in
`app.rs:20` and cloned into every stage. Multi-GPU (§30) means threading a per-stage
`Device` through `Stages::new` — the plumbing is uniform (every model's `load` already
takes a `Device`), so it is tractable, but it is a Phase 3 change.

---

## 7. Project persistence

`crates/koharu-app/src/commands/project.rs:182`

- Library root: `~/Documents/Koharu/`.
- One project = a directory `<name>.khrproj/` containing `state-a.khr` / `state-b.khr`
  (double-buffered scene state) and `blobs/` (content-addressed images).
- `koharu-storage::Session` owns the format; `koharu-scene` is the ECS document
  (entities + components + patches + revisions, with rebase/validate semantics).
- **There is no per-project settings file today.** All settings are global
  (`~/.koharu/config.toml`).

**Consequence:** project profiles (§33) can be a *new sidecar file inside the
`.khrproj` directory* — e.g. `profile.toml`. That neither touches the storage format
nor the scene schema, so existing projects keep opening unchanged, and upstream's
`Session` code is untouched.

---

## 8. UI surface

- Settings live in `packages/koharu/components/preferences/`:
  `SettingsPage.tsx` (tab shell: appearance / pipeline / providers / translation /
  typesetting / shortcuts), `PipelinePreferences.tsx`, `TranslationPreferences.tsx`,
  `GenerationPreferences.tsx`, `ProviderPreferences.tsx`, `TypesettingPreferences.tsx`,
  and shared `PreferenceFields.tsx` (`PreferencePage` / `PreferenceSection` /
  `PreferenceRow` / `NumberField` / `TextField`).
- Saving is **debounced autosave** (`SettingsPage.tsx:145`, 260 ms) through
  `commands.savePreferences(pipeline, providers, typesetting)`, with a serialized
  save queue and generation counter to drop stale responses.
- `packages/koharu/components/preferences/models.ts` holds the frontend's model-name
  tables and `replaceStage`/`stageModel` helpers — a mirror of the Rust enums.
- Types come from `packages/bridge/src/protocol.ts`, generated by
  `cargo run -p koharu-app --bin generate` (`crates/koharu-app/src/bin/generate.rs`).
  **Never hand-edit it.**
- Every new Tauri command must be registered in `commands::bindings()`
  (`commands/mod.rs:57`) and the bindings regenerated.

---

## 9. Constraints this fork inherits

1. **Upstream's `AGENTS.md` says "Never add backward compatibility."** The fork brief
   (§37, §39) requires the opposite for *user data* (config/project migration). These
   are reconcilable: we keep upstream's rule for in-repo API churn, and honour the
   fork's rule for on-disk formats. Where they collide, the fork brief wins and the
   reason gets recorded next to the code.
2. **`koharu-config`'s deep merge already gives additive-field compatibility.** Most
   "migration" is therefore free; only renames and tag changes need explicit handling.
3. **Verification policy** (upstream `AGENTS.md`): run the smallest relevant check on
   the debug profile. No full-suite runs unless asked.
4. **`packages/koharu/AGENTS.md`**: the vendored Next.js differs from public releases —
   consult `node_modules/next/dist/docs/` before writing app-router code.
5. Generated files (`packages/bridge/src/protocol.ts`, WASM output) are derived; change
   the Rust source and regenerate.
6. **The workspace needs rustc ≥ 1.95** (`sysinfo 0.39.6`), but there is no
   `rust-toolchain.toml` pinning it — a stale stable toolchain fails the whole build
   with a dependency error rather than a clear message. Verified building on 1.97.1.

---

## 10. Where each brief requirement lands

| Brief | Anchor in this tree | Nature of change |
| --- | --- | --- |
| §3 dynamic context (keep) | `koharu-ml/src/llm/model.rs:620` | preserve; add optional bounds around it |
| §4 LLM advanced settings | `koharu-translator/src/model.rs:108`, `local/mod.rs:37`, `koharu-ml/src/llm/model.rs:593` | plumb existing fields; add KV-type + flash-attn to `GenerationOptions`/`LoadOptions` |
| §5 custom GGUF | `koharu-translator/src/local/catalog.rs` (`MODELS`, `LocalTranslator::load`), `LocalConfig` | add a runtime registry alongside the static catalog |
| §6 HF GGUF download | `koharu-runtime/src/source/hugging_face.rs`, `downloads.rs` | new repo-listing call; reuse existing transfer + progress |
| §7–9 catalog metadata | new data file + `Model` type in `koharu-translator/src/model.rs` | additive |
| §10–11 VRAM budget | `koharu-pipeline/src/resources*.rs`, `runtime/hardware` | additive telemetry + estimator |
| §12 lifecycle policy | `koharu-pipeline/src/model_cell.rs`, `accelerator.rs`, `stage_runner.rs` | new policy layer around `ModelCell` |
| §13–14 FLUX config | `koharu-ml/src/flux2_klein/mod.rs`, `stages/inpainting.rs:34` | replace consts with resolved paths; widen `Flux2KleinConfig` |
| §16 inpainting router | `koharu-pipeline/src/stages/inpainting.rs` | new pre-stage classifier |
| §18 OCR ensemble | `koharu-pipeline/src/stages/ocr.rs` | new composite processor |
| §19–23 TM / context / prompts | `koharu-translator/src/prompt.rs`, `backend.rs`, project sidecar | prompt templating + per-project data |
| §24–27 QA / retry / multi-pass | `koharu-pipeline/src/stages/translation.rs`, `scheduler.rs` | new stages; `Stage` enum grows |
| §30 multi-GPU | `koharu-pipeline/src/stages/mod.rs:77`, `accelerator.rs` | per-stage `Device` |
| §32–34 profiles | `koharu-config`, new sections + project sidecar | new layered resolver |
| §35 workflow graph | `koharu-pipeline/src/scheduler.rs:145` | replace `prerequisite()` with data |
