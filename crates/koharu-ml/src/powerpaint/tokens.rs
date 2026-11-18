//! PowerPaint's task prompts out of its text encoder. PowerPaint adds each
//! task as ten tokens past CLIP's vocabulary, `P_ctxt_0` to `P_ctxt_9`, and
//! learns their rows of the token embedding. stable-diffusion.cpp holds
//! CLIP's vocabulary as it is, so the encoder it loads keeps only CLIP's
//! rows, and each task's ten rows become a textual inversion named after it,
//! which a prompt word expands to, as PowerPaint's tokenizer expands it.
//! The encoder is saved under Stable Diffusion's own checkpoint names, by
//! which stable-diffusion.cpp tells a Stable Diffusion 1 model from others;
//! diffusers' names alone, which the rest of the model has, it can't tell.
//! Both are made once, beside the encoder they come from.
//! https://github.com/Sanster/IOPaint/blob/61a759fb3f332bacdce8b2813f4837495c9b86e0/iopaint/model/power_paint/powerpaint_tokenizer.py

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use anyhow::{Context as _, Result, bail, ensure};
use safetensors::{SafeTensors, tensor::TensorView};

/// The token embedding's parameter in the text encoder.
const EMBEDDING: &str = "text_model.embeddings.token_embedding.weight";
/// Where a Stable Diffusion 1 checkpoint keeps its text encoder.
const CHECKPOINT: &str = "cond_stage_model.transformer.";

pub(super) struct Tokens {
    /// The text encoder with CLIP's vocabulary only, as a checkpoint.
    pub(super) text_encoder: PathBuf,
    /// Each task's word and its inversion.
    pub(super) embeddings: Vec<(String, PathBuf)>,
}

/// The encoder and inversions made of `text_encoder` and the token ids
/// `added` gives, made now if they aren't yet.
pub(super) fn split(text_encoder: &Path, added: &Path) -> Result<Tokens> {
    let added: BTreeMap<String, u32> = serde_json::from_slice(&fs_err::read(added)?)?;
    // Each task's ids, in its vectors' order.
    let mut tasks = BTreeMap::<String, Vec<(u32, u32)>>::new();
    for (token, id) in &added {
        let (task, index) = token
            .rsplit_once('_')
            .and_then(|(task, index)| Some((task, index.parse::<u32>().ok()?)))
            .with_context(|| format!("{token} isn't a task's token"))?;
        tasks.entry(task.to_owned()).or_default().push((index, *id));
    }
    let vocabulary = *added.values().min().context("PowerPaint adds no tokens")?;
    let folder = text_encoder
        .parent()
        .context("the text encoder is in a folder")?
        .join("koharu");
    let trimmed = folder.join("text-encoder.safetensors");
    let embeddings: Vec<(String, PathBuf)> = tasks
        .keys()
        .map(|task| (task.clone(), folder.join(format!("{task}.safetensors"))))
        .collect();
    if trimmed.is_file() && embeddings.iter().all(|(_, path)| path.is_file()) {
        return Ok(Tokens {
            text_encoder: trimmed,
            embeddings,
        });
    }

    fs_err::create_dir_all(&folder)?;
    let bytes = fs_err::read(text_encoder)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let table = tensors.tensor(EMBEDDING)?;
    let [rows, width] = table.shape() else {
        bail!("the token embedding isn't a table");
    };
    ensure!(
        *rows == vocabulary as usize + added.len(),
        "the token embedding has {rows} rows, not CLIP's and PowerPaint's"
    );
    let row = width * table.dtype().bitsize() / 8;
    let data = table.data();
    for (task, path) in &embeddings {
        let mut ids = tasks[task].clone();
        ids.sort_unstable();
        let rows: Vec<u8> = ids
            .iter()
            .flat_map(|(_, id)| &data[*id as usize * row..(*id as usize + 1) * row])
            .copied()
            .collect();
        let view = TensorView::new(table.dtype(), vec![ids.len(), *width], &rows)?;
        write(path, [("emb_params".to_owned(), view)])?;
    }
    let kept = TensorView::new(
        table.dtype(),
        vec![vocabulary as usize, *width],
        &data[..vocabulary as usize * row],
    )?;
    let encoder = tensors.tensors().into_iter().map(|(name, view)| {
        let view = if name == EMBEDDING {
            kept.clone()
        } else {
            view
        };
        (format!("{CHECKPOINT}{name}"), view)
    });
    write(&trimmed, encoder)?;
    Ok(Tokens {
        text_encoder: trimmed,
        embeddings,
    })
}

/// `tensors` saved to `path`, which appears only once whole.
fn write<'a>(
    path: &Path,
    tensors: impl IntoIterator<Item = (String, TensorView<'a>)>,
) -> Result<()> {
    let partial = path.with_extension("partial");
    safetensors::serialize_to_file(tensors, None, &partial)?;
    fs_err::rename(&partial, path)?;
    Ok(())
}
