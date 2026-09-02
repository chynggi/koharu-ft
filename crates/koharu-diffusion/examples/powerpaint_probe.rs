//! Throwaway probe: can stable-diffusion.cpp take PowerPaint V1?
//!
//! Stage 1 converts the diffusers components into a single GGUF.
//! Stage 2 loads that GGUF and runs one inpaint with the task embedding.
//!
//! ```text
//! cargo run -p koharu-diffusion --example powerpaint_probe -- convert \
//!     <clip_l.safetensors> <unet.safetensors> <vae.safetensors> <out.gguf>
//! cargo run -p koharu-diffusion --example powerpaint_probe -- inpaint \
//!     <model.gguf> <embeddings_dir> <image.png> <mask.png> <out.png>
//! ```

use std::{env, error::Error, path::PathBuf};

use koharu_diffusion::{
    ComponentConversion, Context, ContextParams, Embedding, ImageGenerationParams, Progress,
    WeightType, convert_with_components, set_log_callback, set_progress_callback,
};

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args().skip(1);
    let stage = arguments
        .next()
        .ok_or("usage: powerpaint_probe <convert|inpaint> ...")?;
    match stage.as_str() {
        "convert" => convert_stage(arguments.map(PathBuf::from).collect()),
        "inpaint" => inpaint_stage(arguments.collect()),
        other => Err(format!("unknown stage {other}").into()),
    }
}

fn convert_stage(paths: Vec<PathBuf>) -> Result<(), Box<dyn Error>> {
    let [clip_l, unet, vae, output] = <[PathBuf; 4]>::try_from(paths)
        .map_err(|_| "convert takes <clip_l> <unet> <vae> <output.gguf>, `-` to skip one")?;
    // `-` skips a component so a crash can be narrowed down to a single one.
    let optional = |path: PathBuf| (path != PathBuf::from("-")).then_some(path);

    set_log_callback(|message| eprintln!("[{:?}] {}", message.level, message.text))?;

    let params = ComponentConversion {
        model_path: None,
        clip_l_path: optional(clip_l),
        clip_g_path: None,
        t5xxl_path: None,
        diffusion_model_path: optional(unet),
        vae_path: optional(vae),
        output_path: output.clone(),
        output_type: WeightType::F16,
        tensor_type_rules: None,
        convert_tensor_names: env::var("PROBE_CONVERT_NAMES").as_deref() != Ok("0"),
        n_threads: std::thread::available_parallelism()?.get() as i32,
    };
    convert_with_components(&params)?;
    let size = std::fs::metadata(&output)?.len();
    eprintln!("wrote {} ({:.1} MB)", output.display(), size as f64 / 1e6);
    Ok(())
}

fn inpaint_stage(arguments: Vec<String>) -> Result<(), Box<dyn Error>> {
    let [model, embeddings_dir, image_path, mask_path, output_path] =
        <[String; 5]>::try_from(arguments)
            .map_err(|_| "inpaint takes <model.gguf> <embeddings_dir> <image> <mask> <output>")?;

    let embeddings = ["P_ctxt", "P_obj", "P_shape"]
        .into_iter()
        .map(|name| Embedding {
            name: name.to_string(),
            path: PathBuf::from(&embeddings_dir).join(format!("{name}.safetensors")),
        })
        .collect();

    set_progress_callback(|Progress { step, steps, .. }| eprint!("\rstep {step}/{steps}"))?;

    let context_params = ContextParams {
        model_path: Some(PathBuf::from(model)),
        embeddings,
        ..ContextParams::default()
    };
    let mut context = Context::new(&context_params)?;

    let image = image::open(&image_path)?.to_rgb8();
    let mask = image::open(&mask_path)?.to_luma8();
    let (width, height) = (image.width() as i32, image.height() as i32);

    let generation_params = ImageGenerationParams {
        // The object-removal task prompt, expressed through the extracted embedding.
        prompt: "P_ctxt".into(),
        negative_prompt: "P_obj".into(),
        init_image: Some(image),
        mask_image: Some(mask),
        width,
        height,
        strength: 1.0,
        ..ImageGenerationParams::default()
    };
    let result = context
        .generate_image(&generation_params)?
        .into_iter()
        .next()
        .ok_or("generation returned no images")?;
    result.save(&output_path)?;
    eprintln!("\nwrote {output_path}");
    Ok(())
}
