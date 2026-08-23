use std::{
    collections::BTreeMap,
    io::Cursor,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use anyhow::{Context as _, Result, anyhow, bail, ensure};
use async_trait::async_trait;
use image::{
    DynamicImage, GenericImageView as _, GrayImage, ImageBuffer, ImageFormat, Luma, Rgb, RgbImage,
    Rgba, RgbaImage,
};
use imageproc::region_labelling::{Connectivity, connected_components};
use koharu_ml::{
    aot_inpainting::AotInpainting,
    flux2_klein::{Flux2KleinInpaint, Flux2KleinInpaintOptions, Flux2KleinSource},
    lama::{InpaintRequest, LaMa},
    manga_inpaintor::{MangaInpaintor, MangaSource},
    mi_gan::MiGan,
    powerpaint::{PowerPaint, PowerPaintOptions, PowerPaintPaths},
    rorem_mixed::{DEFAULT_NEGATIVE_PROMPT, DEFAULT_PROMPT, RoremMixed, RoremMixedOptions},
    source::ComponentSource,
};
use koharu_scene::{
    AssetInput, AssetMetadata, AssetRole, At, BubbleRegion, EntityOrigin, Geometry, Origin,
    RasterLayer, RasterLayerKind, Region, RegionSpec,
};
use serde::{Deserialize, Serialize};
use specta::Type;

use super::{StageInput, StageProcessor, finish, generation};
use crate::{InpaintingModel, ModelCell, resources::ResourceMonitor};

const PRODUCER: &str = "dev.koharu.pipeline.inpainting";

/// Where one FLUX.2 Klein checkpoint is loaded from.
///
/// Tagged with `kind` so that `koharu_config`'s merge treats a changed variant
/// as a replaced subtree instead of blending two shapes.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ComponentSourceConfig {
    /// The repository Koharu pins for this component.
    #[default]
    Builtin,
    /// A checkpoint already on disk. Nothing is downloaded and the file is only read.
    LocalFile { path: PathBuf },
    /// Any Hugging Face repository. Without a revision the repository head is used.
    HuggingFace {
        repository: String,
        #[serde(default)]
        revision: Option<String>,
        filename: String,
    },
    /// An arbitrary URL. `digest` is a 64-character BLAKE3 hex digest (case-insensitive).
    Url { url: String, digest: String },
}

impl From<ComponentSourceConfig> for ComponentSource {
    fn from(value: ComponentSourceConfig) -> Self {
        match value {
            ComponentSourceConfig::Builtin => Self::Builtin,
            ComponentSourceConfig::LocalFile { path } => Self::LocalFile(path),
            ComponentSourceConfig::HuggingFace {
                repository,
                revision,
                filename,
            } => Self::HuggingFace {
                repository,
                revision,
                filename,
            },
            ComponentSourceConfig::Url { url, digest } => Self::Url { url, digest },
        }
    }
}

/// The format of the LaMa weights file. The config representation of
/// `koharu_ml::lama::WeightsFormat`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum WeightsFormatConfig {
    #[default]
    SafeTensors,
    TorchScript,
}

impl From<WeightsFormatConfig> for koharu_ml::lama::WeightsFormat {
    fn from(value: WeightsFormatConfig) -> Self {
        match value {
            WeightsFormatConfig::SafeTensors => Self::SafeTensors,
            WeightsFormatConfig::TorchScript => Self::TorchScript,
        }
    }
}

/// LaMa checkpoint selection. Defaults to the `mayocream/lama-manga`
/// safetensors checkpoint, matching the behavior before this field existed.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct LaMaConfig {
    pub source: ComponentSourceConfig,
    pub format: WeightsFormatConfig,
}

impl LaMaConfig {
    fn validate(&self) -> Result<()> {
        ComponentSource::from(self.source.clone())
            .validate()
            .context("LaMa weights")
    }
}

/// MI-GAN checkpoint selection. A prompt-free erase-only model, so it only
/// carries a source.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct MiGanConfig {
    pub source: ComponentSourceConfig,
}

impl MiGanConfig {
    fn validate(&self) -> Result<()> {
        ComponentSource::from(self.source.clone())
            .validate()
            .context("MI-GAN weights")
    }
}

impl From<MangaInpaintorConfig> for MangaSource {
    fn from(value: MangaInpaintorConfig) -> Self {
        Self {
            inpaintor: value.inpaintor.into(),
            line: value.line.into(),
        }
    }
}

/// Manga inpainter checkpoint selection. The pipeline is assembled from an
/// inpaintor and a line model, mirroring `Flux2KleinSourceConfig`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct MangaInpaintorConfig {
    pub inpaintor: ComponentSourceConfig,
    pub line: ComponentSourceConfig,
}

impl MangaInpaintorConfig {
    fn validate(&self) -> Result<()> {
        MangaSource::from(self.clone()).validate()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct Flux2KleinSourceConfig {
    pub transformer: ComponentSourceConfig,
    pub text_encoder: ComponentSourceConfig,
    pub vae: ComponentSourceConfig,
}

impl From<Flux2KleinSourceConfig> for Flux2KleinSource {
    fn from(value: Flux2KleinSourceConfig) -> Self {
        Self {
            transformer: value.transformer.into(),
            text_encoder: value.text_encoder.into(),
            vae: value.vae.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct Flux2KleinConfig {
    pub prompt: String,
    /// Which checkpoints the context is assembled from. Defaults to the pinned
    /// FLUX.2 Klein 4B repositories.
    pub source: Flux2KleinSourceConfig,
    pub steps: u32,
    pub strength: f64,
    /// `-1` draws a fresh seed for every call.
    #[specta(type = f64)]
    pub seed: i64,
    pub padding_mask_crop: Option<u32>,
    /// The working area every tile is shrunk to before denoising.
    pub max_pixels: u32,
}

impl Default for Flux2KleinConfig {
    fn default() -> Self {
        let defaults = Flux2KleinInpaintOptions::default();
        Self {
            prompt: "Remove the text and reconstruct the background.".to_owned(),
            source: Flux2KleinSourceConfig::default(),
            steps: u32::try_from(defaults.num_inference_steps)
                .expect("the default step count fits in u32"),
            strength: defaults.strength,
            seed: defaults.seed,
            padding_mask_crop: defaults.padding_mask_crop,
            max_pixels: defaults.max_pixels,
        }
    }
}

impl Flux2KleinConfig {
    fn validate(&self) -> Result<()> {
        ensure!(!self.prompt.contains('\0'), "FLUX.2 prompt contains NUL");
        ensure!(self.steps > 0, "FLUX.2 steps must be greater than zero");
        ensure!(
            self.strength > 0.0 && self.strength <= 1.0,
            "FLUX.2 strength must be greater than zero and at most one"
        );
        ensure!(
            self.max_pixels >= 64 * 64,
            "FLUX.2 max pixels must be at least {} (64x64)",
            64 * 64
        );
        Flux2KleinSource::from(self.source.clone()).validate()
    }

    fn options(&self) -> Flux2KleinInpaintOptions {
        Flux2KleinInpaintOptions {
            padding_mask_crop: self.padding_mask_crop,
            strength: self.strength,
            num_inference_steps: self.steps as usize,
            seed: self.seed,
            max_pixels: self.max_pixels,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct RoremMixedConfig {
    pub prompt: String,
    pub negative_prompt: String,
}

impl Default for RoremMixedConfig {
    fn default() -> Self {
        Self {
            prompt: DEFAULT_PROMPT.to_owned(),
            negative_prompt: DEFAULT_NEGATIVE_PROMPT.to_owned(),
        }
    }
}

/// PowerPaint ships no pinned repository: the converted GGUF and its task
/// embeddings are produced locally by `scripts/convert_powerpaint.py`, so both
/// paths are required and there is no default that resolves to anything.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct PowerPaintConfig {
    /// Named `model_path` rather than `model` because `InpaintingModel` is
    /// internally tagged on `model`, and a same-named field would collide with
    /// the tag when the variant is serialized.
    pub model_path: PathBuf,
    pub embeddings_dir: PathBuf,
    pub mask_dilation: u8,
    pub steps: i32,
    pub guidance_scale: f32,
    pub strength: f32,
    pub seed: i64,
}

impl PowerPaintConfig {
    fn validate(&self) -> Result<()> {
        self.paths().validate()?;
        let options = self.options();
        ensure!(
            options.num_inference_steps > 0,
            "PowerPaint steps must be greater than zero"
        );
        ensure!(
            options.strength > 0.0 && options.strength <= 1.0,
            "PowerPaint strength must be greater than zero and at most one"
        );
        ensure!(
            options.guidance_scale.is_finite() && options.guidance_scale > 0.0,
            "PowerPaint guidance scale must be finite and greater than zero"
        );
        Ok(())
    }

    fn paths(&self) -> PowerPaintPaths {
        PowerPaintPaths {
            model: self.model_path.clone(),
            embeddings_dir: self.embeddings_dir.clone(),
        }
    }

    /// A zeroed field means "unset" rather than "zero", because `Default` has to
    /// stay derivable for the `unwrap_or_default` the config loader uses.
    fn options(&self) -> PowerPaintOptions {
        let defaults = PowerPaintOptions::default();
        PowerPaintOptions {
            resolution: defaults.resolution,
            mask_dilation: self.mask_dilation,
            num_inference_steps: if self.steps == 0 {
                defaults.num_inference_steps
            } else {
                self.steps
            },
            guidance_scale: if self.guidance_scale == 0.0 {
                defaults.guidance_scale
            } else {
                self.guidance_scale
            },
            strength: if self.strength == 0.0 {
                defaults.strength
            } else {
                self.strength
            },
            seed: if self.seed == 0 {
                defaults.seed
            } else {
                self.seed
            },
        }
    }
}

pub(super) struct Processor {
    config: InpaintingModel,
    device: koharu_ml::Device,
    resources: Arc<ResourceMonitor>,
    model: ModelCell<Model>,
}

impl Processor {
    pub(super) fn new(
        config: InpaintingModel,
        device: koharu_ml::Device,
        resources: Arc<ResourceMonitor>,
    ) -> Result<Self> {
        match &config {
            InpaintingModel::LaMa(settings) => settings.validate()?,
            InpaintingModel::MiGan(settings) => settings.validate()?,
            InpaintingModel::MangaInpaintor(settings) => settings.validate()?,
            InpaintingModel::AotInpainting {} => {}
            InpaintingModel::Flux2Klein(settings) => settings.validate()?,
            InpaintingModel::RoremMixed(settings) => {
                ensure!(
                    !settings.prompt.contains('\0') && !settings.negative_prompt.contains('\0'),
                    "RORem prompt contains NUL"
                );
            }
            InpaintingModel::PowerPaint(settings) => settings.validate()?,
        }

        Ok(Self {
            config,
            device,
            resources,
            model: ModelCell::new(),
        })
    }

    /// The device description handed to a loader, with `memory_free` filled in
    /// from the live monitor.
    ///
    /// `koharu_ml::Device` comes from `Hardware::discover()`, which is a
    /// `OnceLock` probe that never populates `memory_free`, so FLUX's residency
    /// check below it could only ever see `0`. The monitor is the only live
    /// reading available, and its headroom is `budget - used` of one sample, so
    /// it already accounts for whatever else holds memory in the provider's
    /// scope.
    fn device_for_load(&self) -> koharu_ml::Device {
        let mut device = self.device.clone();
        if let Some(headroom) = self.resources.accelerator_headroom() {
            device.memory_free = usize::try_from(headroom.value).unwrap_or(usize::MAX);
        }
        device
    }
}

#[async_trait]
impl StageProcessor for Processor {
    fn model(&self) -> &'static str {
        match self.config {
            InpaintingModel::LaMa(_) => "lama",
            InpaintingModel::MiGan(_) => "mi-gan",
            InpaintingModel::MangaInpaintor(_) => "manga-inpaintor",
            InpaintingModel::AotInpainting {} => "aot-inpainting",
            InpaintingModel::Flux2Klein(_) => "flux2-klein",
            InpaintingModel::RoremMixed(_) => "rorem-mixed",
            InpaintingModel::PowerPaint(_) => "powerpaint",
        }
    }

    fn skip(&self, input: &StageInput) -> Result<bool> {
        if input.inpainting_mask.is_some() {
            return Ok(false);
        }
        let source = AssetRole::new("source")?;
        for entity in input.scene.children(input.page)? {
            if input
                .scene
                .component::<RasterLayer>(entity)?
                .is_some_and(|layer| layer.kind == RasterLayerKind::Cleanup)
                && input.scene.asset(entity, &source)?.is_some()
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn unload(&self) -> bool {
        self.model.unload()
    }

    async fn load(&self) -> Result<()> {
        self.model
            .ensure(|| Model::load(self.device_for_load(), &self.config))
            .await
    }

    async fn process(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        self.model
            .lock()
            .await
            .as_ref()
            .ok_or_else(|| anyhow!("inpainting model is not loaded"))?
            .run(input)
            .await
    }
}

enum Model {
    LaMa(Arc<Mutex<LaMa>>),
    MiGan(Arc<Mutex<MiGan>>),
    MangaInpaintor(Arc<Mutex<MangaInpaintor>>),
    Aot(Arc<Mutex<AotInpainting>>),
    Flux {
        model: Arc<Mutex<Flux2KleinInpaint>>,
        config: Flux2KleinConfig,
    },
    Rorem {
        model: Arc<Mutex<RoremMixed>>,
        config: RoremMixedConfig,
    },
    PowerPaint {
        model: Arc<Mutex<PowerPaint>>,
        options: PowerPaintOptions,
    },
}

/// FLUX exposes no size API once loaded, so the closest honest figure is the
/// three resolved files on disk. Logged rather than displayed: it bounds the
/// weights from above but says nothing about the compute graph.
async fn log_weight_estimate(source: &Flux2KleinSource) {
    let Ok(paths) = koharu_ml::flux2_klein::resolve_paths(source).await else {
        return;
    };
    let sizes = paths
        .iter()
        .map(|path| crate::file_size(path))
        .collect::<Vec<_>>();
    if sizes.iter().any(Option::is_none) {
        return;
    }
    tracing::debug!(
        estimated_weight_bytes = sizes.iter().flatten().map(|bytes| bytes.value).sum::<u64>(),
        "estimated FLUX.2 Klein weights from the resolved files"
    );
}

impl Model {
    async fn load(device: koharu_ml::Device, config: &InpaintingModel) -> Result<Self> {
        match config {
            InpaintingModel::LaMa(config) => Ok(Self::LaMa(Arc::new(Mutex::new(
                LaMa::load(
                    device,
                    &ComponentSource::from(config.source.clone()),
                    config.format.into(),
                )
                .await?,
            )))),
            InpaintingModel::MiGan(config) => Ok(Self::MiGan(Arc::new(Mutex::new(
                MiGan::load(device, &ComponentSource::from(config.source.clone())).await?,
            )))),
            InpaintingModel::MangaInpaintor(config) => Ok(Self::MangaInpaintor(Arc::new(
                Mutex::new(MangaInpaintor::load(device, &config.clone().into()).await?),
            ))),
            InpaintingModel::AotInpainting {} => Ok(Self::Aot(Arc::new(Mutex::new(
                AotInpainting::load(device).await?,
            )))),
            InpaintingModel::Flux2Klein(config) => {
                let source: Flux2KleinSource = config.source.clone().into();
                log_weight_estimate(&source).await;
                Ok(Self::Flux {
                    model: Arc::new(Mutex::new(Flux2KleinInpaint::load(device, &source).await?)),
                    config: config.clone(),
                })
            }
            InpaintingModel::RoremMixed(config) => Ok(Self::Rorem {
                model: Arc::new(Mutex::new(RoremMixed::load(device).await?)),
                config: config.clone(),
            }),
            InpaintingModel::PowerPaint(config) => Ok(Self::PowerPaint {
                model: Arc::new(Mutex::new(PowerPaint::load(device, &config.paths())?)),
                options: config.options(),
            }),
        }
    }

    async fn run(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        let mut prepared = prepare(&input).await?;
        if prepared.mask.as_raw().iter().all(|value| *value == 0) {
            return finish(input.scene.edit());
        }
        let mask = prepared.mask.clone();
        let original = prepared.original.clone();
        let cleanup = prepared.cleanup.take();
        let cleanup_entity = prepared.cleanup_entity;
        let (model_name, image) = match self {
            Self::LaMa(model) => {
                let model = model.clone();
                (
                    "lama",
                    tokio::task::spawn_blocking(move || -> Result<DynamicImage> {
                        let model = model
                            .lock()
                            .map_err(|_| anyhow!("LaMa model lock is poisoned"))?;
                        inpaint_tiled(
                            &prepared.image,
                            &prepared.mask,
                            &prepared.text_mask,
                            &prepared.flat_fill_regions,
                            |image, mask| {
                                Ok(DynamicImage::ImageRgb8(model.inference(
                                    image,
                                    mask,
                                    &InpaintRequest::default(),
                                )?))
                            },
                        )
                    })
                    .await
                    .context("LaMa task panicked")??,
                )
            }
            Self::MiGan(model) => {
                let model = model.clone();
                (
                    "mi-gan",
                    tokio::task::spawn_blocking(move || -> Result<DynamicImage> {
                        let model = model
                            .lock()
                            .map_err(|_| anyhow!("MI-GAN model lock is poisoned"))?;
                        inpaint_tiled(
                            &prepared.image,
                            &prepared.mask,
                            &prepared.text_mask,
                            &prepared.flat_fill_regions,
                            |image, mask| {
                                Ok(DynamicImage::ImageRgb8(model.inference(
                                    image,
                                    mask,
                                    &InpaintRequest::default(),
                                )?))
                            },
                        )
                    })
                    .await
                    .context("MI-GAN task panicked")??,
                )
            }
            Self::MangaInpaintor(model) => {
                let model = model.clone();
                (
                    "manga-inpaintor",
                    tokio::task::spawn_blocking(move || -> Result<DynamicImage> {
                        let model = model
                            .lock()
                            .map_err(|_| anyhow!("Manga inpainter model lock is poisoned"))?;
                        inpaint_tiled(
                            &prepared.image,
                            &prepared.mask,
                            &prepared.text_mask,
                            &prepared.flat_fill_regions,
                            |image, mask| {
                                Ok(DynamicImage::ImageRgb8(model.inference(
                                    image,
                                    mask,
                                    &InpaintRequest::default(),
                                )?))
                            },
                        )
                    })
                    .await
                    .context("Manga inpainter task panicked")??,
                )
            }
            Self::Aot(model) => {
                let model = model.clone();
                (
                    "aot-inpainting",
                    tokio::task::spawn_blocking(move || -> Result<DynamicImage> {
                        let model = model
                            .lock()
                            .map_err(|_| anyhow!("AOT model lock is poisoned"))?;
                        inpaint_tiled(
                            &prepared.image,
                            &prepared.mask,
                            &prepared.text_mask,
                            &prepared.flat_fill_regions,
                            |image, mask| {
                                Ok(DynamicImage::ImageRgb8(model.inference(image, mask)?))
                            },
                        )
                    })
                    .await
                    .context("AOT task panicked")??,
                )
            }
            Self::Flux { model, config } => {
                let model = model.clone();
                let config = config.clone();
                (
                    "flux2-klein",
                    tokio::task::spawn_blocking(move || -> Result<DynamicImage> {
                        let model = model
                            .lock()
                            .map_err(|_| anyhow!("FLUX model lock is poisoned"))?;
                        inpaint_tiled(
                            &prepared.image,
                            &prepared.mask,
                            &prepared.text_mask,
                            &prepared.flat_fill_regions,
                            |image, mask| {
                                model.inference(
                                    &config.prompt,
                                    image,
                                    None,
                                    &DynamicImage::ImageLuma8(mask.clone()),
                                    &config.options(),
                                )
                            },
                        )
                    })
                    .await
                    .context("FLUX task panicked")??,
                )
            }
            Self::Rorem { model, config } => {
                let model = model.clone();
                let config = config.clone();
                (
                    "rorem-mixed",
                    tokio::task::spawn_blocking(move || -> Result<DynamicImage> {
                        let model = model
                            .lock()
                            .map_err(|_| anyhow!("RORem model lock is poisoned"))?;
                        inpaint_tiled(
                            &prepared.image,
                            &prepared.mask,
                            &prepared.text_mask,
                            &prepared.flat_fill_regions,
                            |image, mask| {
                                Ok(DynamicImage::ImageRgb8(model.inference(
                                    image,
                                    mask,
                                    &config.prompt,
                                    &config.negative_prompt,
                                    &RoremMixedOptions::default(),
                                )?))
                            },
                        )
                    })
                    .await
                    .context("RORem task panicked")??,
                )
            }
            Self::PowerPaint { model, options } => {
                let model = model.clone();
                let options = options.clone();
                (
                    "powerpaint",
                    tokio::task::spawn_blocking(move || -> Result<DynamicImage> {
                        let model = model
                            .lock()
                            .map_err(|_| anyhow!("PowerPaint model lock is poisoned"))?;
                        inpaint_tiled(
                            &prepared.image,
                            &prepared.mask,
                            &prepared.text_mask,
                            &prepared.flat_fill_regions,
                            |image, mask| {
                                Ok(DynamicImage::ImageRgb8(
                                    model.inference(image, mask, &options)?,
                                ))
                            },
                        )
                    })
                    .await
                    .context("PowerPaint task panicked")??,
                )
            }
        };
        let page = input.page;
        let manual = input.inpainting_mask.is_some();
        let mut edit = if manual {
            input.scene.edit()
        } else {
            input.scene.edit_as(generation(PRODUCER, model_name)?)
        };
        edit.observe_assets(page)?;
        if let Some(entity) = cleanup_entity {
            edit.observe::<RasterLayer>(entity)?;
            edit.observe_assets(entity)?;
        }
        let image = image.to_rgba8();
        if image.dimensions() != original.dimensions() || image.dimensions() != mask.dimensions() {
            bail!("inpainted image dimensions do not match page {page}");
        }
        let mut overlay = if manual {
            cleanup.unwrap_or_else(|| RgbaImage::new(image.width(), image.height()))
        } else {
            RgbaImage::new(image.width(), image.height())
        };
        for (x, y, target) in overlay.enumerate_pixels_mut() {
            if mask.get_pixel(x, y)[0] < 127 {
                continue;
            }
            let generated = image.get_pixel(x, y);
            // Keep the complete cleanup mask opaque. Transparent texels next to
            // edited pixels let linear canvas sampling reveal the artwork below.
            *target = Rgba([generated[0], generated[1], generated[2], 255]);
        }
        let mut bytes = Cursor::new(Vec::new());
        let width = overlay.width();
        let height = overlay.height();
        DynamicImage::ImageRgba8(overlay).write_to(&mut bytes, ImageFormat::Png)?;
        let cleanup_entity = if let Some(entity) = cleanup_entity {
            if manual {
                let mut layer = input
                    .scene
                    .component::<RasterLayer>(entity)?
                    .context("cleanup entity has no raster layer component")?;
                let generated = layer.origin != Origin::User
                    || input
                        .scene
                        .component::<EntityOrigin>(entity)?
                        .is_some_and(|origin| origin.origin != Origin::User);
                if generated {
                    edit.promote_entity_to_user(entity)?;
                    layer.origin = Origin::User;
                    edit.set(entity, &layer)?;
                }
            }
            entity
        } else {
            let entity = edit.add_entity(page, At::Start)?;
            edit.set(
                entity,
                &RasterLayer {
                    origin: Origin::User,
                    name: "Cleanup".to_owned(),
                    kind: RasterLayerKind::Cleanup,
                },
            )?;
            entity
        };
        edit.set_asset(
            cleanup_entity,
            &AssetRole::new("source")?,
            AssetInput::new(
                Arc::<[u8]>::from(bytes.into_inner()),
                "image/png",
                AssetMetadata {
                    width: Some(width),
                    height: Some(height),
                    attributes: BTreeMap::new(),
                },
            ),
        )?;
        finish(edit)
    }
}

#[derive(Clone, Debug)]
struct FlatFillRegion {
    bounds: [u32; 4],
    polygon: Vec<(f32, f32)>,
}

fn flat_fill_regions(input: &StageInput, width: u32, height: u32) -> Result<Vec<FlatFillRegion>> {
    let mut regions = Vec::new();
    for entity in input.scene.descendants(input.page)? {
        let id = entity.id();
        let is_bubble = input
            .scene
            .component::<Region>(id)?
            .is_some_and(|region| region.kind == BubbleRegion::kind());
        if !is_bubble {
            continue;
        }
        let Some(geometry) = input.scene.component::<Geometry>(id)? else {
            continue;
        };
        let polygon = geometry
            .points
            .iter()
            .map(|point| (point.x as f32, point.y as f32))
            .collect::<Vec<_>>();
        if polygon.len() < 3 {
            continue;
        }
        let (mut left, mut top) = (f32::INFINITY, f32::INFINITY);
        let (mut right, mut bottom) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for &(x, y) in &polygon {
            left = left.min(x);
            top = top.min(y);
            right = right.max(x);
            bottom = bottom.max(y);
        }
        let bounds = [
            left.floor().clamp(0.0, width as f32) as u32,
            top.floor().clamp(0.0, height as f32) as u32,
            right.ceil().clamp(0.0, width as f32) as u32,
            bottom.ceil().clamp(0.0, height as f32) as u32,
        ];
        if bounds[2] > bounds[0] && bounds[3] > bounds[1] {
            regions.push(FlatFillRegion { bounds, polygon });
        }
    }
    Ok(regions)
}

struct InpaintInput {
    image: Arc<DynamicImage>,
    original: Arc<DynamicImage>,
    cleanup_entity: Option<koharu_scene::EntityId>,
    cleanup: Option<RgbaImage>,
    mask: GrayImage,
    text_mask: GrayImage,
    flat_fill_regions: Vec<FlatFillRegion>,
}

async fn prepare(input: &StageInput) -> Result<InpaintInput> {
    let page = input.page;
    let original = input
        .images
        .get(&input.scene, page, "source")
        .await?
        .ok_or_else(|| anyhow!("page {page} has no source image"))?;
    let cleanup_entity = input.scene.children(page)?.find(|entity| {
        input
            .scene
            .component::<RasterLayer>(*entity)
            .ok()
            .flatten()
            .is_some_and(|layer| layer.kind == RasterLayerKind::Cleanup)
    });
    let cleanup = if let Some(entity) = cleanup_entity {
        input
            .images
            .get(&input.scene, entity, "source")
            .await?
            .map(|image| image.to_rgba8())
    } else {
        None
    };
    if cleanup
        .as_ref()
        .is_some_and(|image| image.dimensions() != original.dimensions())
    {
        bail!("cleanup layer dimensions do not match page {page}");
    }
    let source = if input.inpainting_mask.is_some() {
        if let Some(cleanup) = cleanup.as_ref() {
            let mut composite = original.to_rgba8();
            image::imageops::overlay(&mut composite, cleanup, 0, 0);
            Arc::new(DynamicImage::ImageRgba8(composite))
        } else {
            original.clone()
        }
    } else {
        original.clone()
    };
    let mut mask = GrayImage::new(source.width(), source.height());
    let mut text_mask = GrayImage::new(source.width(), source.height());
    if let Some(transient) = &input.inpainting_mask {
        let layer = image::load_from_memory(&transient.png)?.to_luma8();
        if layer.dimensions() != mask.dimensions() {
            bail!("inpainting mask dimensions do not match page {page}");
        }
        mask = layer;
    } else {
        for role in ["text-mask", "coo-mask"] {
            if let Some(image) = input.images.get(&input.scene, page, role).await? {
                let layer = image.to_luma8();
                if layer.dimensions() != mask.dimensions() {
                    bail!("{role} dimensions do not match page {page}");
                }
                for (target, source) in mask.as_mut().iter_mut().zip(layer.as_raw()) {
                    *target = (*target).max(*source);
                }
                if role == "text-mask" {
                    text_mask = layer;
                }
            }
        }
    }
    if let Some(bounds) = input.region {
        for (x, y, pixel) in mask.enumerate_pixels_mut() {
            if f64::from(x + 1) <= bounds.x
                || f64::from(y + 1) <= bounds.y
                || f64::from(x) >= bounds.x + bounds.width
                || f64::from(y) >= bounds.y + bounds.height
            {
                *pixel = Luma([0]);
                text_mask.put_pixel(x, y, Luma([0]));
            }
        }
    }
    let flat_fill_regions = flat_fill_regions(input, source.width(), source.height())?;
    Ok(InpaintInput {
        image: source,
        original,
        cleanup_entity,
        cleanup,
        mask,
        text_mask,
        flat_fill_regions,
    })
}

const TILE_SIZE: u32 = 512;
const TILE_CONTEXT: u32 = 128;
const UNIFORM_BACKGROUND_MIN_PIXELS: usize = 16;
const FLAT_FILL_EDGE_MARGIN: f32 = 3.0;

#[derive(Clone, Debug, PartialEq, Eq)]
struct InpaintTile {
    components: Vec<u32>,
    core: [u32; 4],
    crop: [u32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MaskComponent {
    label: u32,
    bounds: [u32; 4],
}

// BallonsTranslator avoids model inference for a text block when the non-text
// pixels inside its balloon are nearly uniform, and otherwise sends an enlarged
// block crop to the inpainter:
// https://github.com/dmMaze/BallonsTranslator/blob/4bcc635c19f6c63a902872cf77b3d554e14ed1b7/ballontranslator/modules/inpaint/base.py#L168-L200
// Koharu uses the detected bubble polygons as those blocks. Uniform bubbles are
// filled first; only the remaining mask is split into bounded model crops.
fn inpaint_tiled(
    image: &DynamicImage,
    mask: &GrayImage,
    text_mask: &GrayImage,
    flat_fill_regions: &[FlatFillRegion],
    mut inference: impl FnMut(&DynamicImage, &GrayImage) -> Result<DynamicImage>,
) -> Result<DynamicImage> {
    ensure!(
        image.dimensions() == mask.dimensions() && mask.dimensions() == text_mask.dimensions(),
        "image and mask dimensions differ: image={:?}, mask={:?}, text_mask={:?}",
        image.dimensions(),
        mask.dimensions(),
        text_mask.dimensions()
    );

    let mut output = image.to_rgb8();
    let mut pending_mask = mask.clone();
    fill_uniform_regions(&mut output, &mut pending_mask, text_mask, flat_fill_regions);
    let (component_labels, tiles) = inpaint_tiles(&pending_mask);
    for tile in &tiles {
        let [left, top, right, bottom] = tile.crop;
        let crop_width = right - left;
        let crop_height = bottom - top;
        let crop_image =
            image::imageops::crop_imm(&output, left, top, crop_width, crop_height).to_image();
        let crop_mask = crop_tile_mask(&pending_mask, tile);

        let generated = inference(&DynamicImage::ImageRgb8(crop_image), &crop_mask)?;
        let generated = if generated.dimensions() == (crop_width, crop_height) {
            generated.to_rgb8()
        } else {
            generated
                .resize_exact(
                    crop_width,
                    crop_height,
                    image::imageops::FilterType::Lanczos3,
                )
                .to_rgb8()
        };
        composite_generated(&mut output, &component_labels, tile, &generated);
    }
    Ok(DynamicImage::ImageRgb8(output))
}

fn fill_uniform_regions(
    output: &mut RgbImage,
    pending_mask: &mut GrayImage,
    text_mask: &GrayImage,
    regions: &[FlatFillRegion],
) {
    for region in regions {
        let [left, top, right, bottom] = region.bounds;
        let mut targets = Vec::new();
        for y in top..bottom {
            for x in left..right {
                if pending_mask.get_pixel(x, y)[0] >= 127
                    && text_mask.get_pixel(x, y)[0] >= 127
                    && point_in_polygon((x as f32 + 0.5, y as f32 + 0.5), &region.polygon)
                {
                    targets.push((x, y));
                }
            }
        }
        if targets.is_empty() {
            continue;
        }
        let Some(color) = uniform_region_color(output, pending_mask, region) else {
            continue;
        };
        for (x, y) in targets {
            output.put_pixel(x, y, color);
            pending_mask.put_pixel(x, y, Luma([0]));
        }
    }
}

fn inpaint_tiles(mask: &GrayImage) -> (ImageBuffer<Luma<u32>, Vec<u32>>, Vec<InpaintTile>) {
    let (labels, components) = mask_components(mask);
    let mut bounded_tiles: Vec<InpaintTile> = Vec::new();
    let mut split_tiles = Vec::new();

    for component in components {
        let [left, top, right, bottom] = component.bounds;
        if right - left <= TILE_SIZE && bottom - top <= TILE_SIZE {
            let best = bounded_tiles
                .iter()
                .enumerate()
                .filter_map(|(index, tile)| {
                    let bounds = union_bounds(tile.core, component.bounds);
                    ((bounds[2] - bounds[0] <= TILE_SIZE) && (bounds[3] - bounds[1] <= TILE_SIZE))
                        .then_some((bounds_area(bounds) - bounds_area(tile.core), index, bounds))
                })
                .min_by_key(|(growth, index, _)| (*growth, *index));
            if let Some((_, index, bounds)) = best {
                let tile = &mut bounded_tiles[index];
                tile.components.push(component.label);
                tile.core = bounds;
                tile.crop = expand_bounds(bounds, mask.width(), mask.height());
            } else {
                bounded_tiles.push(InpaintTile {
                    components: vec![component.label],
                    core: component.bounds,
                    crop: expand_bounds(component.bounds, mask.width(), mask.height()),
                });
            }
            continue;
        }

        let mut core_top = top;
        while core_top < bottom {
            let core_bottom = core_top.saturating_add(TILE_SIZE).min(bottom);
            let mut core_left = left;
            while core_left < right {
                let core_right = core_left.saturating_add(TILE_SIZE).min(right);
                let core = [core_left, core_top, core_right, core_bottom];
                if let Some(owned_bounds) = component_bounds_in(&labels, component.label, core) {
                    split_tiles.push(InpaintTile {
                        components: vec![component.label],
                        core,
                        crop: expand_bounds(owned_bounds, mask.width(), mask.height()),
                    });
                }
                core_left = core_right;
            }
            core_top = core_bottom;
        }
    }

    bounded_tiles.append(&mut split_tiles);
    bounded_tiles.sort_by_key(|tile| (tile.core[1], tile.core[0], tile.core[3], tile.core[2]));
    (labels, bounded_tiles)
}

fn mask_components(mask: &GrayImage) -> (ImageBuffer<Luma<u32>, Vec<u32>>, Vec<MaskComponent>) {
    let binary = GrayImage::from_fn(mask.width(), mask.height(), |x, y| {
        Luma([(mask.get_pixel(x, y)[0] >= 127) as u8 * u8::MAX])
    });
    let labels = connected_components(&binary, Connectivity::Eight, Luma([0]));
    let mut bounds = Vec::<Option<[u32; 4]>>::new();
    for (x, y, pixel) in labels.enumerate_pixels() {
        let label = pixel[0] as usize;
        if label == 0 {
            continue;
        }
        if label >= bounds.len() {
            bounds.resize(label + 1, None);
        }
        if let Some(bounds) = &mut bounds[label] {
            bounds[0] = bounds[0].min(x);
            bounds[1] = bounds[1].min(y);
            bounds[2] = bounds[2].max(x + 1);
            bounds[3] = bounds[3].max(y + 1);
        } else {
            bounds[label] = Some([x, y, x + 1, y + 1]);
        }
    }
    let components = bounds
        .into_iter()
        .enumerate()
        .skip(1)
        .filter_map(|(label, bounds)| {
            bounds.map(|bounds| MaskComponent {
                label: label as u32,
                bounds,
            })
        })
        .collect();
    (labels, components)
}

fn union_bounds(left: [u32; 4], right: [u32; 4]) -> [u32; 4] {
    [
        left[0].min(right[0]),
        left[1].min(right[1]),
        left[2].max(right[2]),
        left[3].max(right[3]),
    ]
}

fn bounds_area([left, top, right, bottom]: [u32; 4]) -> u64 {
    u64::from(right - left) * u64::from(bottom - top)
}

fn expand_bounds([left, top, right, bottom]: [u32; 4], width: u32, height: u32) -> [u32; 4] {
    [
        left.saturating_sub(TILE_CONTEXT),
        top.saturating_sub(TILE_CONTEXT),
        right.saturating_add(TILE_CONTEXT).min(width),
        bottom.saturating_add(TILE_CONTEXT).min(height),
    ]
}

fn component_bounds_in(
    labels: &ImageBuffer<Luma<u32>, Vec<u32>>,
    label: u32,
    [region_left, region_top, region_right, region_bottom]: [u32; 4],
) -> Option<[u32; 4]> {
    let mut left = region_right;
    let mut top = region_bottom;
    let mut right = 0;
    let mut bottom = 0;
    for y in region_top..region_bottom {
        for x in region_left..region_right {
            if labels.get_pixel(x, y)[0] == label {
                left = left.min(x);
                top = top.min(y);
                right = right.max(x + 1);
                bottom = bottom.max(y + 1);
            }
        }
    }
    (right > left && bottom > top).then_some([left, top, right, bottom])
}

fn crop_tile_mask(mask: &GrayImage, tile: &InpaintTile) -> GrayImage {
    let [left, top, right, bottom] = tile.crop;
    let mut crop = GrayImage::new(right - left, bottom - top);
    for y in top..bottom {
        for x in left..right {
            if mask.get_pixel(x, y)[0] >= 127 {
                crop.put_pixel(x - left, y - top, Luma([u8::MAX]));
            }
        }
    }
    crop
}

fn uniform_region_color(
    image: &RgbImage,
    mask: &GrayImage,
    region: &FlatFillRegion,
) -> Option<Rgb<u8>> {
    let [left, top, right, bottom] = region.bounds;
    let mut channels = [Vec::new(), Vec::new(), Vec::new()];
    for y in top..bottom {
        for x in left..right {
            let point = (x as f32 + 0.5, y as f32 + 0.5);
            if mask.get_pixel(x, y)[0] >= 127
                || !point_in_polygon(point, &region.polygon)
                || polygon_edge_distance_squared(point, &region.polygon)
                    < FLAT_FILL_EDGE_MARGIN * FLAT_FILL_EDGE_MARGIN
            {
                continue;
            }
            let pixel = image.get_pixel(x, y);
            for channel in 0..3 {
                channels[channel].push(pixel[channel]);
            }
        }
    }
    if channels[0].len() < UNIFORM_BACKGROUND_MIN_PIXELS {
        return None;
    }

    let medians = channels.each_mut().map(|values| median(values));
    let deviations = std::array::from_fn::<_, 3, _>(|channel| {
        standard_deviation(&channels[channel], f64::from(medians[channel]))
    });
    let mean_deviation = deviations.iter().sum::<f64>() / deviations.len() as f64;
    let channel_spread = (deviations
        .iter()
        .map(|deviation| (deviation - mean_deviation).powi(2))
        .sum::<f64>()
        / deviations.len() as f64)
        .sqrt();
    let threshold = if channel_spread > 1.0 { 7.0 } else { 10.0 };
    (deviations.iter().copied().fold(0.0, f64::max) < threshold).then_some(Rgb(medians))
}

fn point_in_polygon(point: (f32, f32), polygon: &[(f32, f32)]) -> bool {
    let mut inside = false;
    let mut previous = polygon[polygon.len() - 1];
    for &current in polygon {
        if (current.1 > point.1) != (previous.1 > point.1) {
            let intersection_x = (previous.0 - current.0) * (point.1 - current.1)
                / (previous.1 - current.1)
                + current.0;
            if point.0 < intersection_x {
                inside = !inside;
            }
        }
        previous = current;
    }
    inside
}

fn polygon_edge_distance_squared(point: (f32, f32), polygon: &[(f32, f32)]) -> f32 {
    let mut minimum = f32::INFINITY;
    let mut start = polygon[polygon.len() - 1];
    for &end in polygon {
        let segment = (end.0 - start.0, end.1 - start.1);
        let length_squared = segment.0 * segment.0 + segment.1 * segment.1;
        let projection = if length_squared > 0.0 {
            ((point.0 - start.0) * segment.0 + (point.1 - start.1) * segment.1) / length_squared
        } else {
            0.0
        }
        .clamp(0.0, 1.0);
        let closest = (
            start.0 + segment.0 * projection,
            start.1 + segment.1 * projection,
        );
        let dx = point.0 - closest.0;
        let dy = point.1 - closest.1;
        minimum = minimum.min(dx * dx + dy * dy);
        start = end;
    }
    minimum
}

fn median(values: &mut [u8]) -> u8 {
    values.sort_unstable();
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        ((u16::from(values[middle - 1]) + u16::from(values[middle])) / 2) as u8
    } else {
        values[middle]
    }
}

fn standard_deviation(values: &[u8], center: f64) -> f64 {
    (values
        .iter()
        .map(|value| (f64::from(*value) - center).powi(2))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt()
}

fn composite_generated(
    output: &mut RgbImage,
    labels: &ImageBuffer<Luma<u32>, Vec<u32>>,
    tile: &InpaintTile,
    generated: &RgbImage,
) {
    let [left, top, _, _] = tile.crop;
    let [core_left, core_top, core_right, core_bottom] = tile.core;
    for y in core_top..core_bottom {
        for x in core_left..core_right {
            if tile
                .components
                .binary_search(&labels.get_pixel(x, y)[0])
                .is_ok()
            {
                output.put_pixel(x, y, *generated.get_pixel(x - left, y - top));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn transient_inpainting_mask_replaces_persistent_page_masks() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut page = None;
        let source = DynamicImage::new_rgb8(8, 8);
        let persistent = DynamicImage::ImageLuma8(GrayImage::from_pixel(8, 8, Luma([255])));
        let encode = |image: &DynamicImage| {
            let mut bytes = Cursor::new(Vec::new());
            image.write_to(&mut bytes, ImageFormat::Png).unwrap();
            Arc::<[u8]>::from(bytes.into_inner())
        };
        let patch = session
            .snapshot()
            .patch(|edit| {
                let id = edit.add_page(
                    koharu_scene::PageDraft::new("page", 8.0, 8.0),
                    koharu_scene::At::End,
                )?;
                for (role, image) in [("source", &source), ("text-mask", &persistent)] {
                    edit.set_asset(
                        id,
                        &AssetRole::new(role)?,
                        AssetInput::new(
                            encode(image),
                            "image/png",
                            AssetMetadata {
                                width: Some(8),
                                height: Some(8),
                                attributes: BTreeMap::new(),
                            },
                        ),
                    )?;
                }
                page = Some(id);
                Ok(())
            })
            .unwrap();
        let snapshot = session.commit(patch).await.unwrap().snapshot;
        let page = page.unwrap();
        let mut transient = GrayImage::new(8, 8);
        transient.put_pixel(3, 4, Luma([255]));
        let input = StageInput::new(
            snapshot,
            page,
            None,
            None,
            Arc::new(crate::ImageCache::default()),
            Some(crate::InpaintingMask {
                page,
                png: encode(&DynamicImage::ImageLuma8(transient)),
            }),
        );

        let prepared = prepare(&input).await.unwrap();
        assert_eq!(prepared.mask.get_pixel(3, 4), &Luma([255]));
        assert_eq!(prepared.mask.get_pixel(0, 0), &Luma([0]));
        assert!(prepared.text_mask.pixels().all(|pixel| pixel[0] == 0));
    }

    #[test]
    fn flux_defaults_reproduce_the_previous_inference_options() {
        let config = Flux2KleinConfig::default();
        assert_eq!(config.options(), Flux2KleinInpaintOptions::default());
        assert_eq!(
            Flux2KleinSource::from(config.source),
            Flux2KleinSource::default()
        );
    }

    #[test]
    fn lama_defaults_reproduce_the_previous_checkpoint() {
        let config = LaMaConfig::default();
        assert_eq!(config.source, ComponentSourceConfig::Builtin);
        assert_eq!(config.format, WeightsFormatConfig::SafeTensors);
    }

    #[test]
    fn mi_gan_defaults_are_builtin() {
        let config = MiGanConfig::default();
        assert_eq!(config.source, ComponentSourceConfig::Builtin);
    }

    #[test]
    fn manga_inpaintor_defaults_are_builtin() {
        let config = MangaInpaintorConfig::default();
        assert_eq!(config.inpaintor, ComponentSourceConfig::Builtin);
        assert_eq!(config.line, ComponentSourceConfig::Builtin);
    }

    #[test]
    fn a_flux_section_written_before_the_settings_existed_still_loads() {
        let config: Flux2KleinConfig =
            toml::from_str("prompt = \"Erase the text.\"").expect("legacy section deserializes");
        assert_eq!(config.prompt, "Erase the text.");
        assert_eq!(config.options(), Flux2KleinInpaintOptions::default());
        config.validate().unwrap();
    }

    #[test]
    fn invalid_flux_settings_are_rejected() {
        for config in [
            Flux2KleinConfig {
                steps: 0,
                ..Default::default()
            },
            Flux2KleinConfig {
                strength: 1.5,
                ..Default::default()
            },
            Flux2KleinConfig {
                max_pixels: 1024,
                ..Default::default()
            },
            Flux2KleinConfig {
                source: Flux2KleinSourceConfig {
                    transformer: ComponentSourceConfig::LocalFile {
                        path: PathBuf::from("relative.gguf"),
                    },
                    ..Default::default()
                },
                ..Default::default()
            },
        ] {
            assert!(config.validate().is_err(), "{config:?} should be rejected");
        }
    }

    #[test]
    fn an_invalid_lama_source_is_rejected() {
        let config = LaMaConfig {
            source: ComponentSourceConfig::LocalFile {
                path: PathBuf::from("relative.pt"),
            },
            ..Default::default()
        };
        assert!(config.validate().is_err(), "{config:?} should be rejected");
    }

    #[test]
    fn an_invalid_mi_gan_source_is_rejected() {
        let config = MiGanConfig {
            source: ComponentSourceConfig::LocalFile {
                path: PathBuf::from("relative.pt"),
            },
        };
        assert!(config.validate().is_err(), "{config:?} should be rejected");
    }

    #[test]
    fn an_invalid_manga_inpaintor_source_is_rejected() {
        let config = MangaInpaintorConfig {
            inpaintor: ComponentSourceConfig::LocalFile {
                path: PathBuf::from("relative.jit"),
            },
            ..Default::default()
        };
        assert!(config.validate().is_err(), "{config:?} should be rejected");
    }

    #[test]
    fn flux_settings_reach_the_inference_options() {
        let config = Flux2KleinConfig {
            steps: 8,
            strength: 0.5,
            seed: 42,
            padding_mask_crop: Some(32),
            max_pixels: 512 * 512,
            ..Default::default()
        };
        config.validate().unwrap();
        assert_eq!(
            config.options(),
            Flux2KleinInpaintOptions {
                padding_mask_crop: Some(32),
                strength: 0.5,
                num_inference_steps: 8,
                seed: 42,
                max_pixels: 512 * 512,
            }
        );
    }

    #[test]
    fn a_hugging_face_override_round_trips_through_toml() {
        let config = Flux2KleinConfig {
            source: Flux2KleinSourceConfig {
                transformer: ComponentSourceConfig::HuggingFace {
                    repository: "unsloth/FLUX.2-klein-4B-GGUF".to_owned(),
                    revision: None,
                    filename: "flux-2-klein-4b-Q8_0.gguf".to_owned(),
                },
                ..Default::default()
            },
            ..Default::default()
        };
        config.validate().unwrap();
        let document = toml::to_string(&config).unwrap();
        assert_eq!(
            toml::from_str::<Flux2KleinConfig>(&document).unwrap(),
            config
        );
    }

    #[tokio::test]
    async fn automatic_inpainting_skips_committed_cleanup() {
        let mut session = koharu_scene::Session::memory().await.unwrap();
        let mut page = None;
        let patch = session
            .snapshot()
            .patch(|edit| {
                let id = edit.add_page(
                    koharu_scene::PageDraft::new("page", 8.0, 8.0),
                    koharu_scene::At::End,
                )?;
                let cleanup = edit.add_entity(id, At::Start)?;
                edit.set(
                    cleanup,
                    &RasterLayer {
                        origin: Origin::User,
                        name: "Cleanup".to_owned(),
                        kind: RasterLayerKind::Cleanup,
                    },
                )?;
                edit.set_asset(
                    cleanup,
                    &AssetRole::new("source")?,
                    AssetInput::new(
                        Arc::<[u8]>::from([0]),
                        "image/png",
                        AssetMetadata {
                            width: Some(8),
                            height: Some(8),
                            attributes: BTreeMap::new(),
                        },
                    ),
                )?;
                page = Some(id);
                Ok(())
            })
            .unwrap();
        let snapshot = session.commit(patch).await.unwrap().snapshot;
        let page = page.unwrap();
        let automatic = StageInput::new(
            snapshot.clone(),
            page,
            None,
            None,
            Arc::new(crate::ImageCache::default()),
            None,
        );
        let manual = StageInput::new(
            snapshot,
            page,
            None,
            None,
            Arc::new(crate::ImageCache::default()),
            Some(crate::InpaintingMask {
                page,
                png: Arc::<[u8]>::from([]),
            }),
        );
        let processor = Processor::new(
            InpaintingModel::LaMa(LaMaConfig::default()),
            koharu_ml::Device::cpu(),
            ResourceMonitor::new(&koharu_ml::Device::cpu()),
        )
        .unwrap();

        assert!(processor.skip(&automatic).unwrap());
        assert!(!processor.skip(&manual).unwrap());
    }

    fn rectangle_region([left, top, right, bottom]: [u32; 4]) -> FlatFillRegion {
        FlatFillRegion {
            bounds: [left, top, right, bottom],
            polygon: vec![
                (left as f32, top as f32),
                (right as f32, top as f32),
                (right as f32, bottom as f32),
                (left as f32, bottom as f32),
            ],
        }
    }

    #[test]
    fn uniform_mask_background_is_filled_without_inference() {
        let mut image = RgbImage::from_pixel(96, 96, Rgb([240, 241, 242]));
        let mut mask = GrayImage::new(96, 96);
        for y in 40..56 {
            for x in 32..64 {
                image.put_pixel(x, y, Rgb([20, 20, 20]));
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
        let mut calls = 0;

        let output = inpaint_tiled(
            &DynamicImage::ImageRgb8(image),
            &mask,
            &mask,
            &[rectangle_region([0, 0, 96, 96])],
            |_, _| {
                calls += 1;
                Ok(DynamicImage::new_rgb8(1, 1))
            },
        )
        .unwrap()
        .to_rgb8();

        assert_eq!(calls, 0);
        assert_eq!(output.get_pixel(48, 48), &Rgb([240, 241, 242]));
    }

    #[test]
    fn uniform_bubbles_are_filled_independently_inside_one_textured_tile() {
        let mut image = RgbImage::new(200, 100);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            *pixel = if (x + y).is_multiple_of(2) {
                Rgb([20, 80, 140])
            } else {
                Rgb([220, 80, 120])
            };
        }
        for y in 10..90 {
            for x in 10..90 {
                image.put_pixel(x, y, Rgb([245, 245, 245]));
            }
            for x in 110..190 {
                image.put_pixel(x, y, Rgb([250, 240, 220]));
            }
        }
        let mut mask = GrayImage::new(200, 100);
        for y in 40..60 {
            for x in 30..60 {
                image.put_pixel(x, y, Rgb([10, 10, 10]));
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
            for x in 135..165 {
                image.put_pixel(x, y, Rgb([10, 10, 10]));
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
        let mut calls = 0;

        let output = inpaint_tiled(
            &DynamicImage::ImageRgb8(image),
            &mask,
            &mask,
            &[
                rectangle_region([10, 10, 90, 90]),
                rectangle_region([110, 10, 190, 90]),
            ],
            |_, _| {
                calls += 1;
                Ok(DynamicImage::new_rgb8(1, 1))
            },
        )
        .unwrap()
        .to_rgb8();

        assert_eq!(calls, 0);
        assert_eq!(output.get_pixel(45, 50), &Rgb([245, 245, 245]));
        assert_eq!(output.get_pixel(150, 50), &Rgb([250, 240, 220]));
    }

    #[test]
    fn connected_mask_crossing_both_page_grid_axes_is_inferred_once() {
        let image = RgbImage::from_pixel(900, 900, Rgb([20, 40, 60]));
        let mut mask = GrayImage::new(900, 900);
        for y in 500..525 {
            for x in 500..525 {
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
        let mut calls = 0;

        let output = inpaint_tiled(
            &DynamicImage::ImageRgb8(image),
            &mask,
            &mask,
            &[],
            |tile, tile_mask| {
                calls += 1;
                assert_eq!(
                    tile_mask.pixels().filter(|pixel| pixel[0] >= 127).count(),
                    25 * 25
                );
                Ok(DynamicImage::ImageRgb8(RgbImage::from_pixel(
                    tile.width(),
                    tile.height(),
                    Rgb([1, 2, 3]),
                )))
            },
        )
        .unwrap()
        .to_rgb8();

        assert_eq!(calls, 1);
        assert_eq!(output.get_pixel(500, 500), &Rgb([1, 2, 3]));
        assert_eq!(output.get_pixel(524, 524), &Rgb([1, 2, 3]));
        assert_eq!(output.get_pixel(499, 499), &Rgb([20, 40, 60]));
    }

    #[test]
    fn oversized_component_tiles_mask_their_continuation_context() {
        let mut mask = GrayImage::new(1000, 300);
        for y in 100..120 {
            for x in 50..850 {
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }

        let (labels, tiles) = inpaint_tiles(&mask);

        assert_eq!(tiles.len(), 2);
        for tile in &tiles {
            assert!(tile.crop[2] - tile.crop[0] <= TILE_SIZE + TILE_CONTEXT * 2);
            assert!(tile.crop[3] - tile.crop[1] <= TILE_SIZE + TILE_CONTEXT * 2);
            let label = tile.components[0];
            let [left, top, right, bottom] = tile.core;
            let mut owned = 0;
            for y in top..bottom {
                for x in left..right {
                    if labels.get_pixel(x, y)[0] == label {
                        owned += 1;
                    }
                }
            }
            let masked = crop_tile_mask(&mask, tile)
                .pixels()
                .filter(|pixel| pixel[0] >= 127)
                .count();
            assert!(masked > owned);
        }
    }

    #[test]
    fn textured_mask_regions_are_inferred_as_bounded_tiles() {
        let mut image = RgbImage::new(1200, 700);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            *pixel = if (x + y).is_multiple_of(2) {
                Rgb([0, 40, 80])
            } else {
                Rgb([255, 215, 175])
            };
        }
        let original = image.clone();
        let mut mask = GrayImage::new(1200, 700);
        mask.put_pixel(10, 10, Luma([u8::MAX]));
        mask.put_pixel(1100, 600, Luma([u8::MAX]));
        let mut calls = 0;

        let output = inpaint_tiled(
            &DynamicImage::ImageRgb8(image),
            &mask,
            &mask,
            &[],
            |tile, _| {
                calls += 1;
                assert!(tile.width() <= TILE_CONTEXT * 2 + 1);
                assert!(tile.height() <= TILE_CONTEXT * 2 + 1);
                Ok(DynamicImage::ImageRgb8(RgbImage::from_pixel(
                    tile.width(),
                    tile.height(),
                    Rgb([1, 2, 3]),
                )))
            },
        )
        .unwrap()
        .to_rgb8();

        assert_eq!(calls, 2);
        assert_eq!(output.get_pixel(10, 10), &Rgb([1, 2, 3]));
        assert_eq!(output.get_pixel(1100, 600), &Rgb([1, 2, 3]));
        assert_eq!(output.get_pixel(500, 300), original.get_pixel(500, 300));
    }
}
