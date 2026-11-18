//! What each kind of layer holds: Photoshop's layer kinds, and the parts of a
//! manga page as Clip Studio Paint keeps its frames and balloons.

use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize, de::DeserializeOwned, de::Error as _};
use serde_json::Value;
use specta::Type;

use crate::{
    FillRule, Hash, Layer, MaskMode, Offset, Path, Rgba, Stroke, Text, Transform, layer::part,
};

/// A folder of layers, drawn first to last.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Type, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Group {}

/// An imported file, kept byte for byte.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Image {
    pub file: Hash,
    #[serde(default)]
    pub transform: Transform,
}

/// Editable RGBA pixels.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Type, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Raster {
    #[serde(default)]
    pub offset: Offset,
}

/// A vector path with an optional fill and stroke, in page coordinates.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Shape {
    pub path: Path,
    #[serde(default)]
    pub fill_rule: FillRule,
    #[serde(default)]
    pub fill: Option<ShapeFill>,
    #[serde(default)]
    pub stroke: Option<Stroke>,
}

/// A shape's paint. A solid uses the scene's numeric RGBA representation.
/// Gradient and pattern coordinates are normalized to the shape's path
/// bounds: `(0, 0)` is its top-left and `(1, 1)` its bottom-right, so moving
/// or reshaping the path keeps the paint on it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type, JsonSchema)]
#[serde(untagged)]
pub enum ShapeFill {
    Solid(Rgba),
    Gradient { gradient: Gradient },
    Pattern { pattern: Pattern },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Gradient {
    pub kind: GradientKind,
    pub stops: Vec<GradientStop>,
    #[serde(default)]
    pub extend: PaintExtend,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum GradientKind {
    /// Along the line from `start` to `end`.
    Linear { start: [f64; 2], end: [f64; 2] },
    /// Out from `center`; `radius` is a fraction of the bounds' width.
    Radial { center: [f64; 2], radius: f64 },
    /// Around `center`, clockwise from `start_degrees` to `end_degrees`.
    Sweep {
        center: [f64; 2],
        start_degrees: f64,
        end_degrees: f64,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GradientStop {
    pub offset: f32,
    pub color: Rgba,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, Type, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PaintExtend {
    #[default]
    Pad,
    Repeat,
    Reflect,
}

/// An imported image repeated across the shape: a tile's top-left sits at
/// `origin`, it is `scale` of the bounds' width wide, and it turns clockwise
/// by `rotation_degrees` about its top-left.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Pattern {
    pub file: Hash,
    pub origin: [f64; 2],
    pub scale: f64,
    pub rotation_degrees: f64,
    #[serde(default)]
    pub extend: PaintExtend,
}

impl ShapeFill {
    /// Why the renderer couldn't draw this paint, if it couldn't.
    pub(crate) fn invalid(&self) -> Option<&'static str> {
        let finite = |values: &[f64]| values.iter().all(|value| value.is_finite());
        match self {
            Self::Solid(_) => None,
            Self::Gradient { gradient } => {
                if !(2..=32).contains(&gradient.stops.len()) {
                    return Some("a gradient has 2 to 32 stops");
                }
                if gradient
                    .stops
                    .iter()
                    .any(|stop| !(0.0..=1.0).contains(&stop.offset))
                {
                    return Some("a gradient stop's offset is from 0 to 1");
                }
                match gradient.kind {
                    GradientKind::Linear { start, end }
                        if !finite(&[start, end].concat()) || start == end =>
                    {
                        Some("a linear gradient runs between two distinct points")
                    }
                    GradientKind::Radial { center, radius }
                        if !finite(&center) || !(radius.is_finite() && radius > 0.0) =>
                    {
                        Some("a radial gradient has a center and a positive radius")
                    }
                    GradientKind::Sweep {
                        center,
                        start_degrees,
                        end_degrees,
                    } if !finite(&[center[0], center[1], start_degrees, end_degrees]) => {
                        Some("a sweep gradient has a center and two angles")
                    }
                    _ => None,
                }
            }
            Self::Pattern { pattern } => (!finite(&[
                pattern.origin[0],
                pattern.origin[1],
                pattern.rotation_degrees,
            ]) || !(pattern.scale.is_finite()
                && pattern.scale > 0.0))
                .then_some("a pattern has an origin, a positive scale, and a rotation"),
        }
    }
}

/// Grey pixels that limit its parent's visibility, as Photoshop's and Clip
/// Studio Paint's layer masks do; never drawn. Masks apply in order.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Type, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Mask {
    #[serde(default)]
    pub mode: MaskMode,
    #[serde(default)]
    pub offset: Offset,
}

/// A panel of the page: what's read in it, drawn where it is, past the
/// outline too. The outline is never drawn, as a Clip Studio Paint frame
/// border folder's with Draw border off.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Panel {
    /// A closed path in page coordinates.
    pub outline: Path,
}

/// A balloon of the page, holding its texts. Its outline is never drawn or
/// filled, because the page's background already shows it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Balloon {
    /// A closed path in page coordinates.
    pub outline: Path,
}

/// What a block's text is, as a PAGE-XML text region's `type` gives the
/// nature of its text.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize, Type, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    #[default]
    Text,
    Sfx,
}

/// One text of the page, in a fixed shape: its source and its translation,
/// both text values, as an XLIFF segment's source and target share one
/// content model, and two layers of its own, its `mask` and its `cleanup`, as
/// Photoshop's Generative Fill puts its result on a layer with a layer mask.
/// A patch on the block reaches its parts; painting them takes their own ids.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Block {
    #[serde(default)]
    pub role: Role,
    /// The original text, never drawn, in a box or a closed path, as a
    /// PAGE-XML text region records it.
    pub source: Text,
    /// As an XLIFF segment holds zero or one target.
    #[serde(default)]
    pub translation: Option<Text>,
    /// Where the source's glyphs are: the clean-up shows only inside it while
    /// it's visible.
    #[serde(default = "part::<Mask>", deserialize_with = "strict")]
    #[specta(type = Layer<Mask>)]
    pub mask: Layer<Mask>,
    /// The scan redrawn without the source's glyphs, under the translation.
    #[serde(default = "part::<Raster>", deserialize_with = "strict")]
    #[specta(type = Layer<Raster>)]
    pub cleanup: Layer<Raster>,
}

/// A block's part, refusing a field it doesn't have. A flattened struct takes
/// only the fields it knows, unlike a layer's tagged kind, so serde alone
/// would drop a misspelt one.
fn strict<'de, D, C>(deserializer: D) -> Result<Layer<C>, D::Error>
where
    D: Deserializer<'de>,
    C: Serialize + DeserializeOwned,
{
    let given = Value::deserialize(deserializer)?;
    let part: Layer<C> = serde_json::from_value(given.clone()).map_err(D::Error::custom)?;
    let read = serde_json::to_value(&part).map_err(D::Error::custom)?;
    if let (Value::Object(given), Value::Object(read)) = (&given, &read)
        && let Some(key) = given.keys().find(|key| !read.contains_key(*key))
    {
        return Err(D::Error::custom(format!("unknown field `{key}`")));
    }
    Ok(part)
}

/// A layer's kind and what that kind holds. In JSON its fields sit beside the
/// layer's own, tagged by `kind`.
#[derive(
    Clone, Debug, PartialEq, Serialize, Deserialize, Type, JsonSchema, strum::EnumDiscriminants,
)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[strum_discriminants(
    name(LayerKind),
    doc = "What kind a layer is, without its values.",
    derive(strum::AsRefStr, Serialize, Deserialize),
    strum(serialize_all = "snake_case"),
    serde(rename_all = "snake_case")
)]
pub enum Content {
    Group(Group),
    Image(Image),
    Raster(Raster),
    Shape(Shape),
    Text(Box<Text>),
    Mask(Mask),
    Panel(Panel),
    Balloon(Balloon),
    Block(Box<Block>),
}

impl Content {
    /// The imported file it draws: an image's, or a shape's pattern.
    pub fn file(&self) -> Option<Hash> {
        match self {
            Self::Image(image) => Some(image.file),
            Self::Shape(Shape {
                fill: Some(ShapeFill::Pattern { pattern }),
                ..
            }) => Some(pattern.file),
            _ => None,
        }
    }
}
