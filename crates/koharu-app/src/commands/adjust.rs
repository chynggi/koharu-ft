use anyhow::{Result, bail};
use image::RgbaImage;
use image::imageops::colorops::{brighten_in_place, contrast_in_place, huerotate_in_place};
use serde::Deserialize;
use specta::Type;

/// A single raster colour adjustment. Command arguments are never persisted
/// (B-D7), so the adjustment survives only for the duration of one command and
/// is internally tagged the way this crate's other command arguments are.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ColorAdjust {
    /// `delta` is added to every colour channel.
    Brightness { delta: i32 },
    /// `percent` is a percentage in `-100.0..=100.0`, *not* a `0.0..=1.0`
    /// fraction: `0.0` is a no-op and `100.0` doubles the deviation from
    /// mid-grey. The field name carries the unit to the wire format.
    Contrast { percent: f32 },
    /// The input range stretched onto the full `0..=255` output range.
    Levels { low: u8, high: u8 },
    /// Hue rotation in degrees.
    Hue { degrees: i32 },
    /// Luminance only; alpha is preserved.
    Grayscale,
}

impl ColorAdjust {
    /// Rejects the parameters the kernels cannot express. The range checks
    /// mirror what the underlying functions accept (`stretch_contrast_mut`
    /// panics on `low >= high`) so a bad request fails as an error rather than
    /// a panic.
    pub fn validate(&self) -> Result<()> {
        match *self {
            Self::Brightness { delta } => {
                if !(-255..=255).contains(&delta) {
                    bail!("brightness delta {delta} is outside -255..=255");
                }
            }
            Self::Contrast { percent } => {
                if !percent.is_finite() || !(-100.0..=100.0).contains(&percent) {
                    bail!("contrast percent {percent} is outside -100.0..=100.0");
                }
            }
            Self::Levels { low, high } => {
                if low >= high {
                    bail!("levels low {low} must be below high {high}");
                }
            }
            Self::Hue { degrees } => {
                if !(-180..=180).contains(&degrees) {
                    bail!("hue degrees {degrees} is outside -180..=180");
                }
            }
            Self::Grayscale => {}
        }
        Ok(())
    }
}

/// Applies `adjust` to `image` in place.
///
/// Every adjustment preserves alpha, and a fully transparent pixel keeps its
/// RGB as well: the colour of such a pixel is invisible, and leaving it alone
/// is what makes the operations idempotent if the alpha is raised later.
///
/// The `image` and `imageproc` kernels walk the whole buffer without a way to
/// skip pixels, so the skip cannot live in the loop. Instead the original is
/// cloned, the kernel runs, and the original state is then restored: the whole
/// pixel where the original alpha was zero, and the alpha channel everywhere
/// else. That keeps the crate functions authoritative for the colour
/// arithmetic while this function owns the alpha invariant.
///
/// The alpha restore is not redundant. `contrast_in_place` maps over *every*
/// channel of the pixel (`image::Pixel::map`, not `map_with_alpha`, as
/// `brighten_in_place` uses), so it re-scales alpha along with the colour and
/// turns alpha 200 into 255 at `percent: 50.0`. Copying alpha back is what
/// makes that kernel usable here at all.
pub fn apply_adjust(image: &mut RgbaImage, adjust: ColorAdjust) {
    let before = image.clone();

    match adjust {
        ColorAdjust::Brightness { delta } => brighten_in_place(image, delta),
        ColorAdjust::Contrast { percent } => contrast_in_place(image, percent),
        ColorAdjust::Levels { low, high } => {
            imageproc::contrast::stretch_contrast_mut(image, low, high, 0, 255)
        }
        ColorAdjust::Hue { degrees } => huerotate_in_place(image, degrees),
        ColorAdjust::Grayscale => grayscale_preserving_alpha(image),
    }

    for (pixel, original) in image.pixels_mut().zip(before.pixels()) {
        if original.0[3] == 0 {
            *pixel = *original;
        } else {
            pixel.0[3] = original.0[3];
        }
    }
}

/// Greys the colour channels while preserving alpha.
///
/// `image::imageops::grayscale` returns a `Luma` image and discards alpha,
/// which would turn a raster layer's transparent regions opaque black and hide
/// the artwork below (B-D8). The coefficients are the sRGB luminance weights
/// from Rec. 709, matching the perception-weighted luma that `image` itself
/// uses, evaluated in `f32` so the rounding is not biased by a fixed-point
/// approximation.
fn grayscale_preserving_alpha(image: &mut RgbaImage) {
    for pixel in image.pixels_mut() {
        let [red, green, blue, alpha] = pixel.0;
        let luma = 0.2126 * f32::from(red) + 0.7152 * f32::from(green) + 0.0722 * f32::from(blue);
        let luma = luma.round().clamp(0.0, 255.0) as u8;
        *pixel = image::Rgba([luma, luma, luma, alpha]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    fn pixel(r: u8, g: u8, b: u8, a: u8) -> Rgba<u8> {
        Rgba([r, g, b, a])
    }

    fn image(pixels: &[[u8; 4]]) -> RgbaImage {
        let mut image = RgbaImage::new(pixels.len() as u32, 1);
        for (index, value) in pixels.iter().enumerate() {
            image.put_pixel(index as u32, 0, Rgba(*value));
        }
        image
    }

    #[test]
    fn brightness_raises_the_channels() {
        let mut value = image(&[[100, 100, 100, 255]]);
        apply_adjust(&mut value, ColorAdjust::Brightness { delta: 20 });
        assert_eq!(*value.get_pixel(0, 0), pixel(120, 120, 120, 255));
    }

    #[test]
    fn brightness_clamps_at_the_channel_maximum() {
        let mut value = image(&[[250, 250, 250, 255]]);
        apply_adjust(&mut value, ColorAdjust::Brightness { delta: 20 });
        assert_eq!(*value.get_pixel(0, 0), pixel(255, 255, 255, 255));
    }

    /// `Contrast` takes a percentage, not a -1.0..=1.0 fraction. A value of
    /// 100 doubles the deviation from mid-grey squared; a value of 0 must be a
    /// no-op.
    #[test]
    fn contrast_is_a_percentage_and_zero_changes_nothing() {
        let original = image(&[[100, 150, 200, 255]]);
        let mut unchanged = original.clone();
        apply_adjust(&mut unchanged, ColorAdjust::Contrast { percent: 0.0 });
        assert_eq!(unchanged, original);

        let mut raised = original.clone();
        apply_adjust(&mut raised, ColorAdjust::Contrast { percent: 50.0 });
        assert_ne!(
            raised, original,
            "a non-zero percentage must change the image"
        );
    }

    /// `contrast_in_place` maps over every channel of the pixel
    /// (`image::Pixel::map`), so unlike `brighten_in_place` it re-scales alpha
    /// too. Alpha must survive every variant, not just greyscale (B-D8).
    #[test]
    fn every_adjustment_preserves_the_alpha_of_a_partly_transparent_pixel() {
        let adjusts = [
            ColorAdjust::Brightness { delta: 20 },
            ColorAdjust::Contrast { percent: 50.0 },
            ColorAdjust::Levels {
                low: 100,
                high: 150,
            },
            ColorAdjust::Hue { degrees: 90 },
            ColorAdjust::Grayscale,
        ];
        for adjust in adjusts {
            let mut value = image(&[[100, 150, 200, 200]]);
            apply_adjust(&mut value, adjust);
            assert_eq!(
                value.get_pixel(0, 0)[3],
                200,
                "alpha must survive {adjust:?}"
            );
        }
    }

    /// `ColorAdjust` is a command argument, so it is internally tagged the way
    /// this crate's other command arguments are (`Layer` in `project.rs`).
    /// Task 3's TypeScript type mirrors this shape.
    #[test]
    fn the_wire_shape_is_internally_tagged() {
        let brightness: ColorAdjust =
            serde_json::from_str(r#"{"type":"brightness","delta":20}"#).unwrap();
        assert_eq!(brightness, ColorAdjust::Brightness { delta: 20 });

        let hue: ColorAdjust = serde_json::from_str(r#"{"type":"hue","degrees":-90}"#).unwrap();
        assert_eq!(hue, ColorAdjust::Hue { degrees: -90 });

        let grayscale: ColorAdjust = serde_json::from_str(r#"{"type":"grayscale"}"#).unwrap();
        assert_eq!(grayscale, ColorAdjust::Grayscale);
    }

    #[test]
    fn levels_stretch_the_input_range_onto_the_full_range() {
        let mut value = image(&[[100, 100, 100, 255], [150, 150, 150, 255]]);
        apply_adjust(
            &mut value,
            ColorAdjust::Levels {
                low: 100,
                high: 150,
            },
        );
        assert_eq!(*value.get_pixel(0, 0), pixel(0, 0, 0, 255));
        assert_eq!(*value.get_pixel(1, 0), pixel(255, 255, 255, 255));
    }

    #[test]
    fn hue_rotation_at_zero_changes_nothing() {
        let original = image(&[[200, 40, 40, 255]]);
        let mut value = original.clone();
        apply_adjust(&mut value, ColorAdjust::Hue { degrees: 0 });
        assert_eq!(value, original);
    }

    #[test]
    fn hue_rotation_moves_the_channels() {
        let original = image(&[[200, 40, 40, 255]]);
        let mut value = original.clone();
        apply_adjust(&mut value, ColorAdjust::Hue { degrees: 90 });
        assert_ne!(value, original);
    }

    /// `image::imageops::grayscale` returns a `Luma` image and discards alpha,
    /// which would turn this layer's transparent area opaque black and hide the
    /// artwork below. This task must grey the channels and keep alpha.
    #[test]
    fn grayscale_keeps_alpha_and_greys_the_channels() {
        let mut value = image(&[[200, 100, 50, 128], [200, 100, 50, 0]]);
        apply_adjust(&mut value, ColorAdjust::Grayscale);
        let opaque = *value.get_pixel(0, 0);
        assert_eq!(opaque[0], opaque[1], "channels must be equal after greying");
        assert_eq!(opaque[1], opaque[2]);
        assert_eq!(opaque[3], 128, "alpha must survive greyscale");
        assert_eq!(
            value.get_pixel(1, 0)[3],
            0,
            "a transparent pixel stays transparent"
        );
    }

    /// A fully transparent pixel's colour is invisible, and preserving it keeps
    /// the layer's RGB meaningful if alpha is later raised. Adjusting it would
    /// also make the operation non-idempotent in a surprising way.
    #[test]
    fn a_fully_transparent_pixel_keeps_its_color() {
        let mut value = image(&[[10, 20, 30, 0]]);
        apply_adjust(&mut value, ColorAdjust::Brightness { delta: 50 });
        assert_eq!(*value.get_pixel(0, 0), pixel(10, 20, 30, 0));
    }

    #[test]
    fn validate_rejects_an_inverted_levels_range() {
        let adjust = ColorAdjust::Levels {
            low: 200,
            high: 100,
        };
        assert!(adjust.validate().is_err());
    }

    #[test]
    fn validate_rejects_equal_levels_bounds() {
        let adjust = ColorAdjust::Levels {
            low: 128,
            high: 128,
        };
        assert!(adjust.validate().is_err());
    }

    #[test]
    fn validate_rejects_an_out_of_range_contrast() {
        assert!(ColorAdjust::Contrast { percent: 101.0 }.validate().is_err());
        assert!(
            ColorAdjust::Contrast { percent: -101.0 }
                .validate()
                .is_err()
        );
        assert!(
            ColorAdjust::Contrast { percent: f32::NAN }
                .validate()
                .is_err()
        );
        assert!(ColorAdjust::Contrast { percent: 100.0 }.validate().is_ok());
    }

    #[test]
    fn validate_rejects_an_out_of_range_brightness_and_hue() {
        assert!(ColorAdjust::Brightness { delta: 256 }.validate().is_err());
        assert!(ColorAdjust::Brightness { delta: -256 }.validate().is_err());
        assert!(ColorAdjust::Hue { degrees: 181 }.validate().is_err());
        assert!(ColorAdjust::Hue { degrees: -181 }.validate().is_err());
        assert!(ColorAdjust::Brightness { delta: 0 }.validate().is_ok());
        assert!(ColorAdjust::Hue { degrees: 180 }.validate().is_ok());
    }
}
