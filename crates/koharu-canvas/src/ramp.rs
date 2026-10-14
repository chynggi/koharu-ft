//! The brush-ramp arithmetic the WASM preview uses, on its own so the host can
//! test it.
//!
//! `web.rs` is compiled only for `wasm32` (it needs `web-sys`/`wgpu`), so an
//! inline `mod tests` there would never run under `cargo test -p koharu-canvas
//! --lib` on the host. The preview's ramp position still has to be pinned
//! against the commit path's rule, so the arithmetic lives here and `web.rs`
//! calls it.

use koharu_rasterizer::{BrushRamp, RampAxis};

/// The ramp position for one previewed segment, given the segment's end point.
///
/// Split out from the WASM boundary so the preview's arithmetic can be tested
/// against the commit path's. The two MUST agree: a mismatch means the live
/// preview shows a different gradient than the stroke that lands.
///
/// `cumulative` is the distance travelled to `point` and `total` is the
/// denominator for `AlongStroke`; the caller owns what that denominator means
/// (see the `AlongStroke` divergence note at the `extend_stroke` call site).
/// `FixedAxis` ignores both and normalizes against `size`, the page size,
/// because page-normalized coordinates are what let several strokes continue
/// one ramp.
pub(crate) fn segment_ramp_position(
    ramp: &BrushRamp,
    x: f64,
    y: f64,
    cumulative: f64,
    total: f64,
    size: (u32, u32),
) -> f32 {
    let position = match ramp {
        BrushRamp::AlongStroke { .. } => {
            if total > 0.0 {
                cumulative / total
            } else {
                // A zero-length stroke (a single point, or the dot drawn by
                // `begin_stroke`) has no progress to measure, so it takes the
                // ramp's `from` end rather than dividing by zero.
                0.0
            }
        }
        BrushRamp::FixedAxis { axis, .. } => match axis {
            RampAxis::Horizontal => normalized(x, size.0),
            RampAxis::Vertical => normalized(y, size.1),
        },
    };
    position as f32
}

/// `coordinate / extent`, with an empty page treated as progress `0` instead of
/// producing an infinity that `color_at` would have to clamp.
fn normalized(coordinate: f64, extent: u32) -> f64 {
    if extent == 0 {
        0.0
    } else {
        coordinate / f64::from(extent)
    }
}

/// The preview's `AlongStroke` denominator: the page diagonal.
///
/// The live preview cannot know a stroke's total length, so `AlongStroke` uses
/// this fixed scale for the whole stroke instead of rescaling as the stroke
/// grows; the commit path uses the real total and remains authoritative. See
/// the divergence note in `web.rs::extend_stroke`.
pub(crate) fn page_diagonal(size: (u32, u32)) -> f64 {
    f64::from(size.0).hypot(f64::from(size.1))
}

/// The ramp position for one previewed segment, including the choice of
/// denominator.
///
/// `extend_stroke` must call **this** rather than picking a denominator itself.
/// The denominator is the preview's one deliberate divergence from the commit
/// path, so it needs to be reachable from a host test: `web.rs` is compiled only
/// for `wasm32`, which means anything left in it is invisible to `cargo test`.
/// A test that supplied the denominator as an argument could not tell this
/// function apart from one that used an arbitrary constant.
pub(crate) fn preview_ramp_position(
    ramp: &BrushRamp,
    point: (f64, f64),
    cumulative: f64,
    size: (u32, u32),
) -> f32 {
    segment_ramp_position(
        ramp,
        point.0,
        point.1,
        cumulative,
        page_diagonal(size),
        size,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_preview_ramp_matches_the_commit_arithmetic() {
        let ramp = BrushRamp::AlongStroke {
            from: [0, 0, 0, 255],
            to: [255, 255, 255, 255],
        };
        // Halfway along a 100px stroke.
        let position = segment_ramp_position(&ramp, 50.0, 0.0, 50.0, 100.0, (100, 100));
        assert!((position - 0.5).abs() < 1e-6, "got {position}");
        assert_eq!(ramp.color_at(position), [128, 128, 128, 255]);
    }

    #[test]
    fn a_zero_length_preview_stroke_does_not_divide_by_zero() {
        let ramp = BrushRamp::AlongStroke {
            from: [10, 10, 10, 255],
            to: [200, 200, 200, 255],
        };
        let position = segment_ramp_position(&ramp, 5.0, 5.0, 0.0, 0.0, (10, 10));
        assert!(position.is_finite(), "got {position}");
        assert_eq!(position, 0.0);
    }

    #[test]
    fn a_fixed_axis_preview_uses_page_size() {
        let ramp = BrushRamp::FixedAxis {
            from: [0, 0, 0, 255],
            to: [255, 255, 255, 255],
            axis: RampAxis::Horizontal,
        };
        let position = segment_ramp_position(&ramp, 25.0, 0.0, 0.0, 0.0, (100, 50));
        assert!((position - 0.25).abs() < 1e-6, "got {position}");
    }

    /// `FixedAxis` is page-normalized on the axis it names and ignores the
    /// other coordinate, matching `rasterize_stroke`.
    #[test]
    fn a_vertical_fixed_axis_preview_uses_page_height() {
        let ramp = BrushRamp::FixedAxis {
            from: [0, 0, 0, 255],
            to: [255, 255, 255, 255],
            axis: RampAxis::Vertical,
        };
        // Same x, different y: only y may move the ramp.
        let low = segment_ramp_position(&ramp, 90.0, 10.0, 0.0, 0.0, (100, 50));
        let high = segment_ramp_position(&ramp, 10.0, 40.0, 0.0, 0.0, (100, 50));
        assert!((low - 0.2).abs() < 1e-6, "got {low}");
        assert!((high - 0.8).abs() < 1e-6, "got {high}");
    }

    /// An empty page would make the fixed-axis normalization `x / 0`. The
    /// preview must stay finite rather than feed an infinity to `color_at`.
    #[test]
    fn an_empty_page_does_not_divide_by_zero() {
        let ramp = BrushRamp::FixedAxis {
            from: [0, 0, 0, 255],
            to: [255, 255, 255, 255],
            axis: RampAxis::Horizontal,
        };
        let position = segment_ramp_position(&ramp, 5.0, 5.0, 0.0, 0.0, (0, 0));
        assert!(position.is_finite(), "got {position}");
        assert_eq!(position, 0.0);
    }

    /// The `AlongStroke` rule is a plain cumulative ratio, so the preview and
    /// the commit path agree wherever they are given the same fraction. The
    /// commit path evaluates `(cumulative + t * segment) / total` per pixel and
    /// the preview evaluates the same expression at the segment's end (`t = 1`).
    #[test]
    fn the_along_stroke_rule_is_a_plain_cumulative_ratio() {
        let ramp = BrushRamp::AlongStroke {
            from: [0, 0, 0, 255],
            to: [100, 100, 100, 255],
        };
        let position = segment_ramp_position(&ramp, 0.0, 0.0, 25.0, 100.0, (100, 100));
        assert!((position - 0.25).abs() < 1e-6, "got {position}");
        // The preview's position must carry the same colour the commit path
        // would compute for that fraction.
        assert_eq!(ramp.color_at(position), ramp.color_at(0.25));
        assert_eq!(
            ramp.color_at(segment_ramp_position(
                &ramp,
                0.0,
                0.0,
                100.0,
                100.0,
                (100, 100)
            )),
            [100, 100, 100, 255]
        );
    }

    /// The preview's `AlongStroke` denominator is the page diagonal, not the
    /// length drawn so far. Rescaling as the stroke grows would recolor already
    /// drawn segments, so a stroke's position must depend only on how far the
    /// pointer has travelled and the page size — never on when it was sampled.
    #[test]
    fn the_preview_along_stroke_denominator_is_the_page_diagonal() {
        let ramp = BrushRamp::AlongStroke {
            from: [0, 0, 0, 255],
            to: [255, 255, 255, 255],
        };
        let size = (30, 40);
        // 3-4-5: the diagonal is exactly 50, so half of it is position 0.5.
        assert_eq!(page_diagonal(size), 50.0);
        // Exercise `preview_ramp_position` — the function `extend_stroke` calls —
        // rather than supplying the denominator here. A test that passed the
        // diagonal itself would still pass if the production function used an
        // arbitrary constant, which is exactly the regression this guards.
        let halfway = preview_ramp_position(&ramp, (0.0, 0.0), 25.0, size);
        assert!((halfway - 0.5).abs() < 1e-6, "got {halfway}");

        // The same cumulative distance must give the same position no matter
        // which earlier samples produced it, and the position keeps growing
        // monotonically with cumulative distance.
        let earlier = preview_ramp_position(&ramp, (0.0, 0.0), 10.0, size);
        let later = preview_ramp_position(&ramp, (0.0, 0.0), 40.0, size);
        assert!(earlier < later, "the ramp must not run backwards");
        // Past the diagonal the ramp saturates rather than wrapping.
        let beyond = preview_ramp_position(&ramp, (0.0, 0.0), 200.0, size);
        assert_eq!(ramp.color_at(beyond), ramp.color_at(1.0));

        // A page whose diagonal is not a round number must still scale with the
        // page: 100x100 has diagonal 141.42, not 100, so half of 100 units is
        // well below 0.5. This is what distinguishes the diagonal from the
        // page width, a constant, or the length so far.
        let square = (100, 100);
        let quarterish = preview_ramp_position(&ramp, (0.0, 0.0), 50.0, square);
        assert!(
            (quarterish - (50.0 / (10_000.0f64 + 10_000.0).sqrt()) as f32).abs() < 1e-6,
            "the denominator must be the diagonal, got {quarterish}"
        );
        assert!(
            quarterish < 0.4,
            "50 units is far less than half the diagonal"
        );
    }

    /// The preview must not scale its ramp by the flat stroke colour's alpha as
    /// well as the ramp's: the segment colours already carry the ramp alpha, so
    /// doing both would composite alpha twice and show a lighter stroke than the
    /// commit produces.
    #[test]
    fn the_preview_does_not_apply_a_second_alpha_when_a_ramp_is_active() {
        let ramp = BrushRamp::AlongStroke {
            from: [0, 0, 0, 128],
            to: [255, 255, 255, 128],
        };
        let position = preview_ramp_position(&ramp, (0.0, 0.0), 0.0, (10, 10));
        // The position is independent of any colour alpha; the segment colour
        // already carries the ramp's 128. The layer opacity must therefore be
        // 1.0 while a ramp is active, which is asserted where the opacity is
        // computed rather than here.
        assert_eq!(ramp.color_at(position)[3], 128);
    }
}
