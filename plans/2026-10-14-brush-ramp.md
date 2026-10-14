# 브러시 그라디언트 (B1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 붓이 단색 대신 두 색 사이의 램프를 칠하게 한다. 스트로크를 따라 변하는 램프(잉크 번짐)와 페이지의 고정 축을 따라 변하는 램프(배경 칠하기) 둘 다.

**Architecture:** `koharu-rasterizer`에 `BrushRamp`(두 색 + 매개변수화 규칙)를 두고, 커밋 경로(`rasterize_stroke`)가 픽셀마다 램프에서 색을 얻는다. 라이브 미리보기는 `draw_segment`가 **세그먼트마다** 색을 계산한다 — 스트로크 램프는 누적 길이 비율, 고정축 램프는 그 점의 페이지 좌표다. vello의 `scene.stroke`가 임의 브러시를 받으므로 1b에서 쓴 peniko 선형 그라디언트를 그대로 쓸 수 있다.

**Tech Stack:** Rust (koharu-rasterizer, koharu-canvas, koharu-app, koharu-rpc), vello/peniko/kurbo, TypeScript (Next.js, zustand, i18next), `cargo test`, vitest.

**Spec:** `specs/2026-10-14-paint-fill-adjust-design.md` (B1 절, B-D2, B-D8)

**이 계획의 범위:** B1만이다. B2(채우기)와 B3(색보정)는 완료되었고, 1c(패턴)는 별도다.

---

## Global Constraints

- **`BrushRamp`는 `koharu-scene`의 `GradientSpec`을 재사용하지 않는다**(B-D2). 두 색과 매개변수화 규칙만 필요하고, `GradientSpec`의 정규화 프레임 좌표는 스트로크에 의미가 없다. 억지로 쓰면 `kind`·`stops`·`extend` 중 아무것도 쓰지 않는 필드를 저장하게 되고, "이 좌표가 무엇을 뜻하는가"에 답이 없어진다.
- **두 램프 방식 모두 구현한다**(B-D1). `AlongStroke`(스트로크 진행 방향)와 `FixedAxis`(페이지 고정 축, 가로/세로)는 서로 대체하지 못한다.
- **`FixedAxis`는 페이지 정규화 좌표를 쓴다** — 프레임이 아니라 페이지다. 이것이 "여러 획이 하나의 램프를 잇는다"를 만드는 유일한 방법이고, 램프의 목적이다.
- **알파를 보존한다**(B-D8). 램프 색의 알파는 스트로크 합성에 그대로 참여해야 한다. 램프가 있다고 해서 알파 처리가 달라지지 않는다.
- **커밋은 한 번**이고 램프 파라미터는 **저장하지 않는다**(B-D7과 같은 원칙 — 붓은 상태이지 문서 데이터가 아니다).
- **미리보기와 커밋이 같은 색을 내야 한다.** 1b에서 프론트와 서버의 회전 규칙이 어긋나 실제 버그가 났다. 여기서는 **두 램프의 산술이 같은 함수에서 계산되도록** 하고, 같은 입력에 대해 같은 색을 내는지 리터럴 테스트로 양쪽을 고정한다.
- 하위 호환·별칭 금지 (AGENTS.md). 주석은 불변식을 설명한다.
- `-j 6`을 모든 cargo 명령에 붙인다 (12코어, 무제한 병렬은 데스크톱을 멈춘다).
- 검증은 가장 작은 관련 검사만 debug 프로필로 한 번 돌린다 (AGENTS.md).

## Review Focus

스펙이 명시하지 않았지만 실제로 깨질 수 있는 지점이다. 각 항목의 테스트는 그 코드를 소유한 태스크에 넣는다.

1. **`AlongStroke`의 누적 길이가 세그먼트 로컬 `t`와 다르다.** `rasterize_stroke`의 기존 `t`는 **세그먼트별 투영 매개변수**(`project.rs:2338-2342`)이고, 스트로크 램프가 필요한 것은 **전체 누적 길이 비율**이다. 둘을 혼동하면 램프가 세그먼트마다 리셋되어 획이 조각난 것처럼 보인다. 특히 **한 점짜리 스트로크**(길이 0)에서 0으로 나누지 않아야 한다(Task 2).
2. **`t`가 역방향 세그먼트에서 뒤집히지 않는지.** 뒤로 그은 획에서 램프가 되감기면 안 된다 — 누적 길이는 단조 증가여야 한다(Task 2).
3. **미리보기와 커밋의 램프 산술이 같은지.** 미리보기는 세그먼트마다 한 색을 칠하므로(균일 색), 커밋은 픽셀마다 색을 계산한다(연속 램프). **두 방식은 다르다** — 미리보기는 계단식으로 보일 수 있다. 이것을 결함으로 만들지 않으려면 미리보기의 세그먼트가 충분히 촘촘해야 한다. 판단하고 테스트로 고정한다(Task 3, Task 4).
4. **`FixedAxis`의 정규화 기준이 페이지 크기인지.** 프레임이나 뷰포트를 쓰면 확대/축소에 따라 색이 달라진다(Task 2).
5. **램프의 두 색이 같을 때.** 단색과 같은 결과여야 하고, 특별 취급이 필요 없다(Task 1).
6. **기존 단색 경로가 그대로인지.** `ramp: None`일 때 `color`가 정확히 예전처럼 동작해야 한다 — B3·B2가 `apply_raster_stroke`를 공유 자산 경로로 리팩터링했으므로 회귀 위험이 있다(Task 5).

---

## File Structure

**수정**

| 파일 | 변경 |
|---|---|
| `crates/koharu-rasterizer/src/prepared.rs` | `BrushRamp`, `RampAxis` 타입 |
| `crates/koharu-rasterizer/src/lib.rs` | 재수출 |
| `crates/koharu-app/src/commands/project.rs` | `rasterize_stroke`의 램프, `apply_raster_stroke` 전달 |
| `crates/koharu-app/src/commands/canvas.rs` | **`PaintBrush` 정의가 여기 있다**(`:32`). `ramp: Option<BrushRamp>` 추가 + `commit_raster_stroke`가 램프를 넘김 |
| `crates/koharu-rpc/src/routes/canvas.rs` | 요청 타입에 램프 |
| `crates/koharu-canvas/src/web.rs` | `begin_stroke`가 램프를 받고 `extend_stroke`가 세그먼트 색을 계산 |
| `crates/koharu-canvas/src/ramp.rs` | **신규.** 세그먼트 램프 위치 계산을 `web.rs`에서 떼어낸 순수 함수 + 테스트. `web.rs`는 `wasm32`에서만 컴파일되므로 호스트 `--lib` 테스트가 닿지 않는다 — 이 분리가 없으면 미리보기 산술을 테스트할 수 없다 |
| `crates/koharu-canvas/src/lib.rs` | `ramp` 모듈 등록 (`cfg(any(wasm32, test))`) |
| `packages/bridge/src/protocol.ts` | `PaintBrush`에 `ramp`, `BrushRamp` 타입 |
| `packages/koharu/lib/store.ts` | `CanvasBrush.ramp` |
| `packages/koharu/components/editor/ToolBar.tsx` | 램프 종류 선택 + 두 색 |
| `packages/koharu/public/locales/*/translation.json` | 9개 로케일 |

---

### Task 1: `BrushRamp` 타입

램프를 `koharu-rasterizer`에 둔다. `koharu-canvas`와 `koharu-app`이 둘 다 그 크레이트에 의존하므로 미리보기와 커밋이 **같은 타입과 같은 산술**을 쓴다 — 1b에서 미리보기와 커밋이 어긋난 버그를 막는 구조적 장치다.

**Files:**
- Modify: `crates/koharu-rasterizer/src/prepared.rs`
- Modify: `crates/koharu-rasterizer/src/lib.rs`
- Test: `crates/koharu-rasterizer/src/prepared.rs` (인라인 `mod tests`)

**Interfaces:**
- Produces:
  - `BrushRamp::{AlongStroke { from: [u8; 4], to: [u8; 4] }, FixedAxis { from: [u8; 4], to: [u8; 4], axis: RampAxis }}`
  - `RampAxis::{Horizontal, Vertical}`

  - `BrushRamp::color_at(&self, position: f32) -> [u8; 4]` — `position`은 `0.0..=1.0`으로 클램프된다

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`prepared.rs`의 `mod tests`에 추가한다. 기존 테스트의 스타일(`PreparedGradient` 검증 테스트)을 따른다.

```rust
    #[test]
    fn a_ramp_interpolates_between_its_ends() {
        let ramp = BrushRamp::AlongStroke {
            from: [0, 0, 0, 255],
            to: [255, 255, 255, 255],
        };
        assert_eq!(ramp.color_at(0.0), [0, 0, 0, 255]);
        assert_eq!(ramp.color_at(1.0), [255, 255, 255, 255]);
        // Midpoint must be the rounded halfway value, not one end.
        assert_eq!(ramp.color_at(0.5), [128, 128, 128, 255]);
    }

    #[test]
    fn a_ramp_interpolates_alpha_too() {
        let ramp = BrushRamp::AlongStroke {
            from: [10, 20, 30, 0],
            to: [10, 20, 30, 255],
        };
        assert_eq!(ramp.color_at(0.0)[3], 0);
        assert_eq!(ramp.color_at(1.0)[3], 255);
        // Alpha must move; a ramp that only interpolates RGB would return 0 here.
        assert!(ramp.color_at(0.5)[3] > 0, "alpha must interpolate");
    }

    #[test]
    fn a_ramp_clamps_outside_the_unit_interval() {
        let ramp = BrushRamp::AlongStroke {
            from: [0, 0, 0, 255],
            to: [100, 100, 100, 255],
        };
        assert_eq!(ramp.color_at(-1.0), [0, 0, 0, 255]);
        assert_eq!(ramp.color_at(2.0), [100, 100, 100, 255]);
    }

    #[test]
    fn a_ramp_with_equal_ends_is_a_solid_colour() {
        let ramp = BrushRamp::FixedAxis {
            from: [7, 8, 9, 255],
            to: [7, 8, 9, 255],
            axis: RampAxis::Horizontal,
        };
        for position in [0.0, 0.25, 0.5, 1.0] {
            assert_eq!(ramp.color_at(position), [7, 8, 9, 255]);
        }
    }

    #[test]
    fn a_non_finite_position_does_not_panic() {
        let ramp = BrushRamp::AlongStroke {
            from: [0, 0, 0, 255],
            to: [255, 255, 255, 255],
        };
        // NaN must not produce a panic or an out-of-range channel; clamping has
        // to happen before the cast.
        let color = ramp.color_at(f32::NAN);
        assert_eq!(color.len(), 4);
    }

    #[test]
    fn the_ramp_types_are_reachable_from_the_crate_root() {
        // The 1b review found gradient types that could not be named outside the
        // crate. Pin the re-export so the same gap cannot reappear.
        assert_eq!(
            koharu_rasterizer::BrushRamp::AlongStroke {
                from: [0, 0, 0, 0],
                to: [0, 0, 0, 0],
            }
            .color_at(0.0),
            [0, 0, 0, 0]
        );
    }
```

- [ ] **Step 2: 실패를 확인한다**

Run: `cargo test -j 6 -p koharu-rasterizer --lib brush_ramp`
Expected: 컴파일 실패 (`BrushRamp` 미정의)

- [ ] **Step 3: 구현한다**

`prepared.rs`의 그라디언트 타입들 옆에 둔다. 파생은 기존 타입과 맞춘다(`Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Type`).

```rust
pub enum BrushRamp {
    /// 진행 방향: 스트로크 시작이 `from`, 끝이 `to`.
    AlongStroke { from: [u8; 4], to: [u8; 4] },
    /// 페이지의 고정 축: `axis`의 0.0 쪽이 `from`, 1.0 쪽이 `to`.
    /// 페이지 정규화 좌표를 쓰므로 여러 획이 하나의 램프를 잇는다.
    FixedAxis { from: [u8; 4], to: [u8; 4], axis: RampAxis },
}

pub enum RampAxis {
    Horizontal,
    Vertical,
}
```

`#[serde(tag = "kind", rename_all = "snake_case")]`을 붙인다. **커맨드 인자이므로 내부 태그다** — `ColorAdjust`(`adjust.rs:11`)와 같고, 영속 컴포넌트인 `ShapeFill`의 외부 태그와 다르다. `RampAxis`는 `#[serde(rename_all = "snake_case")]`만.

`color_at`:

```rust
pub fn color_at(&self, position: f32) -> [u8; 4] {
    let (from, to) = match self {
        Self::AlongStroke { from, to } | Self::FixedAxis { from, to, .. } => (from, to),
    };
    // Clamp before any arithmetic: a NaN position would otherwise survive a
    // comparison-based clamp (every comparison with NaN is false) and then
    // produce a garbage channel via `as u8`.
    let t = if position.is_nan() {
        0.0
    } else {
        position.clamp(0.0, 1.0)
    };
    let mut out = [0u8; 4];
    for (index, channel) in out.iter_mut().enumerate() {
        let start = f32::from(from[index]);
        let end = f32::from(to[index]);
        *channel = (start + (end - start) * t).round().clamp(0.0, 255.0) as u8;
    }
    out
}
```

**알파를 포함한 네 채널을 모두 보간한다** — `from[3]`/`to[3]`가 스트로크 합성의 알파이므로(B-D8) 빠뜨리면 램프가 투명도를 무시한다. 그 이유를 주석에 남긴다.

`validate`는 **거부할 것이 없다**: 두 색은 항상 `[u8; 4]`이고 `axis`는 열거형이다. 정규화가 필요한 값이 없으므로 **`validate`를 만들지 않는다.** 검증할 것이 없는데 함수를 두면 "무엇인가 검사한다"는 잘못된 신호를 준다.

`lib.rs`의 재수출에 `BrushRamp`, `RampAxis`를 추가한다.

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

Run: `cargo test -j 6 -p koharu-rasterizer --lib`
Expected: 기존 + 6 passed

- [ ] **Step 5: 커밋**

```bash
git add crates/koharu-rasterizer/src/prepared.rs crates/koharu-rasterizer/src/lib.rs
git commit -m "feat(rasterizer): add the brush ramp type"
```

---

### Task 2: 커밋 경로의 램프

`rasterize_stroke`가 램프를 받아 픽셀마다 색을 계산한다.

**Files:**
- Modify: `crates/koharu-app/src/commands/project.rs`
- Modify: `crates/koharu-app/src/commands/canvas.rs` (`PaintBrush` 정의 + `commit_raster_stroke` 호출부)
- Test: `crates/koharu-app/src/commands/project.rs` (인라인 `mod tests`)

**Interfaces:**
- Consumes: Task 1의 `BrushRamp`, `RampAxis`
- Produces: `rasterize_stroke(image, mode, color, ramp: Option<BrushRamp>, diameter, points)`
- Produces: `PaintBrush { diameter, color, ramp: Option<BrushRamp> }`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`rasterize_stroke`는 순수 함수이므로 프로젝트 없이 테스트한다. 기존 테스트가 그렇게 하는지 확인하고 맞춘다.

```rust
    #[test]
    fn an_along_stroke_ramp_varies_over_the_whole_stroke() {
        // Two well-separated segments, so cumulative length differs per pixel.
        let mut image = RgbaImage::new(64, 8);
        let points = vec![
            ScenePoint { x: 4.0, y: 4.0 },
            ScenePoint { x: 60.0, y: 4.0 },
        ];
        rasterize_stroke(
            &mut image,
            RasterStrokeMode::Paint,
            [0, 0, 0, 255],
            Some(BrushRamp::AlongStroke {
                from: [255, 0, 0, 255],
                to: [0, 0, 255, 255],
            }),
            4.0,
            &points,
        );
        let start = image.get_pixel(4, 4);
        let end = image.get_pixel(58, 4);
        assert!(
            start[0] > start[2],
            "the stroke start must be red-heavy, got {start:?}"
        );
        assert!(
            end[2] > end[0],
            "the stroke end must be blue-heavy, got {end:?}"
        );
    }

    /// A ramp must advance with *cumulative* length, not restart per segment.
    /// Two segments of equal length should give the same colour at equal
    /// cumulative fractions, and the second must not return to the first's
    /// colour.
    #[test]
    fn an_along_stroke_ramp_does_not_restart_at_each_segment() {
        let mut image = RgbaImage::new(64, 8);
        let points = vec![
            ScenePoint { x: 4.0, y: 4.0 },
            ScenePoint { x: 32.0, y: 4.0 },
            ScenePoint { x: 60.0, y: 4.0 },
        ];
        rasterize_stroke(
            &mut image,
            RasterStrokeMode::Paint,
            [0, 0, 0, 255],
            Some(BrushRamp::AlongStroke {
                from: [255, 0, 0, 255],
                to: [0, 0, 255, 255],
            }),
            4.0,
            &points,
        );
        let first = image.get_pixel(6, 4);
        let second = image.get_pixel(34, 4);
        // The second segment's start is past the halfway point, so it must be
        // bluer than the first segment's start. A per-segment restart would make
        // them nearly identical.
        assert!(
            second[2] > first[2],
            "the ramp must not restart per segment: first {first:?}, second {second:?}"
        );
    }

    #[test]
    fn a_fixed_axis_ramp_uses_page_coordinates() {
        // Horizontal: colour depends on x, and two points at the same x but
        // different y must match.
        let mut image = RgbaImage::new(64, 8);
        let ramp = BrushRamp::FixedAxis {
            from: [255, 0, 0, 255],
            to: [0, 0, 255, 255],
            axis: RampAxis::Horizontal,
        };
        rasterize_stroke(
            &mut image,
            RasterStrokeMode::Paint,
            [0, 0, 0, 255],
            Some(ramp),
            4.0,
            &[
                ScenePoint { x: 4.0, y: 2.0 },
                ScenePoint { x: 4.0, y: 6.0 },
                ScenePoint { x: 60.0, y: 6.0 },
            ],
        );
        let left = image.get_pixel(4, 2);
        let right = image.get_pixel(60, 6);
        assert!(left[0] > left[2], "the left edge must be red-heavy");
        assert!(right[2] > right[0], "the right edge must be blue-heavy");
    }

    #[test]
    fn a_vertical_fixed_axis_ramp_ignores_x() {
        let mut image = RgbaImage::new(8, 64);
        let ramp = BrushRamp::FixedAxis {
            from: [255, 0, 0, 255],
            to: [0, 0, 255, 255],
            axis: RampAxis::Vertical,
        };
        rasterize_stroke(
            &mut image,
            RasterStrokeMode::Paint,
            [0, 0, 0, 255],
            Some(ramp),
            4.0,
            &[ScenePoint { x: 4.0, y: 4.0 }, ScenePoint { x: 4.0, y: 60.0 }],
        );
        assert!(image.get_pixel(4, 4)[0] > image.get_pixel(4, 4)[2]);
        assert!(image.get_pixel(4, 60)[2] > image.get_pixel(4, 60)[0]);
    }

    /// The two ramp kinds must be computed by the same arithmetic the preview
    /// uses, so a single-point stroke cannot divide by zero and a reversed
    /// stroke must not run its ramp backwards.
    #[test]
    fn a_ramp_survives_degenerate_strokes() {
        let mut image = RgbaImage::new(16, 16);
        let ramp = BrushRamp::AlongStroke {
            from: [255, 0, 0, 255],
            to: [0, 0, 255, 255],
        };
        // A single point has zero length.
        rasterize_stroke(
            &mut image,
            RasterStrokeMode::Paint,
            [0, 0, 0, 255],
            Some(ramp),
            6.0,
            &[ScenePoint { x: 8.0, y: 8.0 }],
        );
        let single = image.get_pixel(8, 8);
        assert!(
            single[0] > single[2],
            "a zero-length stroke takes the ramp's start colour, got {single:?}"
        );

        // A stroke drawn right-to-left must ramp from its own start, not invert.
        let mut reversed = RgbaImage::new(64, 8);
        rasterize_stroke(
            &mut reversed,
            RasterStrokeMode::Paint,
            [0, 0, 0, 255],
            Some(ramp),
            4.0,
            &[
                ScenePoint { x: 58.0, y: 4.0 },
                ScenePoint { x: 6.0, y: 4.0 },
            ],
        );
        let start = reversed.get_pixel(58, 4);
        let end = reversed.get_pixel(8, 4);
        assert!(
            start[0] > start[2],
            "the ramp follows the stroke direction, so its start is red: {start:?}"
        );
        assert!(end[2] > end[0], "and its end is blue: {end:?}");
    }

    #[test]
    fn no_ramp_keeps_the_plain_stroke_colour() {
        let mut image = RgbaImage::new(32, 8);
        rasterize_stroke(
            &mut image,
            RasterStrokeMode::Paint,
            [12, 34, 56, 255],
            None,
            4.0,
            &[ScenePoint { x: 4.0, y: 4.0 }, ScenePoint { x: 28.0, y: 4.0 }],
        );
        // Every painted pixel must carry the flat colour; a ramp must not leak in.
        for pixel in image.pixels().filter(|pixel| pixel[3] > 0) {
            assert_eq!(
                [pixel[0], pixel[1], pixel[2]],
                [12, 34, 56],
                "without a ramp the colour is uniform"
            );
        }
    }
```

- [ ] **Step 2: 실패를 확인한다**

Run: `cargo test -j 6 -p koharu-app --lib raster`
Expected: 컴파일 실패 (인자 수 불일치)

- [ ] **Step 3: 구현한다**

`rasterize_stroke`의 시그니처에 `ramp: Option<BrushRamp>`를 `color` **뒤에** 넣는다. 기존 호출부(`apply_raster_stroke`, `rasterize_stroke` 테스트)를 모두 갱신한다.

**누적 길이를 미리 계산한다.** 각 점에 대해 `cumulative[i]` = 점 `0..=i`까지의 거리 합, `total = cumulative[last]`. 루프 안에서는:

```rust
let along = if total > 0.0 {
    (cumulative[index] + t * segment_length) / total
} else {
    0.0
};
```

`position`을 정한다:
- `AlongStroke` → `along`
- `FixedAxis { axis: Horizontal }` → `px / f64::from(image.width())`
- `FixedAxis { axis: Vertical }` → `py / f64::from(image.height())`

`py`/`px`는 **픽셀 중심 + 0.5**이므로 페이지 크기가 0이 아닌 한 `0..1` 안에 들어간다. 페이지 크기가 0이면 루프가 돌지 않는다.

그 다음 `let paint = ramp.map_or(color, |ramp| ramp.color_at(position as f32));`로 색을 정하고, **기존 합성 코드가 `color` 대신 `paint`를 쓰도록** 바꾼다. `Erase` 모드는 색을 쓰지 않으므로 램프를 무시한다 — 그 사실을 주석에 남긴다.

**`cumulative`를 계산할 때 점 개수가 1이면 `total`이 0이다.** 위 `if total > 0.0` 분기가 그것을 처리하지만, 그 이유를 주석에 남긴다.

`PaintBrush`는 **`project.rs`가 아니라 `canvas.rs:32`에 있다.** 거기에 `ramp: Option<BrushRamp>`를 추가하고, `commit_paint` → `apply_raster_stroke` → `rasterize_stroke`로 전달한다. `apply_raster_stroke`의 시그니처가 바뀌므로 **`canvas.rs`의 호출부(`commit_raster_stroke`)도 함께 고쳐야 한다** — `project.rs`만 커밋하면 HEAD가 컴파일되지 않는다. `commit_erase`는 색을 쓰지 않으므로 `None`을 넘긴다.

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

Run: `cargo test -j 6 -p koharu-app --lib`
Expected: 기존 + 6 passed

- [ ] **Step 5: 커밋**

```bash
git add crates/koharu-app/src/commands/project.rs crates/koharu-app/src/commands/canvas.rs
git commit -m "feat(app): paint a stroke with a colour ramp"
```

---

### Task 3: WASM 미리보기

라이브 미리보기가 세그먼트마다 램프 색을 계산한다. **미리보기와 커밋이 같은 색을 내야 한다** — 1b에서 이 어긋남이 실제 버그였다.

**Files:**
- Modify: `crates/koharu-canvas/src/web.rs`
- Test: `crates/koharu-canvas/src/web.rs` (인라인 `mod tests` — 있으면. 없으면 순수 함수를 분리해 테스트한다)

**Interfaces:**
- Consumes: Task 1의 `BrushRamp`, `RampAxis`
- Produces: `begin_stroke(kind, layer, point, diameter, color, ramp)`; `StrokeEdit`에 `ramp`과 누적 길이 추적

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`web.rs`는 WASM 경계라 테스트 인프라가 없을 수 있다. **먼저 확인한다.** 없으면 램프 위치 계산을 **순수 함수로 분리**하고 그것을 테스트한다:

```rust
/// The ramp position for one previewed segment.
///
/// Split out from the WASM boundary so the preview's arithmetic can be tested
/// against the commit path's. The two MUST agree: a mismatch means the live
/// preview shows a different gradient than the stroke that lands.
fn segment_ramp_position(
    ramp: &BrushRamp,
    point: Point,
    cumulative: f64,
    total: f64,
    size: (u32, u32),
) -> f32 { ... }
```

테스트:

```rust
    #[test]
    fn the_preview_ramp_matches_the_commit_arithmetic() {
        let ramp = BrushRamp::AlongStroke {
            from: [0, 0, 0, 255],
            to: [255, 255, 255, 255],
        };
        // Halfway along a 100px stroke.
        let position = segment_ramp_position(
            &ramp,
            Point { x: 50.0, y: 0.0 },
            50.0,
            100.0,
            (100, 100),
        );
        assert!((position - 0.5).abs() < 1e-6, "got {position}");
        assert_eq!(ramp.color_at(position), [128, 128, 128, 255]);
    }

    #[test]
    fn a_zero_length_preview_stroke_does_not_divide_by_zero() {
        let ramp = BrushRamp::AlongStroke {
            from: [10, 10, 10, 255],
            to: [200, 200, 200, 255],
        };
        let position =
            segment_ramp_position(&ramp, Point { x: 5.0, y: 5.0 }, 0.0, 0.0, (10, 10));
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
        let position = segment_ramp_position(
            &ramp,
            Point { x: 25.0, y: 0.0 },
            0.0,
            0.0,
            (100, 50),
        );
        assert!((position - 0.25).abs() < 1e-6, "got {position}");
    }
```

- [ ] **Step 2: 실패를 확인한다**

Run: `cargo test -j 6 -p koharu-canvas --lib`
Expected: 컴파일 실패 또는 실패

- [ ] **Step 3: 구현한다**

1. **`begin_stroke`는 위치 인자 다섯 개를 받고 `color`는 `&Uint8Array`이다**(`web.rs:1551-1558`). 램프는 열거형이라 `Uint8Array`로 표현할 수 없으므로 **`ramp: JsValue`를 추가**하고 기존 `point` 인자가 하는 것처럼 `serde_wasm_bindgen::from_value`로 푼다(`:1559`). 프론트 쪽은 `beginStroke`가 `CanvasStroke` 인터페이스를 받고(`canvas.ts:9`, `:96`) 그 객체를 위치 인자로 풀어 넘기므로(`:223-230`), **`CanvasStroke`에 `ramp?: BrushRamp | null`을 추가**하는 것으로 끝난다.
2. `StrokeEdit`에 `ramp: Option<BrushRamp>`와 `cumulative: f64`(지금까지의 총 길이)를 추가한다.
3. `extend_stroke`에서 세그먼트를 그리기 전에 `cumulative += (previous - point).hypot()`하고, `total`은 **알 수 없다** — 스트로크가 진행 중이기 때문이다. **여기서 결정이 필요하다(Task 3의 핵심 판단).**

   `AlongStroke`의 `total`은 스트로크가 끝나야 확정된다. 진행 중 미리보기에서 세 가지 선택이 있다:
   - **(a)** 지금까지의 `cumulative`를 `total`로 쓴다 → 미리보기가 매번 리스케일되어 색이 계속 변한다. 나쁘다.
   - **(b)** `total`을 추정하지 않고, 미리보기는 **지금까지 그린 구간을 0..1로 정규화**한다 → 역시 리스케일된다.
   - **(c)** 미리보기는 `AlongStroke`에서 **픽셀 단위 길이를 고정 스케일로 나눈다**(예: 페이지 대각선 길이) → 미리보기와 커밋이 어긋난다.

   **결정: (d) 미리보기의 `AlongStroke`는 "지금까지의 누적 길이"를 페이지 대각선 길이로 나눈 값을 쓴다.** 커밋은 실제 총 길이를 쓰므로 **미리보기와 커밋이 어긋난다** — 이것은 피할 수 없다(미리보기는 미래를 모른다). 어긋남을 줄이는 대신 **미리보기가 램프의 방향과 대략적 변화만 보여주면 충분하다**고 정한다. 그 이유를 주석에 남기고, **커밋 결과가 권위**임을 명시한다.

   **이 결정을 보고에 남긴다.** 다른 선택을 원하면 근거와 함께 제안한다.

4. `draw_segment`가 색 대신 `BrushRamp`을 받도록 하거나, 호출부에서 색을 계산해 넘긴다. **`draw_segment`는 색만 받게 두고 호출부에서 `color_at`을 부르는 편이 낫다** — 함수가 하나의 책임을 유지한다.

**미리보기의 세그먼트 색은 세그먼트마다 하나다.** 커밋은 픽셀마다 연속 램프를 그리므로 미리보기가 계단식으로 보일 수 있다. 세그먼트가 촘촘하면(포인터 이벤트마다) 눈에 띄지 않는다. 그 판단을 주석에 남긴다.

- [ ] **Step 4: 확인한다**

Run: `cargo check -j 6 -p koharu-canvas`
Run: `cargo test -j 6 -p koharu-canvas --lib`
Expected: 성공

- [ ] **Step 5: 커밋**

```bash
git add crates/koharu-canvas/src/web.rs crates/koharu-canvas/src/ramp.rs crates/koharu-canvas/src/lib.rs
git commit -m "feat(canvas): preview a ramped stroke"
```

---

### Task 4: 전송 계층과 프론트 UI

**Files:**
- Modify: `crates/koharu-app/src/commands/canvas.rs`
- Modify: `crates/koharu-rpc/src/routes/canvas.rs`
- Modify: `packages/bridge/src/protocol.ts`
- Modify: `packages/koharu/lib/store.ts`
- Modify: `packages/koharu/components/editor/ToolBar.tsx`
- Modify: `packages/koharu/public/locales/*/translation.json` (9개)
- Modify: `packages/koharu/tests/components/editor-components.test.tsx`

**Interfaces:**
- Consumes: Task 2의 `PaintBrush.ramp`, Task 3의 `begin_stroke` 램프 인자
- Produces: `CanvasBrush.ramp`, 툴바 램프 컨트롤

- [ ] **Step 1: `protocol.ts`를 먼저 갱신하고 커밋한다**

```ts
export type RampAxis = "horizontal" | "vertical";

/**
 * A two-colour ramp for the brush. Command arguments, so this is internally
 * tagged like `ColorAdjust` rather than externally tagged like the persisted
 * scene components.
 */
export type BrushRamp =
export type BrushRamp =
	| {
			kind: "along_stroke",
			from: [number, number, number, number],
			to: [number, number, number, number],
	  }
	| {
			kind: "fixed_axis",
			from: [number, number, number, number],
			to: [number, number, number, number],
			axis: RampAxis,
	  };
```

`PaintBrush`에 `ramp: BrushRamp | null`을 추가한다. `commitPaint`는 `brush`를 통째로 보내므로 **시그니처 변경이 없다**.

**`Rgba`라는 이름의 타입은 이 저장소에 없다.** 색은 기존 관례대로 인라인 튜플로 쓴다 — `PaintBrush`가 이미 `color: [number, number, number, number]`이다.

- [ ] **Step 2: 타입 검사**

Run: `bunx tsc --noEmit -p packages/koharu/tsconfig.json`
Expected: exit 0

- [ ] **Step 3: 커밋**

```bash
git add packages/bridge/src/protocol.ts
git commit -m "feat(bridge): declare the brush ramp wire contract"
```

- [ ] **Step 4: Rust 커맨드가 램프를 통과시키는지 확인한다**

`canvas.rs`의 `PaintBrush`는 `koharu-app`의 `PaintBrush`를 재사용하므로 Task 2에서 이미 확장되었다. `routes/canvas.rs`의 `CommitPaintRequest`가 `brush: PaintBrush`를 받는지 확인한다. **별도 작업이 없으면 그 사실을 보고에 남긴다.**

Run: `cargo check -j 6 -p koharu-rpc -p koharu-app`

- [ ] **Step 5: 프론트 상태와 UI**

`store.ts`의 `CanvasBrush`에 `ramp: BrushRamp | null`을 추가하고 기본값 `null`을 준다. **`setBrush`가 램프를 보존하는지 확인한다** — 기존 코드가 붓 객체를 통째로 교체하면 램프가 유지된다.

`ToolBar.tsx`의 브러시 팝오버에 램프 종류 선택(`None`/`Along stroke`/`Fixed axis`)과, 램프가 있을 때 **두 색**(`from`/`to`)과 축 선택(고정축일 때)을 넣는다. 기존 `ColorWell`을 쓴다.

**램프가 켜지면 단색 `ColorWell`을 숨긴다** — 1b에서 확립한 규칙과 같다(두 권위를 만들지 않는다). 그 이유를 주석에 남긴다.

`CanvasWorkspace.tsx`의 `beginStroke` 호출(`:843` 근처의 `CanvasStroke` 구성)에 램프를 넘긴다. **커밋 페이로드도 함께 고친다** — `commitPaint` 호출이 지금 `{ diameter, color }`만 보내므로(`:706-709`) `ramp`을 추가해야 한다. 이걸 빠뜨리면 미리보기만 램프이고 커밋은 단색이 되어 **1b와 같은 종류의 미리보기/커밋 어긋남**이 된다.

- [ ] **Step 6: 9개 로케일에 키를 추가한다**

필요한 키: `tools.brushRamp`, `tools.brushRamp.none`, `.alongStroke`, `.fixedAxis`, `tools.rampFrom`, `tools.rampTo`, `tools.rampAxis`, `tools.rampAxis.horizontal`, `.vertical`.

`ko-KR`은 실제 한국어로("붓 그라디언트", "없음", "획 방향", "고정 축", "시작 색", "끝 색", "축", "가로", "세로"). 나머지 8개는 실제 번역으로.

**키 집합이 9개 파일에서 동일해야 한다.** 편집 후 확인한다.

- [ ] **Step 7: 프론트 검사를 돌린다**

Run: `bun run --filter '@koharu/app' test`
Run: `bunx tsc --noEmit -p packages/koharu/tsconfig.json`
Run: `bunx oxlint -c packages/koharu/.oxlintrc.json packages/koharu`
Expected: 전부 통과

- [ ] **Step 8: 커밋**

```bash
git add packages/koharu/lib/store.ts packages/koharu/components/editor/ToolBar.tsx packages/koharu/components/editor/CanvasWorkspace.tsx packages/koharu/public/locales packages/koharu/tests
git commit -m "feat(ui): choose a brush ramp"
```

---

### Task 5: 회귀 확인

**Files:**
- Test: 없음 (검증만)

- [ ] **Step 1: 관련 크레이트와 프론트를 돌린다**

Run: `cargo test -j 6 -p koharu-rasterizer -p koharu-app -p koharu-canvas -p koharu-rpc --lib`
Run: `cargo check -j 6 --workspace`
Run: `bun run --filter '@koharu/app' test`

- [ ] **Step 2: 램프가 없을 때 기존 단색 경로가 그대로인지 확인한다**

`no_ramp_keeps_the_plain_stroke_colour`가 그것을 덮는다. 추가로 `apply_raster_stroke`가 B2에서 공유 헬퍼를 쓰도록 리팩터링되었으므로, **paint 경로 테스트 2개(`painting_creates_a_layer_and_claims_both_origins`, `painting_an_existing_layer_keeps_its_identity`)가 여전히 통과하는지 확인한다** — 램프 인자 추가가 그 경로를 건드렸다.

- [ ] **Step 3: `Erase` 모드가 램프를 무시하는지 확인한다**

`Erase`는 색을 쓰지 않으므로 램프가 있어도 지우기 동작이 같아야 한다. 테스트가 없으면 확인하고, 없으면 **추가한다**.

- [ ] **Step 4: 미리보기와 커밋의 어긋남을 문서화했는지 확인한다**

Task 3의 결정(미리보기 `AlongStroke`는 페이지 대각선으로 나눈다)이 코드 주석에 있고, 커밋이 권위라는 문장이 있는지 확인한다.

- [ ] **Step 5: 보고**

테스트 결과와 `git diff --stat`를 보고서에 남긴다.

---

## Self-Review

**Spec coverage**

| 스펙 절 | 태스크 |
|---|---|
| `BrushRamp` 2종 (B-D1) | Task 1 |
| `PaintBrush.ramp` | Task 2 |
| `rasterize_stroke`가 램프로 색 결정 | Task 2 |
| `FixedAxis`가 페이지 정규화 좌표 | Task 2 |
| `StrokeCommitDto::Paint`에 램프 | Task 4 (기존 `brush` 통과 경로 재사용) |
| WASM 미리보기 `draw_segment` | Task 3 |
| `CanvasBrush.ramp` | Task 4 |
| 알파 보존 (B-D8) | Task 1 (4채널 보간) |
| `GradientSpec` 미재사용 (B-D2) | Task 1 |
| 테스트 표 (램프 행) | Task 2, Task 3 |

**Spec gap을 계획이 메운 곳**

1. **알파 보간.** 스펙은 "두 색"만 말했고 알파를 명시하지 않았다. B-D8의 "알파 보존"은 색보정에 대한 결정이었지만 램프에도 적용된다 — Task 1이 4채널을 모두 보간하고 테스트로 고정한다.
2. **`AlongStroke`의 미리보기 문제.** 스펙은 "진행 방향 램프는 누적 길이"라고만 했다. 진행 중인 스트로크는 총 길이를 모르므로 **미리보기와 커밋이 반드시 어긋난다.** Task 3이 이 사실을 명시하고, 어긋남을 줄이려 하지 않고 **커밋을 권위로 둔다**고 정한다.
3. **`validate`를 만들지 않는다.** 스펙의 `GradientSpec::validate` 관례를 기계적으로 따르면 검증할 것이 없는 함수가 생긴다. Task 1이 만들지 않고 이유를 남긴다.
4. **세그먼트 로컬 `t`와 누적 길이의 구분.** 기존 `rasterize_stroke`의 `t`는 세그먼트 투영 매개변수이고 램프는 누적 비율이다. Task 2가 둘을 명시적으로 구분하고, 한 점 스트로크(길이 0)를 별도 처리��다.
5. **`Erase`는 램프를 무시한다.** 스펙에 없었다. Task 2와 Task 5가 확인한다.
6. **TS 색 타입.** 스펙은 램프를 "두 색"이라고만 했다. 이 저장소에 `Rgba`라는 이름의 타입은 없고 `PaintBrush`가 인라인 튜플을 쓰므로, Task 4가 같은 관례를 따르도록 명시했다.

**Type consistency 확인**

- `BrushRamp`/`RampAxis`는 Rust에서 `koharu-rasterizer` 소유이고 `koharu-canvas`·`koharu-app`이 공유한다 — 미리보기와 커밋이 **같은 타입**을 쓴다.
- serde는 내부 태그(`tag = "kind"`)이고 TS union의 `kind` 값이 `along_stroke`/`fixed_axis`이다(`snake_case`).
- `RampAxis`는 `horizontal`/`vertical`.
- `rasterize_stroke`의 `ramp`은 `Option<BrushRamp>`이고 `color` 뒤에 온다.
- `color_at(position: f32) -> [u8; 4]` — 네 채널을 항상 돌려준다.

**Proportion**

스펙 B1은 약 20줄, 계획은 약 620줄. 코드 블록은 테스트와 시그니처 위주이고, 구현 본문은 알고리즘이 테스트로 결정되지 않는 곳(Task 2의 누적 길이, Task 3의 미리보기 결정)에만 있다.
