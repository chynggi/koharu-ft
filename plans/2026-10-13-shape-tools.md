# 도형 도구 (1a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 사각형·타원·선·화살표·다각형·자유곡선을 벡터 레이어로 그려 저장하고, 채우기·테두리·점선 스타일을 편집하며, 필요할 때 래스터로 구울 수 있게 한다.

**Architecture:** 도형은 `ShapeLayer` 컴포넌트(폴리곤 + 재편집 파라미터 + 스타일)로 저장한다. `Geometry` 폴리곤이 히트테스트·프레임 복원의 권위이고, `ShapeGeometry` 파라미터가 렌더링의 권위다. 렌더러는 씬 컴파일 시점에 파라미터에서 vello 경로를 만들어 기존 `PreparedContent::Vector` 경로로 흘려보낸다. 래스터화는 기존 `Rasterizer::rasterize_scene`을 재사용한다.

**Tech Stack:** Rust (koharu-scene, koharu-renderer, koharu-rasterizer, koharu-app, koharu-rpc), vello 0.10 / kurbo, TypeScript (Next.js, zustand, i18next), `cargo test`, vitest.

**Spec:** `specs/2026-10-13-shape-tools-design.md`

**계획 위치 메모:** 저장소 관례대로 `plans/`에 둔다. `docs/`는 발행용 문서 사이트라 고아 페이지가 섞인다.

---

## Global Constraints

- 컴포넌트 스키마 비트는 **18**을 쓴다 (17까지 사용 중: `crates/koharu-scene/src/schema.rs:44-63`).
- `ShapeLayer`의 KIND 문자열은 `"dev.koharu.layer.shape"`.
- 폴리곤은 최소 3점 (`Geometry::validate`, `crates/koharu-scene/src/components/spatial.rs:49`).
- 래스터화 배경은 반드시 `[0, 0, 0, 0]`. `[0,0,0,255]`는 도형 상자를 검게 덮는다.
- 래스터화 슈퍼샘플링은 4배 (`RasterOptions::supersampled(4)`).
- `PreparedPath`는 파괴적으로 바꾼다. 구형식 파서를 남기지 않는다 (AGENTS.md "Never add backward compatibility").
- 검증은 가장 작은 관련 검사만 debug 프로필로 한 번 돌린다 (AGENTS.md).
- 도구는 하나(`'shape'`) + 종류 선택. 단축키는 `'r'`.
- 새 Tauri 커맨드는 **반드시** `crates/koharu-rpc/src/routes/canvas.rs`에 라우트를 함께 추가한다 (`fork/HANDOFF.md:213`의 교훈).

## Review Focus

스펙이 명시하지 않았지만 사용자가 반드시 만나게 되는 입력들이다. 각 항목의 테스트는 그 코드를 소유한 태스크에 넣는다.

1. **회전한 타원** — 45° 회전한 원의 프레임이 지름을 유지해야 한다. 폴리곤의 축 정렬 경계 상자를 쓰면 √2배로 커진다 (Task 4).
2. **테두리만 있는 도형(`fill: None`)** — 채우기 명령을 만들지 않아야 한다. 빈 채우기가 배경을 덮으면 안 된다 (Task 3).
3. **`angle_degrees: None`인 회전 도형** — 예외 없이 축 정렬 경계로 폴백해야 한다 (Task 4).
4. **래스터화한 도형 위에 다른 도형** — 아래 그림이 검게 덮이지 않아야 한다 (Task 7).
5. **0 길이 선** — 시작과 끝이 같은 좌표일 때 NaN이나 무한 루프 없이 사라지거나 점을 그려야 한다 (Task 3).

---

## File Structure

**신규**

| 파일 | 책임 |
|---|---|
| `crates/koharu-scene/src/components/shapes.rs` | `ShapeLayer`, `ShapeGeometry`, `ShapeStyle`, `ShapeFill`, `StrokeStyle`, `LineCap`, `LineJoin`, `ArrowHead`와 검증 |
| `crates/koharu-renderer/src/shape_renderer.rs` | `ShapeLayer` → vello 경로(`PreparedPath`) 변환. 도형 종류별 지오메트리 생성 |
| `packages/koharu/lib/shapes.ts` | 프론트 도형 종류 표, 기본 스타일, 프레임 → 도형 폴리곤 생성 |
| `packages/koharu/tests/lib/shapes.test.ts` | 위 모듈 테스트 |

**수정**

| 파일 | 변경 |
|---|---|
| `crates/koharu-scene/src/components.rs` | `shapes` 모듈 + 재수출 |
| `crates/koharu-scene/src/schema.rs` | `SHAPE_LAYER = 18` 등록 |
| `crates/koharu-rasterizer/src/prepared.rs` | `PreparedPath` 확장, `PreparedFill`/`PreparedStroke`, `LayerKind::Shape`, 검증 조합 |
| `crates/koharu-rasterizer/src/frame.rs` | `compile_scene`의 fill/stroke 분기 |
| `crates/koharu-renderer/src/frame.rs` | `LayerKind::Shape`, `NodeDescriptor::Shape`, `ShapeMetadata`, `prepare_frame` 분기, `element_frame` 분기 |
| `crates/koharu-renderer/src/renderer.rs` | `DraftMetadata::Shape`, `assemble_frame` 분기, `build_node` 분기 |
| `crates/koharu-renderer/src/text_renderer.rs` | 글리프 생성의 `PreparedPath` 신형식 대응 |
| `crates/koharu-desktop/src/lib.rs` | `transform_geometries`의 종류 게이트 |
| `crates/koharu-canvas/src/web.rs` | `begin_transform` 게이트, `ShapeEdit` 미리보기 |
| `crates/koharu-app/src/commands/canvas.rs` | `add_shape`, `set_shape_style`, `set_shape_geometries`, `rasterize_shapes` |
| `crates/koharu-app/src/commands/project.rs` | `Project::add_shape`, `::set_shape_style`, `::set_shape_geometries`, `::rasterize_shapes`, `Layer::Shape` 노출 |
| `crates/koharu-rpc/src/routes/canvas.rs` | 4개 라우트 |
| `crates/koharu-psd/src/document.rs` | 도형을 래스터 레이어로 |
| `packages/bridge/src/protocol.ts` | `ShapeLayer` 타입, 4개 명령 |
| `packages/koharu/lib/store.ts` | `'shape'` 도구, `shapeKind`, `shapeStyle` |
| `packages/koharu/lib/geometry.ts` | `selectableLayer`에 `'shape'` |
| `packages/koharu/components/editor/ToolBar.tsx` | 도형 도구 + 스타일 Popover |
| `packages/koharu/components/editor/CanvasWorkspace.tsx` | 도형 드래그 제스처 |
| `packages/koharu/components/editor/Inspector.tsx` | 도형 스타일 섹션 |
| `packages/koharu/public/locales/*/translation.json` | 9개 로케일 |

---

### Task 1: `ShapeLayer` 컴포넌트

저장 모델을 먼저 세운다. 이후 모든 태스크가 이 타입을 소비한다.

**Files:**
- Create: `crates/koharu-scene/src/components/shapes.rs`
- Modify: `crates/koharu-scene/src/components.rs`
- Modify: `crates/koharu-scene/src/schema.rs:44-63`
- Test: `crates/koharu-scene/src/components/shapes.rs` (인라인 `mod tests`)

**Interfaces:**
- Consumes: `crate::components::Origin`, `crate::component::{Component, ValidationContext}`, `crate::{Error, Result}`
- Produces:
  - `ShapeLayer { origin: Origin, name: String, geometry: ShapeGeometry, style: ShapeStyle, angle_degrees: Option<f32>, rasterized: bool }`
  - `ShapeGeometry::{Rectangle{corner_radius: f32}, Ellipse, Line{start: [f64;2], end: [f64;2], head: Option<ArrowHead>}, Polygon, Freeform}`
  - `ArrowHead { length: f32, width: f32 }`
  - `ShapeStyle { fill: Option<ShapeFill>, stroke: Option<StrokeStyle> }`
  - `ShapeFill::Solid([u8;4])`
  - `StrokeStyle { color: [u8;4], width: f32, dash: Option<Vec<f32>>, cap: LineCap, join: LineJoin }`
  - `LineCap::{Butt, Round, Square}`, `LineJoin::{Miter, Round, Bevel}`
  - `ShapeLayer::KIND == "dev.koharu.layer.shape"`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`crates/koharu-scene/src/components/shapes.rs`를 만들고 테스트만 넣는다. `ValidationContext` 생성법은 같은 디렉터리의 `layers.rs` 테스트나 `crates/koharu-scene/src/tests.rs`에서 확인해 맞춘다.

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Origin, Result, component::ValidationContext};

    fn validate(layer: &ShapeLayer) -> Result<()> {
        let record_exists = |_id| true;
        let blob_exists = |_id| true;
        let context = ValidationContext::new(&record_exists, &blob_exists);
        layer.validate(&context)
    }

    fn style() -> ShapeStyle {
        ShapeStyle {
            fill: Some(ShapeFill::Solid([0, 0, 0, 255])),
            stroke: None,
        }
    }

    fn shape(geometry: ShapeGeometry) -> ShapeLayer {
        ShapeLayer {
            origin: Origin::User,
            name: "Rectangle 1".to_owned(),
            geometry,
            style: style(),
            angle_degrees: None,
            rasterized: false,
        }
    }

    #[test]
    fn a_named_shape_validates() {
        let layer = shape(ShapeGeometry::Rectangle { corner_radius: 4.0 });
        assert!(validate(&layer).is_ok());
    }

    #[test]
    fn a_negative_corner_radius_is_rejected() {
        let layer = shape(ShapeGeometry::Rectangle { corner_radius: -1.0 });
        assert!(validate(&layer).is_err());
    }

    #[test]
    fn a_non_finite_angle_is_rejected() {
        let mut layer = shape(ShapeGeometry::Ellipse);
        layer.angle_degrees = Some(f32::NAN);
        assert!(validate(&layer).is_err());
    }

    #[test]
    fn a_zero_dash_element_is_rejected() {
        let mut layer = shape(ShapeGeometry::Polygon);
        layer.style.stroke = Some(StrokeStyle {
            color: [0, 0, 0, 255],
            width: 2.0,
            dash: Some(vec![8.0, 0.0]),
            cap: LineCap::Butt,
            join: LineJoin::Miter,
        });
        assert!(validate(&layer).is_err());
    }

    #[test]
    fn an_empty_name_is_rejected() {
        let mut layer = shape(ShapeGeometry::Freeform);
        layer.name = String::new();
        assert!(validate(&layer).is_err());
    }

    #[test]
    fn a_negative_stroke_width_is_rejected() {
        let mut layer = shape(ShapeGeometry::Polygon);
        layer.style.stroke = Some(StrokeStyle {
            color: [0, 0, 0, 255],
            width: -1.0,
            dash: None,
            cap: LineCap::Butt,
            join: LineJoin::Miter,
        });
        assert!(validate(&layer).is_err());
    }

    #[test]
    fn a_negative_arrow_head_is_rejected() {
        let layer = shape(ShapeGeometry::Line {
            start: [0.0, 0.0],
            end: [10.0, 0.0],
            head: Some(ArrowHead { length: -1.0, width: 4.0 }),
        });
        assert!(validate(&layer).is_err());
    }
}
```

**`ValidationContext`에는 `Default`가 없다.** `crates/koharu-scene/src/component.rs:16`의 `new`는 `pub(crate)`이고 두 클로저를 받는다. 같은 크레이트의 테스트이므로 `ValidationContext::new(&record_exists, &blob_exists)`를 쓸 수 있고, 관용구는 `crates/koharu-scene/src/tests.rs:205-207`에 있다. 위 `validate` 헬퍼가 그 형태다. (테스트 상단의 쓰이지 않는 `context()` 헬퍼는 넣지 않는다 — 실수로 남기면 clippy가 잡는다.)

- [ ] **Step 2: 실패를 확인한다**

Run: `cargo test -j 6 -p koharu-scene --lib components::shapes`
Expected: 컴파일 실패 (`ShapeLayer` 등 미정의)

- [ ] **Step 3: 타입과 검증을 구현한다**

`koharu-scene`은 kurbo/vello에 의존하지 않으므로 `LineCap`/`LineJoin`을 자체 정의한다. 각 열거형에 `#[revisioned(revision = 1)]`, `#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Type)]`, `#[serde(rename_all = "snake_case")]`를 붙인다. `ShapeLayer`는 `layers.rs`의 `RasterLayer`와 같은 파생 집합을 쓴다.

`validate`는 전부 거부한다: `origin.validate()` 실패, 빈 이름/4096바이트 초과/NUL, 음수·비유한 `corner_radius`, 음수·비유한 `stroke.width`, `dash`의 0 이하·비유한 원소, 비유한 `angle_degrees`, 음수·비유한 `ArrowHead` 값.

**폴리곤과 파라미터의 불일치는 거부하지 않는다.** `ShapeLayer`는 폴리곤을 갖지 않으므로(`Geometry`가 별도 컴포넌트) 불일치 검사 자체가 여기 없다. 이 사실을 `validate` 위 주석에 남긴다.

`Component` 구현은 `layers.rs:32-51`의 `RasterLayer`를 그대로 따른다 — `KIND`, `validate`, `origin`, `set_origin`.

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

Run: `cargo test -j 6 -p koharu-scene --lib components::shapes`
Expected: 7 passed

- [ ] **Step 5: 모듈과 스키마에 등록한다**

`crates/koharu-scene/src/components.rs`에 `mod shapes;`와 `pub use shapes::{...};`를 추가한다 (알파벳 순서 유지).

`crates/koharu-scene/src/schema.rs`의 `component_schema!`에 마지막 줄로 추가한다:

```rust
    SHAPE_LAYER = 18 => ShapeLayer,
```

그리고 `schema.rs` 상단 `use crate::{...}` 목록에 `ShapeLayer`를 추가한다.

- [ ] **Step 6: 워크스페이스가 컴파일되는지 확인한다**

Run: `cargo check -j 6 -p koharu-scene`
Expected: 경고 없이 성공

- [ ] **Step 7: 커밋**

```bash
git add crates/koharu-scene/src/components/shapes.rs crates/koharu-scene/src/components.rs crates/koharu-scene/src/schema.rs
git commit -m "feat(scene): add the shape layer component"
```

---

### Task 2: `PreparedPath` 확장

`PreparedPath`를 채우기·테두리 열거형으로 바꾸고 모든 소비자를 한 번에 고친다. 이 태스크가 끝나면 기존 텍스트 렌더링이 같은 결과를 내야 한다.

**Files:**
- Modify: `crates/koharu-rasterizer/src/prepared.rs:692-735`
- Modify: `crates/koharu-rasterizer/src/frame.rs:494-526`
- Modify: `crates/koharu-renderer/src/text_renderer.rs:415-429`
- Modify: `crates/koharu-rasterizer/src/lib.rs` (재수출)

**Interfaces:**
- Consumes: 없음
- Produces:
  - `PreparedPath { transform: [f64;6], elements: Vec<PathElement>, fill: Option<PreparedFill>, stroke: Option<PreparedStroke> }`
  - `PreparedFill::Solid { color: [u8;4], rule: FillRule }`
  - `PreparedStroke { color: [u8;4], width: f32, dash: Option<Vec<f32>>, cap: PreparedLineCap, join: PreparedLineJoin }`
  - `PreparedLineCap::{Butt, Round, Square}`, `PreparedLineJoin::{Miter, Round, Bevel}`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`crates/koharu-rasterizer/src/prepared.rs`의 기존 `mod tests`에 추가한다.

```rust
    #[test]
    fn a_path_may_have_a_fill_and_a_stroke() {
        let path = PreparedPath {
            transform: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            elements: vec![
                PathElement::MoveTo([0.0, 0.0]),
                PathElement::LineTo([10.0, 0.0]),
                PathElement::LineTo([10.0, 10.0]),
                PathElement::Close,
            ],
            fill: Some(PreparedFill::Solid {
                color: [255, 0, 0, 255],
                rule: FillRule::NonZero,
            }),
            stroke: Some(PreparedStroke {
                color: [0, 0, 0, 255],
                width: 2.0,
                dash: None,
                cap: PreparedLineCap::Round,
                join: PreparedLineJoin::Round,
            }),
        };
        assert!(path.validate().is_ok());
    }

    #[test]
    fn a_path_without_fill_or_stroke_validates() {
        let path = PreparedPath {
            transform: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            elements: vec![PathElement::MoveTo([0.0, 0.0]), PathElement::Close],
            fill: None,
            stroke: None,
        };
        assert!(path.validate().is_ok());
    }

    #[test]
    fn a_negative_stroke_width_is_rejected() {
        let path = PreparedPath {
            transform: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            elements: vec![PathElement::MoveTo([0.0, 0.0]), PathElement::Close],
            fill: None,
            stroke: Some(PreparedStroke {
                color: [0, 0, 0, 255],
                width: -1.0,
                dash: None,
                cap: PreparedLineCap::Butt,
                join: PreparedLineJoin::Miter,
            }),
        };
        assert!(path.validate().is_err());
    }
```

`PreparedPath::validate`가 현재 private이면 이 태스크에서 `pub fn validate(&self) -> Result<()>`로 넓힌다 (테스트가 같은 모듈에 있으므로 `pub(crate)`로도 충분하다; 기존 가시성을 확인해 더 좁은 쪽을 유지한다).

- [ ] **Step 2: 실패를 확인한다**

Run: `cargo test -j 6 -p koharu-rasterizer --lib prepared::tests`
Expected: 컴파일 실패

- [ ] **Step 3: 구조체를 바꾸고 검증을 넓힌다**

`PreparedPath`에서 `fill: FillRule`과 `color: [u8;4]`를 제거하고 위 인터페이스대로 바꾼다. `validate`는 기존의 transform·elements 검사에 더해:

- `fill`이 `Some`이면 색은 검사하지 않는다 (`[u8;4]`는 모든 값이 유효)
- `stroke`가 `Some`이면 `width`가 유한하고 0 이상이어야 한다
- `stroke.dash`가 `Some`이면 비어 있지 않고 모든 원소가 유한하며 0보다 커야 한다

- [ ] **Step 4: 두 소비자를 고친다**

`crates/koharu-rasterizer/src/frame.rs:494`의 `PreparedSceneCommand::FillPath` 분기를 다음으로 바꾼다:

1. `prepared.fill`이 `Some(PreparedFill::Solid { color, rule })`이면 기존 `scene.fill(...)`을 `rule`과 `color`로 호출한다. `None`이면 호출하지 않는다.
2. `prepared.stroke`가 `Some`이면 `scene.stroke(&stroke, Affine::new(prepared.transform), rgba(color), None, &path)`를 호출한다. `stroke`는 `kurbo::Stroke::new(width)`에 `.with_caps(...)`, `.with_join(...)`, `dash`가 있으면 `.with_dashes(0.0, dash)`를 적용해 만든다.

`crates/koharu-renderer/src/text_renderer.rs:415`의 `PreparedGlyphRun` 생성은 `PreparedPath`를 만들지 않는다 (`PreparedSceneCommand::GlyphRun`을 쓴다). 컴파일 오류가 나는 곳만 고치면 되며, 확인 결과 이 파일에는 `PreparedPath` 리터럴이 없다. 다른 곳에서 컴파일 오류가 나면 같은 방식으로 `fill: Some(PreparedFill::Solid { color, rule: FillRule::NonZero })`, `stroke: None`으로 바꾼다.

- [ ] **Step 5: 테스트가 통과하는지 확인한다**

Run: `cargo test -j 6 -p koharu-rasterizer --lib`
Expected: 기존 테스트 포함 전부 통과

- [ ] **Step 6: 렌더러가 컴파일되는지 확인한다**

Run: `cargo check -j 6 -p koharu-renderer`
Expected: 성공

- [ ] **Step 7: 커밋**

```bash
git add crates/koharu-rasterizer/src/prepared.rs crates/koharu-rasterizer/src/frame.rs crates/koharu-renderer/src/text_renderer.rs
git commit -m "refactor(rasterizer): express path fill and stroke as optional fills"
```

---

### Task 3: 도형 경로 생성기

`ShapeGeometry` + 프레임 → `PreparedPath`. 여기가 도형 종류별 지오메트리의 유일한 구현이다.

**Files:**
- Create: `crates/koharu-renderer/src/shape_renderer.rs`
- Modify: `crates/koharu-renderer/src/lib.rs` (모듈 등록)

**Interfaces:**
- Consumes: `koharu_scene::{ShapeGeometry, ShapeStyle, ShapeFill, StrokeStyle, LineCap, LineJoin, ArrowHead}` (Task 1), `koharu_rasterizer::{PreparedPath, PreparedFill, PreparedStroke, PreparedLineCap, PreparedLineJoin, PathElement, FillRule}` (Task 2)
- Produces:
  - `pub struct ShapeFrame { pub x: f32, pub y: f32, pub width: f32, pub height: f32, pub angle_degrees: f32 }`
  - `pub fn shape_frame(geometry: &Geometry, angle_degrees: Option<f32>) -> Option<ShapeFrame>`
  - `pub fn shape_path(geometry: &ShapeGeometry, style: &ShapeStyle, frame: ShapeFrame) -> PreparedPath`
  - `pub fn polygon_path(points: &[koharu_scene::Point], style: &ShapeStyle, frame: ShapeFrame) -> PreparedPath`
  - `pub fn shape_layer_path(shape: &ShapeGeometry, points: &[koharu_scene::Point], style: &ShapeStyle, frame: ShapeFrame) -> PreparedPath`

`ShapeFrame`를 새로 두는 이유: `crate::bubble::GeometryFrame`은 `pub(crate)`이고 `koharu-app`(Task 7)이 래스터화에 `shape_path`를 써야 한다. `GeometryFrame`을 공개하는 대신 같은 값을 담는 공개 타입을 두고, `shape_frame`이 그 변환을 캡슐화한다. `shape_frame`은 내부적으로 `crate::bubble::geometry_frame`을 호출하고 `None`이면 `None`을 돌려준다.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`crates/koharu-renderer/src/shape_renderer.rs`에 테스트만 넣는다.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn frame(width: f32, height: f32, angle_degrees: f32) -> ShapeFrame {
        ShapeFrame { x: 0.0, y: 0.0, width, height, angle_degrees }
    }

    fn filled() -> ShapeStyle {
        ShapeStyle {
            fill: Some(ShapeFill::Solid([10, 20, 30, 255])),
            stroke: None,
        }
    }

    #[test]
    fn a_rectangle_becomes_a_closed_four_point_path() {
        let path = shape_path(
            &ShapeGeometry::Rectangle { corner_radius: 0.0 },
            &filled(),
            frame(40.0, 20.0, 0.0),
        );
        assert!(matches!(path.fill, Some(PreparedFill::Solid { .. })));
        assert_eq!(path.elements.last(), Some(&PathElement::Close));
    }

    #[test]
    fn a_rectangle_keeps_its_rotation_in_the_transform() {
        let path = shape_path(
            &ShapeGeometry::Rectangle { corner_radius: 0.0 },
            &filled(),
            frame(40.0, 20.0, 90.0),
        );
        // 90도 회전은 [cos, sin, -sin, cos, ..] = [0, 1, -1, 0, ..]
        assert!((path.transform[0] - 0.0).abs() < 1e-9);
        assert!((path.transform[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn an_unfilled_rectangle_has_no_fill() {
        let style = ShapeStyle { fill: None, stroke: None };
        let path = shape_path(
            &ShapeGeometry::Rectangle { corner_radius: 0.0 },
            &style,
            frame(40.0, 20.0, 0.0),
        );
        assert!(path.fill.is_none());
    }

    #[test]
    fn a_zero_length_line_produces_finite_coordinates() {
        let style = ShapeStyle {
            fill: None,
            stroke: Some(StrokeStyle {
                color: [0, 0, 0, 255],
                width: 2.0,
                dash: None,
                cap: LineCap::Round,
                join: LineJoin::Round,
            }),
        };
        let path = shape_path(
            &ShapeGeometry::Line {
                start: [5.0, 5.0],
                end: [5.0, 5.0],
                head: None,
            },
            &style,
            frame(40.0, 20.0, 0.0),
        );
        for element in &path.elements {
            let values: &[f64] = match element {
                PathElement::MoveTo(v) | PathElement::LineTo(v) => v,
                PathElement::QuadTo(v) => v,
                PathElement::CurveTo(v) => v,
                PathElement::Close => continue,
            };
            assert!(values.iter().all(|value| value.is_finite()));
        }
        assert!(path.validate().is_ok());
    }

    #[test]
    fn an_arrow_head_adds_a_filled_triangle_element() {
        let style = ShapeStyle {
            fill: None,
            stroke: Some(StrokeStyle {
                color: [0, 0, 0, 255],
                width: 2.0,
                dash: None,
                cap: LineCap::Butt,
                join: LineJoin::Miter,
            }),
        };
        let path = shape_path(
            &ShapeGeometry::Line {
                start: [0.0, 0.0],
                end: [50.0, 0.0],
                head: Some(ArrowHead { length: 10.0, width: 8.0 }),
            },
            &style,
            frame(100.0, 100.0, 0.0),
        );
        // 화살촉은 채워진 삼각형이므로 두 번째 FillPath가 필요하다.
        // 이 함수는 하나의 PreparedPath만 돌려주므로, 화살촉은 같은
        // 경로에 별도 subpath로 들어가고 fill이 함께 켜진다.
        assert!(path.fill.is_some());
    }
}
```

- [ ] **Step 2: 실패를 확인한다**

Run: `cargo test -j 6 -p koharu-renderer --lib shape_renderer`
Expected: 컴파일 실패

- [ ] **Step 3: 생성기를 구현한다**

`shape_path`는 프레임의 로컬 좌표계로 경로를 만든다. 모든 도형은 다음 규칙을 따른다:

- 로컬 원점은 프레임 상자의 좌상단. `PathElement` 좌표�� `0.0..width`, `0.0..height` 범위.
- `transform`은 프레임 중심으로 회전·이동하는 아핀 계수다. `frame.bounds`의 중심 `(x + w/2, y + h/2)`와 `frame.angle_degrees`에서 만든다:

```rust
let (sin, cos) = f64::from(frame.angle_degrees).to_radians().sin_cos();
let center_x = f64::from(frame.x + frame.width * 0.5);
let center_y = f64::from(frame.y + frame.height * 0.5);
let transform = [
    cos, sin, -sin, cos,
    center_x - (cos * f64::from(frame.width) * 0.5 - sin * f64::from(frame.height) * 0.5),
    center_y - (sin * f64::from(frame.width) * 0.5 + cos * f64::from(frame.height) * 0.5),
];
```

종류별 생성:

- `Rectangle { corner_radius }` — `corner_radius <= 0.0`이면 네 점 + `Close`. 0보다 크면 각 모서리를 반지름 `min(corner_radius, w/2, h/2)`로 둥글게 만든다. 각 모서리마다 `LineTo` + `QuadTo`를 쓴다.
- `Ellipse` — 중심 `(w/2, h/2)`, 반지름 `(w/2, h/2)`인 타원을 4개의 `CurveTo`로 근사한다 (κ = 0.5522847498307936).
- `Line { start, end, head }` — `start`·`end`는 페이지 좌표이므로 프레임 로컬로 옮긴다 (프레임 중심 기준 회전 후 평행이동). `MoveTo(start)` + `LineTo(end)` + `Close`. `head`가 있으면 끝점에 수직인 삼각형을 만든다.

  **`Line`은 폴리곤과 중복 저장되므로 갱신 시 재계산이 필요하다.** 이것이 이 설계의 유일한 이중 저장 지점이고, 방치하면 실제 버그가 된다: 모든 변형 경로(`set_shape_geometries`, `commit_transform`)는 **폴리곤만** 갱신하는데 `line_path`는 `start`/`end`로 그리므로, 선을 드래그하면 폴리곤은 움직이지만 그려지는 선은 제자리에 남고 `element_frame`(폴리곤을 읽는다)도 선에서 분리된다.

  계약: **폴리곤이 권위이고 `start`/`end`는 파생값이다.** 폴리곤을 쓰는 쪽은 반드시 `Line`의 두 끝점을 폴리곤에서 다시 계산해야 한다. `Line`의 폴리곤은 선을 따라 균등 분할된 3점 이상이므로(아래 "폴리곤 점 개수"), **첫 점이 `start`, 마지막 점이 `end`** 다. 재계산 헬퍼를 `koharu-scene`에 둔다:

  ```rust
  impl ShapeGeometry {
      /// 폴리곤이 권위인 `Line`의 두 끝점을 다시 맞춘다.
      /// 다른 변형은 폴리곤에서 파생되는 값이 없으므로 그대로 둔다.
      pub fn resync_with_polygon(&mut self, points: &[Point]) {
          if let Self::Line { start, end, .. } = self
              && let (Some(first), Some(last)) = (points.first(), points.last())
          {
              *start = [first.x, first.y];
              *end = [last.x, last.y];
          }
      }
  }
  ```

  `set_shape_geometries`는 `edit.set(element, &geometry)` 직전에 이 헬퍼를 호출해야 한다. Task 6이 이 호출과 테스트를 소유한다.
- `Polygon` / `Freeform` — `shape_path`는 이 두 변형에서 빈 경로(`MoveTo(0,0)`만)를 돌려준다. **`polygon_path`가 폴리곤 점을 받아 실제 경로를 만든다.** 두 함수를 고르는 일은 `shape_layer_path`가 한다.

**추가 테스트**: `shape_layer_path`가 `Polygon`에서 `points`를 쓰고 `points`가 비면 빈 경로를 돌려주는지 확인한다.

```rust
    #[test]
    fn shape_layer_path_delegates_polygons_to_the_points() {
        let points = vec![
            koharu_scene::Point { x: 0.0, y: 0.0 },
            koharu_scene::Point { x: 10.0, y: 0.0 },
            koharu_scene::Point { x: 5.0, y: 10.0 },
        ];
        let path = shape_layer_path(
            &ShapeGeometry::Polygon,
            &points,
            &filled(),
            frame(20.0, 20.0, 0.0),
        );
        assert!(path.elements.len() >= 4); // 3 MoveTo/LineTo + Close
    }
```

스타일 매핑: `style.fill`이 `Some(ShapeFill::Solid(color))`이면 `PreparedFill::Solid { color, rule: FillRule::NonZero }`, 아니면 `None`. `style.stroke`가 `Some`이면 `PreparedStroke`로 옮기고 `LineCap`/`LineJoin`을 `PreparedLineCap`/`PreparedLineJoin`으로 매핑한다.

화살촉이 있으면 삼각형 subpath를 경로 끝에 추가하고 `fill`을 `Some`으로 만든다 (테스트 5가 요구하는 동작). 스타일의 `fill`이 `None`이어도 화살촉 때문에 `Some`이 된다 — 이 규칙을 주석에 남긴다.

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

Run: `cargo test -j 6 -p koharu-renderer --lib shape_renderer`
Expected: 6 passed

- [ ] **Step 5: 모듈을 등록한다**

`crates/koharu-renderer/src/lib.rs`에 `mod shape_renderer;`를 추가한다 (다른 `mod` 선언과 같은 가시성·위치).

- [ ] **Step 6: 커밋**

```bash
git add crates/koharu-renderer/src/shape_renderer.rs crates/koharu-renderer/src/lib.rs
git commit -m "feat(renderer): generate prepared paths for shapes"
```

---

### Task 4: 렌더러 배선과 프레임 복원

도형을 렌더러의 1급 레이어로 만들고 `element_frame`을 연다. 여기가 `LayerKind`를 exhaustive match하는 모든 지점을 건드린다.

**Files:**
- Modify: `crates/koharu-renderer/src/frame.rs:322-325, 442-445, 474-544, 602-617`
- Modify: `crates/koharu-renderer/src/renderer.rs:463-470, 1100-1181, 1193-1219`
- Modify: `crates/koharu-rasterizer/src/prepared.rs:441-453, 458-462`
- Test: `crates/koharu-renderer/src/frame.rs` (인라인 `mod tests`)

**Interfaces:**
- Consumes: Task 1의 `ShapeLayer`·`ShapeGeometry`·`ShapeStyle`, Task 3의 `shape_path`
- Produces:
  - `koharu_rasterizer::LayerKind::Shape`
  - `koharu_renderer::LayerKind::Shape(ShapeMetadata)`
  - `ShapeMetadata { geometry: ShapeGeometry, style: ShapeStyle, angle_degrees: Option<f32> }`
  - `NodeDescriptor::Shape(Box<ShapeNodeDescriptor>)` — `ShapeNodeDescriptor { geometry: ShapeGeometry, style: ShapeStyle, angle_degrees: Option<f32>, polygon: Vec<(f64, f64)> }`
  - `DraftMetadata::Shape(ShapeMetadata)`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`crates/koharu-renderer/src/frame.rs`의 `mod tests`에 추가한다. 테스트는 `Layer`를 직접 만드는 대신 `Renderer`가 스냅샷에서 프레임을 만드는 기존 테스트 헬퍼가 있으면 그것을 쓴다. 없으면 `geometry_frame`과 `element_frame`을 단위로 검증한다 — 이 태스크의 핵심 회귀가 회전이기 때문이다.

```rust
    #[test]
    fn a_rotated_shape_frame_preserves_its_dimensions() {
        // 45도 회전한 비원형 타원(rx=50, ry=25)을 32점 폴리곤으로 샘플링한다.
        //
        // 원을 쓰면 안 된다: 원의 축 정렬 경계 상자는 회전에 불변이라
        // 각도를 주든 안 주든 지름과 같다. 비원형 타원이라야 회전 시
        // 축 정렬 상자가 100x50 -> 79.06x79.06 으로 바뀌므로 각도의
        // 필요성이 실제로 고정된다.
        let (rx, ry) = (50.0_f64, 25.0_f64);
        let (sin, cos) = (45.0_f64).to_radians().sin_cos();
        let points = (0..32)
            .map(|index| {
                let theta = std::f64::consts::TAU * f64::from(index) / 32.0;
                let x = rx * theta.cos();
                let y = ry * theta.sin();
                koharu_scene::Point {
                    x: 100.0 + x * cos - y * sin,
                    y: 100.0 + x * sin + y * cos,
                }
            })
            .collect::<Vec<_>>();
        let geometry = koharu_scene::Geometry {
            origin: koharu_scene::Origin::User,
            points,
        };

        // 각도를 주면 작성 축에서 재므로 참 크기(2rx x 2ry)가 나온다.
        let frame = crate::shape_renderer::shape_frame(&geometry, Some(45.0)).unwrap();
        assert!((frame.width - 100.0).abs() < 1.0, "width was {}", frame.width);
        assert!((frame.height - 50.0).abs() < 1.0, "height was {}", frame.height);

        // 각도를 주지 않으면 축 정렬 상자가 되어 100x50 이 79.06x79.06 으로
        // 바뀐다 — 정사각형이 되므로 폭은 줄고 높이는 늘어난다.
        let without = crate::shape_renderer::shape_frame(&geometry, None).unwrap();
        assert!((without.width - 79.06).abs() < 0.5, "width was {}", without.width);
        assert!((without.height - 79.06).abs() < 0.5, "height was {}", without.height);
        assert!(without.height > 60.0, "height was {}", without.height);
    }
```

**이 테스트의 수학은 리뷰 중 검증됐다.** 회전한 타원의 축 정렬 경계 상자 폭은
`2·√(rx²cos²θ + ry²sin²θ)`이고 높이는 `2·√(rx²sin²θ + ry²cos²θ)`이다.
rx=50, ry=25, θ=45° 에서 둘 다 79.0569 이다. 원(rx=ry)이면 두 식 모두
지름과 같아져 회전의 영향이 전혀 보이지 않는다.

```rust
    #[test]
    fn a_shape_without_an_angle_falls_back_to_axis_aligned_bounds() {
        let geometry = koharu_scene::Geometry {
            origin: koharu_scene::Origin::User,
            points: vec![
                koharu_scene::Point { x: 10.0, y: 20.0 },
                koharu_scene::Point { x: 60.0, y: 20.0 },
                koharu_scene::Point { x: 60.0, y: 50.0 },
            ],
        };
        let frame = crate::shape_renderer::shape_frame(&geometry, None).unwrap();
        assert!((frame.width - 50.0).abs() < 1e-6);
        assert!((frame.height - 30.0).abs() < 1e-6);
        assert!((frame.angle_degrees - 0.0).abs() < 1e-6);
    }
```

**이 두 테스트는 `shape_renderer::shape_frame`(Task 3)을 쓴다.** `bubble::geometry_frame`과 `bubble::GeometryFrame`은 `pub(crate)`이므로 `koharu-renderer` 안에서는 쓸 수 있지만, Task 3이 이미 공개 래퍼를 만들었으므로 그쪽이 정확한 계약이다. `GeometryPoint`라는 타입은 없다 — `koharu_scene::Point`가 맞다.

- [ ] **Step 2: 실패를 확인한다**

Run: `cargo test -j 6 -p koharu-renderer --lib frame::tests`
Expected: 컴파일 실패 또는 두 번째 테스트 실패

- [ ] **Step 3: `LayerKind::Shape`를 추가한다**

`crates/koharu-rasterizer/src/prepared.rs:458`의 `LayerKind`에 `Shape`를 추가하고, `:441-453`의 `validate` match에 `(LayerKind::Shape, PreparedContent::Vector(scene)) => scene.validate(resources, referenced)?` 분기를 추가한다.

`crates/koharu-renderer/src/frame.rs:322`의 `LayerKind`에 `Shape(ShapeMetadata)`를 추가하고 위 인터페이스대로 `ShapeMetadata`를 정의한다.

- [ ] **Step 4: `element_frame`에 도형 분기를 추가한다**

`frame.rs:602`를 바꾼다:

```rust
fn element_frame(layer: &Layer) -> Option<PreparedElementFrame> {
    match layer.kind() {
        LayerKind::Text(text) => {
            let bounds = text.rendered_bounds;
            if bounds.width > 0.0 && bounds.height > 0.0 {
                return Some(PreparedElementFrame { /* 기존 그대로 */ });
            }
            geometry_frame(layer.geometry())
        }
        // 도형은 각도를 넘겨야 회전한 비원형 타원이 참 크기를 유지한다.
        // shape_frame이 그 변환을 캡슐화한다 (Task 3).
        LayerKind::Shape(shape) => {
            crate::shape_renderer::shape_frame(layer.geometry(), shape.angle_degrees)
                .map(|frame| PreparedElementFrame {
                    x: frame.x,
                    y: frame.y,
                    width: frame.width,
                    height: frame.height,
                    angle_degrees: frame.angle_degrees,
                })
        }
        LayerKind::Image(_) => None,
    }
}
```

기존 `geometry_frame`(frame.rs:619)은 그대로 둔다.

- [ ] **Step 5: `prepare_frame`·`NodeDescriptor`·`DraftMetadata`·`assemble_frame`·`build_node`를 배선한다**

`frame.rs:474`의 `prepare_frame` 분기에 도형 갈래를 추가한다. `ShapeLayer` 컴포넌트를 읽어 `PreparedLayerKind::Shape` + `PreparedContent::Vector(scene)`를 만든다. `scene`은 **`shape_layer_path`** 가 만든 `PreparedPath` 하나를 `PreparedSceneCommand::FillPath`로 담는다.

`Polygon`/`Freeform`은 도형 파라미터에 점이 없으므로 `Geometry` 컴포넌트에서 폴리곤을 읽어야 한다. 이 변환은 Task 7도 필요로 하므로 **`shape_renderer`가 `polygon_path`를 두고 양쪽이 그것을 쓴다.** 폴리곤이 3점 미만이면 빈 `PreparedScene`을 만들고 `tracing::warn!`을 남긴다.

`shape_path`와 `polygon_path`를 나누는 계약: `shape_path`는 `Rectangle`·`Ellipse`·`Line` 전용이고, `Polygon`·`Freeform`에서는 `polygon_path`를 쓴다. 호출자가 어느 것을 쓸지 고르지 않도록 **`shape_renderer`에 두 함수를 합친 진입점을 둔다**:

```rust
pub fn shape_layer_path(
    shape: &ShapeGeometry,
    points: &[koharu_scene::Point],
    style: &ShapeStyle,
    frame: ShapeFrame,
) -> PreparedPath
```

이 함수가 `Polygon`/`Freeform`이면 `points`로, 나머지는 파라미터로 경로를 만든다. **Task 4와 Task 7은 이 하나만 호출한다.** `shape_path`를 직접 부르면 `Polygon`/`Freeform`이 빈 경로가 된다. Task 3의 테스트는 `shape_path`를 직접 부르고, `shape_layer_path`는 `Polygon` 위임을 검증하는 테스트를 갖는다.

`rasterized == true`인 도형은 벡터 대신 래스터 자산을 그려야 한다. `ShapeLayer.rasterized`가 true면 `PreparedLayerKind::Raster` + 래스터 내용으로 내보낸다 (래스터 자산은 Task 7이 만든다). 래스터 내용을 만드는 방법은 `prepare_frame`의 기존 래스터 갈래(`frame.rs:486-490`)가 `RasterImage`를 받아 `prepare_raster_tiles`를 부르는 것을 따른다. 이 분기를 `prepare_frame`에 넣고 주석으로 이유를 남긴다.

`NodeDescriptor`(frame.rs:442)에 `Shape(Box<ShapeNodeDescriptor>)`를 추가한다. `ShapeNodeDescriptor`는 폴리곤 점도 담아야 `build_node`가 `shape_layer_path`에 넘길 수 있다. `renderer.rs:1100` 부근의 `build_node`에 `NodeDescriptor::Shape` 갈래를 추가해 **`shape_layer_path`** 를 호출하고 `RetainedNode { descriptor, scene, resources: Arc::from([]), local_bounds, image: None, text: None, diagnostics }`를 만든다.

`renderer.rs:468`의 `DraftMetadata`에 `Shape(ShapeMetadata)`를 추가하고, `assemble_frame`(renderer.rs:1193)의 match에 `LayerKind::Shape(metadata)` 갈래를 추가한다.

`renderer.rs`에서 `LayerKind`를 match하는 나머지 지점(`:2131`, `:2134`)은 테스트 헬퍼로 보이므로 컴파일러가 요구하는 대로 갈래를 추가한다.

- [ ] **Step 6: 테스트가 통과하는지 확인한다**

Run: `cargo test -j 6 -p koharu-renderer --lib`
Expected: 전부 통과

- [ ] **Step 7: 래스터라이저와 PSD가 컴파일되는지 확인한다**

Run: `cargo check -j 6 -p koharu-rasterizer -p koharu-psd`
Expected: 성공, 새 경고 없음

- [ ] **Step 8: 커밋**

```bash
git add crates/koharu-renderer/src/frame.rs crates/koharu-renderer/src/renderer.rs crates/koharu-rasterizer/src/prepared.rs
git commit -m "feat(renderer): render shape layers and restore their frames"
```

---

### Task 5: 변형 게이트와 WASM 미리보기

도형을 선택·이동·회전할 수 있게 한다. 이 태스크가 없으면 도형이 화면에만 보이고 손댈 수 없다.

**Files:**
- Modify: `crates/koharu-desktop/src/lib.rs:266`
- Modify: `crates/koharu-canvas/src/web.rs:1395`
- Modify: `crates/koharu-canvas/src/web.rs:202-212, 458-500, 787-886` (`ShapeEdit`)
- Test: `crates/koharu-canvas` (WASM 타깃이라 테스트가 어려우면 `koharu-desktop`에 게이트 테스트를 둔다)

**Interfaces:**
- Consumes: Task 4의 `LayerKind::Shape`
- Produces: `transform_geometries`가 도형을 받아들인다. `WebCanvas`에 `beginShape`/`updateShape`/`finishShape`/`cancelShape` (또는 기존 `beginStroke` 계열에 도형 종류 추가)

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`crates/koharu-desktop/src/lib.rs`의 테스트 모듈(있으면)에, 또는 가장 가까운 통합 테스트에 추가한다. `transform_geometries`가 프레임을 준비해야 하므로 순수 단위 테스트가 어렵다. 대신 **게이트 자체를 작은 함수로 뽑아** 테스트한다:

```rust
    #[test]
    fn shape_and_text_layers_are_transformable() {
        assert!(is_transformable(&LayerKind::Text(/* 가능한 값 */)));
        assert!(is_transformable(&LayerKind::Shape(/* 가능한 값 */)));
        assert!(!is_transformable(&LayerKind::Image(/* 가능한 값 */)));
    }
```

`is_transformable(LayerKind) -> bool`을 `koharu-desktop`에 `pub(crate) fn`으로 두고 `transform_geometries`가 그것을 쓰게 한다. `LayerKind` 값을 만드는 비용이 크면 `matches!` 매크로를 그대로 두고 이 테스트를 생략한다 — 그 경우 Step 4에서 `cargo check`로만 확인한다.

- [ ] **Step 2: 실패를 확인한다**

Run: `cargo test -j 6 -p koharu-desktop --lib is_transformable`
Expected: 컴파일 실패

- [ ] **Step 3: 두 게이트를 연다**

`crates/koharu-desktop/src/lib.rs:266`의 `!matches!(layer.kind(), LayerKind::Text(_))`를 `Text` 또는 `Shape`를 허용하도록 바꾼다.

`crates/koharu-canvas/src/web.rs:1395`의 `layer.kind() != LayerKind::Text`도 같은 방식으로 바꾼다.

- [ ] **Step 4: 도형 미리보기를 추가한다**

`CanvasState`(web.rs:458)에 `shape: Option<ShapeEdit>`를 추가한다. `ShapeEdit`은 `page`, `revision`, `geometry: koharu_rasterizer::PreparedScene`(미리보기용 vello 씬), `pending: bool`을 갖는다. `StrokeEdit`(web.rs:202)의 구조를 따른다.

`composition`(web.rs:854)의 stroke 블록 뒤에 같은 패턴으로 `shape` 블록을 추가한다. `begin_transform`/`begin_stroke`의 상호 배제 검사(web.rs:1373, 1524)에 `shape.is_some()`을 추가한다.

`begin_shape`/`update_shape`/`finish_shape`/`cancel_shape`를 `#[wasm_bindgen]`으로 노출한다. 미리보기 씬은 프론트가 넘긴 `PathElement` 목록을 그대로 받아 만든다 — 도형 종류별 지오메트리는 프론트가 이미 알고 있으므로 브라우저에서 다시 계산하지 않는다.

- [ ] **Step 5: 컴파일을 확인한다**

Run: `cargo check -j 6 -p koharu-desktop -p koharu-canvas --target wasm32-unknown-unknown`
Expected: 성공 (wasm 타깃이 없으면 `cargo check -j 6 -p koharu-desktop`만)

- [ ] **Step 6: 커밋**

```bash
git add crates/koharu-desktop/src/lib.rs crates/koharu-canvas/src/web.rs
git commit -m "feat(canvas): allow transforming and previewing shapes"
```

---

### Task 6: 커맨드와 전송 계층

도형 생성·스타일 편집·기하 편집을 서버에 연결하고 RPC 라우트를 함께 만든다.

**Files:**
- Modify: `crates/koharu-app/src/commands/canvas.rs` (`add_shape`, `set_shape_style`, `set_shape_geometries`)
- Modify: `crates/koharu-app/src/commands/project.rs` (`Project::add_shape`, `::set_shape_style`, `::set_shape_geometries`, `Layer::Shape` 노출)
- Modify: `crates/koharu-rpc/src/routes/canvas.rs`
- Modify: `packages/bridge/src/protocol.ts`
- Test: `crates/koharu-app/src/commands/project.rs` (인라인 `mod tests`)

**Interfaces:**
- Consumes: Task 1의 `ShapeLayer` 등, `Project::commit`, `ensure_revision`
- Produces (Rust):
  - `Project::add_shape(&mut self, page: EntityId, geometry: SceneGeometry, shape: ShapeGeometry, style: ShapeStyle, angle_degrees: f32) -> Result<(Commit, EntityId)>`
  - `Project::set_shape_style(&mut self, updates: Vec<(EntityId, ShapeStyle)>) -> Result<Commit>`
  - `Project::set_shape_geometries(&mut self, updates: Vec<(EntityId, SceneGeometry, f32)>) -> Result<Commit>`
  - `Layer::Shape { id, parent, geometry, angle_degrees, visibility, shape, style, rasterized }`
  - 커맨드: `add_shape(expected_revision, page, geometry, shape, style, angle_degrees) -> LayerCommit`
  - 커맨드: `set_shape_style(expected_revision, updates: Vec<ShapeStyleUpdate>) -> Revision`
  - 커맨드: `set_shape_geometries(expected_revision, updates: Vec<ShapeFrameUpdate>) -> Revision`
- Produces (TS, `protocol.ts`):
  - `ShapeGeometry`, `ShapeStyle`, `ShapeFill`, `StrokeStyle`, `LineCap`, `LineJoin`, `ArrowHead` 타입
  - `Layer`에 `{ type: 'shape'; ... }` 변형
  - `commands.addShape(expectedRevision, page, geometry, shape, style, angleDegrees)`
  - `commands.setShapeStyle(expectedRevision, updates)`
  - `commands.setShapeGeometries(expectedRevision, updates)`
  - 라우트: `POST /canvas/shape`, `POST /canvas/shape/style`, `POST /canvas/shape/geometry`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`crates/koharu-app/src/commands/project.rs`의 `mod tests`에 추가한다. 기존 테스트가 프로젝트를 만드는 헬퍼를 쓰는지 확인해 맞춘다.

```rust
    #[tokio::test]
    async fn a_shape_layer_round_trips_through_the_page_view() {
        let mut project = /* 기존 헬퍼로 프로젝트와 페이지를 만든다 */;
        let page = /* 페이지 id */;
        let geometry = SceneGeometry::rectangle(10.0, 20.0, 100.0, 50.0);
        let style = ShapeStyle {
            fill: Some(ShapeFill::Solid([1, 2, 3, 255])),
            stroke: None,
        };

        let (commit, element) = project
            .add_shape(page, geometry, ShapeGeometry::Rectangle { corner_radius: 0.0 }, style.clone(), 0.0)
            .await
            .unwrap();
        assert!(commit.revision.get() > 0);

        let view = Project::page(project.snapshot(), page).unwrap();
        let layer = view.layers.iter().find(|layer| layer.id == element).unwrap();
        let Layer::Shape { shape, style: view_style, angle_degrees, rasterized, .. } = layer else {
            panic!("expected a shape layer");
        };
        assert!(matches!(shape, ShapeGeometry::Rectangle { .. }));
        assert_eq!(*angle_degrees, Some(0.0));
        assert!(!rasterized);
        assert_eq!(view_style.fill, Some(ShapeFill::Solid([1, 2, 3, 255])));
    }

    #[tokio::test]
    async fn changing_shape_style_keeps_the_geometry() {
        let mut project = /* ... */;
        let page = /* ... */;
        let (_, element) = project
            .add_shape(page, SceneGeometry::rectangle(0.0, 0.0, 10.0, 10.0),
                ShapeGeometry::Ellipse,
                ShapeStyle { fill: Some(ShapeFill::Solid([0, 0, 0, 255])), stroke: None },
                0.0)
            .await
            .unwrap();
        let next = ShapeStyle { fill: None, stroke: None };
        project.set_shape_style(vec![(element, next.clone())]).await.unwrap();

        let view = Project::page(project.snapshot(), page).unwrap();
        let Layer::Shape { style, shape, .. } =
            view.layers.iter().find(|layer| layer.id == element).unwrap() else {
                panic!("expected a shape layer");
            };
        assert_eq!(*style, next);
        assert!(matches!(shape, ShapeGeometry::Ellipse));
    }
```

- [ ] **Step 2: 실패를 확인한다**

Run: `cargo test -j 6 -p koharu-app --lib commands::project::tests`
Expected: 컴파일 실패

- [ ] **Step 3: `Project` 메서드를 구현한다**

`add_shape`는 `add_manual_text_region`(project.rs:782)과 같은 모양으로 patch를 만든다. 차이는 `SceneRegion`·`TextContent` 대신 `SceneShapeLayer`를 넣고 `SceneGeometry`를 같은 엔티티에 붙이며 `TextGroup` 앞에 놓는 것이다.

이름은 기존 래스터 이름 규칙(project.rs:1193)을 따른다: `format!("{} {}", 종류 이름, 기존 도형 수 + 1)`. 종류 이름은 `Rectangle`/`Ellipse`/`Line`/`Polygon`/`Freeform`.

`set_shape_style`은 `set_typography`(project.rs:907)를 따른다. `set_shape_geometries`는 `set_geometries`(project.rs:1104)를 따르되 `SceneTextLayout` 대신 `SceneShapeLayer.angle_degrees`를 갱신한다.

**`set_shape_geometries`는 `Line`의 `start`/`end`를 폴리곤에서 다시 계산해야 한다.** 모든 변형 경로는 폴리곤만 갱신하는데 `line_path`는 `start`/`end`로 그리므로, 재계산하지 않으면 선을 드래그해도 그려지는 선이 제자리에 남고 `element_frame`(폴리곤을 읽는다)이 선에서 분리된다. Task 3이 넣은 `ShapeGeometry::resync_with_polygon(&mut self, points: &[Point])`를 `edit.set(element, &geometry)` 직전에 호출한다 — 첫 점이 `start`, 마지막 점이 `end`다. 이 호출을 검증하는 테스트를 Step 1에 추가한다: 선 폴리곤을 평행이동한 뒤 컴포넌트의 `start`/`end`가 따라 움직였는지 확인한다.

세 메서드 모두 `Origin::User`로 승격하고 `promote_entity_to_user`를 호출한다.

- [ ] **Step 4: `Layer::Shape` 노출을 추가한다**

`project.rs:77`의 `Layer` enum에 `Shape` 변형을 추가하고, `collect_layer_views`(project.rs:1541 부근)에 `SceneShapeLayer` 분기를 추가한다. 순서는 `Raster` 검사 뒤, `Image` 폴백 앞이다.

`angle_degrees`는 `ShapeLayer.angle_degrees`를 그대로 내보내고, `rasterized`도 그대로 내보낸다.

- [ ] **Step 5: 커맨드 3개를 추가한다**

`crates/koharu-app/src/commands/canvas.rs`에 `commit_paint`(canvas.rs:288)를 본떠 커맨드를 만든다. 각 커맨드는 `#[tracing::instrument(target = "koharu_metrics", ...)]`, `#[tauri::command]`, `#[specta::specta]`를 붙이고 `ensure_revision`으로 기대 리비전을 확인한 뒤 커밋하고 `desktop.synchronize` + `canvas_channel.publish`를 호출한다.

`ShapeStyleUpdate { element: EntityId, style: ShapeStyle }`, `ShapeFrameUpdate { element: EntityId, geometry: Geometry, angle_degrees: f32 }`를 `#[derive(Deserialize, Type)]`로 정의한다.

- [ ] **Step 6: 테스트가 통과하는지 확인한다**

Run: `cargo test -j 6 -p koharu-app --lib commands::project::tests`
Expected: 전부 통과

- [ ] **Step 7: RPC 라우트를 추가한다**

`crates/koharu-rpc/src/routes/canvas.rs:20`의 `router()`에 세 라우트를 추가하고 핸들러를 만든다. 기존 `add_text_box` 핸들러가 상태를 꺼내는 방식을 그대로 따른다.

- [ ] **Step 8: `protocol.ts`를 갱신한다**

`packages/bridge/src/protocol.ts`에 도형 타입과 세 명령을 추가한다. `Layer` 유니온(protocol.ts:686)에 `shape` 변형을 넣는다. 기존 `RasterLayerKind`(protocol.ts:956) 옆에 도형 관련 타입을 둔다.

`commands` 객체(protocol.ts:272 부근)에:

```ts
	addShape: (expectedRevision: Revision, page: EntityId, geometry: Geometry, shape: ShapeGeometry, style: ShapeStyle, angleDegrees: number) =>
		post<LayerCommit>("/canvas/shape", {
			expected_revision: expectedRevision,
			page,
			geometry,
			shape,
			style,
			angle_degrees: angleDegrees,
		}),
	setShapeStyle: (expectedRevision: Revision, updates: ShapeStyleUpdate[]) =>
		post<Revision>("/canvas/shape/style", { expected_revision: expectedRevision, updates }),
	setShapeGeometries: (expectedRevision: Revision, updates: ShapeFrameUpdate[]) =>
		post<Revision>("/canvas/shape/geometry", { expected_revision: expectedRevision, updates }),
```

**파일 상단 주석에 "This file is derived; change the Rust source and regenerate"가 있으면 반드시 확인한다.** 포크 HANDOFF §8에 따르면 이 파일은 지금 손으로 쓰는 fetch/SSE 클라이언트이며, 자동 생성기가 아니다. 자동 생성기가 살아 있으면 생성하고, 아니면 손으로 고친다.

- [ ] **Step 9: `tsc`로 확인한다**

Run: `bunx tsc --noEmit -p packages/koharu/tsconfig.json`
Expected: exit 0

- [ ] **Step 10: 커밋**

```bash
git add crates/koharu-app/src/commands/canvas.rs crates/koharu-app/src/commands/project.rs crates/koharu-rpc/src/routes/canvas.rs packages/bridge/src/protocol.ts
git commit -m "feat(app): add shape creation and style commands"
```

---

### Task 7: 래스터화 액션

도형을 픽셀로 굽는다. 투명 배경이 이 태스크의 핵심 회귀다.

**Files:**
- Modify: `crates/koharu-app/src/commands/project.rs` (`Project::rasterize_shapes`)
- Modify: `crates/koharu-app/src/commands/canvas.rs` (`rasterize_shapes` 커맨드)
- Modify: `crates/koharu-rpc/src/routes/canvas.rs`
- Modify: `packages/bridge/src/protocol.ts`
- Test: `crates/koharu-app/src/commands/project.rs` (인라인 `mod tests`)

**Interfaces:**
- Consumes: Task 3의 `shape_path`, `Rasterizer::rasterize_scene`, Task 6의 `Layer::Shape`
- Produces:
  - `Project::rasterize_shapes(&mut self, page: EntityId, shapes: Vec<EntityId>) -> Result<Commit>`
  - 커맨드 `rasterize_shapes(expected_revision, page, shapes) -> Revision`
  - 라우트 `POST /canvas/shape/rasterize`
  - TS `commands.rasterizeShapes(expectedRevision, page, shapes)`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```rust
    #[tokio::test]
    async fn a_rasterized_shape_keeps_a_transparent_background() {
        let mut project = /* ... */;
        let page = /* ... */;
        let style = ShapeStyle {
            fill: Some(ShapeFill::Solid([255, 0, 0, 255])),
            stroke: None,
        };
        let (_, element) = project
            .add_shape(page, SceneGeometry::rectangle(10.0, 10.0, 20.0, 20.0),
                ShapeGeometry::Rectangle { corner_radius: 0.0 }, style, 0.0)
            .await
            .unwrap();
        project.rasterize_shapes(page, vec![element]).await.unwrap();

        let snapshot = project.snapshot();
        let asset = snapshot
            .asset(element, &AssetRole::new("source").unwrap())
            .unwrap()
            .expect("rasterized shape must own a source asset");
        let bytes = snapshot.read_blob(asset.blob).await.unwrap();
        let image = image::load_from_memory(&bytes).unwrap().to_rgba8();

        // 도형 밖은 완전히 투명해야 한다.
        assert_eq!(image.get_pixel(0, 0)[3], 0);
        assert_eq!(image.get_pixel(image.width() - 1, image.height() - 1)[3], 0);
        // 도형 안쪽은 불투명한 빨강이어야 한다.
        let center = image.get_pixel(20, 20);
        assert_eq!(center[3], 255);
        assert!(center[0] > 200);

        let Layer::Shape { rasterized, .. } = Project::page(&snapshot, page)
            .unwrap()
            .layers
            .into_iter()
            .find(|layer| layer.id == element)
            .unwrap() else {
                panic!("expected a shape layer");
            };
        assert!(rasterized);
    }
```

- [ ] **Step 2: 실패를 확인한다**

Run: `cargo test -j 6 -p koharu-app --lib commands::project::tests::a_rasterized_shape`
Expected: 컴파일 실패

- [ ] **Step 3: `rasterize_shapes`를 구현한다**

1. 페이지 크기를 읽는다.
2. 각 도형에 대해 `ShapeLayer`와 `Geometry`를 읽고 `koharu_renderer::shape_renderer::shape_frame(&geometry, shape.angle_degrees)`로 프레임을 얻는다. `None`이면 그 도형을 건너뛴다.
3. `koharu_renderer::shape_renderer::shape_layer_path(&shape.geometry, &geometry.points, &shape.style, frame)`으로 `PreparedPath`를 만든다. 이 한 함수가 `Polygon`/`Freeform`의 폴리곤 위임까지 처리한다 (Task 3 계약).
4. `PreparedScene`에 `PreparedSceneCommand::FillPath`로 담고, `Rasterizer::rasterize_scene(&scene, width, height, [0, 0, 0, 0], RasterOptions::supersampled(4))`로 굽는다.
5. PNG로 인코딩해 `AssetRole::new("source")`에 `edit.set_asset`으로 넣는다.
6. `ShapeLayer.rasterized = true`로 갱신한다.

`koharu-app`에는 `koharu-renderer` 의존이 이미 있다 (`crates/koharu-app/Cargo.toml:28`). `shape_path`와 `shape_frame`은 Task 3에서 `pub`으로 만들어야 이 크레이트에서 닿는다.

`apply_raster_stroke`(project.rs:1134)의 blob 저장 부분을 그대로 참고한다. `Rasterizer` 인스턴스는 `koharu-rasterizer`의 `native` 기능을 쓰며 `koharu-app`이 이미 그 기능으로 의존한다 (`crates/koharu-app/Cargo.toml:29`).

**`rasterize_scene`을 쓰는 이유를 주석에 남긴다**: 슈퍼샘플링·Lanczos 다운샘플이 내장되어 있고, 미리보기와 같은 vello 경로라 결과가 일치한다.

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

Run: `cargo test -j 6 -p koharu-app --lib commands::project::tests::a_rasterized_shape`
Expected: PASS

- [ ] **Step 5: 커맨드와 라우트와 TS를 추가한다**

Task 6과 같은 방식으로 `rasterize_shapes` 커맨드, `POST /canvas/shape/rasterize` 라우트, `commands.rasterizeShapes`를 추가한다.

- [ ] **Step 6: 커밋**

```bash
git add crates/koharu-app/src/commands/project.rs crates/koharu-app/src/commands/canvas.rs crates/koharu-rpc/src/routes/canvas.rs packages/bridge/src/protocol.ts
git commit -m "feat(app): rasterize shape layers onto a transparent surface"
```

---

### Task 8: PSD 내보내기

도형이 래스터 레이어로 나가게 한다.

**Files:**
- Modify: `crates/koharu-psd/src/document.rs:107-206`
- Test: `crates/koharu-psd/src/document.rs` 또는 `crates/koharu-psd/tests/`

**Interfaces:**
- Consumes: Task 4의 `LayerKind::Shape`
- Produces: PSD 레이어 목록에 도형이 래스터로 포함된다

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`koharu-psd`의 기존 테스트 방식(스냅샷 + 프레임을 만들어 PSD를 뽑는지, 헬퍼가 있는지)을 먼저 확인한다. 헬퍼가 있으면 도형 레이어를 하나 넣은 페이지로 PSD를 만들고 레이어 이름 목록을 검증한다:

```rust
    #[tokio::test]
    async fn a_shape_is_exported_as_a_raster_layer() {
        let (snapshot, frame, options, rasterizer) = /* 기존 헬퍼 */;
        let bytes = export_page(&snapshot, &frame, &options, rasterizer).await.unwrap();
        let names = psd_layer_names(&bytes);   // 기존 테스트가 쓰는 리더
        assert!(names.iter().any(|name| name.starts_with("Shape")));
    }
```

기존 테스트가 PSD를 다시 읽는 헬퍼를 안 쓰면, `export_page`가 성공하고 `layers` 개수가 도형을 포함해 하나 늘어나는 것으로 검증한다. 정확한 방법을 파일의 기존 테스트에서 확인해 맞춘다.

- [ ] **Step 2: 실패를 확인한다**

Run: `cargo test -j 6 -p koharu-psd`
Expected: 컴파일 실패 (non-exhaustive match)

- [ ] **Step 3: 세 곳을 고친다**

`document.rs`의 세 match에 `LayerKind::Shape(_)` 갈래를 추가한다:

- `:110` (`has_raster_layers` 계산) — 도형은 래스터 레이어 개수에 포함하지 않는다. `matches!(metadata.kind, Cleanup | Paint)` 조건에 `LayerKind::Shape(_)`가 끼지 않도록 `LayerKind::Image(metadata) if ...` 형태를 유지한다. 이 지점은 컴파일러가 요구하지 않을 수 있으니 확인한다.
- `:140` (`text_entities` 수집) — `LayerKind::Shape(_) => None`
- `:150` (`image` 추출) — `LayerKind::Shape(_) => None`
- `:178` (레이어 방출) — `LayerKind::Shape(_)`이면 `frame.cropped(entity)` + `rasterize`로 래스터 레이어를 만든다. 이름은 `format!("Shape {index:03} {}", visual.entity())` 형태로 텍스트 레이어(문서.rs:195)를 따른다.

PSD 벡터 도형(`SoCo`)을 쓰지 않는 이유를 주석에 남긴다: 포맷이 복잡하고, `koharu-psd`는 이미 래스터 경로를 가지며, 텍스트만 편집 가능 레이어로 유지하는 기존 정책과 일관된다.

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

Run: `cargo test -j 6 -p koharu-psd`
Expected: 전부 통과

- [ ] **Step 5: 커밋**

```bash
git add crates/koharu-psd/src/document.rs
git commit -m "feat(psd): export shape layers as raster layers"
```

---

### Task 9: 프론트엔드 도구와 스타일 UI

도형을 실제로 그릴 수 있게 한다.

**Files:**
- Create: `packages/koharu/lib/shapes.ts`
- Create: `packages/koharu/tests/lib/shapes.test.ts`
- Modify: `packages/koharu/lib/store.ts:20-30, 41-44, 68-88, 90-126`
- Modify: `packages/koharu/lib/geometry.ts:61-63`
- Modify: `packages/koharu/components/editor/ToolBar.tsx:44-54, 101-111`
- Modify: `packages/koharu/components/editor/CanvasWorkspace.tsx:395-415, 722-735, 733-756`
- Modify: `packages/koharu/components/editor/Inspector.tsx`
- Modify: `packages/bridge/src/canvas.ts` — Task 5가 노출한 `beginShape`/`updateShape`/`finishShape`/`cancelShape`를 배선한다. 이게 없으면 그 API는 죽은 코드다 (Step 7 참조)
- Modify: `packages/koharu/public/locales/*/translation.json` (9개)

**Interfaces:**
- Consumes: Task 6의 `protocol.ts` 타입과 명령, Task 5의 `WebCanvas` 도형 프리뷰 API
- Produces:
  - `SHAPE_KINDS: readonly ShapeKind[]`, `type ShapeKind = 'rectangle' | 'ellipse' | 'line' | 'arrow' | 'polygon' | 'freeform'`
  - `defaultShapeStyle(kind: ShapeKind): ShapeStyle`
  - `shapePolygon(kind: ShapeKind, frame: Frame): Geometry`
  - `isShapeTool(tool: CanvasTool): boolean` — 이 태스크에서는 `tool === 'shape'`
  - store: `shapeKind: ShapeKind`, `shapeStyle: ShapeStyle`, `setShapeKind`, `setShapeStyle`
  - `CanvasTool`에 `'shape'` 추가, `defaultShortcuts`에 `shape: 'r'`
  - `Canvas` 인터페이스와 `createCanvas` 반환 객체에 `beginShape`/`updateShape`/`finishShape`/`cancelShape`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`packages/koharu/tests/lib/shapes.test.ts`:

```ts
import { describe, expect, it } from 'vitest'

import { defaultShapeStyle, shapePolygon } from '@/lib/shapes'

const frame = { x: 10, y: 20, width: 100, height: 50, angle_degrees: 0 }

describe('shapePolygon', () => {
  it('builds a closed rectangle from the frame corners', () => {
    expect(shapePolygon('rectangle', frame).points).toEqual([
      { x: 10, y: 20 },
      { x: 110, y: 20 },
      { x: 110, y: 70 },
      { x: 10, y: 70 },
    ])
  })

  it('samples an ellipse into at least three points', () => {
    const points = shapePolygon('ellipse', frame).points
    expect(points.length).toBeGreaterThanOrEqual(3)
    expect(points.every((point) => Number.isFinite(point.x) && Number.isFinite(point.y))).toBe(true)
  })

  it('samples a line into three collinear points', () => {
    const points = shapePolygon('line', frame).points
    expect(points.length).toBe(3)
    expect(points[0]).toEqual({ x: 10, y: 20 })
    expect(points[2]).toEqual({ x: 110, y: 70 })
  })

  it('keeps every point finite for a zero-sized frame', () => {
    const points = shapePolygon('ellipse', { ...frame, width: 0, height: 0 }).points
    expect(points.every((point) => Number.isFinite(point.x) && Number.isFinite(point.y))).toBe(true)
  })
})

describe('defaultShapeStyle', () => {
  it('fills without a stroke', () => {
    const style = defaultShapeStyle('rectangle')
    expect(style.fill).not.toBeNull()
    expect(style.stroke).toBeNull()
  })
})
```

- [ ] **Step 2: 실패를 확인한다**

Run: `bun run --filter '@koharu/app' test -- tests/lib/shapes.test.ts`
Expected: FAIL — `Cannot find module '@/lib/shapes'`

- [ ] **Step 3: `lib/shapes.ts`를 구현한다**

`shapePolygon`은 프레임에서 폴리곤을 만든다. 회전은 폴리곤 점에 직접 적용한다 (`geometry_from_frame`의 방식, project.rs:1597):

- `rectangle` — 네 모서리, 프레임 각도로 회전
- `ellipse` — 32점 타원, 같은 방식으로 회전
- `line` / `arrow` — 시작·끝·중간 3점. 시작은 프레임 좌상단, 끝은 우하단
- `polygon` / `freeform` — 프레임 네 모서리에서 시작 (사용자가 이후 편집)

프레임 크기가 0이어도 점이 유한해야 한다. `width`/`height`가 0이면 최소 `1e-6`으로 클램프한다.

`defaultShapeStyle`은 **면적이 있는 도형**(`rectangle`·`ellipse`·`polygon`·`freeform`)에 대해 `{ fill: { solid: [255, 255, 255, 255] }, stroke: null }`을 반환하고, **Task 6의 `protocol.ts`에 정의한 실제 직렬화 형태와 정확히 일치**해야 한다.

**`line`·`arrow`는 예외다.** 직선분은 면적을 둘러싸지 않으므로 채우기만 있는 스타일은 아무것도 그리지 않는다 — 기본값으로 두면 사용자가 선을 그어도 화면에 아무것도 나타나지 않는다. 이 두 종류는 `{ fill: null, stroke: { color: [255,255,255,255], width: 2, dash: null, cap: 'round', join: 'round' } }`을 반환한다.

처음에 이 계획은 여섯 종류 모두에 채우기 기본값을 요구했으나, 그 제약은 serde가 고정한 **wire 형태**(면적 도형의 채우기 리터럴)를 모든 종류로 과잉 일반화한 것이었다. wire 형태는 어느 쪽이든 동일하므로 두 종류의 예외는 직렬화 계약을 깨지 않는다.

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

Run: `bun run --filter '@koharu/app' test -- tests/lib/shapes.test.ts`
Expected: PASS

- [ ] **Step 5: store와 geometry를 넓힌다**

`store.ts:20`의 `CanvasTool`에 `'shape'`를 추가한다. `isBrushTool`(store.ts:37)은 그대로 둔다 (도형은 브러시 크기를 쓰지 않는다).

`defaultShortcuts`(store.ts:90)에 `shape: 'r'`을 추가한다. `store.ts:68`의 인터페이스와 `:120`의 초기값에 `shapeKind: 'rectangle'`, `shapeStyle: defaultShapeStyle('rectangle')`을 추가하고 `setShapeKind`/`setShapeStyle` 액션을 만든다.

`geometry.ts:61`의 `selectableLayer`에 `layer.type === 'shape'`를 추가한다. **이걸 빠뜨리면 도형을 선택할 수 없다.**

- [ ] **Step 6: 도구막대를 넣는다**

`ToolBar.tsx:44`의 `tools` 배열에 `['shape', Square]`를 넣는다 (`lucide-react`의 `Square`). 도구 순서는 `select`·`text` 다음, `draw` 앞이 자연스럽다.

`ToolBar.tsx:101`의 브러시 블록 뒤에 도형 블록을 추가한다: 종류 선택(6종)과 채우기 색 `ColorWell`, 테두리 두께 `NumberField`. 브러시 블록과 같은 Popover 패턴을 쓴다.

- [ ] **Step 7: 캔버스 제스처를 넣는다**

`CanvasWorkspace.tsx:395`의 Escape·단축키 배열에 `'shape'`를 추가한다.

`CanvasWorkspace.tsx:733`의 `'draw'` 분기 앞에 `'shape'` 분기를 넣는다. `'manual_box'`(CanvasWorkspace.tsx:726)와 같은 `draftFrame` 드래그를 쓴다.

**드래그 중 라이브 미리보기는 `packages/bridge/src/canvas.ts`를 거쳐야 한다.** Task 5가 `WebCanvas`에 `beginShape`/`updateShape`/`finishShape`/`cancelShape`를 노출했지만, `canvas.ts`가 WASM 핸들의 유일한 래퍼이고 여기에 네 메서드가 없으면 그 API는 도달 불가능한 죽은 코드가 된다 (Task 5 리뷰가 이 계획 누락을 지적했다). 따라서:

1. `canvas.ts`의 `Canvas` 인터페이스(~:17-38)와 `CanvasHandle` 인터페이스(~:51-79)에 네 메서드를 추가한다. `CanvasHandle`의 `beginShape`는 `PathElement[]`(또는 그것을 담은 직렬화된 `PreparedPath`)와 페이지·리비전을 받는다.
2. `createCanvas`의 반환 객체(~:103-180)에 네 메서드를 배선한다. `stroke` 계열(~:151-161)이 참고 패턴이다.
3. `CanvasWorkspace.tsx`의 도형 분기가 드래그 시작 시 `canvas.beginShape({...})`, 이동 시 `updateShape`, 포인터 업 시 `finishShape`를 호출한다. `strokeUpdates`(~:162-165)의 `useFrameCommand` 스로틀 패턴을 그대로 쓴다 — 매 프레임 전부 보내면 안 된다.
4. 취소(Escape·blur) 경로는 `cancelGesture`(~:231-248)에 도형 갈래를 추가해 `cancelShape`를 부른다. 기존 `strokeUpdates.clear(); canvas?.cancelStroke()` 옆이다.
5. 포인터 업에서는 `finishShape()`가 돌려준 `{page, revision}` 검증에만 쓰고, **커밋은 `commands.addShape`로 한다** — 프론트가 스타일과 종류를 이미 알고 있으므로 경로를 되돌려받아 서버에 보낼 필요가 없다.

`gesture.current`의 유니온(CanvasWorkspace.tsx:82 부근)에 `kind: 'shape'`를 추가한다.

**미리보기 없이 `commands.addShape`만 부르면 안 된다.** 도형은 드래그하는 동안 모양이 보여야 하고, 두 방식이 공존하면 "왜 안 보이나"를 나중에 디버깅하게 된다. Step 5의 `draftFrame` 윤곽(`DraftOverlay`)만으로도 최소 피드백은 있지만, 실제 채우기·테두리를 보여주는 것이 Task 5가 프리뷰를 만든 이유다.

- [ ] **Step 8: Inspector와 i18n을 넣는다**

`Inspector.tsx`에 선택된 레이어가 `type === 'shape'`일 때 스타일 섹션을 렌더링한다. 기존 텍스트 섹션(Inspector.tsx:120 부근)의 `InspectorField` 패턴을 따른다: 채우기 색, 테두리 색, 테두리 두께, 점선, 래스터화 버튼.

9개 로케일 파일에 키를 추가한다 (`packages/koharu/public/locales/*/translation.json`):

- `tools.shape`
- `inspector.shape.fill`, `inspector.shape.stroke`, `inspector.shape.strokeWidth`, `inspector.shape.dash`, `inspector.shape.rasterize`
- `inspector.shape.kind.rectangle`, `.ellipse`, `.line`, `.arrow`, `.polygon`, `.freeform`

한국어(`ko-KR`)는 사각형/타원/선/화살표/다각형/자유곡선. 다른 로케일은 영어를 기본으로 두고 기존 파일의 번역 관행을 따른다.

- [ ] **Step 9: 프론트 검사를 돌린다**

Run: `bunx tsc --noEmit -p packages/koharu/tsconfig.json`
Expected: exit 0

Run: `bunx oxlint -c packages/koharu/.oxlintrc.json packages/koharu`
Expected: clean

Run: `bun run --filter '@koharu/app' test`
Expected: 전부 통과

- [ ] **Step 10: 커밋**

```bash
git add packages/koharu/lib/shapes.ts packages/koharu/tests/lib/shapes.test.ts packages/koharu/lib/store.ts packages/koharu/lib/geometry.ts packages/koharu/components/editor/ToolBar.tsx packages/koharu/components/editor/CanvasWorkspace.tsx packages/koharu/components/editor/Inspector.tsx packages/koharu/public/locales
git commit -m "feat(ui): draw shapes with a style toolbar and inspector"
```

---

### Task 10: 회귀 확인

기존 프로젝트가 그대로 열리는지 실제로 확인한다. 스펙의 호환 주장을 테스트로 못 박는다.

**Files:**
- Test: `crates/koharu-scene/src/tests.rs` 또는 `crates/koharu-storage/src/tests.rs`

**Interfaces:**
- Consumes: Task 1의 `ShapeLayer`, 기존 `Session`
- Produces: 없음 (검증만)

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`crates/koharu-scene/src/tests.rs`에 추가한다. 기존 테스트가 `Session::memory()`로 스냅샷을 만든다 (tests.rs:68).

```rust
#[tokio::test]
async fn a_scene_without_shapes_round_trips_after_the_component_is_registered() {
    let mut session = Session::memory().await.unwrap();
    // 도형 없는 평범한 페이지를 만든다.
    let page = {
        let snapshot = session.snapshot();
        let patch = snapshot
            .patch(|edit| {
                let page = edit.add_entity(/* 프로젝트 루트 */, At::End)?;
                // 기존 테스트가 페이지를 만드는 방식을 그대로 따른다.
                Ok(())
            })
            .unwrap();
        session.commit(patch).await.unwrap();
        /* page id */
    };

    // 저장했다가 다시 읽는 경로를 탄다.
    let encoded = /* 기존 테스트가 쓰는 인코딩 경로 */;
    let decoded = /* 기존 테스트가 쓰는 디코딩 경로 */;
    assert_eq!(decoded.revision(), encoded.revision());
    assert!(decoded.component::<ShapeLayer>(page).unwrap().is_none());
}
```

기존 테스트에 인코딩/디코딩 왕복 헬퍼가 없으면 `crates/koharu-storage/src/tests.rs`의 세션 저장 왕복 테스트를 참고해 같은 방식으로 만든다. 순수 스냅샷만으로는 이 회귀를 잡지 못한다 — 저장 포맷을 실제로 통과해야 한다.

- [ ] **Step 2: 실패를 확인한다**

Run: `cargo test -j 6 -p koharu-scene --lib a_scene_without_shapes`
Expected: 헬퍼가 없으면 컴파일 실패, 있으면 통과 (통과하면 그대로 둔다 — 회귀가 없다는 증거다)

- [ ] **Step 3: 통과시킨다**

테스트가 컴파일되지 않으면 저장 왕복 헬퍼를 만든다. 스키마 변경이 필요하면 여기서 발견한 것이므로 Task 1로 돌아가 고친다.

- [ ] **Step 4: 전체 관련 검사를 한 번에 돌린다**

Run: `cargo test -j 6 -p koharu-scene -p koharu-rasterizer -p koharu-renderer -p koharu-app -p koharu-psd --lib`
Expected: 전부 통과

Run: `cargo check -j 6 --workspace`
Expected: 성공, 새 경고 없음

Run: `bun run --filter '@koharu/app' test`
Expected: 전부 통과

`-j 6`은 이 머신에서 필수다 (12코어, 무제한 병렬은 데스크톱을 멈춘다 — `fork/HANDOFF.md:44`).

- [ ] **Step 5: 커밋**

```bash
git add crates/koharu-scene/src/tests.rs
git commit -m "test(scene): pin project round-trip compatibility for shapes"
```

---

## Self-Review

**Spec coverage**

| 스펙 절 | 태스크 |
|---|---|
| 저장 모델 (`ShapeLayer`, 종류, 스타일, 검증) | Task 1 |
| 폴리곤 점 개수 (`Line` 3점) | Task 1(타입) + Task 9 (`shapePolygon`) |
| `PreparedPath` 확장 | Task 2 |
| `PreparedLayerKind::Shape` | Task 4 |
| `LayerKind::Shape` (렌더러) + `ShapeMetadata` | Task 4 |
| `element_frame` early-return | Task 4 |
| "무엇이 그리는가" (파라미터가 그린다) | Task 3 |
| "프레임을 어떻게 얻는가" (회전) | Task 4 |
| 캔버스 미리보기 (WASM) | Task 5 |
| 새 커맨드 4개 | Task 6 (`add_shape`, `set_shape_style`, `set_shape_geometries`), Task 7 (`rasterize_shapes`) |
| HTTP 라우트 | Task 6, Task 7 |
| 변형 게이트 3곳 | Task 5 (desktop, canvas), Task 6 (`set_shape_geometries` 분리) |
| 래스터화 흐름 + 알파 배경 | Task 7 |
| 렌더링 우선순위 (`rasterized`) | Task 4 (`prepare_frame` 분기) |
| 기존 프로젝트 호환 | Task 10 |
| PSD | Task 8 |
| UI (도구·store·Inspector·i18n) | Task 9 |
| 테스트 표 | 각 태스크 + Task 10 |
| 실패 시나리오 | Task 1 (검증), Task 3 (0길이 선), Task 4 (회전 폴백), Task 7 (배경) |

**Spec gap을 계획이 메운 곳** (스펙에 없던 결정):

1. `LineCap`/`LineJoin`은 `koharu-scene`에 자체 정의한다. 스펙은 이름만 썼고 소속을 정하지 않았다. `koharu-scene`이 kurbo/vello에 의존하지 않으므로 vello 타입을 쓸 수 없다. 렌더러 경계에서 `PreparedLineCap`/`PreparedLineJoin`으로 매핑한다 (Task 1, Task 3).
2. `Polygon`/`Freeform`은 `ShapeGeometry`에 데이터가 없으므로 `shape_path`가 폴리곤을 받지 못한다. 호출자가 폴리곤에서 직접 경로를 만드는 계약으로 정했다 (Task 3, Task 4).
3. 화살촉은 별도 `PreparedPath`가 아니라 같은 경로의 subpath로 넣고 `fill`을 켠다. `PreparedSceneCommand::FillPath`가 하나의 `PreparedPath`만 받기 때문이다 (Task 3).
4. `rasterized == true`일 때 `prepare_frame`이 래스터로 내보낸다 — 스펙의 "렌더링 우선순위"를 구현하는 지점이 어디인지 명시되지 않았다 (Task 4).

**Type consistency 확인**

- `ShapeGeometry::Line`의 필드는 `start`/`end`/`head`로 Task 1·3·9에서 동일하다.
- `ShapeFill::Solid([u8;4])`는 Rust에서 튜플 변형, TS에서 `{ kind: 'solid', color: [...] }`로 직렬화된다. Task 6이 `protocol.ts` 타입을 정할 때 이 차이를 흡수해야 한다 — **Task 9 Step 3의 `defaultShapeStyle`이 Task 6의 실제 형태와 일치하는지 반드시 대조한다.**
- `shape_path`·`polygon_path`·`shape_layer_path`·`shape_frame`은 모두 `pub`이다. Task 7의 `koharu-app`이 `shape_layer_path`와 `shape_frame`을 쓴다.
- `ShapeFrame`은 `koharu-renderer`의 공개 타입이고, `bubble::GeometryFrame`(`pub(crate)`)과 별개다. `shape_frame`이 둘 사이를 옮긴다.
- `ShapeLayer` 필드 이름은 Task 1·6·7에서 `rasterized`, `angle_degrees`로 동일하다.

**작성 중 고친 계획의 오류** (실제 코드를 확인해 발견):

1. Task 1 테스트가 `ValidationContext::default()`를 썼지만 그 타입에 `Default`가 없다. `new`는 `pub(crate)`이고 두 클로저를 받는다 — `crates/koharu-scene/src/component.rs:16`, 관용구는 `crates/koharu-scene/src/tests.rs:205-207`. `validate` 헬퍼로 고쳤다.
2. Task 4 테스트가 `crate::GeometryPoint`를 썼지만 그런 타입이 없다. `koharu_scene::Point`가 맞다.
3. Task 4 테스트가 `bubble::geometry_frame`의 `frame.bounds.width`를 읽었지만, Task 3이 `ShapeFrame`을 도입하므로 `frame.width`가 맞다.
4. Task 7이 `koharu-app`에서 `shape_path`를 부르는데, 이 계획의 초안은 Task 3을 `pub(crate)`로 적었다. `pub`으로 확정하고 `ShapeFrame`을 도입해 `GeometryFrame`의 `pub(crate)` 제약을 우회했다.
5. `Polygon`/`Freeform` 경로 생성이 Task 4와 Task 7에 중복될 뻔했다. `shape_layer_path` 하나로 합쳤다.

**Proportion**

스펙 538줄, 계획 약 900줄. 코드 블록은 테스트와 정확한 시그니처 위주이며 함수 본문은 알고리즘이 시그니처와 테스트로 결정되지 않는 곳(Task 3의 아핀 계수, Task 7의 래스터 순서)에만 있다.
