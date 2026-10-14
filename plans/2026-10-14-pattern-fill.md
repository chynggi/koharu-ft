# 패턴 채우기 (1c) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 도형의 채우기를 이미지 패턴으로 — 이미지를 타일로 반복하거나 도형에 맞춰 늘린다.

**Architecture:** 1b의 그라디언트 경로와 **구조가 같다**. `ShapeFill::Pattern(PatternSpec)`이 저장 모델에 들어가고, `prepared()`가 정규화 프레임 로컬 좌표를 경로 로컬로 펴서 `PreparedFill::Pattern`에 담고, `append_to`가 peniko `ImageBrush`를 만들어 `Scene::fill`에 넘긴다. **타일링은 `Extend::Repeat`이 하므로 격자 계산이 없다.** 패턴 이미지는 1b의 그라디언트처럼 도형 레이어가 소유한다(`AssetRole::new("pattern")`).

**Tech Stack:** Rust (koharu-scene, koharu-rasterizer, koharu-renderer, koharu-app, koharu-rpc), vello 0.10 / peniko 0.6.1 / kurbo, TypeScript (Next.js, zustand, i18next), `cargo test`, vitest.

**Spec:** `specs/2026-10-14-pattern-fill-design.md` (P-D1 ~ P-D7)

**전제 확인 (계획 전에 코드로 검증함):**
- vello 0.10의 `Scene::fill`은 `impl Into<StyleRef>`를 받고 `ImageBrush`가 그중 하나다 (`vello-0.10.0/src/scene.rs:316`). `Scene::draw_image`도 있다(`:442-449`) — 타일링에는 `fill` + `Repeat`을 쓴다.
- peniko `ImageSampler`는 `x_extend`/`y_extend`/`quality`/`alpha`를 갖고(`peniko-0.6.1/src/image.rs:95-104`), `Extend::{Pad, Repeat, Reflect}`가 있다(`brush.rs:159-168`).
- `RasterImage`(`frame.rs:234-240`)가 `ResourceId`·타일·픽셀을 나르고 `compositor.rs:520-541`이 GPU로 올린다 — **패턴도 이 경로를 쓴다.**

---

## Global Constraints

- **타일링을 직접 구현하지 않는다.** 초판 스펙은 "vello 브러시로는 이미지 타일을 표현할 수 없다"고 썼지만 **거짓이다.** `Extend::Repeat`이 임의 오프셋·스케일·회전에서 반복한다. 격자를 손으로 나누면 `scale`·`origin`·회전 계산을 우리가 소유하게 되고, 그 계산이 커밋과 미리보기 사이에서 어긋날 자리가 생긴다(1b에서 실제로 난 버그 유형).
- **`brush_transform`은 `None`이다.** vello가 경로 변환과 브러시 변환을 합성하므로(`scene.rs:328-331`) 경로 아핀을 또 넘기면 이중 적용된다 — **1b Task 2의 Critical과 같은 함정이다.**
- **패턴은 도형 레이어가 소유한다**(P-D2). `PatternSpec`은 레이어를 참조하지 않는다. 참조하면 "레이어가 지워졌는데 참조가 남는" 상태가 생기고, 1a·1b에서 가장 비쌌던 두 권위 문제가 재현된다.
- **자산이 없거나 디코드에 실패하면 `fallback` 색으로 그린다 — 오류가 아니다**(P-D6). 패턴 자산 하나가 깨졌다고 페이지 전체가 열리지 않으면 안 된다.
- **`ShapeFill`에 변형을 추가하되 `Pattern`을 맨 뒤에 둔다.** revision enum 태그가 변형 id를 하위 비트에 담으므로(`shapes.rs:208-211`의 `Gradient` 주석이 그 규약을 기록한다) 뒤에 붙이면 기존 `Solid`(0)·`Gradient`(1)가 보존된다.
- `scale`은 **양수 유한**, `origin`은 유한, `rotation_degrees`는 `-360..=360`. **`origin`이 `0..1` 밖인 것은 거부하지 않는다** — "패턴을 옮기기"가 그렇게 동작한다.
- **`Stretch`에서 `origin`은 무시된다.** 같은 구조체를 공유하므로 이 사실을 doc 주석에 명시한다 — 숨기면 나중에 누군가 그 필드를 읽는다.
- 하위 호환·별칭 금지 (AGENTS.md). 주석은 불변식을 설명한다.
- `-j 6`을 모든 cargo 명령에 붙인다. 검증은 가장 작은 관련 검사만 debug 프로필로 한 번 (AGENTS.md).

## Review Focus

스펙이 명시하지 않았지만 실제로 깨질 수 있는 지점이다.

1. **`Extend` 매핑이 `repeat`와 일치하는지.** `Tile` → `Extend::Repeat`, `Stretch` → `Extend::Pad`(또는 이미지를 도형 박스로 스케일). 매핑이 뒤바뀌면 타일이 늘어나거나 늘리기가 반복된다(Task 2).
2. **`Stretch`의 스케일 계산.** 이미지 한 장을 도형의 바운딩 박스에 맞추려면 `ImageBrush`에 변환을 줘야 한다. `Tile`의 `scale`과 혼동하면 안 된다(Task 2).
3. **1b의 brush_transform 함정 재발.** `append_to`가 경로 아핀을 `brush_transform`으로 넘기면 이중 적용된다. 1b에서 이 버그가 실제로 났고 회전한 도형에서만 드러났다(Task 2).
4. **자산 픽셀 밀도와 PSD 내보내기.** 1c는 벡터 채우기이므로 PSD는 래스터화한다. **래스터화 경로가 패턴을 그리는지** 확인해야 한다 — 빠뜨리면 PSD에서 도형이 사라진다(Task 5).
5. **`merge_pages`가 `pattern` 역할 자산을 옮기는지.** 1a의 Critical이 `merge_pages`에서 났다(기하를 옮기면서 불변식을 깨뜨렸다). 패턴 자산이 페이지 병합에서 누락되면 병합 후 패턴이 사라진다(Task 5).
6. **`fill()`의 반환 타입.** 1b는 `Option<PreparedFill>`이다. 패턴도 같은 자리이므로 타입을 넓힐 필요가 없다 — 넓히려 한다면 설계가 틀어진 것이다(Task 2).

---

## File Structure

**수정**

| 파일 | 변경 |
|---|---|
| `crates/koharu-scene/src/components/shapes.rs` | `ShapeFill::Pattern`, `PatternSpec`, `PatternRepeat`, `validate` |
| `crates/koharu-scene/src/components.rs` + `lib.rs` | 재수출 (1b에서 재수출 누락이 리뷰 지적이었다) |
| `crates/koharu-rasterizer/src/prepared.rs` | `PreparedFill::Pattern`, `PreparedPattern` |
| `crates/koharu-rasterizer/src/frame.rs` | `append_to`가 `ImageBrush`를 만든다 |
| `crates/koharu-renderer/src/shape_renderer.rs` | `fill()`이 `Pattern`을 펴서 넘긴다 |
| `crates/koharu-app/src/commands/project.rs` | `set_shape_pattern`, `set_shape_pattern_asset` |
| `crates/koharu-app/src/commands/canvas.rs` | 커맨드 2개 |
| `crates/koharu-rpc/src/routes/canvas.rs` | 라우트 2개 |
| `packages/bridge/src/protocol.ts` | `PatternSpec` 타입, `commands.setShapePattern`/`setShapePatternAsset` |
| `packages/bridge/src/canvas.ts` | `CanvasPathFill`에 `Pattern` 변형 |
| `packages/koharu/lib/shapes.ts` | `defaultPatternSpec`, `patternOrigin` |
| `packages/koharu/components/editor/Inspector.tsx` | 패턴 컨트롤 |
| `packages/koharu/public/locales/*/translation.json` | 9개 로케일 |

---

### Task 1: `PatternSpec` 저장 모델

**Files:**
- Modify: `crates/koharu-scene/src/components/shapes.rs`
- Modify: `crates/koharu-scene/src/components.rs`, `crates/koharu-scene/src/lib.rs`
- Test: `crates/koharu-scene/src/components/shapes.rs` (인라인 `mod tests`)

**Interfaces:**
- Produces:
  - `ShapeFill::Pattern(PatternSpec)` (맨 뒤에 추가)
  - `PatternSpec { repeat: PatternRepeat, origin: [f32; 2], scale: f32, rotation_degrees: i32, fallback: [u8; 4] }`
  - `PatternRepeat::{Tile, Stretch}`
  - `PatternSpec::validate(&self) -> Result<()>`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`shapes.rs`의 `mod tests`에 추가한다. 기존 그라디언트 검증 테스트(`validate_spec` 계열)의 스타일을 따른다.

```rust
    fn pattern_spec() -> PatternSpec {
        PatternSpec {
            repeat: PatternRepeat::Tile,
            origin: [0.0, 0.0],
            scale: 0.25,
            rotation_degrees: 0,
            fallback: [255, 255, 255, 255],
        }
    }

    #[test]
    fn a_default_pattern_spec_validates() {
        assert!(pattern_spec().validate().is_ok());
    }

    #[test]
    fn a_pattern_scale_must_be_positive_and_finite() {
        for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let mut spec = pattern_spec();
            spec.scale = scale;
            assert!(spec.validate().is_err(), "scale {scale} must be rejected");
        }
    }

    #[test]
    fn a_pattern_origin_must_be_finite() {
        for origin in [[f32::NAN, 0.0], [0.0, f32::INFINITY]] {
            let mut spec = pattern_spec();
            spec.origin = origin;
            assert!(spec.validate().is_err(), "origin {origin:?} must be rejected");
        }
    }

    #[test]
    fn a_pattern_rotation_is_bounded() {
        let mut spec = pattern_spec();
        spec.rotation_degrees = 361;
        assert!(spec.validate().is_err());
        spec.rotation_degrees = -361;
        assert!(spec.validate().is_err());
        spec.rotation_degrees = 360;
        assert!(spec.validate().is_ok());
    }

    /// An origin outside the unit square is how "move the pattern" works, so it
    /// must not be rejected. Only non-finite values are invalid.
    #[test]
    fn a_pattern_origin_outside_the_unit_square_is_allowed() {
        let mut spec = pattern_spec();
        spec.origin = [-2.5, 3.75];
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn a_pattern_fill_round_trips_through_serde() {
        let fill = ShapeFill::Pattern(pattern_spec());
        let json = serde_json::to_string(&fill).unwrap();
        let decoded: ShapeFill = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, fill);
    }

    /// The revision enum tag carries the variant id in bits 0..=4, so `Pattern`
    /// must be appended after `Gradient` or existing data would decode as the
    /// wrong variant. Pin the ids.
    #[test]
    fn appending_pattern_preserves_the_existing_variant_ids() {
        // `Solid` and `Gradient` keep their wire order; if this changes, saved
        // projects would decode a solid as a gradient.
        assert_eq!(
            serde_json::to_string(&ShapeFill::Solid([1, 2, 3, 255])).unwrap(),
            r#"{"solid":[1,2,3,255]}"#
        );
        assert!(serde_json::to_string(&ShapeFill::Pattern(pattern_spec()))
            .unwrap()
            .starts_with(r#"{"pattern":{"#));
    }

    #[test]
    fn the_pattern_types_are_reachable_from_the_crate_root() {
        // 1b shipped gradient types that could not be named outside the crate;
        // pin the re-export so that gap cannot reappear.
        let _: koharu_scene::PatternRepeat = koharu_scene::PatternRepeat::Tile;
        let _: koharu_scene::PatternSpec = pattern_spec();
    }
```

- [ ] **Step 2: 실패를 확인한다**

Run: `cargo test -j 6 -p koharu-scene --lib pattern`
Expected: 컴파일 실패 (`PatternSpec` 미정의)

- [ ] **Step 3: 구현한다**

`ShapeFill`의 **맨 뒤에** `Pattern(PatternSpec)`을 추가하고, 그 이유를 기존 `Gradient` 주석처럼 남긴다.

```rust
#[revisioned(revision = 1)]
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum PatternRepeat {
    /// `origin`에서 시작해 `scale` 간격으로 반복한다.
    Tile,
    /// 이미지 하나를 도형의 바운딩 박스에 맞춰 늘린다. `origin`은 무시한다.
    Stretch,
}

#[revisioned(revision = 1)]
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Type)]
pub struct PatternSpec {
    pub repeat: PatternRepeat,
    /// 정규화 프레임 로컬 좌표(0..1)의 타일 원점. **`Stretch`에서는 무시된다.**
    /// `0..1` 밖이어도 된다 — 패턴을 옮기는 것이 그렇게 동작한다.
    pub origin: [f32; 2],
    /// 타일 간격: 한 타일이 차지하는 프레임 로컬 폭(정규화). 양수여야 한다.
    /// `Stretch`에서는 무시된다.
    pub scale: f32,
    /// 시계 방향 도. `Tile`에서만 의미를 갖는다.
    pub rotation_degrees: i32,
    /// 자산이 없거나 디코드에 실패했을 때 쓰는 색.
    pub fallback: [u8; 4],
}
```

`validate`가 거부하는 것: `scale`이 비유한이거나 0 이하, `origin`의 두 성분 중 비유한, `rotation_degrees`가 `-360..=360` 밖. **`origin`의 범위는 검사하지 않는다.**

`components.rs`와 `lib.rs`의 `pub use`에 `PatternSpec`, `PatternRepeat`를 추가한다. **1b에서 이 재수출을 빠뜨려 리뷰 지적이 났으므로 두 목록 모두 확인한다.**

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

Run: `cargo test -j 6 -p koharu-scene --lib`
Expected: 기존 + 8 passed. **`ShapeFill`에 변형을 추가했으므로 이 크레이트를 쓰는 모든 곳이 컴파일 실패할 수 있다** — 그 목록을 보고에 남긴다(Task 2가 고친다).

- [ ] **Step 5: 커밋**

```bash
git add crates/koharu-scene/src/components/shapes.rs crates/koharu-scene/src/components.rs crates/koharu-scene/src/lib.rs
git commit -m "feat(scene): add the pattern fill model"
```

---

### Task 2: 준비 계층과 렌더링

**Files:**
- Modify: `crates/koharu-rasterizer/src/prepared.rs`, `crates/koharu-rasterizer/src/frame.rs`
- Modify: `crates/koharu-renderer/src/shape_renderer.rs`
- Test: 위 파일들의 인라인 `mod tests`

**Interfaces:**
- Consumes: Task 1의 `PatternSpec`, `PatternRepeat`
- Produces:
  - `PreparedFill::Pattern { pattern: PreparedPattern, rule: FillRule }`
  - `PreparedPattern { repeat: PreparedPatternRepeat, origin: [f32; 2], scale: f32, rotation_degrees: i32, fallback: [u8; 4], source: ResourceId }`
  - `shape_renderer::fill()`이 `Pattern`을 편다

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`shape_renderer.rs`의 `mod tests`에 추가한다. 1b의 그라디언트 스프레드 테스트와 같은 형태다.

```rust
    #[test]
    fn a_pattern_origin_is_spread_onto_the_frame() {
        let frame = ShapeFrame { x: 10.0, y: 20.0, width: 200.0, height: 100.0, angle_degrees: 0.0 };
        let style = ShapeStyle {
            fill: Some(ShapeFill::Pattern(PatternSpec {
                repeat: PatternRepeat::Tile,
                origin: [0.5, 0.25],
                scale: 0.5,
                rotation_degrees: 0,
                fallback: [0, 0, 0, 255],
            })),
            stroke: None,
        };
        let Some(PreparedFill::Pattern { pattern, .. }) = fill(&style, frame, false) else {
            panic!("expected a pattern fill");
        };
        // The spread happens in exactly one place: `fill()`.
        assert_eq!(pattern.origin, [10.0 + 0.5 * 200.0, 20.0 + 0.25 * 100.0]);
        assert_eq!(pattern.scale, 0.5 * 200.0);
    }

    /// Rotation is carried by the path transform, never by the pattern's own
    /// coordinates — the same rule the gradient follows. Passing an affine as
    /// `brush_transform` would double-apply it (1b's Critical bug).
    #[test]
    fn a_pattern_keeps_its_coordinates_local_when_rotated() {
        let upright = /* frame with angle 0 */;
        let rotated = /* the same frame at 90 degrees */;
        let style = /* the same pattern style */;
        let a = fill(&style, upright, false);
        let b = fill(&style, rotated, false);
        assert_eq!(a, b, "the pattern must be identical for both rotations");
    }

    #[test]
    fn a_stretch_pattern_reports_the_stretch_repeat() {
        let style = ShapeStyle {
            fill: Some(ShapeFill::Pattern(PatternSpec {
                repeat: PatternRepeat::Stretch,
                origin: [0.0, 0.0],
                scale: 1.0,
                rotation_degrees: 0,
                fallback: [0, 0, 0, 255],
            })),
            stroke: None,
        };
        let Some(PreparedFill::Pattern { pattern, .. }) = fill(&style, frame, false) else {
            panic!("expected a pattern fill");
        };
        assert_eq!(pattern.repeat, PreparedPatternRepeat::Stretch);
    }
```

`frame.rs`의 테스트에는 **`Extend` 매핑**과 **`brush_transform`이 `None`**임을 고정한다. 1b가 `appending_a_gradient_path_passes_no_brush_transform`으로 그렇게 했다 — 같은 형태로 `appending_a_pattern_path_passes_no_brush_transform`을 둔다.

- [ ] **Step 2: 실패를 확인한다**

Run: `cargo test -j 6 -p koharu-renderer --lib pattern`, `cargo test -j 6 -p koharu-rasterizer --lib pattern`
Expected: 컴파일 실패

- [ ] **Step 3: 구현한다**

`prepared.rs`: `PreparedFill`에 `Pattern` 변형을 추가한다. `PreparedPatternRepeat`은 `PreparedGradientExtend`와 같은 스타일(PascalCase unit variant, `#[serde(rename_all)]` 없음)을 따른다.

`shape_renderer::fill()`: `ShapeFill::Pattern(spec)` arm에서
- `origin`을 프레임 로컬로 편다: `[frame.x + origin[0] * frame.width, frame.y + origin[1] * frame.height]` (1b의 그라디언트와 **같은 규칙**)
- `scale`을 프레임 **폭**으로 편다: `spec.scale * frame.width` (그라디언트의 `radius * frame.width`와 같은 관례)
- `angle_degrees`는 쓰지 않는다

`frame.rs`의 `PreparedPath::append_to`: `Some(PreparedFill::Pattern { .. })` arm에서
```rust
let brush = ImageBrush::new(ImageData { /* decoded pixels */ })
    .with_extend(match pattern.repeat {
        PreparedPatternRepeat::Tile => Extend::Repeat,
        PreparedPatternRepeat::Stretch => Extend::Pad,
    });
// `brush_transform` stays `None`: vello composes it with the path transform, so
// passing the path affine would apply it twice (see the gradient arm's note).
scene.fill(fill(rule), affine, &brush, None, &path);
```

**`Stretch`의 스케일.** `Extend::Pad`만으로는 이미지가 원래 픽셀 크기로 그려진다. 도형 박스에 맞추려면 `ImageBrush`에 스케일 변환을 줘야 한다 — `ImageBrush`의 변환은 `Scene::fill`의 `brush_transform` 인자가 담당하므로, **`Stretch`일 때만** `Some(도형_박스_크기로_가는_아핀)`을 넘긴다. **`Tile`일 때는 반드시 `None`이다.** 이 비대칭의 이유를 주석에 남긴다 — 1b의 Critical이 정확히 이 인자의 오용이었다.

패턴 자산이 없거나 디코드에 실패하면 **`Solid { color: fallback, rule }`을 돌려준다**(P-D6). `append_to`가 아니라 `fill()`에서 결정하는 편이 낫다 — `fill()`이 이미 `Option<PreparedFill>`을 돌려주므로 폴백이 자연스럽고, `append_to`는 "무엇을 그릴지"를 다시 판단하지 않는다. **어느 쪽을 택했든 이유를 남긴다.**

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

Run: `cargo test -j 6 -p koharu-rasterizer -p koharu-renderer --lib`
Expected: 전부 통과

- [ ] **Step 5: 커밋**

```bash
git add crates/koharu-rasterizer/src/prepared.rs crates/koharu-rasterizer/src/frame.rs crates/koharu-renderer/src/shape_renderer.rs
git commit -m "feat(renderer): fill a shape with a pattern brush"
```

---

### Task 3: 커맨드·라우트·TS 계약

`protocol.ts`를 **먼저** 커밋해 Task 5(UI)를 병렬화할 수 있게 한다.

**Files:**
- Modify: `packages/bridge/src/protocol.ts`
- Modify: `crates/koharu-app/src/commands/project.rs`, `canvas.rs`
- Modify: `crates/koharu-rpc/src/routes/canvas.rs`

**Interfaces:**
- Produces:
  - `Project::set_shape_pattern(layer, Option<PatternSpec>)`, `Project::set_shape_pattern_asset(layer, bytes, media_type)`
  - 커맨드 `set_shape_pattern`, `set_shape_pattern_asset`
  - 라우트 `POST /canvas/shape/pattern`, `POST /canvas/shape/pattern/asset`
  - TS `commands.setShapePattern`, `commands.setShapePatternAsset`

- [ ] **Step 1: `protocol.ts`를 갱신하고 커밋한다**

```ts
export type PatternRepeat = "tile" | "stretch";

export type PatternSpec = {
	repeat: PatternRepeat,
	origin: [number, number],
	scale: number,
	rotation_degrees: number,
	fallback: [number, number, number, number],
};
```

`ShapeFill`에 `{ pattern: PatternSpec }`을 추가한다. `commands`에 두 메서드를 추가한다.

**`origin`과 `scale`은 `Stretch`에서 무시된다**는 사실을 주석에 남긴다.

- [ ] **Step 2: 타입 검사**

Run: `bunx tsc --noEmit -p packages/koharu/tsconfig.json`
Expected: exit 0

- [ ] **Step 3: 커밋**

```bash
git add packages/bridge/src/protocol.ts
git commit -m "feat(bridge): declare the pattern fill wire contract"
```

- [ ] **Step 4: `Project` 메서드 2개를 구현한다**

`set_shape_pattern`은 `set_shape_style` 계열을 본뜬다. `None`이면 `Solid`로 되돌린다.
`set_shape_pattern_asset`은 **도형 레이어에만** 허용하고(P-D2), `AssetRole::new("pattern")`으로 자산을 쓴다. 래스터 레이어의 `pattern` 역할은 거부한다.

테스트: `set_shape_pattern`이 `None`으로 `Solid`가 되는지, 자산 커맨드가 래스터를 거부하는지, **wire 계약 리터럴**(1b Task 4 방식)로 `{"pattern":{…}}`를 고정한다.

- [ ] **Step 5: 커맨드 2개와 라우트 2개를 추가한다**

`canvas.rs`의 `set_shape_style`과 `adjust_raster`를 본뜬다. 순서는 기존과 같다: `ensure_revision` → 프로젝트 메서드 → `record_commit` → `synchronize` → `publish` → `LayerCommit`.

**라우트 문자열이 `router()` 체인에 있는지 grep으로 확인한다** — 핸들러만 있고 라우팅되지 않는 것이 이 규칙이 막으려는 실패다.

- [ ] **Step 6: 컴파일과 테스트**

Run: `cargo check -j 6 -p koharu-rpc -p koharu-app`
Run: `cargo test -j 6 -p koharu-app --lib`
Expected: 성공, 기존 통과

- [ ] **Step 7: 커밋**

```bash
git add crates/koharu-app/src/commands/project.rs crates/koharu-app/src/commands/canvas.rs crates/koharu-rpc/src/routes/canvas.rs
git commit -m "feat(app): set a shape's pattern fill"
```

---

### Task 4: WASM 미리보기

**Files:**
- Modify: `packages/bridge/src/canvas.ts` (타입)
- Modify: `crates/koharu-canvas/src/web.rs`
- Test: `crates/koharu-canvas/src/*.rs`의 순수 함수 테스트 (`web.rs`는 host에서 컴파일되지 않는다 — **B1에서 확인한 제약이다**)

**Interfaces:**
- Produces: `CanvasPathFill`에 `Pattern` 변형; `shapePreviewPath`가 패턴을 조립

- [ ] **Step 1: `CanvasPathFill`에 변형을 추가한다**

`canvas.ts`의 `CanvasPathFill`에 `Pattern`을 그라디언트와 나란히 넣는다. **B1에서 확인했듯 `web.rs`는 `wasm32`에서만 컴파일되므로 `cargo check -p koharu-canvas`는 그 파일을 검사하지 않는다** — 검증에 `--target wasm32-unknown-unknown`을 반드시 포함한다.

- [ ] **Step 2: 프론트 조립과 변환을 구현한다**

`lib/shapes.ts`에 `defaultPatternSpec`과, 그라디언트의 `previewGradient`에 대응하는 패턴 조립을 넣는다. **1b와 같은 회전 규칙을 따른다**: `frame.origin + normalized * extent`를 구한 뒤 `rotatePoint`로 프레임 중심 기준 회전. `angle_degrees`를 그라디언트 조립에 쓰지 않는 것과 같은 이유다.

**정확한 좌표를 단언하는 테스트를 둔다.** 1b에서 미리보기와 커밋이 어긋난 버그는 이 테스트가 없어서 생겼다. 1b가 쓴 값을 그대로 재사용해 회전 규칙이 같음을 확인한다(프레임 `{0,0,200,100,angle 90}`에서 제어점이 `(150,-50)`으로 가는지).

- [ ] **Step 3: 확인한다**

Run: `cargo check -j 6 -p koharu-canvas --target wasm32-unknown-unknown`
Run: `bun run --filter '@koharu/app' test -- tests/lib/shapes.test.ts`
Expected: 성공

- [ ] **Step 4: 커밋**

```bash
git add packages/bridge/src/canvas.ts packages/koharu/lib/shapes.ts packages/koharu/tests/lib/shapes.test.ts crates/koharu-canvas/src/web.rs
git commit -m "feat(ui): preview a pattern fill"
```

---

### Task 5: Inspector UI와 i18n

**Files:**
- Modify: `packages/koharu/components/editor/Inspector.tsx`
- Modify: `packages/koharu/public/locales/*/translation.json` (9개)
- Modify: `packages/koharu/tests/components/editor-components.test.tsx`

**Interfaces:**
- Consumes: Task 3의 `commands.setShapePattern`/`setShapePatternAsset`
- Produces: 채우기 종류에 "패턴" + 반복·원점·스케일·회전 입력 + 이미지 업로드

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`editor-components.test.tsx`에 추가한다. **기존 그라디언트 스톱 편집기 테스트**가 선택·스파이·커밋을 관찰하는 방식을 그대로 따른다.

```tsx
  it('sets a pattern fill and preserves it while editing its parameters', async () => {
    // 패턴 채우기로 전환하고 스케일을 바꾼 뒤,
    // 커밋된 fill에 `pattern` 키가 있고 `solid`/`gradient`가 없음을 단언한다.
  })

  it('hides the flat colour well while a pattern is active', async () => {
    // 두 권위를 만들지 않는다 — 1b의 규칙과 같다.
  })
```

**핵심 단언**: 패턴이 활성일 때 파라미터를 편집해도 커밋이 `{ pattern: … }`으로 유지되고 `{ solid: … }`로 강등되지 않아야 한다 — **B3 Task 6에서 실제로 났던 데이터 손실 경로다.**

- [ ] **Step 2: 컨트롤을 구현한다**

종류 선택에 "패턴"을 추가하고, 활성일 때: 반복(타일/늘리기), 원점(2 숫자), 스케일(1 숫자, 양수), 회전(정수 도), 이미지 업로드(파일 입력).

`Stretch`일 때 **`origin`과 `scale` 입력을 비활성화한다** — 서버가 무시하는 값을 편집하게 두면 "왜 안 바뀌지"가 된다.

**UI가 서버가 거부할 값을 만들 수 없어야 한다**: 스케일은 양수, 회전은 `-360..=360`으로 clamp한다(B3·B1과 같은 규율).

- [ ] **Step 3: 9개 로케일에 키를 추가한다**

`inspector.shape.pattern.*`: `title`, `repeat`, `repeat.tile`, `repeat.stretch`, `originX`, `originY`, `scale`, `rotation`, `upload`, `replace`, `ignoredForStretch`.

`ko-KR`은 실제 한국어로("패턴", "반복", "타일", "늘리기", "원점 X", "원점 Y", "크기", "회전", "이미지 선택", "늘리기에서는 무시됨"). 나머지 8개는 실제 번역.

**키 집합이 9개 파일에서 동일해야 한다**(B1이 595키, 파일은 CRLF).

- [ ] **Step 4: 프론트 검사**

Run: `bun run --filter '@koharu/app' test`
Run: `bunx tsc --noEmit -p packages/koharu/tsconfig.json`
Run: `bunx oxlint -c packages/koharu/.oxlintrc.json packages/koharu`

- [ ] **Step 5: 커밋**

```bash
git add packages/koharu/components/editor/Inspector.tsx packages/koharu/public/locales packages/koharu/tests
git commit -m "feat(ui): choose a pattern fill"
```

---

### Task 6: PSD 내보내기와 페이지 병합, 그리고 회귀

**Files:**
- Modify: `crates/koharu-psd/src/document.rs` (필요하면)
- Modify: `crates/koharu-app/src/commands/project.rs` (`merge_pages`)
- Test: 위 파일들

이 태스크가 **Review Focus 4·5**를 닫는다. 둘 다 1a·1b에서 실제로 버그가 난 자리다.

- [ ] **Step 1: PSD 내보내기가 패턴을 그리는지 확인하고 테스트한다**

도형은 PSD에서 래스터화된다. 래스터화가 `shape_renderer`를 지나므로 패턴도 자동으로 그려져야 하지만, **확인 없이 가정하지 않는다.** 패턴 채우기 도형이 있는 페이지를 내보내고 그 레이어에 `fallback`도 흰색도 아닌 패턴 픽셀이 있는지 단언한다.

- [ ] **Step 2: `merge_pages`가 `pattern` 역할 자산을 옮기는지 확인하고 테스트한다**

1a의 Critical이 `merge_pages`였다. 두 페이지를 병합하고 **한쪽 도형의 패턴이 병합 후에도 살아 있는지** 단언한다.

- [ ] **Step 3: 전체 회귀**

Run: `cargo test -j 6 -p koharu-scene -p koharu-rasterizer -p koharu-renderer -p koharu-app -p koharu-psd -p koharu-rpc --lib`
Run: `cargo check -j 6 --workspace`
Run: `cargo check -j 6 -p koharu-canvas --target wasm32-unknown-unknown`
Run: `bun run --filter '@koharu/app' test`

**1b의 그라디언트 테스트가 그대로 통과하는지 확인한다** — `ShapeFill`에 변형을 추가했으므로 기존 arm이 깨질 수 있다. 특히 `appending_a_gradient_path_passes_no_brush_transform`(1b Critical의 회귀 가드)이 살아 있어야 한다.

- [ ] **Step 4: 보고**

테스트 결과와 `git diff --stat`를 남긴다.

---

## Self-Review

**Spec coverage**

| 스펙 절 | 태스크 |
|---|---|
| `PatternSpec` + `PatternRepeat` (P-D3, P-D4, P-D7) | Task 1 |
| `ShapeFill::Pattern` | Task 1 |
| `validate` 규칙 | Task 1 |
| `prepared()`의 정규화→프레임 로컬 (P-D3) | Task 2 |
| `ImageBrush` + `Extend` (P-D5) | Task 2 |
| `brush_transform = None` | Task 2 |
| 자산 폴백 (P-D6) | Task 2 |
| `pattern` 역할 + 도형 레이어 소유 (P-D1, P-D2) | Task 3 |
| `set_shape_pattern`/`set_shape_pattern_asset` | Task 3 |
| 프론트 규칙 | Task 4, Task 5 |
| `merge_pages` 자산 | Task 6 |
| 테스트 표 | 전 태스크 |

**Spec을 계획이 정정한 곳**

1. **렌더링 방식.** 초판은 "클립 + 반복 draw"였고 그것 때문에 `fill()`이 `Pattern`에 `None`을 돌려주고 격자를 손으로 나누는 설계였다. vello가 `ImageBrush`와 `Extend::Repeat`을 제공하므로 **브러시 하나로 끝난다.** 스펙을 커밋 `3f13da76`에서 정정했다.
2. **P-D5.** "타일링을 직접 구현한다"에서 "라이브러리에 맡긴다"로 바뀌었다.
3. **미리보기.** 초판은 자리표시자였으나 브러시 경로가 생겼으므로 정상 구현한다.
4. **열린 문제 3(타일 상한).** 해결됐다 — 셰이더가 샘플링하므로 타일 수가 draw 수에 영향을 주지 않는다.

**계획이 스펙에 더한 것**

- **`Stretch`의 스케일 변환.** 스펙은 반복 방식만 정했고, 늘리기가 어떻게 도형 박스에 맞는지 말하지 않았다. Task 2가 "`Stretch`일 때만 `brush_transform`에 스케일 아핀을 준다"로 정하고, 그 비대칭(타일은 `None`)의 이유를 남긴다.
- **폴백의 위치.** 스펙은 "폴백한다"만 썼다. Task 2가 `fill()`에서 결정하도록 정하고 이유를 남긴다.
- **`Stretch`에서 `origin`/`scale` 입력 비활성화.** 스펙은 서버가 무시한다고만 했다. UI가 편집 가능한 값을 보여주면 혼란스러우므로 Task 5가 비활성화한다.

**Type consistency**

- `PatternRepeat::{Tile, Stretch}` (Rust) ↔ `"tile"`/`"stretch"` (TS) — `#[serde(rename_all = "snake_case")]`.
- `PatternSpec`의 필드명이 Rust·TS·wire에서 동일하다.
- `PreparedPatternRepeat::{Tile, Stretch}`는 `PreparedGradientExtend`와 같은 PascalCase 관례.
- `ShapeFill::Pattern(PatternSpec)` — 1b의 `Gradient(GradientSpec)`과 같은 형태.

**Proportion**

스펙 약 200줄, 계획 약 700줄. 6 태스크이고, Task 1·2가 모델·렌더링, Task 3이 전송, Task 4·5가 프론트, Task 6이 통합 위험(PSD·병합)이다. 1b와 같은 크기이며 그 경로를 재사용하므로 위험이 낮다.
