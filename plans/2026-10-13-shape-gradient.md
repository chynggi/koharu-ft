# 도형 그라디언트 채우기 (1b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 도형의 채우기를 단색에서 선형·방사(그리고 데이터 모델상 스윕) 그라디언트로 넓히고, 스톱을 편집하는 UI를 붙인다.

**Architecture:** 저장 모델에 `GradientSpec`(정규화 프레임 로컬 좌표)을 추가하고 `ShapeFill`에 변형을 하나 더한다. `prepared()`가 정규화 좌표를 프레임 로컬로 펴서 `PreparedFill::Gradient`에 담고, 렌더러는 `peniko::Gradient`를 만들어 `Scene::fill`에 넘기며 경로와 같은 아핀을 `brush_transform` 인자로 전달한다 — 회전 정렬은 vello가 담당하므로 좌표를 손으로 변환하지 않는다.

**Tech Stack:** Rust (koharu-scene, koharu-renderer, koharu-rasterizer, koharu-app), vello 0.10 / peniko 0.6.1 / kurbo, TypeScript (Next.js, zustand, i18next), `cargo test`, vitest.

**Spec:** `specs/2026-10-13-shape-gradient-design.md`

**계획 위치 메모:** 저장소 관례대로 `plans/`에 둔다. `docs/`는 발행용 문서 사이트라 고아 페이지가 섞인다.

---

## Global Constraints

- 그라디언트 좌표는 **정규화 프레임 로컬**(프레임 좌상단 `(0,0)`, 우하단 `(1,1)`)로 저장한다. 페이지 절대 좌표를 저장하지 않는다.
- 정규화 → 프레임 로컬 변환은 `prepared()` 한 곳에서만 한다: `local = normalized * frame_extent`.
- 회전 정렬은 `Scene::fill`의 `brush_transform` 인자로 한다. 그라디언트 제어점을 손으로 변환하지 않는다.
- `ShapeFill`의 `revisioned` 리비전을 올리지 않고 `Gradient`를 **마지막에** 추가한다.
- 스톱은 2개 이상 32개 이하. `offset`은 `0.0..=1.0`이고 유한. 같은 `offset` 중복 금지.
- 스톱 정렬은 검증하지 않는다(렌더러가 정렬한다).
- `fill: None`이면 그라디언트여도 채우기 명령을 만들지 않는다 (1a 규칙).
- 화살촉이 있으면 `fill`이 `Some`이 된다는 1a 규칙을 유지한다.
- `ShapeFill::Solid`는 유지한다(그라디언트로 통합하지 않는다).
- `-j 6`을 모든 cargo 명령에 붙인다 (12코어, 무제한 병렬은 데스크톱을 멈춘다).
- 검증은 가장 작은 관련 검사만 debug 프로필로 한 번 돌린다 (AGENTS.md).
- 하위 호환·별칭 금지 (AGENTS.md).

## Review Focus

스펙이 명시하지 않았지만 사용자가 만나게 되는 입력들이다. 각 항목의 테스트는 그 코드를 소유한 태스크에 넣는다.

1. **회전한 도형의 그라디언트 정렬** — 90° 회전한 도형의 그라디언트가 경로와 같은 각도로 돌아가야 한다. `brush_transform`을 빠뜨리면 그라디언트만 축정렬로 남는다 (Task 3).
2. **`Solid` 데이터의 하위 호환** — `Gradient` 변형이 생긴 뒤에도 1a가 저장한 `Solid` 바이트가 `Solid`로 읽혀야 한다. 리비전을 잘못 올리면 기존 프로젝트가 깨진다 (Task 1).
3. **스톱이 정렬되지 않은 입력** — `[{0.8, A}, {0.2, B}]`는 거부되지 않고 올바른 순서로 렌더링되어야 한다 (Task 2).
4. **`fill: None` + 그라디언트** — 채우기 명령이 만들어지지 않아야 한다. 빈 그라디언트가 배경을 덮으면 안 된다 (Task 3).
5. **스톱 경계값** — 정확히 2개와 정확히 32개는 허용, 1개와 33개는 거부 (Task 2).
6. **화살표 + 그라디언트** — 화살촉이 같은 그라디언트로 채워져야 한다 (Task 3).

---

## File Structure

**신규**

| 파일 | 책임 |
|---|---|
| (없음) | 1b는 기존 파일만 넓힌다. 그라디언트 렌더링은 `koharu-rasterizer`의 기존 `compile_scene`이, 편집 UI는 기존 `Inspector`가 소유한다 |

**수정**

| 파일 | 변경 |
|---|---|
| `crates/koharu-scene/src/components/shapes.rs` | `GradientSpec`, `GradientKind`, `GradientStop`, `GradientExtend`, `ShapeFill::Gradient`, 검증 |
| `crates/koharu-rasterizer/src/prepared.rs` | `PreparedFill::Gradient`, `PreparedGradient`, 검증 |
| `crates/koharu-rasterizer/src/frame.rs` | `compile_scene`의 그라디언트 조립 + `brush_transform` |
| `crates/koharu-renderer/src/shape_renderer.rs` | `fill()`이 그라디언트를 펴서 넘김, 화살촉 규칙 유지 |
| `crates/koharu-app/src/commands/project.rs` | `serde_json` wire 계약 테스트 (그라디언트 + `Solid` 회귀) |
| `packages/bridge/src/canvas.ts` | `CanvasPathFill`에 그라디언트 변형 |
| `packages/bridge/src/protocol.ts` | `ShapeFill`/`GradientSpec` TS 타입 |
| `packages/koharu/lib/shapes.ts` | `defaultGradientSpec`, `shapePreviewPath` 그라디언트 조립 |
| `packages/koharu/components/editor/Inspector.tsx` | 채우기 종류 선택 + 스톱 편집기 |
| `packages/koharu/public/locales/*/translation.json` | 9개 로케일 키 |

---

### Task 1: `GradientSpec` 저장 모델

그라디언트의 저장 형태와 검증을 세운다. 이후 모든 태스크가 이 타입을 소비한다.

**Files:**
- Modify: `crates/koharu-scene/src/components/shapes.rs`
- Modify: `crates/koharu-scene/src/components.rs:28-30` (재수출 목록)
- Modify: `crates/koharu-scene/src/lib.rs:26-33` (크레이트 루트 재수출 목록)
- Test: `crates/koharu-scene/src/components/shapes.rs` (인라인 `mod tests`)

**인터페이스 도달성 (이 태스크가 반드시 처리한다):** `mod components`와 `mod shapes`가 모두 private이므로, 새 타입 4개는 두 재수출 목록에 명시적으로 넣어야 크레이트 밖에서 도달한다. Task 3이 `koharu_scene::GradientSpec` 등을 **이름으로** 쓰기 때문에 이 단계를 빠뜨리면 Task 3이 첫 컴파일에서 `cannot find type GradientSpec`로 실패한다. 루트 재수출이 실제로 동작하는지 테스트 모듈의 `use crate::{GradientExtend, GradientKind, GradientSpec, GradientStop};` 한 줄로 컴파일 검증한다.

**Interfaces:**
- Consumes: 기존 `ShapeFill`(`shapes.rs:66-69`), `ShapeStyle`
- Produces:
  - `GradientSpec { kind: GradientKind, stops: Vec<GradientStop>, extend: GradientExtend }`
  - `GradientKind::{Linear { start: [f32;2], end: [f32;2] }, Radial { center: [f32;2], radius: f32 }, Sweep { center: [f32;2], start_degrees: f32, end_degrees: f32 }}`
  - `GradientStop { offset: f32, color: [u8; 4] }`
  - `GradientExtend::{Pad, Repeat, Reflect}`
  - `ShapeFill::Gradient(GradientSpec)`
  - `GradientSpec::validate(&self, context: &ValidationContext<'_>) -> Result<()>`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`shapes.rs`의 기존 `mod tests`에 추가한다. 기존 테스트가 쓰는 `validate` 헬퍼(`ValidationContext::new(&record_exists, &blob_exists)` 관용구)를 재사용하거나, 그라디언트용으로 같은 모양의 헬퍼를 하나 둔다.

```rust
    fn spec(kind: GradientKind) -> GradientSpec {
        GradientSpec {
            kind,
            stops: vec![
                GradientStop { offset: 0.0, color: [0, 0, 0, 255] },
                GradientStop { offset: 1.0, color: [255, 255, 255, 255] },
            ],
            extend: GradientExtend::Pad,
        }
    }

    fn linear() -> GradientKind {
        GradientKind::Linear { start: [0.0, 0.0], end: [1.0, 1.0] }
    }

    fn validate_spec(spec: &GradientSpec) -> Result<()> {
        let record_exists = |_id| true;
        let blob_exists = |_id| true;
        let context = ValidationContext::new(&record_exists, &blob_exists);
        spec.validate(&context)
    }

    #[test]
    fn a_two_stop_gradient_validates() {
        assert!(validate_spec(&spec(linear())).is_ok());
    }

    #[test]
    fn a_thirty_two_stop_gradient_validates() {
        let mut gradient = spec(linear());
        gradient.stops = (0..32)
            .map(|index| GradientStop {
                offset: index as f32 / 31.0,
                color: [index as u8, 0, 0, 255],
            })
            .collect();
        assert!(validate_spec(&gradient).is_ok());
    }

    #[test]
    fn a_single_stop_is_rejected() {
        let mut gradient = spec(linear());
        gradient.stops.truncate(1);
        assert!(validate_spec(&gradient).is_err());
    }

    #[test]
    fn thirty_three_stops_are_rejected() {
        let mut gradient = spec(linear());
        gradient.stops = (0..33)
            .map(|index| GradientStop { offset: 0.5, color: [0, 0, 0, 255] })
            .collect();
        assert!(validate_spec(&gradient).is_err());
    }

    #[test]
    fn an_out_of_range_stop_offset_is_rejected() {
        let mut gradient = spec(linear());
        gradient.stops[0].offset = 1.5;
        assert!(validate_spec(&gradient).is_err());
    }

    #[test]
    fn a_non_finite_stop_offset_is_rejected() {
        let mut gradient = spec(linear());
        gradient.stops[0].offset = f32::NAN;
        assert!(validate_spec(&gradient).is_err());
    }

    #[test]
    fn duplicate_stop_offsets_are_rejected() {
        let mut gradient = spec(linear());
        gradient.stops[1].offset = 0.0;
        assert!(validate_spec(&gradient).is_err());
    }

    #[test]
    fn unsorted_stop_offsets_are_allowed() {
        let mut gradient = spec(linear());
        gradient.stops = vec![
            GradientStop { offset: 0.8, color: [255, 0, 0, 255] },
            GradientStop { offset: 0.2, color: [0, 0, 255, 255] },
        ];
        assert!(validate_spec(&gradient).is_ok());
    }

    #[test]
    fn a_zero_length_linear_gradient_is_rejected() {
        let gradient = spec(GradientKind::Linear { start: [0.5, 0.5], end: [0.5, 0.5] });
        assert!(validate_spec(&gradient).is_err());
    }

    #[test]
    fn a_zero_radius_radial_gradient_is_rejected() {
        let gradient = spec(GradientKind::Radial { center: [0.5, 0.5], radius: 0.0 });
        assert!(validate_spec(&gradient).is_err());
    }

    #[test]
    fn a_non_finite_sweep_angle_is_rejected() {
        let gradient = spec(GradientKind::Sweep {
            center: [0.5, 0.5],
            start_degrees: f32::INFINITY,
            end_degrees: 360.0,
        });
        assert!(validate_spec(&gradient).is_err());
    }

    /// `ShapeFill` gains a variant; a `Solid` written before it existed must
    /// still decode as `Solid`. This is the direct evidence the spec requires
    /// instead of trusting that appending a variant preserves ids.
    #[test]
    fn a_solid_fill_round_trips_with_the_gradient_variant_present() {
        let solid = ShapeFill::Solid([1, 2, 3, 255]);
        let bytes = revision::to_vec(&solid).unwrap();
        let decoded: ShapeFill = revision::from_slice(&bytes).unwrap();
        assert_eq!(decoded, solid);
    }
```

`revision::{to_vec, from_slice}`가 `shapes.rs` 테스트에서 닿으려면 `use revision::{to_vec, from_slice}` 또는 정규 경로를 쓴다. 파일 상단에 이미 `use revision::revisioned;`가 있으므로 같은 크레이트다.

- [ ] **Step 2: 실패를 확인한다**

Run: `cargo test -j 6 -p koharu-scene --lib components::shapes`
Expected: 컴파일 실패 (`GradientSpec` 등 미정의)

- [ ] **Step 3: 타입과 검증을 구현한다**

`revisioned(revision = 1)`을 각 새 타입에 붙이고, `ShapeFill`과 같은 파생 집합(`Clone, Debug, PartialEq, Serialize, Deserialize, Type`)과 `#[serde(rename_all = "snake_case")]`을 쓴다. `GradientStop`은 `Copy`도 붙인다(`[u8;4]`와 `f32`뿐이다).

`ShapeFill`은 이렇게 된다:

```rust
pub enum ShapeFill {
    Solid([u8; 4]),
    /// 1b. `Gradient`는 마지막에 둔다 — revision의 enum 태그가 variant id를
    /// 담으므로(비트 0..=4) 뒤에 추가해야 기존 `Solid`의 id가 유지된다.
    Gradient(GradientSpec),
}
```

`GradientSpec::validate`가 거부하는 것(스펙의 "검증" 절과 정확히 일치):

- `stops.len()`이 2 미만 또는 32 초과
- `offset`이 비유한이거나 `0.0..=1.0` 밖
- 같은 `offset`이 둘 이상
- `Linear`: `start`/`end`의 네 값 중 비유한이 있거나 `start == end`
- `Radial`: `radius`가 비유한이거나 0 이하
- `Sweep`: 두 각도 중 비유한이 있음

**거부하지 않는 것**: 스톱 미정렬. 이 사실을 함수 doc 주석에 남긴다(렌더러가 정렬한다).

`ShapeLayer::validate`는 `ShapeFill::Gradient`일 때 `GradientSpec::validate`를 호출한다. 기존 `Solid` 경로는 바뀌지 않는다.

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

Run: `cargo test -j 6 -p koharu-scene --lib components::shapes`
Expected: 새 테스트 12개 포함 전부 통과

- [ ] **Step 5: 크레이트가 컴파일되는지 확인한다**

Run: `cargo check -j 6 -p koharu-scene`
Expected: 성공. `ShapeFill`을 매치하는 다른 크레이트(`koharu-renderer`)는 아직 컴파일되지 않을 수 있다 — 그건 Task 3의 몫이며, 이 태스크에서는 `koharu-scene`만 통과하면 된다.

**재수출을 확인한다**: `cargo doc`을 돌릴 필요는 없고, 테스트 모듈에 다음 한 줄이 컴파일되면 충분하다:

```rust
    // 크레이트 루트 재수출이 실제로 도달 가능한지 컴파일로 검증한다.
    use crate::{GradientExtend, GradientKind, GradientSpec, GradientStop};
```

`GradientKind`는 여러 테스트가 이미 이름으로 쓰므로, 이 import가 컴파일된다는 것 자체가 Task 3의 경로가 열렸다는 증거다.

- [ ] **Step 6: 커밋**

```bash
git add crates/koharu-scene/src/components/shapes.rs
git commit -m "feat(scene): add the gradient fill spec"
```

---

### Task 2: `PreparedFill::Gradient`

준비된 프레임의 그라디언트 표현과 렌더러 조립을 넣는다.

**Files:**
- Modify: `crates/koharu-rasterizer/src/prepared.rs`
- Modify: `crates/koharu-rasterizer/src/frame.rs:494-526` (`compile_scene`의 `FillPath` 분기)
- Modify: `crates/koharu-rasterizer/src/lib.rs` (재수출)
- Test: `crates/koharu-rasterizer/src/prepared.rs` (인라인 `mod tests`), `crates/koharu-rasterizer/src/frame.rs`

**Interfaces:**
- Consumes: Task 1의 `GradientSpec` 등 (간접 — `koharu-renderer`가 변환한다. `koharu-rasterizer`는 `koharu-scene`에 의존하지 않는다는 1a의 경계를 유지한다)
- Produces:
  - `PreparedFill::Gradient { gradient: PreparedGradient, rule: FillRule }`
  - `PreparedGradient { kind: PreparedGradientKind, stops: Vec<PreparedGradientStop>, extend: PreparedGradientExtend }`
  - `PreparedGradientKind::{Linear { start: [f32;2], end: [f32;2] }, Radial { center: [f32;2], radius: f32 }, Sweep { center: [f32;2], start_degrees: f32, end_degrees: f32 }}`
  - `PreparedGradientStop { offset: f32, color: [u8;4] }`
  - `PreparedGradientExtend::{Pad, Repeat, Reflect}`
  - `pub fn stops_sorted(stops: &[PreparedGradientStop]) -> Vec<PreparedGradientStop>` — `offset` ��준 안정 정렬

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`prepared.rs`의 기존 `mod tests`에 추가한다. 기존 `stroked` 헬퍼 같은 것이 있으면 재사용한다.

```rust
    fn gradient_fill() -> PreparedFill {
        PreparedFill::Gradient {
            gradient: PreparedGradient {
                kind: PreparedGradientKind::Linear {
                    start: [0.0, 0.0],
                    end: [100.0, 0.0],
                },
                stops: vec![
                    PreparedGradientStop { offset: 0.0, color: [0, 0, 0, 255] },
                    PreparedGradientStop { offset: 1.0, color: [255, 255, 255, 255] },
                ],
                extend: PreparedGradientExtend::Pad,
            },
            rule: FillRule::NonZero,
        }
    }

    #[test]
    fn a_gradient_fill_validates() {
        let path = PreparedPath {
            transform: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            elements: vec![PathElement::MoveTo([0.0, 0.0]), PathElement::Close],
            fill: Some(gradient_fill()),
            stroke: None,
        };
        assert!(path.validate().is_ok());
    }

    #[test]
    fn a_gradient_with_one_stop_is_rejected() {
        let Some(PreparedFill::Gradient { mut gradient, .. }) = Some(gradient_fill()) else {
            unreachable!()
        };
        gradient.stops.truncate(1);
        let path = PreparedPath {
            transform: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            elements: vec![PathElement::MoveTo([0.0, 0.0]), PathElement::Close],
            fill: Some(PreparedFill::Gradient {
                gradient,
                rule: FillRule::NonZero,
            }),
            stroke: None,
        };
        assert!(path.validate().is_err());
    }

    #[test]
    fn stops_are_sorted_by_offset() {
        let stops = vec![
            PreparedGradientStop { offset: 0.8, color: [255, 0, 0, 255] },
            PreparedGradientStop { offset: 0.2, color: [0, 0, 255, 255] },
        ];
        let sorted = stops_sorted(&stops);
        assert_eq!(sorted[0].offset, 0.2);
        assert_eq!(sorted[1].offset, 0.8);
    }
```

`stops_sorted`가 이미 정렬된 입력을 그대로 돌려주는지, 같은 offset이 있어도 안정적인지도 단언에 넣는다.

- [ ] **Step 2: 실패를 확인한다**

Run: `cargo test -j 6 -p koharu-rasterizer --lib prepared::tests`
Expected: 컴파일 실패

- [ ] **Step 3: 타입과 검증을 구현한다**

`PreparedGradient`를 그라디언트 좌표가 **경로 로컬 좌표계**임을 doc 주석에 명시하고 정의한다.

`PreparedPath::validate`에 그라디언트 검증을 추가한다: `stops.len()`이 2..=32, 모든 `offset`이 유한하고 `0.0..=1.0`, 중복 offset 없음. `Solid` 검증은 바뀌지 않는다.

`PreparedGradientKind`의 `Linear`/`Radial`/`Sweep`은 각각 비유한 좌표와 0 길이/0 반지름을 거부한다. Task 1의 검증과 같은 규칙이지만 **이 크레이트는 `koharu-scene`에 의존하지 않으므로 중복이 아니라 경계 검증이다** — 이 사실을 주석에 남긴다.

- [ ] **Step 4: `compile_scene`에서 `peniko::Gradient`를 조립한다**

`crates/koharu-rasterizer/src/frame.rs`의 `FillPath` 분기에서 `PreparedFill::Gradient`일 때:

1. `stops_sorted(&gradient.stops)`로 정렬한다.
2. `peniko::{ColorStop, ColorStops, Gradient, GradientKind, Extend}`를 만든다. `peniko::Gradient::new_linear(start, end)` / `new_radial(center, radius)` / `new_sweep(center, start_angle, end_angle)`을 쓰고, `stops`를 `ColorStops`로 설정하고, `extend`를 `PreparedGradientExtend`에서 `peniko::Extend`로 매핑한다. `Sweep`의 각도는 도 단위이므로 `to_radians()`로 바꾼다.
3. `scene.fill(rule, affine, &gradient, Some(affine), &path)`를 호출한다. **`brush_transform`에 경로와 같은 아핀을 넘긴다** — 회전 정렬이 여기서 결정되며, 이것이 스펙이 정한 유일한 방법이다.
4. `Solid`는 기존대로 `brush_transform: None`을 넘긴다(색은 변환과 무관하다).

`PreparedFill`을 매치하는 다른 지점이 컴파일 오류로 드러나면 같은 방식으로 처리한다.

- [ ] **Step 5: 테스트가 통과하는지 확인한다**

Run: `cargo test -j 6 -p koharu-rasterizer --lib`
Expected: 기존 테스트 포함 전부 통과

- [ ] **Step 6: 크레이트가 컴파일되는지 확인한다**

Run: `cargo check -j 6 -p koharu-rasterizer`
Expected: 성공

- [ ] **Step 7: 커밋**

```bash
git add crates/koharu-rasterizer/src/prepared.rs crates/koharu-rasterizer/src/frame.rs crates/koharu-rasterizer/src/lib.rs
git commit -m "feat(rasterizer): render gradient fills through peniko"
```

---

### Task 3: 도형 경로에 그라디언트 배선

`shape_renderer`가 `ShapeFill::Gradient`를 정규화 → 로컬 변환해 넘긴다.

**Files:**
- Modify: `crates/koharu-renderer/src/shape_renderer.rs:287-297` (`fill`), `:147-159` (`prepared`)
- Test: `crates/koharu-renderer/src/shape_renderer.rs` (인라인 `mod tests`)

**Interfaces:**
- Consumes: Task 1의 `GradientSpec` 등, Task 2의 `PreparedFill::Gradient`/`PreparedGradient`/`PreparedGradientKind`/`PreparedGradientStop`/`PreparedGradientExtend`
- Produces: `fill()`이 그라디언트를 펴서 반환한다. 시그니처는 그대로 `fn fill(style: &ShapeStyle, frame: ShapeFrame, has_head: bool) -> Option<PreparedFill>` — **`frame` 인자가 추가된다**

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```rust
    fn gradient_style() -> ShapeStyle {
        ShapeStyle {
            fill: Some(ShapeFill::Gradient(GradientSpec {
                kind: GradientKind::Linear { start: [0.0, 0.0], end: [1.0, 1.0] },
                stops: vec![
                    GradientStop { offset: 0.0, color: [255, 0, 0, 255] },
                    GradientStop { offset: 1.0, color: [0, 0, 255, 255] },
                ],
                extend: GradientExtend::Pad,
            })),
            stroke: None,
        }
    }

    #[test]
    fn a_normalized_gradient_is_spread_onto_the_frame() {
        let path = shape_path(
            &ShapeGeometry::Rectangle { corner_radius: 0.0 },
            &gradient_style(),
            frame(200.0, 100.0, 0.0),
        );
        let Some(PreparedFill::Gradient { gradient, .. }) = path.fill else {
            panic!("a gradient fill must survive preparation");
        };
        let PreparedGradientKind::Linear { start, end } = gradient.kind else {
            panic!("the linear kind must survive preparation");
        };
        // 정규화 (0,0)->(1,1) 이 200x100 프레임에서 (0,0)->(200,100) 이 된다.
        assert!((start[0] - 0.0).abs() < 1e-6);
        assert!((end[0] - 200.0).abs() < 1e-6);
        assert!((end[1] - 100.0).abs() < 1e-6);
    }

    #[test]
    fn a_rotated_shape_keeps_its_gradient_in_local_coordinates() {
        // 회전 정렬은 렌더러의 brush_transform 몫이므로, 준비된 그라디언트는
        // 회전한 도형에서도 로컬 좌표에 남아 있어야 한다.
        //
        // Rectangle 을 쓰는 이유: Line 의 끝점은 `to_local` 로 각도에 따라
        // 바뀌므로 `Line` 으로는 이 단언이 성립하지 않는다. 그라디언트 좌표는
        // 프레임의 폭·높이에만 의존하고 각도에는 의존하지 않는다.
        let upright = shape_path(
            &ShapeGeometry::Rectangle { corner_radius: 0.0 },
            &gradient_style(),
            frame(200.0, 100.0, 0.0),
        );
        let rotated = shape_path(
            &ShapeGeometry::Rectangle { corner_radius: 0.0 },
            &gradient_style(),
            frame(200.0, 100.0, 90.0),
        );
        assert_eq!(upright.fill, rotated.fill);
        assert_ne!(upright.transform, rotated.transform);
    }

    #[test]
    fn a_gradient_fill_with_a_none_fill_stays_none() {
        let style = ShapeStyle { fill: None, stroke: None };
        let path = shape_path(
            &ShapeGeometry::Rectangle { corner_radius: 0.0 },
            &style,
            frame(200.0, 100.0, 0.0),
        );
        assert!(path.fill.is_none());
    }

    #[test]
    fn an_arrow_head_gradient_uses_the_stroke_colour_when_fill_is_absent() {
        let style = ShapeStyle {
            fill: None,
            stroke: Some(StrokeStyle {
                color: [9, 9, 9, 255],
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
        assert_eq!(
            path.fill,
            Some(PreparedFill::Solid { color: [9, 9, 9, 255], rule: FillRule::NonZero })
        );
    }
```

기존 `Solid` 테스트들이 `fill`의 시그니처 변경으로 깨지면 `frame(...)` 인자를 넘기도록 고친다.

- [ ] **Step 2: 실패를 확인한다**

Run: `cargo test -j 6 -p koharu-renderer --lib shape_renderer`
Expected: 컴파일 실패 (`ShapeFill::Gradient` 미처리, `fill` 인자 불일치)

- [ ] **Step 3: `fill`을 넓힌다**

`fill(style, frame, has_head)`가:

- `Some(ShapeFill::Solid(color))` → 기존대로 `PreparedFill::Solid`
- `Some(ShapeFill::Gradient(spec))` → `PreparedFill::Gradient`. 각 `GradientKind`의 좌표에 `local = normalized * frame_extent`를 적용한다. `radius`는 프레임 폭 기준이므로 `radius * frame.width`다. `Sweep` 각도는 정규화 대상이 아니므로 그대로 옮긴다. `stops`는 그대로 옮긴다(정렬은 렌더러가 한다).
- `None` + `has_head` → 기존대로 스트로크 색의 `PreparedFill::Solid`
- `None` → `None`

**정규화 → 로컬 변환이 이 함수 한 곳에서만 일어난다는 사실을 주석에 남긴다.** 이것이 스펙 G3의 요점이고, 두 번째 변환 지점이 생기면 Task F1에서 고친 종류의 이중 권위 버그가 된다.

- [ ] **Step 4: `prepared()`가 `frame`을 넘기게 한다**

`prepared()`는 이미 `frame`을 갖고 있으므로 `fill(style, frame, has_head)`로 호출만 바꾼다.

- [ ] **Step 5: 테스트가 통과하는지 확인한다**

Run: `cargo test -j 6 -p koharu-renderer --lib`
Expected: 전부 통과

- [ ] **Step 6: 커밋**

```bash
git add crates/koharu-renderer/src/shape_renderer.rs
git commit -m "feat(renderer): spread gradient fills onto the shape frame"
```

---

### Task 4: wire 계약 고정

TS 클라이언트가 의존할 정확한 JSON을 `serde_json`으로 못 박는다. 1a가 이 테스트로 클라이언트 드리프트를 막았고, 1b도 같다.

**Files:**
- Modify: `crates/koharu-app/src/commands/project.rs` (기존 wire 계약 테스트 옆)
- Test: `crates/koharu-app/src/commands/project.rs` (인라인 `mod tests`)

**Interfaces:**
- Consumes: Task 1의 `GradientSpec` 등, Task 3의 `fill`
- Produces: 없음 (검증만). Task 6의 TS 타입이 이 테스트가 고정한 리터럴에 맞춘다

- [ ] **Step 1: 실패하는 테스트를 쓴다**

기존 `the_shape_wire_shape_matches_the_typescript_client` 옆에 추가한다.

```rust
    #[test]
    fn the_gradient_wire_shape_matches_the_typescript_client() {
        let style = ShapeStyle {
            fill: Some(ShapeFill::Gradient(GradientSpec {
                kind: GradientKind::Linear { start: [0.0, 0.0], end: [1.0, 1.0] },
                stops: vec![
                    GradientStop { offset: 0.0, color: [1, 2, 3, 255] },
                    GradientStop { offset: 1.0, color: [4, 5, 6, 255] },
                ],
                extend: GradientExtend::Repeat,
            })),
            stroke: None,
        };
        assert_eq!(
            serde_json::to_string(&style).unwrap(),
            r#"{"fill":{"gradient":{"kind":{"linear":{"start":[0.0,0.0],"end":[1.0,1.0]}},"stops":[{"offset":0.0,"color":[1,2,3,255]},{"offset":1.0,"color":[4,5,6,255]}],"extend":"repeat"}},"stroke":null}"#
        );

        let radial = ShapeFill::Gradient(GradientSpec {
            kind: GradientKind::Radial { center: [0.5, 0.5], radius: 0.25 },
            stops: vec![
                GradientStop { offset: 0.0, color: [0, 0, 0, 255] },
                GradientStop { offset: 1.0, color: [255, 255, 255, 255] },
            ],
            extend: GradientExtend::Pad,
        });
        assert_eq!(
            serde_json::to_string(&radial).unwrap(),
            r#"{"gradient":{"kind":{"radial":{"center":[0.5,0.5],"radius":0.25}},"stops":[{"offset":0.0,"color":[0,0,0,255]},{"offset":1.0,"color":[255,255,255,255]}],"extend":"pad"}}"#
        );
    }

    #[test]
    fn the_solid_wire_shape_is_unchanged_by_the_gradient_variant() {
        let style = ShapeStyle {
            fill: Some(ShapeFill::Solid([1, 2, 3, 255])),
            stroke: None,
        };
        assert_eq!(
            serde_json::to_string(&style).unwrap(),
            r#"{"fill":{"solid":[1,2,3,255]},"stroke":null}"#
        );
    }
```

`revision`의 `revisioned` 파생이 `serde`에도 영향을 주지 않으므로 `serde_json`은 `ShapeFill`의 serde 파생을 그대로 쓴다. 기존 `Solid` 리터럴은 1a가 고정한 것과 **문자 단위로 같아야** 한다 — 다르면 그 자체가 회귀다.

- [ ] **Step 2: 실패를 확인한다**

Run: `cargo test -j 6 -p koharu-app --lib commands::project::tests::the_gradient_wire`
Expected: 실패 (리터럴이 실제 직렬화와 다르거나 컴파일 실패)

- [ ] **Step 3: 리터럴을 실제 출력에 맞춘다**

위 리터럴은 serde 외부 태그 규칙에서 **유도한 예상값**이다. 기존 1a 테스트가 `f32`를 `2.0`으로, 구조 변형을 `{"rectangle":{...}}`로 고정한 선례가 있으므로(`project.rs:3287`, `:3301`, `:3305`) 그 형식을 따랐다.

실제 출력이 다르면 **테스트를 실제 출력에 맞추되, 그 출력이 스펙의 의도인지 확인한다.** 특히:

- `f32`의 JSON 표기(`0.0` vs `0`) — TS 타입이 이 표기를 받아야 한다
- `extend`의 태그 표기(`"repeat"` snake_case) — `#[serde(rename_all = "snake_case")]`가 있으므로 소문자여야 한다
- `kind`의 외부 태그(`{"linear":{...}}`) — 구조 변형이므로 객체여야 한다

어떤 값이 나오든 **두 리터럴을 확정하는 것이 이 태스크의 목적**이다. Task 5가 그것을 그대로 옮긴다. 실제 출력이 위와 달라진 부분을 보고에 남긴다.

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

Run: `cargo test -j 6 -p koharu-app --lib commands::project::tests`
Expected: 전부 통과 (기존 `Solid` 회귀 테스트 포함)

- [ ] **Step 5: 커밋**

```bash
git add crates/koharu-app/src/commands/project.rs
git commit -m "test(app): pin the gradient wire shape"
```

---

### Task 5: 전송 계층과 프론트 조립

TS 타입과 미리보기 조립을 넓힌다. 서버 커맨드는 바뀌지 않는다 — `add_shape`/`set_shape_style`은 이미 `ShapeStyle` 전체를 받는다.

**Files:**
- Modify: `packages/bridge/src/protocol.ts:1006-1030` (`ShapeFill` 등)
- Modify: `packages/bridge/src/canvas.ts:25-47` (`CanvasPathFill`, `CanvasPath`)
- Modify: `packages/koharu/lib/shapes.ts:108-170` (`defaultShapeStyle`, `shapePreviewPath`, `defaultGradientSpec`)
- Modify: `packages/koharu/tests/lib/shapes.test.ts`
- Test: `packages/koharu/tests/lib/shapes.test.ts`

**Interfaces:**
- Consumes: Task 1의 Rust 타입이 직렬화하는 JSON(Task 4가 고정), Task 2의 `PreparedPath` wire 형태
- Produces:
  - TS `ShapeFill = { solid: [number,number,number,number] } | { gradient: GradientSpec }`
  - TS `GradientSpec { kind: GradientKind, stops: GradientStop[], extend: GradientExtend }`
  - TS `GradientKind = { linear: { start: [number,number], end: [number,number] } } | { radial: { center: [number,number], radius: number } } | { sweep: { center: [number,number], start_degrees: number, end_degrees: number } }`
  - TS `GradientStop { offset: number, color: [number,number,number,number] }`
  - TS `GradientExtend = 'pad' | 'repeat' | 'reflect'`
  - `CanvasPathFill = { Solid: {...} } | { Gradient: { gradient: CanvasGradient, rule: 'NonZero' | 'EvenOdd' } }`
  - `defaultGradientSpec(kind: 'linear' | 'radial'): GradientSpec`
  - `shapePreviewPath`가 그라디언트를 조립

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/lib/shapes.test.ts`에 추가한다.

```ts
describe('defaultGradientSpec', () => {
  it('produces a two-stop linear gradient in normalized coordinates', () => {
    const spec = defaultGradientSpec('linear')
    expect(spec.kind).toEqual({ linear: { start: [0, 0], end: [1, 0] } })
    expect(spec.stops).toHaveLength(2)
    expect(spec.stops[0].offset).toBe(0)
    expect(spec.stops[1].offset).toBe(1)
    expect(spec.extend).toBe('pad')
  })

  it('keeps every normalized coordinate inside the unit square', () => {
    for (const kind of ['linear', 'radial'] as const) {
      const spec = defaultGradientSpec(kind)
      const numbers =
        'linear' in spec.kind
          ? [...spec.kind.linear.start, ...spec.kind.linear.end]
          : [...spec.kind.radial.center, spec.kind.radial.radius]
      expect(numbers.every((value) => Number.isFinite(value))).toBe(true)
      expect(numbers.filter((value) => value < 0 || value > 1)).toHaveLength(0)
    }
  })
})

describe('shapePreviewPath with a gradient', () => {
  it('spreads normalized coordinates onto the frame and spells the wire shape', () => {
    // 리터럴을 직접 넘겨 contextual typing 으로 튜플 타입을 얻는다 — 이 파일의
    // 기존 테스트가 쓰는 방식이다(`as const` 를 쓰지 않는다).
    const path = shapePreviewPath('rectangle', frame, {
      fill: {
        gradient: {
          kind: { linear: { start: [0, 0], end: [1, 1] } },
          stops: [
            { offset: 0, color: [1, 2, 3, 255] },
            { offset: 1, color: [4, 5, 6, 255] },
          ],
          extend: 'pad',
        },
      },
      stroke: null,
    })

    expect(path).toEqual({
      transform: [1, 0, 0, 1, 0, 0],
      elements: [
        { MoveTo: [10, 20] },
        { LineTo: [110, 20] },
        { LineTo: [110, 70] },
        { LineTo: [10, 70] },
        'Close',
      ],
      // 서버와 같은 규칙: local = normalized * frame_extent.
      // `frame` 은 이 파일 상단의 {x:10,y:20,width:100,height:50} 이다.
      fill: {
        Gradient: {
          gradient: {
            kind: { Linear: { start: [10, 20], end: [110, 70] } },
            stops: [
              { offset: 0, color: [1, 2, 3, 255] },
              { offset: 1, color: [4, 5, 6, 255] },
            ],
            extend: 'Pad',
          },
          rule: 'NonZero',
        },
      },
      stroke: null,
    })
  })

  it('leaves the transform to carry the rotation', () => {
    const style = { fill: { gradient: defaultGradientSpec('linear') }, stroke: null }
    const upright = shapePreviewPath(
      'rectangle',
      { x: 0, y: 0, width: 200, height: 100, angle_degrees: 0 },
      style,
    )
    const rotated = shapePreviewPath(
      'rectangle',
      { x: 0, y: 0, width: 200, height: 100, angle_degrees: 90 },
      style,
    )
    // 그라디언트는 로컬 좌표에 남고 회전은 transform 이 담당한다.
    expect(rotated.fill).toEqual(upright.fill)
    expect(rotated.transform).not.toEqual(upright.transform)
  })
})
```

`{ Gradient: ... }`의 정확한 철자(`Gradient` 대문자, `extend: 'Pad'`)는 Task 2가 만든 Rust `PreparedGradientExtend`의 serde 출력에 맞춘다. **Task 2를 마친 뒤 그 크레이트의 `serde_json` 출력이나 `prepared.rs`의 `#[serde(rename_all)]` 속성을 확인해 확정한다** — `PreparedFill::Solid`가 PascalCase(`Solid`)인 선례를 따른다.

**좌표 기대값은 이 파일 상단의 `frame` 상수에서 나온다.** 그 상수를 쓰므로 `start`는 `(10,20)`, `end`는 `(110,70)`이다. 위 리터럴의 값을 손으로 계산해 넣지 말고 그 상수에서 유도한다.

- [ ] **Step 2: 실패를 확인한다**

Run: `bun run --filter '@koharu/app' test -- tests/lib/shapes.test.ts`
Expected: FAIL — `defaultGradientSpec` 미정의

- [ ] **Step 3: TS 타입을 넓힌다**

`protocol.ts`의 `ShapeFill`을 유니온으로 넓히고 `GradientSpec` 계열을 추가한다. **Task 4가 고정한 JSON 리터럴과 정확히 일치**해야 한다.

`canvas.ts`의 `CanvasPathFill`을 넓히고 `CanvasGradient`/`CanvasGradientKind`/`CanvasGradientStop`을 추가한다. 서버 `PreparedGradient`의 serde 형태를 따른다.

- [ ] **Step 4: `defaultGradientSpec`과 `shapePreviewPath`를 구현한다**

`defaultGradientSpec(kind)`는 정규화 좌표의 2스톱 그라디언트를 만든다: linear는 `start:[0,0], end:[1,0]`, radial은 `center:[0.5,0.5], radius:0.5`. 스톱은 흰색→검은색.

`shapePreviewPath`는 `style.fill`이 그라디언트일 때 같은 `local = normalized * frame_extent` 규칙으로 제어점을 펴고, `extend`를 PascalCase로, `cap`/`join`과 같은 방식으로 철자를 맞춘다. **회전은 `transform`이 담당하므로 그라디언트 조립에 각도를 쓰지 않는다.**

- [ ] **Step 5: 테스트가 통과하는지 확인한다**

Run: `bun run --filter '@koharu/app' test -- tests/lib/shapes.test.ts`
Expected: PASS

- [ ] **Step 6: 타입 검사**

Run: `bunx tsc --noEmit -p packages/koharu/tsconfig.json`
Expected: exit 0. `packages/bridge/src/wasm`가 오래되어 `CanvasHandle` 관련 오류가 나면 `bun run --filter @koharu/bridge build`를 먼저 돌린다.

- [ ] **Step 7: 커밋**

```bash
git add packages/bridge/src/protocol.ts packages/bridge/src/canvas.ts packages/koharu/lib/shapes.ts packages/koharu/tests/lib/shapes.test.ts
git commit -m "feat(ui): speak the gradient wire shape in the client"
```

---

### Task 6: 스톱 편집 UI와 i18n

Inspector에서 채우기 종류를 고르고 스톱을 편집한다.

**Files:**
- Modify: `packages/koharu/components/editor/Inspector.tsx:144-280` (`ShapeInspector`)
- Modify: `packages/koharu/components/editor/ToolBar.tsx` (스타일 팝오버의 기본 채우기)
- Modify: `packages/koharu/public/locales/*/translation.json` (9개)
- Modify: `packages/koharu/tests/components/editor-components.test.tsx`

**Interfaces:**
- Consumes: Task 5의 `defaultGradientSpec`, TS `GradientSpec` 계열
- Produces: `ShapeInspector`가 채우기 종류(단색/그라디언트)와 그라디언트 종류(선형/방사), 스톱 목록을 편집한다

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`editor-components.test.tsx`에 추가한다. 기존 shape 테스트가 스냅샷을 만드는 방식을 따른다.

```tsx
  it('switches a shape fill from solid to a gradient', async () => {
    // 기존 헬퍼로 도형 레이어를 선택한 상태를 만든다.
    // ...
    await user.click(screen.getByRole('button', { name: /gradient/i }))
    expect(commitStyle).toHaveBeenCalledWith(
      expect.objectContaining({
        fill: expect.objectContaining({ gradient: expect.anything() }),
      }),
    )
  })

  it('keeps the stop count within the validated range', async () => {
    // 스톱 추가를 32번 넘게 눌러도 32개를 넘지 않는다.
  })
```

`commitStyle`은 스타일��� 커밋하는 경로(기존 `setShapeStyle` 호출)를 가리키는 스파이로 맞춘다. 기존 테스트가 쓰는 방식을 확인해 이름을 맞춘다.

- [ ] **Step 2: 실패를 확인한다**

Run: `bun run --filter '@koharu/app' test -- tests/components/editor-components.test.tsx`
Expected: FAIL

- [ ] **Step 3: `ShapeInspector`를 넓힌다**

기존 채우기 `ColorWell`(`:175-190`)을 **채우기 종류 선택 + 조건부 편집기**로 바꾼다:

- 단색 → 기존 `ColorWell` 그대로
- 그라디언트 → 종류 선택(선형/방사), 스톱 목록(색 `ColorWell` + 위치 `Slider`), 스톱 추가/삭제 버튼

**스톱 상한을 UI에서 강제한다**: 추가 버튼은 32개에서 비활성화되고, 삭제 버튼은 2개에서 비활성화된다. 서버 검증이 최종 방어이지만 UI가 거부당할 입력을 만들 수 있으면 안 된다.

**스윕은 UI에 노출하지 않는다**(스펙 G5). 종류 선택은 선형/방사 둘뿐이다.

스톱을 정렬해서 보여줄 필요는 없다 — 서버가 렌더링 시 정렬하고, 저장 순서는 자유다(스펙). 다만 위치를 바꾸면 목록을 `offset` 순으로 다시 그리면 편집이 자연스러워진다.

- [ ] **Step 4: `ToolBar`의 기본 스타일을 넓힌다**

도구막대 팝오버의 채우기도 같은 선택을 제공한다. 스톱 편집기 전체를 넣을 필요는 없다 — 종류 선택만 하고 세부 편집은 Inspector에서 한다는 결정을 주석에 남긴다.

- [ ] **Step 5: 9개 로케일에 키를 추가한다**

필요한 키:

- `inspector.shape.fillKind.solid`, `.gradient`
- `inspector.shape.gradientKind.linear`, `.radial`
- `inspector.shape.stops`, `inspector.shape.addStop`, `inspector.shape.removeStop`, `inspector.shape.stopOffset`
- `inspector.shape.tooFewStops`, `inspector.shape.tooManyStops`

`ko-KR`은 실제 한국어로("도형" 용어를 1a의 `inspector.shape.*`와 통일). 나머지 8개는 각 파일의 기존 관행을 따른다(1a에서 8개 모두 실제 번역이 들어갔으므로 같은 수준을 유지한다).

**키 개수와 이름이 9개 파일에서 동일해야 한다.** 1a에서 스크립트 편집이 로케일을 손상시킨 사례가 있으므로, 편집 후 키 집합이 9개 모두 같은지 확인한다.

- [ ] **Step 6: 프론트 검사를 돌린다**

Run: `bun run --filter '@koharu/app' test`
Expected: 전부 통과

Run: `bunx oxlint -c packages/koharu/.oxlintrc.json packages/koharu`
Expected: clean

- [ ] **Step 7: 커밋**

```bash
git add packages/koharu/components/editor/Inspector.tsx packages/koharu/components/editor/ToolBar.tsx packages/koharu/public/locales packages/koharu/tests/components/editor-components.test.tsx
git commit -m "feat(ui): edit gradient stops in the shape inspector"
```

---

### Task 7: 회귀 확인과 전체 검사

1a의 `Solid` 경로가 그대로이고 전체가 컴파일되는지 확인한다.

**Files:**
- Test: `crates/koharu-rasterizer/src/frame.rs` (`compile_scene`의 `brush_transform`)

**Interfaces:**
- Consumes: Task 2의 `compile_scene`, Task 3의 `fill`
- Produces: 없음 (검증만)

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`compile_scene`이 그라디언트에 `brush_transform`을 넘기는지, `Solid`에는 넘기지 않는지 고정한다. `Scene`의 내부는 읽을 수 없으므로 **`compile_scene`을 작은 순수 함수로 뽑아** 검증한다:

```rust
/// The brush transform a prepared fill needs: a gradient must rotate with its
/// path, a solid colour does not.
pub(crate) fn brush_transform(fill: &PreparedFill, path: &PreparedPath) -> Option<Affine> {
    match fill {
        PreparedFill::Solid { .. } => None,
        PreparedFill::Gradient { .. } => Some(Affine::new(path.transform)),
    }
}
```

테스트:

```rust
    #[test]
    fn a_gradient_brush_rotates_with_its_path() {
        let path = PreparedPath {
            transform: [0.0, 1.0, -1.0, 0.0, 10.0, 20.0],
            elements: vec![PathElement::MoveTo([0.0, 0.0]), PathElement::Close],
            fill: Some(gradient_fill()),
            stroke: None,
        };
        let Some(fill) = path.fill.as_ref() else { unreachable!() };
        assert_eq!(brush_transform(fill, &path), Some(Affine::new(path.transform)));
    }

    #[test]
    fn a_solid_brush_needs_no_transform() {
        let path = PreparedPath {
            transform: [0.0, 1.0, -1.0, 0.0, 10.0, 20.0],
            elements: vec![PathElement::MoveTo([0.0, 0.0]), PathElement::Close],
            fill: Some(PreparedFill::Solid { color: [1, 2, 3, 255], rule: FillRule::NonZero }),
            stroke: None,
        };
        let Some(fill) = path.fill.as_ref() else { unreachable!() };
        assert_eq!(brush_transform(fill, &path), None);
    }
```

- [ ] **Step 2: 실패를 확인한다**

Run: `cargo test -j 6 -p koharu-rasterizer --lib frame::tests::a_gradient_brush`
Expected: 컴파일 실패

- [ ] **Step 3: `brush_transform`을 구현하고 `compile_scene`이 그것을 쓰게 한다**

Task 2에서 인라인으로 match하던 것을 이 함수로 바꾼다. 동작은 같다 — 이 태스크는 그 결정을 테스트 가능하게 만드는 것이다.

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

Run: `cargo test -j 6 -p koharu-rasterizer --lib`
Expected: 전부 통과

- [ ] **Step 5: 전체를 한 번에 확인한다**

Run: `cargo test -j 6 -p koharu-scene -p koharu-rasterizer -p koharu-renderer -p koharu-app -p koharu-psd --lib`
Expected: 전부 통과

Run: `cargo check -j 6 --workspace`
Expected: 성공, 새 경고 없음

Run: `bun run --filter '@koharu/app' test`
Expected: 전부 통과

`-j 6`은 이 머신에서 필수다 (12코어, 무제한 병렬은 데스크톱을 멈춘다 — `fork/HANDOFF.md:44`).

- [ ] **Step 6: 커밋**

```bash
git add crates/koharu-rasterizer/src/frame.rs
git commit -m "test(rasterizer): pin the gradient brush transform"
```

---

## Self-Review

**Spec coverage**

| 스펙 절 | 태스크 |
|---|---|
| 범위 결정 (패턴은 1c로 분리) | 태스크 없음 — 계획 범위 밖이며 스펙에 기록됨 |
| G1·G2 `GradientSpec` 자체 타입 | Task 1 |
| G3 정규화 프레임 로컬 좌표 | Task 1 (저장) + Task 3 (변환) + Task 5 (프론트 동일 규칙) |
| G4 `PreparedFill` 변형 추가 | Task 2 |
| G5 스윕 포함·UI 미노출 | Task 1·2 (데이터) + Task 6 (UI에서 제외) |
| G6 스톱 2..=32 | Task 1 (검증) + Task 2 (경계 검증) + Task 6 (UI 상한) |
| G7 `Solid` 유지 | Task 3 (match 유지) + Task 4 (회귀 테스트) |
| 저장 모델 타입·검증 | Task 1 |
| wire 형태 | Task 4 (Rust 고정) + Task 5 (TS 일치) |
| `PreparedFill` 확장·`brush_transform` | Task 2 + Task 7 |
| 렌더러 조립 (`peniko::Gradient`) | Task 2 |
| 캔버스 미리보기 | Task 5 |
| UI (Inspector·ToolBar·i18n) | Task 6 |
| 실패 시나리오 표 | Task 1 (검증), Task 2 (경계), Task 3 (`fill: None`, 화살촉) |
| 테스트 표 | 각 태스크 + Task 7 |
| 기존 프로젝트 호환 | Task 1 (`Solid` 바이트 테스트) + Task 4 (`Solid` JSON 회귀) |

**Spec gap을 계획이 메운 곳** (스펙에 없던 결정):

1. **`koharu-rasterizer`가 `koharu-scene`에 의존하지 않는다.** 스펙은 `PreparedGradient`를 정의했지만 검증 주체를 정하지 않았다. 1a의 경계(이 크레이트는 vello를 알고 scene은 모른다)를 유지해야 하므로 `GradientSpec` 검증과 `PreparedGradient` 검증이 **같은 규칙을 두 크레이트에서 각각** 구현한다. 중복이 아니라 경계 검증이며, 주석에 그 이유를 남긴다 (Task 2).
2. **`brush_transform`을 순수 함수로 뽑는다.** 스펙은 그 인자를 쓰라고만 했고 검증 방법을 정하지 않았다. `Scene` 내부를 읽을 수 없으므로 결정을 함수로 떼어내 테스트한다 (Task 7).
3. **`fill`에 `frame` 인자가 추가된다.** 스펙은 "정규화 → 로컬 변환은 준비 단계"라고만 했다. `prepared()`가 이미 `frame`을 갖고 있으므로 시그니처만 넓히면 된다 (Task 3).
4. **`ShapeFill::Gradient`를 마지막 변형으로 둔다.** 스펙은 "리비전을 올리지 않는다"고 했고 revision의 태그 형식으로 그 근거를 댔지만, **어느 위치에 추가하는지**는 명시하지 않았다. 마지막이어야 `Solid`의 id 0이 유지된다 (Task 1).
5. **프론트의 `extend`/`kind` 철자를 Task 2의 serde 출력에서 확정한다.** 스펙은 wire 형태를 예시로만 들었다. Task 5 Step 1이 그 확인 절차를 명시한다.

**Type consistency 확인**

- `GradientKind::Linear { start: [f32;2], end: [f32;2] }`의 필드명 `start`/`end`는 Task 1·5에서 동일하고, `PreparedGradientKind`의 같은 이름과도 대응한다.
- `GradientExtend::{Pad, Repeat, Reflect}`는 Task 1(저장) → Task 2(`peniko::Extend` 매핑) → Task 5(TS `'pad'|'repeat'|'reflect'`)에서 snake_case와 PascalCase가 각각 올바르다: 저장은 `rename_all = "snake_case"`, wire는 외부 태그 단위 변형이라 `"pad"` 소문자다.
- `fill()`의 시그니처는 Task 3에서 `(style, frame, has_head)`로 확정되며 Task 7이 그것을 소비하지 않는다(별개 함수).
- `defaultGradientSpec`은 Task 5에서 정의되고 Task 6이 소비한다. 이름 하나만 쓴다.
- `stops_sorted`는 Task 2에서 `pub`으로 만들고 `compile_scene`만 쓴다. 다른 태스크가 참조하지 않는다.

**Proportion**

스펙 345줄, 계획 약 700줄. 코드 블록은 테스트 리터럴과 정확한 시그니처 위주이며, 함수 본문은 알고리즘이 시그니처와 테스트로 결정되지 않는 곳(Task 2의 peniko 조립, Task 3의 좌표 변환)에만 있다.
