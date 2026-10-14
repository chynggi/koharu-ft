# 색보정 (B3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 래스터 레이어에 밝기·대비·레벨·색조·흑백 보정을 적용하고, 실행취소로 되돌릴 수 있게 한다.

**Architecture:** `Project::adjust_raster`가 대상 레이어의 PNG 자산을 읽어 `RgbaImage`로 디코드하고, `ColorAdjust`에 따라 제자리 보정한 뒤 다시 PNG로 커밋한다. 기존 `apply_raster_stroke`의 자산 경로(`set_asset` + 페이지 크기 검사)를 그대로 재사용하므로 되돌리기·동기화·PSD가 자동으로 맞는다. Inspector의 래스터 섹션에 보정 컨트롤을 붙인다.

**Tech Stack:** Rust (koharu-app, koharu-rpc), `image` 0.25.10, `imageproc` 0.27.0, TypeScript (Next.js, zustand, i18next), `cargo test`, vitest.

**Spec:** `specs/2026-10-14-paint-fill-adjust-design.md`

**계획 위치 메모:** 저장소 관례대로 `plans/`에 둔다. `docs/`는 발행용 문서 사이트다.

**이 계획의 범위:** B의 세 덩어리 중 **B3(색보정)만**이다. B2(칠하기)와 B1(브러시 그라디언트)은 각자의 계획에서 다룬다. B3을 먼저 하는 이유는 가장 독립적이기 때문이다 — 새 알고리즘이 없고 기존 래스터 자산 경로를 재사용한다.

---

## Global Constraints

- **모든 보정은 알파를 보존한다.** 래스터 레이어의 투명 영역이 의미를 갖는다(1a가 래스터화 배경을 `[0,0,0,0]`으로 못 박았다). `image::imageops::grayscale`은 알파를 **버리므로 쓰지 않는다** — 문서에 "Alpha channel is discarded"라고 명시되어 있다(`image-0.25.10/src/imageops/colorops.rs:13`).
- `Contrast`의 인자는 **백분율**이다: `image::imageops::contrast_in_place(image, percent: f32)`의 내부가 `((100.0 + percent) / 100.0).powi(2)`(`image-0.25.10/src/imageops/colorops.rs:132`)이다. `0.0`이 무변화이고 범위는 `-100.0..=100.0`이며 `-1.0..=1.0`이 아니다.
- `Levels { low, high }`는 `stretch_contrast_mut`의 **입력** 범위에 매핑하고 출력은 `0..=255`로 고정한다. 네 인자가 모두 `u8`이다(`imageproc-0.27.0/src/contrast.rs:368-375`).
- 보정 파라미터는 **저장하지 않는다.** 적용 결과가 픽셀이고, 저장하면 도형의 `ShapeStyle`처럼 두 권위가 생겨 1a·1b에서 가장 비쌌던 버그 유형이 재현된다.
- 대상은 **래스터 레이어만**이다. 도형·텍스트 레이어에는 픽셀 연산이 의미를 갖지 않는다.
- 자산 크기가 페이지 크기와 다르면 거부한다 (`apply_raster_stroke`의 기존 관례).
- 새 Tauri 커맨드는 **반드시** HTTP 라우트를 함께 추가한다 (`fork/HANDOFF.md:213`).
- 하위 호환·별칭 금지 (AGENTS.md).
- `-j 6`을 모든 cargo 명령에 붙인다 (12코어, 무제한 병렬은 데스크톱을 멈춘다).
- 검증은 가장 작은 관련 검사만 debug 프로필로 한 번 돌린다 (AGENTS.md).

## Review Focus

스펙이 명시하지 않았지만 실제로 깨질 수 있는 지점이다. 각 항목의 테스트는 그 코드를 소유한 태스크에 넣는다.

1. **투명 영역이 불투명해지지 않는지** — 특히 흑백. `grayscale`을 쓰면 알파가 사라져 페이지가 검게 덮인다 (Task 1).
2. **`Contrast`의 단위** — `-1.0..=1.0`으로 착각하면 `100.0`을 넘겨 대비가 폭발한다. `percent`가 실제로 백분율로 해석되는지 (Task 1).
3. **`Levels`에서 `low == high`** — 0으로 나누거나 모든 픽셀이 한 값이 된다. 거부해야 한다 (Task 2).
4. **보정이 원본을 파괴하지 않는지** — 같은 레이어에 두 번 적용하면 누적되어야 하고, `set_asset`이 새 blob을 쓰므로 이전 blob은 참조 해제되어야 한다 (Task 2).
5. **알파 0인 픽셀의 RGB가 변하지 않는지** — 보정이 완전 투명 픽셀의 색을 건드리면 나중에 알파를 올렸을 때 이상한 색이 나온다 (Task 1).

---

## File Structure

**신규**

| 파일 | 책임 |
|---|---|
| `crates/koharu-app/src/commands/adjust.rs` | `ColorAdjust` 타입, 검증, `RgbaImage`에 대한 순수 보정 함수. Tauri 의존성 없는 순수 로직 |
| `packages/koharu/tests/lib/adjust.test.ts` | 프론트 보정 파라미터 헬퍼 테스트 |

**수정**

| 파일 | 변경 |
|---|---|
| `crates/koharu-app/src/commands/mod.rs` | `adjust` 모듈 등록 |
| `crates/koharu-app/src/commands/project.rs` | `Project::adjust_raster`, 자산 읽기·쓰기 |
| `crates/koharu-app/src/commands/canvas.rs` | `adjust_raster` 커맨드 |
| `crates/koharu-app/Cargo.toml` | `imageproc` 직접 의존 추가 |
| `crates/koharu-rpc/src/routes/canvas.rs` | `POST /canvas/raster/adjust` |
| `packages/bridge/src/protocol.ts` | `ColorAdjust` 타입, `commands.adjustRaster` |
| `packages/koharu/components/editor/Inspector.tsx` | 래스터 색보정 섹션 |
| `packages/koharu/public/locales/*/translation.json` | 9개 로케일 |
| `packages/koharu/tests/components/editor-components.test.tsx` | 보정 컨트롤 테스트 |

---

### Task 1: 보정 커널 (순수 로직)

`RgbaImage`를 받아 보정하는 순수 함수를 세운다. Tauri·프로젝트에 의존하지 않으므로 여기가 이 기능에서 가장 촘촘히 테스트되는 곳이다.

**Files:**
- Create: `crates/koharu-app/src/commands/adjust.rs`
- Modify: `crates/koharu-app/src/commands/mod.rs`
- Modify: `crates/koharu-app/Cargo.toml`
- Test: `crates/koharu-app/src/commands/adjust.rs` (인라인 `mod tests`)

**Interfaces:**
- Consumes: `image::RgbaImage`, `image::imageops::{brighten_in_place, contrast_in_place, huerotate_in_place}`, `imageproc::contrast::stretch_contrast_mut`
- Produces:
  - `ColorAdjust::{Brightness { delta: i32 }, Contrast { percent: f32 }, Levels { low: u8, high: u8 }, Hue { degrees: i32 }, Grayscale}`
  - `ColorAdjust::validate(&self) -> Result<()>`
  - `pub fn apply_adjust(image: &mut RgbaImage, adjust: ColorAdjust)`

- [ ] **Step 1: 모듈과 의존을 등록한다**

`crates/koharu-app/src/commands/mod.rs`에 알파벳 순서로 추가한다:

```rust
pub mod adjust;
```

`crates/koharu-app/Cargo.toml`의 `[dependencies]`에 `image` 옆에 추가한다:

```toml
imageproc = { workspace = true }
```

`imageproc`은 워크스페이스에 이미 고정되어 있다 (`Cargo.toml:59`의 `imageproc = "0.27.0"`).

- [ ] **Step 2: 실패하는 테스트를 쓴다**

`crates/koharu-app/src/commands/adjust.rs`를 만들고 테스트만 먼저 넣는다.

```rust
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
        assert_ne!(raised, original, "a non-zero percentage must change the image");
    }

    #[test]
    fn levels_stretch_the_input_range_onto_the_full_range() {
        let mut value = image(&[[100, 100, 100, 255], [150, 150, 150, 255]]);
        apply_adjust(&mut value, ColorAdjust::Levels { low: 100, high: 150 });
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
        assert_eq!(value.get_pixel(1, 0)[3], 0, "a transparent pixel stays transparent");
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
        let adjust = ColorAdjust::Levels { low: 200, high: 100 };
        assert!(adjust.validate().is_err());
    }

    #[test]
    fn validate_rejects_equal_levels_bounds() {
        let adjust = ColorAdjust::Levels { low: 128, high: 128 };
        assert!(adjust.validate().is_err());
    }

    #[test]
    fn validate_rejects_an_out_of_range_contrast() {
        assert!(ColorAdjust::Contrast { percent: 101.0 }.validate().is_err());
        assert!(ColorAdjust::Contrast { percent: -101.0 }.validate().is_err());
        assert!(ColorAdjust::Contrast { percent: f32::NAN }.validate().is_err());
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
```

- [ ] **Step 3: 실패를 확인한다**

Run: `cargo test -j 6 -p koharu-app --lib commands::adjust`
Expected: 컴파일 실패 (`ColorAdjust` 등 미정의)

- [ ] **Step 4: 타입과 보정 함수를 구현한다**

`ColorAdjust`를 `Debug, Clone, Copy, PartialEq, Deserialize, Type`로 파생하고 `#[serde(tag = "type", rename_all = "snake_case")]`을 붙인다. 이 태그 이름은 이 저장소의 **커맨드 인자** 열거형 관례를 따른다 — `Layer`가 `#[serde(tag = "type", rename_all = "snake_case")]`이다(`crates/koharu-app/src/commands/project.rs:78`). `Deserialize`와 `Type`은 커맨드가 받기 위해, `PartialEq`와 `Copy`는 테스트가 쓰기 위해 필요하다.

**`ShapeFill`은 이 관례를 따르지 않는다.** 그것은 영속 컴포넌트라 외부 태그(`#[serde(rename_all = "snake_case")]`만)를 쓰고 wire 형태가 `{"solid":[…]}`다. `ColorAdjust`는 저장되지 않는 커맨드 인자이므로 내부 태그가 맞고, 그래서 `{"type":"brightness","delta":20}` 꼴이 된다. Task 3의 TS 타입이 이 형태를 그대로 옮긴다.

`validate`가 거부하는 것: `Levels`에서 `low >= high`, `Brightness.delta`가 `-255..=255` 밖, `Hue.degrees`가 `-180..=180` 밖, `Contrast.percent`가 비유한이거나 `-100.0..=100.0` 밖.

`apply_adjust`는 **알파 0인 픽셀을 건너뛴다**: 각 픽셀에 대해 `a == 0`이면 그대로 두고, 아니면 보정한다. 이 규칙을 함수 doc 주석에 남긴다.

각 변형:

- `Brightness { delta }` — `brighten_in_place(image, delta)`
- `Contrast { percent }` — `contrast_in_place(image, percent)`
- `Levels { low, high }` — `stretch_contrast_mut(image, low, high, 0, 255)`
- `Hue { degrees }` — `huerotate_in_place(image, degrees)`
- `Grayscale` — **직접 구현한다.** `image::imageops::grayscale`은 쓰지 않는다. 각 픽셀에 대해 `let luma = (0.2126*r + 0.7152*g + 0.0722*b)` (또는 이미 있는 가중치 관례가 있으면 그것)로 계산하고 `r=g=b=luma`, `a`는 그대로 둔다. 그레이 계수 선택 이유를 주석에 남긴다.

**알파 0 스킵이 `brighten_in_place` 같은 크레이트 함수와 함께 동작하지 않는다는 점에 주의한다.** 그 함수들은 이미지 전체를 순회하므로, 알파 0 스킵을 지키려면 **스킵이 필요한 변형(즉 모든 변형)에 대해 순회를 직접 소유**해야 한다. 다음 구조를 쓴다:

1. 원본을 복제해 `let before = image.clone()`
2. ���레이트 함수를 `image`에 적용
3. `before`와 비교해 `before`의 알파가 0인 픽셀만 `image`에서 되돌린다

이렇게 하면 크레이트 함수를 그대로 쓰면서 불변식을 지킬 수 있다. 이 이유를 주석에 남긴다.

- [ ] **Step 5: 테스트가 통과하는지 확인한다**

Run: `cargo test -j 6 -p koharu-app --lib commands::adjust`
Expected: 12 passed

- [ ] **Step 6: 컴파일을 확인한다**

Run: `cargo check -j 6 -p koharu-app`

- [ ] **Step 7: 커밋**

```bash
git add crates/koharu-app/src/commands/adjust.rs crates/koharu-app/src/commands/mod.rs crates/koharu-app/Cargo.toml Cargo.lock
git commit -m "feat(app): add the raster colour adjustment kernels"
```

---

### Task 2: 프로젝트 메서드와 커맨드

`Project::adjust_raster`가 자산을 읽고·보정하고·커밋한다.

**Files:**
- Modify: `crates/koharu-app/src/commands/project.rs`
- Test: `crates/koharu-app/src/commands/project.rs` (인라인 `mod tests`)

**Interfaces:**
- Consumes: Task 1의 `ColorAdjust`, `apply_adjust`; 기존 `apply_raster_stroke`(`project.rs:1458`)의 자산 경로
- Produces: `Project::adjust_raster(&mut self, layer: EntityId, adjust: ColorAdjust) -> Result<(Commit, EntityId)>`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`project.rs`의 `mod tests`에 추가한다. 기존 래스터 테스트가 쓰는 프로젝트·페이지·래스터 레이어 생성 헬퍼를 재사용한다.

```rust
    #[tokio::test]
    async fn adjusting_a_raster_rewrites_its_pixels() {
        let mut project = /* 기존 헬퍼 */;
        let layer = /* 100,100,100 불투명 픽셀 하나를 가진 래스터 레이어 */;

        let (commit, element) = project
            .adjust_raster(layer, ColorAdjust::Brightness { delta: 50 })
            .await
            .unwrap();
        assert_eq!(element, layer);
        assert!(commit.revision.get() > 0);

        let image = /* 커밋된 자산을 디코드 */;
        assert_eq!(image.get_pixel(0, 0)[0], 150);
    }

    #[tokio::test]
    async fn adjusting_a_raster_keeps_its_transparent_area_transparent() {
        let mut project = /* 기존 헬퍼 */;
        let layer = /* 알파 0 픽셀을 가진 래스터 레이어 */;

        project
            .adjust_raster(layer, ColorAdjust::Grayscale)
            .await
            .unwrap();

        let image = /* 커밋된 자산을 디코드 */;
        assert_eq!(image.get_pixel(0, 0)[3], 0);
    }

    #[tokio::test]
    async fn adjusting_a_non_raster_layer_is_rejected() {
        let mut project = /* 기존 헬퍼 */;
        let text = /* 텍스트 레이어 */;
        assert!(
            project
                .adjust_raster(text, ColorAdjust::Grayscale)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn adjusting_with_an_invalid_parameter_is_rejected() {
        let mut project = /* 기존 헬퍼 */;
        let layer = /* 래스터 레이어 */;
        assert!(
            project
                .adjust_raster(layer, ColorAdjust::Levels { low: 200, high: 100 })
                .await
                .is_err()
        );
    }
```

테스트 헬퍼가 없으면 `apply_raster_stroke`의 기존 테스트가 래스터 레이어를 어떻게 만드는지 보고 같은 방식을 쓴다.

- [ ] **Step 2: 실패를 확인한다**

Run: `cargo test -j 6 -p koharu-app --lib commands::project::tests::adjusting`
Expected: 컴파일 실패

- [ ] **Step 3: `Project::adjust_raster`를 구현한다**

1. `adjust.validate()?`
2. 페이지를 읽어 `width`·`height`를 얻는다
3. 대상이 래스터인지 확인한다 — `SceneRasterLayer` 컴포넌트가 없으면 `"only raster layers can be adjusted"`로 거부
4. `snapshot.asset(layer, &AssetRole::new("source")?)`로 자산을 읽고 `image::load_from_memory(&bytes)?.to_rgba8()`로 디코드한다. 자산이 없으면 빈 페이지 크기 이미지를 만든다 — `apply_raster_stroke`가 그렇게 한다
5. 디코드한 크기가 `(width, height)`와 다르면 거부
6. `apply_adjust(&mut image, adjust)`
7. PNG로 인코딩하고, `apply_raster_stroke`와 **같은 방식**으로 `edit.set_asset(layer, &source, AssetInput::new(bytes, "image/png", AssetMetadata { width: Some(width), height: Some(height), attributes: Default::default() }))`를 호출한다
8. `Origin::User`로 승격한다 (`promote_entity_to_user`)

**`apply_raster_stroke`의 레이어 생성 분기는 필요 없다** — 보정은 기존 레이어를 고치므로 새 레이어를 만들지 않는다. 이 차이를 주석에 남긴다.

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

Run: `cargo test -j 6 -p koharu-app --lib commands::project::tests::adjusting`
Expected: 4 passed

- [ ] **Step 5: 전체 `koharu-app` 스위트를 돌린다**

Run: `cargo test -j 6 -p koharu-app --lib`
Expected: 전부 통과

- [ ] **Step 6: 커밋**

```bash
git add crates/koharu-app/src/commands/project.rs
git commit -m "feat(app): adjust a raster layer's pixels"
```

---

### Task 3: 커맨드와 전송 계층

Tauri 커맨드, HTTP 라우트, TS 클라이언트를 잇는다.

**Files:**
- Modify: `crates/koharu-app/src/commands/canvas.rs`
- Modify: `crates/koharu-rpc/src/routes/canvas.rs`
- Modify: `packages/bridge/src/protocol.ts`

**Interfaces:**
- Consumes: Task 2의 `Project::adjust_raster`
- Produces:
  - 커맨드 `adjust_raster(expected_revision, layer, adjust) -> LayerCommit`
  - 라우트 `POST /canvas/raster/adjust`
  - TS `commands.adjustRaster(expectedRevision, layer, adjust)`

- [ ] **Step 1: 커맨드를 추가한다**

`crates/koharu-app/src/commands/canvas.rs`에 `commit_paint`(`:288`)를 본떠 만든다. `#[tracing::instrument(target = "koharu_metrics", name = "raster_adjusted", skip_all, fields(...))]`, `#[tauri::command]`, `#[specta::specta]`를 붙이고, `ensure_revision`으로 기대 리비전을 확인한 뒤 `project.adjust_raster`를 호출하고 `desktop.synchronize` + `canvas_channel.publish`를 한다. `commit_raster_stroke`(`canvas.rs:343`)의 마무리 부분이 정확한 참고다.

- [ ] **Step 2: RPC 라우트를 추가한다**

`crates/koharu-rpc/src/routes/canvas.rs`의 `router()`에 추가하고 핸들러를 만든다:

```rust
.route("/canvas/raster/adjust", post(adjust_raster))
```

`CommitPaintRequest`(같은 파일)의 형태를 본떠 `AdjustRasterRequest { expected_revision: Revision, layer: EntityId, adjust: ColorAdjust }`를 `Deserialize`로 정의한다. `commit_paint` 핸들러(`:251`)가 상태를 꺼내는 방식을 그대로 따른다.

- [ ] **Step 3: `koharu-rpc`가 컴파일되는지 확인한다**

Run: `cargo check -j 6 -p koharu-rpc`
Expected: 성공

- [ ] **Step 4: `protocol.ts`를 갱신한다**

`packages/bridge/src/protocol.ts`에 `ColorAdjust` 타입을 추가한다. **Task 1의 serde 태그 형태와 정확히 일치**해야 한다 (`#[serde(tag = "type", rename_all = "snake_case")]`이므로 `{ type: "brightness", delta }` 꼴):

```ts
export type ColorAdjust =
  | { type: "brightness"; delta: number }
  | { type: "contrast"; percent: number }
  | { type: "levels"; low: number; high: number }
  | { type: "hue"; degrees: number }
  | { type: "grayscale" }
```

`commands` 객체에 추가한다:

```ts
	adjustRaster: (expectedRevision: Revision, layer: EntityId, adjust: ColorAdjust) =>
		post<LayerCommit>("/canvas/raster/adjust", {
			expected_revision: expectedRevision,
			layer,
			adjust,
		}),
```

- [ ] **Step 5: 타입 검사**

Run: `bunx tsc --noEmit -p packages/koharu/tsconfig.json`
Expected: exit 0. `packages/bridge/src/wasm`가 오래되어 `CanvasHandle` ��류가 나면 `bun run --filter @koharu/bridge build`를 먼저 돌린다.

- [ ] **Step 6: 커밋**

```bash
git add crates/koharu-app/src/commands/canvas.rs crates/koharu-rpc/src/routes/canvas.rs packages/bridge/src/protocol.ts
git commit -m "feat(app): expose the raster adjustment command"
```

---

### Task 4: Inspector UI와 i18n

래스터 레이어를 선택하면 색보정 컨트롤이 나온다.

**Files:**
- Modify: `packages/koharu/components/editor/Inspector.tsx`
- Modify: `packages/koharu/public/locales/*/translation.json` (9개)
- Modify: `packages/koharu/tests/components/editor-components.test.tsx`
- Test: `packages/koharu/tests/lib/adjust.test.ts` (신규, 컨트롤 헬퍼가 있으면)

**Interfaces:**
- Consumes: Task 3의 `commands.adjustRaster`, `ColorAdjust`
- Produces: `AdjustInspector` 컴포넌트 — 래스터 레이어의 보정 컨트롤

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`editor-components.test.tsx`에 추가한다. 기존 래스터 레이어 테스트가 레이어를 선택하는 방식을 확인해 맞춘다.

```tsx
  it('applies a brightness adjustment to a selected raster layer', async () => {
    // 기존 헬퍼로 래스터 레이어를 선택한 상태를 만든다.
    // 밝기 입력을 조정하고 적용 버튼을 누른다.
    await user.click(screen.getByRole('button', { name: /apply/i }))
    expect(adjustRaster).toHaveBeenCalledWith(
      expect.anything(),
      'raster-layer-id',
      { type: 'brightness', delta: expect.any(Number) },
    )
  })

  it('shows the adjustment section only for a raster layer', async () => {
    // 텍스트 레이어를 선택하면 색보정 섹션이 없다.
  })
```

`adjustRaster`는 `vi.spyOn(commands, 'adjustRaster')`로 맞춘다. 기존 테스트가 쓰는 스파이 방식을 확인해 이름을 맞춘다.

- [ ] **Step 2: 실패를 확인한다**

Run: `bun run --filter '@koharu/app' test -- tests/components/editor-components.test.tsx`
Expected: FAIL

- [ ] **Step 3: `AdjustInspector`를 추가한다**

`Inspector.tsx`에 넣는다. 기존 `ShapeInspector`(`:144`)의 구조와 `InspectorField` 패턴을 따른다. 대상은 **래스터 레이어**이므로 기존 텍스트 전용 분기와 나란히 둔다.

컨트롤:
- 종류 선택: 밝기 / 대비 / 레벨 / 색조 / 흑백
- 선택된 종류의 파라미터 입력 (`NumberField` 또는 `Slider`)
- **적용 버튼** — 이 버튼이 `commands.adjustRaster`를 호출한다

**적용 즉시 한 번의 커밋이다.** 슬라이더를 움직이는 동안 커밋하지 않는다(스펙 B-D6: 미리보기 없이 적용). 이 결정을 주석에 남긴다.

각 변형의 입력 범위는 서버 검증과 일치시킨다: 밝기 `-255..=255`, 대비 `-100..=100`, 레벨 `0..=255`이면서 `low < high`, 색조 `-180..=180`. **UI가 서버가 거부할 값을 만들 수 없어야 한다** — 범위 밖 입력을 막고, 레벨에서 `low >= high`면 적용 버튼을 비활성화한다.

- [ ] **Step 4: 9개 로케일에 키를 추가한다**

필요한 키 (`inspector.adjust.*`):

- `title`, `brightness`, `contrast`, `levels`, `hue`, `grayscale`, `apply`
- `low`, `high`, `degrees`, `percent`, `delta`
- `invalidLevels`

`ko-KR`은 실제 한국어로 1a·1b가 쓴 도형·채우기 용어와 일관되게. 나머지 8개는 기존 수준의 실제 번역으로.

**키 집합이 9개 파일에서 동일해야 한다.** 편집 후 9개가 같은 키 집합인지 확인한다(1a에서 스크립트 편집이 로케일을 손상시킨 사례가 있다).

- [ ] **Step 5: 프론트 검사를 돌린다**

Run: `bun run --filter '@koharu/app' test`
Expected: 전부 통과

Run: `bunx oxlint -c packages/koharu/.oxlintrc.json packages/koharu`
Expected: clean

Run: `bunx tsc --noEmit -p packages/koharu/tsconfig.json`
Expected: exit 0

- [ ] **Step 6: 커밋**

```bash
git add packages/koharu/components/editor/Inspector.tsx packages/koharu/public/locales packages/koharu/tests
git commit -m "feat(ui): adjust raster colours from the inspector"
```

---

### Task 5: 회귀 확인

B3이 전체에 해를 끼치지 않았는지 확인한다.

**Files:**
- Test: 없음 (검증만)

- [ ] **Step 1: 관련 크레이트를 한 번에 돌린다**

Run: `cargo test -j 6 -p koharu-app -p koharu-rpc --lib`
Expected: 전부 통과

Run: `cargo check -j 6 --workspace`
Expected: 성공, 새 경고 없음

Run: `bun run --filter '@koharu/app' test`
Expected: 전부 통과

`-j 6`은 이 머신에서 필수다 (12코어, 무제한 병렬은 데스크톱을 멈춘다 — `fork/HANDOFF.md:44`).

- [ ] **Step 2: 기존 래스터 경로가 그대로인지 확인한다**

`apply_raster_stroke`가 바뀌지 않았는지 `git diff`로 확인한다. B3은 그 함수를 **호출하지 않고 참고만** 한다 — 자산 쓰기 코드를 복사했다면 그 사실을 보고에 남기고, 중복이 크면 헬퍼 추출을 제안한다.

- [ ] **Step 3: 보고**

테스트 결과와 `git diff --stat`를 보고서에 남긴다.

---

## Self-Review

**Spec coverage**

| 스펙 절 | 태스크 |
|---|---|
| B3 — 색보정 커맨드 | Task 2, Task 3 |
| `ColorAdjust` 5변형 | Task 1 |
| 거부 규칙 (비래스터, 크기 불일치, 범위) | Task 1 (파라미터), Task 2 (대상) |
| B-D3 래스터 전용 | Task 2 |
| B-D5 한 번의 커밋 | Task 2 |
| B-D6 미리보기 없음 | Task 4 |
| B-D7 파라미터 미저장 | Task 2 (저장하지 않음) |
| B-D8 알파 보존 | Task 1 |
| UI + i18n | Task 4 |
| 테스트 표 (색보정 행) | Task 1, Task 2 |
| 실패 시나리오 (`low == high`) | Task 1 |

B1·B2는 이 계획의 범위 밖이다.

**Spec gap을 계획이 메운 곳**

1. **알파 0 픽셀의 RGB를 보존한다.** 스펙은 "알파 보존"만 요구했고, 완전 투명 픽셀의 RGB는 언급하지 않았다. 크레이트 함수가 이미지 전체를 순회하므로, 알파 0 스킵을 지키려면 원본을 복제해 되돌리는 구조가 필요하다. Task 1이 그 구조와 이유를 명시한다.
2. **`imageproc` 의존 추가.** 스펙은 "직접 의존을 추가해야 한다"고 했고 Task 1이 정확한 위치를 지정한다.
3. **`ColorAdjust`의 serde 태그.** 스펙은 타입만 정의했고 wire 형태를 정하지 않았다. 커맨드 인자이므로 저장 컴포넌트와 달리 **내부 태그**(`tag = "type"`)를 쓰고, 이는 `Layer`(`project.rs:78`)의 관례와 같다. Task 1이 정하고 Task 3이 그 형태로 TS를 쓴다.
4. **자산이 없을 때.** `apply_raster_stroke`는 자산이 없으면 빈 이미지를 만든다. 보정이 그 관례를 따를지 스펙에 없었다. Task 2가 따른다.
5. **흑백 그레이 계수.** 스펙은 `grayscale_alpha` 경로를 쓰라고 했지만 계수를 정하지 않았다. Task 1이 `0.2126/0.7152/0.0722`를 제시하고 이유를 남긴다.

**Type consistency 확인**

- `ColorAdjust` 변형 이름은 Task 1(Rust) → Task 3(TS `kind` 태그) → Task 4(UI)에서 동일하다.
- `percent`라는 필드명은 `contrast`의 백분율 단위를 드러내기 위해 스펙 B-D와 함께 쓴다.
- `Project::adjust_raster`는 `(Commit, EntityId)`를 돌려주고, 커맨드는 `LayerCommit { revision, layer }`로 감싼다 — `commit_raster_stroke`와 같은 형태다.
- `apply_adjust`는 `&mut RgbaImage`를 받고 아무것도 돌려주지 않는다. 테스트가 제자리 변경을 확인한다.

**Proportion**

스펙 211줄, 계획 약 620줄. 코드 블록은 테스트와 정확한 시그니처 위주이며, 함수 본문은 알고리즘이 테스트로 결정되지 않는 곳(Task 1의 알파 0 보존 구조)에만 있다.
