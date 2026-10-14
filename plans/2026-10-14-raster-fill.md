# 버킷 채우기 (B2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 붓 도구 옆에 버킷 도구를 붙여, 클릭한 지점과 같은 색이 이어진 영역을 한 색으로 칠한다.

**Architecture:** `Project::fill_raster`가 대상 래스터의 PNG 자산을 디코드하고, `imageproc::drawing::flood_fill_mut`으로 시작 픽셀과 **정확히 같은** 색이 이어진 영역을 칠한 뒤 다시 PNG로 한 번에 커밋한다. B3가 만든 자산 경로(디코드 → 제자리 수정 → `set_asset` → 승격 + `origin` 재설정)를 그대로 재사용하므로 되돌리기·동기화·PSD가 자동으로 맞는다. 프론트는 새 `fill` 도구와 클릭 지점 전달, 색상 선택 UI를 얻는다.

**Tech Stack:** Rust (koharu-app, koharu-rpc), `image` 0.25.10, `imageproc` 0.27.0, TypeScript (Next.js, zustand, i18next), `cargo test`, vitest.

**Spec:** `specs/2026-10-14-paint-fill-adjust-design.md` (B2 절, B-D3/B-D4/B-D5/B-D7)

**이 계획의 범위:** B2만이다. 그라디언트 채우기(`FillSpec::Gradient`)는 이 계획에 **포함하지 않는다** — 아래 "무엇을 미루는가" 참조. B1(브러시 그라디언트)과 1c(패턴)는 각자의 계획이다.

---

## Global Constraints

- **`imageproc::flood_fill_mut`을 그대로 쓴다.** 4방향이고 허용 오차가 **없다** — `P: Pixel + PartialEq`로 비교하므로(`imageproc-0.27.0/src/drawing/fill.rs:60-63`) 시작 픽셀과 정확히 같은 픽셀만 칠한다. 순회 로직을 복사하지 않는다(B-D4).
- **래스터 레이어에만** 적용한다(B-D3). 도형·텍스트에는 픽셀 연산이 의미를 갖지 않는다.
- **한 번의 커밋**(B-D5). 파라미터를 저장하지 않는다(B-D7).
- 자산 크기가 페이지 크기와 다르면 거부한다.
- **페이지 해석은 조상 탐색으로 한다.** `Snapshot::parent`는 직계 부모를 돌려주고 `Snapshot::page`는 페이지가 아니면 실패한다. 레이어는 그룹 아래에 있을 수 있으므로 단일 `parent` 호출은 그룹 레이어를 `EntityNotFound`로 잘못 거부한다. **B3에서 실제로 발생한 버그이며**(`project.rs`의 `adjust_raster`가 커밋 `d2b08b1d`에서 수정됨), 같은 실수를 반복하지 않는다. **`adjust_raster`의 조상 탐색을 헬퍼로 추출해 공유한다.**
- 새 Tauri 커맨드는 **반드시** HTTP 라우트를 함께 추가한다.
- 하위 호환·별칭 금지 (AGENTS.md). 주석은 불변식을 설명한다.
- `-j 6`을 모든 cargo 명령에 붙인다 (12코어, 무제한 병렬은 데스크톱을 멈춘다).
- 검증은 가장 작은 관련 검사만 debug 프로필로 한 번 돌린다 (AGENTS.md).

## Review Focus

스펙이 명시하지 않았지만 실제로 깨질 수 있는 지점이다. 각 항목의 테스트는 그 코드를 소유한 태스크에 넣는다.

1. **`i32` 좌표 변환에서의 음수·경계 처리.** 클릭 지점은 `f64` 페이지 좌표이고 `flood_fill_mut`은 `u32`를 받는다. 음수나 페이지 밖 좌표를 그대로 캐스팅하면 거대한 `u32`가 되어 **패닉하거나 엉뚱한 픽셀을 칠한다**. 범위 검사를 캐스팅보다 먼저 한다(Task 2).
2. **투명 영역이 채워지는지.** 래스터 레이어의 빈 영역은 `[0,0,0,0]`이다 — 버킷은 그것도 "하나의 색 영역"이므로 **칠해야 맞다**. 반대로 알파 0 픽셀의 RGB를 건드리는 것이 문제가 아니다(B3과 다른 점). 이 구분을 테스트로 못 박는다(Task 2).
3. **시작 픽셀이 이미 채울 색일 때.** `flood_fill_mut`은 그대로 두고 커밋은 일어난다(스펙의 실패 시나리오 표). 무한 루프가 없음을 확인한다(Task 2).
4. **가장자리에서의 클램프.** 시작점이 페이지 경계에 있을 때 이웃이 없어도 정상 동작해야 한다(Task 2).
5. **도구 상태가 다른 도구와 섞이지 않는지.** `fill` 도구를 고르면 `draw`의 붓 상태를 쓰지 않고 자체 색을 갖는다(Task 4).
6. **B3와 같은 페이지 해석 헬퍼를 쓰는지.** 두 곳이 각자 조상 탐색을 구현하면 한쪽만 고쳐지는 드리프트가 생긴다(Task 1).

---

## File Structure

**수정**

| 파일 | 변경 |
|---|---|
| `crates/koharu-app/src/commands/project.rs` | 페이지 조상 탐색 헬퍼 추출, `Project::fill_raster` 추가 |
| `crates/koharu-app/src/commands/canvas.rs` | `fill_raster` 커맨드 |
| `crates/koharu-rpc/src/routes/canvas.rs` | `POST /canvas/raster/fill` |
| `packages/bridge/src/protocol.ts` | `FillRequest` 타입, `commands.fillRaster` |
| `packages/koharu/lib/store.ts` | `CanvasTool`에 `fill` 추가, 단축키, 채우기 색 |
| `packages/koharu/components/editor/CanvasWorkspace.tsx` | `fill` 도구 클릭 처리 |
| `packages/koharu/components/editor/ToolBar.tsx` | 버킷 도구 버튼 + 색상 |
| `packages/koharu/public/locales/*/translation.json` | 9개 로케일 |
| `packages/koharu/tests/components/editor-components.test.tsx` | 도구 배선 테스트 |

---

### Task 1: 페이지 조상 탐색 헬퍼 추출

B3의 `adjust_raster`가 조상 탐색을 인라인으로 갖고 있다. B2가 같은 로직을 필요로 하므로 먼저 추출한다. **이 태스크는 동작을 바꾸지 않는다** — 순수 리팩터링이고 기존 테스트가 그대로 통과해야 한다.

**Files:**
- Modify: `crates/koharu-app/src/commands/project.rs`
- Test: `crates/koharu-app/src/commands/project.rs` (인라인 `mod tests` — 기존 6개가 회귀 가드)

**Interfaces:**
- Produces: `fn owning_page(snapshot: &Snapshot, entity: EntityId) -> Result<EntityId>`

- [ ] **Step 1: 현재 인라인 로직을 확인한다**

`adjust_raster` 안의 다음 블록을 찾는다(주석 포함):

```rust
        let mut page = layer;
        while let Some(parent) = snapshot.parent(page)? {
            page = parent;
        }
        let page_value = snapshot.page(page)?.page()?;
```

- [ ] **Step 2: 헬퍼로 추출한다**

`adjust_raster` 위(같은 `impl Project` 바깥의 자유 함수로 두는 편이 낫다 — `Project` 상태가 필요 없다)에 추가한다:

```rust
/// The page an entity ultimately belongs to.
///
/// `Snapshot::parent` returns the *immediate* parent and `Snapshot::page` only
/// accepts a page id, so a single `parent` call rejects any nested layer with a
/// misleading `EntityNotFound`. Only a page has no parent, so walking up until
/// `parent` returns `None` terminates at the owning page; `page(entity)` then
/// fails loudly if the chain did not in fact reach one.
fn owning_page(snapshot: &Snapshot, entity: EntityId) -> Result<EntityId> {
    let mut page = entity;
    while let Some(parent) = snapshot.parent(page)? {
        page = parent;
    }
    Ok(page)
}
```

`adjust_raster`는 다음으로 바꾼다:

```rust
        let page = owning_page(&snapshot, layer)?;
        let page_value = snapshot.page(page)?.page()?;
```

`snapshot.page(page)?`가 페이지가 아님을 검증하므로 헬퍼는 별도 검사를 하지 않는다 — 그 이유를 헬퍼 doc에 남긴다.

- [ ] **Step 3: 동작이 바뀌지 않았음을 확인한다**

Run: `cargo test -j 6 -p koharu-app --lib commands::project::tests::adjusting`
Expected: **6 passed** — 특히 `adjusting_a_grouped_raster_layer_still_finds_its_page`와 `adjusting_a_non_raster_layer_is_rejected`(래스터 가드 메시지를 단언한다). 이 둘이 통과하면 추출이 동작을 보존한 것이다.

- [ ] **Step 4: 컴파일과 전체 스위트**

Run: `cargo test -j 6 -p koharu-app --lib`
Expected: 62 passed

- [ ] **Step 5: 커밋**

```bash
git add crates/koharu-app/src/commands/project.rs
git commit -m "refactor(app): extract the owning-page walk shared by raster edits"
```

---

### Task 2: `Project::fill_raster` (순수 로직)

채우기를 프로젝트 계층에 넣는다. **그라디언트는 이 태스크에 없다** — 단색 채우기만이다.

**Files:**
- Modify: `crates/koharu-app/src/commands/project.rs`
- Test: `crates/koharu-app/src/commands/project.rs` (인라인 `mod tests`)

**Interfaces:**
- Consumes: Task 1의 `owning_page`; B3의 자산 경로(`adjust_raster`)를 참고
- Produces: `Project::fill_raster(&mut self, layer: EntityId, point: Point, color: [u8; 4]) -> Result<(Commit, EntityId)>`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`project.rs`의 `mod tests`에 추가한다. 기존 `project_with_raster`/`source_image` 헬퍼를 재사용한다.

```rust
    #[tokio::test]
    async fn filling_replaces_only_the_connected_region() {
        // A 4x1 image: the two left pixels are identical, the right two differ.
        let mut source = RgbaImage::new(4, 1);
        source.put_pixel(0, 0, image::Rgba([10, 10, 10, 255]));
        source.put_pixel(1, 0, image::Rgba([10, 10, 10, 255]));
        source.put_pixel(2, 0, image::Rgba([20, 20, 20, 255]));
        source.put_pixel(3, 0, image::Rgba([20, 20, 20, 255]));
        let (mut project, layer) = project_with_raster(source).await;

        project
            .fill_raster(layer, Point { x: 0.0, y: 0.0 }, [255, 0, 0, 255])
            .await
            .unwrap();

        let image = source_image(&project.snapshot(), layer).await;
        assert_eq!(
            image.get_pixel(0, 0),
            &image::Rgba([255, 0, 0, 255]),
            "the start pixel is filled"
        );
        assert_eq!(
            image.get_pixel(1, 0),
            &image::Rgba([255, 0, 0, 255]),
            "the connected neighbour is filled"
        );
        assert_eq!(
            image.get_pixel(2, 0),
            &image::Rgba([20, 20, 20, 255]),
            "a differently coloured neighbour is untouched"
        );
        assert_eq!(
            image.get_pixel(3, 0),
            &image::Rgba([20, 20, 20, 255]),
            "the region must not leak past the boundary"
        );
    }

    /// A raster layer's empty area is fully transparent, and a bucket must treat
    /// it as a region like any other — otherwise the tool cannot fill the blank
    /// space authors actually use it on.
    #[tokio::test]
    async fn filling_covers_a_transparent_region() {
        let mut source = RgbaImage::new(4, 1);
        source.put_pixel(0, 0, image::Rgba([0, 0, 0, 0]));
        source.put_pixel(1, 0, image::Rgba([0, 0, 0, 0]));
        source.put_pixel(2, 0, image::Rgba([200, 0, 0, 255]));
        source.put_pixel(3, 0, image::Rgba([0, 0, 0, 0]));
        let (mut project, layer) = project_with_raster(source).await;

        project
            .fill_raster(layer, Point { x: 0.0, y: 0.0 }, [255, 255, 0, 255])
            .await
            .unwrap();

        let image = source_image(&project.snapshot(), layer).await;
        assert_eq!(image.get_pixel(0, 0), &image::Rgba([255, 255, 0, 255]));
        assert_eq!(image.get_pixel(1, 0), &image::Rgba([255, 255, 0, 255]));
        assert_eq!(
            image.get_pixel(2, 0),
            &image::Rgba([200, 0, 0, 255]),
            "the opaque divider stops the fill"
        );
        assert_eq!(
            image.get_pixel(3, 0),
            &image::Rgba([0, 0, 0, 0]),
            "the far transparent pixel is a different region and stays empty"
        );
    }

    #[tokio::test]
    async fn filling_with_the_existing_colour_is_a_harmless_commit() {
        let (mut project, layer) = project_with_raster(RgbaImage::from_pixel(
            4,
            1,
            image::Rgba([10, 10, 10, 255]),
        ))
        .await;

        let (commit, _) = project
            .fill_raster(layer, Point { x: 0.0, y: 0.0 }, [10, 10, 10, 255])
            .await
            .expect("filling with the same colour is not an error");

        assert!(commit.revision.get() > 0);
        let image = source_image(&project.snapshot(), layer).await;
        assert_eq!(image.get_pixel(0, 0), &image::Rgba([10, 10, 10, 255]));
    }

    /// The click point is `f64` page space and `flood_fill_mut` takes `u32`.
    /// Casting before checking would turn `-1.0` into a huge `u32` and panic or
    /// fill the wrong pixel, so the bounds check must come first.
    #[tokio::test]
    async fn filling_outside_the_page_is_rejected() {
        let (mut project, layer) =
            project_with_raster(RgbaImage::from_pixel(4, 4, image::Rgba([10, 10, 10, 255]))).await;

        for point in [
            Point { x: -1.0, y: 0.0 },
            Point { x: 0.0, y: -1.0 },
            Point { x: 4.0, y: 0.0 },
            Point { x: 0.0, y: 4.0 },
            Point { x: f64::NAN, y: 0.0 },
        ] {
            assert!(
                project.fill_raster(layer, point, [1, 2, 3, 255]).await.is_err(),
                "a point outside the page must be rejected: {point:?}"
            );
        }
    }

    #[tokio::test]
    async fn filling_a_non_raster_layer_is_rejected() {
        let (mut project, text) = project_with_text_layer().await;
        let error = project
            .fill_raster(text, Point { x: 0.0, y: 0.0 }, [1, 2, 3, 255])
            .await
            .expect_err("a text layer cannot be filled");
        assert!(
            error.to_string().contains("only raster layers"),
            "expected the raster guard, got: {error}"
        );
    }

    /// The bottom-right corner is the boundary case: the start pixel is valid
    /// but has no in-bounds neighbour, so an off-by-one in the fill or the
    /// bounds check shows up here and nowhere else.
    #[tokio::test]
    async fn filling_at_the_corner_stops_at_the_edge() {
        let (mut project, layer) =
            project_with_raster(RgbaImage::from_pixel(3, 3, image::Rgba([5, 5, 5, 255]))).await;

        project
            .fill_raster(layer, Point { x: 2.0, y: 2.0 }, [9, 9, 9, 255])
            .await
            .unwrap();

        let image = source_image(&project.snapshot(), layer).await;
        assert_eq!(image.get_pixel(2, 2), &image::Rgba([9, 9, 9, 255]));
        assert_eq!(
            image.get_pixel(0, 0),
            &image::Rgba([9, 9, 9, 255]),
            "a uniform image fills entirely, corner included"
        );
    }
```

- [ ] **Step 2: 실패를 확인한다**

Run: `cargo test -j 6 -p koharu-app --lib commands::project::tests::filling`
Expected: 컴파일 실패 (`fill_raster` 미정의)

- [ ] **Step 3: 구현한다**

`adjust_raster`를 옆에 두고 **같은 순서**를 따른다. 차이만 서술한다:

1. `let snapshot = self.snapshot();`
2. `let page = owning_page(&snapshot, layer)?;` (Task 1) → `width`/`height` 계산, 0이면 거부
3. 래스터 확인 — `component::<SceneRasterLayer>(layer)?.context("only raster layers can be filled")?`
   **B3와 메시지가 다르다**("adjusted" vs "filled") — 사용자가 어느 도구에서 실패했는지 알 수 있어야 한다.
4. **좌표 범위 검사** — 이것이 B3에 없는 단계다:
   ```rust
   let x = point.x.floor();
   let y = point.y.floor();
   if !x.is_finite() || !y.is_finite() || x < 0.0 || y < 0.0 || x >= f64::from(width) || y >= f64::from(height) {
       bail!("the fill point is outside the page");
   }
   let (x, y) = (x as u32, y as u32);
   ```
   **`is_finite`를 먼저 검사한다** — `NaN`은 모든 비교가 거짓이라 `< 0.0`/`>= width` 검사를 통과한 뒤 `as u32`에서 0이 되거나 정의되지 않은 값을 만든다.
5. 자산 읽기·디코드·크기 검사 (B3과 동일)
6. **채우기**:
   ```rust
   imageproc::drawing::flood_fill_mut(&mut image, x, y, image::Rgba(color));
   ```
   `image`는 `RgbaImage` = `ImageBuffer<Rgba<u8>, Vec<u8>>`이므로 `P = Rgba<u8>`이다.
7. PNG 인코딩, `edit.set_asset(...)`, 승격 + `origin` 재설정 — **`adjust_raster`의 블록과 정확히 같은 형태**. B3의 주석(`RasterLayer::origin`이 파이프라인 게이트라는 설명)을 복사한다.
8. `self.commit(...)` 한 번.

**채워진 픽셀 수가 0이어도 성공으로 커밋한다.** 스펙의 실패 시나리오 표가 그렇게 정했다 — 시작 픽셀이 이미 그 색이면 `flood_fill_mut`이 아무것도 바꾸지 않는다. 이 결정을 주석에 남긴다.

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

Run: `cargo test -j 6 -p koharu-app --lib commands::project::tests::filling`
Expected: 6 passed

- [ ] **Step 5: 전체 스위트**

Run: `cargo test -j 6 -p koharu-app --lib`
Expected: 68 passed (62 + 6)

- [ ] **Step 6: 커밋**

```bash
git add crates/koharu-app/src/commands/project.rs
git commit -m "feat(app): fill a connected raster region with a colour"
```

---

### Task 3: 커맨드·라우트·TS 계약

B3 Task 3과 같은 형태다. **`protocol.ts`를 먼저 커밋해 Task 4를 병렬화할 수 있게 한다.**

**Files:**
- Modify: `crates/koharu-app/src/commands/canvas.rs`
- Modify: `crates/koharu-rpc/src/routes/canvas.rs`
- Modify: `packages/bridge/src/protocol.ts`

**Interfaces:**
- Consumes: Task 2의 `Project::fill_raster`
- Produces:
  - 커맨드 `fill_raster(expected_revision, layer, point, color) -> LayerCommit`
  - 라우트 `POST /canvas/raster/fill`
  - TS `commands.fillRaster(expectedRevision, layer, point, color)`

- [ ] **Step 1: `protocol.ts`를 갱신한다**

`ColorAdjust` 옆에 추가한다:

```ts
/** A bucket fill: the click point in page pixels, and the colour to flood with. */
export type FillRequest = {
	point: Point,
	color: [number, number, number, number],
};
```

`commands` 객체에 `adjustRaster` 옆에 추가한다:

```ts
	fillRaster: (
		expectedRevision: Revision,
		layer: EntityId,
		point: Point,
		color: [number, number, number, number],
	) =>
		post<LayerCommit>("/canvas/raster/fill", {
			expected_revision: expectedRevision,
			layer,
			point,
			color,
		}),
```

- [ ] **Step 2: 타입 검사**

Run: `bunx tsc --noEmit -p packages/koharu/tsconfig.json`
Expected: exit 0

- [ ] **Step 3: 커맨드를 추가한다**

`crates/koharu-app/src/commands/canvas.rs`의 `adjust_raster`(`:344`)를 본떠 만든다. 인자는 `(expected_revision, layer, point: Point, color: [u8; 4])`이고, `#[tracing::instrument(target = "koharu_metrics", name = "raster_filled", skip_all, fields(...))]`을 붙인다. 본문은 `ensure_revision` → `project.fill_raster` → `record_commit` → `synchronize` → `publish` → `LayerCommit`. **`adjust_raster`와 순서가 같아야 한다.**

- [ ] **Step 4: RPC 라우트를 추가한다**

`routes/canvas.rs`의 `router()`에서 `raster/adjust` 옆에 추가한다:

```rust
.route("/canvas/raster/fill", post(fill_raster))
```

`AdjustRasterRequest` 옆에 요청 타입을 정의한다:

```rust
#[derive(Deserialize)]
struct FillRasterRequest {
    expected_revision: Revision,
    layer: EntityId,
    point: Point,
    color: [u8; 4],
}
```

핸들러는 `adjust_raster` 핸들러(`:278`)와 같은 형태로 상태를 꺼내 전달한다.

- [ ] **Step 5: 컴파일과 라우트 등록을 확인한다**

Run: `cargo check -j 6 -p koharu-rpc -p koharu-app`
Expected: 성공, 새 경고 없음

**라우트 문자열이 `router()` 체인에 있는지 grep으로 확인한다** — 핸들러만 정의되고 라우팅되지 않는 것이 이 제약이 막으려는 실패다.

- [ ] **Step 6: 커밋**

`protocol.ts`를 **먼저** 별도 커밋해 Task 4가 의존할 수 있게 한다:

```bash
git add packages/bridge/src/protocol.ts
git commit -m "feat(bridge): declare the bucket fill wire contract"

git add crates/koharu-app/src/commands/canvas.rs crates/koharu-rpc/src/routes/canvas.rs
git commit -m "feat(app): expose the bucket fill command"
```

---

### Task 4: `fill` 도구와 색상 UI

**Files:**
- Modify: `packages/koharu/lib/store.ts`
- Modify: `packages/koharu/components/editor/CanvasWorkspace.tsx`
- Modify: `packages/koharu/components/editor/ToolBar.tsx`
- Modify: `packages/koharu/public/locales/*/translation.json` (9개)
- Modify: `packages/koharu/tests/components/editor-components.test.tsx`

**Interfaces:**
- Consumes: Task 3의 `commands.fillRaster`
- Produces: `fill` 도구와 그 색 상태

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`editor-components.test.tsx`에 추가한다. 기존 도구 테스트가 도구를 고르고 스파이를 맞추는 방식을 확인해 그대로 따른다.

```tsx
  it('sends the clicked page point and colour when the fill tool is used', async () => {
    // 래스터 레이어를 선택하고 fill 도구를 고른 뒤 캔버스를 클릭한다.
    expect(fillRaster).toHaveBeenCalledWith(
      expect.anything(),
      'raster-layer-id',
      expect.objectContaining({ x: expect.any(Number), y: expect.any(Number) }),
      expect.any(Array),
    )
  })

  it('does not offer the fill tool without a raster layer', async () => {
    // 텍스트만 있는 페이지에서 fill 도구가 비활성이거나 채우기를 호출하지 않는다.
  })
```

`fillRaster`는 `vi.spyOn(commands, 'fillRaster')`로 맞춘다.

- [ ] **Step 2: 실패를 확인한다**

Run: `bun run --filter '@koharu/app' test -- tests/components/editor-components.test.tsx`
Expected: FAIL

- [ ] **Step 3: 도구 상태를 추가한다**

`store.ts`:

```ts
export type CanvasTool =
  | 'select'
  | 'text'
  | 'shape'
  | 'draw'
  | 'fill'
  | 'eraser'
  | 'color_picker'
  | 'remove'
  | 'manual_ocr_box'
  | 'manual_ocr_lasso'
  | 'pan'
```

`defaultShortcuts`(`store.ts:98-110`)에 `fill: 'g'`를 추가한다. **`g`는 현재 비어 있다** (기존 키는 v/t/r/b/e/i/j/o/l/h/0). `ShortcutAction`이 `CanvasTool | 'fit'`이므로(`:50`) `CanvasTool`에 `fill`을 넣으면 타입이 자동으로 따라온다.

`fillColor: string` 상태와 `setFillColor`를 추가한다. **붓의 `color`를 재사용하지 않는다** — 두 도구는 다른 색을 쓸 수 있고, 공유하면 한쪽을 바꿀 때 다른 쪽이 따라 바뀐다. 기본값은 `#000000`.

- [ ] **Step 4: 클릭 처리를 배선한다**

`CanvasWorkspace.tsx`의 `onPointerDown`에서 `color_picker` 분기(`:908`) 옆에 `fill` 분기를 추가한다. 좌표는 **`clientPagePoint`로 얻는다**(`:764` — `physical`이 아니라 페이지 좌표다). 채우기 대상은 **선택된 래스터 레이어**이고, 선택이 없으면 아무것도 하지 않는다(오류 토스트를 띄우지 않는다 — 도구가 비활성인 편이 낫다).

호출 후 `refresh`로 레이어를 새로 고친다. 픽셀 자체는 B3과 같이 `canvas_channel` publish → SSE 경로로 갱신된다.

- [ ] **Step 5: 도구막대에 버튼과 색상을 추가한다**

`ToolBar.tsx`의 `hasBrush` 블록(`:123`) 근처에 fill 도구 버튼을 넣고, `fill` 도구일 때 `ColorWell`을 `fillColor`에 연결한다. 기존 `hasBrush` 조건을 그대로 확장하지 말고 **별도 조건**으로 둔다 — 붓 크기 슬라이더는 채우기에 의미가 없다.

- [ ] **Step 6: 9개 로케일에 키를 추가한다**

필요한 키: `tools.fill`, `tools.fillColor`.

`ko-KR`은 실제 한국어로("채우기", "채우기 색"). 나머지 8개는 기존 수준의 실제 번역으로.

**키 집합이 9개 파일에서 동일해야 한다.** 편집 후 9개가 같은 키 집합인지 확인한다.

- [ ] **Step 7: 프론트 검사를 돌린다**

Run: `bun run --filter '@koharu/app' test`
Run: `bunx tsc --noEmit -p packages/koharu/tsconfig.json`
Run: `bunx oxlint -c packages/koharu/.oxlintrc.json packages/koharu`
Expected: 전부 통과

- [ ] **Step 8: 커밋**

```bash
git add packages/koharu/lib/store.ts packages/koharu/components/editor/CanvasWorkspace.tsx packages/koharu/components/editor/ToolBar.tsx packages/koharu/public/locales packages/koharu/tests
git commit -m "feat(ui): add the bucket fill tool"
```

---

### Task 5: 회귀 확인

**Files:**
- Test: 없음 (검증만)

- [ ] **Step 1: 관련 크레이트와 프론트를 돌린다**

Run: `cargo test -j 6 -p koharu-app -p koharu-rpc --lib`
Run: `cargo check -j 6 --workspace`
Run: `bun run --filter '@koharu/app' test`

- [ ] **Step 2: B3의 커맨드가 그대로인지 확인한다**

`git diff`로 `adjust_raster`가 바뀌지 않았는지 확인한다. Task 1의 헬퍼 추출이 유일한 변경이어야 한다.

`apply_raster_stroke`가 **여전히 무수정**인지도 확인한다 — B3 검증에서 `+414/-0`이었고, B2는 그것을 건드리지 않아야 한다.

- [ ] **Step 3: 페이지 해석 헬퍼가 두 곳에서 쓰이는지 확인한다**

`owning_page`가 `adjust_raster`와 `fill_raster` **양쪽**에서 호출되는지 grep으로 확인한다. 한 곳만 쓰면 B3의 인라인 로직이 남은 것이다.

- [ ] **Step 4: 보고**

테스트 결과와 `git diff --stat`를 보고서에 남긴다.

---

## 무엇을 미루는가 (그라디언트 채우기)

스펙의 `FillSpec::Gradient { gradient: GradientSpec, axis: GradientAxis }`를 **이 계획에서 구현하지 않는다.** 두 가지 이유다.

**1. `axis` 필드는 중복이다.** `GradientSpec::kind`가 이미 방향과 범위를 담는다 — `Linear { start: [f32; 2], end: [f32; 2] }`가 두 점을 갖고 있고, 그것이 곧 축이다. `axis: GradientAxis`를 함께 저장하면 "축이 무엇인가"에 대한 답이 둘이 되어, 1a·1b에서 가장 비쌌던 두 권위 문제가 재현된다. 스펙의 이 부분은 **틀렸고 계획에서 정정한다**: `FillSpec::Gradient { gradient: GradientSpec }`만 둔다.

**2. 마스크 유도 방식이 스펙에 미완으로 남아 있다.** 스펙은 "표식 색으로 `flood_fill_mut`하고 원본과 달라진 픽셀을 마스크로 정의한다"고 썼다. 그런데 `GradientSpec`의 좌표는 **프레임 로컬 정규화**(1b의 G3)인데, 채우기 마스크는 **페이지 픽셀**이다. 이 둘을 잇는 프레임이 채우기에는 없다 — 도형이 없기 때문이다. 그러므로 그라디언트 채우기는 "무엇을 프레임으로 삼는가"를 먼저 정해야 한다.

**권장 답**: 마스크의 **바운딩 박스**를 프레임으로 삼는다. 그러면 정규화 좌표가 자연스럽게 정의되고, 1b의 `prepared()` 변환을 그대로 쓸 수 있다. 이것은 별도 계획에서 다룬다 — 정규화 좌표를 새로 정의하는 결정이고, 단색 채우기와 함께 검증하면 리뷰가 두 관심사를 동시에 봐야 한다.

**그러므로 이 계획은 단색 채우기만 낸다.** 사용자 가치는 "빈 영역을 한 색으로 칠하기"이고, 그것이 버킷 도구의 주 용도다.

---

## Self-Review

**Spec coverage**

| 스펙 절 | 태스크 |
|---|---|
| `fill_raster` 커맨드 | Task 2, Task 3 |
| `flood_fill_mut` 그대로 사용 (B-D4) | Task 2 |
| 래스터 전용 (B-D3) | Task 2 |
| 한 번의 커밋 (B-D5) | Task 2 |
| 파라미터 미저장 (B-D7) | Task 2 |
| 자산 크기 불일치 거부 | Task 2 |
| 시작점이 페이지 밖이면 거부 | Task 2 |
| 시작 픽셀이 이미 그 색이면 무해한 커밋 | Task 2 |
| UI + i18n | Task 4 |
| 테스트 표 (버킷 채우기 행) | Task 2 |
| `FillSpec::Gradient` | **미룸** — 위 "무엇을 미루는가" |

**Spec gap을 계획이 메운 곳**

1. **`axis` 필드 제거.** 스펙의 `FillSpec::Gradient`에 중복 필드가 있었다. 미루면서 정정했다.
2. **좌표 범위 검사.** 스펙은 "시작점이 페이지 밖이면 거부한다"고만 썼고, `f64`→`u32` 캐스팅 위험은 언급하지 않았다. Task 2가 `is_finite`를 먼저 검사하는 순서까지 명시한다.
3. **페이지 조상 탐색 헬퍼.** B3에서 실제 버그였고 스펙에 없었다. Task 1이 추출해 두 곳이 공유하게 한다 — 드리프트 방지가 목적이다.
4. **투명 영역 채우기.** 스펙은 언급하지 않았다. 버킷의 주 용도이므로 Task 2가 명시적으로 테스트한다.
5. **`fill` 도구의 색이 붓과 별개.** 스펙에 없었다. Task 4가 분리하고 이유를 남긴다.

**Type consistency 확인**

- `Point`는 `canvas::Point`(`canvas.rs:26`, `{ x: f64, y: f64 }`)이고 `project.rs`가 이미 그것을 import한다(`project.rs:26`). **`ScenePoint`와 모양은 같지만 다른 타입이므로 섞지 않는다** — `add_point_text`도 `canvas::Point`를 받는다(`project.rs:678`). TS에서는 `Point`다.
- 색은 `[u8; 4]`(Rust) / `[number, number, number, number]`(TS)로, `ShapeFill::Solid`의 기존 관례와 같다.
- `fill_raster`는 `(Commit, EntityId)`를 돌려주고 커맨드가 `LayerCommit`으로 감싼다 — `adjust_raster`와 같은 형태다.
- `owning_page`는 자유 함수이고 `Project` 상태를 쓰지 않는다.

**Proportion**

스펙 B2는 약 30줄, 계획은 약 560줄. 코드 블록은 테스트와 시그니처 위주이고, 구현 본문은 알고리즘이 테스트로 결정되지 않는 곳(좌표 검사 순서, 자산 경로)에만 있다. Task 1이 순수 리팩터링으로 분리되어 있어 나머지 태스크의 diff가 작다.
