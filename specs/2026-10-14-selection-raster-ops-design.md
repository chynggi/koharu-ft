# 선택 영역과 래스터 레이어 조작 설계 (C)

작성일: 2026-10-14

## 배경

지금 캔버스에는 **영역 개념이 없다.** 붓(B1 예정)은 점 목록을 그리고, 버킷 채우기
(B2 예정)는 flood fill로 영역을 추정하며, 색보정(B3)은 레이어 전체에 적용된다.
"여기만" 작업하려면 영역을 먼저 고르는 수단이 필요하다.

조사 결과 **올가미의 골격이 이미 있다**: `manual_ocr_lasso` 도구(`store.ts:32`,
단축키 `l`), `CanvasOverlay`의 `lasso: Point[] | null`(`:25`), `LassoOverlay`
(`:199`), 그리고 `CanvasWorkspace`의 커서 매핑(`:76`). 그것은 OCR용 수동 박스를
긋는 도구지만, **점 목록을 받아 화면에 그리고 좌표를 얻는 부분은 그대로 재사용
가능하다.**

C는 그 골격 위에 **일반 선택 영역**과, 그 영역이 있을 때 래스터 레이어를 잘라내기·
복사·이동·삭제하는 조작을 얹는다.

## 범위

| 덩어리 | 내용 |
|---|---|
| C1 | 선택 영역 (사각형, 올가미, 전체 선택/해제) |
| C2 | 영역 기반 래스터 조작 (잘라내기·복사·붙여넣기·영역 삭제·영역만 남기기) |
| C3 | 레이어 조작 (복제, 순서 이동, 병합) |

**C를 한 스펙으로 다룬다.** 셋은 같은 "선택 영역" 개념을 공유하고, C1 없이 C2는
의미가 없으며 C3은 C2와 같은 커밋 경로를 쓴다. 다만 구현은 C1 → C2 → C3 순서다.

## 결정 사항

| # | 결정 | 근거 |
|---|---|---|
| S-D1 | 선택 영역은 **페이지 픽셀 좌표의 다각형**으로 표현한다. 마스크가 아니다 | 마스크(래스터)로 표현하면 히스토리에 큰 버퍼가 들어가고, 선택 영역이 저장 대상인지 아닌지가 모호해진다. 다각형은 작고, 되돌리기와 직렬화가 자명하며, 올가미가 이미 점 목록을 준다. **앤티에일리어싱된 선택 경계는 포기한다** — 만화 편집에서 도형 선택은 하드 엣지가 정상이다 |
| S-D2 | 선택 영역은 **저장하지 않는다.** 클라이언트 상태다 | 선택은 편집 중의 임시 상태이고, 저장하면 "페이지를 열었더니 예전 선택이 남아 있다"는 놀라움이 된다. 1b의 B-D7과 같은 원칙: 결과(픽셀)만 저장한다 |
| S-D3 | 영역 조작은 **래스터 레이어에만** 적용한다 | 도형·텍스트 레이어는 벡터 파라미터를 갖고 있어 "영역만 지우기"가 의미를 갖지 않는다. B-D3과 같은 경계다 |
| S-D4 | 선택 영역은 **한 번에 하나**이고 페이지 단위다 | 다중 선택·레이어별 선택은 새 개념(선택 집합, 레이어 연결)을 만든다. 1차는 하나로 충분하다 |
| S-D5 | 클립보드는 **앱 내부**에 있고 시스템 클립보드를 쓰지 않는다 | 시스템 클립보드는 이미지·텍스트·HTML을 함께 다루고 OS마다 다르다. 내부 클립보드는 `RgbaImage` + 위치 하나면 되고, 직렬화도 필요 없다(붙여넣기는 같은 세션에서 일어난다) |
| S-D6 | 영역 밖은 **삭제(알파 0)** 로 처리한다. "흰색으로 채우기"가 아니다 | 래스터 레이어의 투명은 의미를 갖는다(1a가 래스터화 배경을 `[0,0,0,0]`으로 못 박았다). 흰색으로 채우면 아래 레이어를 덮는다. 색을 채우고 싶으면 B2의 채우기 도구를 쓴다 |
| S-D7 | 레이어 병합은 **픽셀 합성**이고, 대상 레이어 하나만 남는다 | PSD 내보내기와 미리보기가 같은 합성을 봐야 한다. 합성 규칙은 `RasterStrokeMode::Paint`가 이미 쓰는 source-over를 재사용한다 |

### S-D1이 가장 중요한 결정이다

선택 영역을 마스크로 표현하면:
- 히스토리 한 항목이 페이지 크기 버퍼를 들고 있다 (1000×1500 = 1.5MB)
- 직렬화 대상인지 정해야 하고, 그 결정이 저장 포맷을 바꾼다
- 마스크와 다각형이 둘 다 있으면 **두 권위**가 된다 (1a·1b의 반복된 실패 유형)

다각형으로 표현하면 되돌리기가 점 목록 몇 개이고, 저장하지 않기로 한 결정(S-D2)이
자명해지며, "선택 영역은 무엇인가"에 대한 답이 하나뿐이다.

**대가**: 올가미가 만든 다각형은 자기 교차할 수 있고, 오목할 수 있다. 픽셀을
고르는 판정(`point_in_polygon`)은 그 둘을 모두 처리해야 한다. 이것은 표준
알고리즘이며 테스트로 못 박는다.

## 선택 영역 모델 (클라이언트)

```ts
export type SelectionShape =
  | { kind: 'rect'; x: number; y: number; width: number; height: number }
  | { kind: 'lasso'; points: Point[] }
```

좌표는 **페이지 픽셀 좌표**다(S-D1). 사각형을 별도 변형으로 두는 이유는 UI가
사각형과 올가미를 다르게 다루기 때문이다 — 사각형은 핸들로 재조정되고 올가미는
점 목록이다.

`store.ts`에 `selection: SelectionShape | null`과 `setSelection`, `clearSelection`을
추가한다. **저장 경로에 넣지 않는다** (S-D2).

**정규화 함수가 서버와 클라이언트 양쪽에 필요하다.** 서버는 커맨드 인자로 받은
선택을 픽셀 마스크로 바꿔야 하고, 클라이언트는 오버레이를 그려야 한다. 서버가
권위이고(픽셀을 실제로 고치는 쪽), 클라이언트는 표시만 한다. **두 구현이
어긋나면 보이는 선택과 실제로 지워지는 영역이 달라진다** — 1b에서 미리보기와
커밋이 어긋난 것과 같은 종류의 버그다. 그러므로 **같은 판정 규칙을 리터럴
테스트로 양쪽에 고정한다** (1b Task 4의 방식).

## 커맨드

### C2 — 영역 기반 래스터 조작

```rust
pub async fn apply_raster_selection(
    expected_revision: Revision,
    layer: EntityId,
    selection: SelectionShape,
    operation: SelectionOperation,
) -> Result<LayerCommit, Error>

pub enum SelectionOperation {
    /// 선택 영역 안의 픽셀만 알파 0으로 만든다.
    Erase,
    /// 선택 영역 **밖**의 픽셀을 알파 0으로 만든다.
    Keep,
    /// 선택 영역의 픽셀을 새 레이어로 복사한다.
    CopyToNewLayer,
    /// 선택 영역의 픽셀을 잘라내 새 레이어로 옮긴다(원본은 알파 0).
    CutToNewLayer,
}
```

`Erase`와 `Keep`은 같은 순회의 반대 조건이므로 한 곳에서 구현한다.

**`CopyToNewLayer`/`CutToNewLayer`가 새 레이어를 만든다.** B3(색보정)는 기존
레이어를 고쳤지만 여기는 새 레이어가 필요하므로, `apply_raster_stroke`의
**레이어 생성 분기를 재사용**한다 — 그 분기가 `SceneRasterLayer { origin: User, kind }`를
만들고 자산을 설정하는 정확한 참조다(`project.rs:1542-1568`).

붙여넣기와 이동은 별도 커맨드다:

```rust
pub async fn paste_raster_selection(
    expected_revision: Revision,
    layer: EntityId,
    offset: Point,
) -> Result<LayerCommit, Error>

pub async fn move_raster_selection(
    expected_revision: Revision,
    layer: EntityId,
    selection: SelectionShape,
    delta: Point,
) -> Result<LayerCommit, Error>
```

`paste_raster_selection`은 세션 클립보드(서버 측 `RasterClipboard`)에서 읽는다(S-D5).
`move_raster_selection`은 선택 영역을 `delta`만큼 옮긴다 — 원본을 지우고 옮긴
자리에 그린다.

**`move_raster_selection`은 원본을 먼저 복사해야 한다.** 제자리에서 옮기면 겹치는
부분이 자기 자신을 덮어써서 번진다. 원본에서 읽어 새 버퍼에 그린 뒤 통째로 바꾼다.
이 순서를 주석에 남긴다 — 1a의 `write_geometry`와 같은 종류의 "순서가 곧 정확성"이다.

### C3 — 레이어 조작

```rust
pub async fn duplicate_raster_layer(
    expected_revision: Revision,
    layer: EntityId,
) -> Result<LayerCommit, Error>

pub async fn reorder_layer(
    expected_revision: Revision,
    layer: EntityId,
    at: LayerPosition,
) -> Result<LayerCommit, Error>

pub async fn merge_layers(
    expected_revision: Revision,
    layers: Vec<EntityId>,
    into: EntityId,
) -> Result<LayerCommit, Error>
```

`merge_layers`는 `into`를 대상으로 삼아 나머지를 **아래에서 위로** 합성하고,
합성된 레이어들 중 `into`가 아닌 것을 삭제한다. 합성 순서는 z 순서이고, source-over다(S-D7).

**`merge_layers`가 1a의 Critical과 같은 함정을 갖는다.** 1a에서 `merge_pages`가
`ShapeLayer`의 기하를 옮기면서 `Line`의 끝점 불변식을 깨뜨렸다. 여기서는 레이어를
삭제하면서 **그 레이어의 자산을 참조하는 것이 남지 않는지** 확인해야 한다 —
특히 1c가 도입하는 `pattern` 역할 자산(P-D2)이 대상이다. **합성 후 삭제되는
레이어의 모든 역할 자산이 함께 정리되는지 테스트한다.**

삭제에는 `Edit::remove_entity(entity, RemovePolicy)`를 쓴다(`edit.rs:202`).
`RemovePolicy`는 `RejectNonEmpty`/`Cascade`/`PromoteChildren` 셋이다(`edit.rs:25-29`).
**합성으로 소비된 레이어는 `Cascade`로 지운다** — 그 레이어가 자식이나 자산을
갖고 있어도 함께 정리되어야 하기 때문이다. `RejectNonEmpty`를 쓰면 자식 있는
레이어에서 실패하고, `PromoteChildren`은 자식을 남겨 의도와 다르다. 이 선택을
주석에 남긴다.

레이어 순서 변경은 `Edit::move_entity(entity, parent, at)`이다(`edit.rs:134`).
`At`은 `Start`/`End`/`Before(anchor)`/`After(anchor)`이고, `At::Before`를 쓰려면
앵커가 같은 페이지에 있어야 한다 — 다른 페이지의 앵커를 주면 거부된다.

## 프론트

| 요소 | 변경 |
|---|---|
| `CanvasTool` (`store.ts:23`) | `select_rect`, `select_lasso` 추가. 기존 `manual_ocr_lasso`는 남긴다 |
| `store.ts` | `selection`, `setSelection`, `clearSelection`, `clipboard` 상태 |
| `CanvasOverlay` | 기존 `LassoOverlay`(`:199`)를 일반화해 선택 오버레이로 쓴다. **점선 애니메이션(행진하는 개미)은 넣지 않는다** — 정적 점선으로 충분하고 애니메이션은 렌더 비용과 테스트 복잡도를 만든다 |
| `CanvasWorkspace` | 선택 도구의 드래그 처리, `Escape`로 해제, `Ctrl+A` 전체 선택 |
| `Inspector` | 선택이 있을 때 영역 조작 버튼 |
| 단축키 | `Delete` 영역 삭제, `Ctrl+C`/`Ctrl+X`/`Ctrl+V`, `Ctrl+J` 레이어 복제 |
| i18n | 9개 로케일 |

## 테스트

| 계층 | 테스트 |
|---|---|
| 프론트 | `point_in_polygon` — 볼록·오목·자기 교차 다각형, 경계 위의 점 |
| 프론트 | 사각형 선택의 정규화 (음수 폭·높이를 정규화하는지) |
| `koharu-app` | `point_in_polygon`의 Rust 구현이 프론트와 **같은 리터럴 케이스**를 통과하는지 |
| `koharu-app` | `Erase`가 선택 안만 알파 0으로 만드는지, 밖은 그대로인지 |
| `koharu-app` | `Keep`이 선택 밖만 알파 0으로 만드는지 |
| `koharu-app` | `CopyToNewLayer`가 새 레이어를 만들고 원본을 바꾸지 않는지 |
| `koharu-app` | `CutToNewLayer`가 원본 선택 영역을 알파 0으로 만드는지 |
| `koharu-app` | `move_raster_selection`이 겹치는 이동에서 **자기 복사를 하지 않는지** |
| `koharu-app` | 붙여넣기가 클립보드가 비었을 때 거부되는지 |
| `koharu-app` | `merge_layers`가 z 순서로 합성하고 나머지를 삭제하는지 |
| `koharu-app` | **`merge_layers`가 삭제된 레이어의 모든 역할 자산을 정리하는지** (1a Critical 재발 방지) |
| `koharu-app` | 도형·텍스트 레이어에 영역 조작이 거부되는지 (S-D3) |
| `koharu-app` | 선택 영역이 페이지 밖이어도 오류가 아닌지 (가장자리 클램프) |
| 프론트 | 선택 오버레이가 페이지 좌표를 화면 좌표로 옳게 옮기는지 |
| 프론트 | 영역 조작 버튼이 선택 없을 때 비활성인지 |

## 실패 시나리오

| 상황 | 처리 |
|---|---|
| 선택 영역이 페이지와 교차하지 않음 | 조작은 성공하되 아무 픽셀도 안 바뀐다. 거부하지 않는다 |
| 선택 다각형이 점 2개 이하 | 거부 (`point_in_polygon`이 의미를 갖지 않는다) |
| 선택 영역 자기 교차 | 허용. even-odd 규칙으로 판정하고 그 규칙을 테스트에 고정한다. **정하라: even-odd vs nonzero — 계획에서 정한다** |
| 클립보드가 비었는데 붙여넣기 | 거부 |
| 붙여넣기 오프셋이 페이지 밖 | 허용하되 화면 밖 부분은 잘린다. 거부하지 않는다 |
| `merge_layers`에 레이어가 하나만 | 거부 (합성할 것이 없다) |
| `merge_layers`의 `into`가 목록에 없음 | 거부 |
| `reorder_layer`의 대상이 같은 페이지가 아님 | 거부 |

## 다른 서브프로젝트와의 관계

- **B2(버킷 채우기)와 상호작용한다.** 버킷은 flood fill로 영역을 정하고, 선택은
  다각형으로 정한다. 선택이 있을 때 버킷이 "선택 안으로 제한"되어야 하는지는
  **B2의 결정**이며, C는 선택 상태를 노출하기만 한다.
- **B1(브러시 그라디언트)과 상호작용한다.** 같은 질문 — 선택이 있으면 붓이
  선택 안으로 제한되는가. C는 답하지 않는다.
- **E(만화 제작 보조)가 C를 쓴다.** 톤 붙이기·말풍선 자르기는 선택 영역을 쓴다.

## 열린 문제

1. **`point_in_polygon`의 규칙**: even-odd와 nonzero 중 무엇인가. 자기 교차
   올가미에서 결과가 달라진다. **계획에서 정하고 테스트에 고정한다.**
2. **선택 오버레이의 렌더 계층.** 선택은 캔버스(WASM) 위에 SVG로 그릴지, WASM
   안에 그릴지. 기존 `LassoOverlay`가 SVG이므로 SVG가 자연스럽지만, 확대/축소
   시 점선 두께가 화면 기준인지 페이지 기준인지 정해야 한다.
3. **클립보드의 위치 기준.** 복사한 픽셀의 원래 위치를 기억해 같은 자리에
   붙여넣을지, 커서 위치에 붙여넣을지. 1차는 오프셋 인자를 받는 것으로 정했으나
   UI가 그 오프셋을 어떻게 정하는지는 미정이다.
