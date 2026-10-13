# 도형 도구 설계 (1a)

작성일: 2026-10-13

## 배경

Koharu의 캔버스는 9개 도구만 제공한다 — `select`, `text`, `draw`,
`eraser`, `color_picker`, `remove`, `manual_ocr_box`, `manual_ocr_lasso`,
`pan` (`packages/koharu/lib/store.ts:20`). 사각형·타원·선 같은 도형을
그릴 방법이 없다.

편집자가 도형을 필요로 하는 지점은 두 가지다. 미번역 원문·효과음·스캔
아티팩트를 덮는 보정, 그리고 말풍선·강조 박스처럼 번역 결과물에 없던
요소를 새로 넣는 작업이다. 현재는 둘 다 브러시로만 가능하므로 곧은
테두리나 정확한 원을 그릴 수 없다.

기존 편집 기능을 먼저 정리하면 다음과 같다.

| 영역 | 있는 것 | 근거 |
|---|---|---|
| 래스터 페인트 | 브러시·지우개·인페인트 브러시, 전체페이지 PNG 레이어 | `project.rs:1134` |
| 텍스트 | 포인트/문단 텍스트, 폰트·크기·색·테두리·정렬·세로쓰기 | `Inspector.tsx:120` |
| 변형 | 이동·크기조절·회전 (텍스트·이미지) | `SelectionControls.tsx:77` |
| 레이어 | 순서·삭제·불투명도·표시·그룹·잠금 | `project.rs:1079` |
| 실행취소 | 문서 단위 undo/redo | `editing.rs:400` |
| 내보내기 | PNG / PSD / CBZ | `output.rs:36` |

## 범위

이 문서는 **1a**만 다룬다. 1a는 도형을 벡터 레이어로 저장하고,
기본 도형·채우기·선 스타일·점선·화살표·래스터화를 포함한다.

**1b로 미루는 것**: 그라디언트·패턴 채우기. 별도 스펙에서 다룬다.
1a는 `ShapeFill`을 처음부터 열거형으로 설계해 1b가 변형을 추가하기만
하면 되도록 한다.

**로드맵에서 아직 스펙이 없는 서브프로젝트** (이 문서는 이들의 기반이
아니다. 다만 1a의 표현 방식에 의존한다):

- B — 칠하기·그라디언트·색보정 (브러시 그라디언트 2종 포함)
- C — 선택 영역 + 래스터 레이어 조작
- D — 페이지 변형 + 이미지 삽입
- E — 만화 제작 보조 (스말풍선·스크린톤·클론·스냅)

## 결정 사항

| # | 결정 | 근거 |
|---|---|---|
| D1 | 도형을 벡터 레이어로 저장하고, 래스터화를 별도 액션으로 처음부터 제공 | 재편집 가능성이 래스터 굽기보다 중요하고, 되돌릴 수 없는 손실을 피한다 |
| D2 | `Geometry` 폴리곤이 히트테스트·프레임 복원의 권위. 렌더링은 `ShapeGeometry` 파라미터를 쓴다 | 기존 검증된 코드를 재사용한다. 아래 "왜 이 소유권인가"와 "무엇이 그리는가" 참조 |
| D3 | 회전 각도는 `ShapeLayer.angle_degrees`에 저장 | 폴리곤에서 복원 가능한 도형(사각형)과 불가능한 도형(타원·곡선)이 섞여 있다 |
| D4 | `ShapeLayer`에 `name` 필드를 둔다 | 래스터 레이어와 같은 이유로 레이어 목록에서 식별 가능해야 한다 |
| D5 | 래스터화 여부는 `rasterized: bool` 하나로 관리한다 | 별도 상태 머신이 필요 없다 |
| D6 | `ShapeFill`은 처음부터 열거형. 1a는 `Solid`만 갖는다 | 1b가 스키마를 갈아엎지 않게 한다 |
| D7 | 도구는 "도형" 하나 + 종류 선택 | 도구막대가 짧고 단축키 관리가 쉽다 |
| D8 | `ShapeLayer`는 `Geometry`를 참조하지 않고 같은 엔티티에 병존 | 기존 `TextLayout` + `Geometry` 패턴과 동일 |

### 왜 이 소유권인가 (D2)

대안은 `ShapeLayer`가 `frame`을 소유하고 `Geometry`를 파생으로 두는
것이었다. 진실의 원천이 하나라는 점에서 개념적으로 깔끔하다. 채택하지
않은 이유는 **기존 코드가 `Geometry` 폴리곤을 읽도록 이미 완성되어
있기 때문**이다.

- `geometry_frame`이 회전 사각형 4점에서 x/y/w/h/angle을 정확히
  복원한다 (`crates/koharu-renderer/src/frame.rs:628-643`)
- 프론트 `layerFrame`이 같은 계산을 `oppositeLengthsMatch` /
  `perpendicular` / `diagonalsBisect` 검사와 함께 수행하고,
  테스트가 이를 고정한다 (`packages/koharu/lib/geometry.ts:96-125`)
- 히트테스트·선택 사각형·PSD 내보내기가 모두 폴리곤을 읽는다

프레임을 도형이 소유하면 이 소비자 전부에 분기가 필요하다. 폴리곤을
권위로 두면 `element_frame`의 early-return 한 곳만 열면 된다.

## 저장 모델

### 새 컴포넌트

`crates/koharu-scene/src/components/layers.rs`에 추가하고,
`schema.rs`의 `component_schema!`에 비트 18로 등록한다 (현재 17까지
사용 중 — `crates/koharu-scene/src/schema.rs:44-63`).

```rust
#[revisioned(revision = 1)]
pub struct ShapeLayer {
    pub origin: Origin,
    pub name: String,
    /// 폴리곤에서 복원할 수 없는 종류를 위한 편집 파라미터.
    pub geometry: ShapeGeometry,
    pub style: ShapeStyle,
    /// 폴리곤이 표현하지 못하는 회전. TextLayout.angle_degrees와 같은 역할.
    pub angle_degrees: Option<f32>,
    pub rasterized: bool,
}
```

### 도형 종류

```rust
pub enum ShapeGeometry {
    /// 폴리곤 4점에서 x/y/w/h/angle이 복원되므로 모서리만 남긴다.
    Rectangle { corner_radius: f32 },
    /// 프레임에 내접하는 타원을 그린다.
    Ellipse,
    /// 폴리곤은 히트테스트용 샘플이고, 직선은 이 두 점으로 그린다.
    Line {
        start: [f64; 2],
        end: [f64; 2],
        head: Option<ArrowHead>,
    },
    /// 폴리곤이 곧 도형이다. 파라미터 없음.
    Polygon,
    /// 폴리곤이 곧 도형이다. 파라미터 없음.
    Freeform,
}

pub struct ArrowHead {
    pub length: f32,
    pub width: f32,
}
```

### 스타일

```rust
pub struct ShapeStyle {
    /// None이면 채우지 않는다. 테두리만 그리는 도형을 표현한다.
    pub fill: Option<ShapeFill>,
    /// None이면 테두리 없음.
    pub stroke: Option<StrokeStyle>,
}

/// 1a는 Solid만 갖는다. 1b가 Gradient 변형을 추가한다.
pub enum ShapeFill {
    Solid([u8; 4]),
}

pub struct StrokeStyle {
    pub color: [u8; 4],
    pub width: f32,
    /// None 또는 빈 배열은 실선. 0 이하 원소는 검증에서 거부한다.
    pub dash: Option<Vec<f32>>,
    pub cap: LineCap,
    pub join: LineJoin,
}
```

`fill: None`(칠하지 않음)은 채우기 명령을 **아예 만들지 않는다**는
뜻이다. `Some(Solid([r, g, b, 0]))`은 알파 0인 채우기 명령을 만들되
그리지 않는다는 뜻이다. 화면 결과는 같지만 전자가 낫다 — 명령이
없으면 합성 비용이 들지 않고, 기하 검증도 건너뛴다. 테두리만 있는
도형은 전자를 쓴다.

`fill`을 `Option`으로 두는 두 번째 이유는 1b 때문이다. `ShapeFill`이
열거형이 되면 `None`은 "채우기 종류를 고르지 않음"으로 그대로 남고,
`Some(Gradient(..))`가 추가된다. `Option` 없이 열거형만 두면
"칠하지 않음"을 `ShapeFill::None` 같은 변형으로 표현해야 하는데,
그러면 열거형이 "무엇으로 칠하는가"와 "칠하는가"를 섞는다.

### 검증

`ShapeLayer::validate`가 거부하는 것:

- `name`이 비었거나 4096바이트 초과하거나 NUL 포함 (`RasterLayer`와 동일 규칙)
- `corner_radius`가 음수 또는 비유한
- `stroke.width`가 음수 또는 비유한
- `dash` 원소에 0 이하 또는 비유한 값 (무한 루프 방지)
- `angle_degrees`가 `Some`인데 비유한
- `ArrowHead`의 `length`/`width`가 음수 또는 비유한

**거부하지 않는 것**: 폴리곤과 파라미터의 불일치. 예를 들어
`ShapeGeometry::Ellipse`인데 폴리곤이 3점인 경우다. 이때는
`Ellipse` 파라미터를 신뢰하고 그 3점이 만드는 프레임에 내접하는
타원을 그린다. 검증으로 강제하면 다른 도구가 만든 데이터가 열리지
않는다.

### 무엇이 그리는가 (권위의 정확한 범위)

D2의 "폴리곤이 권위"는 **기하의 출처**를 뜻하지, **그리는 데 쓰는
값**을 뜻하지 않는다. 두 가지를 구분한다.

| 용도 | 쓰는 값 |
|---|---|
| 히트테스트, 선택 사각형, `element_frame` 복원 | **폴리곤** (기존 코드 그대로) |
| 화면 렌더링, 래스터화, 내보내기 | **`ShapeGeometry` 파라미터 + 프레임** |

파라미터가 그리는 이유는 명확하다. 직선을 폴리곤 점으로 그리면
직선이 아니라 꺾은선이 되고, 타원을 폴리곤으로 그리면 점 개수만큼
각진 다각형이 된다. 파라미터는 손실 없는 표현이고, 폴리곤은
히트테스트와 프레임을 위한 근사다.

따라서:

- `Rectangle { corner_radius }` → 프레임에서 둥근 사각형을 그린다.
- `Ellipse` → 프레임에 내접하는 타원을 그린다.
- `Line { start, end, head }` → `start`·`end` 두 점으로 직선을 그리고,
  `head`가 있으면 `end` 쪽에 화살촉을 붙인다. 폴리곤은 히트테스트용
  샘플일 뿐이다.
- `Polygon` / `Freeform` → **폴리곤 자체가 도형이므로 폴리곤을 그린다.**
  이 둘에는 파라미터가 없다.

### 프레임을 어떻게 얻는가 — 회전이 핵심이다

"프레임"은 `GeometryFrame { bounds, angle_degrees }`, 즉 로컬 축의
상자와 각도다. 기존 함수 `geometry_frame(geometry, angle_degrees)`
(`crates/koharu-renderer/src/bubble.rs:643`)가 이 계산을 이미 한다.

**`angle_degrees`를 반드시 넘겨야 한다.** 이유는 회전된 타원에 있다:

회전된 타원의 외곽선 샘플 폴리곤의 **축 정렬 경계 상자는 원래
타원보다 크다.** 45° 회전한 원의 축 정렬 경계 상자는 변의 길이가
지름의 √2배인 정사각형이다. 그 상자에 내접하는 원을 그리면 원본보다
큰 원이 나온다.

`geometry_frame`은 `angle_degrees`가 주어지면 폴리곤을 그 각도만큼
역회전시켜 **작성된 축에서** 경계를 잰다
(`bubble.rs:647-685`). 그러면 경계가 정확한 지름이 되고, 그 로컬
상자에 타원을 그린 뒤 다시 각도를 적용하면 원본과 일치한다.

`angle_degrees`가 `None`이면 `geometry_frame`은 축 정렬 경계로
폴백한다(`bubble.rs:687-733`). 이는 회전하지 않은 도형에서 정확하고,
회전했지만 각도를 잃은 데이터에서는 근사치다. 각도를 저장하는
이유가 이것이다.

`Rectangle`도 같은 경로를 쓴다. 사각형은 폴리곤 4점이 회전을
정확히 담고 있어 각도 없이도 복원되지만, 메커니즘을 하나로 유지하는
편이 검증할 지점이 적다. `Polygon`과 `Freeform`은 각도 개념 없이
폴리곤을 그대로 그리므로 `element_frame`은 축 정렬 경계를 쓴다.

### 폴리곤 점 개수

`Geometry::validate`가 이미 3점 이상을 요구한다
(`crates/koharu-scene/src/components/spatial.rs:49`). `Line`은 시작과
끝 2점만 필요하므로 **이 제약과 충돌한다.**

해결: `Line`의 폴리곤은 **선을 따라 3점 이상으로 샘플링**한다 —
시작, 끝, 그리고 그 사이. 폴리곤은 히트테스트와 프레임 계산용이고,
렌더링은 `Line { start, end, head }` 파라미터로 정확한 직선을 그린다
(위 "무엇이 그리는가" 참조).

`Line`의 폴리곤을 만들 때는 시작·끝 사이를 **균등 분할**한다. 점
개수는 선 길이에 비례시키지 않고 고정한다 — 폴리곤은 히트테스트
정밀도만 좌우하므로, 히트테스트에 충분한 3점이면 충분하다.

### `Line`의 이중 저장 — 권위는 폴리곤이다

`Line`은 `start`/`end`와 폴리곤을 함께 저장하는 **유일한 이중 저장
지점**이다. 이 사실을 방치하면 실제 버그가 된다:

- 모든 변형 경로는 폴리곤만 갱신한다.
- `line_path`는 `start`/`end`로만 그린다 (폴리곤을 읽지 않는다).
- `element_frame`은 폴리곤을 읽는다.

결과: 선을 끌면 선택 상자(폴리곤)는 움직이는데 그려진 선은 제자리에
남고, 둘이 눈에 띄게 분리된다.

**계약: 폴리곤이 권위이고 `start`/`end`는 파생값이다.** 폴리곤을 쓰는
모든 경로는 `ShapeGeometry::resync_with_polygon(&mut self, points)`를
호출해 첫 점을 `start`, 마지막 점을 `end`로 다시 맞춘다. 호출 지점은
`set_geometries`의 도형 분기 **한 곳**이다 (위 "새 커맨드" 참조).

이 이중 저장을 없애는 선택지도 있었다 — `Line`에서 `start`/`end`를
지우고 폴리곤 첫·마지막 점을 쓰는 것이다. 채택하지 않은 이유는
`Line`의 두 끝점이 화살촉 방향 계산에 필요하고, 폴리곤이 3점 이상
균등 분할이므로 매번 첫·마지막을 꺼내는 것이 같은 정보를 다른
표현으로 다시 얻는 일이기 때문이다. 대신 파생값임을 명시하고
재계산 지점을 하나로 모은다.

## 렌더러

### `PreparedPath` 확장

`crates/koharu-rasterizer/src/prepared.rs:692`의 구조를 바꾼다.

```rust
pub struct PreparedPath {
    pub transform: [f64; 6],
    pub elements: Vec<PathElement>,
    pub fill: Option<PreparedFill>,
    pub stroke: Option<PreparedStroke>,
}

pub enum PreparedFill {
    Solid { color: [u8; 4], rule: FillRule },
    // 1b: Gradient(GradientSpec)
}

pub struct PreparedStroke {
    pub color: [u8; 4],
    pub width: f32,
    pub dash: Option<Vec<f32>>,
    pub cap: LineCap,
    pub join: LineJoin,
}
```

이것은 **파괴적 변경**이다. `fill: FillRule`과 `color` 필드가
사라지므로 글리프 생성 경로(`crates/koharu-renderer/src/text_renderer.rs:427`)와
`compile_scene`(`crates/koharu-rasterizer/src/frame.rs:494`)이 함께
바뀐다. AGENTS.md의 "Never add backward compatibility"에 따라 구형식
파서를 남기지 않고 모든 소비자를 한 번에 고친다.

**저장 데이터에는 영향이 없다.** `PreparedFrameManifest`와
`PREPARED_FRAME_MANIFEST_VERSION`(`prepared.rs:14`)은 런타임 전용이며
디스크에 남지 않는다. 매 프레임 `prepare_frame`이 다시 만든다
(`crates/koharu-renderer/src/renderer.rs:1244`).

### `PreparedLayerKind::Shape`

`prepare_frame`(`crates/koharu-renderer/src/frame.rs:474`)의 분기에
`Shape`를 추가하고, `PreparedLayer::validate`
(`prepared.rs:441`)에 `(Shape, Vector)` 조합을 허용한다.

도형과 텍스트가 같은 `PreparedContent::Vector`를 공유한다. 렌더링은
동일하고 `LayerKind`만 다르다.

### `LayerKind::Shape` (렌더러)

`crates/koharu-renderer/src/frame.rs:322`에 `Shape(ShapeMetadata)`를
추가한다. `NodeDescriptor`(`frame.rs:442`)에도 `Shape` 변형이 필요하다.

```rust
pub struct ShapeMetadata {
    pub geometry: ShapeGeometry,
    pub style: ShapeStyle,
    pub angle_degrees: Option<f32>,
}
```

`ShapeMetadata`는 `ShapeLayer`에서 렌더링에 필요한 값을 그대로
복사한 것이다. `element_frame`과 PSD 내보내기가 이 값을 읽는다
— 위 "`element_frame`의 early-return" 참조.

**도형의 `PreparedScene`은 씬 컴파일 시점에 만든다.** `ShapeLayer`
컴포넌트의 폴리곤과 파라미터에서 매번 경로를 생성한다. 저장 시점에
미리 만들지 않는 이유는 스타일을 바꿀 때마다 재생성이 필요하고,
폴리곤과 중복되기 때문이다.

이 변경은 `LayerKind`를 exhaustive match하는 모든 지점을 컴파일러가
찾아내게 한다. 확인된 소비자: `desktop/lib.rs:266`, `web.rs:1395`,
`prepared.rs:442`, `frame.rs:488,491,603`, `renderer.rs:1194,1206`,
`document.rs:110,140,150,178`.

### `element_frame`의 early-return

`element_frame`(`crates/koharu-renderer/src/frame.rs:602`)은 현재
텍스트가 아니면 즉시 `None`을 반환한다:

```rust
let LayerKind::Text(text) = layer.kind() else {
    return None;
};
```

도형은 여기서 `None`을 받으면 프레임이 없어 선택·이동·회전이 모두
불가능하다. `Shape` 분기를 추가하고 다음 순서로 값을 얻는다:

1. `ShapeMetadata.angle_degrees`(셰이프 컴파일 시 `ShapeLayer`에서
   복사해 둔 값)를 꺼낸다
2. `geometry_frame(layer.geometry(), angle)`을 호출한다
   (`crates/koharu-renderer/src/bubble.rs:643`)
3. 결과가 `None`이면 `None`을 반환한다 — 프레임 없는 도형은
   선택할 수 없고, 이는 폴리곤이 유효하지 않다는 뜻이다

`ShapeMetadata`에 `angle_degrees`를 두는 이유는 `element_frame`이
`PreparedLayer`가 아니라 `Layer`를 받기 때문이다. `ShapeLayer`
컴포넌트에서 매번 읽으려면 스냅샷 접근이 필요한데, `prepare_frame`
시점에 이미 `NodeDescriptor`를 만들고 있으므로 거기서 옮겨 담는다.

## 캔버스 미리보기 (WASM)

`CanvasState::composition`(`crates/koharu-canvas/src/web.rs:787`)이
이미 래스터/벡터로 분기한다:

```rust
if let Some(image) = layer.raster_image() {
    // 래스터 경로
} else {
    layer.append_vector_with_presentation(...)   // 851행
}
```

도형은 벡터이므로 **기존 `else` 갈래로 자동 진입한다. 새 코드가
필요 없다.**

열어야 할 게이트는 `begin_transform`의 종류 검사
(`web.rs:1395`)뿐이다.

그리는 동안의 실시간 미리보기는 `StrokeEdit`과 같은 방식의
`ShapeEdit` 상태를 `CanvasState`에 추가하고, `composition`의 기존
stroke 블록(`web.rs:854-878`)과 같은 패턴으로 합성한다.

## 커맨드와 전송 계층

### 새 커맨드

`crates/koharu-app/src/commands/canvas.rs`:

```rust
pub async fn add_shape(
    expected_revision: Revision,
    shape: ShapeGeometry,
    style: ShapeStyle,
    frame: Frame,
) -> Result<LayerCommit, Error>

pub async fn set_shape_style(
    expected_revision: Revision,
    shapes: Vec<ShapeStyleUpdate>,
) -> Result<Revision, Error>

pub async fn set_shape_geometries(
    expected_revision: Revision,
    shapes: Vec<ShapeFrameUpdate>,
) -> Result<Revision, Error>

pub async fn rasterize_shapes(
    expected_revision: Revision,
    shapes: Vec<EntityId>,
) -> Result<Revision, Error>
```

`set_shape_geometries`는 도형 전용 계약을 유지하되(비도형 요소를 사전
거부한다), 실제 구현은 `set_geometries`가 소유한다. 처음에는 두 함수를
분리하려 했으나, 변형 경로가 두 개가 되면 "폴리곤을 갱신할 때 `Line`의
끝점을 재계산한다"는 불변식을 두 곳에 두게 되어 한쪽에서 잊힐 수 있다.
`set_geometries`가 내부적으로 `Target::{Text, Shape}`로 갈라지고
`set_shape_geometries`가 그것에 위임한다.

**`Line`의 끝점 재계산이 이 통일의 이유다.** `Line`은 `start`/`end`와
폴리곤을 함께 저장하는 유일한 이중 저장 지점이다. 폴리곤이 권위이고
끝점은 파생값이므로, 폴리곤을 쓰는 모든 경로가
`ShapeGeometry::resync_with_polygon`을 호출해야 한다. 이 호출 지점이
하나여야 잊히지 않는다.

### HTTP 라우트

**Tauri 커맨드만 만들고 끝내면 안 된다.** `crates/koharu-rpc/src/routes/`에
캔버스 라우트가 이미 있으므로 거기에도 추가한다. 포크 HANDOFF §8이
기록한 교훈이다 — `get_llm_capabilities`가 Tauri 커맨드로만 있고
라우트가 없어서 RPC 클라이언트에서 깨졌다 (`fork/HANDOFF.md:213`).

### 변형 게이트 3곳

| 위치 | 현재 | 변경 |
|---|---|---|
| `crates/koharu-desktop/src/lib.rs:266` | `!matches!(kind, Text(_))` → 에러 | `Shape(_)` 허용 (`is_transformable` 술어로 추출) |
| `crates/koharu-app/src/commands/project.rs:1114` | `SceneTextLayout` 필수 | `set_geometries`가 `Target::{Text, Shape}`로 분기해 도형도 받는다 |
| `crates/koharu-canvas/src/web.rs:1395` | `kind != Text` | `Shape` 허용 |

`transform_geometries`는 `matches!`를 쓰므로 도형 변형을 추가해도
**컴파일 오류가 나지 않는다.** 즉 이 게이트는 조용한 동작 게이트이고,
열지 않으면 도형을 끌 수 없다는 사실이 아무 데도 드러나지 않는다.
그래서 술어를 떼어내 테스트로 고정한다.

## 래스터화

### 흐름

```
도형 레이어 선택 → rasterize_shapes
  → 그 도형만 PreparedScene으로 컴파일
  → Rasterizer::rasterize_scene(scene, page_w, page_h, [0,0,0,0], supersampled 4)
  → PNG 인코딩
  → Paint 래스터 레이어로 커밋 (apply_raster_stroke와 같은 blob 경로)
  → 원본 ShapeLayer.rasterized = true
```

`rasterize_scene`(`crates/koharu-rasterizer/src/native.rs:212`)이
슈퍼샘플링과 Lanczos 다운샘플을 내장하고
(`native.rs:519-540`), 미리보기와 내���내기가 같은 vello 경로를 쓰므로
결과가 일치한다.

### 알파 배경

`rasterize_scene`은 `background` 인자를 받는다
(`native.rs:217`). 도형은 투명 배경이어야 하므로 `[0, 0, 0, 0]`을
넘긴다. **`[0, 0, 0, 255]`를 넘기면 도형의 경계 상자 전체가 검게
칠해져 아래 그림을 덮는다.** 테스트로 못 박는다.

### 렌더링 우선순위

`rasterized: bool`만 두므로 래스터화 후에도 `ShapeLayer`는 남는다.

- `rasterized == false` → 벡터로 그린다
- `rasterized == true` → 래스터 자산을 그린다. 벡터는 무시한다

되돌리기가 데이터상 가능하다. 되돌리기 UI는 1a 범위 밖이다.

**래스터화된 도형의 편집 정책은 1a에서 정하지 않는다.** 렌더러는
`rasterized == true`인 도형을 래스터 레이어로 내보내므로, 그 도형은
`LayerKind::Image`가 되어 선택·이동 대상에서 빠진다(`selectableLayer`가
`image`를 포함하지만 텍스트·도형과 다른 경로다). 만약 선택되어
`set_geometries`가 도달하면 폴리곤만 바뀌고 그려지는 것은 래스터라
**조용한 시각적 무변화**가 된다.

1a는 이 상태를 "래스터화 = 편집 종료"로 둔다. 래스터화된 도형을 다시
벡터로 되돌리거나, 래스터 상태에서도 이동을 허용하는 정책은 별도
스펙에서 다룬다. 이 결정을 1a에서 명시하는 이유는 침묵이 "이동이
되어야 한다"는 기대로 읽히기 때문이다.

## 내보내기

### 기존 프로젝트 호환

**기존 프로젝트는 그대로 열린다.** 세 계층 모두 확인했다.

- **프로젝트 파일 포맷**: `VERSION: u32 = 1`(`crates/koharu-storage/src/format.rs:12`)은
  컨테이너 레이아웃을 가리키며 바뀌지 않는다
- **컴포넌트 스키마**: 각 컴포넌트가 자체 스키마 번호를 저장하고
  (`crates/koharu-scene/src/component.rs:172`), 디코딩 시 현재보다
  크면 거부한다(`component.rs:192-198`). 기존 프로젝트에는
  `ShapeLayer`가 없으므로 디코딩을 시도하지 않는다
- **`component_schema!` 비트**: `component_mask`는 런타임 조회이며
  (`crates/koharu-scene/src/schema.rs:20`) 저장 데이터에 기록되지
  않는다
- **미등록 컴포넌트**: `validate_component`가 `Ok(())`를 반환하며
  보존한다 (`schema.rs:36`)

도형이 추가된 프로젝트를 **구버전 앱**으로 열면 그 페이지만
`UnsupportedComponent`로 실패할 수 있다. 이는 방향이 반대인
경우이며, upstream AGENTS.md의 "Never add backward compatibility"와
일치한다.

### PSD

도형은 **래스터 레이어로 내보낸다.** PSD의 벡터 도형(`SoCo`)은
포맷이 복잡하고, `koharu-psd`는 이미 `frame.cropped(entity)` +
`rasterize` 경로를 갖고 있다 (`crates/koharu-psd/src/document.rs:158-161`).
텍스트만 편집 가능 레이어로 유지하는 기존 정책
(`TextLayerMode`, `document.rs:27`)과 일관된다.

수정 지점: `document.rs:110`, `:140`, `:150`, `:178`.

PNG/CBZ는 페이지를 통째로 래스터화하므로 변경이 없다.

## UI

| 요소 | 변경 |
|---|---|
| `CanvasTool` (`store.ts:20`) | `'shape'` 추가 |
| 단축키 (`store.ts:90`) | `shape: 'r'` 추가 |
| `ToolBar` (`ToolBar.tsx:44`) | 도구 아이콘 + 스타일 Popover (종류·채우기·선·두께·점선) |
| `selectableLayer` (`geometry.ts:61`) | `'shape'` 추가 — 안 하면 선택·이동이 안 된다 |
| `isLockedLayer` (`document.ts:11`) | `artwork`만 잠그므로 수정 불필요 |
| `Inspector` | 선택된 도형의 스타일 편집 섹션 |
| i18n | 9개 로케일에 `tools.shape`, `inspector.shape.*` 키 추가 |
| store | `shapeKind`·`shapeStyle` 상태를 `brush`와 같은 패턴으로 추가 |

그리기 제스처는 기존 `draftFrame`을 재사용한다
(`packages/koharu/components/editor/CanvasWorkspace.tsx:524-529`).
`'manual_box'`와 같은 드래그 UX이고 커밋 대상만 다르다.

## 테스트

AGENTS.md의 검증 정책에 따라 가장 작은 관련 검사만 돌린다.

| 계층 | 테스트 |
|---|---|
| `koharu-scene` | `ShapeLayer::validate` 경계값 (음수 반경, NaN, dash 0) |
| `koharu-rasterizer` | `PreparedPath` fill/stroke 조합, `(Shape, Vector)` 검증 통과 |
| `koharu-renderer` | 도형 → `PreparedScene` 경로 생성, `element_frame`이 도형에서 프레임 반환 |
| `koharu-renderer` | **45° 회전한 원**의 프레임이 지름을 유지하는지 (축 정렬 상자 함정) |
| `koharu-renderer` | 화살표 `Line`이 폴리곤 샘플이 아니라 `start`/`end`로 그려지는지 |
| `koharu-app` | **투명 배경 래스터화** — 알파 0 유지 |
| `koharu-psd` | 도형이 래스터 레이어로 나오는지 |
| 프론트 | `selectableLayer('shape')`, 드래그 → 프레임 계산, 스타일 store |
| 회귀 | 도형 없는 페이지를 열고 저장한 뒤 다시 여는 왕복 |

## 실패 시나리오

| 상황 | 처리 |
|---|---|
| 폴리곤이 자기교차 | vello가 `FillRule`에 따라 처리. 거부하지 않는다 |
| `stroke.width`가 도형보다 큼 | 정상 렌더링. 경계 밖으로 나간다 |
| `dash` 패턴 합이 0 | 검증에서 거부 |
| 곡선 점이 2개뿐 | `Geometry::validate`가 3점 미만 거부 |
| 폴리곤과 파라미터 불일치 | `ShapeGeometry` 파라미터를 신뢰한다. 폴리곤은 경계와 히트테스트에만 쓴다 |
| 회전된 도형의 `bounds` | `geometry_frame`에 `angle_degrees`를 넘겨 로컬 축에서 계산한다. 안 넘기면 폴리곤의 축 정렬 상자가 되어 회전 타원이 커진다 |
| `angle_degrees`가 `None`인 회전 도형 | `geometry_frame`이 축 정렬 경계로 폴백한다. 근사치이며, 1a가 만든 데이터는 각도를 항상 채우므로 정상 경로에서는 발생하지 않는다 |
| 래스터화 시 배경 | `[0,0,0,0]` 고정 |
| 같은 도형을 두 번 래스터화 | `rasterized == true`면 기존 래스터를 교체한다 |

## 열린 문제

없음. 1b(그라디언트·패턴)와 B~E 서브프로젝트는 이 문서의 범위 밖이며,
각자 별도 스펙에서 다룬다.
