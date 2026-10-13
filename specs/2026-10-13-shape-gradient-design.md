# 도형 그라디언트 채우기 설계 (1b)

작성일: 2026-10-13

## 배경

1a는 도형 도구를 완성했다: `ShapeLayer` 컴포넌트, 벡터 경로 생성, 렌더러 배선,
캔버스 미리보기, 4개 커맨드와 HTTP 라우트, 래스터화, PSD 내보내기, 도구 UI.
도형의 채우기는 단색 하나뿐이다.

1a는 1b를 위해 `ShapeFill`을 처음부터 열거형으로 설계했다
(`specs/2026-10-13-shape-tools-design.md:63` D6). 실제 형태는
`crates/koharu-scene/src/components/shapes.rs`의 `ShapeFill::Solid([u8;4])`
하나이고, wire 형태는 `{solid:[r,g,b,a]}`로 serde에 고정되어 있다.

1b는 여기에 **그라디언트 채우기**를 추가한다.

## 범위

이 문서는 **1b**만 다룬다. 그라디언트 채우기(선형·방사·스윕)와 그 편집 UI다.

### 패턴 채우기는 1c로 분리한다

1a 시점에 "1b = 그라디언트 + 패턴"으로 묶여 있었으나, 조사 결과 두 기능의
비용이 크게 다르다.

**그라디언트는 대부분 이미 있다.** `peniko` 0.6.1이 필요한 타입을 전부
제공한다:

| 타입 | 위치 | 역할 |
|---|---|---|
| `Gradient` | `peniko-0.6.1/src/gradient.rs:301` | `kind` + `extend` + `stops` + 보간 설정 |
| `GradientKind` | `:269` | `Linear` / `Radial` / `Sweep` |
| `LinearGradientPosition` | `:144` | `start`·`end` 점 |
| `RadialGradientPosition` | `:164` | 중심·반지름 (2점 변형 지원) |
| `SweepGradientPosition` | `:209` | 중심·시작/끝 각도 |
| `ColorStop` / `ColorStops` | `:28` / `:101` | `offset: f32` + `DynamicColor` |
| `Extend` | `brush.rs:159` | `Pad` / `Repeat` / `Reflect` |

그리고 `vello` 0.10은 `peniko::Brush`를 받으므로 `Scene::fill`에 그라디언트를
그대로 넘길 수 있다. `koharu-rasterizer`는 이미 vello/peniko에 의존한다.
즉 1b는 **새 렌더링 엔진이 아니라 열거형 변형 추가**다 — 1a의 D6이 노린 것.

**패턴은 그렇지 않다.** 타일 이미지를 그리려면 새 리소스 종류가 필요하다:

- `PreparedResourceKind`는 현재 `Font`와 `EncodedRaster { width, height }`
  뿐이다 (`crates/koharu-rasterizer/src/prepared.rs:792-795`).
- `PreparedResource`는 `Font { bytes }`와 `EncodedRaster { width, height,
  media_type, bytes }`다. 패턴 이미지는 후자로 표현할 수 있으나,
  **`ResourceId`가 `w/h/media_type/bytes`로 해시되므로 같은 이미지를 두 곳에서
  참조할 때 중복 저장**되고, `validate_frame`이 "참조되지 않은 리소스"를
  거부하므로 패턴용 리소스의 수명 규칙을 새로 정해야 한다
  (`prepared.rs:911-915`).
- 캔버스 WASM은 리소스를 `install_resource`로 개별 설치한다
  (`crates/koharu-canvas/src/web.rs:1254`). 패턴은 도형마다 이미지가 붙으므로
  매니페스트에 새 리소스 참조를 실어 보내는 경로가 필요하다.
- 패턴 이미지를 어디서 얻는가(파일? 클립보드? 프로젝트 자산?)는 별도 결정이다.
  `D — 페이지 변형 + 이미지 삽입` 서브프로젝트와 겹친다.

따라서 **1b는 그라디언트만** 하고, 패턴은 1c로 분리한다. 패턴을 1b에 밀어
넣으면 "저장 모델 + 리소스 수명 + 전송 + 이미지 출처"가 한 스펙에 섞여
리뷰가 불가능해진다.

**1b에 포함하지 않는 것** (각각 별도 스펙):

- **1c — 패턴 채우기**: 타일 이미지 리소스, 리소스 수명 규칙, 이미지 출처
- **B — 칠하기·색보정 + 브러시 그라디언트**: 1b의 `GradientSpec` 타입을
  재사용하되 래스터 픽셀 합성과 스트로크 파이프라인을 다룬다

## 결정 사항

| # | 결정 | 근거 |
|---|---|---|
| G1 | 그라디언트는 `peniko::Gradient`를 그대로 쓰지 않고 자체 `GradientSpec`으로 저장한다 | `DynamicColor`·`ColorSpaceTag`·`HueDirection`은 저장 스키마에 필요 없는 복잡도다. 자체 타입이면 sRGB 4채널로 고정되고 `revisioned` 스키마가 단순해진다 |
| G2 | `GradientSpec`은 `kind` + `stops` + `extend`만 갖는다 | 보간 색공간·hue 방향·알파 공간은 1b에서 sRGB 기본값으로 고정한다. 나중에 필요하면 필드를 추가한다(깊은 병합이 흡수) |
| G3 | 좌표는 **프레임 로컬 정규화 좌표**(0..1)로 저장한다 | 도형을 옮기거나 크기를 바꿔도 그라디언트가 따라간다. 페이지 절대 좌표로 저장하면 변형마다 그라디언트를 다시 계산해야 한다 |
| G4 | `PreparedFill`에 `Gradient` 변형을 추가한다 | 1a가 `PreparedFill`을 열거형으로 만든 이유가 이것이다 |
| G5 | 스윕 그라디언트도 1b에 포함한다 | `peniko`가 `Sweep`을 이미 제공하고 추가 비용이 `GradientKind` 변형 하나뿐이다. 다만 UI는 선형·방사만 노출하고 스윕은 데이터 모델에만 둔다 |
| G6 | 스톱은 최소 2개, 최대 32개 | 2개 미만이면 그라디언트가 성립하지 않는다. 32개는 편집 UI와 성능의 실용 상한 |
| G7 | `ShapeFill::Solid`는 그대로 둔다 | 단색이 압도적으로 흔하고, 그라디언트로 표현하면 스톱 2개가 되어 wire 크기와 UI가 모두 나빠진다 |

### 왜 좌표를 정규화하는가 (G3)

도형은 이동·크기조절·회전된다. 그라디언트 ��표를 페이지 절대값으로 저장하면
`set_geometries`가 폴리곤을 갱신할 때마다 그라디언트 좌표도 함께 변환해야
한다. **이것이 1a에서 `Line`의 이중 저장 때문에 Critical 버그를 낳은 바로 그
패턴이다** — 두 개의 권위가 어긋날 수 있게 만드는 것.

정규화 좌표(프레임의 좌상단이 `(0,0)`, 우하단이 `(1,1)`)로 저장하면 폴리곤이
바뀌어도 그라디언트는 자동으로 따라간다. 변환해야 할 것이 없다.

렌더러는 `shape_frame`이 이미 계산한 프레임 상자를 써서 정규화 좌표를 페이지
좌표로 편다:

```
page_x = frame.x + normalized_x * frame.width
page_y = frame.y + normalized_y * frame.height
```

회전은 경로의 `transform`이 이미 담당하므로 그라디언트는 **로컬 좌표계**에서
정의하면 된다. 다만 `peniko::Gradient`는 브러시 자체의 변환을 갖지 않으므로,
회전한 도형의 그라디언트는 **경로와 같은 아핀 변환을 그라디언트 좌표에도
적용**해야 정렬된다. 이 점이 1b의 유일한 좌표 함정이며 테스트로 못 박는다.

## 저장 모델

### 새 타입

`crates/koharu-scene/src/components/shapes.rs`에 추가한다.

```rust
/// 1b가 추가하는 그라디언트 채우기.
///
/// 좌표는 프레임 로컬 정규화 값이다: 프레임 좌상단이 (0,0), 우하단이 (1,1).
/// 폴리곤이 바뀌어도 그라디언트는 다시 계산할 필요가 없다.
#[revisioned(revision = 1)]
pub struct GradientSpec {
    pub kind: GradientKind,
    pub stops: Vec<GradientStop>,
    pub extend: GradientExtend,
}

#[revisioned(revision = 1)]
pub enum GradientKind {
    /// start에서 end로 가는 직선을 따라 보간한다. 두 점 모두 정규화 좌표다.
    Linear { start: [f32; 2], end: [f32; 2] },
    /// 중심과 반지름(프레임 폭 기준 비율)으로 방사한다.
    Radial { center: [f32; 2], radius: f32 },
    /// 중심에서 시작 각도와 끝 각도 사이를 쓸어간다. 각도는 도(degree)다.
    Sweep { center: [f32; 2], start_degrees: f32, end_degrees: f32 },
}

#[revisioned(revision = 1)]
pub struct GradientStop {
    /// 0.0 ~ 1.0. 정렬되어 있지 않아도 되며, 렌더러가 정렬해 쓴다.
    pub offset: f32,
    pub color: [u8; 4],
}

#[revisioned(revision = 1)]
pub enum GradientExtend {
    /// 마지막 스톱 색으로 채운다 (기본).
    Pad,
    /// 반복한다.
    Repeat,
    /// 거울처럼 뒤집어 반복한다.
    Reflect,
}
```

`ShapeFill`에 변형을 추가한다:

```rust
pub enum ShapeFill {
    Solid([u8; 4]),
    Gradient(GradientSpec),
}
```

`ShapeFill` 자체의 `revisioned` 리비전은 **올리지 않는다.** 근거는 `revision`
0.30의 wire 형식이다: 열거형은 `u8 tag || payload`로 인코딩되고 태그의
bits 0..=4가 variant id다(`revision-0.30.0/README.md:545`, `:548-549`).
`Gradient`를 **마지막에** 추가하면 기존 `Solid`의 id 0이 바뀌지 않으므로
기존 데이터가 그대로 읽힌다. 파생 타입의 리비전 헤더(`u16`, 같은 문서 `:347`)도
그대로 1이다.

**이 추론을 테스트로 고정한다.** 1a에서 "그럴 것이다"라는 추론이 여러 번
틀렸으므로, 1b는 다음을 명시적 테스트로 요구한다:

1. `ShapeFill::Solid([1,2,3,255])`를 `revision::to_vec`으로 인코딩한다
2. 그 바이트를 `Gradient` 변형이 존재하는 상태의 `ShapeFill`로 디코딩한다
3. `Solid`로 읽히고 색이 보존되었는지 단언한다

이 테스트는 `Gradient` 추가가 기존 저장 데이터를 깨뜨리지 않는다는 유일한
직접 증거다. 리비전 번호를 올렸다면 이 테스트가 실패한다.

### 검증

`GradientSpec::validate`가 거부하는 것:

- `stops.len()`이 2 미만 또는 32 초과
- 스톱 `offset`이 비유한이거나 0.0..=1.0 밖
- 같은 `offset`을 가진 스톱이 둘 이상 (어느 색이 이기는지 정의되지 않는다)
- `Linear`의 `start`/`end`에 비유한 값이 있거나 두 점이 같음 (길이 0이면 방향이 없다)
- `Radial`의 `center`가 비유한이거나 `radius`가 비유한이거나 0 이하
- `Sweep`의 `center`가 비유한이거나 각도가 비유한

**거부하지 않는 것**: 스톱이 정렬되어 있지 않은 것. 렌더러가 정렬해 쓰므로
저장 순서는 자유다. 검증으로 강제하면 UI가 스톱을 중간에 삽입할 때마다
전체를 다시 정렬해야 한다.

`ShapeLayer::validate`는 `ShapeFill::Gradient`일 때 위 검증을 호출한다.

### wire 형태

1a의 `ShapeFill`은 serde 외부 태그를 쓴다(`{solid:[r,g,b,a]}`). 새 변형은
구조 변형이므로 `{gradient:{kind:{linear:{start:[..],end:[..]}},stops:[..],extend:"pad"}}`
형태가 된다. `GradientKind`도 외부 태그이므로
`{"linear":{"start":[0.0,0.0],"end":[1.0,1.0]}}`,
`{"radial":{"center":[0.5,0.5],"radius":0.5}}`,
`{"sweep":{"center":[0.5,0.5],"start_degrees":0.0,"end_degrees":360.0}}`다.

**1a와 마찬가지로 이 형태를 `serde_json` 테스트로 못 박는다.** TypeScript
타입이 여기에 맞춰야 하고, 그 테스트가 유일한 강제 수단이다.

## 렌더러

### `PreparedFill` 확장

```rust
pub enum PreparedFill {
    Solid { color: [u8; 4], rule: FillRule },
    Gradient { gradient: PreparedGradient, rule: FillRule },
}

pub struct PreparedGradient {
    /// 페이지 좌표계. 정규화 → 페이지 변환을 이미 적용한 값이다.
    pub kind: PreparedGradientKind,
    pub stops: Vec<PreparedGradientStop>,
    pub extend: PreparedGradientExtend,
}
```

좌표를 **경로 로컬 좌표계로 펴서 저장**하는 이유: 경로(`PreparedPath.elements`)가
로컬 좌표이고 그 변환은 `PreparedPath.transform`에 있으므로, 그라디언트도 같은
계에 두면 회전·이동이 브러시에도 자동으로 적용된다(아래 "회전 정렬" 참조).
정규화 좌표를 여기까지 들고 가면 소비자마다 프레임 크기를 다시 알아야 한다.

### 렌더링

`compile_scene`(`crates/koharu-rasterizer/src/frame.rs`)의 `FillPath` 분기가
`PreparedFill::Gradient`일 때 `peniko::Gradient`를 만들어 `Scene::fill`에
넘긴다. `peniko::Gradient::new_linear`/`new_radial`/`new_sweep`이 각각 있고
(`gradient.rs:344`, `:356`, `:393`), `stops`는 `ColorStops`로 감싼다.

**회전 정렬에는 `brush_transform`을 쓰지 않는다 — `None`을 넘긴다.**

초판은 "브러시 변환에 경로와 같은 아핀을 넘긴다"고 썼으나 **그것이 틀렸다.**
`Scene::fill`은 브러시 변환을 경로 변환에 **합성**한다:

```rust
// vello-0.10.0/src/scene.rs:328-331
if let Some(brush_transform) = brush_transform
    && self.encoding.encode_transform(Transform::from_kurbo(&(transform * brush_transform)))
```

따라서 `brush_transform = Some(T)`는 브러시가 `T * T`를 받게 만든다 — 경로
변환을 이중 적용한다. 회전한 도형의 그라디언트가 도형보다 두 배 각도로
돌아가거나(또는 특정 회전에서 단색처럼 뭉개진다).

정답은 **`None`**이다. `PreparedFill::Gradient`의 제어점이 이미 경로와 같은
로컬 좌표계에 있고, vello가 경로 변환을 브러시에 자동으로 적용하므로 추가
변환이 필요 없다. `Some(Affine::IDENTITY)`도 같은 결과지만(`T * I == T`)
`None`이 의도를 더 분명히 한다.

vello 자신의 `draw_blurred_rounded_rect`가 같은 규율을 쓴다 — 브러시에는
**추가** 변환만 넘기고 경로 변환을 반복하지 않는다
(`vello-0.10.0/src/scene.rs:296-298`).

`Solid`도 `None`을 넘긴다(색은 변환과 무관하다). 결과적으로 `brush_transform`
인자는 이 코드베이스에서 항상 `None`이며, Task 7이 그 사실을 테스트로 고정한다.

**정규화 → 로컬 변환은 준비 단계에서 한다.** `shape_renderer`의 도형별 경로
생성에서 정규화 좌표(0..1)를 프레임 로컬 좌표로 편다:

```
local_x = normalized_x * frame.width
local_y = normalized_y * frame.height
```

`frame`은 `shape_frame`이 이미 계산한 `ShapeFrame`이다. `Polygon`/`Freeform`
위임 경로도 같은 함수를 지나므로 함께 처리된다.

### 캔버스 미리보기

`CanvasPathFill`(`packages/bridge/src/canvas.ts:32`)에 그라디언트 변형을
추가한다. 현재는 `{ Solid: { color, rule } }` 하나다. WASM
`begin_shape`/`update_shape`는 `PreparedPath`를 그대로 받으므로
(`crates/koharu-canvas/src/web.rs:1731`, `:1759`), 서버 변경 없이 wire 타입만
넓히면 된다.

**프론트의 `shapePreviewPath`가 그라디언트를 조립해야 한다.** 1a에서 이
함수가 WASM `PreparedPath`의 wire 형태를 손으로 만드는 유일한 지점이었고
(테스트로 못 박음), 그라디언트가 추가되면 그 조립 코드도 함께 늘어난다.

**중복은 정규화 → 로컬 변환 한 곳뿐이고, 회전 처리는 두 경로가 다르다.**

**서버:** `PreparedPath.transform`이 회전을 담고 `brush_transform`은 `None`이므로,
그라디언트 제어점은 **프레임 로컬** 좌표에 둔다. vello가 경로 변환을 브러시에
자동 적용한다.

**프론트:** `shapePreviewPath`의 `transform`은 항상 `identity`이고
(`packages/koharu/lib/shapes.ts:147`), 회전은 `shapePolygon`이 **폴리곤 점에
직접 굽는다** (`shapes.ts:73`, `:79`의 `rotatePoint`). 그러므로 프론트의
그라디언트 제어점도 **폴리곤과 같은 방식으로 페이지 좌표로 회전**해야 경로와
정렬된다. 프레임 로컬에 그대로 두면 회전한 도형의 그라디언트가 축정렬로 남아
경로와 어긋난다.

두 경로의 **최종 좌표계는 같아야 한다** — 어느 쪽이든 그려지는 그라디언트는
페이지 위에서 같은 위치·방향이어야 한다. 서버는 vello가 대신 회전해 주고
프론트는 직접 회전한다는 차이만 있다.

두 구현이 어긋나면 미리보기와 커밋 결과가 달라지므로, **양쪽을 같은
리터럴 테스트로 고정한다** (서버는 Rust 테스트, 프론트는 vitest — 프론트
테스트는 회전한 프레임에서 제어점이 실제로 돌아갔는지까지 단언한다). 별도
공유 모듈을 만드는 것은 WASM 경계를 넘나드는 코드에서는 과잉이다.

## UI

| 요소 | 변경 |
|---|---|
| `Inspector` | 채우기 종류 선택(단색/그라디언트), 그라디언트 종류(선형/방사), 스톱 편집기 |
| `ToolBar` | 도구막대 팝오버에도 같은 선택 (기본 스타일) |
| `lib/shapes.ts` | `defaultGradientSpec()`, `defaultShapeStyle`이 그라디언트 선택 시 그라디언트를 반환 |
| store | `shapeStyle`이 이미 `ShapeStyle` 전체를 담으므로 변경 없음 |
| i18n | 9개 로케일에 `inspector.shape.fill.solid`/`.gradient`, `inspector.shape.gradient.linear`/`.radial`, 스톱 관련 키 |

**스톱 편집기의 최소 형태**: 스톱 목록(색 + 위치), 스톱 추가/삭제, 위치 슬라이더.
그라디언트 미리보기 바를 클릭해 스톱을 추가하는 방식은 1b 범위 밖이다.

**스윕은 UI에서 노출하지 않는다** (G5). 저장 모델과 렌더링�� 지원하되 편집
UI가 없으므로, 1b가 만든 데이터에 스윕이 나타날 수는 없다.

## 실패 시나리오

| 상황 | 처리 |
|---|---|
| 스톱 1개 | 검증에서 거부 |
| 스톱 33개 | 검증에서 거부 |
| 모든 스톱이 같은 offset | 검증에서 거부 |
| 스톱이 정렬되지 않음 | 거부하지 않는다. 렌더러가 정렬한다 |
| `Linear`의 두 점이 같음 | 검증에서 거부 (방향 없음) |
| `Radial`의 반지름 0 이하 | 검증에서 거부 |
| 그라디언트가 프레임 밖으로 나감 | 정상. `extend` 규칙이 처리한다 |
| 회전한 도형의 그라디언트 | `compile_scene`이 `brush_transform = None`을 넘긴다. vello가 경로 변환을 브러시에 자동 적용하므로 제어점이 로컬 좌표에 있으면 회전이 따라온다 |
| `fill: None` | 그라디언트여도 채우기 명령을 만들지 않는다 (1a 규칙 유지) |
| 화살촉 + 그라디언트 채우기 | 화살촉은 같은 경로의 subpath이므로 같은 그라디언트로 채워진다 (1a 동작 유지) |

## 테스트

| 계층 | 테스트 |
|---|---|
| `koharu-scene` | `GradientSpec::validate` 경계값 — 스톱 1/2/32/33, offset 범위, 중복 offset, 0길이 선형, 0반지름 |
| `koharu-scene` | 스톱 미정렬 허용 |
| `koharu-scene` | **`Solid` 바이트를 `Gradient` 변형이 있는 상태에서 디코딩해도 `Solid`로 읽히는지** (variant id 보존의 직접 증거) |
| `koharu-app` | `serde_json`으로 `ShapeFill::Gradient`의 정확한 wire JSON 고정 (TS 계약) |
| `koharu-app` | `serde_json`으로 `ShapeFill::Solid`의 JSON이 1a와 **문자 단위로 동일**한지 (회귀) |
| `koharu-renderer` | 정규화 좌표 → 프레임 로컬 좌표 변환 |
| `koharu-renderer` | 회전한 도형에서 그라디언트 제어점이 로컬 좌표에 남아 있는지 (회전이 경로 변환 몫임을 고정) |
| `koharu-rasterizer` | `PreparedFill::Gradient` 검증, `Solid` 하위 호환 |
| `koharu-rasterizer` | **`compile_scene`이 그라디언트와 `Solid` 모두에 `brush_transform = None`을 넘기는지** (이중 적용 방지의 핵심 함정) |
| `koharu-rasterizer` | `peniko::Gradient` 조립 — 스톱 정렬, `Extend` 매핑 |
| 프론트 | `shapePreviewPath`의 그라디언트 wire 형태 리터럴 단언 |
| 프론트 | **회전한 프레임에서 그라디언트 제어점이 페이지 좌표로 돌아갔는지** (`transform`은 identity이므로 회전이 제어점에 있어야 한다) |
| 프론트 | `defaultGradientSpec`이 검증을 통과하는 값인지 |

## 기존 프로젝트 호환

**기존 프로젝트는 그대로 열린다.** 1a와 같은 세 계층 논리가 적용된다:

- 컨테이너 `VERSION`은 바뀌지 않는다 (`crates/koharu-storage/src/format.rs:12`)
- `ShapeFill`에 변형을 추가하는 것은 `revisioned` 리비전 변경이 아니다.
  열거형 wire 형식이 `u8 tag || payload`이고 태그가 variant id를 담으므로
  (`revision-0.30.0/README.md:545`, `:548-549`), `Gradient`를 마지막에
  추가하면 `Solid`의 id 0이 유지된다. 기존 `Solid` 데이터는 그대로 읽힌다
- `PreparedFill`/`PreparedGradient`는 런타임 전용이며 매 프레임 재생성된다

이 논리는 **테스트로 고정한다** (위 테스트 표의 `Solid` 바이트 디코딩 항목).
Task 10이 1a에서 저장 포맷 왕복을 실제로 확인했고, 1b는 그 위에 변형 하나를
더할 뿐이다. 다만 **`ShapeFill::Gradient`를 가진 프로젝트를 구버전 앱으로
열면 그 도형이 `UnsupportedComponent`로 실패**한다 — 방향이 반대인
경우이며 AGENTS.md의 "Never add backward compatibility"와 일치한다.

## 열린 문제

없음. 패턴 채우기(1c)와 B~E 서브프로젝트는 이 문서의 범위 밖이다.
