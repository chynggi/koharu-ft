# 칠하기·그라디언트·색보정 설계 (B)

작성일: 2026-10-14

## 배경

1a는 도형 도구를, 1b는 도형 그라디언트를 완성했다. 그 과정에서 **재사용 가능한
그라디언트 타입**(`GradientSpec`/`GradientKind`/`GradientStop`/`GradientExtend`)이
`koharu-scene`에 들어갔고, 캔버스는 스트로크 파이프라인(WASM `beginStroke` →
`commit_paint` → `apply_raster_stroke` → `rasterize_stroke`)을 이미 갖고 있다.

B는 그 두 가지 위에 **래스터 영역 작업**을 얹는다. 브러시는 현재 단색
(`CanvasBrush { diameter, color }`)뿐이고, 채우기·그라디언트·색보정 도구가 없다.

### 무엇이 이미 있는가 (조사 결과)

**칠하기(버킷)의 알고리즘은 이미 있다.** `imageproc` 0.27.0의
`flood_fill_mut<P>(image: &mut Image<P>, x, y, fill_with: P)`가 4방향 span
filling을 구현한다(`imageproc-0.27.0/src/drawing/fill.rs:16`, 알고리즘은
`:5`의 MS-Paint 버킷 등가물). `imageproc`은 이미 워크스페이스
의존(`Cargo.toml:59`)이며 `koharu-ml`·`koharu-pipeline`이 쓴다. `koharu-app`에
**직접 의존을 추가**해야 한다.

**다만 이 함수는 허용 오차를 받지 않는다.** 내부 비교가
`P: Pixel + PartialEq`이고(`fill.rs:60-63`의 `inside`), 대상 픽셀과 **정확히
같은** 픽셀만 채운다. 그러므로:

- **허용 오차가 필요 없으면** `flood_fill_mut`을 그대로 쓴다.
- **허용 오차가 필요하면 직접 써야 한다.** 같은 span-filling 골격을
  복사하되 비교를 "채널별 차이가 `tolerance` 이하"로 바꾼다. `imageproc`의
  구현은 `stack`에 `(x1, x2, y, dy)`를 넣는 형태이므로 비교 함수 하나만
  바꾸면 되지만, 그 순회 로직을 우리가 소유하게 된다.

**결정: 1차는 `flood_fill_mut`을 그대로 쓰고 허용 오차를 넣지 않는다.** 이유는
(a) 오차 허용은 UI로 노출할 값이 아니고, (b) 앤티에일리어싱된 경계에서 오차가
"새는" 동작이 그려지지 않은 결과를 낳으며, (c) 무엇보다 순회 로직을 복사하면
그 알고리즘의 유지보수를 우리가 떠안는다. 허용 오차가 필요해지면 그때 직접
구현하고 그 결정을 그 스펙에 남긴다.

**색보정도 대부분 이미 있다.**
- `image::imageops::brighten_in_place(image, value: i32)` — 밝기
- `image::imageops::contrast_in_place(image, contrast: f32)` — 대비 (단위는 아래 참조)
- `image::imageops::huerotate_in_place(image, value: i32)` — 색조 (도 단위, `i32`)
- `imageproc::contrast::stretch_contrast_mut(image, input_min, input_max, output_min, output_max)` — 레벨.
  네 인자가 모두 `u8`이므로(`imageproc-0.27.0/src/contrast.rs:368-375`)
  우리 `Levels { low, high }`는 **입력 범위**로 매핑하고 출력은 `0..=255`로
  고정한다.

**흑백은 `grayscale`이 아니라 `grayscale_alpha`를 쓴다.** `grayscale`은 타입이
`ImageBuffer<Luma<_>, _>`로 바뀌며 **알파를 버린다** — 문서에 "Alpha channel is
discarded"라고 명시되어 있다(`image-0.25.10/src/imageops/colorops.rs:13`).
래스터 레이어는 페이지 크기 RGBA이고 투명 영역이 의미를 갖는다(1a가 래스터화
배경을 `[0,0,0,0]`으로 못 박은 이유다). `grayscale`을 쓰면 투명 영역이 불투명
검정이 되어 아래 그림을 덮는다. `grayscale_alpha`(`:20-25`)는 `LumaA`를 돌려주고
알파를 보존하므로, 채널만 회색으로 바꾸고 알파는 그대로 두는 처리를 직접 한다 —
`RgbaImage`를 순회하며 `r=g=b=luma(원래 rgb)`, `a=원래 a`.

**브러시 그라디언트는 스트로크 파이프라인 전체를 건드린다.** 현재 색은
`StrokeCommitDto::Paint { color: [u8;4] }`(`crates/koharu-canvas/src/web.rs:116`)로
단일값이고, 굽는 함수는 `rasterize_stroke(image, mode, color, diameter, points)`
(`crates/koharu-app/src/commands/project.rs:2067`)로 픽셀마다 같은 색을 쓴다.

## 범위

이 문서는 **B**만 다룬다. 세 덩어리다.

| 덩어리 | 내용 |
|---|---|
| B1 | 브러시 그라디언트 2종 (스트로크 진행 방향 / 고정 공간축) |
| B2 | 버킷 채우기 + 그라디언트 채우기 도구 |
| B3 | 색보정 (밝기·대비·레벨·색조·흑백) |

**B를 한 번에 하지 않는다.** 셋은 서로 다른 계층을 건드리고, 특히 B1은
**픽셀을 그리는 방식**을, B2는 **영역을 고르는 방식**을, B3은 **이미 있는
픽셀을 고치는 방식**을 바꾼다. 한 스펙에 담으면 리뷰가 불가능하다.

따라서 이 문서는 **셋의 공통 결정과 순서**를 정하고, 각 덩어리는 자기 스펙에서
다룬다. 착수 순서는 **B3 → B2 → B1**을 제안한다:
- B3(색보정)이 가장 독립적이고 기존 코드를 거의 안 건드린다 — `RgbaImage`를
  받아 제자리에서 고치는 순수 함수와 커맨드 하나면 된다.
- B2(칠하기)는 `flood_fill_mut` 래핑 + 도구 UI + 커맨드다. B3이 만든 "래스터
  레이어에 픽셀 작업을 적용하는" 경로를 재사용한다.
- B1(브러시 그라디언트)이 가장 파급이 크다 — WASM 미리보기, 커밋 DTO,
  `rasterize_stroke`, 프론트 브러시 상태가 모두 바뀐다.

**B에 포함하지 않는 것**: 패턴 채우기(1c), 선택 영역(C), 클론 스탬프·스크린톤(E).

## 결정 사항

| # | 결정 | 근거 |
|---|---|---|
| B-D1 | 브러시 그라디언트는 **두 방식 모두** 제공한다: `Stroke`(진행 방향)와 `Space`(고정 축) | 1b 브레인스토밍에서 확정. 전자는 잉크 번짐, 후자는 배경 칠하기에 쓰이며 서로 대체하지 못한다 |
| B-D2 | 스트로크 그라디언트는 `GradientSpec`을 **재사용하지 않는다** — 별도 `BrushRamp { from: [u8;4], to: [u8;4] }` | `GradientSpec`의 좌표는 정규화 프레임 로컬이고 스톱은 임의 개수다. 스트로크는 **두 색과 매개변수화 규칙**만 필요하고, 정규화 좌표가 의미를 갖지 않는다. 재사용하면 `kind`·`stops`·`extend` 중 아무것도 쓰지 않는 필드를 저장하게 된다 |
| B-D3 | 채우기·색보정은 **래스터 레이어에만** 적용한다 | 도형 레이어에는 벡터 파라미터가 있어 픽셀 연산이 의미를 갖지 않는다. 대상이 래스터가 아니면 커맨드가 거부한다 |
| B-D4 | 버킷 채우기는 `imageproc::flood_fill_mut`을 그대로 쓴다 — 4방향이고 **허용 오차가 없다**(그 함수가 `PartialEq`로 정확히 같은 픽셀만 채운다). 오차 허용은 넣지 않는다 | 8방향은 대각선으로 새어 나가고, 오차 허용은 UI로 노출할 값이 아니며, 무엇보다 순회 로직을 복사하면 그 알고리즘의 유지보수를 우리가 떠안는다. 필요해지면 그때 직접 구현한다 |
| B-D5 | 채우기는 **한 번의 커밋**으로 전체 페이지 PNG를 다시 쓴다 | `apply_raster_stroke`가 이미 그렇게 한다(페이지 크기 이미지 → PNG → `set_asset`). 같은 경로를 쓰면 되돌리기·동기화·PSD가 자동으로 맞는다 |
| B-D6 | 색보정은 **미리보기 없이** 적용 즉시 커밋한다 | 미리보기는 WASM에 픽셀을 밀어넣는 새 경로가 필요하다. 색보정은 반복 조정이 잦으므로 미리보기가 값을 하지만, 1차는 "적용 → 결과 확인 → 되돌리기(Ctrl+Z)"로 충분하다. 미리보기는 별도 스펙 |
| B-D7 | 색보정 파라미터는 `i32`/`f32` 원시값으로 커맨드에 넘기고 **저장하지 않는다** | 적용 결과가 픽셀이므로 파라미터는 재편집 대상이 아니다. 저장하면 도형의 `ShapeStyle`처럼 두 권위가 생긴다 |
| B-D8 | **모든 색보정은 알파를 보존한다** — 흑백만 해당되지 않으므로 흑백도 `grayscale_alpha` 경로로 직접 구현한다 | 래스터 레이어의 투명 영역이 의미를 갖는다(1a가 래스터화 배경을 `[0,0,0,0]`으로 못 박았다). `image::imageops::grayscale`은 알파를 버리므로 투명 영역이 불투명 검정이 되어 아래 그림을 덮는다 |

### 왜 B-D2인가 (가장 중요한 결정)

1a와 1b에서 **가장 비싼 버그는 모두 "두 권위가 어긋날 수 있는 이중 저장"**이었다:
`Line`의 `start`/`end` 대 폴리곤, 그리고 도형 그라디언트의 로컬 좌표 대 브러시 변환.

스트로크 그라디언트에 `GradientSpec`을 재사용하면 같은 위험이 생긴다. `GradientSpec`은
"이 프레임 안 어디에 어떤 색이 있는가"를 정규화 좌표로 표현하는데, 스트로크에는
그 좌표계가 없다 — 스트로크는 페이지를 가로지르는 선이고, 그 위의 매개변수는
"선을 따라 몇 %" 또는 "페이지의 어느 축"이다. 정규화 프레임 좌표를 억지로
쓰면 그 필드가 무엇을 뜻하는지 아무도 말할 수 없게 된다.

**두 색과 매개변수화 규칙**이면 충분하다:

```rust
pub enum BrushRamp {
    /// 스트로크 시작이 `from`, 끝이 `to`. 획마다 리셋된다.
    AlongStroke { from: [u8; 4], to: [u8; 4] },
    /// 페이지의 고정 축을 따라 색이 정해진다. 여러 획이 하나의 램프를 잇는다.
    FixedAxis {
        from: [u8; 4],
        to: [u8; 4],
        axis: RampAxis,
    },
}

pub enum RampAxis {
    /// 페이지 왼쪽(0.0)에서 오른쪽(1.0).
    Horizontal,
    /// 페이지 위(0.0)에서 아래(1.0).
    Vertical,
}
```

`FixedAxis`는 **페이지 정규화 좌표**를 쓴다 — 프레임이 아니라 페이지다. 이것이
"여러 획이 하나의 램프를 잇는다"를 만드는 유일한 방법이다.

## 커맨드

### B3 — 색보정

```rust
pub async fn adjust_raster(
    expected_revision: Revision,
    layer: EntityId,
    adjust: ColorAdjust,
) -> Result<LayerCommit, Error>
```

`ColorAdjust`는 열거형이다(한 번에 하나):

```rust
pub enum ColorAdjust {
    Brightness { delta: i32 },       // -255..=255
    Contrast { percent: f32 },       // -100.0..=100.0 (image 크레이트 규약)
    Levels { low: u8, high: u8 },    // low < high
    Hue { degrees: i32 },            // -180..=180
    Grayscale,
}
```

**`Contrast`의 단위에 주의한다.** `image::imageops::contrast_in_place(image, contrast: f32)`의
인자는 **백분율**이다 — 내부가 `((100.0 + contrast) / 100.0).powi(2)`
(`image-0.25.10/src/imageops/colorops.rs:132`)이므로 `0.0`은 무변화,
`100.0`은 대비를 제곱배로 키운다. `-1.0..=1.0`이 아니다. 필드 이름을
`percent`로 두어 단위를 드러낸다.

거부 규칙: `layer`가 래스터가 아님, 페이지 크기와 자산 크기가 다름,
`Levels`에서 `low >= high`, `delta`/`degrees`가 범위 밖, `percent`가
비유한이거나 `-100.0..=100.0` 밖.

### B2 — 채우기

```rust
pub async fn fill_raster(
    expected_revision: Revision,
    layer: EntityId,
    point: Point,
    fill: FillSpec,
) -> Result<LayerCommit, Error>

pub enum FillSpec {
    Solid { color: [u8; 4] },
    Gradient { gradient: GradientSpec, axis: GradientAxis },
}
```

`Solid`는 `flood_fill_mut`을 그대로 쓴다 — **허용 오차가 없으므로** 시작 픽셀과
정확히 같은 픽셀만 채운다. `Gradient`는 **`point`에서 시작하는 flood fill로
마스크를 만들고**, 그 마스크 안에서만 그라디언트 색을 칠한다 — 즉 "영역을
고르는 것"과 "그 영역을 어떻게 칠하는가"를 분리한다.

마스크를 만드는 방법: 원본을 복제해 시작 픽셀과 **확실히 다른** 표식 색으로
`flood_fill_mut`하고, **원본과 달라진 픽셀**을 마스크로 정의한다. 표식 색은
시작 픽셀의 각 채널을 `wrapping_add(1)`한 값이면 시작 픽셀과 반드시 다르다
(`flood_fill_mut`은 `target`과 **정확히 같은** 픽셀만 채우므로, 채워진 픽셀은
정의상 전부 `target`이고 표식과 겹치지 않는다). "원본에 존재하지 않는 색을
고른다"는 보장은 필요 없다 — 마스크는 색이 아니라 **변화**로 정의되기 때문이다.

`Point`는 페이지 픽셀 좌표이고, 채우기 시작점이 페이지 밖이면 거부한다.

**같은 색으로 채울 때는 `flood_fill_mut`을 부르면 안 된다.** 함수의 루프는
`inside`가 거짓이 될 때 끝나는데, `inside`는 "그 픽셀이 아직 시작 픽셀과 같은가"를
묻는다(`imageproc-0.27.0/src/drawing/fill.rs:70`). `fill_with`가 `target`과 같으면
`put_pixel`이 같은 값을 다시 쓰므로 `inside`가 **영원히 참**이고, 각 span이 자기
행을 다시 덮으며 위·아래 행에 새 span을 무한히 쌓는다 — 반환하지 않는다.

**한 행짜리 이미지로는 재현되지 않는다.** 위·아래 행이 범위 밖이라 아무것도
push되지 않기 때문이다. 실제로 이 함정은 4×1 이미지로 쓴 테스트를 통과했고,
`H >= 2` 회귀 테스트를 추가한 뒤에야 드러났다. 그러므로 시작 픽셀이 이미 목표
색이면 **호출을 건너뛰고 커밋한다** — 결과가 같으므로 의미상 동일하다.

### B1 — 브러시 그라디언트

`commit_paint`의 `PaintBrush`가 넓어진다:

```rust
pub struct PaintBrush {
    pub diameter: f32,
    pub color: [u8; 4],              // 램프가 없을 때 쓰는 단색
    pub ramp: Option<BrushRamp>,     // 있으면 color 대신 램프가 색을 정한다
}
```

`rasterize_stroke`가 램프를 받아 픽셀마다 색을 계산한다. `AlongStroke`는
스트로크의 누적 길이 비율, `FixedAxis`는 페이지 정규화 좌표다.

`StrokeCommitDto::Paint`도 `ramp: Option<BrushRamp>`를 싣는다. WASM 미리보기는
`draw_segment`가 세그먼트마다 색을 정하도록 바뀐다 — 진행 방향 램프는 누적
길이를, 고정축 램프는 그 점의 좌표를 쓴다.

## 프론트

| 요소 | 변경 |
|---|---|
| `CanvasBrush` (`store.ts:45`) | `ramp: BrushRamp \| null` 추가 |
| `ToolBar` | 브러시 팝오버에 램프 종류 선택 + 두 색 |
| 새 도구 | `fill`(버킷) 도구. `CanvasTool`에 추가, 단축키 `g` |
| `Inspector` | 래스터 레이어 선택 시 색보정 섹션 |
| i18n | 9개 로케일 |

## 테스트

| 계층 | 테스트 |
|---|---|
| `koharu-app` | 색보정 각 변형의 픽셀 결과 (밝기 증가, 대비, 레벨, 색조, 흑백) |
| `koharu-app` | **모든 색보정 변형이 알파를 보존하는지** — 특히 흑백이 투명 픽셀을 불투명하게 만들지 않는지 (B-D8) |
| `koharu-app` | 색보정 거부 — 비래스터 대상, 페이지 밖, 잘못된 범위 |
| `koharu-app` | 버킷 채우기가 연결 영역만 칠하고 경계를 넘지 않는지 (오차 없음이므로 정확히 같은 색만) |
| `koharu-app` | 그라디언트 채우기가 마스크 안에서만 칠하는지 |
| `koharu-app` | `AlongStroke` 램프가 누적 길이를 따라 변하는지 (0%·50%·100%) |
| `koharu-app` | `FixedAxis` 램프가 여러 획에 걸쳐 이어지는지 |
| `koharu-app` | **투명 배경 유지** — 래스터 편집이 알파 0 영역을 불투명하게 만들지 않는지 |
| 프론트 | 버킷 도구 배선, 램프 UI, 좌표 변환 |

## 실패 시나리오

| 상황 | 처리 |
|---|---|
| 채우기 시작점이 이미 채울 색 | **`flood_fill_mut`을 호출하지 않고** 그대로 커밋한다(무변경). 호출하면 **무한 루프**다 — 아래 참조 |
| 채우기 시작점이 페이지 밖 | 거부 |
| 그라디언트 채우기 마스크가 비어 있음 | 채울 픽셀이 하나도 없으므로 거부하거나 무변경 커밋. **정하라**: 무변경 커밋이 낫다(사용자 의도가 "이 영역"이므로) |
| 색보정 `Levels`에서 `low == high` | 거부 (0으로 나눔) |
| 램프의 `from == to` | 허용. 단색과 같은 결과다 |
| 램프가 `Some`인데 `points`가 1개 | `AlongStroke`는 길이 0이므로 `from` 색. 거부하지 않는다 |
| 자산이 페이지 크기와 다름 | 거부 (기존 `apply_raster_stroke` 관례) |

## 기존 프로젝트 호환

**영향 없음.** B는 새 커맨드·새 도구·새 옵션 필드만 추가하고 기존 저장 데이터의
의미를 바꾸지 않는다. `PaintBrush`/`StrokeCommitDto`의 `ramp`는
`Option`이므로 기존 클라이언트·기존 요청이 그대로 유효하다 — 이는 1a·1b의
스키마 변경과 달리 **wire 하위 호환이 유지되는 드문 경우**이며, 이유는
`Option` 필드 추가가 serde에서 누락을 허용하기 때문이다.

## 열린 문제

없음. B1·B2·B3의 상세 설계는 각자의 스펙에서 정한다. 1c(패턴)와 C~E는 범위 밖이다.
