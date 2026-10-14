# 패턴 채우기 설계 (1c)

작성일: 2026-10-14

## 배경

1b는 `ShapeFill::Solid` 옆에 `ShapeFill::Gradient(GradientSpec)`를 추가했고, 그 과정에서
`PreparedFill::{Solid, Gradient}` · `PreparedGradient*` · `fill()`의 정규화→로컬 변환 ·
wire 계약 테스트라는 재사용 가능한 골격을 만들었다. 1c는 **같은 자리에 세 번째
변형**을 넣는다.

브레인스토밍 시점에 1c를 1b에서 분리한 이유는 세 가지가 미정이었기 때문이다:
**자원 종류**(패턴 이미지를 무엇으로 저장하는가), **수명 규칙**(자원이 사라지면
채우기는 어떻게 되는가), **이미지 원점**(타일을 어디서부터 반복하는가). 이 문서가
그 셋을 정한다.

## 범위

도형 채우기에 **이미지 패턴**을 추가한다. 텍스처·스크린톤·망점을 도형 안에
반복해 넣는 것이 목적이다.

**포함**: 반복 방식(타일/늘리기), 원점, 스케일, 회전, 자원 수명 규칙.
**제외**: 만화 스크린톤 자동 생성(E), 붓 패턴(브러시는 B1이 다루며 패턴은 범위 밖),
패턴 자체를 편집하는 도구.

## 결정 사항

| # | 결정 | 근거 |
|---|---|---|
| P-D1 | 패턴 이미지는 **`AssetRole::new("pattern")`** 으로 레이어에 붙인다 | `apply_raster_stroke`가 `AssetRole::new("source")`를 쓰는 것과 같은 메커니즘이다(`project.rs`). 새 자원 종류를 만들면 저장소·PSD·동기화가 각각 그 종류를 알아야 한다. 역할 문자열은 이미 임의이고, 같은 레이어에 여러 역할이 공존할 수 있다 — 래스터 레이어는 "source"(픽셀)와 "pattern"(패턴)을 동시에 가질 수 있다 |
| P-D2 | `PatternSpec`은 **레이어의 패턴 자산을 참조하지 않고 스스로는 아무것도 참조하지 않는다** — 패턴은 **도형 레이어가 소유**한다 | 1a·1b에서 가장 비쌌던 버그는 두 권위의 불일치였다. `PatternSpec`이 레이어 ID를 참조하면 "그 레이어가 지워졌는데 `PatternSpec`이 남는" 상태가 생긴다. 대신 **패턴을 쓰는 도형 레이어 자신이 `AssetRole::new("pattern")` 자산을 갖고**, `PatternSpec`은 그 자산을 어떻게 반복할지만 정한다 |
| P-D3 | `PatternSpec`은 **정규화 프레임 로컬 좌표**로 원점을 갖는다 (1b의 G3와 동일) | 1b가 그렇게 정했고, 다른 좌표계를 쓰면 `fill()`이 두 규칙을 알아야 한다 |
| P-D4 | 반복 방식은 **`Tile`과 `Stretch` 둘** | `Tile`은 텍스처, `Stretch`는 "이미지 하나를 도형에 맞춰 늘리기"다. 둘은 서로 대체하지 못하고, 둘 다 `image` 크레이트의 기존 함수로 구현된다 |
| P-D5 | 타일링은 **`image::imageops::tile`이 아니라 직접 구현**한다 | `tile`은 이미지를 정수배로만 반복해 도형 경계에서 잘린다. 우리는 도형의 경계 안에서 임의 위치·스케일로 반복해야 하므로, 페인트 순서를 우리가 소유해야 한다 |
| P-D6 | 자산이 없거나 디코드에 실패하면 **`Solid`로 폴백하고 채우기는 성공한다** | 실패를 오류로 만들면 패턴 자산 하나가 깨졌을 때 페이지 전체가 열리지 않는다. 1b에서 그라디언트 스톱이 비정상일 때의 처리와 같은 원칙이다. 폴백 색은 `PatternSpec::fallback`에 저장한다 |
| P-D7 | 패턴 회전은 **도(degree) 정수**로 저장한다 | 1b의 `ShapeLayer.angle_degrees`와 단위를 맞춘다. 부동소수를 저장하면 직렬화 왕복에서 값이 흔들린다 |

### P-D2가 이 서브프로젝트의 핵심 결정이다

1a의 Critical은 `Line`의 `start`/`end`와 폴리곤이 **두 권위**였던 것이고, 1b의 Critical은
경로 변환과 `brush_transform`이 **두 권위**였던 것이다. 둘 다 "같은 사실을 두 곳이
기억한다"에서 나왔다.

패턴에 레이어 참조를 넣으면 세 번째가 된다: `PatternSpec.layer`와 그 레이어의 실제
존재. 레이어가 삭제되면 참조가 매달리거나, 삭제를 막거나, 참조를 지우는 셋 중 하나를
골라야 하고 **셋 다 새 불변식**을 만든다.

**패턴을 도형 레이어가 소유하면 그 문제가 사라진다.** 패턴 자산은 그 레이어와 함께
살고 함께 죽는다 — 1a가 래스터 자산에 대해 이미 그렇게 한다. `PatternSpec`은
순수한 "어떻게 반복할까"이고, 참조를 하나도 갖지 않으므로 매달릴 수 없다.

### 저장 모델

```rust
pub enum ShapeFill {
    Solid([u8; 4]),
    Gradient(GradientSpec),
    Pattern(PatternSpec),        // 새 변형
}

pub struct PatternSpec {
    /// 반복 방식.
    pub repeat: PatternRepeat,
    /// 정규화 프레임 로컬 좌표(0..1)의 타일 원점. `Tile`에서만 의미를 갖는다.
    pub origin: [f32; 2],
    /// 이미지 한 픽셀이 차지하는 프레임 로컬 크기(정규화). 1.0이면 프레임 폭만큼.
    pub scale: f32,
    /// 시계 방향 도. 0이면 축정렬.
    pub rotation_degrees: i32,
    /// 자산이 없거나 깨졌을 때 쓰는 색.
    pub fallback: [u8; 4],
}

pub enum PatternRepeat {
    /// `origin`에서 시작해 `scale` 간격으로 반복한다.
    Tile,
    /// 이미지 하나를 도형의 바운딩 박스에 맞춰 늘린다. `origin`은 무시한다.
    Stretch,
}
```

**`origin`이 `Stretch`에서 무시된다는 사실을 문서화한다.** 두 변형이 같은 구조체를
공유하므로 "이 필드는 지금 의미가 없다"를 숨기면 나중에 누군가 그 필드를 읽는
코드를 쓴다. `PatternRepeat::Stretch`의 doc 주석에 명시한다.

**`validate`가 거부하는 것**:
- `scale`이 비유한이거나 0 이하
- `origin`의 두 성분 중 비유한이 있는 것
- `rotation_degrees`가 `-360..=360` 밖 (범위를 넓게 두되 무한을 막는다)

**거부하지 않는 것**: `origin`이 `0..1` 밖. 타일 원점은 도형 밖에서 시작해도 되며,
오히려 "패턴을 옮기기"가 그렇게 동작한다.

### 렌더링

1b의 경로를 그대로 따른다:

1. `prepared()`가 `PatternSpec`을 `PreparedFill::Pattern(PreparedPattern)`으로 펴고,
   **정규화 → 프레임 로컬 변환을 여기서 한다** (1b의 G3와 같은 단일 변환 지점).
2. `compile_scene`이 `PreparedPattern`을 만나면 **패턴 자산을 디코드해 캔버스에
   직접 그린다** — vello의 브러시로는 이미지 타일을 표현할 수 없다.
3. 그리기는 **도형의 클립 안에서** 일어난다: 도형 경로를 클립으로 설정하고, 그 안에
   변환된 타일을 반복해 그린다.

**이것이 1b와 다른 점이다.** 1b는 peniko 브러시 하나로 끝났지만, 패턴은 vello의
`Scene::fill`이 아니라 **클립 + 반복 draw**로 표현된다. 그러므로:
- `PreparedFill`에 `Pattern` 변형을 넣되, `shape_renderer::fill()`은 `Pattern`을
  **`None`으로 돌려주고** 채우기를 별도 단계로 넘긴다. `fill()`의 반환 타입이
  `Option<PreparedFill>`이므로 이 표현이 자연스럽다 — "vello 브러시로 표현할 수
  없는 채우기"가 `None`이 아니라 **별도 플래그**가 되어야 한다면 그때 타입을 넓힌다.
- 클립은 중첩될 수 있고 vello는 클립 스택을 지원한다. 도형 하나당 클립 하나이므로
  중첩 깊이는 1이다.

**`tile`을 직접 구현하는 이유(P-D5)**: 도형의 바운딩 박스를 프레임 로컬에서 구해,
그 박스를 `scale` 간격의 격자로 나누고, 각 격자 칸에 `origin` 기준으로 오프셋한
이미지를 그린다. 정수배 반복이 아니므로 임의 오프셋과 회전이 가능하다.

### 자산 수명 규칙 (P-D6)

| 상황 | 동작 |
|---|---|
| 도형 레이어에 `pattern` 자산이 없음 | `fallback` 색으로 `Solid` 채우기와 동일하게 그린다. **오류가 아니다** |
| 자산이 있으나 디코드 실패 | 위와 같다. 로그를 남긴다 |
| 자산이 있으나 크기가 0 | 위와 같다 |
| 자산이 정상 | 패턴을 그린다 |
| 도형 레이어가 삭제됨 | 자산이 함께 사라진다. 매달린 참조가 없다(P-D2) |
| 페이지 병합 | 1a가 `merge_pages`에서 자산을 옮기는 경로를 이미 갖는다. 패턴 자산도 같은 경로를 탄다 — **`merge_pages`가 `pattern` 역할을 빠뜨리지 않는지 테스트한다** |

## 커맨드

```rust
pub async fn set_shape_pattern(
    expected_revision: Revision,
    layer: EntityId,
    pattern: Option<PatternSpec>,
) -> Result<LayerCommit, Error>

pub async fn set_shape_pattern_asset(
    expected_revision: Revision,
    layer: EntityId,
    bytes: Vec<u8>,
    media_type: String,
) -> Result<LayerCommit, Error>
```

`set_shape_pattern`은 `PatternSpec`을 설정하고(`None`이면 `Solid`로 되돌린다),
`set_shape_pattern_asset`은 `AssetRole::new("pattern")` 자산을 쓴다. **둘을 분리하는
이유**: 자산 업로드는 크기가 크고 실패할 수 있으며, `PatternSpec` 설정은 즉시
일어난다. 하나의 커맨드로 묶으면 큰 업로드가 파라미터 변경까지 막는다.

`set_shape_pattern_asset`은 **도형 레이어에만** 허용한다(P-D2). 래스터 레이어의
`pattern` 역할은 거부한다 — 그 레이어는 패턴을 소유하지 않는다.

## 프론트

| 요소 | 변경 |
|---|---|
| `ShapeFill` (`protocol.ts`) | `pattern` 변형 추가 |
| `CanvasPathFill` (`canvas.ts`) | 패턴은 브러시가 아니므로 **넣지 않는다**. 미리보기는 별도 경로로 그린다 |
| `lib/shapes.ts` | `defaultPatternSpec`, `patternOrigin`(정규화→페이지, **1b와 같은 회전 규칙**) |
| `Inspector` | 채우기 종류에 "패턴" 추가, 반복 방식 선택, 원점·스케일·회전 입력, 이미지 업로드 |
| `i18n` | 9개 로케일 |

**미리보기가 1b보다 복잡하다.** 1b는 그라디언트를 `CanvasPathFill`에 실어 WASM이
그렸지만, 패턴은 WASM이 클립+반복 draw를 해야 한다. 두 선택지:

- **(a)** WASM에 패턴 그리기 경로를 추가한다 — 미리보기가 커밋 결과와 정확히 같아진다.
- **(b)** 미리보기는 자리표시자(회색 사선)로 두고 커밋 후 결과를 본다.

**1c는 (b)로 시작한다.** (a)는 새 WASM 페인트 경로이고, 1b에서 미리보기와 커밋이
어긋나는 버그를 이미 한 번 겪었다(프론트 회전 규칙). 서로 다른 두 렌더러가 같은
패턴을 그리게 하는 것은 그 버그를 다시 만드는 일이다. **미리보기가 틀릴 수 있으면
미리보기를 하지 않는 편이 낫다** — 사용자는 커밋 결과를 보고 그것이 권위다.

## 테스트

| 계층 | 테스트 |
|---|---|
| `koharu-scene` | `PatternSpec::validate` — scale 0·음수·비유한, origin 비유한, rotation 범위 |
| `koharu-scene` | `origin`이 `0..1` 밖이어도 검증을 통과하는지 (거부하지 않기로 했으므로) |
| `koharu-scene` | `ShapeFill::Pattern` 라운드트립 |
| `koharu-rasterizer` | 정규화→프레임 로컬 변환이 패턴 원점에도 적용되는지 |
| `koharu-rasterizer` | 타일 격자가 도형 바운딩 박스를 정확히 덮는지 |
| `koharu-rasterizer` | 자산이 없을 때 `fallback` 색으로 그려지는지 (오류 아님) |
| `koharu-rasterizer` | 디코드 실패 자산이 오류가 아니라 폴백인지 |
| `koharu-app` | wire 계약 — `{"pattern":{…}}` 리터럴 고정 (1b Task 4의 방식) |
| `koharu-app` | `set_shape_pattern`이 `None`으로 `Solid`로 되돌리는지 |
| `koharu-app` | `set_shape_pattern_asset`이 래스터 레이어를 거부하는지 |
| `koharu-app` | **`merge_pages`가 `pattern` 역할 자산을 옮기는지** (1a의 Critical 재발 방지) |
| 프론트 | `defaultPatternSpec`이 검증을 통과하는지, 변환 규칙 |

## 열린 문제

1. **패턴 자산의 픽셀 밀도.** `scale`을 "이미지 한 픽셀 = 프레임 로컬 얼마"로
   정의했으므로 페이지 해상도가 바뀌면 패턴의 물리적 크기가 바뀐다. 이는 의도다
   (도형이 페이지와 함께 스케일된다). 다만 PSD 내보내기가 래스터화할 때 같은
   규칙을 쓰는지 확인해야 한다 — **계획 단계에서 확인한다.**
2. **`Stretch`에서 `origin` 무시** — 필드를 남길지 별도 구조체로 나눌지. 지금은
   한 구조체 + 문서화로 정한다. 계획에서 불편하면 재검토한다.
3. 타일 개수 상한이 없다. 도형이 크고 `scale`이 작으면 수십만 개의 draw가 될 수
   있다. **계획에서 상한을 정하고 초과 시 동작을 정한다** (폴백 또는 클램프).
