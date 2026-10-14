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
| P-D5 | 타일링은 **그래픽 라이브러리에 맡긴다** — peniko `ImageSampler`의 `Extend::Repeat` | 초판은 "`image::imageops::tile`은 정수배만 반복하므로 직접 구현한다"고 했다. **전제가 틀렸다.** 우리가 쓰는 것은 `image`의 타일 함수가 아니라 **vello의 이미지 브러시**이고, 그것의 `Extend::Repeat`은 임의 오프셋·스케일·회전에서 연속적으로 반복한다(`peniko-0.6.1/src/image.rs:95-104`, `brush.rs:159-168`). 직접 격자를 계산하면 `scale`·`origin`·회전을 우리가 소유하게 되고, 그 계산이 미리보기와 어긋날 자리가 생긴다 |
| P-D8 | 패턴의 **스케일·원점·회전은 `brush_transform`이 나른다** (`Stretch` 포함, `None`이 아니다) | `ImageBrush`의 행렬은 `transform_inverse(composed)`이고(`vello_shaders-0.10.0/shader/draw_leaf.wgsl:250`), 셰이더는 그 역행렬로 얻은 좌표를 **이미지 픽셀 단위**로 취급한다(`fine.wgsl:1326`, `extend_mode(..., max = image.extents)`). 따라서 `composed`는 **이미지 픽셀 → 페이지**를 보내는 사상이고, `brush_transform: None`이면 1 이미지 픽셀 = 1 경로 로컬 단위가 되어 **`scale`이 아무 효과를 갖지 못한다.** 브러시 변환을 주는 것은 안전하다 — 1b의 함정은 *경로* 아핀을 준 것이었고, 브러시 변환은 vello가 정확히 한 번 합성한다(`scene.rs:328-331`) |
| P-D6 | 자산이 없거나 디코드에 실패하면 **`Solid`로 폴백하고 채우기는 성공한다** | 실패를 오류로 만들면 패턴 자산 하나가 깨졌을 때 페이지 전체가 열리지 않는다. 1b에서 그라디언트 스톱이 비정상일 때의 처리와 같은 원칙이다. 폴백 색은 `PatternSpec::fallback`에 저장한다 |
| P-D7 | 패턴 회전은 **도(degree) 정수**로 저장한다 | 1b의 `ShapeLayer.angle_degrees`와 단위를 맞춘다. 부동소수를 저장하면 직렬화 왕복에서 값이 흔들린다 |
| P-D9 | **`PreparedPattern.rotation_degrees`는 `f32`다** (저장되는 `PatternSpec`의 `i32`와 다르다) | `PreparedPattern`은 저장되지 않는 **일시적 커맨드 인자**다. 브라우저 미리보기는 경로 변환이 항등이므로 브러시가 **도형 각도 + 패턴 각도**를 함께 나눠야 하는데(`P = R(θ_frame) + 이동`이므로 `composed = T(P(origin))·R(θ_frame+θ_pattern)·S`), `θ_frame`이 `f32`라 합을 정수로 표현할 수 없다. 반올림하면 미리보기가 커밋된 도형에서 최대 0.5° 어긋난다 — 1b의 Critical과 이 서브프로젝트의 단일 권위 규칙이 막으려는 바로 그 실패다. **저장 스펙은 `i32`를 유지**한다(P-D7의 직렬화 안정성). `fill()`은 `spec.rotation_degrees as f32`로 정확히 변환하고, 미리보기는 합을 넘긴다. 회전은 주기적이므로 커맨드 인자의 회전은 **유한성만** 검증한다(±360 제한 없음) |

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

**패턴은 그라디언트와 똑같이 브러시다.** vello 0.10은 이미지 브러시를 지원한다 —
`Scene::fill`이 `impl Into<StyleRef>`를 받고 `ImageBrush`가 그중 하나이며
(`vello-0.10.0/src/scene.rs:316`, `:442-449`), peniko의 `ImageSampler`가
`Extend::Repeat`을 지원한다(`peniko-0.6.1/src/image.rs:95-104`,
`brush.rs:159-168`). 즉 **타일링은 그래픽 라이브러리가 한다.**

1b의 경로를 그대로 따른다:

1. `prepared()`가 `PatternSpec`을 `PreparedFill::Pattern(PreparedPattern)`으로 펴고,
   **정규화 → 프레임 로컬 변환을 여기서 한다** (1b의 G3와 같은 단일 변환 지점).
   단, 결과는 **경로 로컬**이다 — 그라디언트처럼 `frame.x`/`frame.y`를 **더하지 않는다.**
   경로 로컬 원점이 곧 프레임 박스의 좌상단이므로 `frame.x`를 더하면 두 번 offset된다
   (`shape_renderer.rs:311`의 그라디언트가 `frame.x`를 쓰지 않는 것과 같은 이유).
2. `append_to`가 `ImageBrush`를 만들어 `Scene::fill`에 넘긴다.
3. `Extend::Repeat`(타일) 또는 `Extend::Pad`(늘리기)를 `ImageSampler`에 설정한다.

### P-D8: 브러시 변환이 스케일·원점·회전을 나른다

**계획 초판은 "`brush_transform`은 `None`"이라고 썼고, 그것이 틀렸다.** 셰이더와
인코딩 경로를 읽어 확인한 사실은 다음과 같다:

- `draw_leaf.wgsl:250`은 `DRAWTAG_FILL_IMAGE`에 대해 `inv = transform_inverse(transform)`을
  저장한다. 여기서 `transform`은 `Scene::fill`이 합성한 값, 즉 `composed`다.
- `fine.wgsl:1344-1346`은 그 행렬로 `atlas_uv = image.matrx * my_xy + image.xlat`을 구하고,
  곧바로 `extend_mode(atlas_uv.x, image.x_extend_mode, image.extents.x)`로 **이미지 픽셀
  단위**로 반복시킨다(`fine.wgsl:881-893`).
- 즉 `image.matrx = inverse(composed)`이고, 따라서 **`composed`는 이미지 픽셀 → 페이지**를
  보낸다. `Scene::fill`은 `composed = transform * brush_transform`으로 합성한다
  (`scene.rs:328-331`).

`brush_transform: None`이면 `composed = P`(경로 아핀)이므로 **1 이미지 픽셀 = 1 경로 로컬
단위**가 되고, `scale`이 타일 간격에 **아무 영향을 주지 못한다.** 따라서 패턴의
스케일·원점·회전은 **브러시 변환**으로 넣어야 한다.

**1b의 함정과 혼동하지 말 것.** 1b의 Critical은 *경로* 아핀을 `brush_transform`으로 준
것이었다. 브러시 변환을 주는 것 자체는 안전하고 오히려 필수다 — vello는 브러시 변환을
정확히 한 번만 합성한다.

**경로 로컬 기준 변환**(프레임 박스가 `(0,0)`–`(w,h)`, `pitch = scale * w`):

| 반복 | `brush_transform` |
|---|---|
| `Tile` | `translate(origin) * rotate(θ) * scale(pitch)` |
| `Stretch` | `translate(c) * rotate(θ) * translate(-c) * scale_non_uniform(w/iw, h/ih)`, `c` = 박스 중심 |

`Tile`에서 이미지 픽셀 `(0,0)`이 로컬 `origin`에 놓이고, 픽셀 하나가 `pitch` 로컬 단위를
차지한다 — 스펙의 "이미지 한 픽셀이 차지하는 프레임 로컬 크기" 정의와 일치한다.
`Stretch`는 비균일 스케일로 이미지 한 장을 박스에 꽉 채운다(종횡비를 보존하지 않는다 —
스펙의 "박스에 맞춰 늘린다" 그대로).

**초판은 이 부분을 틀리게 썼다.** 초판은 "vello 브러시로는 이미지 타일을 표현할 수
없으므로 클립 + 반복 draw로 그린다"고 했고, 그래서 `fill()`이 `Pattern`에 `None`을
돌려주고 격자를 손으로 나누는 설계를 넣었다. **그 전제가 거짓이다** — vello는
이미지 브러시와 `Repeat` 확장을 모두 제공하므로 클립도 격자 계산도 필요 없다.
손으로 타일링하면 `scale`·`origin`·회전을 전부 우리가 계산해야 하고 그 계산이
미리보기와 어긋날 자리가 생긴다(1b에서 실제로 났던 버그 유형). 라이브러리에
맡기면 그 위험이 사라진다.

**남는 제약**: `PreparedPattern`은 디코드된 픽셀이나 그 `ResourceId`를 렌더러에
넘겨야 한다. 이미 있는 `RasterImage`(`frame.rs:234-240`)가 타일·픽셀·`ResourceId`를
이미 나르고 `compositor.rs:520-541`이 그것을 GPU로 올리므로, 패턴도 그 경로를
쓴다 — 새 업로드 경로를 만들지 않는다.

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
| `CanvasPathFill` (`canvas.ts`) | `pattern` 변형 추가 — 그라디언트와 같은 자리다 |
| `lib/shapes.ts` | `defaultPatternSpec`, `patternOrigin`(정규화→페이지, **1b와 같은 회전 규칙**) |
| `Inspector` | 채우기 종류에 "패턴" 추가, 반복 방식 선택, 원점·스케일·회전 입력, 이미지 업로드 |
| `i18n` | 9개 로케일 |

**미리보기도 그라디언트와 같은 경로를 탄다.** 패턴이 브러시이므로 `CanvasPathFill`의
`fill`에 `Gradient`와 나란히 `Pattern` 변형을 넣고, WASM이 `ImageBrush`를 만들어
`Scene::fill`에 넘긴다. **초판은 이 자리를 비워 두고 "(b) 자리표시자로 시작한다"고
했는데, 그 근거(클립+반복 draw)가 사라졌으므로 미리보기도 정상적으로 구현한다.**

프론트가 정해야 하는 것은 **정규화 → 페이지 변환과 회전**뿐이고, 그 규칙은 1b가
이미 확립했다(`shapes.ts`의 `previewGradient`가 `frame.origin + normalized * extent`를
구한 뒤 `rotatePoint`로 회전한다). `patternOrigin`이 같은 규칙을 따르고, **1b처럼
정확한 좌표를 단언하는 테스트로 못 박는다** — 1b에서 미리보기와 커밋이 어긋난
버그는 그 테스트가 없어서 생겼다.

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
3. **~~타일 개수 상한~~** — **해결됨.** 초판은 "수십만 개의 draw"를 걱정했지만,
   그 우려는 손으로 격자를 그리는 설계에서 나온 것이었다. `Extend::Repeat`은
   셰이더가 샘플링하므로 **타일 수가 draw 수에 영향을 주지 않는다.** 상한이 없다.
