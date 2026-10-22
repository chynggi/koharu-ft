# 말풍선 배경 채우기 판정 강건화 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 외곽선 이상치가 섞인 불투명 말풍선과 축 방향 그라데이션 말풍선을 인페인팅 모델 없이 채운다.

**Architecture:** 인페인팅 단계의 사전 채우기(`fill_uniform_regions`)가 쓰는 판정을 표준편차에서 inlier 비율로 바꾸고, 판정 결과를 단색 하나가 아니라 배경 모델(`Flat` / `Rows` / `Columns`)로 돌려준다. 받아들인 모델은 text-mask 픽셀마다 배경색을 예측해 채우고, 그 픽셀을 모델 마스크에서 뺀다. 나머지 타일 계획과 모델 호출은 그대로다.

**Tech Stack:** Rust, `image` 크레이트(`RgbImage`, `GrayImage`, `Rgb`, `Luma`)

**Spec:** `specs/2026-10-22-bubble-background-fill-design.md`

---

## 파일 구조

수정 파일은 하나다.

- Modify: `crates/koharu-pipeline/src/stages/inpainting.rs`
  - 상수(현재 1097-1100행 부근): inlier 상수 2개, 프로파일 반경 상수 1개 추가
  - BallonsTranslator 주석(현재 1115-1120행): 의도적 차이 한 줄 추가
  - `fill_uniform_regions`(현재 1204-1234행): 배경 모델로 픽셀마다 색 결정
  - `uniform_region_color`(현재 1387-1427행) → `region_background`로 교체, `RegionBackground`, `inlier_ratio`, `median_color`, `axis_profile`, `lerp_color` 추가
  - `standard_deviation`(현재 1480-1487행): 삭제
  - 테스트 모듈(`mod tests`, `rectangle_region` 헬퍼 1860행 부근 뒤): 테스트 4개 추가

검증 명령은 전부 debug 프로필, 인페인팅 테스트만이다.

```bash
cargo test -p koharu-pipeline --lib inpainting::tests
```

처음 실행은 `koharu-pipeline` 의존성 빌드 때문에 몇 분 걸린다. 백그라운드로 돌린다.

---

### Task 1: 단색 판정을 inlier 비율로 바꾸기

**Files:**
- Modify: `crates/koharu-pipeline/src/stages/inpainting.rs`
- Test: 같은 파일의 `mod tests`

- [ ] **Step 1: 실패하는 회귀 테스트 작성**

`mod tests` 안, `rectangle_region` 헬퍼 바로 뒤(`uniform_mask_background_is_filled_without_inference` 앞)에 추가한다.

```rust
    #[test]
    fn outline_ink_inside_an_opaque_bubble_does_not_block_the_flat_fill() {
        let mut image = RgbImage::from_pixel(100, 100, Rgb([250, 250, 250]));
        // The polygon catches the bubble outline: two dark columns past the
        // edge margin, about 4% of the sampled background.
        for y in 14..86 {
            image.put_pixel(14, y, Rgb([0, 0, 0]));
            image.put_pixel(85, y, Rgb([0, 0, 0]));
        }
        let mut mask = GrayImage::new(100, 100);
        for y in 30..70 {
            for x in 30..70 {
                image.put_pixel(x, y, Rgb([10, 10, 10]));
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
        let mut output = image.clone();
        let mut pending_mask = mask.clone();

        fill_uniform_regions(
            &mut output,
            &mut pending_mask,
            &mask,
            &[rectangle_region([10, 10, 90, 90])],
        );

        assert_eq!(output.get_pixel(50, 50), &Rgb([250, 250, 250]));
        assert_eq!(pending_mask.get_pixel(50, 50), &Luma([0]));
    }
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `cargo test -p koharu-pipeline --lib inpainting::tests::outline_ink_inside_an_opaque_bubble_does_not_block_the_flat_fill`
Expected: FAIL. `output.get_pixel(50, 50)`이 `Rgb([10, 10, 10])`이다. 지금 판정은 표준편차가 약 49라서 채우지 않는다.

- [ ] **Step 3: 상수 추가**

`FLAT_FILL_EDGE_MARGIN` 바로 아래에 추가한다.

```rust
const UNIFORM_BACKGROUND_MIN_PIXELS: usize = 16;
const FLAT_FILL_EDGE_MARGIN: f32 = 3.0;
const BACKGROUND_INLIER_TOLERANCE: u8 = 12;
const BACKGROUND_INLIER_RATIO: f64 = 0.9;
```

- [ ] **Step 4: `uniform_region_color`를 `region_background`로 교체**

`fn uniform_region_color(...) { ... }` 전체(현재 1387-1427행)를 아래로 바꾼다.

```rust
/// The background a bubble's lettering sits on, predicted for any pixel inside
/// the bubble's bounds.
#[derive(Clone, Debug)]
enum RegionBackground {
    Flat(Rgb<u8>),
}

impl RegionBackground {
    fn color(&self, _bounds: [u32; 4], _x: u32, _y: u32) -> Rgb<u8> {
        match self {
            Self::Flat(color) => *color,
        }
    }
}

fn region_background(
    image: &RgbImage,
    mask: &GrayImage,
    region: &FlatFillRegion,
) -> Option<RegionBackground> {
    let [left, top, right, bottom] = region.bounds;
    let mut samples = Vec::new();
    for y in top..bottom {
        for x in left..right {
            let point = (x as f32 + 0.5, y as f32 + 0.5);
            if mask.get_pixel(x, y)[0] >= 127
                || !point_in_polygon(point, &region.polygon)
                || polygon_edge_distance_squared(point, &region.polygon)
                    < FLAT_FILL_EDGE_MARGIN * FLAT_FILL_EDGE_MARGIN
            {
                continue;
            }
            samples.push((x, y, *image.get_pixel(x, y)));
        }
    }
    if samples.len() < UNIFORM_BACKGROUND_MIN_PIXELS {
        return None;
    }

    let flat = RegionBackground::Flat(median_color(samples.iter().map(|&(_, _, color)| color)));
    (inlier_ratio(&flat, region.bounds, &samples) >= BACKGROUND_INLIER_RATIO).then_some(flat)
}

/// The share of samples the background predicts within the tolerance on every
/// channel. An outline pixel the polygon catches lowers it only by its own share.
fn inlier_ratio(
    background: &RegionBackground,
    bounds: [u32; 4],
    samples: &[(u32, u32, Rgb<u8>)],
) -> f64 {
    let inliers = samples
        .iter()
        .filter(|&&(x, y, color)| {
            let predicted = background.color(bounds, x, y);
            (0..3).all(|channel| {
                color[channel].abs_diff(predicted[channel]) <= BACKGROUND_INLIER_TOLERANCE
            })
        })
        .count();
    inliers as f64 / samples.len() as f64
}

fn median_color(colors: impl Iterator<Item = Rgb<u8>>) -> Rgb<u8> {
    let mut channels = [Vec::new(), Vec::new(), Vec::new()];
    for color in colors {
        for channel in 0..3 {
            channels[channel].push(color[channel]);
        }
    }
    Rgb(channels.each_mut().map(|values| median(values)))
}
```

- [ ] **Step 5: `standard_deviation` 삭제**

`fn standard_deviation(values: &[u8], center: f64) -> f64 { ... }` 전체(현재 1480-1487행)를 지운다. 다른 호출처는 없다.

- [ ] **Step 6: `fill_uniform_regions`가 배경 모델을 쓰게 바꾸기**

`fill_uniform_regions`의 끝부분을 바꾼다.

변경 전:

```rust
        let Some(color) = uniform_region_color(output, pending_mask, region) else {
            continue;
        };
        for (x, y) in targets {
            output.put_pixel(x, y, color);
            pending_mask.put_pixel(x, y, Luma([0]));
        }
```

변경 후:

```rust
        let Some(background) = region_background(output, pending_mask, region) else {
            continue;
        };
        for (x, y) in targets {
            output.put_pixel(x, y, background.color(region.bounds, x, y));
            pending_mask.put_pixel(x, y, Luma([0]));
        }
```

- [ ] **Step 7: 인페인팅 테스트 전체가 통과하는지 확인**

Run: `cargo test -p koharu-pipeline --lib inpainting::tests`
Expected: PASS. 새 테스트와 기존 `uniform_mask_background_is_filled_without_inference`, `uniform_bubbles_are_filled_independently_inside_one_textured_tile`, `textured_mask_regions_are_inferred_as_bounded_tiles`가 모두 통과한다. 경고가 없어야 한다.

- [ ] **Step 8: 커밋**

```bash
git add crates/koharu-pipeline/src/stages/inpainting.rs
git commit -m "fix(inpainting): accept flat bubbles by inlier share instead of deviation" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: 행·열 그라데이션 배경 모델 추가

**Files:**
- Modify: `crates/koharu-pipeline/src/stages/inpainting.rs`
- Test: 같은 파일의 `mod tests`

- [ ] **Step 1: 실패하는 그라데이션 테스트 작성**

Task 1에서 추가한 테스트 바로 뒤에 추가한다.

```rust
    fn vertical_gradient(width: u32, height: u32) -> RgbImage {
        RgbImage::from_fn(width, height, |_, y| {
            let value = (y * 255 / (height - 1)) as u8;
            Rgb([value, value, value])
        })
    }

    #[test]
    fn a_vertical_gradient_box_is_filled_from_its_row_profile() {
        let mut image = vertical_gradient(60, 200);
        let expected = *image.get_pixel(30, 100);
        let mut mask = GrayImage::new(60, 200);
        for y in 20..180 {
            for x in 15..45 {
                image.put_pixel(x, y, Rgb([255, 255, 255]));
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
        let mut output = image.clone();
        let mut pending_mask = mask.clone();

        fill_uniform_regions(
            &mut output,
            &mut pending_mask,
            &mask,
            &[rectangle_region([0, 0, 60, 200])],
        );

        let filled = output.get_pixel(30, 100);
        assert!(
            (0..3).all(|channel| filled[channel].abs_diff(expected[channel]) <= 2),
            "filled {filled:?}, expected about {expected:?}"
        );
        assert_eq!(pending_mask.get_pixel(30, 100), &Luma([0]));
    }
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `cargo test -p koharu-pipeline --lib inpainting::tests::a_vertical_gradient_box_is_filled_from_its_row_profile`
Expected: FAIL. `pending_mask.get_pixel(30, 100)`이 255로 남는다. Flat 판정의 inlier 비율이 10% 정도라 채우지 않는다.

- [ ] **Step 3: 기각 경계 테스트 작성**

이어서 추가한다. 이 두 테스트는 지금 코드에서도 통과한다. 새 프로파일 경로가 질감 배경과 외삽을 받아들이지 않는지 지키는 테스트다.

```rust
    #[test]
    fn a_screentone_bubble_is_left_for_the_model() {
        let mut image = RgbImage::from_fn(100, 100, |x, y| {
            if (x / 2 + y / 2).is_multiple_of(2) {
                Rgb([40, 40, 40])
            } else {
                Rgb([220, 220, 220])
            }
        });
        let mut mask = GrayImage::new(100, 100);
        for y in 30..70 {
            for x in 30..70 {
                image.put_pixel(x, y, Rgb([10, 10, 10]));
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
        let mut output = image.clone();
        let mut pending_mask = mask.clone();

        fill_uniform_regions(
            &mut output,
            &mut pending_mask,
            &mask,
            &[rectangle_region([10, 10, 90, 90])],
        );

        assert_eq!(output.get_pixel(50, 50), &Rgb([10, 10, 10]));
        assert_eq!(pending_mask.get_pixel(50, 50), &Luma([u8::MAX]));
    }

    #[test]
    fn a_gradient_is_not_extrapolated_past_its_sampled_rows() {
        let mut image = vertical_gradient(60, 200);
        let mut mask = GrayImage::new(60, 200);
        // Lettering spans the full width of the bottom rows, so no row below
        // 150 has a background sample.
        for y in 150..200 {
            for x in 0..60 {
                image.put_pixel(x, y, Rgb([255, 255, 255]));
                mask.put_pixel(x, y, Luma([u8::MAX]));
            }
        }
        let mut output = image.clone();
        let mut pending_mask = mask.clone();

        fill_uniform_regions(
            &mut output,
            &mut pending_mask,
            &mask,
            &[rectangle_region([0, 0, 60, 200])],
        );

        assert_eq!(pending_mask.get_pixel(30, 170), &Luma([u8::MAX]));
    }
```

- [ ] **Step 4: 프로파일 반경 상수 추가**

Task 1에서 추가한 상수 아래에 추가한다.

```rust
const BACKGROUND_INLIER_RATIO: f64 = 0.9;
const BACKGROUND_PROFILE_RADIUS: usize = 4;
```

- [ ] **Step 5: `RegionBackground`에 `Rows`/`Columns` 추가**

Task 1의 `enum RegionBackground`와 `impl RegionBackground`를 아래로 바꾼다.

```rust
/// The background a bubble's lettering sits on, predicted for any pixel inside
/// the bubble's bounds.
#[derive(Clone, Debug)]
enum RegionBackground {
    Flat(Rgb<u8>),
    /// One color per row, starting at the bounds' top edge.
    Rows(Vec<Rgb<u8>>),
    /// One color per column, starting at the bounds' left edge.
    Columns(Vec<Rgb<u8>>),
}

impl RegionBackground {
    fn color(&self, bounds: [u32; 4], x: u32, y: u32) -> Rgb<u8> {
        match self {
            Self::Flat(color) => *color,
            Self::Rows(colors) => colors[(y - bounds[1]) as usize],
            Self::Columns(colors) => colors[(x - bounds[0]) as usize],
        }
    }
}
```

- [ ] **Step 6: `region_background`가 프로파일 모델을 시도하게 바꾸기**

Task 1의 `fn region_background(...) { ... }` 전체를 아래로 바꾼다. 채울 픽셀(`targets`)을 받아 외삽 여부를 판단한다.

```rust
fn region_background(
    image: &RgbImage,
    mask: &GrayImage,
    region: &FlatFillRegion,
    targets: &[(u32, u32)],
) -> Option<RegionBackground> {
    let [left, top, right, bottom] = region.bounds;
    let mut samples = Vec::new();
    for y in top..bottom {
        for x in left..right {
            let point = (x as f32 + 0.5, y as f32 + 0.5);
            if mask.get_pixel(x, y)[0] >= 127
                || !point_in_polygon(point, &region.polygon)
                || polygon_edge_distance_squared(point, &region.polygon)
                    < FLAT_FILL_EDGE_MARGIN * FLAT_FILL_EDGE_MARGIN
            {
                continue;
            }
            samples.push((x, y, *image.get_pixel(x, y)));
        }
    }
    if samples.len() < UNIFORM_BACKGROUND_MIN_PIXELS {
        return None;
    }

    let flat = RegionBackground::Flat(median_color(samples.iter().map(|&(_, _, color)| color)));
    if inlier_ratio(&flat, region.bounds, &samples) >= BACKGROUND_INLIER_RATIO {
        return Some(flat);
    }
    let ratio = |background: &Option<RegionBackground>| {
        background
            .as_ref()
            .map_or(0.0, |background| inlier_ratio(background, region.bounds, &samples))
    };
    let rows = axis_profile(&samples, targets, |_, y| y, top, bottom).map(RegionBackground::Rows);
    let columns =
        axis_profile(&samples, targets, |x, _| x, left, right).map(RegionBackground::Columns);
    let (rows_ratio, columns_ratio) = (ratio(&rows), ratio(&columns));
    let (best, best_ratio) = if columns_ratio > rows_ratio {
        (columns, columns_ratio)
    } else {
        (rows, rows_ratio)
    };
    best.filter(|_| best_ratio >= BACKGROUND_INLIER_RATIO)
}
```

- [ ] **Step 7: `axis_profile`와 `lerp_color` 추가**

`median_color` 바로 뒤에 추가한다.

```rust
/// One color for every line from `start` to `end` along the axis `line`
/// selects. A sampled line takes the median of the samples within
/// `BACKGROUND_PROFILE_RADIUS` lines of it, so a line with few samples cannot
/// fit them alone; an unsampled line interpolates between its sampled
/// neighbors. `None` when a target lies outside the sampled lines, because the
/// profile would have to extrapolate.
fn axis_profile(
    samples: &[(u32, u32, Rgb<u8>)],
    targets: &[(u32, u32)],
    line: impl Fn(u32, u32) -> u32,
    start: u32,
    end: u32,
) -> Option<Vec<Rgb<u8>>> {
    let length = (end - start) as usize;
    let mut lines = vec![Vec::new(); length];
    for &(x, y, color) in samples {
        lines[(line(x, y) - start) as usize].push(color);
    }
    let first = lines.iter().position(|colors| !colors.is_empty())?;
    let last = lines.iter().rposition(|colors| !colors.is_empty())?;
    if targets.iter().any(|&(x, y)| {
        let index = (line(x, y) - start) as usize;
        index < first || index > last
    }) {
        return None;
    }

    let known = (0..length)
        .filter(|&index| !lines[index].is_empty())
        .map(|index| {
            let window = index.saturating_sub(BACKGROUND_PROFILE_RADIUS)
                ..(index + BACKGROUND_PROFILE_RADIUS + 1).min(length);
            (index, median_color(lines[window].iter().flatten().copied()))
        })
        .collect::<Vec<_>>();
    let colors = (0..length)
        .map(|index| {
            let after = known.partition_point(|&(known_index, _)| known_index < index);
            match (after.checked_sub(1).map(|before| known[before]), known.get(after)) {
                (_, Some(&(known_index, color))) if known_index == index => color,
                (Some((from_index, from)), Some(&(to_index, to))) => lerp_color(
                    from,
                    to,
                    (index - from_index) as f32 / (to_index - from_index) as f32,
                ),
                (Some((_, color)), None) | (None, Some(&(_, color))) => color,
                (None, None) => unreachable!("a sampled line exists"),
            }
        })
        .collect();
    Some(colors)
}

fn lerp_color(from: Rgb<u8>, to: Rgb<u8>, t: f32) -> Rgb<u8> {
    Rgb(std::array::from_fn(|channel| {
        let from = f32::from(from[channel]);
        (from + (f32::from(to[channel]) - from) * t).round() as u8
    }))
}
```

- [ ] **Step 8: `fill_uniform_regions`가 `targets`를 넘기게 바꾸기**

변경 전:

```rust
        let Some(background) = region_background(output, pending_mask, region) else {
```

변경 후:

```rust
        let Some(background) = region_background(output, pending_mask, region, &targets) else {
```

- [ ] **Step 9: BallonsTranslator 주석에 의도적 차이 추가**

`struct TilePlan` 위 주석의 마지막 두 줄을 바꾼다.

변경 전:

```rust
// Koharu uses the detected bubble polygons as those blocks. Uniform bubbles are
// filled first; only the remaining mask is split into bounded model crops.
```

변경 후:

```rust
// Koharu uses the detected bubble polygons as those blocks. Uniform bubbles are
// filled first; only the remaining mask is split into bounded model crops.
// Unlike the deviation test there, a bubble passes when nearly all of its
// background matches a flat color or a row/column gradient, so outline pixels
// the polygon catches do not send an opaque bubble to the model.
```

- [ ] **Step 10: 인페인팅 테스트 전체가 통과하는지 확인**

Run: `cargo test -p koharu-pipeline --lib inpainting::tests`
Expected: PASS. Task 1·2에서 추가한 테스트 4개와 기존 인페인팅 테스트가 모두 통과한다. 경고가 없어야 한다.

- [ ] **Step 11: 커밋**

```bash
git add crates/koharu-pipeline/src/stages/inpainting.rs
git commit -m "feat(inpainting): fill row and column gradient bubbles without the model" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: 실제 페이지 확인 (사용자)

코드 변경은 없다. 실행자는 이 단계를 하지 않고, 사용자에게 확인을 요청한다.

- [ ] **Step 1: 확인 요청**

사용자에게 `C:\Users\chyng\Koharu_test\03.webp`를 앱에서 다시 탐지·인페인팅해 달라고 요청한다. 확인할 항목은 다음과 같다.

- 흰 말풍선 5개(「パンパカパーン」「これからよろしくね」「今日は許して」「はい！」「先生！」)와 검은 박스 「ふふ…生徒と」가 모델 생성 흔적 없이 단색으로 지워졌는가.
- 그라데이션 박스 「何を…期待して」가 그라데이션 그대로 지워지고, 성 같은 생성 그림이 없는가.
