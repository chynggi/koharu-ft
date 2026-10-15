# NeKoharu 0.84.1 이식 설계

작성일: 2026-10-15

## 배경

`nekojin116/NeKoharu`는 이 저장소의 포크다. 이미 설정된 원격(`nekoharu`)이고, 태그
`v0.84.1` = `56c5c64f`는 `a8f4acb7`의 자손이다. 포크의 분기점은 `f9b22ba0`이며, 그것은
이 저장소의 `main`이 이미 병합한 지점(`plans/2026-10-12-nekoharu-fork-merge-notes.md`)과
같다. 따라서 이번 작업은 **재구현이 아니라 포트**다.

이식 대상은 기준선 `5b3ea688^`(=`f9b22ba0`)부터 `a8f4acb7`까지의 **7개 커밋**이다.
그중 `a8f4acb7`은 버전 범프이고 나머지 6개가 기능이다.

| 커밋 | 제목 | 기능 |
|---|---|---|
| `5b3ea688` | feat(editor): duplicate text layers with Ctrl+D | 텍스트 복제 |
| `d214095d` | fix(editor): keep pasted text copies horizontal like their original | 텍스트 상자 복사 버그 |
| `5408bdfe` | feat(fonts): favorite fonts and pin them at the top of the picker | 글꼴 즐겨찾기 |
| `050dd2a0` | feat(rail): show each page's order number on its thumbnail | 페이지 번호 (배지) |
| `d1d832c9` | feat(rail): show the page order number under the page name | 페이지 번호 (재배치) |
| `8af42854` | feat(pages): merge spreads right to left and add NeKoharu Features settings | RTL 병합 + 설정 탭 |

## 범위

포크의 0.84.1 기능을 이 저장소의 구조로 옮긴다.

**포함**: 위 표의 6개 커밋이 만드는 동작 전부 — 텍스트 복제, 붙여넣기 방향 수정, 글꼴
즐겨찾기, 페이지 레일 순서 번호, RTL 스프레드 병합, NeKoharu Features 설정 탭, 9개
로케일 번역.

**제외**:
- **버전 범프(`a8f4acb7`)** — N-D1 참조.
- 포크 브랜딩(`README.md`, `AboutDialog.tsx`, `cliff.toml`, 릴리스 워크플로,
  `tauri.conf.json`의 업데이터 엔드포인트) — 포크 소유이며 기존 병합에서도 되돌린 항목이다.
- 포크의 Tauri invoke 브리지와 specta 바인딩 — 이 저장소는 `fcdba40a`부터 손으로 쓴
  HTTP 클라이언트를 쓰고 `specta`는 휴면 상태다.
- 포크의 프런트 테스트를 그대로 받는 것 — 포크 테스트는 Tauri invoke와 포크 메뉴바를
  전제한다. **동작은 이식하되 테스트는 이 저장소의 UI 기준으로 다시 쓴다.**

## 결정 사항

| # | 결정 | 근거 |
|---|---|---|
| N-D1 | **버전 범프를 이식하지 않는다.** `Cargo.toml`은 `0.83.5`를 유지한다 | `upstream/main`의 `Cargo.toml`도 `0.83.5`이고 `CHANGELOG.md`의 최신 항목도 `0.83.5`다(확인함). 포크의 `0.84.1`은 포크가 upstream `0.84.0`을 앞서 가기 때문이며, 우리가 그 번호를 채택하면 **존재하지 않는 upstream 릴리스**와 어긋나고 `/meta`가 보고하는 버전(`env!("CARGO_PKG_VERSION")`, `crates/koharu-rpc/src/routes/meta.rs`)이 거짓이 된다 |
| N-D2 | **스프레드 병합을 오른쪽→왼쪽으로 전환한다** | 만화 읽기 순서다. 포크의 의도적 발산이며, 우리가 이를 받아들이기로 확정했다. 이 저장소의 기존 테스트 4개 중 2개가 LTR 위치를 고정하고 있어 함께 갱신한다(N-D6) |
| N-D3 | **`mergeDeleteOriginals`의 기본값은 `true`다** — 포크의 `false`와 다르다 | 포크의 기본값은 "원본 유지"이고, 그것은 이 저장소의 현재 동작(항상 삭제, `project.rs`의 `remove_entity` 루프)을 **바꾼다**. 발산을 택해 기존 사용자 동작을 보존한다. 기본값은 `forkPreferenceDefaults` 한 곳에서 정의하고, 포크와 다르다는 사실을 그 자리에 주석으로 남긴다 |
| N-D4 | 설정은 **`localStorage`** 에 저장한다 | 포크 구현이 그러하고, 이 저장소에 같은 패턴이 이미 있다(`lib/colorHistory.ts`). 서버 `Preferences`(`crates/koharu-app/src/commands/preferences.rs`)는 파이프라인·자격증명처럼 **서버가 소유해야 하는** 설정을 담는다. 이 두 토글은 클라이언트 UI 상태이고 프로젝트 파일과 무관하므로 서버로 올리면 저장 경로와 수명주기만 늘어난다 |
| N-D5 | wire 필드 이름은 **`delete_originals`**(snake_case) | 포크는 Tauri invoke에 `deleteOriginals`(camelCase)를 넘기지만, 이 저장소의 `protocol.ts`는 `chapter_start`(`:205`), `expected_revision`(`:291`)처럼 snake_case를 보내고 라우트 구조체에는 `rename_all`이 없다(확인함: `crates/koharu-rpc/src/routes`에 `rename_all` 없음). camelCase로 보내면 `MergePagesRequest`가 역직렬화에 실패한다 |
| N-D6 | **삭제 여부 배선은 4곳이다** — 포크의 1줄이 아니다 | `protocol.ts:209` → `routes/pages.rs:398`(`MergePagesRequest`) → `:403`(핸들러) → `editing.rs:150`(커맨드) → `project.rs:561`(`merge_pages`). 포크는 Tauri invoke라 `protocol.ts`만 고치면 됐다. 우리는 HTTP 라우트를 지나므로 네 곳이 한 계약이다 |
| N-D7 | `FontList`의 행을 **`<button>`에서 `<div role='option'>`로 바꾼다** | 즐겨찾기 별 버튼이 행 안에 들어가는데, HTML은 `<button>` 안에 `<button>`을 허용하지 않는다. `role='option'`은 유지되므로 `getByRole('option')` 조회는 계속 동작한다. 행 클릭은 `div`의 `onClick`으로 남는다 |
| N-D8 | 페이지 번호의 두 커밋(`050dd2a0` + `d1d832c9`)을 **한 태스크로 합친다** | `d1d832c9`는 `050dd2a0`이 만든 배지를 같은 릴리스 안에서 재배치한 커밋이다(축소판 위 → 레이블 아래). 둘을 별도 태스크로 두면 "축소판을 가리는 배지"라는, 포크가 스스로 버린 중간 상태를 우리 트리에 커밋하게 된다 |
| N-D9 | 설정 탭 이름은 포크의 **"NeKoharu Features"** 를 유지한다 | 파트너가 명시적으로 승인했다. 이 저장소에 `NeKoharu` 문자열이 하나도 없게 되지만(확인함), 승인된 사항이다. 탭 식별자는 `features`, 아이콘은 `Sparkles` |
| N-D10 | **`deleteOriginals`가 `false`일 때 원본 페이지의 자식은 이미 스프레드로 옮겨진 뒤다** | 포크 구현에서 `remove_entity` 루프만 조건부가 되고 자식 재부모화는 그대로다. 결과적으로 원본은 **이미지 전용 페이지**로 남는다. 이것이 의도된 동작이며 문구("레거시 원본은 이미지 전용으로 남는다")와 일치한다 |

### N-D2가 이 서브프로젝트에서 가장 위험한 결정이다

RTL 전환은 픽셀 offset과 기하 이동을 **동시에** 뒤집는다. 포크의 diff는 두 곳을 고친다:

```rust
// 픽셀: 각 페이지는 자기 뒤 페이지들의 폭만큼 오른쪽에 놓인다
offsets = offsets.into_iter().zip(&widths)
    .map(|(offset, &width)| total_width - offset - width).collect();
```

LTR에서 `offsets[i]`는 "왼쪽 페이지들의 누적 폭"이고, RTL에서는 "오른쪽 페이지들의
누적 폭"이다. `widths`를 새로 모으는 이유가 이것이다(`total_width - offset - width`가
`i`번째 페이지의 오른쪽 이웃 합이 되려면 자기 폭을 알아야 한다).

**기하와 픽셀이 같은 `offsets` 벡터를 공유하므로 둘이 어긋날 수 없다** — 이것이 이
설계에서 안전한 부분이다. `compose_offsets = offsets.clone()`으로 픽셀 합성에 넘기고
(`project.rs:615`), 같은 벡터로 `roots`의 `offset`을 정한다(`:687`). 재패딩 경로
(`:595-613`)도 같은 `offsets[index]`를 쓴다.

**우리 트리에서 1c가 추가한 경로**: 재패딩 대상 판정이 `SceneRasterLayer` **또는**
`SceneShapeLayer.rasterized`다(`project.rs:598-601`). RTL 전환은 `offsets`의 **값**만
바꾸므로 이 경로를 그대로 통과한다. 다만 이 경로를 덮는 테스트
(`merging_pages_repads_a_rasterized_shape_asset`)가 위치를 단언하지 않으므로 RTL
회귀를 잡아주지 못한다 — 별도 확인이 필요하다(N-D6의 테스트 갱신 대상 참조).

**기존 테스트 중 실제로 깨지는 것** (확인함):

| 테스트 | 페이지 폭 | 현재 단언 (LTR) | RTL 기대값 | 비고 |
|---|---|---|---|---|
| `merging_adjacent_pages_composes_a_wide_spread_and_keeps_layers` | `(2, 4)` | `region.points[0].x == 2.0` | 첫 페이지 `4.0`, 둘째 `0.0` | LTR/RTL이 서로 다른 값이라 회귀를 잡는다 |
| `merging_pages_shifts_a_line_shape_with_its_polygon` | `(20, 30)` | `first.x == 20.0` | **`0.0` — 도형을 옮겨야 한다(아래)** | 도형이 `right`=**마지막 페이지**라 RTL offset이 0이다 |
| `merging_pages_repads_a_rasterized_shape_asset` | `(20, 30)` | 폭이 `(50, 10)`인지만 단언 | `(50, 10)` 그대로 | 도형이 역시 `right`라 RTL offset 0 → 재패딩이 항등이다 |
| `merging_pages_keeps_a_shapes_pattern_asset` | `(20, 30)` | 부모·픽셀값만 단언 | **통과** | 패턴 자산은 엔티티에 붙어 함께 이동하고 재패딩 대상이 아니다(`rasterized: false`) |

계산한 offset(`total_width - offset - width`):

```
spread (2,4):  total=6   LTR=[0,2]  RTL=[4,0]   → 어긋난다. 회귀를 잡는다
line  (20,30): total=50  LTR=[0,20] RTL=[30,0]  → right(idx1)는 RTL에서 0이다
repad (20,30): 도형이 right → RTL offset 0 → 재패딩이 항등. 치수만 단언하므로 통과하나
               위치 회귀를 전혀 잡지 못한다
```

**두 테스트가 조용히 무력해진다 — 이것이 RTL 전환의 진짜 함정이다.** 위 표의 `line`과 `repad`는 도형을 **`right`, 즉 마지막 페이지**에 둔다. RTL에서 마지막 페이지의 offset은 **0**이므로:

- `line` 테스트: `if offset == 0.0 { continue; }`(`project.rs:722-724`) 때문에 **shift 경로를 한 번도 실행하지 않는다.** 이 테스트가 존재하는 이유(1c에서 `write_geometry`가 선의 파생 끝점을 함께 옮기지 않아 생긴 결함을 고정)가 사라진다.
- `repad` 테스트: 재패딩이 항등이 되어, RTL을 검출할 픽셀 경계가 존재하지 않는다.

**그래서 두 테스트의 도형을 `left`(인덱스 0)로 옮긴다.** 그러면 offset이 `LTR=0 → RTL=30`이 되어 양쪽 방향에서 shift 경로를 실행하고 검출 경계가 생긴다. 이동에 따라 `line`의 기대값은 `20.0 → 30.0`이 되고, `repad`의 베이크 치수 단언은 `(30, 10) → (20, 10)`이 된다.

포크도 자기 테스트를 `left`/`right` → `first`/`second`로 개명하고 `4.0`/`0.0`으로 바꿨다. 우리는 기존 테스트를 **RTL 기대값으로 갱신**하고, RTL을 되돌리면 실패하는지 확인해 회귀를 실제로 고정한다.

**주의 — 두 페이지 폭이 같은 경우.** `widths = [w, w]`이면 RTL offset도 `[w, 0]`이
아니라 `[0, w]`이므로 여전히 구분된다(`offset = total - offset - w`에서 `total = 2w`).
그러나 **테스트를 새로 쓸 때 위 두 케이스처럼 폭을 다르게 두어야** 어느 페이지가 어디로
갔는지가 위치만으로 결정된다. 같은 폭을 쓰면 "첫 페이지가 오른쪽"이라는 사실이
우연히 offset에 드러나지 않는다. `merging_can_keep_the_original_pages`(포크 테스트)도
폭이 `2`와 `4`로 다르다.

**의도적 발산임을 코드 옆에 남긴다.** 포크의 `merge_pages` doc comment는 이 결정을
"NeKoharu diverges from upstream here: manga reads right to left"로 명시한다. 그 문장을
offset 재계산 바로 옆에 유지하지 않으면, 미래의 독자가 이것을 버그로 보고 왼쪽→오른쪽으로
"고칠" 것이다. 이 저장소의 `AGENTS.md`가 요구하는 "의도적 발산을 해당 코드 옆에 설명"이
정확히 이 경우다.

## 데이터 모델

변경 없음. RTL은 `offsets` 계산과 테스트 기대값에만 영향을 주고, `Page`·`SceneGeometry`·
자산 스키마는 그대로다. `delete_originals`는 커맨드 인자이고 저장되지 않는다.

## 커맨드

### `merge_pages` — 시그니처 확장

```rust
// crates/koharu-app/src/commands/editing.rs
pub async fn merge_pages(
    pages: Vec<EntityId>,
    delete_originals: bool,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<EntityId, Error>

// crates/koharu-app/src/commands/project.rs
pub(crate) async fn merge_pages(
    &mut self,
    pages: Vec<EntityId>,
    delete_originals: bool,
) -> Result<(Commit, EntityId)>
```

`editing.rs`의 커맨드는 이 저장소에서 이미 `pub`이고 라우트가 다른 크레이트에서 부른다
(기존 병합에서 `pub(crate)` → `pub`로 고친 지점). 발행은 이 저장소 방식인
`canvas_channel.publish`를 유지한다.

### 라우트 — 요청 구조체 확장

```rust
// crates/koharu-rpc/src/routes/pages.rs
#[derive(Deserialize)]
struct MergePagesRequest {
    pages: Vec<EntityId>,
    delete_originals: bool,
}
```

### `protocol.ts` — 클라이언트 시그니처

```ts
mergePages: (pages: EntityId[], deleteOriginals: boolean) =>
  post<EntityId>("/pages/merge", { pages, delete_originals: deleteOriginals }),
```

호출부는 `PageRail.tsx:267` 한 곳이다. `useForkPreferences`의
`preferences.mergeDeleteOriginals`를 넘긴다.

### 새 커맨드 없음

글꼴 즐겨찾기·설정 토글·페이지 번호는 모두 클라이언트 전용이다. 텍스트 복제는 **기존
`pasteTextLayers`를 재사용**한다(포크와 동일).

## 프론트

### 새 파일 2개

| 파일 | 책임 |
|---|---|
| `packages/koharu/lib/fontFavorites.ts` | 즐겨찾기 글꼴 이름 배열의 zustand 스토어 + localStorage 영속. `FONT_FAVORITES_STORAGE_KEY = 'koharu.font-favorites'`. 로드 시 문자열 필터 + 중복 제거 |
| `packages/koharu/lib/forkPreferences.ts` | `ForkPreferences { mergeDeleteOriginals, showPageNumbers }` 스토어 + localStorage 영속. `FORK_PREFERENCES_STORAGE_KEY = 'koharu.fork-preferences'` |

두 파일 모두 `lib/colorHistory.ts`의 구조(스토어 + `load*`/`persist*` + `typeof window`
가드 + `try/catch`)를 따른다.

**`loadForkPreferences`의 기본값은 둘 다 `true`다** (N-D3):

```ts
mergeDeleteOriginals: stored.mergeDeleteOriginals !== false,  // 없으면 true (N-D3)
showPageNumbers: stored.showPageNumbers !== false,            // 없으면 true
```

**포크와 다르다.** 포크는 `mergeDeleteOriginals: stored.mergeDeleteOriginals === true`
(없으면 `false`, 즉 "원본 유지가 기본")인 반면 우리는 `!== false`(없으면 `true`, 즉
"원본 삭제가 기본")다. 이 비대칭은 N-D3의 결정 그대로이며, **왜 갈라지는지 그 자리에
주석으로 남긴다** — 그렇지 않으면 다음 포크 병합에서 "포크와 다르다"는 이유로 되돌려질
자리다. 두 필드가 같은 형태(`!== false`)가 되므로 우연이 아니라 결정임이 코드에서
드러난다.

### 수정 파일

| 파일 | 변경 |
|---|---|
| `components/editor/CanvasWorkspace.tsx` | `duplicateSelectedTextLayers` 임포트 + `Ctrl/Cmd+D` 분기(기존 `c`/`v`/`z`/`a` 분기 사이) |
| `lib/layerClipboard.ts` | `selectedTextLayerIds` 추출, `paste()` 공통 함수 추출, `duplicateSelectedTextLayers` 추가 |
| `components/controls/FontPicker.tsx` | `useFontFavorites` 사용, `FontRow` 유니온 + 헤더, `rowOffset`, 행을 `div`로(N-D7) |
| `components/editor/PageRail.tsx` | `PageItem`에 `index` prop, 순서 번호를 레이블 아래 텍스트 열에, `mergePages`에 설정 전달, `showPageNumbers` 게이트 |
| `components/preferences/SettingsPage.tsx` | `features` 탭 추가(`Sparkles`), `FeaturesPreferences` 렌더 |
| `public/locales/*/translation.json` (9개) | `fontPicker.{addFavorite,allFonts,favorites,removeFavorite}` + `settings.features.*` + `settings.tabs.features` |

### 새 컴포넌트

`components/preferences/FeaturesPreferences.tsx` — `PreferencePage`/`PreferenceSection`/
`PreferenceRow`(기존 `PreferenceFields.tsx`가 내보냄) + `Switch`
(`@koharu/ui/components/switch`) 두 개. 각 `Switch`는 `aria-label`로 행 제목을 그대로
쓴다(포크와 동일, 테스트가 이 이름으로 조회한다).

### 번역

포크의 9개 로케일 문구를 그대로 옮긴다. `tests/lib/localization.test.ts`가 **모든 로케일의
flattened leaf 키 집합이 `en-US`와 동일**할 것을 요구하므로, 하나라도 빠지면 테스트가
실패한다. 이것이 이 태스크의 번역 누락 방지 장치다.

## 테스트

### Rust

| 테스트 | 검증 |
|---|---|
| `pasted_copies_only_keep_user_authored_writing_direction` (포크 `d214095d`) | 파이프라인 추론 방향은 복사본에서 지워지고(`writing_mode == None`), 사용자가 명시한 방향은 보존된다. `origin`은 둘 다 `User`, 색은 보존 |
| `merging_adjacent_pages_...` (갱신) | RTL offset을 단언. 첫 페이지 4.0, 둘째 0.0 |
| `merging_pages_shifts_a_line_shape_with_its_polygon` (갱신) | RTL에서 선 도형의 다각형이 같은 offset으로 이동 |
| `merging_can_keep_the_original_pages` (포크 `8af42854`) | `delete_originals=false`면 페이지가 3개(원본 2 + 스프레드 1)이고, 원본의 레이어는 여전히 스프레드로 이동 |
| RTL 회귀 고정 (신규) | `merging_pages_repads_a_rasterized_shape_asset`에 위치 단언 추가 — RTL을 되돌리면 실패해야 한다 |

**기존 `merge_pages` 호출부 5곳**(테스트 내)에 `true`를 넘겨야 컴파일된다.

### 프런트

| 테스트 | 검증 |
|---|---|
| 복제 (workspace.test) | `Ctrl+D`가 `pasteTextLayers`를 선택 레이어로 호출하고, 복사본을 선택하며, **클립보드를 건드리지 않는다** |
| 즐겨찾기 (editor-components) | 별 클릭 → 헤더 아래 상단 고정 → localStorage 영속 → 해제 시 헤더 사라짐. `getByRole('option')` 순서로 고정을 확인 |
| 페이지 번호 (editor-components) | 레이블과 순서가 다를 때 `[data-page-number]`가 1,2를 표시. **번호가 축소판 div 안에 없음**을 단언(N-D8) |
| 페이지 번호 숨김 (editor-components) | 설정이 꺼지면 `[data-page-number]`가 없고 레이어 수 텍스트는 남음 |
| 병합 설정 (editor-components) | 기본값이 포크의 `false`가 아니라 N-D3의 `true`로 호출되고, 토글 후 `false`로 호출됨 |
| 설정 탭 (editor-components) | 탭 클릭 → 제목 → 두 스위치 토글 → localStorage 값 |
| 로케일 (localization, 기존) | 9개 로케일 키 집합 동일 |

## 검증 계획

| 검사 | 명령 |
|---|---|
| 앱 단위 테스트 | `cargo test -p koharu-app --lib` |
| RPC 라우트 | `cargo test -p koharu-rpc --lib` |
| 워크스페이스 | `cargo check --workspace` |
| 프런트 테스트 | `bun x vitest run` (`packages/koharu`) |
| 타입 | `bun x tsc --noEmit` (`packages/bridge`, `packages/koharu`) |
| 린트 | `bun run --filter @koharu/app lint` |

## 열린 문제

- **RTL 픽셀 합성의 육안 확인.** 단위 테스트는 offset과 기하를 덮지만 합성된 스프레드
  이미지에서 좌우가 실제로 바뀌었는지는 픽셀을 읽어야 안다. 포크도 이 부분에 테스트가
  없다. 재패딩 테스트처럼 합성 결과의 픽셀을 확인하는 단언을 추가할지 검토한다.
  (기존 병합 문서 §5가 "실기 동작은 확인하지 않았다"로 남긴 구멍과 같은 종류다.)
- **`showPageNumbers`의 위치.** 포크는 레이블 아래 텍스트 열에 번호를 넣고 `' · '`로
  레이어 수와 잇는다. 이 저장소의 레일 행 높이(`estimateSize: 72`)가 포크와 같은지
  확인하지 않았다. 다르면 배치가 어긋날 수 있다 — 구현 시 실제 렌더를 확인한다.
