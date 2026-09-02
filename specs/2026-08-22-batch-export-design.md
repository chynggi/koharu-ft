# 일괄 내보내기 설계

작성일: 2026-08-22

## 배경

백엔드는 이미 일괄 내보내기를 한다. `koharu-app/src/commands/output.rs`의
`export_pages_to`는 페이지 목록을 받아 4개씩 동시에 렌더링하고
`0001_라벨.png` 형식으로 저장한다. 빈 목록이면 전체 페이지가 대상이다.

막혀 있는 쪽은 프론트엔드와 작업 모델이다.

- `TitleBar.tsx:278`의 `exportSelection()`이 선택된 페이지 또는 현재 페이지
  하나만 넘긴다. 전체를 내보내려면 네비게이터에서 전부 선택해야 한다.
- 메뉴는 `PNG로 내보내기` / `PSD로 내보내기` 두 항목뿐이고 진행률도 취소도
  없다. 200페이지 PSD를 뽑으면 멈춘 것처럼 보인다.
- 브라우저 경로(`export_download`)는 ZIP 전체를 메모리에 만든 뒤에야 응답한다.
  응답이 끝에서야 나오므로 이 구조로는 진행률을 보낼 수 없다.

## 목표

1. 선택 작업 없이 전체 페이지를 한 번에 내보낸다.
2. 진행률을 표시하고 중단할 수 있다.
3. 형식·대상·파일명·폴더 구조를 한 화면에서 고른다.
4. PNG와 PSD를 한 번의 실행으로 함께 낸다.

데스크톱 창(네이티브 폴더 선택 후 디스크 직접 쓰기)과 원격 브라우저(ZIP
다운로드)를 모두 지원한다.

### 범위 밖

**PNG 투명 배경.** 검토 중 전제가 틀린 것으로 확인됐다.
`RasterOptions`(`koharu-rasterizer/src/native.rs:47`)에는 배경 필드가 없다.
`rasterize_frame_inner`는 투명한 표면에서 시작해 프레임의 합성 명령을 그리므로
출력은 이미 RGBA이고, 만화 페이지는 원본 이미지가 전면을 덮어 결과적으로
불투명해질 뿐이다. 따라서 "투명 배경"은 체크박스가 아니라 원본 에셋 레이어를
제외하고 조판 결과만 렌더링한다는 뜻이 되고, 렌더러 레이어 필터링이 필요하다.
그런데 PNG 출력물 자체가 번역·조판이 끝난 결과물이므로 원본 위에 합성된 현재
기본값이 정확히 의도한 산출물이다. 이 옵션은 넣지 않는다.

## 접근

기존 Job 인프라를 재사용한다. `JobChannel`, SSE `job` 이벤트, `/process/stop`,
`ActivityCenter`의 진행률 UI가 모두 이미 있으므로 새로 만드는 개념이 없다.

`Processing.stops`의 단일 작업 제약(`processing.rs:113`,
`if !stops.is_empty()`)을 그대로 물려받는다. 따라서 번역 파이프라인이 도는
중에는 내보내기가 거부된다. 둘 다 GPU를 많이 쓰므로 이 제약은 손해가 아니라
안전장치다.

검토했으나 채택하지 않은 대안:

- **전용 Export 작업 상태 신설.** 파이프라인과 동시 실행이 가능해지지만
  백엔드 상태·라우트와 프론트 스토어·타입·i18n까지 신규 표면이 두세 배가 된다.
  동시 실행이 필요하다는 근거가 없다.
- **동기 유지, 진행률만 추가.** 가장 작지만 취소가 불가능해 목표 2를 못 맞춘다.

## 설계

### 1. 내보내기를 Job으로

`export_pages_to`의 시그니처를 바꾼다.

```rust
pub struct ExportOptions {
    pub formats: Vec<ExportFormat>,  // 최소 1개, PNG+PSD 동시 가능
    pub pattern: String,             // 파일명 템플릿
    pub subfolders: bool,            // 형식별 png/ psd/ 분리
}

pub async fn export_pages_to(
    directory: PathBuf,
    pages: Vec<EntityId>,       // 빈 목록 = 전체 (기존 동작 유지)
    options: ExportOptions,
    progress: Arc<dyn Fn(usize, usize, EntityId) + Send + Sync>,
    stop: StopToken,
    project: State<'_, CurrentProject>,
    desktop: State<'_, Desktop>,
) -> Result<(), Error>
```

`progress`는 `Option`이 아니다. 세 라우트가 전부 Job이 되므로 진행률을 원하지
않는 호출자가 없다.

`total = pages.len() × formats.len()`. 기존 `buffer_unordered(4)` 루프에서 각
항목을 시작하기 전에 `stop.stopped()`를 확인해 조기 종료하고, 완료마다
`progress`를 호출한다. 파이프라인의 취소와 같은 협조적 방식이라 이미 진행
중인 최대 4건은 마저 끝난다.

`total`이 페이지×형식인 것과 달리 **`{index}`는 형식과 무관하게 페이지의
1-기반 순번이다.** PNG와 PSD를 함께 내면 같은 페이지의 두 파일이 같은 번호를
갖는다 (`0001_page-01.png`, `0001_page-01.psd`). 진행률의 분모와 파일명의
번호는 서로 다른 것을 센다.

그 위에 Job 래퍼를 둔다. `process()`(`processing.rs:110-130`)와 같은 순서로
`Processing.stops`에 등록하고, `JobChannel`로
`Job { state, completed, total, page }`를 발행한다. `stage`와 `model`은
파이프라인 전용이므로 `None`이다. 종료 시 상태는 `Finished`,
`Failed { error }`, `Stopped` 중 하나다. 취소는 기존 `/process/stop`이 그대로
처리한다.

### 2. 파일명 템플릿

`crates/koharu-app/src/commands/naming.rs`에 Tauri 의존성 없는 순수 함수로
분리한다. 자유 템플릿을 고른 대가가 여기 몰리므로 단위 테스트가 붙을 수 있는
경계에 둔다.

| 토큰 | 의미 |
|---|---|
| `{index}` | 1부터. `{index:04}`로 자릿수 지정 |
| `{label}` | 페이지 라벨. 확장자 제거 + 금지문자 `_` 치환 (기존 규칙) |

기본값은 현재 동작과 같은 `{index:04}_{label}`이다.

검증 규칙:

- 알 수 없는 토큰이나 닫히지 않은 `{`는 오류다.
- 결과에 `/`, `\`, `..`가 있으면 거부한다. 라벨은 이미 정제되지만 템플릿
  리터럴로도 경로 이탈이 가능하다.
- 결과가 비면 `page`로 대체한다 (기존 동작).
- 이름이 충돌하면 뒤에 `_2`, `_3`을 붙인다. `{label}`만 쓰는 패턴에서 충돌이
  생긴다.

### 3. HTTP API

`Job`에 `kind: JobKind { Processing, Export }`를 추가한다. 작업 종류를 구분할
방법이 지금 없고, `model` 같은 기존 필드를 전용하는 것보다 정직하다. 프론트는
`kind`로 라벨을 고른다.

라우트는 제자리에서 바꾼다. 이 프로토콜의 소비자는 `@koharu/app` 하나뿐이라
버전을 나눌 이유가 없다.

| 라우트 | 변경 |
|---|---|
| `POST /pages/export` | `{pages, options, directory}` → `JobId` 즉시 반환 |
| `POST /pages/export/dialog` | 폴더 선택 대기 후 `JobId`. 사용자가 취소하면 `null` |
| `POST /pages/export/download` | `JobId` 반환, 스테이징 디렉터리에 렌더링 |
| `GET /pages/export/download/{job}` | 신규. ZIP 반환 후 스테이징 정리 |

**`export_dialog`은 네이티브 폴더 선택창을 띄우기 전에 단일 작업 제약을
확인한다.** 현재 순서(`pages.rs:225`)대로라면 파이프라인이 도는 중에도 폴더를
고르게 한 다음에야 "another process is already running"으로 거절하게 된다.
거절할 것이면 사용자를 붙잡기 전에 거절한다.

브라우저 경로가 2단계가 되면서 임시 디렉터리가 요청보다 오래 살아야 한다.
`Mutex<Option<(JobId, TempDir)>>` 하나를 `ExportStaging` 상태로 둔다. 단일
작업 제약 덕에 동시에 하나뿐이므로 맵이 필요 없고, 새 내보내기가 시작되면
이전 것이 교체되며 자동 삭제된다.

다운로드가 소비하면 비워진다. 실패하거나 취소된 내보내기의 스테이징은 그
자리에 남았다가 **다음 내보내기가 시작될 때** 교체되며 지워진다. 작업이 끝나는
시점에 즉시 지우려면 `start_export`가 스테이징을 알아야 하는데, 그 결합을
피하려고 슬롯을 라우트 쪽에 둔 것이다. 남는 것은 언제나 최대 한 개이고 다음
실행이 회수하므로 이 정도는 감수한다.

`GET /pages/export/download/{job}`은 요청된 job의 스테이징이 없으면 404를
반환한다. job이 `stopped`나 `failed`로 끝난 경우가 여기 해당한다.

**`archive_directory`는 재귀해야 한다.** 현재 구현(`pages.rs:429`)은
`std::fs::read_dir`로 한 겹만 읽고 `entry.path().is_file()`로 거른다. 형식별
하위 폴더를 켠 채 브라우저에서 내보내면 걸러진 결과가 비어
`bail!("the export produced no files")`로 실패한다. 렌더링은 멀쩡히 끝난
뒤이므로 오류 문구가 원인을 정반대로 가리킨다. 하위 경로를 ZIP 엔트리 이름으로
보존하도록 재귀시킨다.

`packages/bridge/src/protocol.ts`는 생성되지 않고 **손으로 쓴 파일이다**
(파일 첫 줄 주석). `Job`에 `kind`를 더하고 세 내보내기 명령의 시그니처를
바꾸는 변경은 전부 여기에 직접 반영해야 한다. 어딘가에서 자동 생성되기를
기다릴 것이 없다.

### 4. 프론트엔드

`components/app/ExportDialog.tsx`를 새로 만든다. `TitleBar`의
`menu.exportPng` / `menu.exportPsd` 두 항목은 대화상자를 여는 한 항목으로
대체한다.

대화상자 항목:

- **대상** — 전체 / 선택한 N개 (라디오). 선택이 없으면 전체만 활성
- **형식** — PNG / PSD 체크박스. 최소 하나 강제
- **파일명 패턴** — 텍스트 입력, 실시간 미리보기(`0001_page-01.png`), 파싱
  오류 즉시 표시
- **형식별 하위 폴더** — 체크박스. 형식을 둘 다 골랐을 때만 활성이며,
  비활성일 때는 체크 상태와 무관하게 `false`로 전송한다

시작을 누르면 대화상자가 닫히고 진행 상황은 `ActivityCenter`에 나타난다.
`JobItem`(`ActivityCenter.tsx:76`)은 이미 진행률·프로그레스바·중단 버튼을
렌더링하므로, `job.kind`로 라벨만 고르도록 고친다. 중단은 기존 `stopJob`
그대로다.

브라우저에서는 job이 `finished`가 되는 것을 SSE로 보고 ZIP을 받아
`saveBlob`으로 저장한다. `lib/transfer.ts`의 `runExport`가 이 흐름을 담는다.

### 5. i18n

신규 문구는 `packages/koharu/public/locales`의 9개 로케일(en-US, es-ES,
ja-JP, ko-KR, pt-BR, ru-RU, tr-TR, zh-CN, zh-TW)에 모두 채운다.

## 오류 처리

프로젝트 없음, 대상 디렉터리 없음, 다른 작업 실행 중
(`another process is already running`)은 기존 문구를 그대로 쓴다.

잘못된 템플릿은 대화상자에서 막되 **서버에서도 다시 검증한다.** HTTP API는
신뢰 경계이고, 경로 이탈을 클라이언트 검증에만 맡길 수 없다.

렌더링 실패는 `Job::Failed { error }`로 흐르고 `ActivityCenter`의 기존
`Failure`가 표시한다.

**취소 시 이미 쓴 파일은 지우지 않고 남긴다.** 되돌리려면 어느 파일이 이번
실행의 소산인지 추적해야 하는데, 사용자가 기존 파일을 덮어썼을 수 있어
안전하게 되돌릴 수 없다. 부분 결과를 남기고 알리는 편이 정직하다.

## 테스트

- `naming.rs` 단위 테스트가 중심이다. 자릿수 지정, 알 수 없는 토큰, 닫히지
  않은 중괄호, 경로 이탈, 충돌 접미사, 빈 결과.
- `tests/lib/transfer.test.ts`를 2단계 브라우저 흐름으로 확장한다.
- `ExportDialog`의 검증 동작(형식 최소 하나, 패턴 오류 표시)에 테스트를 붙인다.
- rpc 라우트는 `AppState`가 살아 있는 Tauri 핸들이라 단위 테스트가 어렵다.
  제외한다.

## 명세 위치에 대한 메모

기본 경로는 `docs/superpowers/specs/`지만 `docs/`는 `docs_dir = "."`에 명시적
`nav`를 가진 발행용 문서 사이트다. 여기에 두면 nav에 없는 고아 페이지로
빌드돼 사용자용 사이트에 섞이므로, 발행 대상이 아닌 `specs/`에 둔다.
