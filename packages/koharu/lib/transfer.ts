'use client'

import { useKoharuStore } from '@/lib/store'
import {
  commands,
  type EntityId,
  type ExportOptions,
  type JobId,
  type PageImportSource,
  type PageSummary,
} from '@koharu/bridge/protocol'

/**
 * Whether this document is the desktop CEF window rather than a browser.
 *
 * This is the question that decides how pages get in and out. In the desktop
 * window the server is the same machine as the user, so a native dialog is
 * both possible and better — it yields real paths and the bytes never move.
 * In a browser nothing can produce a path the server could open, so the files
 * themselves have to travel.
 *
 * NOTE: `WindowChrome` and `Updater` each carry their own copy of this check.
 * They predate this module and are left alone rather than refactored here.
 */
export function isEmbedded(): boolean {
  return typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window
}

/** Mirrors the extensions `koharu_app::commands::import::Format` accepts. */
const IMPORT_ACCEPT = '.png,.jpg,.jpeg,.webp,.cbz,.zip,.rar,.pdf'

const IMPORT_EXTENSIONS = IMPORT_ACCEPT.split(',')

function isImportable(name: string): boolean {
  const lower = name.toLocaleLowerCase()
  return IMPORT_EXTENSIONS.some((extension) => lower.endsWith(extension))
}

/**
 * Ask the browser for files. Resolves empty when the user dismisses the
 * picker, which callers should treat as "nothing to do" rather than an error.
 */
export function pickFiles(source: PageImportSource): Promise<File[]> {
  return new Promise((resolve) => {
    const input = document.createElement('input')
    input.type = 'file'
    input.multiple = true
    // A directory pick cannot also filter by extension, so the server-side
    // format check is what rejects anything unwanted in that case.
    if (source === 'folder') input.webkitdirectory = true
    else input.accept = IMPORT_ACCEPT
    input.addEventListener('change', () => resolve(Array.from(input.files ?? [])), { once: true })
    input.addEventListener('cancel', () => resolve([]), { once: true })
    input.click()
  })
}

/** Import pages by whichever route this client can actually use. */
export async function runImport(source: PageImportSource): Promise<PageSummary[]> {
  if (isEmbedded()) return commands.importPagesDialog(source)
  const files = await pickFiles(source)
  if (files.length === 0) return []
  return commands.importPagesUpload(files)
}

/**
 * 데스크톱 창에 떨어진 파일·폴더를 가져온다.
 *
 * 네이티브 드롭은 실제 경로를 주므로 바이트가 움직이지 않는다 — 다이얼로그와
 * 같은 `/pages/import` 경로를 쓴다. 서버가 디렉터리를 재귀 탐색한다.
 */
export function importDroppedPaths(paths: string[]): Promise<PageSummary[]> {
  if (!isEmbedded() || paths.length === 0) return Promise.resolve([])
  return commands.importPages(paths)
}

/** 브라우저에 떨어진 파일을 가져온다. 경로가 없으므로 업로드한다. */
export function importDroppedFiles(files: File[]): Promise<PageSummary[]> {
  if (isEmbedded() || files.length === 0) return Promise.resolve([])
  return commands.importPagesUpload(files)
}

async function readEntry(entry: FileSystemEntry, files: File[]): Promise<void> {
  if (entry.isFile) {
    const file = await new Promise<File | null>((resolve) => {
      ;(entry as FileSystemFileEntry).file(
        (value) => resolve(value),
        () => resolve(null),
      )
    })
    if (file && isImportable(file.name)) files.push(file)
    return
  }
  if (!entry.isDirectory) return
  const reader = (entry as FileSystemDirectoryEntry).createReader()
  // `readEntries`는 배치 단위로 돌려주며 끝에서 빈 배열을 준다. 한 번에 전부
  // 오지 않으므로 빌 때까지 반복해야 폴더가 잘리지 않는다.
  for (;;) {
    const batch = await new Promise<FileSystemEntry[]>((resolve) => {
      reader.readEntries(
        (entries) => resolve(entries),
        () => resolve([]),
      )
    })
    if (batch.length === 0) break
    for (const child of batch) await readEntry(child, files)
  }
}

/**
 * 브라우저 드롭에서 가져올 파일들을 모은다.
 *
 * 폴더는 `DataTransfer.files`에 담기지 않으므로 엔트리를 재귀 탐색한다. 지원
 * 형식이 아닌 파일은 조용히 건너뛴다 — 서버도 같은 확장자만 디코드한다.
 */
export async function filesFromDataTransfer(dataTransfer: DataTransfer): Promise<File[]> {
  const items = dataTransfer.items
  if (!items || items.length === 0) {
    return Array.from(dataTransfer.files).filter((file) => isImportable(file.name))
  }
  const files: File[] = []
  const entries: FileSystemEntry[] = []
  for (const item of Array.from(items)) {
    if (item.kind !== 'file') continue
    const entry = item.webkitGetAsEntry()
    if (entry) entries.push(entry)
    else {
      const file = item.getAsFile()
      if (file && isImportable(file.name)) files.push(file)
    }
  }
  for (const entry of entries) await readEntry(entry, files)
  return files
}

/**
 * 내보내기 Job을 시작한다. 렌더링은 백그라운드에서 돌고, 진행률은 job
 * 이벤트로 온다. 데스크톱에서 사용자가 폴더 선택을 취소하면 `null`이다.
 */
export async function runExport(
  pages: EntityId[],
  options: ExportOptions,
): Promise<JobId | null> {
  if (isEmbedded()) return commands.exportPagesDialog(pages, options)
  return commands.exportPagesDownload(pages, options)
}

/**
 * Job이 끝난 뒤 결과를 사용자에게 넘긴다.
 *
 * 데스크톱은 이미 사용자가 고른 폴더에 파일이 들어 있으므로 할 일이 없다.
 * 브라우저는 여기서 ZIP을 받아 저장한다 — 서버가 임시 디렉터리를 붙들고
 * 있으므로 이 호출이 그것을 비우는 역할도 한다.
 */
export async function finishExport(job: JobId): Promise<void> {
  if (isEmbedded()) return
  saveBlob(await commands.getExportArchive(job), 'koharu-export.zip')
}

/**
 * Job이 끝날 때까지 기다렸다가 `finishExport`를 부른다.
 *
 * 진행률은 이미 `ActivityCenter`가 보여주므로 여기서는 완료만 본다. 스토어
 * 구독을 쓰는 것은 job 이벤트가 SSE로 오기 때문이다 — 폴링할 것이 없다.
 */
export function finishExportWhenDone(job: JobId): Promise<void> {
  return new Promise((resolve) => {
    const check = (state: { jobs: Record<string, { state: string }> }) => {
      const current = state.jobs[job]
      if (!current || current.state === 'running') return
      unsubscribe()
      if (current.state === 'finished') void finishExport(job).then(resolve, () => resolve())
      else resolve()
    }
    const unsubscribe = useKoharuStore.subscribe(check)
    check(useKoharuStore.getState())
  })
}

/** Hand a blob to the browser as a download. */
export function saveBlob(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = name
  anchor.click()
  // Revoked on the next tick rather than immediately: some browsers read the
  // object URL asynchronously after the click, and pulling it out from under
  // them cancels the download.
  setTimeout(() => URL.revokeObjectURL(url), 0)
}
