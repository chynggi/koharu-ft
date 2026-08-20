'use client'

import {
  commands,
  type EntityId,
  type ExportFormat,
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

/** Export pages by whichever route this client can actually use. */
export async function runExport(pages: EntityId[], format: ExportFormat): Promise<void> {
  if (isEmbedded()) {
    await commands.exportPagesDialog(pages, format)
    return
  }
  saveBlob(await commands.exportPagesDownload(pages, format), 'koharu-export.zip')
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
