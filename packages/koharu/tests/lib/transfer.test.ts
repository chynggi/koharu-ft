import { afterEach, describe, expect, it, vi } from 'vitest'

import { runExport, runImport } from '@/lib/transfer'
import { commands } from '@koharu/bridge/protocol'

function pretendEmbedded(): void {
  Object.defineProperty(window, '__TAURI_INTERNALS__', { value: {}, configurable: true })
}

afterEach(() => {
  Reflect.deleteProperty(window, '__TAURI_INTERNALS__')
  vi.restoreAllMocks()
})

describe('page transfer', () => {
  it('imports through the native dialog in the desktop window', async () => {
    pretendEmbedded()
    const dialog = vi.spyOn(commands, 'importPagesDialog').mockResolvedValue([])
    const upload = vi.spyOn(commands, 'importPagesUpload')

    await runImport('files')

    expect(dialog).toHaveBeenCalledWith('files')
    expect(upload).not.toHaveBeenCalled()
  })

  it('exports through the native dialog in the desktop window', async () => {
    pretendEmbedded()
    const dialog = vi.spyOn(commands, 'exportPagesDialog').mockResolvedValue(null)
    const downloadPages = vi.spyOn(commands, 'exportPagesDownload')

    await runExport(['page-1'], 'png')

    expect(dialog).toHaveBeenCalledWith(['page-1'], 'png')
    expect(downloadPages).not.toHaveBeenCalled()
  })

  it('exports as a downloaded archive in a browser', async () => {
    // No `__TAURI_INTERNALS__`: a plain browser, where a server-side folder
    // picker would choose a directory the user cannot reach.
    const archive = new Blob(['zip'], { type: 'application/zip' })
    const downloadPages = vi.spyOn(commands, 'exportPagesDownload').mockResolvedValue(archive)
    const dialog = vi.spyOn(commands, 'exportPagesDialog')
    // jsdom implements neither, and `saveBlob` needs both.
    const createObjectURL = vi.fn(() => 'blob:koharu')
    const revokeObjectURL = vi.fn()
    vi.stubGlobal('URL', { ...URL, createObjectURL, revokeObjectURL })
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined)

    await runExport(['page-1'], 'psd')

    expect(downloadPages).toHaveBeenCalledWith(['page-1'], 'psd')
    expect(dialog).not.toHaveBeenCalled()
    expect(createObjectURL).toHaveBeenCalledWith(archive)
    expect(click).toHaveBeenCalledTimes(1)
    vi.unstubAllGlobals()
  })
})
