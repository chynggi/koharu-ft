import { afterEach, describe, expect, it, vi } from 'vitest'

import { finishExport, runExport, runImport } from '@/lib/transfer'
import { commands, type ExportFormat } from '@koharu/bridge/protocol'

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

})

describe('runExport', () => {
  const options = { formats: ['png'] as ExportFormat[], pattern: '{index:04}_{label}', subfolders: false }

  it('데스크톱에서는 대화상자 명령만 부르고 job id를 돌려준다', async () => {
    pretendEmbedded()
    const dialog = vi.spyOn(commands, 'exportPagesDialog').mockResolvedValue('job-1')
    const download = vi.spyOn(commands, 'exportPagesDownload')

    await expect(runExport(['page-1'], options)).resolves.toBe('job-1')
    expect(dialog).toHaveBeenCalledWith(['page-1'], options)
    expect(download).not.toHaveBeenCalled()
  })

  it('브라우저에서는 다운로드 job을 시작하고 아카이브는 아직 받지 않는다', async () => {
    // No `__TAURI_INTERNALS__`: a plain browser.
    const download = vi.spyOn(commands, 'exportPagesDownload').mockResolvedValue('job-2')
    const archive = vi.spyOn(commands, 'getExportArchive')

    await expect(runExport([], options)).resolves.toBe('job-2')
    expect(download).toHaveBeenCalledWith([], options)
    expect(archive).not.toHaveBeenCalled()
  })
})

describe('finishExport', () => {
  it('브라우저에서만 아카이브를 받아 저장한다', async () => {
    const archive = vi.spyOn(commands, 'getExportArchive').mockResolvedValue(new Blob(['zip']))
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined)

    await finishExport('job-3')
    expect(archive).toHaveBeenCalledWith('job-3')
  })

  it('데스크톱에서는 아무것도 하지 않는다', async () => {
    pretendEmbedded()
    const archive = vi.spyOn(commands, 'getExportArchive')

    await finishExport('job-4')
    expect(archive).not.toHaveBeenCalled()
  })
})
