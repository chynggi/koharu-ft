import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useTranslation } from 'react-i18next'
import { beforeAll, describe, expect, it, vi } from 'vitest'

import { ExportDialog } from '@/components/app/ExportDialog'

// `export.*` 키는 아직 번역 파일에 없다. 없으면 i18next가 키를 그대로
// 돌려주는데, 그러면 취소 버튼까지 `/export/i`에 걸려 쿼리가 모호해진다.
// 덮어쓰기를 끈 채로 채워 넣으므로 실제 번역이 들어오면 그쪽이 이긴다.
beforeAll(() => {
  const { i18n } = useTranslation()
  i18n.addResourceBundle(
    'en-US',
    'translation',
    {
      export: {
        title: 'Export pages',
        description: 'Render the finished pages to disk.',
        scope: 'Pages',
        scopeAll: 'All pages',
        scopeSelected: 'Selected pages',
        formats: 'Formats',
        pattern: 'Filename pattern',
        patternError: {
          unclosed: 'Unclosed brace.',
          width: 'Width must be a number.',
          token: 'Unknown token.',
          separator: 'Path separators are not allowed.',
        },
        subfolders: 'One subfolder per format',
        cancel: 'Cancel',
        start: 'Export',
      },
    },
    true,
    false,
  )
})

function open(onStart = vi.fn()) {
  render(
    <ExportDialog
      open
      onOpenChange={() => {}}
      pages={['a', 'b']}
      selected={['a']}
      onStart={onStart}
    />,
  )
  return onStart
}

describe('ExportDialog', () => {
  it('형식을 모두 끄면 시작할 수 없다', async () => {
    const user = userEvent.setup()
    open()
    await user.click(screen.getByRole('checkbox', { name: /png/i }))
    expect(screen.getByRole('button', { name: /export/i })).toBeDisabled()
  })

  it('잘못된 패턴은 오류를 보여주고 시작을 막는다', async () => {
    const user = userEvent.setup()
    open()
    const pattern = screen.getByLabelText(/pattern/i)
    await user.clear(pattern)
    await user.type(pattern, '{{page}')
    expect(screen.getByRole('button', { name: /export/i })).toBeDisabled()
    expect(screen.getByRole('alert')).toBeInTheDocument()
  })

  it('기본 패턴의 미리보기를 보여준다', () => {
    open()
    expect(screen.getByText('0001_page-01.png')).toBeInTheDocument()
  })

  it('고른 옵션으로 시작한다', async () => {
    const user = userEvent.setup()
    const onStart = open()
    await user.click(screen.getByRole('button', { name: /export/i }))
    expect(onStart).toHaveBeenCalledWith(['a'], {
      formats: ['png'],
      pattern: '{index:04}_{label}',
      subfolders: false,
    })
  })

  it('CBZ를 고르면 아카이브까지 함께 요청한다', async () => {
    const user = userEvent.setup()
    const onStart = open()
    await user.click(screen.getByRole('checkbox', { name: /cbz/i }))
    await user.click(screen.getByRole('button', { name: /export/i }))
    expect(onStart).toHaveBeenCalledWith(['a'], {
      formats: ['png', 'cbz'],
      pattern: '{index:04}_{label}',
      subfolders: false,
    })
  })

  it('CBZ만 고르면 항목 이름은 그대로 PNG로 미리보기한다', async () => {
    const user = userEvent.setup()
    const onStart = open()
    await user.click(screen.getByRole('checkbox', { name: /png/i }))
    await user.click(screen.getByRole('checkbox', { name: /cbz/i }))
    expect(screen.getByText('0001_page-01.png')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /export/i }))
    expect(onStart).toHaveBeenCalledWith(['a'], {
      formats: ['cbz'],
      pattern: '{index:04}_{label}',
      subfolders: false,
    })
  })

  it('하위 폴더는 아카이브를 세지 않고 페이지 파일 형식만 센다', async () => {
    const user = userEvent.setup()
    open()
    await user.click(screen.getByRole('checkbox', { name: /cbz/i }))
    // PNG와 CBZ를 골랐지만 페이지 파일 형식은 하나뿐이라 나눌 것이 없다.
    // Base UI 체크박스는 네이티브 disabled가 아니라 aria-disabled를 쓴다.
    const subfolders = screen.getByRole('checkbox', { name: /separate folders/i })
    expect(subfolders).toHaveAttribute('aria-disabled', 'true')
    await user.click(screen.getByRole('checkbox', { name: /psd/i }))
    expect(subfolders).not.toHaveAttribute('aria-disabled', 'true')
  })
})
