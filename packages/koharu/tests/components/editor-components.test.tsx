import { QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render as testingRender, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { StrictMode, type ReactNode } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { TitleBar } from '@/components/app/TitleBar'
import { WindowControls } from '@/components/app/WindowChrome'
import { FontPicker } from '@/components/controls/FontPicker'
import { PageRail } from '@/components/editor/PageRail'
import { FONT_FAVORITES_STORAGE_KEY, useFontFavorites } from '@/lib/fontFavorites'
import { pageKey, projectKey, queryClient } from '@/lib/queries'
import { useKoharuStore } from '@/lib/store'
import { commands } from '@koharu/ipc/protocol'

const nativeWindow = vi.hoisted(() => ({
  close: vi.fn(async () => undefined),
  isMaximized: vi.fn(async () => false),
  minimize: vi.fn(async () => undefined),
  onResized: vi.fn(async () => () => undefined),
  startResizeDragging: vi.fn(async () => undefined),
  toggleMaximize: vi.fn(async () => undefined),
}))
const nativeOpenUrl = vi.hoisted(() => vi.fn(async () => undefined))
const nativeGetVersion = vi.hoisted(() => vi.fn(async () => '0.62.0'))

vi.mock('@tauri-apps/api/window', () => ({ getCurrentWindow: () => nativeWindow }))
vi.mock('@tauri-apps/api/app', () => ({ getVersion: nativeGetVersion }))
vi.mock('@tauri-apps/plugin-opener', () => ({ openUrl: nativeOpenUrl }))

function render(ui: ReactNode) {
  return testingRender(<QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>)
}

describe('window chrome', () => {
  it('registers one native resize listener in React strict mode', async () => {
    const unlisten = vi.fn()
    nativeWindow.onResized.mockClear()
    nativeWindow.onResized.mockResolvedValueOnce(unlisten)

    const view = render(
      <StrictMode>
        <WindowControls />
      </StrictMode>,
    )

    await waitFor(() => expect(nativeWindow.onResized).toHaveBeenCalledTimes(1))
    view.unmount()
    await waitFor(() => expect(unlisten).toHaveBeenCalledTimes(1))
  })

  it('starts native resize dragging from every frameless window edge', () => {
    nativeWindow.startResizeDragging.mockClear()
    const view = render(<WindowControls />)
    const directions = [
      'North',
      'South',
      'East',
      'West',
      'NorthEast',
      'NorthWest',
      'SouthEast',
      'SouthWest',
    ]

    for (const direction of directions) {
      fireEvent.pointerDown(
        view.container.querySelector(`[data-window-resize-handle="${direction}"]`)!,
        { button: 0 },
      )
    }

    expect(nativeWindow.startResizeDragging.mock.calls).toEqual(
      directions.map((direction) => [direction]),
    )
  })

  it('removes resize handles while the window is maximized', async () => {
    nativeWindow.isMaximized.mockResolvedValueOnce(true)
    const view = render(<WindowControls />)

    await waitFor(() =>
      expect(view.container.querySelector('[data-window-resize-handle]')).not.toBeInTheDocument(),
    )
  })

  it('opens community links through the Tauri opener plugin', async () => {
    nativeOpenUrl.mockClear()
    const user = userEvent.setup()
    render(<TitleBar />)

    await user.click(screen.getByRole('menuitem', { name: 'Help' }))
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Discord' }))
    expect(nativeOpenUrl).toHaveBeenLastCalledWith('https://discord.gg/mHvHkxGnUY')

    await user.click(screen.getByRole('menuitem', { name: 'Help' }))
    fireEvent.click(await screen.findByRole('menuitem', { name: 'GitHub' }))
    expect(nativeOpenUrl).toHaveBeenLastCalledWith('https://github.com/koharu-rs/koharu')
  })

  it('shows the current version and author in About', async () => {
    const user = userEvent.setup()
    render(<TitleBar />)

    await user.click(screen.getByRole('menuitem', { name: 'Help' }))
    await user.click(await screen.findByRole('menuitem', { name: 'About' }))

    expect(await screen.findByRole('heading', { name: 'Koharu' })).toBeInTheDocument()
    expect(await screen.findByText('0.62.0')).toBeInTheDocument()
    expect(screen.getByText('Mayo Takanashi')).toBeInTheDocument()
    expect(nativeGetVersion).toHaveBeenCalledTimes(1)
  })
})

describe('font favorites', () => {
  it('pins a favorite without choosing it', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    const families = [
      { name: 'Alpha', faces: [] },
      { name: 'Zeta', faces: [] },
    ]
    try {
      render(
        <FontPicker ariaLabel='Font family' value='' families={families} onChange={onChange} />,
      )
      await user.click(screen.getByRole('button', { name: 'Font family' }))
      const alpha = await screen.findByRole('button', { name: 'Alpha' })
      const zeta = screen.getByRole('button', { name: 'Zeta' })
      expect(alpha.compareDocumentPosition(zeta) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()

      await user.click(screen.getByRole('button', { name: /Zeta.*favorite|favorite.*Zeta/i }))
      expect(zeta.compareDocumentPosition(alpha) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
      expect(JSON.parse(localStorage.getItem(FONT_FAVORITES_STORAGE_KEY) ?? 'null')).toEqual([
        'Zeta',
      ])
      expect(onChange).not.toHaveBeenCalled()

      await user.click(zeta)
      expect(onChange).toHaveBeenCalledWith('Zeta')
    } finally {
      useFontFavorites.setState({ favorites: [] })
      localStorage.removeItem(FONT_FAVORITES_STORAGE_KEY)
    }
  })
})

describe('page shortcuts', () => {
  it('uses the rail selection and leaves arrows to a focused field', () => {
    const show = vi.spyOn(commands, 'show').mockResolvedValue(null)
    const first = { id: 'first', size: { width: 100, height: 100 }, background: null, layers: [] }
    const second = { ...first, id: 'second' }
    queryClient.setQueryData(projectKey, {
      name: 'Book',
      page: 'first',
      pages: [first, second],
      revisions: {},
    })
    queryClient.setQueryData(pageKey, first)
    useKoharuStore.setState({ selectedPages: ['first'], selectedLayers: ['layer'] })
    render(<PageRail />)

    const field = document.body.appendChild(document.createElement('input'))
    field.focus()
    fireEvent.keyDown(field, { key: 'ArrowRight' })
    expect(show).not.toHaveBeenCalled()

    field.remove()
    fireEvent.keyDown(document.body, { key: 'ArrowRight' })
    expect(show).toHaveBeenCalledWith('second')
    expect(useKoharuStore.getState().selectedPages).toEqual(['second'])
    expect(useKoharuStore.getState().selectedLayers).toEqual([])
  })
})
