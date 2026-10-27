import { QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { FindPanel } from '@/components/editor/FindPanel'
import { pageKey, pagesKey, projectKey, queryClient } from '@/lib/queries'
import { useKoharuStore } from '@/lib/store'
import * as canvasRuntime from '@koharu/bridge/canvas'
import { commands, type TranslationMatch } from '@koharu/bridge/protocol'

const size = { width: 1000, height: 1500 }

const matches: TranslationMatch[] = [
  { page: 'page-1', page_index: 0, layer: 'layer-a', text: 'Onii-chan!', count: 1 },
  { page: 'page-3', page_index: 2, layer: 'layer-b', text: 'Thanks, onii-chan.', count: 1 },
  { page: 'page-3', page_index: 2, layer: 'layer-c', text: 'Onii-chan, onii-chan…', count: 2 },
]

function installProject() {
  queryClient.setQueryData(projectKey, {
    name: 'Book',
    revision: 1,
    active_page: 'page-1',
    can_undo: false,
    can_redo: false,
  })
  queryClient.setQueryData(
    pagesKey,
    ['page-1', 'page-2', 'page-3'].map((id, index) => ({
      id,
      label: `page_00${index + 1}.png`,
      size,
      source_asset: null,
      chapter_start: false,
      layer_count: 1,
    })),
  )
  queryClient.setQueryData(pageKey, {
    id: 'page-1',
    label: 'page_001.png',
    size,
    layers: [],
    regions: [],
  })
}

function renderPanel() {
  return render(
    <QueryClientProvider client={queryClient}>
      <FindPanel />
    </QueryClientProvider>,
  )
}

describe('FindPanel', () => {
  it('lists matches by page and replaces them all as one command', async () => {
    installProject()
    const find = vi.spyOn(commands, 'findTranslations').mockResolvedValue(matches)
    const replace = vi.spyOn(commands, 'replaceTranslations').mockResolvedValue(4)
    renderPanel()

    fireEvent.change(screen.getByRole('textbox', { name: 'Find in translations' }), {
      target: { value: 'onii-chan' },
    })
    await waitFor(() => expect(find).toHaveBeenCalledWith('onii-chan', false))
    expect(await screen.findByText('4 matches on 2 page(s)')).toBeInTheDocument()
    expect(screen.getByText('Page 1 · page_001.png')).toBeInTheDocument()
    expect(screen.getByText('Page 3 · page_003.png')).toBeInTheDocument()

    fireEvent.change(screen.getByRole('textbox', { name: 'Replace with' }), {
      target: { value: 'big bro' },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Replace all' }))
    await waitFor(() => expect(replace).toHaveBeenCalledWith('onii-chan', 'big bro', false, null))
  })

  it('searches with match case and replaces a single layer', async () => {
    installProject()
    const find = vi.spyOn(commands, 'findTranslations').mockResolvedValue(matches.slice(1, 2))
    const replace = vi.spyOn(commands, 'replaceTranslations').mockResolvedValue(1)
    renderPanel()

    fireEvent.click(screen.getByRole('button', { name: 'Match case' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'Find in translations' }), {
      target: { value: 'onii' },
    })
    await waitFor(() => expect(find).toHaveBeenCalledWith('onii', true))
    fireEvent.click(await screen.findByRole('button', { name: 'Replace' }))
    await waitFor(() => expect(replace).toHaveBeenCalledWith('onii', '', true, ['layer-b']))
  })

  it('opens the page of a result and selects its text layer', async () => {
    installProject()
    vi.spyOn(commands, 'findTranslations').mockResolvedValue(matches)
    vi.spyOn(canvasRuntime, 'showCanvasPage').mockReturnValue(false)
    const selectPage = vi.spyOn(commands, 'selectPage').mockResolvedValue({
      project: {
        name: 'Book',
        revision: 1,
        active_page: 'page-3',
        can_undo: false,
        can_redo: false,
      },
      page: { id: 'page-3', label: 'page_003.png', size, layers: [], regions: [] },
    })
    renderPanel()

    fireEvent.change(screen.getByRole('textbox', { name: 'Find in translations' }), {
      target: { value: 'thanks' },
    })
    fireEvent.click(await screen.findByRole('button', { name: /Thanks,/ }))

    await waitFor(() => expect(selectPage).toHaveBeenCalledWith('page-3'))
    await waitFor(() => expect(useKoharuStore.getState().selectedLayers).toEqual(['layer-b']))
    expect(useKoharuStore.getState().selectedPages).toEqual(['page-3'])
  })

  it('opens the Find tab with Ctrl+F', () => {
    installProject()
    renderPanel()
    expect(useKoharuStore.getState().sidebarPanel).toBe('properties')

    fireEvent.keyDown(window, { key: 'f', ctrlKey: true })

    expect(useKoharuStore.getState().sidebarPanel).toBe('find')
  })
})
