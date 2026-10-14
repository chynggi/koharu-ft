import { QueryClientProvider } from '@tanstack/react-query'
import {
  act,
  fireEvent,
  render as testingRender,
  renderHook,
  screen,
  waitFor,
} from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ThemeProvider } from 'next-themes'
import { StrictMode, type ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { TitleBar } from '@/components/app/TitleBar'
import { WindowControls } from '@/components/app/WindowChrome'
import { ActivityCenter } from '@/components/editor/ActivityCenter'
import { CanvasCommandBar } from '@/components/editor/CanvasCommandBar'
import { Inspector } from '@/components/editor/Inspector'
import { PageRail } from '@/components/editor/PageRail'
import { ResourceMonitor } from '@/components/editor/ResourceMonitor'
import { StatusBar } from '@/components/editor/StatusBar'
import { ToolBar } from '@/components/editor/ToolBar'
import { ProviderPreferences } from '@/components/preferences/ProviderPreferences'
import { SettingsPage } from '@/components/preferences/SettingsPage'
import {
  fontsKey,
  pageKey,
  pagesKey,
  preparedPageKey,
  projectKey,
  queryClient,
  useCommand,
} from '@/lib/queries'
import { fillKind } from '@/lib/shapes'
import { useKoharuStore } from '@/lib/store'
import * as canvasRuntime from '@koharu/bridge/canvas'
import {
  commands,
  type Layer,
  type PageSummary,
  type PatternSpec,
  type Preferences,
  type ProjectInfo,
  type ShapeFill,
} from '@koharu/bridge/protocol'
import { TooltipProvider } from '@koharu/ui/components/tooltip'

const nativeWindow = vi.hoisted(() => ({
  close: vi.fn(async () => undefined),
  isMaximized: vi.fn(async () => false),
  minimize: vi.fn(async () => undefined),
  onResized: vi.fn(async () => () => undefined),
  startResizeDragging: vi.fn(async () => undefined),
  toggleMaximize: vi.fn(async () => undefined),
}))

vi.mock('@tauri-apps/api/window', () => ({ getCurrentWindow: () => nativeWindow }))

const emptyCredential = () => ({
  configured: false,
  editable: true,
  environment_variable: null,
  value: null,
  clear: false,
})

const textLayer: Layer = {
  type: 'text',
  angle_degrees: 0,
  id: 'element',
  parent: 'page',
  geometry: {
    points: [
      { x: 10, y: 20 },
      { x: 110, y: 20 },
      { x: 110, y: 70 },
      { x: 10, y: 70 },
    ],
  },
  visibility: { visible: true, opacity: 1 },
  content: {
    id: 'content',
    source: { text: 'こんにちは', language: 'ja' },
    translation: { text: 'Hello', language: null },
    role: null,
    source_region: null,
  },
  typography: {
    preferred_font: 'Noto Sans',
    font_weight: 400,
    font_style: 'normal',
    size: null,
    auto_fit: true,
    color: [0, 0, 0, 255],
    stroke_color: [255, 255, 255, 255],
    stroke_width: 0,
    alignment: 'Center',
    writing_mode: null,
  },
  layout: 'paragraph',
  automatic_region: null,
}

const secondLayer: Layer = {
  ...textLayer,
  id: 'second',
  content: {
    ...textLayer.content,
    id: 'second-content',
    translation: { text: 'Second', language: null },
  },
}

const thirdLayer: Layer = {
  ...textLayer,
  id: 'third',
  content: {
    ...textLayer.content,
    id: 'third-content',
    translation: { text: 'Third', language: null },
  },
}

const shapeLayer: Layer = {
  type: 'shape',
  id: 'shape',
  parent: 'page',
  geometry: {
    points: [
      { x: 10, y: 20 },
      { x: 110, y: 20 },
      { x: 110, y: 70 },
      { x: 10, y: 70 },
    ],
  },
  angle_degrees: 0,
  visibility: { visible: true, opacity: 1 },
  name: 'Rectangle 1',
  shape: { rectangle: { corner_radius: 0 } },
  style: { fill: { solid: [255, 255, 255, 255] }, stroke: null },
  rasterized: false,
}

const gradientShapeLayer: Layer = {
  type: 'shape',
  id: 'shape-gradient',
  parent: 'page',
  geometry: {
    points: [
      { x: 10, y: 20 },
      { x: 110, y: 20 },
      { x: 110, y: 70 },
      { x: 10, y: 70 },
    ],
  },
  angle_degrees: 0,
  visibility: { visible: true, opacity: 1 },
  name: 'Rectangle 2',
  shape: { rectangle: { corner_radius: 0 } },
  style: {
    fill: {
      gradient: {
        kind: { linear: { start: [0, 0], end: [1, 0] } },
        stops: [
          { offset: 0, color: [255, 255, 255, 255] },
          { offset: 1, color: [0, 0, 0, 255] },
        ],
        extend: 'pad',
      },
    },
    stroke: null,
  },
  rasterized: false,
}

const patternSpec: PatternSpec = {
  repeat: 'tile',
  origin: [0, 0],
  scale: 0.25,
  rotation_degrees: 0,
  fallback: [255, 255, 255, 255],
}

const patternShapeLayer: Layer = {
  ...gradientShapeLayer,
  id: 'shape-pattern',
  name: 'Rectangle 3',
  style: { fill: { pattern: patternSpec }, stroke: null },
}

const artworkLayer: Layer = {
  type: 'artwork',
  id: 'artwork',
  parent: 'page',
  geometry: {
    points: [
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: 10, y: 10 },
      { x: 0, y: 10 },
    ],
  },
  visibility: { visible: true, opacity: 1 },
  image: 'source',
}

const rasterLayer: Layer = {
  type: 'raster',
  id: 'raster',
  parent: 'page',
  visibility: { visible: true, opacity: 1 },
  image: 'paint',
  name: 'Paint 1',
  kind: 'paint',
}

const preferences: Preferences = {
  pipeline: {
    detection: { model: 'koharu-layout-rfdetr-seg-2xl' },
    ocr: { model: 'paddleocr-vl-1.6' },
    translation: {
      model: {
        provider: 'local',
        model: 'gemma4-e2b-it',
        quantization: null,
        vision: true,
        reasoning: true,
      },
      generation: { vision: true, reasoning: false },
      target_language: 'en-US',
      instructions: null,
    },
    inpainting: { model: 'lama' },
    processor: {},
  },
  providers: {
    entries: [
      {
        name: 'Local',
        config: { provider: 'local', settings: {} },
        credential: null,
      },
      {
        name: 'OpenAI-compatible',
        config: {
          provider: 'openai-compatible',
          settings: { base_url: 'http://localhost:11434/v1' },
        },
        credential: emptyCredential(),
      },
      {
        name: 'LM Studio',
        config: { provider: 'lm-studio', settings: { base_url: 'http://localhost:1234' } },
        credential: emptyCredential(),
      },
      {
        name: 'DeepL',
        config: { provider: 'deepl', settings: { base_url: null } },
        credential: emptyCredential(),
      },
    ],
  },
  typesetting: {
    font_families: ['Noto Sans'],
  },
  languages: [
    { tag: 'en-US', name: 'English' },
    { tag: 'ja-JP', name: 'Japanese' },
  ],
}

function installProject() {
  const page = {
    id: 'page',
    label: 'Page 1',
    size: { width: 1000, height: 1500 },
    layers: [textLayer],
    regions: [],
  }
  queryClient.setQueryData(projectKey, {
    name: 'Book',
    revision: 1,
    active_page: 'page',
    can_undo: true,
    can_redo: false,
  })
  queryClient.setQueryData(pagesKey, [
    {
      id: 'page',
      label: 'Page 1',
      size: { width: 1000, height: 1500 },
      source_asset: 'source',
      layer_count: 1,
    },
  ])
  queryClient.setQueryData(pageKey, page)
  useKoharuStore.setState({
    preferences,
    translationModels: [
      {
        provider: 'local',
        model: 'gemma4-e2b-it',
        name: 'Gemma 4 E2B Instruct',
        quantizations: [],
        vision: true,
        reasoning: true,
      },
    ],
    selectedPages: ['page'],
    selectedLayers: ['element'],
    layerFrames: {
      element: { x: 10, y: 20, width: 100, height: 50, angle_degrees: 0 },
    },
  })
  vi.spyOn(commands, 'getPreferences').mockImplementation(
    async () => useKoharuStore.getState().preferences!,
  )
  vi.spyOn(commands, 'getTranslationModels').mockImplementation(async () => [
    ...useKoharuStore.getState().translationModels,
  ])
}

function render(ui: ReactNode) {
  return testingRender(<QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>)
}

function SettingsNavigationHarness() {
  const settingsOpen = useKoharuStore((state) => state.settingsOpen)
  return settingsOpen ? <SettingsPage /> : <CanvasCommandBar />
}

describe('greenfield editor', () => {
  beforeEach(() => {
    Object.defineProperty(window, '__TAURI_INTERNALS__', {
      configurable: true,
      value: {},
    })
  })

  afterEach(() => {
    delete (window as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__
  })

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

  it('shows import activity and prevents duplicate imports', async () => {
    const user = userEvent.setup()
    installProject()
    // This test is about the in-flight UI, not the transport, so it takes the
    // desktop route: without `__TAURI_INTERNALS__` the import would open a
    // browser file picker, which never resolves under jsdom.
    Object.defineProperty(window, '__TAURI_INTERNALS__', {
      value: {},
      configurable: true,
    })
    let finishImport: (() => void) | undefined
    const importPages = vi.spyOn(commands, 'importPagesDialog').mockImplementation(
      () =>
        new Promise<PageSummary[]>((resolve) => {
          finishImport = () => resolve([])
        }),
    )
    render(
      <>
        <TitleBar />
        <PageRail />
        <ActivityCenter />
      </>,
    )

    expect(screen.getByText('/')).toHaveClass('mx-2')
    expect(screen.queryByRole('button', { name: 'Import pages' })).not.toBeInTheDocument()
    await user.click(screen.getByRole('menuitem', { name: 'File' }))
    await user.hover(await screen.findByRole('menuitem', { name: 'Import Pages' }))
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Files…' }))

    expect(await screen.findByRole('status')).toHaveTextContent('Importing pages…')
    expect(screen.getByRole('complementary', { name: 'Activity' })).toBeInTheDocument()
    expect(importPages).toHaveBeenCalledTimes(1)

    await user.click(screen.getByRole('menuitem', { name: 'File' }))
    expect(await screen.findByRole('menuitem', { name: 'Importing pages…' })).toHaveAttribute(
      'aria-disabled',
      'true',
    )

    finishImport?.()
    await waitFor(() => expect(screen.queryByRole('status')).not.toBeInTheDocument())
    Reflect.deleteProperty(window, '__TAURI_INTERNALS__')
  })

  it('opens community links in a new tab', async () => {
    const openSpy = vi.spyOn(window, 'open').mockReturnValue(null)
    const user = userEvent.setup()
    render(<TitleBar />)

    await user.click(screen.getByRole('menuitem', { name: 'Help' }))
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Discord' }))
    expect(openSpy).toHaveBeenLastCalledWith(
      'https://discord.gg/mHvHkxGnUY',
      '_blank',
      'noopener,noreferrer',
    )

    await user.click(screen.getByRole('menuitem', { name: 'Help' }))
    fireEvent.click(await screen.findByRole('menuitem', { name: 'GitHub' }))
    expect(openSpy).toHaveBeenLastCalledWith(
      'https://github.com/koharu-rs/koharu',
      '_blank',
      'noopener,noreferrer',
    )
  })

  it('shows the current version and author in About', async () => {
    const getMeta = vi
      .spyOn(commands, 'getMeta')
      .mockResolvedValue({ name: 'koharu', version: '0.62.0' })
    const user = userEvent.setup()
    render(<TitleBar />)

    await user.click(screen.getByRole('menuitem', { name: 'Help' }))
    await user.click(await screen.findByRole('menuitem', { name: 'About' }))

    expect(await screen.findByRole('heading', { name: 'Koharu' })).toBeInTheDocument()
    expect(await screen.findByText('0.62.0')).toBeInTheDocument()
    expect(screen.getByText('Mayo Takanashi')).toBeInTheDocument()
    expect(getMeta).toHaveBeenCalledTimes(1)
  })

  it('shows how many pages are selected above the filter', async () => {
    installProject()
    act(() => {
      useKoharuStore.setState({ selectedPages: ['page-1'] })
    })
    render(<PageRail />)

    // One page behaves like acting on the active page, so it is not worth
    // announcing.
    expect(screen.queryByRole('status')).not.toBeInTheDocument()

    act(() => {
      useKoharuStore.setState({ selectedPages: ['page-1', 'page-2', 'page-3'] })
    })

    // A status region, so the count reaches assistive technology when it
    // changes rather than only being visible.
    expect(await screen.findByRole('status')).toHaveTextContent('3 selected')
  })

  it('loads page thumbnails into the filmstrip', async () => {
    installProject()
    const thumbnail = vi
      .spyOn(commands, 'getThumbnail')
      .mockResolvedValue(new Uint8Array([1]).buffer)
    render(<PageRail />)
    await waitFor(() => expect(thumbnail).toHaveBeenCalledWith('page'))
    expect(await screen.findByRole('img', { name: 'Page 1' })).toHaveAttribute(
      'src',
      'blob:koharu-thumbnail',
    )
    expect(screen.queryByText('01')).not.toBeInTheDocument()
  })

  it('keeps rapid page switches on the latest native selection', async () => {
    installProject()
    const pages = [
      { id: 'page', label: 'Page 1', size: { width: 1000, height: 1500 }, layers: [], regions: [] },
      {
        id: 'page-2',
        label: 'Page 2',
        size: { width: 1000, height: 1500 },
        layers: [],
        regions: [],
      },
      {
        id: 'page-3',
        label: 'Page 3',
        size: { width: 1000, height: 1500 },
        layers: [],
        regions: [],
      },
    ]
    queryClient.setQueryData(
      pagesKey,
      pages.map((page) => ({
        id: page.id,
        label: page.label,
        size: page.size,
        source_asset: null,
        layer_count: 0,
      })),
    )
    vi.spyOn(canvasRuntime, 'showCanvasPage').mockReturnValue(false)
    const pending = new Map<
      string,
      (selection: Awaited<ReturnType<typeof commands.selectPage>>) => void
    >()
    const selectPage = vi.spyOn(commands, 'selectPage').mockImplementation(
      (page) =>
        new Promise((resolve) => {
          pending.set(page, resolve)
        }),
    )
    render(<PageRail />)

    fireEvent.click(screen.getByText('Page 2').closest('article')!)
    expect(selectPage).toHaveBeenLastCalledWith('page-2')

    fireEvent.click(screen.getByText('Page 3').closest('article')!)
    expect(selectPage).toHaveBeenLastCalledWith('page-3')

    await act(async () => {
      pending.get('page-3')!({
        project: {
          name: 'Book',
          revision: 1,
          active_page: 'page-3',
          can_undo: true,
          can_redo: false,
        },
        page: pages[2]!,
      })
      await Promise.resolve()
    })
    await act(async () => {
      pending.get('page-2')!({
        project: {
          name: 'Book',
          revision: 1,
          active_page: 'page-2',
          can_undo: true,
          can_redo: false,
        },
        page: pages[1]!,
      })
      await Promise.resolve()
    })

    expect(queryClient.getQueryData<{ active_page: string }>(projectKey)?.active_page).toBe(
      'page-3',
    )
    expect(queryClient.getQueryData<{ id: string }>(pageKey)?.id).toBe('page-3')
  })

  it('lets a cached canvas frame paint before synchronizing native page state', async () => {
    installProject()
    queryClient.setQueryData(pagesKey, [
      ...(queryClient.getQueryData<PageSummary[]>(pagesKey) ?? []),
      {
        id: 'page-2',
        label: 'Page 2',
        size: { width: 1000, height: 1500 },
        source_asset: null,
        layer_count: 0,
      },
    ])
    vi.spyOn(canvasRuntime, 'showCanvasPage').mockReturnValue(true)
    const selectPage = vi.spyOn(commands, 'selectPage')
    let paint: FrameRequestCallback | undefined
    vi.spyOn(window, 'requestAnimationFrame').mockImplementation((callback) => {
      paint = callback
      return 1
    })
    render(<PageRail />)

    fireEvent.click(screen.getByText('Page 2').closest('article')!)

    expect(selectPage).not.toHaveBeenCalled()
    await act(async () => {
      paint?.(performance.now())
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    expect(selectPage).toHaveBeenCalledWith('page-2')
  })

  it('prefetches an inactive page on pointer intent and caches its preparation', async () => {
    installProject()
    const page = {
      id: 'page-2',
      label: 'Page 2',
      size: { width: 1000, height: 1500 },
      layers: [],
      regions: [],
    }
    queryClient.setQueryData(pagesKey, [
      ...(queryClient.getQueryData<PageSummary[]>(pagesKey) ?? []),
      {
        id: page.id,
        label: page.label,
        size: page.size,
        source_asset: null,
        layer_count: 0,
      },
    ])
    const prepared = { revision: 1, page }
    const prefetch = vi.spyOn(canvasRuntime, 'prefetchCanvasPages').mockResolvedValue([prepared])
    const selectPage = vi.spyOn(commands, 'selectPage')
    render(<PageRail />)

    fireEvent.pointerEnter(screen.getByText('Page 2').closest('article')!)

    await waitFor(() => expect(prefetch).toHaveBeenCalledWith(['page-2']))
    await waitFor(() =>
      expect(queryClient.getQueryData(preparedPageKey('page-2'))).toEqual(prepared),
    )
    expect(selectPage).not.toHaveBeenCalled()
  })

  it('deduplicates page intent within a project revision and retries on a newer revision', async () => {
    installProject()
    const page = {
      id: 'page-2',
      label: 'Page 2',
      size: { width: 1000, height: 1500 },
      layers: [],
      regions: [],
    }
    queryClient.setQueryData(pagesKey, [
      ...(queryClient.getQueryData<PageSummary[]>(pagesKey) ?? []),
      {
        id: page.id,
        label: page.label,
        size: page.size,
        source_asset: null,
        layer_count: 0,
      },
    ])
    const prefetch = vi
      .spyOn(canvasRuntime, 'prefetchCanvasPages')
      .mockResolvedValueOnce([{ revision: 1, page }])
      .mockResolvedValueOnce([{ revision: 2, page }])
    render(<PageRail />)
    const item = screen.getByText('Page 2').closest('article')!

    fireEvent.pointerEnter(item)
    fireEvent.focus(screen.getByRole('button', { name: 'Actions for Page 2' }))
    await waitFor(() => expect(prefetch).toHaveBeenCalledTimes(1))
    fireEvent.pointerEnter(item)
    expect(prefetch).toHaveBeenCalledTimes(1)

    queryClient.setQueryData(projectKey, {
      ...queryClient.getQueryData<ProjectInfo>(projectKey)!,
      revision: 2,
    })
    fireEvent.focus(item)

    await waitFor(() => expect(prefetch).toHaveBeenCalledTimes(2))
    expect(prefetch).toHaveBeenLastCalledWith(['page-2'])
  })

  it('switches tools and applies typography from the contextual inspector', async () => {
    const user = userEvent.setup()
    installProject()
    const setTypography = vi.spyOn(commands, 'setTypography').mockResolvedValue(null)
    render(
      <TooltipProvider>
        <ToolBar />
        <Inspector />
      </TooltipProvider>,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Brush' }))
    expect(useKoharuStore.getState().tool).toBe('draw')
    fireEvent.click(screen.getByRole('button', { name: 'Text' }))
    expect(screen.getByTestId('type-inspector')).toBeInTheDocument()
    expect(screen.getByTestId('type-font-picker')).toHaveTextContent('Noto Sans')
    expect(screen.getByTestId('type-size')).toHaveValue('')
    expect(screen.getByTestId('type-size')).toHaveAttribute('placeholder', 'Auto')
    expect(screen.getByRole('combobox', { name: 'Text direction' })).toHaveTextContent('Auto')
    await user.clear(screen.getByTestId('type-size'))
    await user.type(screen.getByTestId('type-size'), '18')
    await user.tab()
    await waitFor(() =>
      expect(setTypography).toHaveBeenCalledWith(
        expect.arrayContaining([
          expect.objectContaining({
            layer: 'element',
            typography: expect.objectContaining({ size: 18, writing_mode: null }),
          }),
        ]),
      ),
    )
  })

  it('defaults vertical text alignment to top and maps end to bottom', async () => {
    const user = userEvent.setup()
    installProject()
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: page.layers.map((layer) =>
        layer.type === 'text'
          ? {
              ...layer,
              typography: { ...layer.typography, alignment: null, writing_mode: 'Vertical' },
            }
          : layer,
      ),
    }))
    const setTypography = vi.spyOn(commands, 'setTypography').mockResolvedValue(null)
    render(<Inspector />)

    expect(screen.getByRole('button', { name: 'Align top' })).toHaveAttribute(
      'aria-pressed',
      'true',
    )
    await user.click(screen.getByRole('button', { name: 'Align bottom' }))
    await waitFor(() =>
      expect(setTypography).toHaveBeenCalledWith([
        expect.objectContaining({
          layer: 'element',
          typography: expect.objectContaining({ alignment: 'End', writing_mode: 'Vertical' }),
        }),
      ]),
    )
  })

  it('switches the shape kind and seeds a style that can draw it', async () => {
    const user = userEvent.setup()
    installProject()
    render(
      <TooltipProvider>
        <ToolBar />
      </TooltipProvider>,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Shape' }))
    expect(useKoharuStore.getState().tool).toBe('shape')

    await user.click(screen.getByRole('button', { name: 'Shape kind: Rectangle' }))
    // The popover heading must be a real string, not i18next's "returned an
    // object instead of string" diagnostic for a branch key.
    expect(await screen.findByText('Shape kind')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Line' }))

    expect(useKoharuStore.getState().shapeKind).toBe('line')
    expect(useKoharuStore.getState().shapeStyle).toEqual({
      fill: null,
      stroke: { color: [255, 255, 255, 255], width: 2, dash: null, cap: 'round', join: 'round' },
    })
  })

  it('offers a fill colour apart from the brush colour', async () => {
    const user = userEvent.setup()
    installProject()
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [...page.layers, rasterLayer],
    }))
    useKoharuStore.setState({
      selectedLayers: ['raster'],
      canvasRevision: 1,
      brush: { diameter: 48, color: '#FFFFFF', ramp: null },
    })
    render(
      <TooltipProvider>
        <ToolBar />
      </TooltipProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Fill' }))
    expect(useKoharuStore.getState().tool).toBe('fill')
    // The brush-size slider is meaningless for filling, so the rail shows only
    // the colour well.
    expect(screen.queryByRole('button', { name: /Brush size/ })).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Fill color' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'Hex color code' }), {
      target: { value: '#123456' },
    })

    expect(useKoharuStore.getState().fillColor).toBe('#123456')
    expect(useKoharuStore.getState().brush.color).toBe('#FFFFFF')
  })

  it('builds a brush ramp and hides the flat colour it replaces', async () => {
    const user = userEvent.setup()
    installProject()
    useKoharuStore.setState({
      tool: 'draw',
      brush: { diameter: 48, color: '#204060', ramp: null },
    })
    render(
      <TooltipProvider>
        <ToolBar />
      </TooltipProvider>,
    )

    // No ramp: the single flat colour well is the only colour authority.
    expect(screen.getByRole('button', { name: 'Brush color' })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Brush gradient' }))
    await user.click(await screen.findByRole('radio', { name: 'Along stroke' }))

    // Once a ramp is on, the flat well must be gone: two controls writing
    // "what colour is this stroke" would disagree about which one Rust reads.
    expect(screen.queryByRole('button', { name: 'Brush color' })).not.toBeInTheDocument()
    expect(useKoharuStore.getState().brush.ramp).toEqual({
      kind: 'along_stroke',
      from: [32, 64, 96, 255],
      to: [0, 0, 0, 255],
    })

    expect(screen.getByRole('button', { name: 'Start color' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'End color' })).toBeInTheDocument()
    expect(screen.queryByRole('radiogroup', { name: 'Axis' })).not.toBeInTheDocument()

    await user.click(await screen.findByRole('radio', { name: 'Fixed axis' }))

    expect(useKoharuStore.getState().brush.ramp).toEqual({
      kind: 'fixed_axis',
      from: [32, 64, 96, 255],
      to: [0, 0, 0, 255],
      axis: 'horizontal',
    })
    expect(screen.getByRole('radiogroup', { name: 'Axis' })).toBeInTheDocument()

    await user.click(screen.getByRole('radio', { name: 'Vertical' }))
    expect(useKoharuStore.getState().brush.ramp).toMatchObject({ axis: 'vertical' })
  })

  it('turns the ramp back off without disturbing the flat brush colour', async () => {
    const user = userEvent.setup()
    installProject()
    useKoharuStore.setState({
      tool: 'draw',
      brush: {
        diameter: 48,
        color: '#204060',
        ramp: { kind: 'along_stroke', from: [255, 0, 0, 255], to: [0, 0, 255, 255] },
      },
    })
    render(
      <TooltipProvider>
        <ToolBar />
      </TooltipProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Brush gradient' }))
    await user.click(await screen.findByRole('radio', { name: 'None' }))

    expect(useKoharuStore.getState().brush.ramp).toBeNull()
    expect(useKoharuStore.getState().brush.color).toBe('#204060')
    expect(screen.getByRole('button', { name: 'Brush color' })).toBeInTheDocument()
  })

  it('disables the fill tool without a selected raster layer', () => {
    installProject()
    render(
      <TooltipProvider>
        <ToolBar />
      </TooltipProvider>,
    )

    expect(screen.getByRole('button', { name: 'Fill' })).toBeDisabled()
  })

  it('offers a pattern as a new-shape preset and reports it as active', async () => {
    // Without this the preset could only ever be solid or gradient, so
    // `previewPattern` and the pattern branch of a fresh shape were
    // unreachable: a user could choose a pattern only after drawing.
    installProject()
    const user = userEvent.setup()
    render(
      <TooltipProvider>
        <ToolBar />
      </TooltipProvider>,
    )

    // The shape presets only mount while the shape tool is active, and the
    // fill-kind radios live inside the kind popover.
    await user.click(screen.getByRole('button', { name: 'Shape' }))
    await user.click(screen.getByRole('button', { name: 'Shape kind: Rectangle' }))
    const pattern = await screen.findByRole('radio', { name: 'Pattern' })
    expect(pattern).toHaveAttribute('aria-checked', 'false')
    await user.click(pattern)

    // The store preset is the pattern kind, and the radio reflects it. The
    // elimination-style test this replaced reported a pattern as a *solid*,
    // which is why the assertion is on all three radios.
    expect(fillKind(useKoharuStore.getState().shapeStyle.fill)).toBe('pattern')
    expect(screen.getByRole('radio', { name: 'Pattern' })).toHaveAttribute('aria-checked', 'true')
    expect(screen.getByRole('radio', { name: 'Solid' })).toHaveAttribute('aria-checked', 'false')
    expect(screen.getByRole('radio', { name: 'Gradient' })).toHaveAttribute('aria-checked', 'false')
    // A preset carries no image, so it must still have a drawable colour.
    const spec = useKoharuStore.getState().shapeStyle.fill
    if (!spec || !('pattern' in spec)) throw new Error('expected a pattern preset')
    expect(spec.pattern.fallback).toEqual([255, 255, 255, 255])
  })

  it('edits a selected shape style and rasterizes it from the inspector', async () => {
    installProject()
    const user = userEvent.setup()
    const setShapeStyle = vi.spyOn(commands, 'setShapeStyle').mockResolvedValue(2)
    const rasterize = vi.spyOn(commands, 'rasterizeShapes').mockResolvedValue(3)
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [...page.layers, shapeLayer],
    }))
    useKoharuStore.setState({ selectedLayers: ['shape'], canvasRevision: 1 })
    render(<Inspector />)

    await user.click(screen.getByRole('button', { name: 'Fill' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'Hex color code' }), {
      target: { value: '#FF0000' },
    })
    await waitFor(() =>
      expect(setShapeStyle).toHaveBeenCalledWith(1, [
        { element: 'shape', style: expect.objectContaining({ fill: { solid: [255, 0, 0, 255] } }) },
      ]),
    )

    fireEvent.click(screen.getByRole('button', { name: 'Rasterize' }))
    await waitFor(() => expect(rasterize).toHaveBeenCalledWith(1, 'page', ['shape']))
  })

  it('switches a shape fill from solid to a gradient', async () => {
    installProject()
    const user = userEvent.setup()
    const setShapeStyle = vi.spyOn(commands, 'setShapeStyle').mockResolvedValue(2)
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [...page.layers, shapeLayer],
    }))
    useKoharuStore.setState({ selectedLayers: ['shape'], canvasRevision: 1 })
    render(<Inspector />)

    await user.click(screen.getByRole('radio', { name: 'Gradient' }))

    await waitFor(() =>
      expect(setShapeStyle).toHaveBeenCalledWith(1, [
        {
          element: 'shape',
          style: expect.objectContaining({
            fill: expect.objectContaining({ gradient: expect.anything() }),
          }),
        },
      ]),
    )
  })

  it('keeps a gradient fill a gradient when a stop colour changes', async () => {
    // The fill colour well used to write `{ solid }` unconditionally, which
    // silently downgraded a gradient. Editing a stop must stay in the gradient
    // branch and must not disturb the other stop.
    installProject()
    const user = userEvent.setup()
    const setShapeStyle = vi.spyOn(commands, 'setShapeStyle').mockResolvedValue(2)
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [...page.layers, gradientShapeLayer],
    }))
    useKoharuStore.setState({ selectedLayers: ['shape-gradient'], canvasRevision: 1 })
    render(<Inspector />)

    await user.click(screen.getByRole('button', { name: 'Stop 1 color' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'Hex color code' }), {
      target: { value: '#00FF00' },
    })

    await waitFor(() => expect(setShapeStyle).toHaveBeenCalled())
    const fill = setShapeStyle.mock.calls.at(-1)?.[1][0]?.style.fill
    expect(fill).not.toHaveProperty('solid')
    expect(fill).toEqual({
      gradient: expect.objectContaining({
        stops: [
          { offset: 0, color: [0, 255, 0, 255] },
          { offset: 1, color: [0, 0, 0, 255] },
        ],
      }),
    })
  })

  // This test clicks "Add stop" 32 times to reach the ceiling, and each click
  // commits through the mocked command and re-renders the whole stop list from
  // the query cache, so it does ~1000 row renders. That is real work and it
  // measures just over vitest's 5s default; the budget is raised rather than
  // shrinking the click count, which is the point of the test.
  it('keeps the stop count within the validated range', { timeout: 30_000 }, async () => {
    installProject()
    const user = userEvent.setup()
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [...page.layers, gradientShapeLayer],
    }))
    // The inspector renders from the page query, not from a local draft, so the
    // stub has to publish the committed style back or the button would never
    // reach the 32-stop ceiling. The real command round-trips the same way.
    const setShapeStyle = vi.spyOn(commands, 'setShapeStyle').mockImplementation(async (_r, updates) => {
      queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
        ...page,
        layers: page.layers.map((layer) => {
          const update = updates.find((entry) => entry.element === layer.id)
          return update ? { ...layer, style: update.style } : layer
        }),
      }))
      return 2
    })
    useKoharuStore.setState({ selectedLayers: ['shape-gradient'], canvasRevision: 1 })
    render(<Inspector />)

    const add = screen.getByRole('button', { name: 'Add stop' })
    // Two stops is the floor, so removal starts disabled.
    expect(screen.getAllByRole('button', { name: 'Remove stop' })[0]).toBeDisabled()

    const stopCounts = () =>
      setShapeStyle.mock.calls.map(([, updates]) => {
        const fill = updates[0]?.style.fill
        if (!fill || !('gradient' in fill)) throw new Error('expected a gradient fill')
        return fill.gradient.stops
      })

    for (let index = 0; index < 32; index += 1) await user.click(add)

    expect(add).toBeDisabled()
    const committed = stopCounts().at(-1)
    expect(committed).toHaveLength(32)
    // The server rejects more than 32 stops and rejects duplicate offsets, so
    // neither may ever appear in a commit.
    expect(stopCounts().every((stops) => stops.length <= 32)).toBe(true)
    expect(new Set(committed?.map((stop) => stop.offset)).size).toBe(32)
    expect(committed?.every((stop) => stop.offset >= 0 && stop.offset <= 1)).toBe(true)
  })

  it('never commits a duplicate offset when a gradient has tight neighbours', async () => {
    // A spec read from disk may hold offsets closer than the editor's grid; the
    // server accepts that. Rounding a midpoint between such neighbours lands on
    // one of them, which the server then rejects. Every commit from the editor
    // must stay server-valid: distinct offsets in 0..=1.
    installProject()
    const user = userEvent.setup()
    const setShapeStyle = vi.spyOn(commands, 'setShapeStyle').mockResolvedValue(2)
    const tightGradient: ShapeFill = {
      gradient: {
        kind: { linear: { start: [0, 0], end: [1, 0] } },
        stops: [
          { offset: 0.7865, color: [255, 255, 255, 255] },
          { offset: 0.3, color: [255, 0, 0, 255] },
          { offset: 0.7865004, color: [0, 255, 0, 255] },
          { offset: 0.9, color: [0, 0, 255, 255] },
        ],
        extend: 'pad',
      },
    }
    const tight: Layer = { ...gradientShapeLayer, style: { fill: tightGradient, stroke: null } }
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [...page.layers, tight],
    }))
    useKoharuStore.setState({ selectedLayers: ['shape-gradient'], canvasRevision: 1 })
    render(<Inspector />)

    const committed = () =>
      setShapeStyle.mock.calls.flatMap(([, updates]) => {
        const fill = updates[0]?.style.fill
        if (!fill || !('gradient' in fill)) return []
        return [fill.gradient.stops.map((stop) => stop.offset)]
      })

    const add = screen.getByRole('button', { name: 'Add stop' })
    for (let index = 0; index < 8 && !add.hasAttribute('disabled'); index += 1) {
      await user.click(add)
    }
    for (const offsets of committed()) {
      expect(new Set(offsets).size, `duplicate in ${offsets.join(', ')}`).toBe(offsets.length)
      expect(offsets.every((offset) => offset >= 0 && offset <= 1)).toBe(true)
    }
    // The tight pair was pulled apart rather than rounded together.
    expect(setShapeStyle).toHaveBeenCalled()
  })

  it('does not show the shape controls for a text selection', () => {
    installProject()
    render(<Inspector />)

    expect(screen.queryByTestId('shape-inspector')).not.toBeInTheDocument()
  })

  it('switches a shape fill from solid to a pattern', async () => {
    installProject()
    const user = userEvent.setup()
    const setShapeStyle = vi.spyOn(commands, 'setShapeStyle').mockResolvedValue(2)
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [...page.layers, shapeLayer],
    }))
    useKoharuStore.setState({ selectedLayers: ['shape'], canvasRevision: 1 })
    render(<Inspector />)

    // A pattern is the third kind the fill picker offers, beside solid and
    // gradient, and a solid fill leaves it unselected.
    const pattern = screen.getByRole('radio', { name: 'Pattern' })
    expect(pattern).toHaveAttribute('aria-checked', 'false')
    expect(screen.getByRole('radio', { name: 'Solid' })).toHaveAttribute('aria-checked', 'true')

    await user.click(pattern)

    await waitFor(() =>
      expect(setShapeStyle).toHaveBeenCalledWith(1, [
        {
          element: 'shape',
          style: expect.objectContaining({
            fill: expect.objectContaining({ pattern: expect.anything() }),
          }),
        },
      ]),
    )
  })

  it('keeps a pattern fill a pattern when one of its parameters changes', async () => {
    // The fill colour well used to write `{ solid }` unconditionally, which
    // silently downgraded a gradient. A pattern parameter edit walks the same
    // path: it must stay in the pattern branch and must not become a solid or a
    // gradient. This is the data-loss path B3 Task 6 hit.
    installProject()
    const setShapeStyle = vi.spyOn(commands, 'setShapeStyle').mockResolvedValue(2)
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [...page.layers, patternShapeLayer],
    }))
    useKoharuStore.setState({ selectedLayers: ['shape-pattern'], canvasRevision: 1 })
    render(<Inspector />)

    // The pattern panel is the only fill editor on screen for a pattern.
    expect(screen.getByTestId('pattern-fill')).toBeInTheDocument()

    fireEvent.change(screen.getByRole('textbox', { name: 'Scale' }), { target: { value: '0.5' } })

    await waitFor(() => expect(setShapeStyle).toHaveBeenCalled())
    const fill = setShapeStyle.mock.calls.at(-1)?.[1][0]?.style.fill
    expect(fill).not.toHaveProperty('solid')
    expect(fill).not.toHaveProperty('gradient')
    expect(fill).toEqual({
      pattern: expect.objectContaining({ scale: 0.5, repeat: 'tile' }),
    })
  })

  it('repurposes the flat colour well onto the pattern fallback', async () => {
    // The fill colour well is the only authority for a pattern's colour. While a
    // pattern is active it stops naming the flat fill and edits `fallback`
    // instead, so no second control can disagree about the colour and no edit
    // demotes the fill to `{ solid }`.
    installProject()
    const user = userEvent.setup()
    const setShapeStyle = vi.spyOn(commands, 'setShapeStyle').mockResolvedValue(2)
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [...page.layers, patternShapeLayer],
    }))
    useKoharuStore.setState({ selectedLayers: ['shape-pattern'], canvasRevision: 1 })
    render(<Inspector />)

    expect(screen.queryByRole('button', { name: 'Fill' })).not.toBeInTheDocument()
    expect(screen.queryByTestId('gradient-stops')).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Pattern' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'Hex color code' }), {
      target: { value: '#00FF00' },
    })

    await waitFor(() => expect(setShapeStyle).toHaveBeenCalled())
    const fill = setShapeStyle.mock.calls.at(-1)?.[1][0]?.style.fill
    expect(fill).not.toHaveProperty('solid')
    expect(fill).not.toHaveProperty('gradient')
    expect(fill).toEqual({
      pattern: expect.objectContaining({ fallback: [0, 255, 0, 255] }),
    })
  })

  it('disables a pattern origin and scale while it is stretched', () => {
    // The server ignores both under `stretch`, so leaving them editable would
    // invite an edit that changes nothing.
    installProject()
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [
        ...page.layers,
        {
          ...patternShapeLayer,
          style: {
            fill: { pattern: { ...patternSpec, repeat: 'stretch' } },
            stroke: null,
          },
        } as Layer,
      ],
    }))
    useKoharuStore.setState({ selectedLayers: ['shape-pattern'], canvasRevision: 1 })
    render(<Inspector />)

    expect(screen.getByRole('radio', { name: 'Stretch' })).toHaveAttribute('aria-checked', 'true')
    expect(screen.getByRole('textbox', { name: 'Origin X' })).toBeDisabled()
    expect(screen.getByRole('textbox', { name: 'Origin Y' })).toBeDisabled()
    expect(screen.getByRole('textbox', { name: 'Scale' })).toBeDisabled()
    expect(screen.getByRole('textbox', { name: 'Rotation' })).toBeEnabled()
    expect(screen.getByText('Origin and scale are ignored while stretching.')).toBeInTheDocument()
  })

  it('applies a brightness adjustment to a selected raster layer', async () => {
    installProject()
    const user = userEvent.setup()
    const adjustRaster = vi.spyOn(commands, 'adjustRaster').mockResolvedValue({ revision: 2, layer: 'raster' })
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [...page.layers, rasterLayer],
    }))
    useKoharuStore.setState({ selectedLayers: ['raster'], canvasRevision: 1 })
    render(<Inspector />)

    expect(screen.getByTestId('adjust-inspector')).toBeInTheDocument()
    // Nothing is committed until the user applies: the adjustment is a single
    // explicit commit, not a live preview.
    expect(adjustRaster).not.toHaveBeenCalled()

    fireEvent.change(screen.getByRole('textbox', { name: 'Amount' }), {
      target: { value: '40' },
    })
    expect(adjustRaster).not.toHaveBeenCalled()

    await user.click(screen.getByRole('button', { name: 'Apply' }))

    await waitFor(() =>
      expect(adjustRaster).toHaveBeenCalledWith(1, 'raster', { type: 'brightness', delta: 40 }),
    )
  })

  it('shows the adjustment section only for a raster layer', () => {
    installProject()
    render(<Inspector />)

    expect(screen.queryByTestId('adjust-inspector')).not.toBeInTheDocument()
  })

  it('does not show the adjustment section for a shape or group selection', () => {
    installProject()
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [...page.layers, shapeLayer],
    }))
    useKoharuStore.setState({ selectedLayers: ['shape'], canvasRevision: 1 })
    render(<Inspector />)

    expect(screen.queryByTestId('adjust-inspector')).not.toBeInTheDocument()
  })

  it('does not show the adjustment section for a group holding a raster child', () => {
    // The shape inspector expands a group selection; the adjustment section must
    // not, or selecting the group would offer a control aimed at one child.
    installProject()
    const group: Layer = {
      type: 'group',
      id: 'group',
      parent: 'page',
      visibility: { visible: true, opacity: 1 },
      name: 'Group 1',
      role: null,
    }
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [...page.layers, group, { ...rasterLayer, parent: 'group' }],
    }))
    useKoharuStore.setState({ selectedLayers: ['group'], canvasRevision: 1 })
    render(<Inspector />)

    expect(screen.queryByTestId('adjust-inspector')).not.toBeInTheDocument()
  })

  it('applies the adjustment kind selected from the dropdown', async () => {
    installProject()
    const user = userEvent.setup()
    const adjustRaster = vi.spyOn(commands, 'adjustRaster').mockResolvedValue({ revision: 2, layer: 'raster' })
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [...page.layers, rasterLayer],
    }))
    useKoharuStore.setState({ selectedLayers: ['raster'], canvasRevision: 1 })
    render(<Inspector />)

    await user.click(screen.getByRole('combobox', { name: 'Adjustment kind' }))
    await user.click(await screen.findByRole('option', { name: 'Levels' }))

    fireEvent.change(screen.getByRole('textbox', { name: 'Low' }), { target: { value: '10' } })
    fireEvent.change(screen.getByRole('textbox', { name: 'High' }), { target: { value: '240' } })
    await user.click(screen.getByRole('button', { name: 'Apply' }))

    await waitFor(() =>
      expect(adjustRaster).toHaveBeenCalledWith(1, 'raster', { type: 'levels', low: 10, high: 240 }),
    )
  })

  it('refuses to apply inverted levels', async () => {
    // The server rejects `low >= high`; the UI must not be able to send it.
    installProject()
    const user = userEvent.setup()
    const adjustRaster = vi.spyOn(commands, 'adjustRaster').mockResolvedValue({ revision: 2, layer: 'raster' })
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [...page.layers, rasterLayer],
    }))
    useKoharuStore.setState({ selectedLayers: ['raster'], canvasRevision: 1 })
    render(<Inspector />)

    await user.click(screen.getByRole('combobox', { name: 'Adjustment kind' }))
    await user.click(await screen.findByRole('option', { name: 'Levels' }))

    fireEvent.change(screen.getByRole('textbox', { name: 'Low' }), { target: { value: '200' } })
    fireEvent.change(screen.getByRole('textbox', { name: 'High' }), { target: { value: '50' } })

    const apply = screen.getByRole('button', { name: 'Apply' })
    expect(apply).toBeDisabled()
    expect(screen.getByText('Low must be below high.')).toBeInTheDocument()
    await user.click(apply)
    expect(adjustRaster).not.toHaveBeenCalled()
  })

  it('clamps adjustment inputs to the server ranges', async () => {
    installProject()
    const user = userEvent.setup()
    const adjustRaster = vi.spyOn(commands, 'adjustRaster').mockResolvedValue({ revision: 2, layer: 'raster' })
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [...page.layers, rasterLayer],
    }))
    useKoharuStore.setState({ selectedLayers: ['raster'], canvasRevision: 1 })
    render(<Inspector />)

    // Brightness is -255..=255; a typed 9990 must not reach the command.
    fireEvent.change(screen.getByRole('textbox', { name: 'Amount' }), {
      target: { value: '9990' },
    })
    await user.click(screen.getByRole('button', { name: 'Apply' }))

    await waitFor(() => expect(adjustRaster).toHaveBeenCalled())
    expect(adjustRaster.mock.calls.at(-1)?.[2]).toEqual({ type: 'brightness', delta: 255 })

    adjustRaster.mockClear()
    await user.click(screen.getByRole('combobox', { name: 'Adjustment kind' }))
    await user.click(await screen.findByRole('option', { name: 'Hue' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'Degrees' }), {
      target: { value: '-900' },
    })
    await user.click(screen.getByRole('button', { name: 'Apply' }))

    await waitFor(() =>
      expect(adjustRaster).toHaveBeenCalledWith(1, 'raster', { type: 'hue', degrees: -180 }),
    )
  })

  it('applies a grayscale adjustment with no parameters', async () => {
    installProject()
    const user = userEvent.setup()
    const adjustRaster = vi.spyOn(commands, 'adjustRaster').mockResolvedValue({ revision: 2, layer: 'raster' })
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [...page.layers, rasterLayer],
    }))
    useKoharuStore.setState({ selectedLayers: ['raster'], canvasRevision: 1 })
    render(<Inspector />)

    await user.click(screen.getByRole('combobox', { name: 'Adjustment kind' }))
    await user.click(await screen.findByRole('option', { name: 'Grayscale' }))
    await user.click(screen.getByRole('button', { name: 'Apply' }))

    await waitFor(() =>
      expect(adjustRaster).toHaveBeenCalledWith(1, 'raster', { type: 'grayscale' }),
    )
  })

  it('disables apply when more than one raster layer is selected', async () => {
    // The command takes a single layer, and applying once per layer would need
    // one undo per layer for one click. Disabling says that plainly instead of
    // silently adjusting whichever layer happens to be first.
    installProject()
    const user = userEvent.setup()
    const adjustRaster = vi.spyOn(commands, 'adjustRaster').mockResolvedValue({ revision: 2, layer: 'raster' })
    const second: Layer = { ...rasterLayer, id: 'raster-2', name: 'Paint 2' }
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [...page.layers, rasterLayer, second],
    }))
    useKoharuStore.setState({ selectedLayers: ['raster', 'raster-2'], canvasRevision: 1 })
    render(<Inspector />)

    const apply = screen.getByRole('button', { name: 'Apply' })
    expect(apply).toBeDisabled()
    expect(screen.getByText('Select a single raster layer to adjust it.')).toBeInTheDocument()
    await user.click(apply)
    expect(adjustRaster).not.toHaveBeenCalled()
  })

  it('re-enables apply when the selection returns to a single raster layer', async () => {
    installProject()
    const user = userEvent.setup()
    vi.spyOn(commands, 'adjustRaster').mockResolvedValue({ revision: 2, layer: 'raster' })
    const second: Layer = { ...rasterLayer, id: 'raster-2', name: 'Paint 2' }
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [...page.layers, rasterLayer, second],
    }))
    useKoharuStore.setState({ selectedLayers: ['raster', 'raster-2'], canvasRevision: 1 })
    render(<Inspector />)
    expect(screen.getByRole('button', { name: 'Apply' })).toBeDisabled()

    act(() => useKoharuStore.setState({ selectedLayers: ['raster'] }))

    expect(screen.getByRole('button', { name: 'Apply' })).not.toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Apply' }))
  })

  it('ignores a second apply while the first is still in flight', async () => {
    // Both clicks would carry the same expected revision, so the second would
    // fail `ensure_revision` and surface an error the user cannot act on.
    installProject()
    const user = userEvent.setup()
    let settle: ((value: { revision: number; layer: string }) => void) | undefined
    const adjustRaster = vi
      .spyOn(commands, 'adjustRaster')
      .mockReturnValue(new Promise((resolve) => {
        settle = resolve
      }))
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [...page.layers, rasterLayer],
    }))
    useKoharuStore.setState({ selectedLayers: ['raster'], canvasRevision: 1 })
    render(<Inspector />)

    await user.click(screen.getByRole('button', { name: 'Apply' }))
    expect(screen.getByRole('button', { name: 'Apply' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Apply' }))
    expect(adjustRaster).toHaveBeenCalledTimes(1)

    await act(async () => {
      settle?.({ revision: 2, layer: 'raster' })
    })
    await waitFor(() => expect(screen.getByRole('button', { name: 'Apply' })).not.toBeDisabled())
  })

  it('resets the adjustment fields when the selected raster layer changes', async () => {
    installProject()
    vi.spyOn(commands, 'adjustRaster').mockResolvedValue({ revision: 2, layer: 'raster' })
    const second: Layer = { ...rasterLayer, id: 'raster-2', name: 'Paint 2' }
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [...page.layers, rasterLayer, second],
    }))
    useKoharuStore.setState({ selectedLayers: ['raster'], canvasRevision: 1 })
    render(<Inspector />)

    fireEvent.change(screen.getByRole('textbox', { name: 'Amount' }), { target: { value: '77' } })
    expect(screen.getByRole('textbox', { name: 'Amount' })).toHaveValue('77')
    // Selecting another layer in the list blurs the field first, which is what
    // settles typed text back to its canonical form; the reset has to hold for
    // the text the field then displays.
    fireEvent.blur(screen.getByRole('textbox', { name: 'Amount' }))

    act(() => useKoharuStore.setState({ selectedLayers: ['raster-2'] }))

    await waitFor(() => expect(screen.getByRole('textbox', { name: 'Amount' })).toHaveValue('0'))
  })

  it('names a shape layer from its stored name and kind in the layer list', async () => {
    // Decision D4: `ShapeLayer.name` exists so a shape is identifiable in the
    // layer list. Without a `shape` branch the row falls through to the image
    // arm and every shape reads "Image N" with an image icon.
    installProject()
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [...page.layers, { ...shapeLayer, name: 'Rectangle 3' }],
    }))
    render(<Inspector />)

    expect(await screen.findByText('Rectangle 3')).toBeInTheDocument()
    expect(screen.queryByText('Image 2')).not.toBeInTheDocument()
    // The kind line distinguishes a shape from an image without relying on the
    // name alone.
    expect(screen.getByText('Shape')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Edit Rectangle 3' })).toBeInTheDocument()
  })

  it('adjusts brush size from the toolbar popover', async () => {
    const user = userEvent.setup()
    installProject()
    useKoharuStore.setState({ tool: 'draw' })
    render(
      <TooltipProvider>
        <ToolBar />
      </TooltipProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Brush size: 48 pixels' }))
    expect(screen.getByRole('textbox', { name: 'Brush size' })).toHaveValue('48')

    await user.click(screen.getByRole('button', { name: 'Increase brush size' }))
    expect(useKoharuStore.getState().brush.diameter).toBe(49)
  })

  it('uses the border color well to enable and disable the border', async () => {
    const user = userEvent.setup()
    installProject()
    const setTypography = vi.spyOn(commands, 'setTypography').mockResolvedValue(null)
    render(<Inspector />)

    expect(screen.queryByRole('button', { name: 'Enable text border' })).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Border color' }))
    await user.click(screen.getByRole('button', { name: 'Transparent' }))
    await waitFor(() =>
      expect(setTypography).toHaveBeenCalledWith([
        expect.objectContaining({
          layer: 'element',
          typography: expect.objectContaining({
            stroke_color: [255, 255, 255, 0],
            stroke_width: 1.5,
          }),
        }),
      ]),
    )

    setTypography.mockClear()
    fireEvent.change(screen.getByRole('textbox', { name: 'Hex color code' }), {
      target: { value: '#FF0000' },
    })
    await waitFor(() =>
      expect(setTypography).toHaveBeenCalledWith([
        expect.objectContaining({
          layer: 'element',
          typography: expect.objectContaining({
            stroke_color: [255, 0, 0, 255],
            stroke_width: 1.5,
          }),
        }),
      ]),
    )
  })

  it('only offers styles and weights available for the selected font family', async () => {
    const user = userEvent.setup()
    installProject()
    const setTypography = vi.spyOn(commands, 'setTypography').mockResolvedValue(null)
    queryClient.setQueryData(fontsKey, [
      {
        name: 'Noto Sans',
        metadata: {
          primary_script: 'latn',
          scripts: ['latn'],
          languages: ['en'],
          category: 'SANS_SERIF',
          classifications: ['sans-serif'],
          use_cases: ['body-text'],
        },
        sources: ['system'],
        faces: [
          {
            postscript_name: 'NotoSans-Regular',
            weight: 400,
            weight_range: null,
            style: 'normal',
          },
          {
            postscript_name: 'NotoSans-Bold',
            weight: 700,
            weight_range: null,
            style: 'normal',
          },
          {
            postscript_name: 'NotoSans-Italic',
            weight: 400,
            weight_range: null,
            style: 'italic',
          },
        ],
      },
    ])
    const { unmount } = render(<Inspector />)

    fireEvent.click(screen.getByRole('combobox', { name: 'Font weight' }))

    expect(screen.getByRole('option', { name: '400' })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: '700' })).toBeInTheDocument()
    expect(screen.queryByRole('option', { name: '100' })).not.toBeInTheDocument()

    unmount()
    render(<Inspector />)
    fireEvent.click(screen.getByRole('combobox', { name: 'Font style' }))
    expect(screen.getByRole('option', { name: 'Regular' })).toBeInTheDocument()
    await user.click(screen.getByRole('option', { name: 'Italic' }))
    await waitFor(() =>
      expect(setTypography).toHaveBeenCalledWith([
        expect.objectContaining({
          layer: 'element',
          typography: expect.objectContaining({ font_style: 'italic', font_weight: 400 }),
        }),
      ]),
    )
  })

  it('never writes a weight that the selected family does not offer', async () => {
    installProject()
    const setTypography = vi.spyOn(commands, 'setTypography').mockResolvedValue(null)
    const face = (name: string, weight: number) => ({
      postscript_name: name,
      weight,
      weight_range: null,
      style: 'normal' as const,
    })
    const family = (name: string, weights: number[]) => ({
      name,
      metadata: {
        primary_script: 'latn',
        scripts: ['latn'],
        languages: ['en'],
        category: 'SANS_SERIF',
        classifications: ['sans-serif'],
        use_cases: ['body-text'],
      },
      sources: ['system' as const],
      faces: weights.map((weight) => face(`${name}-${weight}`, weight)),
    })
    queryClient.setQueryData(fontsKey, [family('Alpha', [700]), family('Beta', [400])])
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [
        {
          ...textLayer,
          typography: { ...textLayer.typography, preferred_font: 'Alpha', font_weight: 700 },
        },
        {
          ...secondLayer,
          typography: { ...textLayer.typography, preferred_font: 'Beta', font_weight: 700 },
        },
      ],
    }))
    render(<Inspector />)
    const written = () =>
      setTypography.mock.calls.flatMap(([updates]) =>
        updates.map((update) => update.typography.font_weight),
      )

    fireEvent.click(screen.getByRole('combobox', { name: 'Font weight' }))
    act(() => useKoharuStore.setState({ selectedLayers: ['second'] }))
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    expect(written()).toEqual([])

    await userEvent.click(screen.getByRole('option', { name: '400' }))
    await waitFor(() => expect(written()).toEqual([400]))
  })

  it('debounces layer text editing and flushes it when focus leaves the field', async () => {
    installProject()
    const save = vi.spyOn(commands, 'setSourceText').mockResolvedValue(null)
    render(<Inspector />)
    const layer = screen.getByRole('button', { name: 'Edit Hello' })
    expect(screen.getByTestId('edit-source-element')).toBeInTheDocument()
    fireEvent.click(layer)
    expect(screen.queryByTestId('edit-source-element')).not.toBeInTheDocument()
    fireEvent.click(layer)
    const source = screen.getByTestId('edit-source-element')
    fireEvent.change(source, { target: { value: 'corrected OCR' } })
    fireEvent.blur(source)
    await waitFor(() => expect(save).toHaveBeenCalledWith('element', 'corrected OCR'))
  })

  it('shows actual layers with only the useful text-role distinction', () => {
    installProject()
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [
        ...page.layers.map((layer) =>
          layer.type === 'text'
            ? { ...layer, content: { ...layer.content, role: 'dev.koharu.text.onomatopoeia' } }
            : layer,
        ),
        {
          ...textLayer,
          id: 'dialogue',
          content: {
            ...textLayer.content,
            id: 'dialogue-content',
            translation: { text: 'Dialogue line', language: null },
            role: 'dev.koharu.text.dialogue',
          },
        },
        {
          ...textLayer,
          id: 'free-text',
          content: {
            ...textLayer.content,
            id: 'free-text-content',
            translation: { text: 'Caption', language: null },
            role: 'dev.koharu.text.free-text',
          },
        },
      ],
    }))
    render(<Inspector />)

    expect(screen.queryByRole('button', { name: /Filter layers by type/ })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Edit Hello' })).toHaveTextContent('Text')
    expect(screen.getByRole('button', { name: 'Edit Dialogue line' })).toHaveTextContent('Dialogue')
    expect(screen.getByRole('button', { name: 'Edit Caption' })).toHaveTextContent('Free text')
    expect(screen.queryByText('Onomatopoeia')).not.toBeInTheDocument()
  })

  function installLayerSelectionFixture() {
    installProject()
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [...page.layers, secondLayer, artworkLayer, thirdLayer],
    }))
  }

  it('toggles layers in the selection with ctrl-click', () => {
    installLayerSelectionFixture()
    render(<Inspector />)

    expect(useKoharuStore.getState().selectedLayers).toEqual(['element'])
    fireEvent.click(screen.getByRole('button', { name: 'Edit Second' }), { ctrlKey: true })
    expect(useKoharuStore.getState().selectedLayers).toEqual(['element', 'second'])
    fireEvent.click(screen.getByRole('button', { name: 'Edit Hello' }), { ctrlKey: true })
    expect(useKoharuStore.getState().selectedLayers).toEqual(['second'])
    fireEvent.click(screen.getByRole('button', { name: 'Edit Second' }), { ctrlKey: true })
    expect(useKoharuStore.getState().selectedLayers).toEqual([])
  })

  it('toggles layers in the selection with meta-click', () => {
    installLayerSelectionFixture()
    render(<Inspector />)

    fireEvent.click(screen.getByRole('button', { name: 'Edit Second' }), { metaKey: true })
    expect(useKoharuStore.getState().selectedLayers).toEqual(['element', 'second'])
    fireEvent.click(screen.getByRole('button', { name: 'Edit Second' }), { metaKey: true })
    expect(useKoharuStore.getState().selectedLayers).toEqual(['element'])
  })

  it('selects a display range with shift-click, skipping locked layers', () => {
    installLayerSelectionFixture()
    render(<Inspector />)

    fireEvent.click(screen.getByRole('button', { name: 'Edit Third' }))
    expect(useKoharuStore.getState().selectedLayers).toEqual(['third'])
    fireEvent.click(screen.getByRole('button', { name: 'Edit Hello' }), { shiftKey: true })
    expect(useKoharuStore.getState().selectedLayers).toEqual(['third', 'second', 'element'])
  })

  it('unions a shift range with the existing selection when ctrl is held', () => {
    installLayerSelectionFixture()
    render(<Inspector />)

    fireEvent.click(screen.getByRole('button', { name: 'Edit Second' }))
    fireEvent.click(screen.getByRole('button', { name: 'Edit Third' }), {
      ctrlKey: true,
      shiftKey: true,
    })
    expect(useKoharuStore.getState().selectedLayers).toEqual(['second', 'third'])
  })

  it('falls back to single selection when shift-clicking without an anchor', () => {
    installLayerSelectionFixture()
    render(<Inspector />)

    fireEvent.click(screen.getByRole('button', { name: 'Edit Third' }), { shiftKey: true })
    expect(useKoharuStore.getState().selectedLayers).toEqual(['third'])
  })

  it('collapses a multi-selection back to a single layer on plain click', () => {
    installLayerSelectionFixture()
    render(<Inspector />)

    fireEvent.click(screen.getByRole('button', { name: 'Edit Second' }), { ctrlKey: true })
    expect(useKoharuStore.getState().selectedLayers).toEqual(['element', 'second'])
    fireEvent.click(screen.getByRole('button', { name: 'Edit Hello' }))
    expect(useKoharuStore.getState().selectedLayers).toEqual(['element'])
  })

  it('resets a custom text frame to its automatic region', async () => {
    installProject()
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: page.layers.map((layer) =>
        layer.type === 'text' ? { ...layer, automatic_region: 'bubble' } : layer,
      ),
    }))
    const reset = vi.spyOn(commands, 'setGeometry').mockResolvedValue(null)
    render(<Inspector />)

    expect(screen.getByText('Custom frame')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Reset to auto fit' }))

    await waitFor(() => expect(reset).toHaveBeenCalledWith([{ layer: 'element', points: null }]))
  })

  it('shows zoom before page size without a fit control', () => {
    installProject()
    useKoharuStore.setState({ camera: { zoom: 1.25, translation: [0, 0], fitted: false } })
    render(<StatusBar onZoomChange={vi.fn()} />)

    const zoom = screen.getByText('125%')
    const size = screen.getByText('1000 × 1500 px')
    expect(zoom.compareDocumentPosition(size) & Node.DOCUMENT_POSITION_FOLLOWING).not.toBe(0)
    expect(screen.queryByRole('button', { name: 'Fit window' })).not.toBeInTheDocument()
  })

  it('changes the pipeline scope and selected stages from the runtime selector', async () => {
    installProject()
    const run = vi.spyOn(commands, 'process').mockResolvedValue('job')
    render(<CanvasCommandBar />)

    fireEvent.click(screen.getByRole('button', { name: 'Processing settings' }))
    fireEvent.click(screen.getByRole('button', { name: /Scope Selection/ }))
    fireEvent.click(screen.getByRole('button', { name: /Entire project/ }))
    fireEvent.click(screen.getByRole('button', { name: /Stages 4 stages/ }))
    fireEvent.click(screen.getByRole('button', { name: /Translation/ }))
    fireEvent.click(screen.getByRole('button', { name: /Inpainting/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Run processing' }))
    await waitFor(() =>
      expect(run).toHaveBeenLastCalledWith(
        { scope: 'project' },
        { operation: 'stages', stages: ['detection', 'ocr'] },
      ),
    )

    fireEvent.click(screen.getByRole('button', { name: 'Processing settings' }))
    fireEvent.click(screen.getByRole('button', { name: /Scope Project/ }))
    fireEvent.click(screen.getByRole('button', { name: /Selected pages/ }))
    fireEvent.click(screen.getByRole('button', { name: /Stages 2 stages/ }))
    fireEvent.click(screen.getByRole('button', { name: /Translation/ }))
    fireEvent.click(screen.getByRole('button', { name: /Inpainting/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Run processing' }))
    await waitFor(() =>
      expect(run).toHaveBeenLastCalledWith(
        { scope: 'pages', value: ['page'] },
        { operation: 'full' },
      ),
    )
  })

  it('runs the selected pages by default and exposes the runtime shortcuts', async () => {
    installProject()
    const run = vi.spyOn(commands, 'process').mockResolvedValue('job')
    render(<CanvasCommandBar />)

    fireEvent.click(screen.getByRole('button', { name: 'Run processing' }))
    await waitFor(() =>
      expect(run).toHaveBeenLastCalledWith(
        { scope: 'pages', value: ['page'] },
        { operation: 'full' },
      ),
    )

    const selector = screen.getByRole('button', { name: 'Processing settings' })
    const runButton = screen.getByRole('button', { name: 'Run processing' })
    expect(selector).toHaveClass('h-7')
    expect(runButton).toHaveClass('h-7', 'bg-primary/80', 'hover:bg-primary/90')
    fireEvent.click(selector)
    await waitFor(() => expect(commands.getTranslationModels).toHaveBeenCalled())
    expect(screen.getByRole('button', { name: /Model Gemma 4 E2B Instruct/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Scope Selection/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Stages 4 stages/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Output English/ })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }))
    expect(useKoharuStore.getState().settingsOpen).toBe(true)
  })

  it('configures translation output from the runtime selector', async () => {
    installProject()
    const user = userEvent.setup()
    const nextPreferences: Preferences = {
      ...preferences,
      pipeline: {
        ...preferences.pipeline,
        translation: {
          ...preferences.pipeline.translation,
          target_language: 'ja-JP',
          instructions: 'Keep character names unchanged.',
        },
      },
    }
    let finishSave!: (saved: Preferences) => void
    const pendingSave = new Promise<Preferences>((resolve) => {
      finishSave = resolve
    })
    const save = vi.spyOn(commands, 'savePreferences').mockImplementation(() => pendingSave)
    render(<CanvasCommandBar />)

    await user.click(screen.getByRole('button', { name: 'Processing settings' }))
    await waitFor(() => expect(commands.getTranslationModels).toHaveBeenCalled())
    await user.click(screen.getByRole('button', { name: /Output English/ }))
    const language = screen.getByRole('combobox', { name: 'Target language' })
    expect(language).toHaveTextContent('English')
    expect(language).not.toHaveTextContent('en-US')
    await user.click(language)
    await user.click(await screen.findByRole('option', { name: 'Japanese' }))
    const instructions = screen.getByRole('textbox', { name: 'Translation instructions' })
    expect(instructions).toHaveClass('max-h-20', 'overflow-y-auto')
    await user.type(instructions, 'Keep character names unchanged.')
    expect(screen.queryByRole('button', { name: 'Apply output' })).not.toBeInTheDocument()

    await waitFor(() =>
      expect(save).toHaveBeenCalledWith(
        nextPreferences.pipeline,
        preferences.providers,
        preferences.typesetting,
      ),
    )
    expect(
      screen.queryByLabelText(/Saving output settings|outputPicker\.saving/),
    ).not.toBeInTheDocument()
    expect(language).not.toBeDisabled()
    expect(instructions).not.toBeDisabled()
    expect(instructions).toHaveFocus()
    await act(async () => finishSave(nextPreferences))
    await waitFor(() =>
      expect(useKoharuStore.getState().preferences?.pipeline.translation).toEqual(
        nextPreferences.pipeline.translation,
      ),
    )
    expect(instructions).toBeInTheDocument()
    expect(instructions).toHaveFocus()
  })

  it('preserves the runtime shortcuts while visiting settings', async () => {
    installProject()
    const user = userEvent.setup()
    const save = vi
      .spyOn(commands, 'savePreferences')
      .mockImplementation(async (pipeline, providers, typesetting) => ({
        ...preferences,
        pipeline,
        providers,
        typesetting,
      }))
    render(
      <ThemeProvider attribute='class'>
        <SettingsNavigationHarness />
      </ThemeProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Processing settings' }))
    await user.click(screen.getByRole('button', { name: /Scope Selection/ }))
    await user.click(screen.getByRole('button', { name: /Entire project/ }))
    await user.click(screen.getByRole('button', { name: /Stages 4 stages/ }))
    await user.click(screen.getByRole('button', { name: /Translation/ }))
    await user.click(screen.getByRole('button', { name: /Inpainting/ }))
    await user.click(screen.getByRole('button', { name: 'Back' }))
    await user.click(screen.getByRole('button', { name: /Output English/ }))
    await user.click(screen.getByRole('combobox', { name: 'Target language' }))
    await user.click(await screen.findByRole('option', { name: 'Japanese' }))

    act(() => useKoharuStore.getState().setSettingsOpen(true))
    await waitFor(() => expect(save).toHaveBeenCalled())
    await user.click(screen.getByRole('button', { name: 'Back to editor' }))
    await user.click(screen.getByRole('button', { name: 'Processing settings' }))

    expect(screen.getByRole('button', { name: /Scope Project/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Stages 2 stages/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Output Japanese/ })).toBeInTheDocument()
  })

  it('changes the translation model without re-enabling vision or reasoning', async () => {
    installProject()
    const user = userEvent.setup()
    const currentPreferences: Preferences = {
      ...preferences,
      pipeline: {
        ...preferences.pipeline,
        translation: {
          ...preferences.pipeline.translation,
          generation: {
            ...preferences.pipeline.translation.generation,
            vision: false,
            reasoning: false,
          },
        },
      },
    }
    useKoharuStore.setState({ preferences: currentPreferences })
    const nextPreferences: Preferences = {
      ...currentPreferences,
      pipeline: {
        ...currentPreferences.pipeline,
        translation: {
          ...currentPreferences.pipeline.translation,
          model: {
            provider: 'local',
            model: 'gemma4-12b-it',
            quantization: null,
            vision: true,
            reasoning: true,
          },
        },
      },
    }
    const save = vi.spyOn(commands, 'savePreferences').mockResolvedValue(nextPreferences)
    useKoharuStore.setState({
      translationModels: [
        ...useKoharuStore.getState().translationModels,
        {
          provider: 'local',
          model: 'gemma4-12b-it',
          name: 'Gemma 4 12B',
          quantizations: [],
          vision: true,
          reasoning: true,
        },
      ],
    })
    render(<CanvasCommandBar />)

    fireEvent.click(screen.getByRole('button', { name: 'Processing settings' }))
    fireEvent.click(screen.getByRole('button', { name: /Model Gemma 4 E2B Instruct/ }))
    const search = screen.getByRole('textbox', { name: 'Search models' })
    expect(search).toHaveFocus()
    await user.type(search, '12b')
    expect(
      screen.queryByRole('button', { name: 'Use Gemma 4 E2B Instruct from Local' }),
    ).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Use Gemma 4 12B from Local' }))

    await waitFor(() =>
      expect(save).toHaveBeenCalledWith(
        nextPreferences.pipeline,
        currentPreferences.providers,
        currentPreferences.typesetting,
      ),
    )
    expect(useKoharuStore.getState().preferences?.pipeline.translation).toEqual(
      nextPreferences.pipeline.translation,
    )
  })

  it('preserves vision when the runtime selector chooses a text-only model', async () => {
    installProject()
    const nextPreferences: Preferences = {
      ...preferences,
      pipeline: {
        ...preferences.pipeline,
        translation: {
          ...preferences.pipeline.translation,
          model: {
            provider: 'deepseek',
            model: 'deepseek-chat',
            quantization: null,
            vision: false,
            reasoning: true,
          },
          generation: {
            ...preferences.pipeline.translation.generation,
            vision: true,
            reasoning: false,
          },
        },
      },
    }
    const save = vi.spyOn(commands, 'savePreferences').mockResolvedValue(nextPreferences)
    useKoharuStore.setState({
      translationModels: [
        ...useKoharuStore.getState().translationModels,
        {
          provider: 'deepseek',
          model: 'deepseek-chat',
          name: 'DeepSeek Chat',
          quantizations: [],
          vision: false,
          reasoning: true,
        },
      ],
    })
    render(<CanvasCommandBar />)

    fireEvent.click(screen.getByRole('button', { name: 'Processing settings' }))
    fireEvent.click(screen.getByRole('button', { name: /Model Gemma 4 E2B Instruct/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Use DeepSeek Chat from deepseek' }))

    await waitFor(() =>
      expect(save).toHaveBeenCalledWith(
        nextPreferences.pipeline,
        preferences.providers,
        preferences.typesetting,
      ),
    )
    expect(useKoharuStore.getState().preferences?.pipeline.translation).toEqual(
      nextPreferences.pipeline.translation,
    )
  })

  it('constrains long model names inside the runtime selector', async () => {
    installProject()
    const longName = 'Llama 3.2 8x3b Moe Dark Champion Instruct Uncensored Abliterated 18.4b'
    useKoharuStore.setState({
      translationModels: [
        ...useKoharuStore.getState().translationModels,
        {
          provider: 'local',
          model: 'long-model',
          name: longName,
          quantizations: [],
          vision: true,
          reasoning: false,
        },
      ],
    })
    render(<CanvasCommandBar />)

    fireEvent.click(screen.getByRole('button', { name: 'Processing settings' }))
    fireEvent.click(screen.getByRole('button', { name: /Model Gemma 4 E2B Instruct/ }))

    const label = await screen.findByText(longName)
    await waitFor(() => expect(commands.getTranslationModels).toHaveBeenCalled())
    expect(label).toHaveClass('truncate')
    expect(label.parentElement).toHaveClass('overflow-hidden')
    expect(label.closest('button')).toHaveClass('max-w-full', 'min-w-0', 'overflow-hidden')
    const content = label.closest('[data-slot="scroll-area-content"]')
    expect(content).toHaveStyle({ width: '100%' })
    expect(content?.closest('[data-slot="scroll-area-viewport"]')).toHaveClass('h-auto', 'max-h-64')
  })

  it('edits and persists pipeline and translation preferences from the settings page', async () => {
    installProject()
    const user = userEvent.setup()
    useKoharuStore.setState({ settingsOpen: true })
    const save = vi.spyOn(commands, 'savePreferences').mockResolvedValue(preferences)
    render(
      <ThemeProvider attribute='class'>
        <SettingsPage />
      </ThemeProvider>,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Pipeline' }))
    expect(screen.getByRole('heading', { level: 2, name: 'Pipeline' })).toBeInTheDocument()
    expect(screen.getAllByRole('combobox')).toHaveLength(5)
    fireEvent.change(screen.getByRole('spinbutton', { name: 'Text threshold' }), {
      target: { value: '0.42' },
    })
    await waitFor(() =>
      expect(save).toHaveBeenCalledWith(
        expect.objectContaining({
          processor: expect.objectContaining({
            'koharu-layout-rfdetr-seg-2xl': expect.objectContaining({ text_threshold: 0.42 }),
          }),
        }),
        preferences.providers,
        preferences.typesetting,
      ),
    )
    fireEvent.click(screen.getByRole('button', { name: 'Providers' }))
    expect(screen.getByRole('heading', { level: 2, name: 'Providers' })).toBeInTheDocument()
    expect(screen.getByLabelText('DeepL credential')).toBeInTheDocument()
    expect(screen.getAllByLabelText('Base URL')).toHaveLength(3)
    expect(screen.queryByRole('switch', { name: 'Vision' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Translation' }))
    expect(screen.getByRole('heading', { level: 2, name: 'Translation' })).toBeInTheDocument()
    expect(screen.getByText('Choose the model used to translate text.')).toBeInTheDocument()
    expect(
      screen.getByText('Control how the model selects and varies generated text.'),
    ).toBeInTheDocument()
    expect(screen.getByText('Use model reasoning during translation.')).toBeInTheDocument()
    const reasoning = screen.getByRole('switch', { name: 'Enable reasoning' })
    expect(reasoning).not.toHaveAttribute('aria-disabled', 'true')
    expect(screen.queryByText('Enable reasoning')).not.toBeInTheDocument()
    save.mockClear()
    await user.click(reasoning)
    await waitFor(() =>
      expect(save).toHaveBeenCalledWith(
        expect.objectContaining({
          translation: expect.objectContaining({
            generation: expect.objectContaining({ reasoning: true }),
          }),
        }),
        preferences.providers,
        preferences.typesetting,
      ),
    )
    const vision = screen.getByRole('switch', { name: 'Vision' })
    expect(screen.getByText('Feed page images to the LLM during translation.')).toBeInTheDocument()
    expect(vision).toBeChecked()
    save.mockClear()
    await user.click(vision)
    await waitFor(() =>
      expect(save).toHaveBeenCalledWith(
        expect.objectContaining({
          translation: expect.objectContaining({
            generation: expect.objectContaining({ vision: false }),
          }),
        }),
        preferences.providers,
        preferences.typesetting,
      ),
    )
    expect(screen.getByLabelText('Translation model')).toHaveTextContent('Gemma 4 E2B Instruct')
    await user.click(screen.getByLabelText('Translation model'))
    const modelSearch = screen.getByRole('textbox', { name: 'Search models' })
    expect(modelSearch).toHaveFocus()
    await user.type(modelSearch, 'missing model')
    expect(screen.getByText('No models match this search.')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Clear search' }))
    expect(
      screen.getByRole('button', { name: 'Use Gemma 4 E2B Instruct from Local' }),
    ).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Back' }))
    expect(screen.getByLabelText('Target language')).toHaveTextContent('English')
    expect(screen.getByText('Choose the language to translate text into.')).toBeInTheDocument()
    expect(screen.getByLabelText('Translation instructions')).toHaveClass(
      'field-sizing-fixed',
      'overflow-y-auto',
    )
  })

  it('keeps generation controls independent from model capabilities', async () => {
    installProject()
    const user = userEvent.setup()
    const configured: Preferences = {
      ...preferences,
      pipeline: {
        ...preferences.pipeline,
        translation: {
          ...preferences.pipeline.translation,
          model: {
            provider: 'deepseek',
            model: 'deepseek-chat',
            quantization: null,
            vision: false,
            reasoning: true,
          },
          generation: {
            ...preferences.pipeline.translation.generation,
            reasoning: false,
          },
        },
      },
    }
    useKoharuStore.setState({
      settingsOpen: true,
      preferences: configured,
      translationModels: [
        {
          provider: 'deepseek',
          model: 'deepseek-chat',
          name: 'DeepSeek Chat',
          quantizations: [],
          vision: false,
          reasoning: true,
        },
      ],
    })
    const save = vi.spyOn(commands, 'savePreferences').mockResolvedValue(configured)
    render(
      <ThemeProvider attribute='class'>
        <SettingsPage />
      </ThemeProvider>,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Translation' }))
    const reasoning = screen.getByRole('switch', { name: 'Enable reasoning' })
    expect(reasoning).not.toHaveAttribute('aria-disabled', 'true')
    expect(screen.getByRole('switch', { name: 'Vision' })).not.toHaveAttribute(
      'aria-disabled',
      'true',
    )
    await user.click(reasoning)
    await waitFor(() =>
      expect(save).toHaveBeenCalledWith(
        expect.objectContaining({
          translation: expect.objectContaining({
            generation: expect.objectContaining({ reasoning: true }),
          }),
        }),
        configured.providers,
        configured.typesetting,
      ),
    )
  })

  it('keeps a configured provider credential private and allows clearing it', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    const providers = {
      entries: [
        {
          name: 'DeepL',
          config: { provider: 'deepl' as const, settings: { base_url: null } },
          credential: { ...emptyCredential(), configured: true },
        },
      ],
    }
    render(<ProviderPreferences value={providers} onChange={onChange} />)

    const input = screen.getByLabelText('DeepL credential')
    expect(screen.getByText('Credential')).toHaveAttribute('for', input.id)
    expect(input).toHaveAttribute('type', 'text')
    expect(input).toHaveAttribute('autocomplete', 'off')
    expect(input).toHaveAttribute('autocapitalize', 'none')
    expect(input).toHaveAttribute('spellcheck', 'false')
    expect(input).toHaveClass(
      '[-webkit-text-security:disc]',
      '[&::placeholder]:[-webkit-text-security:none]',
    )
    expect(input).toHaveValue('')
    expect(input).toHaveAttribute('placeholder', 'Configured')
    expect(
      screen.queryByRole('button', { name: 'Reveal DeepL credential' }),
    ).not.toBeInTheDocument()

    const clear = screen.getByRole('button', { name: 'Clear DeepL credential' })
    expect(clear.querySelector('.lucide-eraser')).toBeInTheDocument()
    expect(clear).not.toHaveClass('text-destructive')
    await user.click(clear)
    expect(onChange).toHaveBeenCalledWith({
      entries: [
        expect.objectContaining({
          credential: { ...emptyCredential(), clear: true },
        }),
      ],
    })
  })

  it('shows Linux environment credentials as read-only', () => {
    const providers = {
      entries: [
        {
          name: 'OpenAI',
          config: { provider: 'openai' as const, settings: {} },
          credential: {
            configured: true,
            editable: false,
            environment_variable: 'OPENAI_API_KEY',
            value: null,
            clear: false,
          },
        },
      ],
    }
    render(<ProviderPreferences value={providers} onChange={vi.fn()} />)

    expect(screen.getByText('OPENAI_API_KEY')).toBeInTheDocument()
    expect(screen.getByLabelText('OpenAI credential')).toBeDisabled()
    expect(screen.getByLabelText('OpenAI credential')).toHaveAttribute('placeholder', 'Configured')
    expect(
      screen.queryByRole('button', { name: 'Clear OpenAI credential' }),
    ).not.toBeInTheDocument()
  })

  it('preserves a credential draft and focus when autosave finishes', async () => {
    installProject()
    const user = userEvent.setup()
    useKoharuStore.setState({ settingsOpen: true })
    const save = vi
      .spyOn(commands, 'savePreferences')
      .mockImplementation(async (pipeline, providers, typesetting) => ({
        ...preferences,
        pipeline,
        providers: {
          entries: providers.entries.map((entry) => ({
            ...entry,
            credential: entry.credential?.value
              ? { ...entry.credential, configured: true, value: null, clear: false }
              : entry.credential,
          })),
        },
        typesetting,
      }))
    render(
      <ThemeProvider attribute='class'>
        <SettingsPage />
      </ThemeProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Providers' }))
    const input = screen.getByLabelText('DeepL credential')
    await user.type(input, 's')
    await waitFor(() => expect(save).toHaveBeenCalled())

    await waitFor(() => {
      expect(screen.getByLabelText('DeepL credential')).toBe(input)
      expect(input).toHaveFocus()
      expect(input).toHaveValue('s')
    })
  })

  it('clamps a threshold typed past its bounds before saving it', async () => {
    installProject()
    useKoharuStore.setState({ settingsOpen: true })
    const save = vi.spyOn(commands, 'savePreferences').mockResolvedValue(preferences)
    render(
      <ThemeProvider attribute='class'>
        <SettingsPage />
      </ThemeProvider>,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Pipeline' }))
    fireEvent.change(screen.getByRole('spinbutton', { name: 'Text threshold' }), {
      target: { value: '15' },
    })
    await waitFor(() =>
      expect(save).toHaveBeenCalledWith(
        expect.objectContaining({
          processor: expect.objectContaining({
            'koharu-layout-rfdetr-seg-2xl': expect.objectContaining({ text_threshold: 1 }),
          }),
        }),
        preferences.providers,
        preferences.typesetting,
      ),
    )
  })

  it('saves a newer OpenRouter selection after an in-flight provider save', async () => {
    installProject()
    const user = userEvent.setup()
    const configured: Preferences = {
      ...preferences,
      providers: {
        entries: [
          ...preferences.providers.entries,
          {
            name: 'OpenRouter',
            config: { provider: 'openrouter', settings: {} },
            credential: emptyCredential(),
          },
        ],
      },
    }
    useKoharuStore.setState({
      settingsOpen: true,
      preferences: configured,
      translationModels: [
        ...useKoharuStore.getState().translationModels,
        {
          provider: 'openrouter',
          model: 'openrouter/auto',
          name: 'OpenRouter Auto',
          quantizations: [],
          vision: true,
          reasoning: true,
        },
      ],
    })
    let resolveFirst: ((value: Preferences) => void) | undefined
    const first = new Promise<Preferences>((resolve) => {
      resolveFirst = resolve
    })
    let firstResult: Preferences | undefined
    let invocation = 0
    const save = vi
      .spyOn(commands, 'savePreferences')
      .mockImplementation(async (pipeline, providers, typesetting) => {
        const saved = { ...configured, pipeline, providers, typesetting }
        if (invocation++ === 0) {
          firstResult = saved
          return first
        }
        return saved
      })
    render(
      <ThemeProvider attribute='class'>
        <SettingsPage />
      </ThemeProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Providers' }))
    await user.type(screen.getByLabelText('OpenRouter credential'), 'secret')
    await waitFor(() => expect(save).toHaveBeenCalledTimes(1))
    await user.click(screen.getByRole('button', { name: 'Translation' }))
    await user.click(screen.getByLabelText('Translation model'))
    await user.click(screen.getByRole('button', { name: 'Use OpenRouter Auto from OpenRouter' }))
    await user.click(screen.getByRole('button', { name: 'Back to editor' }))

    expect(save).toHaveBeenCalledTimes(1)
    await act(async () => {
      resolveFirst?.(firstResult!)
      await first
    })
    await waitFor(() => expect(save).toHaveBeenCalledTimes(2))
    await waitFor(() => {
      expect(useKoharuStore.getState().settingsOpen).toBe(false)
      expect(useKoharuStore.getState().preferences?.pipeline.translation.model).toMatchObject({
        provider: 'openrouter',
        model: 'openrouter/auto',
      })
    })
  })

  it('adds and removes default font families from typesetting settings', async () => {
    installProject()
    const user = userEvent.setup()
    useKoharuStore.setState({ settingsOpen: true })
    queryClient.setQueryData(fontsKey, [
      {
        name: 'Noto Sans',
        metadata: {
          primary_script: 'latn',
          scripts: ['latn'],
          languages: ['en'],
          category: 'SANS_SERIF',
          classifications: ['sans-serif'],
          use_cases: ['body-text'],
        },
        sources: ['system'],
        faces: [
          {
            postscript_name: 'NotoSans-Regular',
            weight: 400,
            weight_range: null,
            style: 'normal',
          },
        ],
      },
      {
        name: 'Arial',
        metadata: {
          primary_script: 'latn',
          scripts: ['latn'],
          languages: ['en'],
          category: 'SANS_SERIF',
          classifications: ['sans-serif'],
          use_cases: ['body-text'],
        },
        sources: ['system'],
        faces: [
          {
            postscript_name: 'ArialMT',
            weight: 400,
            weight_range: null,
            style: 'normal',
          },
        ],
      },
    ])
    const save = vi
      .spyOn(commands, 'savePreferences')
      .mockImplementation(async (pipeline, providers, typesetting) => ({
        ...preferences,
        pipeline,
        providers,
        typesetting,
      }))
    render(
      <ThemeProvider attribute='class'>
        <SettingsPage />
      </ThemeProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Typesetting' }))
    expect(screen.getByRole('heading', { level: 2, name: 'Typesetting' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Add font family' }))
    await user.click(await screen.findByRole('option', { name: 'Arial, System' }))

    await waitFor(() =>
      expect(save).toHaveBeenLastCalledWith(preferences.pipeline, preferences.providers, {
        font_families: ['Noto Sans', 'Arial'],
      }),
    )
    await user.click(screen.getByRole('button', { name: 'Remove Noto Sans' }))
    await waitFor(() =>
      expect(save).toHaveBeenLastCalledWith(preferences.pipeline, preferences.providers, {
        font_families: ['Arial'],
      }),
    )
  })

  it('shows model resources in the left sidebar footer', () => {
    useKoharuStore.setState({
      resources: {
        process_memory: 2 * 1024 ** 3,
        system_memory: 64 * 1024 ** 3,
        process_cpu: 8,
        devices: [
          {
            name: 'GPU',
            selected: true,
            memory_budget: 16 * 1024 ** 3,
            memory_used: 6 * 1024 ** 3,
            memory_scope: 'device',
            utilization: 40,
          },
        ],
      },
    })
    render(<ResourceMonitor />)
    expect(screen.getByText('8%')).toBeInTheDocument()
    expect(screen.getByText('3%')).toBeInTheDocument()
  })

  it('says which processes a vram reading covers', async () => {
    useKoharuStore.setState({
      resources: {
        process_memory: 2 * 1024 ** 3,
        system_memory: 64 * 1024 ** 3,
        process_cpu: 8,
        devices: [
          {
            name: 'GPU',
            selected: true,
            memory_budget: 16 * 1024 ** 3,
            memory_used: 6 * 1024 ** 3,
            memory_scope: 'device',
            utilization: 40,
          },
        ],
      },
    })
    render(<ResourceMonitor />)
    fireEvent.click(screen.getAllByRole('button')[0])
    expect(
      await screen.findByText('6.0/16.0 GB VRAM · this device, all applications'),
    ).toBeInTheDocument()
  })

  it('tracks arbitrary concurrent commands until each one settles', async () => {
    const first = Promise.withResolvers<string>()
    const second = Promise.withResolvers<string>()
    const command = vi
      .fn<(name: string, count: number) => Promise<string>>()
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise)
    const { result } = renderHook(
      () => useCommand(['rebuild-index'], command, 'Rebuilding index…'),
      {
        wrapper: ({ children }) => (
          <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
        ),
      },
    )
    render(<ActivityCenter />)

    act(() => {
      result.current.run('first', 2)
      result.current.run('second', 3)
    })
    await waitFor(() => expect(screen.getAllByRole('status')).toHaveLength(2))
    expect(screen.getAllByText('Rebuilding index…')).toHaveLength(2)
    expect(command).toHaveBeenNthCalledWith(1, 'first', 2)
    expect(command).toHaveBeenNthCalledWith(2, 'second', 3)
    expect(result.current.busy).toBe(true)

    await act(async () => first.resolve('done'))
    await waitFor(() => expect(screen.getAllByRole('status')).toHaveLength(1))
    expect(result.current.busy).toBe(true)

    await act(async () => second.reject(new Error('Index failed')))
    await waitFor(() => expect(screen.queryByRole('status')).not.toBeInTheDocument())
    expect(result.current.busy).toBe(false)
    expect(screen.queryByRole('complementary', { name: 'Activity' })).not.toBeInTheDocument()
  })

  it('keeps running work visible and stoppable', async () => {
    installProject()
    useKoharuStore.setState({
      jobs: {
        job: {
          state: 'running',
          id: 'job',
          kind: 'processing',
          completed: 1,
          total: 4,
          page: 'page',
          stage: 'ocr',
          model: 'manga-ocr',
          error: null,
        },
      },
    })
    // A real page label is a filename, which says nothing about position in
    // the run, so the row numbers it too.
    queryClient.setQueryData(pagesKey, [
      {
        id: 'page',
        label: 'cover.png',
        size: { width: 1000, height: 1500 },
        source_asset: 'source',
        layer_count: 1,
      },
    ])
    const stop = vi.spyOn(commands, 'stopJob').mockResolvedValue(null)
    render(<ActivityCenter />)
    expect(screen.getByText('25%')).toBeInTheDocument()
    // Separate elements, so a long label truncates without taking the model.
    expect(screen.getByText('Page 1: cover.png')).toBeInTheDocument()
    expect(screen.getByText('manga-ocr')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Stop' }))
    await waitFor(() => expect(stop).toHaveBeenCalledWith('job'))
  })

  it('combines concurrent downloads into one progress bar', () => {
    useKoharuStore.setState({
      downloads: {
        one: {
          id: 1,
          state: 'running',
          name: 'one.ttf',
          completed: 25,
          total: 100,
          error: null,
        },
        two: {
          id: 2,
          state: 'running',
          name: 'two.ttf',
          completed: 75,
          total: 100,
          error: null,
        },
      },
    })

    render(<ActivityCenter />)

    expect(screen.getByText('Downloading 2 files')).toBeInTheDocument()
    expect(screen.getByText('50%')).toBeInTheDocument()
    expect(screen.getAllByRole('progressbar')).toHaveLength(1)
    expect(screen.queryByText('one.ttf')).not.toBeInTheDocument()
    expect(screen.queryByText('two.ttf')).not.toBeInTheDocument()
  })
})
