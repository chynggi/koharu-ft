import { QueryClientProvider } from '@tanstack/react-query'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { CanvasWorkspace } from '@/components/editor/CanvasWorkspace'
import { pageKey, pagesKey, projectKey, queryClient } from '@/lib/queries'
import { useKoharuStore } from '@/lib/store'
import { commands, type BrushRamp, type Layer } from '@koharu/bridge/protocol'
import { TooltipProvider } from '@koharu/ui/components/tooltip'

const canvas = vi.hoisted(() => ({
  resize: vi.fn(),
  setView: vi.fn(),
  stageManifest: vi.fn(),
  installResource: vi.fn(),
  activateFrame: vi.fn(),
  activatePage: vi.fn(),
  clear: vi.fn(),
  previewOpacity: vi.fn(),
  beginTransform: vi.fn(),
  updateTransform: vi.fn(),
  finishTransform: vi.fn(),
  cancelTransform: vi.fn(),
  beginStroke: vi.fn(),
  extendStroke: vi.fn(),
  finishStroke: vi.fn(),
  cancelStroke: vi.fn(),
  beginShape: vi.fn(),
  updateShape: vi.fn(),
  finishShape: vi.fn(),
  cancelShape: vi.fn(),
  sampleColor: vi.fn(),
  dispose: vi.fn(),
}))

const canvasState = vi.hoisted(() => ({
  canvas,
  error: null as Error | null,
  generation: 1 as number | null,
  hasFrame: true,
  retry: vi.fn(),
  status: 'ready' as 'loading' | 'switching' | 'ready' | 'recovering' | 'error',
}))
const prefetchCanvasPages = vi.hoisted(() => vi.fn(async () => []))

vi.mock('@/components/editor/useCanvas', () => ({
  useCanvas: () => canvasState,
}))
vi.mock('@koharu/bridge/canvas', () => ({
  prefetchCanvasPages,
  workspaceColor: () => [245, 245, 245],
}))

const layer: Layer = {
  type: 'image',
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
  image: 'image',
}

const paintLayer: Layer = {
  type: 'raster',
  id: 'paint',
  parent: 'page',
  visibility: { visible: true, opacity: 1 },
  image: 'paint-image',
  name: 'Paint 1',
  kind: 'paint',
}

const textLayer: Layer = {
  type: 'text',
  id: 'sound-effect',
  parent: 'text-group',
  geometry: {
    points: [
      { x: 10, y: 20 },
      { x: 110, y: 20 },
      { x: 110, y: 70 },
      { x: 10, y: 70 },
    ],
  },
  angle_degrees: null,
  visibility: { visible: true, opacity: 1 },
  content: {
    id: 'sound-effect-content',
    source: { text: 'ドン', language: 'ja' },
    translation: { text: 'BOOM', language: 'en' },
    role: null,
    source_region: null,
  },
  typography: null,
  layout: 'paragraph',
  automatic_region: null,
}

let nextAnimationFrame = 1
let animationFrames = new Map<number, FrameRequestCallback>()

beforeEach(() => {
  nextAnimationFrame = 1
  animationFrames = new Map()
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
    const frame = nextAnimationFrame++
    animationFrames.set(frame, callback)
    return frame
  })
  vi.stubGlobal('cancelAnimationFrame', (frame: number) => animationFrames.delete(frame))
  canvasState.error = null
  canvasState.generation = 1
  canvasState.hasFrame = true
  canvasState.status = 'ready'
  prefetchCanvasPages.mockClear()
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

function flushAnimationFrames() {
  const pending = [...animationFrames.values()]
  animationFrames.clear()
  for (const callback of pending) callback(performance.now())
}

function installProject() {
  const page = {
    id: 'page',
    label: 'Page',
    size: { width: 1000, height: 1000 },
    layers: [layer],
    regions: [],
  }
  queryClient.setQueryData(projectKey, {
    name: 'Book',
    revision: 1,
    active_page: 'page',
    can_undo: false,
    can_redo: false,
  })
  queryClient.setQueryData(pagesKey, [])
  queryClient.setQueryData(pageKey, page)
  useKoharuStore.setState({ selectedLayers: [], copiedTextLayers: [], tool: 'select' })
  useKoharuStore.setState({
    canvasPage: 'page',
    canvasRevision: 1,
    canvasGeneration: 1,
    canvasSize: [1000, 1000],
  })
}

function renderWorkspace() {
  render(
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <CanvasWorkspace />
      </TooltipProvider>
    </QueryClientProvider>,
  )
  const surface = screen.getByLabelText('Koharu canvas')
  Object.defineProperty(surface, 'getBoundingClientRect', {
    value: () => ({ x: 10, y: 20, width: 800, height: 600 }),
  })
  return surface
}

describe('canvas interaction adapter', () => {
  it('prefetches only after the authoritative canvas page and generation are active', async () => {
    installProject()
    queryClient.setQueryData(pagesKey, [
      {
        id: 'page',
        label: 'Page',
        size: { width: 1000, height: 1000 },
        source_asset: null,
        chapter_start: false,
        layer_count: 1,
      },
      {
        id: 'next',
        label: 'Next',
        size: { width: 1000, height: 1000 },
        source_asset: null,
        chapter_start: false,
        layer_count: 1,
      },
    ])
    useKoharuStore.setState({ canvasPage: 'previous' })
    renderWorkspace()

    await Promise.resolve()
    expect(prefetchCanvasPages).not.toHaveBeenCalled()

    act(() => useKoharuStore.setState({ canvasPage: 'page' }))
    await waitFor(() => expect(prefetchCanvasPages).toHaveBeenCalledWith(['next']))
  })

  it('renders an accessible browser canvas and keeps camera updates local', async () => {
    installProject()
    const surface = renderWorkspace()
    expect(screen.getByTestId('webgpu-canvas')).toBeInstanceOf(HTMLCanvasElement)

    fireEvent.wheel(surface, { clientX: 100, clientY: 100, deltaY: 4 })
    fireEvent.wheel(surface, { clientX: 100, clientY: 100, deltaY: 6 })

    await waitFor(() => expect(canvas.setView).toHaveBeenLastCalledWith(1, [0, -10]))
  })

  it('preserves a manual camera when a newer canvas generation becomes ready', () => {
    installProject()
    renderWorkspace()
    const camera = { zoom: 2, translation: [-120, -80] as [number, number], fitted: false }

    act(() => useKoharuStore.setState({ camera }))
    act(() => {
      canvasState.status = 'switching'
      useKoharuStore.setState({ canvasGeneration: 2 })
    })
    act(() => {
      canvasState.status = 'ready'
      canvasState.generation = 2
      useKoharuStore.setState({ canvasRevision: 2 })
    })

    expect(useKoharuStore.getState().camera).toEqual(camera)
  })

  it('suppresses Alt browser handling while an editor input retains focus', () => {
    installProject()
    renderWorkspace()
    const input = document.createElement('input')
    document.body.append(input)
    input.focus()

    const event = new KeyboardEvent('keydown', { key: 'Alt', bubbles: true, cancelable: true })
    input.dispatchEvent(event)

    expect(event.defaultPrevented).toBe(true)
    input.remove()
  })

  it('copies and pastes selected text layers with keyboard shortcuts', async () => {
    installProject()
    queryClient.setQueryData(pageKey, {
      id: 'page',
      label: 'Page',
      size: { width: 1000, height: 1000 },
      layers: [textLayer],
      regions: [],
    })
    useKoharuStore.setState({ selectedLayers: [textLayer.id] })
    const paste = vi.spyOn(commands, 'pasteTextLayers').mockResolvedValue(['pasted-layer'])
    vi.spyOn(commands, 'getProject').mockResolvedValue({
      name: 'Book',
      revision: 2,
      active_page: 'page',
      can_undo: true,
      can_redo: false,
    })
    vi.spyOn(commands, 'getPages').mockResolvedValue([])
    vi.spyOn(commands, 'getPage').mockResolvedValue({
      id: 'page',
      label: 'Page',
      size: { width: 1000, height: 1000 },
      layers: [],
      regions: [],
    })
    renderWorkspace()

    fireEvent.keyDown(window, { key: 'c', ctrlKey: true })
    expect(useKoharuStore.getState().copiedTextLayers).toEqual([textLayer.id])

    fireEvent.keyDown(window, { key: 'v', ctrlKey: true })

    await waitFor(() => expect(paste).toHaveBeenCalledWith([textLayer.id], 'page'))
    await waitFor(() => expect(useKoharuStore.getState().selectedLayers).toEqual(['pasted-layer']))
  })

  it('duplicates selected text layers with Ctrl+D without using the clipboard', async () => {
    installProject()
    queryClient.setQueryData(pageKey, {
      id: 'page',
      label: 'Page',
      size: { width: 1000, height: 1000 },
      layers: [textLayer],
      regions: [],
    })
    useKoharuStore.setState({ selectedLayers: [textLayer.id], copiedTextLayers: [] })
    const paste = vi.spyOn(commands, 'pasteTextLayers').mockResolvedValue(['duplicated-layer'])
    vi.spyOn(commands, 'getProject').mockResolvedValue({
      name: 'Book',
      revision: 2,
      active_page: 'page',
      can_undo: true,
      can_redo: false,
    })
    vi.spyOn(commands, 'getPages').mockResolvedValue([])
    vi.spyOn(commands, 'getPage').mockResolvedValue({
      id: 'page',
      label: 'Page',
      size: { width: 1000, height: 1000 },
      layers: [],
      regions: [],
    })
    renderWorkspace()

    const event = new KeyboardEvent('keydown', {
      key: 'd',
      ctrlKey: true,
      bubbles: true,
      cancelable: true,
    })
    window.dispatchEvent(event)

    await waitFor(() => expect(paste).toHaveBeenCalledWith([textLayer.id], 'page'))
    await waitFor(() =>
      expect(useKoharuStore.getState().selectedLayers).toEqual(['duplicated-layer']),
    )
    // Duplicating must not clobber the clipboard.
    expect(useKoharuStore.getState().copiedTextLayers).toEqual([])
    expect(event.defaultPrevented).toBe(true)
  })

  it('leaves Ctrl+D to the browser when no text layer is selected', () => {
    installProject()
    queryClient.setQueryData(pageKey, {
      id: 'page',
      label: 'Page',
      size: { width: 1000, height: 1000 },
      layers: [layer],
      regions: [],
    })
    // An image layer is selected: duplication would find no text layer and do
    // nothing, so the shortcut must not be swallowed.
    useKoharuStore.setState({ selectedLayers: [layer.id] })
    const paste = vi.spyOn(commands, 'pasteTextLayers')
    renderWorkspace()

    const event = new KeyboardEvent('keydown', {
      key: 'd',
      ctrlKey: true,
      bubbles: true,
      cancelable: true,
    })
    window.dispatchEvent(event)

    expect(event.defaultPrevented).toBe(false)
    expect(paste).not.toHaveBeenCalled()
  })

  it('allows bare tool shortcuts from a slider while leaving control keys alone', () => {
    installProject()
    renderWorkspace()
    const input = document.createElement('input')
    input.type = 'range'
    document.body.append(input)
    input.focus()
    const undo = vi.spyOn(commands, 'undo').mockResolvedValue(null)
    const remove = vi.spyOn(commands, 'deleteLayers').mockResolvedValue(null)
    act(() => useKoharuStore.setState({ selectedLayers: ['element'] }))

    for (const options of [
      { key: ' ', code: 'Space' },
      { key: 'ArrowLeft' },
      { key: 'Delete' },
      { key: 'z', ctrlKey: true },
    ]) {
      const event = new KeyboardEvent('keydown', { ...options, bubbles: true, cancelable: true })
      fireEvent(input, event)
      expect(event.defaultPrevented).toBe(false)
    }
    expect(undo).not.toHaveBeenCalled()
    expect(remove).not.toHaveBeenCalled()
    expect(useKoharuStore.getState().selectedLayers).toEqual(['element'])

    fireEvent.keyDown(input, { key: 't' })
    expect(useKoharuStore.getState().tool).toBe('text')
    input.remove()
  })

  it('keeps text entry, composition and handled keys out of tool shortcuts', () => {
    installProject()
    renderWorkspace()
    for (const tag of ['input', 'textarea', 'select']) {
      const control = document.createElement(tag)
      document.body.append(control)
      fireEvent.keyDown(control, { key: 't' })
      expect(useKoharuStore.getState().tool).toBe('select')
      control.remove()
    }
    fireEvent.keyDown(window, { key: 't', isComposing: true })
    expect(useKoharuStore.getState().tool).toBe('select')
    const handled = new KeyboardEvent('keydown', { key: 't', cancelable: true })
    handled.preventDefault()
    fireEvent(window, handled)
    expect(useKoharuStore.getState().tool).toBe('select')
  })

  it('announces WebGPU startup failures and offers recovery', () => {
    installProject()
    canvasState.status = 'error'
    canvasState.error = new Error('No compatible WebGPU adapter was found.')
    renderWorkspace()
    expect(screen.getByRole('alert')).toHaveTextContent('WebGPU canvas unavailable')
    expect(screen.getByRole('alert')).toHaveTextContent('No compatible WebGPU adapter was found.')
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }))
    expect(canvasState.retry).toHaveBeenCalledOnce()
  })

  it('previews brush input locally and sends only the durable paint commit to Rust', async () => {
    installProject()
    useKoharuStore.setState({ tool: 'draw', brush: { diameter: 48, color: '#FFFFFF', ramp: null } })
    const commit = vi
      .spyOn(commands, 'commitPaint')
      .mockResolvedValue({ revision: 2, layer: 'paint' })
    const surface = renderWorkspace()
    expect(surface).toHaveStyle({ cursor: 'none' })

    fireEvent.pointerDown(surface, { button: 0, pointerId: 7, clientX: 30, clientY: 40 })
    fireEvent.pointerMove(surface, { pointerId: 7, clientX: 55, clientY: 65 })
    fireEvent.pointerUp(surface, { pointerId: 7, clientX: 58, clientY: 70 })

    await waitFor(() => expect(commit).toHaveBeenCalledOnce())
    expect(canvas.beginStroke).toHaveBeenCalledWith({
      kind: 'paint',
      layer: null,
      point: { x: 20, y: 20 },
      diameter: 48,
      color: [255, 255, 255, 255],
      ramp: null,
    })
    expect(canvas.extendStroke).toHaveBeenCalledWith(expect.arrayContaining([{ x: 45, y: 45 }]))
    expect(canvas.finishStroke).toHaveBeenCalledOnce()
    expect(commit).toHaveBeenCalledWith(1, null, expect.arrayContaining([{ x: 45, y: 45 }]), {
      diameter: 48,
      color: [255, 255, 255, 255],
      ramp: null,
    })
  })

  it('sends the active brush ramp to both the preview and the paint commit', async () => {
    installProject()
    const ramp: BrushRamp = {
      kind: 'along_stroke',
      from: [255, 0, 0, 255],
      to: [0, 0, 255, 255],
    }
    useKoharuStore.setState({
      tool: 'draw',
      brush: { diameter: 48, color: '#FFFFFF', ramp },
    })
    const commit = vi
      .spyOn(commands, 'commitPaint')
      .mockResolvedValue({ revision: 2, layer: 'paint' })
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 9, clientX: 30, clientY: 40 })
    fireEvent.pointerMove(surface, { pointerId: 9, clientX: 55, clientY: 65 })
    fireEvent.pointerUp(surface, { pointerId: 9, clientX: 58, clientY: 70 })

    await waitFor(() => expect(commit).toHaveBeenCalledOnce())
    expect(canvas.beginStroke).toHaveBeenCalledWith({
      kind: 'paint',
      layer: null,
      point: { x: 20, y: 20 },
      diameter: 48,
      color: [255, 255, 255, 255],
      ramp,
    })
    // The preview and the committed raster must agree: a ramp sent only to
    // `beginStroke` would ramp on screen and land flat in the document.
    expect(commit).toHaveBeenCalledWith(1, null, expect.arrayContaining([{ x: 45, y: 45 }]), {
      diameter: 48,
      color: [255, 255, 255, 255],
      ramp,
    })
  })
  it.each([
    {
      keys: ['ArrowRight'],
      repeat: 'ArrowRight',
      release: 'ArrowRight',
      expected: { x: 32, y: 40 },
    },
    {
      keys: ['ArrowUp', 'ArrowRight'],
      repeat: 'ArrowRight',
      release: 'ArrowRight',
      expected: { x: 32, y: 36 },
    },
    {
      keys: ['ArrowRight', 'ArrowUp'],
      repeat: 'ArrowUp',
      release: 'ArrowRight',
      expected: { x: 33, y: 37 },
    },
    {
      keys: ['ArrowLeft', 'ArrowRight'],
      repeat: 'ArrowRight',
      release: 'ArrowRight',
      expected: { x: 28, y: 40 },
    },
    {
      keys: ['ArrowUp', 'ArrowDown'],
      repeat: 'ArrowDown',
      release: 'ArrowDown',
      expected: { x: 30, y: 38 },
    },
  ])('text box nudge with arrow key: $keys', async ({ keys, repeat, release, expected }) => {
    vi.useFakeTimers()
    installProject()

    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [
        {
          type: 'text',
          id: 'element',
          parent: 'page',
          geometry: layer.geometry,
          angle_degrees: 0,
          visibility: { visible: true, opacity: 1 },
          content: {
            id: 'content',
            source: { text: 'Source', language: 'en' },
            translation: { text: 'Rendered', language: null },
            role: null,
            source_region: null,
          },
          typography: null,
          layout: 'paragraph',
          automatic_region: null,
        },
      ],
    }))

    useKoharuStore.setState({
      selectedLayers: ['element'],
      layerFrames: {
        element: { x: 30, y: 40, width: 50, height: 20, angle_degrees: 0 },
      },
    })

    const commit = vi.spyOn(commands, 'commitTransform').mockResolvedValue(2)
    const surface = renderWorkspace()

    act(() =>
      useKoharuStore.setState({
        camera: { zoom: 4, translation: [0, 0], fitted: false },
      }),
    )

    for (const key of keys) fireEvent.keyDown(surface, { key })

    act(() => {
      vi.advanceTimersByTime(300)
    })

    fireEvent.keyDown(surface, { key: repeat, repeat: true })

    fireEvent.keyUp(surface, { key: release })
    const remaining = keys.filter((key) => key !== release)

    if (remaining.length) {
      expect(commit).not.toHaveBeenCalled()
    }

    act(() => {
      vi.advanceTimersByTime(30)
    })

    for (const key of remaining) {
      fireEvent.keyUp(surface, { key })
    }

    vi.useRealTimers()

    await waitFor(() =>
      expect(commit).toHaveBeenCalledWith(1, [
        {
          element: 'element',
          frame: {
            ...expected,
            width: 50,
            height: 20,
            angle_degrees: 0,
          },
        },
      ]),
    )
  })
  it.each([
    ['revision', { canvasRevision: 2 }],
    ['generation', { canvasGeneration: 2 }],
  ])('cancels an active gesture when the canvas %s changes', async (_name, update) => {
    installProject()
    useKoharuStore.setState({ tool: 'draw', brush: { diameter: 48, color: '#FFFFFF', ramp: null } })
    const commit = vi
      .spyOn(commands, 'commitPaint')
      .mockResolvedValue({ revision: 2, layer: 'paint' })
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 8, clientX: 30, clientY: 40 })
    expect(canvas.beginStroke).toHaveBeenCalledOnce()

    act(() => {
      useKoharuStore.setState(update)
    })

    await waitFor(() => expect(canvas.cancelStroke).toHaveBeenCalledOnce())
    fireEvent.pointerUp(surface, { pointerId: 8, clientX: 30, clientY: 40 })
    expect(canvas.finishStroke).not.toHaveBeenCalled()
    expect(commit).not.toHaveBeenCalled()
  })

  it('uses rendered text bounds for hit testing and semantic transforms', async () => {
    installProject()
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [
        {
          type: 'text',
          id: 'element',
          parent: 'page',
          geometry: layer.geometry,
          angle_degrees: 0,
          visibility: { visible: true, opacity: 1 },
          content: {
            id: 'content',
            source: { text: 'Source', language: 'en' },
            translation: { text: 'Rendered', language: null },
            role: null,
            source_region: null,
          },
          typography: null,
          layout: 'paragraph',
          automatic_region: null,
        },
      ],
    }))
    useKoharuStore.setState({
      layerFrames: {
        element: { x: 30, y: 40, width: 50, height: 20, angle_degrees: 0 },
      },
    })
    const commit = vi.spyOn(commands, 'commitTransform').mockResolvedValue(2)
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 9, clientX: 50, clientY: 60 })
    fireEvent.pointerMove(surface, { pointerId: 9, clientX: 70, clientY: 80 })
    fireEvent.pointerUp(surface, { pointerId: 9, clientX: 70, clientY: 80 })

    await waitFor(() => expect(commit).toHaveBeenCalledOnce())
    expect(useKoharuStore.getState().selectedLayers).toEqual(['element'])
    expect(canvas.beginTransform).toHaveBeenCalledWith([
      {
        element: 'element',
        frame: { x: 30, y: 40, width: 50, height: 20, angle_degrees: 0 },
      },
    ])
    expect(canvas.updateTransform).toHaveBeenCalledWith(
      expect.arrayContaining([
        expect.objectContaining({
          element: 'element',
          frame: expect.objectContaining({ x: 50, y: 60 }),
        }),
      ]),
    )
    expect(canvas.finishTransform).toHaveBeenCalledOnce()
    expect(commit).toHaveBeenCalledWith(
      1,
      expect.arrayContaining([
        expect.objectContaining({
          element: 'element',
          frame: expect.objectContaining({ x: 50, y: 60 }),
        }),
      ]),
    )
  })

  it('resizes a selected layer through Koharu selection controls', async () => {
    installProject()
    useKoharuStore.setState({ selectedLayers: ['element'] })
    const commit = vi.spyOn(commands, 'commitTransform').mockResolvedValue(2)
    renderWorkspace()
    Object.defineProperty(screen.getByTestId('canvas-overlay'), 'getBoundingClientRect', {
      value: () => ({ x: 10, y: 20, width: 800, height: 600 }),
    })
    const handle = document.querySelector<HTMLElement>('[data-resize-handle="e"]')!

    fireEvent.pointerDown(handle, { button: 0, pointerId: 10, clientX: 120, clientY: 65 })
    fireEvent.pointerMove(handle, { pointerId: 10, clientX: 140, clientY: 65 })
    fireEvent.pointerUp(handle, { pointerId: 10, clientX: 140, clientY: 65 })

    await waitFor(() => expect(commit).toHaveBeenCalledOnce())
    expect(canvas.beginTransform).toHaveBeenCalledWith([
      { element: 'element', frame: { x: 10, y: 20, width: 100, height: 50, angle_degrees: 0 } },
    ])
    expect(canvas.updateTransform).toHaveBeenCalledWith(
      expect.arrayContaining([
        {
          element: 'element',
          frame: { x: 10, y: 20, width: 120, height: 50, angle_degrees: 0 },
        },
      ]),
    )
    expect(commit).toHaveBeenCalledWith(
      1,
      expect.arrayContaining([
        {
          element: 'element',
          frame: { x: 10, y: 20, width: 120, height: 50, angle_degrees: 0 },
        },
      ]),
    )
  })

  it('shows the automatic region behind the selected text controls', () => {
    installProject()
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [
        {
          type: 'text',
          id: 'element',
          parent: 'page',
          geometry: null,
          angle_degrees: null,
          visibility: { visible: true, opacity: 1 },
          content: {
            id: 'content',
            source: { text: 'Source', language: 'en' },
            translation: { text: 'Rendered', language: null },
            role: null,
            source_region: null,
          },
          typography: null,
          layout: 'paragraph',
          automatic_region: 'bubble',
        },
      ],
      regions: [
        {
          id: 'bubble',
          parent: 'page',
          geometry: {
            points: [
              { x: 20, y: 30 },
              { x: 100, y: 30 },
              { x: 100, y: 90 },
              { x: 20, y: 90 },
            ],
          },
          kind: 'bubble',
          label: null,
        },
      ],
    }))
    useKoharuStore.setState({
      selectedLayers: ['element'],
      layerFrames: {
        element: { x: 30, y: 40, width: 50, height: 20, angle_degrees: 0 },
      },
    })

    renderWorkspace()

    expect(screen.getByTestId('text-fit-region').querySelector('polygon')).toHaveAttribute(
      'points',
      '20,30 100,30 100,90 20,90',
    )
  })

  it('targets the selected raster layer with the eraser', async () => {
    installProject()
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [...page.layers, paintLayer],
    }))
    useKoharuStore.setState({ tool: 'eraser', selectedLayers: ['paint'] })
    const commit = vi
      .spyOn(commands, 'commitErase')
      .mockResolvedValue({ revision: 2, layer: 'paint' })
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 11, clientX: 30, clientY: 40 })
    fireEvent.pointerUp(surface, { pointerId: 11, clientX: 30, clientY: 40 })

    await waitFor(() => expect(commit).toHaveBeenCalledOnce())
    expect(canvas.beginStroke).toHaveBeenCalledWith({
      kind: 'erase',
      layer: 'paint',
      point: { x: 20, y: 20 },
      diameter: 48,
    })
    expect(commit).toHaveBeenCalledWith(1, 'paint', expect.arrayContaining([{ x: 20, y: 20 }]), 48)
  })

  it('fills the selected raster layer at the clicked page point with the fill colour', async () => {
    installProject()
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [...page.layers, paintLayer],
    }))
    useKoharuStore.setState({ tool: 'fill', selectedLayers: ['paint'], fillColor: '#102030' })
    const fill = vi.spyOn(commands, 'fillRaster').mockResolvedValue({ revision: 2, layer: 'paint' })
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 30, clientX: 30, clientY: 40 })

    await waitFor(() => expect(fill).toHaveBeenCalledOnce())
    // (30, 40) is a client point; the fill command takes page pixels, which the
    // identity camera maps to (20, 20) — screen space would be wrong here.
    expect(fill).toHaveBeenCalledWith(1, 'paint', { x: 20, y: 20 }, [16, 32, 48, 255])
  })

  it('does not fill without a selected raster layer', () => {
    installProject()
    useKoharuStore.setState({ tool: 'fill', selectedLayers: [] })
    const fill = vi.spyOn(commands, 'fillRaster')
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 31, clientX: 30, clientY: 40 })

    expect(fill).not.toHaveBeenCalled()
  })

  it('does not fill when the selection is not a raster layer', () => {
    installProject()
    // `installProject` selects the image layer 'element'.
    useKoharuStore.setState({ tool: 'fill', selectedLayers: ['element'] })
    const fill = vi.spyOn(commands, 'fillRaster')
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 32, clientX: 30, clientY: 40 })

    expect(fill).not.toHaveBeenCalled()
  })

  it('maps the Remove tool to an inpainting mask gesture', async () => {
    installProject()
    useKoharuStore.setState({ tool: 'remove' })
    const commit = vi.spyOn(commands, 'commitInpaint').mockResolvedValue('job')
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 12, clientX: 30, clientY: 40 })
    fireEvent.pointerUp(surface, { pointerId: 12, clientX: 30, clientY: 40 })

    await waitFor(() => expect(commit).toHaveBeenCalledOnce())
    expect(canvas.beginStroke).toHaveBeenCalledWith({
      kind: 'inpaint',
      layer: null,
      point: { x: 20, y: 20 },
      diameter: 48,
    })
    expect(commit).toHaveBeenCalledWith(1, expect.arrayContaining([{ x: 20, y: 20 }]), 48)
  })

  it('creates point text on click and paragraph text on drag', async () => {
    installProject()
    useKoharuStore.setState({ tool: 'text' })
    const point = vi
      .spyOn(commands, 'addPointText')
      .mockResolvedValue({ revision: 2, layer: 'point-text' })
    const box = vi
      .spyOn(commands, 'addTextBox')
      .mockResolvedValue({ revision: 3, layer: 'box-text' })
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 13, clientX: 30, clientY: 40 })
    fireEvent.pointerUp(surface, { pointerId: 13, clientX: 30, clientY: 40 })
    await waitFor(() => expect(point).toHaveBeenCalledWith({ x: 20, y: 20 }))

    fireEvent.pointerDown(surface, { button: 0, pointerId: 14, clientX: 40, clientY: 50 })
    fireEvent.pointerMove(surface, { pointerId: 14, clientX: 140, clientY: 110 })
    fireEvent.pointerUp(surface, { pointerId: 14, clientX: 140, clientY: 110 })
    await waitFor(() => expect(box).toHaveBeenCalledOnce())
    expect(box).toHaveBeenCalledWith({
      x: 30,
      y: 30,
      width: 100,
      height: 60,
      angle_degrees: 0,
    })
  })

  it('runs manual OCR on a drawn box and selects the new region layer', async () => {
    installProject()
    useKoharuStore.setState({ tool: 'manual_ocr_box' })
    const manual = vi
      .spyOn(commands, 'manualOcr')
      .mockResolvedValue({ revision: 2, region: 'region', layer: 'manual', job: 'job' })
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 15, clientX: 30, clientY: 40 })
    fireEvent.pointerMove(surface, { pointerId: 15, clientX: 140, clientY: 110 })
    fireEvent.pointerUp(surface, { pointerId: 15, clientX: 140, clientY: 110 })

    await waitFor(() => expect(manual).toHaveBeenCalledOnce())
    expect(manual).toHaveBeenCalledWith([
      { x: 20, y: 20 },
      { x: 130, y: 90 },
    ])
    await waitFor(() => expect(useKoharuStore.getState().selectedLayers).toEqual(['manual']))
  })

  it('runs manual OCR on a freeform lasso and previews the drawing', async () => {
    installProject()
    useKoharuStore.setState({ tool: 'manual_ocr_lasso' })
    const manual = vi
      .spyOn(commands, 'manualOcr')
      .mockResolvedValue({ revision: 2, region: 'region', layer: 'manual', job: 'job' })
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 16, clientX: 30, clientY: 40 })
    fireEvent.pointerMove(surface, { pointerId: 16, clientX: 100, clientY: 40 })
    fireEvent.pointerMove(surface, { pointerId: 16, clientX: 100, clientY: 100 })
    fireEvent.pointerMove(surface, { pointerId: 16, clientX: 30, clientY: 100 })
    expect(screen.getByTestId('manual-ocr-lasso')).toBeInTheDocument()
    fireEvent.pointerUp(surface, { pointerId: 16, clientX: 30, clientY: 100 })

    await waitFor(() => expect(manual).toHaveBeenCalledOnce())
    expect(manual).toHaveBeenCalledWith([
      { x: 20, y: 20 },
      { x: 90, y: 20 },
      { x: 90, y: 80 },
      { x: 20, y: 80 },
    ])
  })

  it('previews a dragged shape on the canvas and commits it once', async () => {
    installProject()
    useKoharuStore.setState({ tool: 'shape', shapeKind: 'ellipse' })
    const addShape = vi
      .spyOn(commands, 'addShape')
      .mockResolvedValue({ revision: 2, layer: 'shape' })
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 20, clientX: 30, clientY: 40 })
    expect(canvas.beginShape).toHaveBeenCalledOnce()
    fireEvent.pointerMove(surface, { pointerId: 20, clientX: 140, clientY: 110 })
    fireEvent.pointerUp(surface, { pointerId: 20, clientX: 140, clientY: 110 })

    await waitFor(() => expect(addShape).toHaveBeenCalledOnce())
    expect(canvas.updateShape).toHaveBeenCalledOnce()
    expect(canvas.finishShape).toHaveBeenCalledOnce()
    expect(addShape).toHaveBeenCalledWith(
      1,
      'page',
      { points: expect.any(Array) },
      'ellipse',
      { fill: { solid: [255, 255, 255, 255] }, stroke: null },
      0,
    )
    await waitFor(() => expect(useKoharuStore.getState().selectedLayers).toEqual(['shape']))
  })

  it('keeps a line shape drawn from the polygon the drag produced', async () => {
    installProject()
    useKoharuStore.setState({ tool: 'shape', shapeKind: 'line' })
    const addShape = vi
      .spyOn(commands, 'addShape')
      .mockResolvedValue({ revision: 2, layer: 'shape' })
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 21, clientX: 30, clientY: 40 })
    fireEvent.pointerMove(surface, { pointerId: 21, clientX: 140, clientY: 110 })
    fireEvent.pointerUp(surface, { pointerId: 21, clientX: 140, clientY: 110 })

    await waitFor(() => expect(addShape).toHaveBeenCalledOnce())
    const points = addShape.mock.calls[0][2].points
    expect(addShape.mock.calls[0][3]).toEqual({
      line: {
        start: [points[0].x, points[0].y],
        end: [points[2].x, points[2].y],
        head: null,
      },
    })
  })

  it('coalesces a burst of shape drags into one throttled preview frame', () => {
    installProject()
    useKoharuStore.setState({ tool: 'shape' })
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 28, clientX: 30, clientY: 40 })
    // While the shape gesture is live the preview must be throttled, not sent
    // once per pointer event.
    for (let x = 40; x <= 140; x += 10) {
      fireEvent.pointerMove(surface, { pointerId: 28, clientX: x, clientY: x })
    }
    expect(canvas.updateShape).not.toHaveBeenCalled()

    flushAnimationFrames()
    expect(canvas.updateShape).toHaveBeenCalledOnce()
  })

  it('cancels a shape preview without committing it', async () => {
    installProject()
    useKoharuStore.setState({ tool: 'shape' })
    const addShape = vi.spyOn(commands, 'addShape')
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 22, clientX: 30, clientY: 40 })
    fireEvent.pointerMove(surface, { pointerId: 22, clientX: 140, clientY: 110 })
    fireEvent.pointerCancel(surface)

    expect(canvas.cancelShape).toHaveBeenCalledOnce()
    expect(canvas.finishShape).not.toHaveBeenCalled()
    expect(addShape).not.toHaveBeenCalled()
  })

  it('does not create a shape from a click with no drag', () => {
    installProject()
    useKoharuStore.setState({ tool: 'shape' })
    const addShape = vi.spyOn(commands, 'addShape')
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 23, clientX: 30, clientY: 40 })
    fireEvent.pointerUp(surface, { pointerId: 23, clientX: 30, clientY: 40 })

    expect(canvas.cancelShape).toHaveBeenCalledOnce()
    expect(canvas.finishShape).not.toHaveBeenCalled()
    expect(addShape).not.toHaveBeenCalled()
  })

  it('creates a horizontal arrow instead of discarding the drag', async () => {
    installProject()
    useKoharuStore.setState({ tool: 'shape', shapeKind: 'arrow' })
    const addShape = vi
      .spyOn(commands, 'addShape')
      .mockResolvedValue({ revision: 2, layer: 'shape' })
    const surface = renderWorkspace()

    // A purely horizontal drag leaves the frame one unit tall, which a
    // both-axes minimum would reject.
    fireEvent.pointerDown(surface, { button: 0, pointerId: 24, clientX: 30, clientY: 40 })
    fireEvent.pointerMove(surface, { pointerId: 24, clientX: 160, clientY: 40 })
    fireEvent.pointerUp(surface, { pointerId: 24, clientX: 160, clientY: 40 })

    await waitFor(() => expect(addShape).toHaveBeenCalledOnce())
    expect(canvas.cancelShape).not.toHaveBeenCalled()
    expect(canvas.finishShape).toHaveBeenCalledOnce()
  })

  it('creates a vertical line instead of discarding the drag', async () => {
    installProject()
    useKoharuStore.setState({ tool: 'shape', shapeKind: 'line' })
    const addShape = vi
      .spyOn(commands, 'addShape')
      .mockResolvedValue({ revision: 2, layer: 'shape' })
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 25, clientX: 30, clientY: 40 })
    fireEvent.pointerMove(surface, { pointerId: 25, clientX: 30, clientY: 170 })
    fireEvent.pointerUp(surface, { pointerId: 25, clientX: 30, clientY: 170 })

    await waitFor(() => expect(addShape).toHaveBeenCalledOnce())
    expect(canvas.cancelShape).not.toHaveBeenCalled()
  })

  it('creates a thin but long rectangle', async () => {
    installProject()
    useKoharuStore.setState({ tool: 'shape', shapeKind: 'rectangle' })
    const addShape = vi
      .spyOn(commands, 'addShape')
      .mockResolvedValue({ revision: 2, layer: 'shape' })
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 26, clientX: 30, clientY: 40 })
    fireEvent.pointerMove(surface, { pointerId: 26, clientX: 200, clientY: 41 })
    fireEvent.pointerUp(surface, { pointerId: 26, clientX: 200, clientY: 41 })

    await waitFor(() => expect(addShape).toHaveBeenCalledOnce())
    expect(canvas.cancelShape).not.toHaveBeenCalled()
  })

  it('releases the native shape edit when finishing it fails', () => {
    installProject()
    useKoharuStore.setState({ tool: 'shape' })
    const addShape = vi.spyOn(commands, 'addShape')
    canvas.finishShape.mockImplementationOnce(() => {
      throw new Error('no shape preview is active')
    })
    const surface = renderWorkspace()

    fireEvent.pointerDown(surface, { button: 0, pointerId: 27, clientX: 30, clientY: 40 })
    fireEvent.pointerMove(surface, { pointerId: 27, clientX: 140, clientY: 110 })
    fireEvent.pointerUp(surface, { pointerId: 27, clientX: 140, clientY: 110 })

    // Leaving the WASM edit pending would block every later edit with
    // "another canvas edit is already active".
    expect(canvas.cancelShape).toHaveBeenCalledOnce()
    expect(addShape).not.toHaveBeenCalled()
  })
})
