'use client'

import { useEffect, useRef } from 'react'
import { useTranslation } from 'react-i18next'

import { useColorSampling } from '@/components/controls/ColorSampling'
import { CanvasCommandBar } from '@/components/editor/CanvasCommandBar'
import { FillBar } from '@/components/editor/FillBar'
import { HistoryWindow } from '@/components/editor/HistoryWindow'
import { ShapeBar } from '@/components/editor/ShapeBar'
import { StatusBar } from '@/components/editor/StatusBar'
import { ToolBar } from '@/components/editor/ToolBar'
import { TypeBar } from '@/components/editor/TypeBar'
import { call, dispatch } from '@/lib/backend'
import { usePage } from '@/lib/queries'
import { copiedLayers, deleteSelection, duplicateLayers } from '@/lib/scene'
import { hexToRgba, toolKeys, useKoharuStore, type CanvasTool } from '@/lib/store'
import { commands, type LayerId, type PageId, type Rgba } from '@koharu/ipc/protocol'

/** The canvas's place in the window. The native canvas draws the page under the web interface
 * wherever this element and its ancestors paint nothing, and takes the pointer there; the
 * interface sets its tool, brush, and colours, and hears its camera and selection. */
export function CanvasWorkspace() {
  const { t } = useTranslation()
  const surface = useRef<HTMLDivElement>(null)
  const page = usePage().data
  const tool = useKoharuStore((state) => state.tool)
  const brush = useKoharuStore((state) => state.brush)
  const colorSampling = useColorSampling()
  /** The layers Ctrl+C took, which Ctrl+V copies onto the page shown. */
  const clipboard = useRef<{ page: PageId; layers: LayerId[] } | null>(null)

  useEffect(() => {
    const element = surface.current
    if (!element) return
    const report = () => {
      const { x, y, width, height } = element.getBoundingClientRect()
      dispatch(commands.setViewport, x, y, width, height)
    }
    report()
    const resize = new ResizeObserver(report)
    resize.observe(element)
    window.addEventListener('resize', report)
    window.visualViewport?.addEventListener('resize', report)
    return () => {
      resize.disconnect()
      window.removeEventListener('resize', report)
      window.visualViewport?.removeEventListener('resize', report)
      // Nothing to draw while the editor is away, as under the preferences.
      dispatch(commands.setViewport, 0, 0, 0, 0)
    }
  }, [])

  useEffect(() => dispatch(commands.setTool, tool), [tool])
  // The canvas numbers the texts and outlines their frames while Story shows.
  const panel = useKoharuStore((state) => state.panel)
  const storyHover = useKoharuStore((state) => state.storyHover)
  useEffect(
    () => dispatch(commands.setStory, { shown: panel === 'story', hovered: storyHover }),
    [panel, storyHover],
  )

  useEffect(
    () =>
      dispatch(commands.setBrush, {
        diameter: brush.diameter,
        color: hexToRgba(brush.color),
        ramp: brush.ramp && { to: hexToRgba(brush.ramp.to), along: brush.ramp.along },
      }),
    [brush],
  )

  useEffect(() => {
    const send = () =>
      dispatch(commands.setColors, {
        workspace: cssColor('--workspace-background'),
        selection: cssColor('--canvas-selection'),
      })
    send()
    const theme = new MutationObserver(send)
    theme.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class', 'data-palette'],
    })
    return () => theme.disconnect()
  }, [])

  // A selection the interface made reaches the canvas; one the canvas published doesn't go back.
  useEffect(
    () =>
      useKoharuStore.subscribe((state, previous) => {
        if (state.selectedLayers === previous.selectedLayers) return
        if (sameLayers(state.selectedLayers, state.canvasSelection)) return
        useKoharuStore.setState({ canvasSelection: state.selectedLayers })
        dispatch(commands.select, state.selectedLayers)
      }),
    [],
  )

  useEffect(() => {
    const editable = (target: EventTarget | null) =>
      target instanceof HTMLInputElement ||
      target instanceof HTMLTextAreaElement ||
      (target instanceof HTMLElement && target.isContentEditable)
    // The tool Space replaced, put back when it's released, as Photoshop's hand tool.
    let held: CanvasTool | null = null

    const down = (event: KeyboardEvent) => {
      if (event.key === 'Alt') {
        event.preventDefault()
        return
      }
      if (editable(event.target)) return
      const state = useKoharuStore.getState()
      if (event.code === 'Space') {
        event.preventDefault()
        if (held === null && state.tool !== 'pan') {
          held = state.tool
          state.setTool('pan')
        }
        return
      }
      const command = event.ctrlKey || event.metaKey
      if (command && event.key.toLowerCase() === 'z') {
        event.preventDefault()
        dispatch(event.shiftKey ? commands.redo : commands.undo)
        return
      }
      if (command && event.key.toLowerCase() === 'a' && page) {
        event.preventDefault()
        state.selectLayers(page.layers.map((layer) => layer.id))
        return
      }
      if (command && event.key.toLowerCase() === 'c' && page) {
        const layers = copiedLayers(state.selectedLayers, page.layers)
        if (!layers.length) return
        event.preventDefault()
        clipboard.current = { page: page.id, layers }
        return
      }
      // A paste onto the page copied from lands above each original, as Photoshop's Duplicate
      // Layer (Ctrl+J) does; onto another page, at its top.
      const copy = (layers: LayerId[], onto: PageId | null) => {
        void call(duplicateLayers, layers, onto)
          .then((copies) => useKoharuStore.getState().selectLayers(copies))
          .catch(() => undefined)
      }
      if (command && event.key.toLowerCase() === 'v' && page && clipboard.current) {
        event.preventDefault()
        const { page: from, layers } = clipboard.current
        copy(layers, from === page.id ? null : page.id)
        return
      }
      if (command && event.key.toLowerCase() === 'j' && page) {
        const layers = copiedLayers(state.selectedLayers, page.layers)
        if (!layers.length) return
        event.preventDefault()
        copy(layers, null)
        return
      }
      // Photoshop's Deselect, of the area the marquee and the lasso marked.
      if (command && event.key.toLowerCase() === 'd') {
        event.preventDefault()
        dispatch(commands.deselect)
        return
      }
      if ((event.key === 'Delete' || event.key === 'Backspace') && state.selectedLayers.length) {
        event.preventDefault()
        dispatch(deleteSelection, state.selectedLayers, page?.layers ?? [])
        return
      }
      // Photoshop's Fit on Screen.
      if (command && event.key === '0') {
        event.preventDefault()
        dispatch(commands.fit)
        return
      }
      if (event.key === 'Escape') {
        colorSampling?.cancel()
        state.selectLayers([])
        return
      }
      if (command || event.altKey) return
      const key = event.key.toLowerCase()
      const keyed = (Object.keys(toolKeys) as CanvasTool[]).filter((tool) => toolKeys[tool] === key)
      const next = keyed[(keyed.indexOf(state.tool) + 1) % keyed.length]
      if (next) state.setTool(next)
    }

    const up = (event: KeyboardEvent) => {
      if (event.key === 'Alt') event.preventDefault()
      if (event.code === 'Space' && held !== null) {
        useKoharuStore.getState().setTool(held)
        held = null
      }
    }
    const blur = () => {
      if (held === null) return
      useKoharuStore.getState().setTool(held)
      held = null
    }

    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    window.addEventListener('blur', blur)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
      window.removeEventListener('blur', blur)
    }
  }, [colorSampling, page])

  return (
    <main className='relative flex h-full min-h-0 min-w-0 flex-col overflow-hidden rounded-tl-2xl'>
      <CanvasCommandBar />
      <div className='relative flex min-h-0 min-w-0 flex-1'>
        <ToolBar />
        <div
          ref={surface}
          data-testid='canvas-viewport'
          // Clipped, never scrolled: a focused control past its edge would scroll it, and every
          // overlay with it, off the canvas the overlays sit on.
          className='relative min-h-0 min-w-0 flex-1 overflow-clip'
        >
          {page && <TypeBar />}
          {page && <ShapeBar />}
          {page && <FillBar />}
          {page && <HistoryWindow />}
          {!page && (
            <div className='pointer-events-none absolute inset-0 grid place-items-center'>
              <p className='text-[12px] text-muted-foreground'>{t('canvas.empty')}</p>
            </div>
          )}
        </div>
      </div>
      <StatusBar onZoomChange={(zoom) => dispatch(commands.setZoom, zoom)} />
    </main>
  )
}

function sameLayers(a: string[], b: string[]): boolean {
  return a.length === b.length && a.every((layer, index) => layer === b[index])
}

/** A theme colour, whatever CSS notation it's written in, as the canvas's RGBA. */
function cssColor(variable: string): Rgba {
  const probe = document.createElement('canvas')
  probe.width = probe.height = 1
  const context = probe.getContext('2d', { willReadFrequently: true })
  if (!context) return [0, 0, 0, 1]
  context.fillStyle =
    getComputedStyle(document.documentElement).getPropertyValue(variable).trim() || '#000000'
  context.fillRect(0, 0, 1, 1)
  const [r = 0, g = 0, b = 0, a = 255] = context.getImageData(0, 0, 1, 1).data
  return [r / 255, g / 255, b / 255, a / 255]
}
