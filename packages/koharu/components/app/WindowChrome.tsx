'use client'

import { commands, type ResizeDirection } from '@koharu/bridge/protocol'
import { getCurrentWindow } from '@tauri-apps/api/window'
import { Copy, Minus, Square, X } from 'lucide-react'
import {
  useEffect,
  useState,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
} from 'react'
import { useTranslation } from 'react-i18next'

const resizeHandles = [
  { direction: 'North', className: 'top-0 right-2 left-2 h-1 cursor-n-resize' },
  { direction: 'South', className: 'right-2 bottom-0 left-2 h-1 cursor-s-resize' },
  { direction: 'East', className: 'top-2 right-0 bottom-2 w-1 cursor-e-resize' },
  { direction: 'West', className: 'top-2 bottom-2 left-0 w-1 cursor-w-resize' },
  { direction: 'NorthEast', className: 'top-0 right-0 size-2 cursor-ne-resize' },
  { direction: 'NorthWest', className: 'top-0 left-0 size-2 cursor-nw-resize' },
  { direction: 'SouthEast', className: 'right-0 bottom-0 size-2 cursor-se-resize' },
  { direction: 'SouthWest', className: 'bottom-0 left-0 size-2 cursor-sw-resize' },
] as const

export function useMacOS() {
  const [macOS, setMacOS] = useState(false)

  useEffect(() => {
    setMacOS(
      navigator.userAgent.includes('Macintosh') ||
        navigator.platform.toLowerCase().startsWith('mac'),
    )
  }, [])

  return macOS
}

function isEmbedded(): boolean {
  return typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window
}

// On Linux the runtime's own window dragging never sees the pointer press,
// which lands on the CEF window, so koharu-rpc hands the move or resize to the
// window manager instead.
function isLinux(): boolean {
  return navigator.userAgent.includes('Linux') && !navigator.userAgent.includes('Android')
}

const CLICKABLE_TAGS = new Set(['A', 'BUTTON', 'INPUT', 'SELECT', 'TEXTAREA', 'LABEL', 'SUMMARY'])
const INTERACTIVE_ROLES = new Set([
  'button',
  'link',
  'menuitem',
  'tab',
  'checkbox',
  'radio',
  'switch',
  'option',
])

/**
 * Capture-phase mouse-down handler for `data-tauri-drag-region='deep'`
 * elements. It has to run in the capture phase: Tauri's drag script listens on
 * the document, which is also React's root here, and stops the event before
 * React's bubble listener sees it. That script still handles double-click
 * maximizing; this only starts the move on Linux, skipping the same
 * interactive descendants that script skips.
 */
export function startWindowDrag(event: ReactMouseEvent<HTMLElement>) {
  if (event.button !== 0 || event.detail !== 1 || !isEmbedded() || !isLinux()) return
  for (
    let element = event.target as HTMLElement | null;
    element && element !== event.currentTarget;
    element = element.parentElement
  ) {
    const clickable =
      CLICKABLE_TAGS.has(element.tagName) ||
      (element.hasAttribute('contenteditable') &&
        element.getAttribute('contenteditable') !== 'false') ||
      (element.hasAttribute('tabindex') && element.getAttribute('tabindex') !== '-1') ||
      INTERACTIVE_ROLES.has(element.getAttribute('role') ?? '')
    if (clickable || element.getAttribute('data-tauri-drag-region') === 'false') return
  }
  void commands.startWindowMoveResize().catch(() => undefined)
}

function startWindowResize(direction: ResizeDirection) {
  const request = isLinux()
    ? commands.startWindowMoveResize(direction)
    : getCurrentWindow().startResizeDragging(direction)
  void request.catch(() => undefined)
}

export function WindowControls() {
  const { t } = useTranslation()
  const [maximized, setMaximized] = useState(false)
  const embedded = isEmbedded()

  useEffect(() => {
    if (!embedded) return
    const window = getCurrentWindow()
    let disposed = false
    let unlisten: (() => void) | undefined
    const synchronize = () => {
      void window.isMaximized().then((value) => {
        if (!disposed) setMaximized(value)
      })
    }
    synchronize()
    queueMicrotask(() => {
      if (disposed) return
      void window.onResized(synchronize).then((stop) => {
        if (disposed) void Promise.resolve(stop()).catch(() => undefined)
        else unlisten = stop
      })
    })
    return () => {
      disposed = true
      if (unlisten) void Promise.resolve(unlisten()).catch(() => undefined)
    }
  }, [embedded])

  const toggleMaximize = async () => {
    const window = getCurrentWindow()
    await window.toggleMaximize()
    setMaximized(await window.isMaximized())
  }

  if (!embedded) return null

  return (
    <>
      {!maximized && <WindowResizeHandles />}
      <div className='flex h-full shrink-0'>
        <WindowButton
          label={t('window.minimize')}
          onClick={() => void getCurrentWindow().minimize()}
        >
          <Minus />
        </WindowButton>
        <WindowButton
          label={t(maximized ? 'window.restore' : 'window.maximize')}
          onClick={() => void toggleMaximize()}
        >
          {maximized ? <Copy /> : <Square />}
        </WindowButton>
        <WindowButton
          label={t('window.close')}
          className='hover:text-destructive-foreground hover:bg-destructive'
          onClick={() => void getCurrentWindow().close()}
        >
          <X />
        </WindowButton>
      </div>
    </>
  )
}

function WindowResizeHandles() {
  const startResize =
    (direction: (typeof resizeHandles)[number]['direction']) =>
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (event.button !== 0) return
      event.preventDefault()
      event.stopPropagation()
      startWindowResize(direction)
    }

  return resizeHandles.map(({ direction, className }) => (
    <div
      key={direction}
      aria-hidden='true'
      data-window-resize-handle={direction}
      className={`window-resize-handle fixed z-50 ${className}`}
      onPointerDown={startResize(direction)}
    />
  ))
}

function WindowButton({
  label,
  className = '',
  children,
  onClick,
}: {
  label: string
  className?: string
  children: ReactNode
  onClick: () => void
}) {
  return (
    <button
      type='button'
      aria-label={label}
      className={`grid h-full w-11 place-items-center text-muted-foreground transition-colors hover:bg-primary/10 hover:text-foreground [&_svg]:size-3.5 ${className}`}
      onClick={onClick}
    >
      {children}
    </button>
  )
}
