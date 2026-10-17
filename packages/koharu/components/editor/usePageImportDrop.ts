'use client'

import { getCurrentWindow } from '@tauri-apps/api/window'
import { useCallback, useEffect, useRef, useState, type DragEvent, type RefObject } from 'react'

import { filesFromDataTransfer, isEmbedded } from '@/lib/transfer'

export interface DroppedImport {
  paths?: string[]
  files?: File[]
}

/**
 * 물리 픽셀 좌표가 요소 안에 있는지 본다.
 *
 * Tauri의 드롭 좌표는 웹뷰 기준 물리 픽셀이고 `getBoundingClientRect`는 CSS
 * 픽셀이므로, 배율만큼 곱해 맞춘 뒤 비교한다.
 */
export function dropInsideElement(
  rect: { left: number; top: number; right: number; bottom: number },
  position: { x: number; y: number },
  scale: number,
): boolean {
  const x = position.x / scale
  const y = position.y / scale
  return x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom
}

/**
 * 페이지 레일에 떨어진 파일·폴더를 가져오도록 배선한다.
 *
 * 데스크톱(CEF)은 OS 드롭이 네이티브 경로로 오므로 `onDragDropEvent`를 쓰고,
 * 브라우저는 DOM 드래그 이벤트에서 파일을 모은다. 두 경로는 배타적이므로
 * `isEmbedded()`로 한쪽만 건다.
 */
export function usePageImportDrop(
  target: RefObject<HTMLElement | null>,
  importDropped: (dropped: DroppedImport) => void,
) {
  const [overDropZone, setOverDropZone] = useState(false)
  const embedded = isEmbedded()
  // The caller's callback is usually a fresh closure each render; keeping it in
  // a ref stops the native listener below from being torn down and re-registered
  // on every render.
  const importRef = useRef(importDropped)
  useEffect(() => {
    importRef.current = importDropped
  }, [importDropped])

  useEffect(() => {
    if (!embedded) return
    let disposed = false
    let unlisten: (() => void) | undefined
    const inside = (position: { x: number; y: number }) => {
      const rect = target.current?.getBoundingClientRect()
      if (!rect) return false
      return dropInsideElement(rect, position, window.devicePixelRatio || 1)
    }
    void getCurrentWindow()
      .onDragDropEvent((event) => {
        const payload = event.payload
        if (payload.type === 'leave') {
          setOverDropZone(false)
          return
        }
        if (payload.type === 'enter' || payload.type === 'over') {
          setOverDropZone(inside(payload.position))
          return
        }
        const dropped = inside(payload.position)
        setOverDropZone(false)
        if (dropped) importRef.current({ paths: payload.paths })
      })
      .then((stop) => {
        if (disposed) stop()
        else unlisten = stop
      })
      .catch(() => undefined)
    return () => {
      disposed = true
      unlisten?.()
    }
  }, [embedded, target])

  // A file drag carries no `text/plain`, so the page-reorder handlers above can
  // tell the two apart and let this one own the drop.
  const onDragOver = useCallback(
    (event: DragEvent<HTMLElement>) => {
      if (embedded || !Array.from(event.dataTransfer.types).includes('Files')) return
      event.preventDefault()
      event.dataTransfer.dropEffect = 'copy'
      setOverDropZone(true)
    },
    [embedded],
  )

  const onDragLeave = useCallback((event: DragEvent<HTMLElement>) => {
    // Leaving onto a child still bubbles a leave whose relatedTarget is inside,
    // so the highlight would flicker without this check.
    const next = event.relatedTarget as Node | null
    if (!next || !event.currentTarget.contains(next)) setOverDropZone(false)
  }, [])

  const onDrop = useCallback(
    (event: DragEvent<HTMLElement>) => {
      if (embedded || !Array.from(event.dataTransfer.types).includes('Files')) return
      event.preventDefault()
      setOverDropZone(false)
      void filesFromDataTransfer(event.dataTransfer).then((files) => {
        if (files.length > 0) importDropped({ files })
      })
    },
    [embedded, importDropped],
  )

  const dropHandlers = embedded ? {} : { onDragOver, onDragLeave, onDrop }

  return { overDropZone, dropHandlers }
}
