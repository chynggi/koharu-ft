'use client'

import { call } from '@/lib/backend'
import { pageKey, preparedPageKey, projectKey, queryClient } from '@/lib/queries'
import { useKoharuStore } from '@/lib/store'
import { showCanvasPage } from '@koharu/bridge/canvas'
import {
  commands,
  type CanvasPagePreparation,
  type Page,
  type ProjectInfo,
} from '@koharu/bridge/protocol'

let activationRequest = 0

/** Opens `page` with `selectedPages` selected in the rail. A page the canvas
 * has already prepared shows at once through the optimistic query cache, and
 * the native selection then confirms it; a failed switch rolls the cache back.
 * Only the newest request applies, so resolves `true` once this page is open
 * and `false` when a newer switch superseded it or it failed. */
export function activatePage(page: string, selectedPages: string[]): Promise<boolean> {
  const { selectPages, selectLayers } = useKoharuStore.getState()
  const previousProject = queryClient.getQueryData<ProjectInfo | null>(projectKey)
  const previousPage = queryClient.getQueryData<Page | null>(pageKey)
  const prepared = queryClient.getQueryData<CanvasPagePreparation>(preparedPageKey(page))
  const activated = showCanvasPage(page, previousProject?.revision ?? null)
  const request = ++activationRequest
  const synchronize = async () => {
    if (activationRequest !== request) return false
    if (activated && previousProject && prepared?.revision === previousProject.revision) {
      queryClient.setQueryData(projectKey, { ...previousProject, active_page: page })
      queryClient.setQueryData(pageKey, prepared.page)
    }
    selectPages(selectedPages)
    selectLayers([])
    try {
      const selection = await call(commands.selectPage, page)
      if (activationRequest !== request) return false
      queryClient.setQueryData(projectKey, selection.project)
      queryClient.setQueryData(pageKey, selection.page)
      return true
    } catch {
      if (activationRequest !== request) return false
      if (queryClient.getQueryData<ProjectInfo | null>(projectKey)?.active_page === page) {
        queryClient.setQueryData(projectKey, previousProject)
        queryClient.setQueryData(pageKey, previousPage)
      }
      return false
    }
  }
  if (!activated) return synchronize()
  // Let the canvas present the prepared frame before the editor follows it.
  return new Promise((resolve) =>
    requestAnimationFrame(() => window.setTimeout(() => void synchronize().then(resolve), 0)),
  )
}

/** Drops any page switch still waiting to apply. */
export function cancelPageActivation(): void {
  activationRequest += 1
}
