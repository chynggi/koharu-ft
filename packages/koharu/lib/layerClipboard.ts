import type { Page } from '@koharu/bridge/protocol'
import { commands } from '@koharu/bridge/protocol'

import { call } from './backend'
import { pageKey, pagesKey, projectKey, refresh } from './queries'
import { useKoharuStore } from './store'

function selectedTextLayerIds(page: Page | undefined): string[] {
  if (!page) return []
  const selected = new Set(useKoharuStore.getState().selectedLayers)
  return page.layers
    .filter((layer) => layer.type === 'text' && selected.has(layer.id))
    .map((layer) => layer.id)
}

export function copySelectedTextLayers(page: Page | undefined): boolean {
  const layers = selectedTextLayerIds(page)
  if (!layers.length) return false
  useKoharuStore.getState().setCopiedTextLayers(layers)
  return true
}

export async function pasteCopiedTextLayers(page: Page | undefined): Promise<void> {
  const copied = useKoharuStore.getState().copiedTextLayers
  if (!page || !copied.length) return
  await paste(page, copied)
}

/** Duplicates the selected text layers in place; the clipboard stays
/// untouched. Returns whether anything was duplicated. */
export async function duplicateSelectedTextLayers(page: Page | undefined): Promise<boolean> {
  if (!page) return false
  const layers = selectedTextLayerIds(page)
  if (!layers.length) return false
  await paste(page, layers)
  return true
}

async function paste(page: Page, layers: string[]): Promise<void> {
  const pasted = await call(commands.pasteTextLayers, layers, page.id)
  useKoharuStore.getState().selectLayers(pasted)
  await refresh(projectKey, pagesKey, pageKey)
}
