import type { Page } from '@koharu/bridge/protocol'
import { commands } from '@koharu/bridge/protocol'

import { call } from './backend'
import { pageKey, pagesKey, projectKey, refresh } from './queries'
import { useKoharuStore } from './store'

export function copySelectedTextLayers(page: Page | undefined): boolean {
  if (!page) return false
  const selected = new Set(useKoharuStore.getState().selectedLayers)
  const layers = page.layers
    .filter((layer) => layer.type === 'text' && selected.has(layer.id))
    .map((layer) => layer.id)
  if (!layers.length) return false
  useKoharuStore.getState().setCopiedTextLayers(layers)
  return true
}

export async function pasteCopiedTextLayers(page: Page | undefined): Promise<void> {
  const copied = useKoharuStore.getState().copiedTextLayers
  if (!page || !copied.length) return
  const pasted = await call(commands.pasteTextLayers, copied, page.id)
  useKoharuStore.getState().selectLayers(pasted)
  await refresh(projectKey, pagesKey, pageKey)
}
