'use client'

import {
  commands,
  type Change,
  type ColorAdjust,
  type Layer,
  type LayerId,
  type Op,
  type PageId,
  type ShapeFill,
  type Stroke,
  type Value,
} from '@koharu/ipc/protocol'

import { findPart, isBlockLayer, isTextLayer, walkLayers, type Move, type Part } from './document'

/** The panels' edits, each one undo step through the scene's own `edit`, labelled by a key the
 * History panel translates. */

function edit(label: string, ops: Op[]): Promise<Change> {
  return commands.edit({ label, ops })
}

/** Shows, hides, or fades layers. */
export function setVisibility(
  layers: LayerId[],
  visible: boolean | null,
  opacity: number | null,
): Promise<Change> {
  const patch: { [key in string]: Value } = {}
  if (visible !== null) patch.visible = visible
  if (opacity !== null) patch.opacity = opacity
  return edit(
    'set_visibility',
    layers.map((layer) => ({ op: 'update_layer', layer, patch })),
  )
}

/** Shows or hides a block's part: a hidden mask stops limiting the clean-up, and a hidden clean-up
 * shows the original. */
export function setPartVisibility(block: LayerId, part: Part, visible: boolean): Promise<Change> {
  return edit('set_visibility', [
    { op: 'update_layer', layer: block, patch: { [part]: { visible } } },
  ])
}

export function setLocked(layers: LayerId[], locked: boolean): Promise<Change> {
  return edit(
    'set_locked',
    layers.map((layer) => ({ op: 'update_layer', layer, patch: { locked } })),
  )
}

export function deleteLayers(layers: LayerId[]): Promise<Change> {
  return edit(
    'delete_layers',
    layers.map((layer) => ({ op: 'remove_layer', layer })),
  )
}

/** What Delete does when the page has the keys: deletes the selected layers, skipping a block's
 * selected part, which goes only with its block; the eraser clears a clean-up, and hiding a mask
 * stops it limiting the clean-up. */
export async function deleteSelection(selected: LayerId[], layers: Layer[]): Promise<void> {
  const whole = selected.filter((id) => !findPart(layers, id))
  if (whole.length) await deleteLayers(whole)
}

/** Moves layers in one undo step, each under its parent, or to the page's top level, at its
 * index counted after it's taken out, in order, as a drop in the Layers panel places them. */
export function moveLayers(moves: Move[]): Promise<Change> {
  return edit(
    'move_layer',
    moves.map(({ layer, parent, index }) => ({ op: 'move_layer', layer, parent, index })),
  )
}

/** A text layer's text, or a block's translation. */
export function setText(layer: LayerId, text: string): Promise<Change> {
  return edit('set_text', [{ op: 'set_text', layer, text }])
}

/** A block's source text, as OCR would have read it. */
export function setSourceText(layer: LayerId, text: string): Promise<Change> {
  return edit('set_source_text', [
    { op: 'update_layer', layer, patch: { source: { content: text } } },
  ])
}

/** Merges `style`, a `TextStyle` patch, into each layer's lettering: a text's own style, and a
 * block's translation's, which is what it draws. */
export function setStyle(layers: Layer[], style: Value): Promise<Change> {
  return edit(
    'set_style',
    layers
      .filter((layer) => isTextLayer(layer) || isBlockLayer(layer))
      .map((layer): Op => {
        const patch: Value = isBlockLayer(layer) ? { translation: { style } } : { style }
        return { op: 'update_layer', layer: layer.id, patch }
      }),
  )
}

/** A JSON Merge Patch of a layer's values, for the Inspector's frame, fit, anchor, direction, and
 * language. */
export function updateLayer(layer: LayerId, patch: Value): Promise<Change> {
  return edit('update_layer', [{ op: 'update_layer', layer, patch }])
}

export function deletePages(pages: PageId[]): Promise<Change> {
  return edit(
    'delete_pages',
    pages.map((page) => ({ op: 'remove_page', page })),
  )
}

/** Moves a page to `index`, counted after it's taken out. */
export function movePage(page: PageId, index: number): Promise<Change> {
  return edit('move_page', [{ op: 'move_page', page, index }])
}

/** Sets a page's name and the chapter it starts; an empty chapter clears it. */
export function updatePage(page: PageId, name: string, chapter: string): Promise<Change> {
  return edit('update_page', [
    { op: 'update_page', page, patch: { name, chapter: chapter || null } },
  ])
}

/** Joins pages into one spread where the first of them was, replacing them. */
export function mergePages(pages: PageId[], rtl: boolean): Promise<Change> {
  return edit('merge_pages', [{ op: 'merge_pages', pages, rtl, keep_originals: false }])
}

/** The selected layers a copy takes, in drawing order: a layer inside another selected one goes
 * with it, and a block's part only with its block. */
export function copiedLayers(selected: LayerId[], layers: Layer[]): LayerId[] {
  const chosen = new Set(selected)
  return walkLayers(layers)
    .filter(
      ({ layer, ancestors }) =>
        chosen.has(layer.id) && !ancestors.some((ancestor) => chosen.has(ancestor.id)),
    )
    .map(({ layer }) => layer.id)
}

/** Copies layers in one undo step, each just above itself, or at the top of `page` in order, as
 * Photoshop's Duplicate Layer; the copies' ids. */
export async function duplicateLayers(layers: LayerId[], page: PageId | null): Promise<LayerId[]> {
  const change = await edit(
    'duplicate_layer',
    layers.map((layer, index) => ({ op: 'duplicate_layer', layer, page, id: `$copy${index}` })),
  )
  const labels = new Set(layers.map((_, index) => `copy${index}`))
  return change.created.flatMap((created) =>
    created.kind === 'layer' && created.label !== null && labels.has(created.label)
      ? [created.id]
      : [],
  )
}

/** Gives shapes `fill`, or none, in one undo step. The old fill goes first, since a merge patch
 * would keep another kind's keys. */
export function setFill(layers: LayerId[], fill: ShapeFill | null): Promise<Change> {
  return edit(
    'set_fill',
    layers.flatMap((layer): Op[] => [
      { op: 'update_layer', layer, patch: { fill: null } },
      ...(fill === null ? [] : [{ op: 'update_layer' as const, layer, patch: { fill } as Value }]),
    ]),
  )
}

export function setStroke(layers: LayerId[], stroke: Stroke | null): Promise<Change> {
  return edit(
    'set_stroke',
    layers.map((layer) => ({ op: 'update_layer', layer, patch: { stroke } as Value })),
  )
}

/** Adjusts the colours of pictures, each adjustment in order, in one undo step. */
export function adjustColors(layers: LayerId[], adjusts: ColorAdjust[]): Promise<Change> {
  return edit(
    'adjust_colors',
    layers.flatMap((layer) =>
      adjusts.map((adjust): Op => ({ op: 'adjust_colors', layer, adjust })),
    ),
  )
}

/** The colour pictures among the selected layers that an adjustment takes: rasters, and blocks'
 * clean-ups. */
export function adjustable(selected: LayerId[], layers: Layer[]): LayerId[] {
  return selected.filter(
    (id) =>
      findPart(layers, id)?.part === 'cleanup' ||
      walkLayers(layers).some(({ layer }) => layer.id === id && layer.kind === 'raster'),
  )
}
