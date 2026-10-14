'use client'

import { create } from 'zustand'

import type {
  CanvasState,
  Download,
  EntityId,
  Job,
  Model,
  Frame,
  Preferences,
  Stage,
  StartupState,
  ModelResources,
  PageSize,
  ShapeStyle,
  BrushRamp,
} from '@koharu/bridge/protocol'
import { toast } from '@koharu/ui/components/toast'

import { defaultShapeStyle, type ShapeKind } from './shapes'

export type CanvasTool =
  | 'select'
  | 'text'
  | 'shape'
  | 'draw'
  | 'fill'
  | 'eraser'
  | 'color_picker'
  | 'remove'
  | 'manual_ocr_box'
  | 'manual_ocr_lasso'
  | 'pan'
export const MIN_BRUSH_DIAMETER = 1
export function maxBrushDiameter(size: PageSize): number {
  // Assume that for all sizes below 2048 width, 128 is sufficient.
  // For anything above, scale by percentage greater than 2048 width.
  return Math.round(Math.max(size.width / 2048, 1.0) * 128)
}

export function isBrushTool(tool: CanvasTool): boolean {
  return tool === 'draw' || tool === 'eraser' || tool === 'remove'
}

export interface CanvasBrush {
  diameter: number
  color: string
  /**
   * The ramp that replaces `color` while it is set. It is a command argument,
   * not persisted state, so it survives only as long as this session's brush.
   */
  ramp: BrushRamp | null
}
export type InspectorSection = 'copy' | 'type' | 'layers'
export type ShortcutAction = CanvasTool | 'fit'
export type Shortcuts = Record<ShortcutAction, string>
export type PipelineScope = 'page' | 'selected-pages' | 'project'
export const pipelineStages: readonly Stage[] = ['detection', 'ocr', 'translation', 'inpainting']

interface KoharuStore {
  initialized: boolean
  preferences: Preferences | null
  translationModels: Model[]
  resources: ModelResources | null
  jobs: Record<string, Job>
  downloads: Record<string, Download>
  camera: { zoom: number; translation: [number, number]; fitted: boolean }
  canvasPage: EntityId | null
  canvasRevision: number | null
  canvasGeneration: number
  canvasSize: [number, number]
  fitCanvasRequest: number
  layerFrames: Record<EntityId, Frame>
  selectedLayers: EntityId[]
  copiedTextLayers: EntityId[]
  selectedPages: EntityId[]
  tool: CanvasTool
  brush: CanvasBrush
  // The bucket fill keeps its own colour: it is a different operation from
  // painting, so sharing `brush.color` would make each tool silently retarget
  // the other's colour.
  fillColor: string
  shapeKind: ShapeKind
  shapeStyle: ShapeStyle
  inspector: InspectorSection
  processingScope: PipelineScope
  processingStages: Stage[]
  settingsOpen: boolean
  shortcuts: Shortcuts
  selectPages: (pages: EntityId[]) => void
  showInspector: (section: InspectorSection) => void
  setProcessingScope: (scope: PipelineScope) => void
  setProcessingStages: (stages: Stage[]) => void
  setSettingsOpen: (open: boolean) => void
  selectLayers: (layers: EntityId[]) => void
  setCopiedTextLayers: (layers: EntityId[]) => void
  setTool: (tool: CanvasTool, cleanupLayer?: EntityId) => void
  setBrush: (brush: CanvasBrush) => void
  setFillColor: (color: string) => void
  setShapeKind: (kind: ShapeKind) => void
  setShapeStyle: (style: ShapeStyle) => void
  setShortcut: (action: ShortcutAction, key: string) => void
  requestCanvasFit: () => void
  dismissJob: (id: string) => void
  dismissDownload: (id: number) => void
}

export const defaultShortcuts: Shortcuts = {
  select: 'v',
  text: 't',
  shape: 'r',
  draw: 'b',
  fill: 'g',
  eraser: 'e',
  color_picker: 'i',
  remove: 'j',
  manual_ocr_box: 'o',
  manual_ocr_lasso: 'l',
  pan: 'h',
  fit: '0',
}

export const useKoharuStore = create<KoharuStore>()((set) => ({
  initialized: false,
  preferences: null,
  translationModels: [],
  resources: null,
  jobs: {},
  downloads: {},
  camera: { zoom: 1, translation: [0, 0], fitted: true },
  canvasPage: null,
  canvasRevision: null,
  canvasGeneration: 0,
  canvasSize: [0, 0],
  fitCanvasRequest: 0,
  layerFrames: {},
  selectedLayers: [],
  copiedTextLayers: [],
  selectedPages: [],
  tool: 'select',
  brush: { diameter: 48, color: '#FFFFFF', ramp: null },
  fillColor: '#000000',
  shapeKind: 'rectangle',
  shapeStyle: defaultShapeStyle('rectangle'),
  inspector: 'copy',
  processingScope: 'selected-pages',
  processingStages: [...pipelineStages],
  settingsOpen: false,
  shortcuts: defaultShortcuts,
  selectPages: (selectedPages) => set({ selectedPages: [...new Set(selectedPages)] }),
  showInspector: (inspector) => set({ inspector }),
  setProcessingScope: (processingScope) => set({ processingScope }),
  setProcessingStages: (processingStages) => set({ processingStages }),
  setSettingsOpen: (settingsOpen) => set({ settingsOpen }),
  selectLayers: (selectedLayers) => set({ selectedLayers: [...new Set(selectedLayers)] }),
  setCopiedTextLayers: (copiedTextLayers) => set({ copiedTextLayers: [...copiedTextLayers] }),
  // 붓·지우개로 넘어갈 때만 cleanup 레이어를 고른다. 다른 도구는 선택을 건드리지
  // 않으므로 이전 상태를 읽을 필요가 없다.
  setTool: (tool, cleanupLayer) =>
    set({
      tool,
      ...(cleanupLayer && (tool === 'draw' || tool === 'eraser')
        ? { selectedLayers: [cleanupLayer] }
        : {}),
    }),
  // The whole brush object is replaced, so a caller that changes one field must
  // spread the current brush. That is what keeps `ramp` alive when the colour
  // picker samples a pixel, rather than silently dropping back to a flat stroke.
  setBrush: (brush) => set({ brush }),
  setFillColor: (fillColor) => set({ fillColor }),
  // Each kind ships the style that can actually draw it, so switching to a line
  // never leaves the fill-only style of a rectangle behind.
  setShapeKind: (shapeKind) => set({ shapeKind, shapeStyle: defaultShapeStyle(shapeKind) }),
  setShapeStyle: (shapeStyle) => set({ shapeStyle }),
  setShortcut: (action, key) =>
    set((state) => ({
      shortcuts: { ...state.shortcuts, [action]: key.toLowerCase().slice(0, 1) },
    })),
  requestCanvasFit: () => set((state) => ({ fitCanvasRequest: state.fitCanvasRequest + 1 })),
  dismissJob: (id) =>
    set((state) => {
      const jobs = { ...state.jobs }
      delete jobs[id]
      return { jobs }
    }),
  dismissDownload: (id) =>
    set((state) => {
      const downloads = { ...state.downloads }
      delete downloads[String(id)]
      return { downloads }
    }),
}))

export function receiveStartupState(state: StartupState): void {
  useKoharuStore.setState({
    initialized: true,
    preferences: state.preferences,
    jobs: byId(state.jobs),
    ...canvasSnapshot(state.canvas),
  })
}

export function receiveCanvas(canvas: CanvasState): void {
  useKoharuStore.setState(canvasSnapshot(canvas))
}

export function receiveJob(job: Job): void {
  useKoharuStore.setState((state) => ({ jobs: { ...state.jobs, [job.id]: job } }))
}

export function receiveDownload(download: Download): void {
  useKoharuStore.setState((state) => {
    const downloads = { ...state.downloads }
    if (download.state === 'finished') delete downloads[String(download.id)]
    else downloads[String(download.id)] = download
    return { downloads }
  })
}

export function receivePreferences(preferences: Preferences): void {
  useKoharuStore.setState({ preferences })
}

export function receiveTranslationModels(translationModels: Model[]): void {
  useKoharuStore.setState({ translationModels })
}

export function receiveResources(resources: ModelResources): void {
  useKoharuStore.setState({ resources })
}

export function receiveError(message: string): void {
  toast.add({ type: 'error', title: 'Could not complete that action', description: message })
}

function canvasSnapshot(canvas: CanvasState) {
  return {
    canvasPage: canvas.page,
    canvasRevision: canvas.revision,
    canvasGeneration: canvas.generation,
    canvasSize: canvas.size,
    layerFrames: Object.fromEntries(
      canvas.element_frames.map(({ element, frame }) => [element, frame]),
    ),
  }
}

function byId<T extends { id: string | number }>(items: T[]): Record<string, T> {
  return Object.fromEntries(items.map((item) => [String(item.id), item]))
}
