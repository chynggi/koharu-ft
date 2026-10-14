import type {
  ArrowHead,
  Frame,
  Geometry,
  GradientExtend,
  GradientSpec,
  GradientStop,
  Point,
  ShapeGeometry,
  ShapeStyle,
} from '@koharu/bridge/protocol'
import type { CanvasGradient, CanvasPath, CanvasPathElement } from '@koharu/bridge/canvas'

import type { CanvasTool } from './store'

export type ShapeKind = 'rectangle' | 'ellipse' | 'line' | 'arrow'

// 1a exposes only the kinds that have a real authored geometry. `polygon` and
// `freeform` exist in `ShapeGeometry` and the renderer's polygon path, but a
// user drawing one would get a rectangle with a different tag: there is no
// vertex-editing UI and `shapePolygon` returns the drag box's four corners for
// both. They are deferred with the editing surface, not shipped as placeholders
// (see "1a 범위" in `specs/2026-10-13-shape-tools-design.md`).
export const SHAPE_KINDS: readonly ShapeKind[] = ['rectangle', 'ellipse', 'line', 'arrow']

// Shapes are drawn from a polygon the frontend owns (the renderer only rebuilds
// rectangle/ellipse/line from their parameters), so every kind is approximated
// here and the polygon doubles as the hit-test and selection outline.
const ELLIPSE_SAMPLES = 32

// A frame can be dragged to zero size, and the backend rejects non-finite or
// non-positive dimensions, so the polygon keeps a degenerate floor.
const MIN_FRAME_EXTENT = 1e-6

const ARROW_HEAD: ArrowHead = { length: 18, width: 12 }

export function isShapeTool(tool: CanvasTool): tool is 'shape' {
  return tool === 'shape'
}

/**
 * The outline for one shape kind, already rotated into page coordinates.
 *
 * Rotation happens on the polygon points, matching the backend's
 * `geometry_from_frame`: the frame's own axes are the authored ones, and the
 * angle is how those axes sit on the page.
 */
export function shapePolygon(kind: ShapeKind, frame: Frame): Geometry {
  const width = Math.max(frame.width, MIN_FRAME_EXTENT)
  const height = Math.max(frame.height, MIN_FRAME_EXTENT)
  if (kind === 'ellipse') {
    const center = { x: frame.x + width * 0.5, y: frame.y + height * 0.5 }
    const radius = { x: width * 0.5, y: height * 0.5 }
    const points: Point[] = []
    for (let index = 0; index < ELLIPSE_SAMPLES; index += 1) {
      const angle = (index / ELLIPSE_SAMPLES) * Math.PI * 2
      points.push(
        rotatePoint(
          { x: center.x + Math.cos(angle) * radius.x, y: center.y + Math.sin(angle) * radius.y },
          center,
          frame.angle_degrees,
        ),
      )
    }
    return { points }
  }
  const topLeft = { x: frame.x, y: frame.y }
  const topRight = { x: frame.x + width, y: frame.y }
  const bottomRight = { x: frame.x + width, y: frame.y + height }
  const bottomLeft = { x: frame.x, y: frame.y + height }
  const center = { x: frame.x + width * 0.5, y: frame.y + height * 0.5 }
  if (kind === 'line' || kind === 'arrow') {
    const middle = { x: (topLeft.x + bottomRight.x) * 0.5, y: (topLeft.y + bottomRight.y) * 0.5 }
    return {
      points: [topLeft, middle, bottomRight].map((point) =>
        rotatePoint(point, center, frame.angle_degrees),
      ),
    }
  }
  return {
    points: [topLeft, topRight, bottomRight, bottomLeft].map((point) =>
      rotatePoint(point, center, frame.angle_degrees),
    ),
  }
}

/**
 * The regenerable parameters that draw the shape.
 *
 * `Line`'s endpoints are derived from the polygon rather than stored
 * separately: the backend's `resync_with_polygon` keeps them equal to the first
 * and last point, and the polygon is what a later move rewrites.
 */
export function shapeGeometry(kind: ShapeKind, points: Point[]): ShapeGeometry {
  if (kind === 'rectangle') return { rectangle: { corner_radius: 0 } }
  if (kind === 'ellipse') return 'ellipse'
  const first = points[0] ?? { x: 0, y: 0 }
  const last = points.at(-1) ?? first
  return {
    line: {
      start: [first.x, first.y],
      end: [last.x, last.y],
      head: kind === 'arrow' ? ARROW_HEAD : null,
    },
  }
}

export function defaultShapeStyle(kind: ShapeKind): ShapeStyle {
  // Matches serde's external tagging for `ShapeFill::Solid([u8; 4])`, which is
  // pinned by `the_shape_wire_shape_matches_the_typescript_client`.
  if (kind === 'line' || kind === 'arrow') {
    // A straight segment encloses no area, so it is stroked, never filled.
    return {
      fill: null,
      stroke: {
        color: [255, 255, 255, 255],
        width: 2,
        dash: null,
        cap: 'round',
        join: 'round',
      },
    }
  }
  return { fill: { solid: [255, 255, 255, 255] }, stroke: null }
}

export type GradientKindName = 'linear' | 'radial'

// The renderer's `PreparedGradient` validation rejects a stop count outside
// this range, so the editor must never build one: the inspector disables the
// add button at the ceiling and the remove button at the floor.
export const MIN_GRADIENT_STOPS = 2
export const MAX_GRADIENT_STOPS = 32

/**
 * The gradient a freshly created shape starts with: white to black across the
 * frame, in the normalized frame-local coordinates `GradientSpec` stores.
 *
 * Linear runs left to right (`(0,0)` to `(1,0)`) and radial grows from the
 * frame centre by half its width, so both read the same on a wide frame and
 * neither escapes the unit square that validation requires. `pad` is the
 * editor's default: the last stop's colour continues past the endpoints.
 */
export function defaultGradientSpec(kind: GradientKindName): GradientSpec {
  const stops: GradientStop[] = [
    { offset: 0, color: [255, 255, 255, 255] },
    { offset: 1, color: [0, 0, 0, 255] },
  ]
  const spec: GradientSpec =
    kind === 'linear'
      ? { kind: { linear: { start: [0, 0], end: [1, 0] } }, stops, extend: 'pad' }
      : { kind: { radial: { center: [0.5, 0.5], radius: 0.5 } }, stops, extend: 'pad' }
  return spec
}

/**
 * The gradient kind a spec is authored with, or `null` for a spec the editor
 * cannot name. `sweep` is not exposed by the UI (spec G5), so a spec that
 * carries one leaves both picker options unchecked rather than claiming to be
 * linear; its stops stay editable and picking a kind replaces the geometry.
 */
export function gradientKindName(spec: GradientSpec): GradientKindName | null {
  if ('linear' in spec.kind) return 'linear'
  if ('radial' in spec.kind) return 'radial'
  return null
}

/**
 * The style a fill-kind choice produces.
 *
 * Converting keeps the user's colour: a gradient's first stop becomes the solid
 * colour, and a solid becomes the gradient's first stop. When that would leave
 * both stops identical the second stop takes the kind's default leading colour,
 * so a switch always produces a visible gradient instead of a flat fill.
 * A fill that is already the requested kind is returned untouched.
 */
export function withFillKind(
  style: ShapeStyle,
  kind: 'solid' | 'gradient',
  gradientKind: GradientKindName = 'linear',
): ShapeStyle {
  if (kind === 'solid') {
    if (style.fill && 'solid' in style.fill) return style
    const color = style.fill?.gradient.stops[0].color ?? [255, 255, 255, 255]
    return { ...style, fill: { solid: color } }
  }
  if (style.fill && 'gradient' in style.fill) return style
  const solid = style.fill?.solid ?? null
  const spec = defaultGradientSpec(gradientKind)
  const stops = [...spec.stops] as GradientStop[]
  if (solid) {
    stops[0] = { ...stops[0], color: solid }
    // The second stop takes whichever default differs from the solid colour, so
    // a switch never yields two identical stops and a flat fill.
    const last = stops.length - 1
    stops[last] = {
      ...stops[last],
      color: spec.stops[0].color.every((channel, index) => channel === solid[index])
        ? spec.stops[1].color
        : spec.stops[0].color,
    }
  }
  return { ...style, fill: { gradient: { ...spec, stops } } }
}

/** Replaces a gradient's control geometry, keeping its stops and extend rule. */
export function withGradientKind(spec: GradientSpec, kind: GradientKindName): GradientSpec {
  return { ...spec, kind: defaultGradientSpec(kind).kind }
}

/**
 * Adds a stop in the widest gap between two existing offsets.
 *
 * Splitting the widest gap keeps offsets unique, which the server requires,
 * and the new stop copies its left neighbour's colour so adding one does not
 * change the gradient until the user edits it. The result is ordered by offset.
 */
export function addGradientStop(stops: GradientStop[]): GradientStop[] {
  const sorted = sortStops(stops)
  let widest = 0
  for (let index = 1; index < sorted.length - 1; index += 1) {
    if (sorted[index + 1].offset - sorted[index].offset > sorted[widest + 1].offset - sorted[widest].offset) {
      widest = index
    }
  }
  const left = sorted[widest]
  const right = sorted[widest + 1]
  // Six decimals always lands strictly inside the gap: editing a position keeps
  // neighbours at least `STOP_OFFSET_STEP` apart, so no rounding collision is
  // possible and the new offset never duplicates an existing one.
  const offset = Math.round(((left.offset + right.offset) / 2) * 1e6) / 1e6
  return sortStops([...stops, { offset, color: left.color }])
}

/** The finest offset a stop may be moved to, matching the offset slider. */
export const STOP_OFFSET_STEP = 0.001

/** The offsets a stop at sorted `index` may occupy without colliding. */
export function gradientStopBounds(stops: GradientStop[], index: number): [number, number] {
  const sorted = sortStops(stops)
  const round = (value: number) => Math.round(value / STOP_OFFSET_STEP) * STOP_OFFSET_STEP
  return [
    index === 0 ? 0 : round(sorted[index - 1].offset) + STOP_OFFSET_STEP,
    index === sorted.length - 1 ? 1 : round(sorted[index + 1].offset) - STOP_OFFSET_STEP,
  ]
}

/**
 * Moves one stop to `offset`, or returns `null` when the move is impossible.
 *
 * The offset is snapped and clamped inside the neighbours' bounds so a drag can
 * never produce the duplicate offsets the server rejects. Neighbours closer than
 * two steps leave no legal offset, and the caller simply keeps the old value.
 */
export function moveGradientStop(
  spec: GradientSpec,
  index: number,
  offset: number,
): GradientSpec | null {
  const sorted = sortStops(spec.stops)
  const [lower, upper] = gradientStopBounds(sorted, index)
  if (lower > upper) return null
  const snapped = Math.round(Math.min(Math.max(offset, lower), upper) / STOP_OFFSET_STEP) * STOP_OFFSET_STEP
  const rounded = Math.round(snapped * 1e6) / 1e6
  return sortGradientStops(replaceGradientStop(spec, spec.stops.indexOf(sorted[index]), { offset: rounded }))
}

export function replaceGradientStop(
  spec: GradientSpec,
  index: number,
  patch: Partial<GradientStop>,
): GradientSpec {
  return { ...spec, stops: spec.stops.map((stop, at) => (at === index ? { ...stop, ...patch } : stop)) }
}

export function removeGradientStop(spec: GradientSpec, index: number): GradientSpec {
  return { ...spec, stops: spec.stops.filter((_stop, at) => at !== index) }
}

export function sortGradientStops(spec: GradientSpec): GradientSpec {
  return { ...spec, stops: sortStops(spec.stops) }
}

function sortStops(stops: GradientStop[]): GradientStop[] {
  return [...stops].sort((left, right) => left.offset - right.offset)
}

/**
 * The live WASM preview for a shape being dragged.
 *
 * This is the polygon the frontend already computed, in page coordinates, so
 * the transform is the identity. It exists because the browser canvas draws
 * what it is handed and nothing else — the preview and the committed shape can
 * therefore never disagree about the outline.
 */
export function shapePreviewPath(
  kind: ShapeKind,
  frame: Frame,
  style: ShapeStyle,
): CanvasPath {
  const points = shapePolygon(kind, frame).points
  const elements: CanvasPathElement[] = []
  points.forEach((point, index) => {
    elements.push(index === 0 ? { MoveTo: [point.x, point.y] } : { LineTo: [point.x, point.y] })
  })
  elements.push('Close')
  const arrow = kind === 'arrow' ? arrowHeadPath(points) : []
  elements.push(...arrow)
  const fill = style.fill
  return {
    transform: [1, 0, 0, 1, 0, 0],
    elements,
    // An arrow head is a filled triangle, so it forces a fill even when the
    // author set none, and then it borrows the stroke's colour.
    fill:
      fill && 'solid' in fill
        ? { Solid: { color: fill.solid, rule: 'NonZero' } }
        : fill && 'gradient' in fill
          ? { Gradient: { gradient: previewGradient(fill.gradient, frame), rule: 'NonZero' } }
          : arrow.length > 0 && style.stroke
            ? { Solid: { color: style.stroke.color, rule: 'NonZero' } }
            : null,
    stroke: style.stroke
      ? {
          color: style.stroke.color,
          width: style.stroke.width,
          dash: style.stroke.dash,
          cap: capName(style.stroke.cap),
          join: joinName(style.stroke.join),
        }
      : null,
  }
}

/**
 * Lifts a persisted gradient into the WASM preview's page coordinates.
 *
 * The server leaves this to `koharu_renderer::shape_renderer::fill`, which
 * multiplies normalized coordinates by the frame extent **and stops there**:
 * the rotation lives in `PreparedPath::transform`, which also shifts the frame's
 * local origin onto the page, so vello applies both to the brush for free. This
 * function cannot copy that, because the preview's `transform` is the identity
 * and `shapePolygon` bakes the frame's position and angle into the outline
 * points instead. Doing only the multiplication would leave the control points
 * in the frame's own box — unrotated *and* offset by the frame origin whenever
 * the frame is not at `(0,0)` — while the outline moved. Applying the same
 * `rotatePoint` about the same centre the polygon uses, to the same
 * `frame.origin + normalized * extent` the server's affine would produce, is
 * what keeps the two in step.
 */
function previewGradient(spec: GradientSpec, frame: Frame): CanvasGradient {
  // The same degenerate floor `shapePolygon` applies, and for the same reason:
  // a zero-width frame would otherwise give a linear gradient two identical
  // endpoints and a radial one a zero radius, both of which
  // `PreparedPath::validate` rejects during the drag. Sharing the floor keeps
  // the control points and the outline describing one box.
  const width = Math.max(frame.width, MIN_FRAME_EXTENT)
  const height = Math.max(frame.height, MIN_FRAME_EXTENT)
  const center = { x: frame.x + width * 0.5, y: frame.y + height * 0.5 }
  const toPage = ([x, y]: [number, number]): [number, number] => {
    const boxed = { x: frame.x + x * width, y: frame.y + y * height }
    const rotated = rotatePoint(boxed, center, frame.angle_degrees)
    return [rotated.x, rotated.y]
  }
  const kind = spec.kind
  return {
    kind:
      'linear' in kind
        ? {
            Linear: {
              start: toPage(kind.linear.start),
              end: toPage(kind.linear.end),
            },
          }
        : 'radial' in kind
          ? {
              Radial: {
                center: toPage(kind.radial.center),
                // The radius is a fraction of the frame's width, matching the
                // server, and rotation leaves its length unchanged.
                radius: kind.radial.radius * width,
              },
            }
          : {
              Sweep: {
                center: toPage(kind.sweep.center),
                // Angles are absolute degrees, not normalized: no frame factor
                // applies. They *do* rotate, though — vello would rotate a
                // sweep's angular field along with the path, so the preview has
                // to bake that rotation into the angles to land in the same
                // place. 1b's editor never creates a sweep (spec G5); this arm
                // exists so an imported one still previews in alignment.
                start_degrees: kind.sweep.start_degrees + frame.angle_degrees,
                end_degrees: kind.sweep.end_degrees + frame.angle_degrees,
              },
            },
    stops: spec.stops,
    extend: extendName(spec.extend),
  }
}

export function hexToColor(hex: string, alpha = 255): [number, number, number, number] {
  const digits = hex.startsWith('#') ? hex.slice(1) : hex
  return [
    Number.parseInt(digits.slice(0, 2), 16),
    Number.parseInt(digits.slice(2, 4), 16),
    Number.parseInt(digits.slice(4, 6), 16),
    alpha,
  ]
}

export function colorToHex([red, green, blue]: [number, number, number, number]): string {
  return `#${[red, green, blue]
    .map((channel) => channel.toString(16).padStart(2, '0'))
    .join('')}`.toUpperCase()
}

function arrowHeadPath(points: Point[]): CanvasPathElement[] {
  const start = points[0]
  const end = points.at(-1)
  if (!start || !end) return []
  const delta = { x: end.x - start.x, y: end.y - start.y }
  const length = Math.hypot(delta.x, delta.y)
  // A zero-length drag has no direction, so there is no perpendicular to place
  // a head on; the backend drops the head for the same reason.
  if (!Number.isFinite(length) || length <= 0) return []
  const direction = { x: delta.x / length, y: delta.y / length }
  const base = {
    x: end.x - direction.x * ARROW_HEAD.length,
    y: end.y - direction.y * ARROW_HEAD.length,
  }
  const perpendicular = {
    x: -direction.y * ARROW_HEAD.width * 0.5,
    y: direction.x * ARROW_HEAD.width * 0.5,
  }
  return [
    { MoveTo: [end.x, end.y] },
    { LineTo: [base.x + perpendicular.x, base.y + perpendicular.y] },
    { LineTo: [base.x - perpendicular.x, base.y - perpendicular.y] },
    'Close',
  ]
}

function rotatePoint(point: Point, center: Point, angleDegrees: number): Point {
  if (!angleDegrees) return point
  const angle = (angleDegrees * Math.PI) / 180
  const cos = Math.cos(angle)
  const sin = Math.sin(angle)
  const x = point.x - center.x
  const y = point.y - center.y
  return { x: center.x + x * cos - y * sin, y: center.y + x * sin + y * cos }
}

function capName(cap: 'butt' | 'round' | 'square') {
  return cap === 'butt' ? 'Butt' : cap === 'round' ? 'Round' : 'Square'
}

function joinName(join: 'miter' | 'round' | 'bevel') {
  return join === 'miter' ? 'Miter' : join === 'round' ? 'Round' : 'Bevel'
}

// `PreparedGradientExtend` is a unit-variant enum, so serde spells it with the
// Rust variant names just as `PreparedLineCap`/`PreparedLineJoin` do.
function extendName(extend: GradientExtend) {
  return extend === 'pad' ? 'Pad' : extend === 'repeat' ? 'Repeat' : 'Reflect'
}

