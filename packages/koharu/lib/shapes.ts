import type {
  ArrowHead,
  Frame,
  Geometry,
  GradientExtend,
  GradientSpec,
  GradientStop,
  PatternSpec,
  Point,
  ShapeFill,
  ShapeGeometry,
  ShapeStyle,
} from '@koharu/bridge/protocol'
import type {
  CanvasGradient,
  CanvasPath,
  CanvasPathElement,
  CanvasPattern,
  CanvasPatternImage,
} from '@koharu/bridge/canvas'

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

/**
 * The pattern a freshly created shape starts with.
 *
 * `tile` with a quarter-frame pitch reads as a texture rather than a single
 * stretched image, which is what makes the mode's parameters visible
 * immediately. The origin sits at the frame's top-left and the rotation is zero,
 * so the initial state is the axis-aligned grid a user expects. `fallback` is
 * the colour the renderer draws until an image is uploaded — it is deliberately
 * the user's current colour, so switching kinds never changes what is on screen.
 */
export function defaultPatternSpec(
  fallback: [number, number, number, number] = [255, 255, 255, 255],
): PatternSpec {
  return {
    repeat: 'tile',
    origin: [0, 0],
    scale: 0.25,
    rotation_degrees: 0,
    fallback,
  }
}

// The renderer's `PreparedGradient` validation rejects a stop count outside
// this range, so the editor must never build one: the inspector disables the
// add button at the ceiling and the remove button at the floor. Because the
// buttons are disabled there is no message to show for either bound, so this
// feature deliberately ships no `tooFewStops`/`tooManyStops` key rather than
// carrying translations no code path renders.
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
 *
 * `pattern` carries its colour in `fallback` rather than in stops, so converting
 * to or from a pattern moves the colour through that field. The pattern's own
 * parameters are replaced with defaults on the way in: a `PatternSpec` has no
 * meaningful identity to preserve from a solid or a gradient.
 */
export function withFillKind(
  style: ShapeStyle,
  kind: 'solid' | 'gradient' | 'pattern',
  gradientKind: GradientKindName = 'linear',
): ShapeStyle {
  if (kind === 'solid') {
    if (style.fill && 'solid' in style.fill) return style
    const color = fillColor(style.fill) ?? [255, 255, 255, 255]
    return { ...style, fill: { solid: color } }
  }
  if (kind === 'pattern') {
    if (style.fill && 'pattern' in style.fill) return style
    const color = fillColor(style.fill) ?? [255, 255, 255, 255]
    return { ...style, fill: { pattern: defaultPatternSpec(color) } }
  }
  if (style.fill && 'gradient' in style.fill) return style
  // The source colour comes from whichever kind is active, not from `solid`
  // alone: a pattern carries its colour in `fallback`, so reading only `solid`
  // silently dropped it and a pattern-to-gradient switch came out white. The
  // `withFillKind` test "keeps a gradient visible when converting a pattern
  // back" pins this.
  const source = fillColor(style.fill)
  const spec = defaultGradientSpec(gradientKind)
  const stops = [...spec.stops] as GradientStop[]
  if (source) {
    stops[0] = { ...stops[0], color: source }
    // The second stop takes whichever default differs from the source colour, so
    // a switch never yields two identical stops and a flat fill.
    const last = stops.length - 1
    stops[last] = {
      ...stops[last],
      color: spec.stops[0].color.every((channel, index) => channel === source[index])
        ? spec.stops[1].color
        : spec.stops[0].color,
    }
  }
  return { ...style, fill: { gradient: { ...spec, stops } } }
}

/**
 * The representative colour of a fill, for the flat colour well and for
 * conversions between fill kinds that must not lose the user's colour.
 *
 * A pattern's `fallback` is exactly the colour it draws when its asset is
 * missing, which is the closest thing it has to a colour of its own.
 */
export function fillColor(fill: ShapeFill | null): [number, number, number, number] | null {
  if (!fill) return null
  if ('solid' in fill) return fill.solid
  if ('gradient' in fill) return fill.gradient.stops[0]?.color ?? null
  return fill.pattern.fallback
}

/**
 * Which fill kind a style is currently showing.
 *
 * A null fill reads as `solid`, matching {@link defaultShapeStyle}: a shape with
 * no fill at all behaves like one whose colour is simply not drawn.
 *
 * This exists because a pattern fill has **neither** the `solid` nor the
 * `gradient` key. A caller that tested `'gradient' in fill` to mean "not solid"
 * would classify a pattern as a solid and highlight the wrong control, so every
 * kind test goes through this one function.
 */
export function fillKind(fill: ShapeFill | null): 'solid' | 'gradient' | 'pattern' {
  if (!fill) return 'solid'
  if ('solid' in fill) return 'solid'
  if ('gradient' in fill) return 'gradient'
  return 'pattern'
}

/** Replaces a gradient's control geometry, keeping its stops and extend rule. */
export function withGradientKind(spec: GradientSpec, kind: GradientKindName): GradientSpec {
  return { ...spec, kind: defaultGradientSpec(kind).kind }
}

/** The offset grid every edit is placed on, in normalized frame coordinates. */
export const STOP_OFFSET_STEP = 0.001

/**
 * The number of distinct offsets the grid can hold.
 *
 * `MAX_GRADIENT_STOPS` is far below this, which is what makes normalisation
 * total: any server-acceptable spec can be spread onto free grid points.
 */
const GRID_POINTS = Math.round(1 / STOP_OFFSET_STEP) + 1

/** The grid point nearest `offset`, clamped into the unit interval. */
function snapToGrid(offset: number): number {
  const snapped = Math.round(offset / STOP_OFFSET_STEP) * STOP_OFFSET_STEP
  return Math.round(Math.min(Math.max(snapped, 0), 1) * 1e6) / 1e6
}

/**
 * Moves every stop onto the offset grid while preserving order, count, and
 * storage positions.
 *
 * The server only requires distinct finite offsets in `0..=1`, so a spec read
 * from disk can carry neighbours closer together than one grid step (or off the
 * grid entirely). Editing such a list on the grid would otherwise round two
 * stops onto the same point and commit an offset pair the server rejects. This
 * establishes the editor's invariant — offsets are distinct grid points in
 * ascending order — before any edit is built, so every later operation can rely
 * on it.
 *
 * Offsets are re-assigned by sorted rank back into their original slots, so the
 * caller's indices stay valid and the stored order is left alone. Collisions are
 * resolved by a forward sweep and, if that would push past `1`, a backward one;
 * `MAX_GRADIENT_STOPS` stops need at most 31 of the grid's points, so both
 * sweeps always fit.
 */
export function normalizeGradientStops(stops: GradientStop[]): GradientStop[] {
  const ranked = stops
    .map((stop, index) => ({ index, offset: snapToGrid(stop.offset) }))
    .sort((left, right) => left.offset - right.offset || left.index - right.index)
  for (let rank = 1; rank < ranked.length; rank += 1) {
    if (ranked[rank].offset <= ranked[rank - 1].offset) {
      ranked[rank].offset = snapToGrid(ranked[rank - 1].offset + STOP_OFFSET_STEP)
    }
  }
  for (let rank = ranked.length - 2; rank >= 0; rank -= 1) {
    if (ranked[rank].offset >= ranked[rank + 1].offset) {
      ranked[rank].offset = snapToGrid(ranked[rank + 1].offset - STOP_OFFSET_STEP)
    }
  }
  const offsets = new Map(ranked.map((entry) => [entry.index, entry.offset]))
  return stops.map((stop, index) => ({ ...stop, offset: offsets.get(index)! }))
}

/** The spec with its stops normalised onto the offset grid. */
export function normalizeGradientSpec(spec: GradientSpec): GradientSpec {
  return { ...spec, stops: normalizeGradientStops(spec.stops) }
}

/**
 * Adds a stop in the widest gap between two existing offsets, or returns `null`
 * when no free grid point is left.
 *
 * The offset is the grid midpoint of the widest gap, so it is a distinct grid
 * point strictly between two existing stops whenever that gap spans at least two
 * steps; a gap of one step has no point between its ends and the insertion is
 * refused rather than committed as a duplicate. The new stop copies its left
 * neighbour's colour, so adding one does not change the gradient until the user
 * edits it. The result is ordered by offset.
 *
 * `null` is the signal the caller uses to disable the add control, so the UI can
 * never build the duplicate offsets or the out-of-range count the server rejects.
 */
export function addGradientStop(stops: GradientStop[]): GradientStop[] | null {
  const normalized = normalizeGradientStops(stops)
  const sorted = sortStops(normalized)
  if (sorted.length >= MAX_GRADIENT_STOPS || sorted.length + 1 > GRID_POINTS) return null
  let widest = 0
  let widestGap = 0
  for (let index = 0; index < sorted.length - 1; index += 1) {
    const gap = sorted[index + 1].offset - sorted[index].offset
    if (gap > widestGap) {
      widest = index
      widestGap = gap
    }
  }
  const left = sorted[widest]
  const right = sorted[widest + 1]
  // A gap of one step leaves nothing strictly between its ends, and rounding a
  // fractional midpoint would land on a neighbour.
  if (Math.round(widestGap / STOP_OFFSET_STEP) < 2) return null
  const offset = snapToGrid((left.offset + right.offset) / 2)
  if (offset <= left.offset || offset >= right.offset) return null
  return sortStops([...sorted, { offset, color: left.color }])
}

/**
 * The offsets a stop at sorted `index` may occupy without colliding.
 *
 * The ends are the grid points immediately inside the neighbours, so a value
 * taken from this range is always strictly between them and never duplicates
 * one. `lower > upper` means the neighbours leave no free grid point and the
 * control must be disabled; the range always contains the stop's own offset,
 * because normalisation places it on a grid point strictly between its
 * neighbours, so a slider thumb is never clamped away from its value.
 */
export function gradientStopBounds(stops: GradientStop[], index: number): [number, number] {
  const normalized = normalizeGradientStops(stops)
  const sorted = sortStops(normalized)
  const lower = index === 0 ? 0 : snapToGrid(sorted[index - 1].offset + STOP_OFFSET_STEP)
  const upper =
    index === sorted.length - 1 ? 1 : snapToGrid(sorted[index + 1].offset - STOP_OFFSET_STEP)
  return [lower, upper]
}

/**
 * Moves one stop to `offset`, or returns `null` when the move is impossible.
 *
 * The offset is snapped to the grid and clamped inside the neighbours' bounds,
 * so the result is always a distinct grid point strictly between them. When the
 * neighbours leave no free grid point the move is refused and the caller keeps
 * the old value, rather than committing a duplicate the server rejects.
 */
export function moveGradientStop(
  spec: GradientSpec,
  index: number,
  offset: number,
): GradientSpec | null {
  const normalized = normalizeGradientSpec(spec)
  const sorted = sortStops(normalized.stops)
  const [lower, upper] = gradientStopBounds(sorted, index)
  if (lower > upper) return null
  const snapped = snapToGrid(Math.min(Math.max(offset, lower), upper))
  if (snapped < lower || snapped > upper) return null
  return sortGradientStops(
    replaceGradientStop(normalized, normalized.stops.indexOf(sorted[index]), { offset: snapped }),
  )
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
  patternImage: CanvasPatternImage | null = null,
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
          : fill && 'pattern' in fill
            ? {
                Pattern: {
                  pattern: previewPattern(fill.pattern, frame, patternImage),
                  rule: 'NonZero',
                },
              }
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

/**
 * Lifts a persisted pattern into the WASM preview's page coordinates.
 *
 * This deliberately does **not** copy `previewGradient`'s rule, and the
 * difference is structural rather than an oversight. A gradient is defined by
 * its control points, so rotating those points rotates the ramp and nothing else
 * is needed. A pattern's *tiling direction* must rotate as well, and rotating
 * only the tile's origin leaves the grid upright.
 *
 * The server draws `composed = P · T(origin_local) · R(θ_pattern) · S(pitch)`
 * where `P = R(θ_frame) + translate` (`shape_renderer::transform` is rotation
 * and translation with no scale). That identity equals
 * `T(P(origin_local)) · R(θ_frame + θ_pattern) · S(pitch)`, so with the preview's
 * identity path transform the brush has to carry the page-space origin **and the
 * summed rotation**. Hence `rotation_degrees` here is a sum, and hence
 * `PreparedPattern::rotation_degrees` is an `f32` while the saved `PatternSpec`
 * stays an integer.
 */
export function previewPattern(
  spec: PatternSpec,
  frame: Frame,
  image: CanvasPatternImage | null,
): CanvasPattern {
  const width = Math.max(frame.width, MIN_FRAME_EXTENT)
  const height = Math.max(frame.height, MIN_FRAME_EXTENT)
  const center = { x: frame.x + width * 0.5, y: frame.y + height * 0.5 }
  const origin = rotatePoint(
    { x: frame.x + spec.origin[0] * width, y: frame.y + spec.origin[1] * height },
    center,
    frame.angle_degrees,
  )
  return {
    repeat: spec.repeat === 'tile' ? 'Tile' : 'Stretch',
    // The server spreads `scale` by the frame's width, the same convention the
    // gradient's radius uses.
    origin: [origin.x, origin.y],
    scale: spec.scale * width,
    // The sum the identity path transform cannot supply. See the note above.
    rotation_degrees: frame.angle_degrees + spec.rotation_degrees,
    fallback: spec.fallback,
    extent: [width, height],
    image,
  }
}

export function hexToColor(hex: string, alpha = 255): [number, number, number, number] {  const digits = hex.startsWith('#') ? hex.slice(1) : hex
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

