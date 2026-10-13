import type {
  ArrowHead,
  Frame,
  Geometry,
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
  const center = { x: frame.x + frame.width * 0.5, y: frame.y + frame.height * 0.5 }
  const toPage = ([x, y]: [number, number]): [number, number] => {
    const boxed = { x: frame.x + x * frame.width, y: frame.y + y * frame.height }
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
                radius: kind.radial.radius * frame.width,
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
function extendName(extend: 'pad' | 'repeat' | 'reflect') {
  return extend === 'pad' ? 'Pad' : extend === 'repeat' ? 'Repeat' : 'Reflect'
}

