import { describe, expect, it } from 'vitest'

import {
  MAX_GRADIENT_STOPS,
  MIN_GRADIENT_STOPS,
  SHAPE_KINDS,
  addGradientStop,
  defaultGradientSpec,
  defaultShapeStyle,
  gradientKindName,
  gradientStopBounds,
  isShapeTool,
  moveGradientStop,
  removeGradientStop,
  shapeGeometry,
  shapePolygon,
  shapePreviewPath,
  withFillKind,
  withGradientKind,
} from '@/lib/shapes'

const frame = { x: 10, y: 20, width: 100, height: 50, angle_degrees: 0 }

describe('shapePolygon', () => {
  it('builds a closed rectangle from the frame corners', () => {
    expect(shapePolygon('rectangle', frame).points).toEqual([
      { x: 10, y: 20 },
      { x: 110, y: 20 },
      { x: 110, y: 70 },
      { x: 10, y: 70 },
    ])
  })

  it('samples an ellipse into at least three points', () => {
    const points = shapePolygon('ellipse', frame).points
    expect(points.length).toBeGreaterThanOrEqual(3)
    expect(points.every((point) => Number.isFinite(point.x) && Number.isFinite(point.y))).toBe(true)
  })

  it('samples a line into three collinear points', () => {
    const points = shapePolygon('line', frame).points
    expect(points.length).toBe(3)
    expect(points[0]).toEqual({ x: 10, y: 20 })
    expect(points[2]).toEqual({ x: 110, y: 70 })
  })

  it('keeps every point finite for a zero-sized frame', () => {
    const points = shapePolygon('ellipse', { ...frame, width: 0, height: 0 }).points
    expect(points.every((point) => Number.isFinite(point.x) && Number.isFinite(point.y))).toBe(true)
  })

  it('rotates the rectangle corners about the frame centre', () => {
    const points = shapePolygon('rectangle', {
      x: 0,
      y: 0,
      width: 100,
      height: 50,
      angle_degrees: 90,
    }).points
    const [topLeft, topRight, bottomRight, bottomLeft] = points
    expect(topLeft.x).toBeCloseTo(75, 6)
    expect(topLeft.y).toBeCloseTo(-25, 6)
    expect(topRight.x).toBeCloseTo(75, 6)
    expect(topRight.y).toBeCloseTo(75, 6)
    expect(bottomRight.x).toBeCloseTo(25, 6)
    expect(bottomRight.y).toBeCloseTo(75, 6)
    expect(bottomLeft.x).toBeCloseTo(25, 6)
    expect(bottomLeft.y).toBeCloseTo(-25, 6)
  })

  it('preserves the frame centre when a tall ellipse rotates', () => {
    const points = shapePolygon('ellipse', {
      x: 0,
      y: 0,
      width: 40,
      height: 120,
      angle_degrees: 37,
    }).points
    const centre = points.reduce(
      (sum, point) => ({ x: sum.x + point.x, y: sum.y + point.y }),
      { x: 0, y: 0 },
    )
    expect(centre.x / points.length).toBeCloseTo(20, 6)
    expect(centre.y / points.length).toBeCloseTo(60, 6)
  })
})

describe('shapeGeometry', () => {
  it('keeps the line endpoints in step with the polygon that moves', () => {
    const points = shapePolygon('line', frame).points
    const geometry = shapeGeometry('line', points)
    expect(geometry).toEqual({
      line: { start: [10, 20], end: [110, 70], head: null },
    })
  })

  it('gives an arrow a finite head even for a zero-length drag', () => {
    const points = shapePolygon('arrow', { ...frame, width: 0, height: 0 }).points
    const head = (shapeGeometry('arrow', points) as { line: { head: unknown } }).line.head
    expect(head).not.toBeNull()
    expect(Object.values(head as Record<string, number>).every(Number.isFinite)).toBe(true)
  })

  it('describes a rectangle with its corner radius', () => {
    expect(shapeGeometry('rectangle', shapePolygon('rectangle', frame).points)).toEqual({
      rectangle: { corner_radius: 0 },
    })
    expect(shapeGeometry('ellipse', shapePolygon('ellipse', frame).points)).toBe('ellipse')
  })
})

describe('defaultShapeStyle', () => {
  it('fills without a stroke', () => {
    const style = defaultShapeStyle('rectangle')
    expect(style.fill).not.toBeNull()
    expect(style.stroke).toBeNull()
  })

  it('serializes the fill the way serde does', () => {
    expect(defaultShapeStyle('ellipse')).toEqual({ fill: { solid: [255, 255, 255, 255] }, stroke: null })
  })

  it('gives a line a stroke, because a line has no area to fill', () => {
    const line = defaultShapeStyle('line')
    expect(line.fill).toBeNull()
    expect(line.stroke).toMatchObject({ width: 2, cap: 'round', join: 'round', dash: null })
    expect(defaultShapeStyle('arrow').stroke).toMatchObject({ width: 2 })
  })
})

describe('defaultGradientSpec', () => {
  it('produces a two-stop linear gradient in normalized coordinates', () => {
    const spec = defaultGradientSpec('linear')
    expect(spec.kind).toEqual({ linear: { start: [0, 0], end: [1, 0] } })
    expect(spec.stops).toHaveLength(2)
    expect(spec.stops[0].offset).toBe(0)
    expect(spec.stops[1].offset).toBe(1)
    expect(spec.extend).toBe('pad')
  })

  it('keeps every normalized coordinate inside the unit square', () => {
    for (const kind of ['linear', 'radial'] as const) {
      const spec = defaultGradientSpec(kind)
      // `GradientSpec.kind` also carries `sweep`, which `defaultGradientSpec`
      // never produces; naming that case keeps this loop from passing on an
      // empty coordinate list if it ever did.
      if ('sweep' in spec.kind) throw new Error('defaultGradientSpec produced a sweep')
      const numbers =
        'linear' in spec.kind
          ? [...spec.kind.linear.start, ...spec.kind.linear.end]
          : [...spec.kind.radial.center, spec.kind.radial.radius]
      expect(numbers.every((value) => Number.isFinite(value))).toBe(true)
      expect(numbers.filter((value) => value < 0 || value > 1)).toHaveLength(0)
    }
  })
})

describe('withFillKind', () => {
  const solid = { fill: { solid: [10, 20, 30, 255] as [number, number, number, number] }, stroke: null }

  it('turns a solid into a gradient that starts from its colour', () => {
    const style = withFillKind(solid, 'gradient')
    expect(style.fill && 'gradient' in style.fill).toBe(true)
    if (!style.fill || !('gradient' in style.fill)) throw new Error('expected a gradient')
    expect(style.fill.gradient.stops[0].color).toEqual([10, 20, 30, 255])
    // A white-to-black default would collapse to one colour here; the second
    // stop must keep the gradient visible.
    expect(style.fill.gradient.stops[1].color).not.toEqual([10, 20, 30, 255])
    expect(gradientKindName(style.fill.gradient)).toBe('linear')
  })

  it('honours the requested gradient kind', () => {
    const style = withFillKind(solid, 'gradient', 'radial')
    if (!style.fill || !('gradient' in style.fill)) throw new Error('expected a gradient')
    expect(gradientKindName(style.fill.gradient)).toBe('radial')
  })

  it('turns a gradient back into the colour of its first stop', () => {
    const gradient = withFillKind(solid, 'gradient')
    expect(withFillKind(gradient, 'solid')).toEqual(solid)
  })

  it('is idempotent for the kind already in use', () => {
    expect(withFillKind(solid, 'solid')).toBe(solid)
    const gradient = withFillKind(solid, 'gradient')
    expect(withFillKind(gradient, 'gradient')).toBe(gradient)
  })

  it('produces a fill even when the shape had none', () => {
    expect(withFillKind({ fill: null, stroke: null }, 'gradient')).toEqual({
      fill: { gradient: defaultGradientSpec('linear') },
      stroke: null,
    })
  })
})

describe('gradient stop editing', () => {
  const spec = defaultGradientSpec('linear')

  it('adds a stop without duplicating an offset', () => {
    let current = spec
    for (let index = 0; index < MAX_GRADIENT_STOPS - MIN_GRADIENT_STOPS; index += 1) {
      current = { ...current, stops: addGradientStop(current.stops) }
    }
    const offsets = current.stops.map((stop) => stop.offset)
    expect(offsets).toHaveLength(MAX_GRADIENT_STOPS)
    expect(new Set(offsets).size).toBe(MAX_GRADIENT_STOPS)
    expect(offsets.every((offset) => offset >= 0 && offset <= 1)).toBe(true)
    expect(current.stops.map((stop) => stop.offset)).toEqual([...offsets].sort((a, b) => a - b))
  })

  it('keeps a moved stop strictly between its neighbours', () => {
    const three = { ...spec, stops: addGradientStop(spec.stops) }
    const moved = moveGradientStop(three, 1, 5)
    if (!moved) throw new Error('expected a move')
    // A drag past the last stop clamps one step short of it rather than
    // colliding with it, which the server rejects as a duplicate offset.
    expect(moved.stops.map((stop) => stop.offset)).toEqual([0, 0.999, 1])
  })

  it('refuses a move when the neighbours leave no room', () => {
    const tight = {
      ...spec,
      stops: [
        { offset: 0, color: [0, 0, 0, 255] as [number, number, number, number] },
        { offset: 0.0005, color: [255, 255, 255, 255] as [number, number, number, number] },
        { offset: 0.001, color: [0, 0, 0, 255] as [number, number, number, number] },
      ],
    }
    const [lower, upper] = gradientStopBounds(tight.stops, 1)
    expect(lower).toBeGreaterThan(upper)
    expect(moveGradientStop(tight, 1, 0.0005)).toBeNull()
  })

  it('removes a stop and replaces one gradient kind in place', () => {
    expect(removeGradientStop(spec, 0).stops).toEqual([spec.stops[1]])
    expect(gradientKindName(withGradientKind(spec, 'radial'))).toBe('radial')
    expect(withGradientKind(spec, 'radial').stops).toEqual(spec.stops)
  })
})

describe('shapePreviewPath', () => {
  // This is the only place the frontend hand-builds the WASM `PreparedPath`
  // wire shape, so the literals are asserted rather than inferred. A wrong
  // spelling makes `beginShape`'s `path.validate()` throw during a drag.
  const stroke = {
    color: [0, 0, 0, 255] as [number, number, number, number],
    width: 2,
    dash: null,
    cap: 'butt' as const,
    join: 'miter' as const,
  }

  it('serializes a filled rectangle with PascalCase fill and no arrow tail', () => {
    const path = shapePreviewPath('rectangle', frame, {
      fill: { solid: [10, 20, 30, 255] },
      stroke: null,
    })

    expect(path).toEqual({
      transform: [1, 0, 0, 1, 0, 0],
      elements: [
        { MoveTo: [10, 20] },
        { LineTo: [110, 20] },
        { LineTo: [110, 70] },
        { LineTo: [10, 70] },
        'Close',
      ],
      fill: { Solid: { color: [10, 20, 30, 255], rule: 'NonZero' } },
      stroke: null,
    })
  })

  it('serializes an arrow with a stroked fill borrowed from the stroke', () => {
    const path = shapePreviewPath('arrow', frame, { fill: null, stroke })

    const points = shapePolygon('arrow', frame).points
    const start = points[0]
    const end = points.at(-1)!
    const delta = { x: end.x - start.x, y: end.y - start.y }
    const length = Math.hypot(delta.x, delta.y)
    const direction = { x: delta.x / length, y: delta.y / length }
    // The head geometry lives in `ARROW_HEAD`; assert the wire tail shape and
    // let the numeric values come from the same constants the module uses.
    const head = { length: 18, width: 12 }
    const base = {
      x: end.x - direction.x * head.length,
      y: end.y - direction.y * head.length,
    }
    const perpendicular = {
      x: (-direction.y * head.width) / 2,
      y: (direction.x * head.width) / 2,
    }

    expect(path.transform).toEqual([1, 0, 0, 1, 0, 0])
    expect(path.elements.slice(0, 3)).toEqual([
      { MoveTo: [start.x, start.y] },
      { LineTo: [points[1].x, points[1].y] },
      { LineTo: [end.x, end.y] },
    ])
    expect(path.elements[3]).toBe('Close')
    expect(path.elements.slice(4)).toEqual([
      { MoveTo: [end.x, end.y] },
      { LineTo: [base.x + perpendicular.x, base.y + perpendicular.y] },
      { LineTo: [base.x - perpendicular.x, base.y - perpendicular.y] },
      'Close',
    ])
    // The head is a filled triangle, so it forces a fill borrowing the stroke.
    expect(path.fill).toEqual({ Solid: { color: [0, 0, 0, 255], rule: 'NonZero' } })
    // Cap/join are PascalCase enum spellings, matching serde.
    expect(path.stroke).toEqual({
      color: [0, 0, 0, 255],
      width: 2,
      dash: null,
      cap: 'Butt',
      join: 'Miter',
    })
  })

  it('keeps a round/square stroke spelled in PascalCase', () => {
    const path = shapePreviewPath('line', frame, {
      fill: null,
      stroke: { ...stroke, cap: 'round', join: 'bevel' },
    })

    expect(path.stroke).toMatchObject({ cap: 'Round', join: 'Bevel' })
  })
})

describe('shapePreviewPath with a gradient', () => {
  type PreviewPath = ReturnType<typeof shapePreviewPath>

  const gradientOf = (path: PreviewPath) => {
    const fill = path.fill
    if (!fill || !('Gradient' in fill)) throw new Error('expected a gradient fill')
    return fill.Gradient.gradient
  }

  const linearOf = (path: PreviewPath) => {
    const kind = gradientOf(path).kind
    if (!('Linear' in kind)) throw new Error('expected a linear gradient')
    return kind.Linear
  }

  it('spreads normalized coordinates onto the frame and spells the wire shape', () => {
    // 리터럴을 직접 넘겨 contextual typing 으로 튜플 타입을 얻는다 — 이 파일의
    // 기존 테스트가 쓰는 방식이다(`as const` 를 쓰지 않는다).
    const path = shapePreviewPath('rectangle', frame, {
      fill: {
        gradient: {
          kind: { linear: { start: [0, 0], end: [1, 1] } },
          stops: [
            { offset: 0, color: [1, 2, 3, 255] },
            { offset: 1, color: [4, 5, 6, 255] },
          ],
          extend: 'pad',
        },
      },
      stroke: null,
    })

    expect(path).toEqual({
      transform: [1, 0, 0, 1, 0, 0],
      elements: [
        { MoveTo: [10, 20] },
        { LineTo: [110, 20] },
        { LineTo: [110, 70] },
        { LineTo: [10, 70] },
        'Close',
      ],
      // 서버와 같은 규칙: local = normalized * frame_extent.
      // `frame` 은 이 파일 상단의 {x:10,y:20,width:100,height:50} 이다.
      fill: {
        Gradient: {
          gradient: {
            kind: { Linear: { start: [10, 20], end: [110, 70] } },
            stops: [
              { offset: 0, color: [1, 2, 3, 255] },
              { offset: 1, color: [4, 5, 6, 255] },
            ],
            extend: 'Pad',
          },
          rule: 'NonZero',
        },
      },
      stroke: null,
    })
  })

  // The frontend's `transform` is the identity and `shapePolygon` bakes the
  // rotation into the outline points (`shapes.ts`'s `rotatePoint`), so the
  // gradient control points must be rotated into page coordinates too. These
  // assertions pin exact coordinates rather than "it moved": a rotation in the
  // wrong direction, about the wrong centre, or left axis-aligned would all
  // satisfy a `not.toEqual`. `toBeCloseTo` because cos 90° is 6.12e-17, not 0.
  it('leaves an unrotated gradient spanning the frame', () => {
    const linear = linearOf(
      shapePreviewPath(
        'rectangle',
        { x: 0, y: 0, width: 200, height: 100, angle_degrees: 0 },
        { fill: { gradient: defaultGradientSpec('linear') }, stroke: null },
      ),
    )

    expect(linear.start[0]).toBeCloseTo(0, 6)
    expect(linear.start[1]).toBeCloseTo(0, 6)
    expect(linear.end[0]).toBeCloseTo(200, 6)
    expect(linear.end[1]).toBeCloseTo(0, 6)
  })

  it('rotates the gradient about the frame centre by the frame angle', () => {
    const rotated = linearOf(
      shapePreviewPath(
        'rectangle',
        { x: 0, y: 0, width: 200, height: 100, angle_degrees: 90 },
        { fill: { gradient: defaultGradientSpec('linear') }, stroke: null },
      ),
    )

    // (0,0) and (200,0) turned 90° about (100,50).
    expect(rotated.start[0]).toBeCloseTo(150, 6)
    expect(rotated.start[1]).toBeCloseTo(-50, 6)
    expect(rotated.end[0]).toBeCloseTo(150, 6)
    expect(rotated.end[1]).toBeCloseTo(150, 6)
  })

  it('offsets an unrotated gradient by the frame origin', () => {
    // The server's translation lives in `PreparedPath::transform`; the preview
    // has no transform, so it must fold the frame origin into the points. A
    // bare `normalized * extent` would put these at (0,0)->(100,0).
    const linear = linearOf(
      shapePreviewPath('rectangle', frame, {
        fill: { gradient: defaultGradientSpec('linear') },
        stroke: null,
      }),
    )

    expect(linear.start[0]).toBeCloseTo(10, 6)
    expect(linear.start[1]).toBeCloseTo(20, 6)
    expect(linear.end[0]).toBeCloseTo(110, 6)
    expect(linear.end[1]).toBeCloseTo(20, 6)
  })

  it('scales a radial gradient radius by the frame width', () => {
    const radial = gradientOf(
      shapePreviewPath(
        'rectangle',
        { x: 0, y: 0, width: 200, height: 100, angle_degrees: 0 },
        { fill: { gradient: defaultGradientSpec('radial') }, stroke: null },
      ),
    ).kind
    if (!('Radial' in radial)) throw new Error('expected a radial gradient')

    // Centre (0.5,0.5) on a 200x100 frame, radius 0.5 of the width.
    expect(radial.Radial.center[0]).toBeCloseTo(100, 6)
    expect(radial.Radial.center[1]).toBeCloseTo(50, 6)
    expect(radial.Radial.radius).toBeCloseTo(100, 6)
  })

  it('keeps a degenerate frame drawable', () => {
    // `PreparedPath::validate` rejects a linear gradient whose endpoints are
    // equal, so a zero-width frame must not produce one during the drag.
    const linear = linearOf(
      shapePreviewPath(
        'rectangle',
        { x: 0, y: 0, width: 0, height: 0, angle_degrees: 0 },
        { fill: { gradient: defaultGradientSpec('linear') }, stroke: null },
      ),
    )

    expect(Number.isFinite(linear.start[0] + linear.start[1])).toBe(true)
    expect(linear.start).not.toEqual(linear.end)
  })

  it('bakes the frame angle into sweep angles as well as its centre', () => {
    // 1b's editor never creates a sweep (spec G5), but vello rotates a sweep's
    // angular field with the path, so the preview has to add the frame angle
    // here too or an imported sweep would drift out of alignment.
    const sweep = gradientOf(
      shapePreviewPath(
        'rectangle',
        { x: 0, y: 0, width: 200, height: 100, angle_degrees: 90 },
        {
          fill: {
            gradient: {
              kind: { sweep: { center: [0.5, 0.5], start_degrees: 0, end_degrees: 360 } },
              stops: [
                { offset: 0, color: [1, 2, 3, 255] },
                { offset: 1, color: [4, 5, 6, 255] },
              ],
              extend: 'pad',
            },
          },
          stroke: null,
        },
      ),
    ).kind
    if (!('Sweep' in sweep)) throw new Error('expected a sweep gradient')

    expect(sweep.Sweep.center[0]).toBeCloseTo(100, 6)
    expect(sweep.Sweep.center[1]).toBeCloseTo(50, 6)
    expect(sweep.Sweep.start_degrees).toBeCloseTo(90, 6)
    expect(sweep.Sweep.end_degrees).toBeCloseTo(450, 6)
  })
})

describe('isShapeTool', () => {
  it('accepts only the shape tool', () => {
    // 1a ships the four kinds with an authored geometry. `polygon`/`freeform`
    // are deferred with the vertex-editing surface, because a picker entry that
    // only draws a differently tagged rectangle would be misleading.
    expect(SHAPE_KINDS).toEqual(['rectangle', 'ellipse', 'line', 'arrow'])
    expect(isShapeTool('shape')).toBe(true)
    expect(isShapeTool('draw')).toBe(false)
  })
})
