import { describe, expect, it } from 'vitest'

import {
  SHAPE_KINDS,
  defaultShapeStyle,
  isShapeTool,
  shapeGeometry,
  shapePolygon,
  shapePreviewPath,
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
