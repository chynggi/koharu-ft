import { describe, expect, it } from 'vitest'

import {
  SHAPE_KINDS,
  defaultShapeStyle,
  isShapeTool,
  shapeGeometry,
  shapePolygon,
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
    expect(shapeGeometry('polygon', points)).toBe('polygon')
    expect(shapeGeometry('freeform', points)).toBe('freeform')
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

describe('isShapeTool', () => {
  it('accepts only the shape tool', () => {
    expect(SHAPE_KINDS).toEqual([
      'rectangle',
      'ellipse',
      'line',
      'arrow',
      'polygon',
      'freeform',
    ])
    expect(isShapeTool('shape')).toBe(true)
    expect(isShapeTool('draw')).toBe(false)
  })
})
