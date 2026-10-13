import { describe, expect, it } from 'vitest'

import {
  controlFrame,
  hitTestLayers,
  layerFrame,
  resizeFrame,
  rotateFrame,
  selectableLayer,
} from '@/lib/geometry'
import type { Frame, Layer } from '@koharu/bridge/protocol'

const frame: Frame = { x: 10, y: 20, width: 100, height: 50, angle_degrees: 0 }

const shapeLayer: Layer = {
  type: 'shape',
  id: 'shape',
  parent: 'page',
  geometry: {
    points: [
      { x: 10, y: 20 },
      { x: 110, y: 20 },
      { x: 110, y: 70 },
      { x: 10, y: 70 },
    ],
  },
  angle_degrees: 0,
  visibility: { visible: true, opacity: 1 },
  shape: { rectangle: { corner_radius: 0 } },
  style: { fill: { solid: [255, 255, 255, 255] }, stroke: null },
  rasterized: false,
}

describe('shape selection geometry', () => {
  it('selects and hit-tests a shape from its polygon', () => {
    expect(selectableLayer(shapeLayer)).toBe(true)
    expect(layerFrame(shapeLayer)).toEqual({ x: 10, y: 20, width: 100, height: 50, angle_degrees: 0 })
    expect(controlFrame(shapeLayer, {})).toEqual({
      x: 10,
      y: 20,
      width: 100,
      height: 50,
      angle_degrees: 0,
    })
    expect(hitTestLayers([shapeLayer], { x: 60, y: 45 }, {})).toBe(shapeLayer)
    expect(hitTestLayers([shapeLayer], { x: 300, y: 300 }, {})).toBeNull()
  })

  it('restores the authored angle of a rotated shape polygon', () => {
    const rotated: Layer = {
      ...shapeLayer,
      angle_degrees: 90,
      geometry: {
        points: [
          { x: 75, y: -25 },
          { x: 75, y: 75 },
          { x: 25, y: 75 },
          { x: 25, y: -25 },
        ],
      },
    }
    const restored = layerFrame(rotated)!
    expect(restored.angle_degrees).toBe(90)
    expect(restored.width).toBeCloseTo(100, 6)
    expect(restored.height).toBeCloseTo(50, 6)
    expect(restored.x).toBeCloseTo(0, 6)
    expect(restored.y).toBeCloseTo(0, 6)
  })

  it('selects a horizontal line, whose polygon has zero height', () => {
    const horizontal: Layer = {
      ...shapeLayer,
      shape: { line: { start: [10, 40], end: [110, 40], head: null } },
      geometry: {
        points: [
          { x: 10, y: 40 },
          { x: 60, y: 40 },
          { x: 110, y: 40 },
        ],
      },
    }
    expect(layerFrame(horizontal)).not.toBeNull()
    expect(hitTestLayers([horizontal], { x: 60, y: 40 }, {})).toBe(horizontal)
  })
})

describe('selection control geometry', () => {
  it('resizes from a handle while keeping the opposite edge fixed', () => {
    expect(resizeFrame(frame, 'e', { x: 140, y: 45 }, 1)).toEqual({
      ...frame,
      width: 130,
    })
    expect(resizeFrame(frame, 'nw', { x: 0, y: 10 }, 1)).toEqual({
      ...frame,
      x: 0,
      y: 10,
      width: 110,
      height: 60,
    })
  })

  it('resizes in the frame local axes when it is rotated', () => {
    const result = resizeFrame({ ...frame, angle_degrees: 90 }, 'e', { x: 60, y: 125 }, 1)
    expect(result.x).toBeCloseTo(-5)
    expect(result.y).toBeCloseTo(35)
    expect(result.width).toBeCloseTo(130)
    expect(result.height).toBe(50)
  })

  it('rotates around the frame center', () => {
    expect(rotateFrame(frame, { x: 60, y: -5 }, { x: 110, y: 45 })).toEqual({
      ...frame,
      angle_degrees: 90,
    })
  })
})
