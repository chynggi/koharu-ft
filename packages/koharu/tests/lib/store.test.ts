import { describe, expect, it } from 'vitest'

import { receiveDownload, useKoharuStore } from '@/lib/store'

describe('editor state', () => {
  it('retains failed downloads until they are dismissed', () => {
    useKoharuStore.setState({ downloads: {} })
    receiveDownload({
      state: 'failed',
      id: 9,
      name: 'weights.bin',
      completed: 0,
      total: 0,
      error: 'network unavailable',
    })
    expect(useKoharuStore.getState().downloads[9]).toMatchObject({
      state: 'failed',
      name: 'weights.bin',
    })

    useKoharuStore.getState().dismissDownload(9)
    expect(useKoharuStore.getState().downloads[9]).toBeUndefined()
  })

  it('defaults paint brush color to pure white (#FFFFFF)', () => {
    expect(useKoharuStore.getState().brush.color).toBe('#FFFFFF')
  })

  it('reseeds the shape style so a new kind can actually draw itself', () => {
    expect(useKoharuStore.getState().shapeKind).toBe('rectangle')
    expect(useKoharuStore.getState().shapeStyle).toEqual({
      fill: { solid: [255, 255, 255, 255] },
      stroke: null,
    })

    useKoharuStore.getState().setShapeKind('line')

    expect(useKoharuStore.getState().shapeKind).toBe('line')
    expect(useKoharuStore.getState().shapeStyle.stroke).not.toBeNull()
    expect(useKoharuStore.getState().shapeStyle.fill).toBeNull()
  })
})
