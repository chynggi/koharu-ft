import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { Flux3KleinOptions } from '@/components/preferences/Flux3KleinOptions'
import { defaultModel, modelOptions } from '@/components/preferences/models'

vi.mock('@tauri-apps/plugin-opener', () => ({ openUrl: vi.fn(async () => undefined) }))

describe('FLUX 3 Image Klein preferences', () => {
  it('is offered after FLUX.2 Klein with the backend default request', () => {
    const inpainting = modelOptions.inpainting as readonly string[]
    expect(inpainting.indexOf('flux3-klein')).toBe(inpainting.indexOf('flux2-klein') + 1)
    // Mirrors `koharu_ml::flux3_klein::DEFAULT_REQUEST`.
    expect(defaultModel('flux3-klein')).toEqual({
      model: 'flux3-klein',
      request: 'Remove all text and lettering, and reconstruct the background behind it.',
    })
  })

  it('edits the upsampler request', () => {
    const onChange = vi.fn()
    render(<Flux3KleinOptions value={{}} onChange={onChange} />)
    fireEvent.change(screen.getAllByRole('textbox')[0]!, {
      target: { value: 'Erase the lettering.' },
    })
    expect(onChange).toHaveBeenCalledWith({ request: 'Erase the lettering.' })
  })
})
