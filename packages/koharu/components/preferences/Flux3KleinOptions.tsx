'use client'

import { openUrl } from '@tauri-apps/plugin-opener'
import { useTranslation } from 'react-i18next'

import { ComponentSourceField, NumberField, TextField } from '@/components/preferences/PreferenceFields'
import type { Flux3KleinConfig } from '@koharu/bridge/protocol'

const DEFAULT_REQUEST = 'Remove all text and lettering, and reconstruct the background behind it.'
const DEFAULT_STEPS = 16
const DEFAULT_STRENGTH = 0.8
const DEFAULT_SEED = -1
const DEFAULT_MAX_PIXELS = 1024 * 1024

// Gate G4 (specs/2026-10-17-flux3-klein-inpainting-design.md): the inpainting
// stage peaks at 8,010 MiB and the first download is 12.36 GB. G4 did not
// record host RAM, so RAM_GB is still the spec's estimate. The locale strings
// interpolate these, so a new measurement changes only here.
const VRAM_GB = 8
const RAM_GB = 12
const DOWNLOAD_GB = 13

const LICENSE_URL =
  'https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md'

const COMPONENTS = [
  'transformer',
  'text_encoder',
  'vae',
  'upsampler',
  'upsampler_mmproj',
] as const

/**
 * Inference settings, checkpoint overrides and the license and resource
 * notices for FLUX 3 Image Klein. Unset fields fall back to the backend's
 * defaults.
 */
export function Flux3KleinOptions({
  value,
  onChange,
}: {
  value: Flux3KleinConfig
  onChange: (changes: Partial<Flux3KleinConfig>) => void
}) {
  const { t } = useTranslation()
  const source = value.source ?? {}
  return (
    <div className='grid gap-3'>
      <div className='grid gap-1 rounded-md border border-amber-500/40 bg-amber-500/10 p-2 text-[11px] text-amber-700 dark:text-amber-300'>
        <p>
          {t('settings.pipeline.options.flux3.license')}{' '}
          <a
            href={LICENSE_URL}
            className='underline'
            onClick={(event) => {
              event.preventDefault()
              void openUrl(LICENSE_URL).catch(() => undefined)
            }}
          >
            {t('settings.pipeline.options.flux3.licenseLink')}
          </a>
        </p>
        <p>
          {t('settings.pipeline.options.flux3.resources', {
            vram: VRAM_GB,
            ram: RAM_GB,
            download: DOWNLOAD_GB,
          })}
        </p>
        <p>{t('settings.pipeline.options.flux3.speed')}</p>
      </div>
      <div className='grid gap-1'>
        <TextField
          label={t('settings.pipeline.options.request')}
          value={value.request ?? DEFAULT_REQUEST}
          onChange={(request) => onChange({ request })}
        />
        <p className='text-[10px] text-muted-foreground'>
          {t('settings.pipeline.options.requestDescription')}
        </p>
      </div>
      <div className='grid grid-cols-3 gap-2'>
        <NumberField
          label={t('settings.pipeline.options.steps')}
          value={value.steps ?? DEFAULT_STEPS}
          min={1}
          step={1}
          onChange={(steps) => onChange({ steps: steps ?? DEFAULT_STEPS })}
        />
        <NumberField
          label={t('settings.pipeline.options.strength')}
          value={value.strength ?? DEFAULT_STRENGTH}
          min={0.05}
          max={1}
          step={0.05}
          onChange={(strength) => onChange({ strength: strength ?? DEFAULT_STRENGTH })}
        />
        <NumberField
          label={t('settings.pipeline.options.seed')}
          value={value.seed ?? DEFAULT_SEED}
          step={1}
          onChange={(seed) => onChange({ seed: seed ?? DEFAULT_SEED })}
        />
        <NumberField
          label={t('settings.pipeline.options.maxPixels')}
          value={value.max_pixels ?? DEFAULT_MAX_PIXELS}
          min={64 * 64}
          step={65536}
          onChange={(max_pixels) => onChange({ max_pixels: max_pixels ?? DEFAULT_MAX_PIXELS })}
        />
        <NumberField
          label={t('settings.pipeline.options.paddingMaskCrop')}
          value={value.padding_mask_crop ?? null}
          min={0}
          step={1}
          onChange={(padding_mask_crop) => onChange({ padding_mask_crop })}
        />
      </div>
      <div className='grid gap-2'>
        <p className='text-[10px] text-muted-foreground'>
          {t('settings.pipeline.options.sourceDescription')}
        </p>
        {COMPONENTS.map((component) => (
          <ComponentSourceField
            key={component}
            label={t(`settings.pipeline.options.source.${component}`)}
            value={source[component] ?? { kind: 'builtin' }}
            onChange={(replacement) =>
              onChange({ source: { ...source, [component]: replacement } })
            }
          />
        ))}
      </div>
    </div>
  )
}
