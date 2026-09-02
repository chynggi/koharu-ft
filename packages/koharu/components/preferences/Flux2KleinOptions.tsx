'use client'

import { useTranslation } from 'react-i18next'

import { ComponentSourceField, NumberField, TextField } from '@/components/preferences/PreferenceFields'
import type { Flux2KleinConfig } from '@koharu/bridge/protocol'

const DEFAULT_PROMPT = 'Remove the text and reconstruct the background.'
const DEFAULT_STEPS = 4
const DEFAULT_STRENGTH = 0.8
const DEFAULT_SEED = -1
const DEFAULT_MAX_PIXELS = 1024 * 1024

const COMPONENTS = ['transformer', 'text_encoder', 'vae'] as const

/**
 * Inference settings and checkpoint overrides for FLUX.2 Klein. Every field is
 * optional in the configuration, so an unset field falls back to the same
 * default the backend uses.
 */
export function Flux2KleinOptions({
  value,
  onChange,
}: {
  value: Flux2KleinConfig
  onChange: (changes: Partial<Flux2KleinConfig>) => void
}) {
  const { t } = useTranslation()
  const source = value.source ?? {}
  return (
    <div className='grid gap-3'>
      <TextField
        label={t('settings.pipeline.options.prompt')}
        value={value.prompt ?? DEFAULT_PROMPT}
        onChange={(prompt) => onChange({ prompt })}
      />
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

