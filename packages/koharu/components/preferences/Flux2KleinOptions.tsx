'use client'

import { useTranslation } from 'react-i18next'

import { NumberField, TextField } from '@/components/preferences/PreferenceFields'
import type { ComponentSourceConfig, Flux2KleinConfig } from '@koharu/bridge/protocol'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@koharu/ui/components/select'

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

export function ComponentSourceField({
  label,
  value,
  onChange,
}: {
  label: string
  value: ComponentSourceConfig
  onChange: (value: ComponentSourceConfig) => void
}) {
  const { t } = useTranslation()
  const kinds = {
    builtin: t('settings.pipeline.options.sourceKind.builtin'),
    local_file: t('settings.pipeline.options.sourceKind.localFile'),
    hugging_face: t('settings.pipeline.options.sourceKind.huggingFace'),
    url: t('settings.pipeline.options.sourceKind.url'),
  }
  return (
    <div className='grid gap-1'>
      <label className='grid gap-1 text-[10px] text-muted-foreground'>
        {label}
        <Select
          value={value.kind}
          items={kinds}
          onValueChange={(kind) => {
            if (kind) onChange(emptySource(kind as ComponentSourceConfig['kind']))
          }}
        >
          <SelectTrigger aria-label={label} className='h-8 text-[11px] text-foreground'>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {Object.entries(kinds).map(([kind, name]) => (
              <SelectItem key={kind} value={kind}>
                {name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </label>
      {value.kind === 'local_file' && (
        <TextField
          label={t('settings.pipeline.options.sourcePath')}
          value={value.path}
          onChange={(path) => onChange({ ...value, path })}
        />
      )}
      {value.kind === 'hugging_face' && (
        <div className='grid grid-cols-3 gap-2'>
          <TextField
            label={t('settings.pipeline.options.sourceRepository')}
            value={value.repository}
            onChange={(repository) => onChange({ ...value, repository })}
          />
          <TextField
            label={t('settings.pipeline.options.sourceFilename')}
            value={value.filename}
            onChange={(filename) => onChange({ ...value, filename })}
          />
          <TextField
            label={t('settings.pipeline.options.sourceRevision')}
            value={value.revision ?? ''}
            onChange={(revision) => onChange({ ...value, revision: revision || null })}
          />
        </div>
      )}
      {value.kind === 'url' && (
        <div className='grid grid-cols-2 gap-2'>
          <TextField
            label={t('settings.pipeline.options.sourceUrl')}
            value={value.url}
            onChange={(url) => onChange({ ...value, url })}
          />
          <TextField
            label={t('settings.pipeline.options.sourceDigest')}
            value={value.digest}
            onChange={(digest) => onChange({ ...value, digest })}
          />
        </div>
      )}
    </div>
  )
}

export function emptySource(kind: ComponentSourceConfig['kind']): ComponentSourceConfig {
  switch (kind) {
    case 'builtin':
      return { kind }
    case 'local_file':
      return { kind, path: '' }
    case 'hugging_face':
      return { kind, repository: '', revision: null, filename: '' }
    case 'url':
      return { kind, url: '', digest: '' }
  }
}
