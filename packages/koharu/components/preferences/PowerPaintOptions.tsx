'use client'

import { useTranslation } from 'react-i18next'

import { NumberField, TextField } from '@/components/preferences/PreferenceFields'
import type { PowerPaintConfig } from '@koharu/bridge/protocol'

/**
 * PowerPaint reads files the user converts themselves, so it asks for paths
 * rather than a checkpoint source. The task prompt is fixed to context-aware
 * filling, which is what erasing lettering needs, so it is not exposed.
 */
export function PowerPaintOptions({
  value,
  onChange,
}: {
  value: PowerPaintConfig
  onChange: (changes: Partial<PowerPaintConfig>) => void
}) {
  const { t } = useTranslation()
  return (
    <div className='grid gap-3'>
      <TextField
        label={t('settings.pipeline.options.powerpaintModel')}
        value={value.model_path ?? ''}
        onChange={(model_path) => onChange({ model_path })}
      />
      <TextField
        label={t('settings.pipeline.options.powerpaintEmbeddings')}
        value={value.embeddings_dir ?? ''}
        onChange={(embeddings_dir) => onChange({ embeddings_dir })}
      />
      {/* Zero reads as "unset" on the Rust side, so clearing the field
          restores the default step count rather than disabling sampling. */}
      <NumberField
        label={t('settings.pipeline.options.steps')}
        value={value.steps ? value.steps : null}
        min={1}
        onChange={(steps) => onChange({ steps: steps ?? 0 })}
      />
    </div>
  )
}
