'use client'

import { useTranslation } from 'react-i18next'

import { ComponentSourceField } from '@/components/preferences/PreferenceFields'
import type { MiGanConfig } from '@koharu/bridge/protocol'

/**
 * Checkpoint selection for MI-GAN. A prompt-free erase-only model, so the
 * source is the only setting.
 */
export function MiGanOptions({
  value,
  onChange,
}: {
  value: MiGanConfig
  onChange: (changes: Partial<MiGanConfig>) => void
}) {
  const { t } = useTranslation()
  return (
    <ComponentSourceField
      label={t('settings.pipeline.options.checkpointSource')}
      value={value.source ?? { kind: 'builtin' }}
      onChange={(source) => onChange({ source })}
    />
  )
}
