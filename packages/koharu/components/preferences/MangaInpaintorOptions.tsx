'use client'

import { useTranslation } from 'react-i18next'

import { ComponentSourceField } from '@/components/preferences/PreferenceFields'
import type { MangaInpaintorConfig } from '@koharu/bridge/protocol'

/**
 * Checkpoint selection for the Manga inpainter. The pipeline is assembled
 * from an inpaintor and a line model, so both sources are configurable.
 */
export function MangaInpaintorOptions({
  value,
  onChange,
}: {
  value: MangaInpaintorConfig
  onChange: (changes: Partial<MangaInpaintorConfig>) => void
}) {
  const { t } = useTranslation()
  return (
    <div className='grid gap-3'>
      <ComponentSourceField
        label={t('settings.pipeline.options.source.inpaintor')}
        value={value.inpaintor ?? { kind: 'builtin' }}
        onChange={(inpaintor) => onChange({ inpaintor })}
      />
      <ComponentSourceField
        label={t('settings.pipeline.options.source.line')}
        value={value.line ?? { kind: 'builtin' }}
        onChange={(line) => onChange({ line })}
      />
    </div>
  )
}
