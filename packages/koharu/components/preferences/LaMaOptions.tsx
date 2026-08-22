'use client'

import { useTranslation } from 'react-i18next'

import { ComponentSourceField } from '@/components/preferences/Flux2KleinOptions'
import type { LaMaConfig, WeightsFormatConfig } from '@koharu/bridge/protocol'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@koharu/ui/components/select'

/**
 * Checkpoint and weight format selection for LaMa. Every field is optional in
 * the configuration, so an unset field falls back to the same default the
 * backend uses.
 */
export function LaMaOptions({
  value,
  onChange,
}: {
  value: LaMaConfig
  onChange: (changes: Partial<LaMaConfig>) => void
}) {
  const { t } = useTranslation()
  const formats = {
    safe_tensors: t('settings.pipeline.options.formatKind.safeTensors'),
    torch_script: t('settings.pipeline.options.formatKind.torchScript'),
  }
  return (
    <div className='grid gap-3'>
      <label className='grid gap-1 text-[10px] text-muted-foreground'>
        {t('settings.pipeline.options.format')}
        <Select
          value={value.format ?? 'safe_tensors'}
          items={formats}
          onValueChange={(format) => {
            if (format) onChange({ format: format as WeightsFormatConfig })
          }}
        >
          <SelectTrigger
            aria-label={t('settings.pipeline.options.format')}
            className='h-8 text-[11px] text-foreground'
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {Object.entries(formats).map(([format, name]) => (
              <SelectItem key={format} value={format}>
                {name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </label>
      <ComponentSourceField
        label={t('settings.pipeline.options.checkpointSource')}
        value={value.source ?? { kind: 'builtin' }}
        onChange={(source) => onChange({ source })}
      />
    </div>
  )
}
