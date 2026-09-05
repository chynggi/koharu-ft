'use client'

import { FolderOpen, Plus, Trash2 } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { LlmRuntimePreferences } from '@/components/preferences/LlmRuntimePreferences'
import {
  PreferencePage,
  PreferenceRow,
  PreferenceSection,
} from '@/components/preferences/PreferenceFields'
import { call } from '@/lib/backend'
import {
  commands,
  type CustomModel,
  type LlmCapabilities,
  type LocalConfig,
  type ProviderPreference,
  type ProviderPreferences as ProviderSettings,
} from '@koharu/bridge/protocol'
import { Button } from '@koharu/ui/components/button'
import { Input } from '@koharu/ui/components/input'

/**
 * Registered GGUF files and the llama.cpp runtime settings. Both live in the
 * local provider's configuration, so they travel through the existing
 * preferences save path.
 */
export function LocalModelPreferences({
  value,
  onChange,
}: {
  value: ProviderSettings
  onChange: (value: ProviderSettings) => void
}) {
  const { t } = useTranslation()
  const [capabilities, setCapabilities] = useState<LlmCapabilities | null>(null)
  // Set to a string once the native picker declines (a remote browser), which
  // switches the row over to typing a path on the machine running Koharu.
  const [manualPath, setManualPath] = useState<string | null>(null)

  useEffect(() => {
    let active = true
    void call(commands.getLlmCapabilities)
      .then((result) => {
        if (active) setCapabilities(result)
      })
      .catch(() => undefined)
    return () => {
      active = false
    }
  }, [])

  const entry = value.entries.find((candidate) => candidate.config.provider === 'local')
  if (!entry || entry.config.provider !== 'local') return null
  const config = entry.config.settings
  const models = config.models ?? []

  const update = (changes: Partial<LocalConfig>) =>
    onChange(replaceLocal(value, entry, { ...config, ...changes }))

  const register = (path: string) =>
    update({ models: [...models, draftModel(path, models)] })

  const commitManualPath = () => {
    const path = manualPath?.trim()
    if (!path) return
    register(path)
    setManualPath(null)
  }

  return (
    <PreferencePage title={t('settings.models.title')} description={t('settings.models.description')}>
      <PreferenceSection
        title={t('settings.models.registered')}
        description={t('settings.models.registeredDescription')}
      >
        {models.length === 0 && (
          <PreferenceRow title={t('settings.models.none')}>
            <span className='text-[11px] text-muted-foreground'>
              {t('settings.models.noneDescription')}
            </span>
          </PreferenceRow>
        )}
        {models.map((model, index) => (
          <CustomModelRow
            key={model.id}
            model={model}
            onChange={(replacement) =>
              update({ models: models.map((entry, at) => (at === index ? replacement : entry)) })
            }
            onRemove={() => update({ models: models.filter((_, at) => at !== index) })}
          />
        ))}
        <PreferenceRow
          title={t('settings.models.add')}
          description={t('settings.models.addDescription')}
          align='start'
        >
          <div className='grid justify-items-end gap-2'>
            <Button
              type='button'
              variant='outline'
              size='sm'
              className='h-8 gap-1.5 text-[11px]'
              onClick={() => {
                void call(commands.pickGgufFile)
                  .then((path) => {
                    if (path) register(path)
                    else setManualPath('')
                  })
                  .catch(() => undefined)
              }}
            >
              <Plus className='size-3.5' /> {t('settings.models.addLocal')}
            </Button>
            {manualPath !== null && (
              <div className='flex w-full gap-2'>
                <Input
                  aria-label={t('settings.models.path')}
                  value={manualPath}
                  placeholder={t('settings.models.pathPlaceholder')}
                  className='h-8 min-w-0 flex-1 text-[12px]'
                  onChange={(event) => setManualPath(event.currentTarget.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') commitManualPath()
                  }}
                />
                <Button
                  type='button'
                  variant='outline'
                  size='icon'
                  aria-label={t('settings.models.addLocal')}
                  onClick={commitManualPath}
                >
                  <Plus />
                </Button>
              </div>
            )}
          </div>
        </PreferenceRow>
      </PreferenceSection>

      <LlmRuntimePreferences
        value={config.runtime ?? {}}
        capabilities={capabilities}
        onChange={(runtime) => update({ runtime })}
      />
    </PreferencePage>
  )
}

function CustomModelRow({
  model,
  onChange,
  onRemove,
}: {
  model: CustomModel
  onChange: (model: CustomModel) => void
  onRemove: () => void
}) {
  const { t } = useTranslation()
  return (
    <PreferenceRow title={model.name || model.id} description={model.path} align='start'>
      <div className='grid gap-2'>
        <Input
          aria-label={t('settings.models.name')}
          value={model.name}
          placeholder={t('settings.models.name')}
          className='h-8 text-[12px]'
          onChange={(event) => onChange({ ...model, name: event.currentTarget.value })}
        />
        <div className='flex gap-2'>
          <Input
            aria-label={t('settings.models.projector')}
            value={model.projector ?? ''}
            placeholder={t('settings.models.projectorPlaceholder')}
            className='h-8 min-w-0 flex-1 text-[12px]'
            onChange={(event) =>
              onChange({ ...model, projector: event.currentTarget.value || null })
            }
          />
          <Button
            type='button'
            variant='outline'
            size='icon'
            aria-label={t('settings.models.browseProjector')}
            onClick={() => {
              void call(commands.pickGgufFile)
                .then((projector) => {
                  if (projector) onChange({ ...model, projector })
                })
                .catch(() => undefined)
            }}
          >
            <FolderOpen />
          </Button>
          <Button
            type='button'
            variant='destructive'
            size='icon'
            aria-label={t('settings.models.remove', { model: model.name || model.id })}
            onClick={onRemove}
          >
            <Trash2 />
          </Button>
        </div>
      </div>
    </PreferenceRow>
  )
}

/**
 * Derives an id and display name from the chosen file. The id is what the
 * pipeline stores, so it must stay unique and free of separators the backend
 * rejects.
 */
function draftModel(path: string, existing: CustomModel[]): CustomModel {
  const file = path.split(/[\\/]/).pop() ?? 'model.gguf'
  const name = file.replace(/\.gguf$/i, '')
  const slug = name
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, '-')
    .replace(/^-+|-+$/g, '')
  const base = slug ? `custom-${slug}` : 'custom-model'
  let id = base
  for (let suffix = 2; existing.some((model) => model.id === id); suffix += 1) {
    id = `${base}-${suffix}`
  }
  return { id, name, path, projector: null }
}

function replaceLocal(
  preferences: ProviderSettings,
  entry: ProviderPreference,
  settings: LocalConfig,
): ProviderSettings {
  return {
    entries: preferences.entries.map((candidate) =>
      candidate.config.provider === 'local'
        ? { ...entry, config: { provider: 'local', settings } }
        : candidate,
    ),
  }
}
