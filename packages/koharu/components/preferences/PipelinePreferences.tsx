'use client'

import { useMutation, useQuery } from '@tanstack/react-query'
import { ChevronDown, Eraser } from 'lucide-react'
import { type ReactNode, useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { Choice } from '@/components/controls/Choice'
import { ModelPicker } from '@/components/controls/ModelPicker'
import {
  NumberField,
  PreferencePage,
  PreferenceRow,
  PreferenceSection,
} from '@/components/preferences/PreferenceFields'
import { detectors, inpainters, modelName, REASONING, recognizers, vendorOf } from '@/lib/models'
import { languageNames } from '@/lib/translation'
import { vendorName } from '@/lib/vendors'
import {
  commands,
  type Detector,
  type Inpainter,
  type Model,
  type PipelineConfig,
  type Recognizer,
  type TranslatorConfig,
} from '@koharu/ipc/protocol'
import { Badge } from '@koharu/ui/components/badge'
import { Button } from '@koharu/ui/components/button'
import { Input } from '@koharu/ui/components/input'
import { Popover, PopoverContent, PopoverTrigger } from '@koharu/ui/components/popover'
import { Switch } from '@koharu/ui/components/switch'
import { Textarea } from '@koharu/ui/components/textarea'

type StageName = 'detect' | 'recognize' | 'inpaint' | 'translate'

/** The `[pipeline]` stages in the order they run, a section each with its model first: the
 * detector with its thresholds, the recognizer, the inpainter with its prompt, and the translator
 * with its options. */
export function PipelinePreferences({
  value,
  models,
  onChange,
}: {
  value: PipelineConfig
  models: Model[]
  onChange: (value: PipelineConfig) => void
}) {
  const { t } = useTranslation()
  const detector = value.detector ?? {}
  const inpainter = value.inpainter ?? {}
  const translator = value.translator ?? {}
  const thresholds = ['text', 'onomatopoeia', 'bubble', 'panel'] as const

  return (
    <PreferencePage
      title={t('settings.pipeline.title')}
      description={t('settings.pipeline.description')}
    >
      <Stage
        stage='detect'
        model={
          <ModelSelect
            stage='detect'
            value={detector.model ?? 'koharu-layout-highres'}
            names={detectors}
            onChange={(model) => onChange({ ...value, detector: { ...detector, model } })}
          />
        }
      >
        <PreferenceRow title={t('settings.pipeline.thresholds')} align='start'>
          <div className='grid grid-cols-2 gap-2'>
            {thresholds.map((threshold) => (
              <NumberField
                key={threshold}
                label={t(`settings.pipeline.options.${threshold}Threshold`)}
                value={detector[threshold] ?? null}
                min={0}
                max={1}
                step={0.05}
                placeholder={t('model.default')}
                onChange={(next) =>
                  onChange({ ...value, detector: { ...detector, [threshold]: next } })
                }
              />
            ))}
          </div>
        </PreferenceRow>
      </Stage>
      <Stage
        stage='recognize'
        model={
          <ModelSelect
            stage='recognize'
            value={value.recognizer?.model ?? 'hayai-ocr'}
            names={recognizers}
            onChange={(model: Recognizer) => onChange({ ...value, recognizer: { model } })}
          />
        }
      />
      <Stage
        stage='inpaint'
        model={
          <ModelSelect
            stage='inpaint'
            value={inpainter.model ?? 'lama'}
            names={inpainters}
            onChange={(model: Inpainter) =>
              onChange({ ...value, inpainter: { ...inpainter, model } })
            }
          />
        }
      >
        <PreferenceRow
          title={t('settings.pipeline.options.prompt')}
          description={t('settings.pipeline.options.promptDescription')}
          align='start'
        >
          <Textarea
            aria-label={t('settings.pipeline.options.prompt')}
            value={inpainter.prompt ?? ''}
            className='min-h-28 resize-none text-[12px] leading-5'
            onChange={(event) =>
              onChange({
                ...value,
                inpainter: { ...inpainter, prompt: event.currentTarget.value },
              })
            }
          />
        </PreferenceRow>
      </Stage>
      <Stage
        stage='translate'
        model={
          <TranslatorModel
            value={translator.model ?? ''}
            models={models}
            label={t('settings.pipeline.modelLabel', {
              stage: t('settings.pipeline.stages.translate.title'),
            })}
            onChange={(model) => onChange({ ...value, translator: { ...translator, model } })}
          />
        }
      >
        <TranslatorOptions
          value={translator}
          onChange={(next) => onChange({ ...value, translator: next })}
        />
      </Stage>
      <RemoteServer
        url={value.remote?.url ?? ''}
        onChange={(url) => onChange({ ...value, remote: url ? { url } : null })}
      />
    </PreferencePage>
  )
}

/** The processing server runs go to, the stages' models above running there: its address, kept
 * with the settings, and its token, kept as an API key is. */
function RemoteServer({ url, onChange }: { url: string; onChange: (url: string) => void }) {
  const { t } = useTranslation()
  const [address, setAddress] = useState(url)
  const [token, setToken] = useState('')
  const stored = useQuery({
    queryKey: ['server-token'],
    queryFn: () => commands.hasServerToken(),
  })
  const storeToken = useMutation({
    scope: { id: 'server-token' },
    mutationFn: (next: string | null) => commands.setServerToken(next),
    onSuccess: async () => {
      setToken('')
      await stored.refetch()
    },
  })
  useEffect(() => setAddress(url), [url])

  return (
    <PreferenceSection
      title={t('settings.pipeline.remote.title')}
      description={t('settings.pipeline.remote.description')}
    >
      <PreferenceRow title={t('settings.pipeline.remote.url')}>
        <Input
          aria-label={t('settings.pipeline.remote.url')}
          type='url'
          spellCheck={false}
          placeholder='http://192.168.0.10:47823'
          className='h-8 w-72 text-[12px]'
          value={address}
          onChange={(event) => setAddress(event.currentTarget.value)}
          onBlur={() => address.trim() !== url && onChange(address.trim())}
          onKeyDown={(event) => {
            if (event.key === 'Enter') event.currentTarget.blur()
          }}
        />
      </PreferenceRow>
      <PreferenceRow title={t('settings.pipeline.remote.token')}>
        <div className='flex w-72 gap-2'>
          <Input
            aria-label={t('settings.pipeline.remote.token')}
            type='text'
            autoComplete='off'
            spellCheck={false}
            placeholder={
              stored.data
                ? t('settings.providers.configured')
                : t('settings.providers.notConfigured')
            }
            className='h-8 min-w-0 flex-1 text-[12px] [-webkit-text-security:disc] [&::placeholder]:[-webkit-text-security:none]'
            value={token}
            onChange={(event) => setToken(event.currentTarget.value)}
            onBlur={() => token && storeToken.mutate(token)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') event.currentTarget.blur()
            }}
          />
          {stored.data && (
            <Button
              type='button'
              variant='outline'
              size='icon'
              aria-label={t('settings.pipeline.remote.clearToken')}
              onClick={() => storeToken.mutate(null)}
            >
              <Eraser />
            </Button>
          )}
        </div>
      </PreferenceRow>
    </PreferenceSection>
  )
}

/** A stage's section: what it does, its model, then its options. */
function Stage({
  stage,
  model,
  children,
}: {
  stage: StageName
  model: ReactNode
  children?: ReactNode
}) {
  const { t } = useTranslation()
  return (
    <PreferenceSection
      title={t(`settings.pipeline.stages.${stage}.title`)}
      description={t(`settings.pipeline.stages.${stage}.description`)}
    >
      <PreferenceRow title={t('settings.pipeline.model')}>{model}</PreferenceRow>
      {children}
    </PreferenceSection>
  )
}

function ModelSelect<Model extends Detector | Recognizer | Inpainter>({
  stage,
  value,
  names,
  onChange,
}: {
  stage: Exclude<StageName, 'translate'>
  value: Model
  names: Record<Model, string>
  onChange: (model: Model) => void
}) {
  const { t } = useTranslation()
  return (
    <Choice
      value={value}
      choices={names}
      label={t('settings.pipeline.modelLabel', {
        stage: t(`settings.pipeline.stages.${stage}.title`),
      })}
      onChange={onChange}
    />
  )
}

/** The translator's model, any provider's, labelled with where it runs. */
function TranslatorModel({
  value,
  models,
  label,
  onChange,
}: {
  value: string
  models: Model[]
  label: string
  onChange: (model: string) => void
}) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger
        type='button'
        aria-label={label}
        className='flex h-8 w-full min-w-0 items-center justify-between gap-2 rounded-lg border border-input bg-transparent px-2.5 text-[11px] transition-colors outline-none hover:bg-foreground/[0.03] focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50'
      >
        <span className='flex min-w-0 flex-1 items-center gap-2 text-left'>
          {value && (
            <Badge variant='outline' className='shrink-0 px-1.5 py-0 text-[9px] font-medium'>
              {vendorName(vendorOf(value), t)}
            </Badge>
          )}
          <span className='truncate'>{modelName(models, value) || t('inference.noModel')}</span>
        </span>
        <ChevronDown className='size-3.5 shrink-0 text-muted-foreground' />
      </PopoverTrigger>
      <PopoverContent
        align='start'
        sideOffset={4}
        className='w-(--anchor-width) min-w-64 gap-0 overflow-hidden p-1'
      >
        <ModelPicker
          value={value}
          models={models}
          onSelect={(reference) => {
            onChange(reference)
            setOpen(false)
          }}
        />
      </PopoverContent>
    </Popover>
  )
}

/** The translator's options under `[pipeline.translator]`, as rows of its section: whether the
 * model sees each page, how much it reasons, the target language, and the user's instructions,
 * which Koharu adds to its own fixed system prompt. */
function TranslatorOptions({
  value,
  onChange,
}: {
  value: TranslatorConfig
  onChange: (value: TranslatorConfig) => void
}) {
  const { t, i18n } = useTranslation()
  const languageChoices = useMemo(() => languageNames(i18n.language), [i18n.language])
  const reasoning = value.reasoning ?? 'default'

  return (
    <>
      <PreferenceRow
        title={t('settings.pipeline.translator.vision')}
        description={t('settings.pipeline.translator.visionDescription')}
      >
        <Switch
          aria-label={t('settings.pipeline.translator.vision')}
          checked={value.vision ?? false}
          onCheckedChange={(vision) => onChange({ ...value, vision })}
        />
      </PreferenceRow>
      <PreferenceRow
        title={t('settings.pipeline.translator.reasoning')}
        description={t('settings.pipeline.translator.reasoningDescription')}
      >
        <Choice
          value={reasoning}
          choices={Object.fromEntries([
            ['default', t('inference.providerDefault')],
            ...REASONING.map((level) => [level, t(`reasoning.${level}`)]),
          ])}
          label={t('settings.pipeline.translator.reasoning')}
          onChange={(level) =>
            onChange({ ...value, reasoning: level === 'default' ? null : level })
          }
        />
      </PreferenceRow>
      <PreferenceRow
        title={t('model.targetLanguage')}
        description={t('settings.pipeline.translator.targetLanguageDescription')}
      >
        <Choice
          value={value.target ?? 'en'}
          choices={languageChoices}
          label={t('model.targetLanguage')}
          onChange={(target) => onChange({ ...value, target })}
        />
      </PreferenceRow>
      <PreferenceRow
        title={t('settings.pipeline.translator.instructions')}
        description={t('settings.pipeline.translator.instructionsDescription')}
        align='start'
      >
        <Textarea
          aria-label={t('settings.pipeline.translator.instructions')}
          value={value.instructions ?? ''}
          className='min-h-28 resize-none text-[12px] leading-5'
          onChange={(event) => onChange({ ...value, instructions: event.currentTarget.value })}
        />
      </PreferenceRow>
    </>
  )
}
