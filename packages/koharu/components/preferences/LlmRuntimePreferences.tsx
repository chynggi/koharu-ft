'use client'

import { Info } from 'lucide-react'
import { useTranslation } from 'react-i18next'

import {
  NumberField,
  PreferenceRow,
  PreferenceSection,
} from '@/components/preferences/PreferenceFields'
import type {
  ContextMode,
  FlashAttentionMode,
  GpuLayers,
  KvCacheChoice,
  LlmCapabilities,
  LlmRuntimeConfig,
} from '@koharu/bridge/protocol'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@koharu/ui/components/select'

const contextKinds = ['dynamic', 'fixed', 'bounded'] as const
const gpuLayerKinds = ['all', 'custom'] as const
const flashAttentionModes: FlashAttentionMode[] = ['auto', 'on', 'off']
const kvCacheChoices: KvCacheChoice[] = [
  'f16',
  'bf16',
  'q8_0',
  'q5_1',
  'q5_0',
  'q4_1',
  'q4_0',
  'iq4_nl',
  'f32',
]

const DEFAULT_VALUE = '__default__'

/**
 * The llama.cpp runtime controls. Every field is optional: leaving one unset
 * keeps Koharu's own behaviour rather than pinning a value, so an untouched
 * panel changes nothing.
 */
export function LlmRuntimePreferences({
  value,
  capabilities,
  onChange,
}: {
  value: LlmRuntimeConfig
  capabilities: LlmCapabilities | null
  onChange: (value: LlmRuntimeConfig) => void
}) {
  const { t } = useTranslation()
  const update = (changes: Partial<LlmRuntimeConfig>) => onChange({ ...value, ...changes })
  const context = value.context ?? { kind: 'dynamic' }
  const gpuLayers = value.gpu_layers ?? { kind: 'all' }
  const deferred = new Map(capabilities?.deferred.map((entry) => [entry.setting, entry.reason]))

  return (
    <>
      <PreferenceSection
        title={t('settings.llm.context')}
        description={t('settings.llm.contextDescription')}
      >
        <PreferenceRow
          title={t('settings.llm.contextMode')}
          description={t('settings.llm.contextModeDescription')}
          align='start'
        >
          <div className='grid gap-2'>
            <Select
              value={context.kind}
              onValueChange={(kind) => kind && update({ context: emptyContext(kind as ContextKind) })}
            >
              <SelectTrigger
                aria-label={t('settings.llm.contextMode')}
                className='h-8 w-full text-[11px]'
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {contextKinds.map((kind) => (
                  <SelectItem key={kind} value={kind}>
                    {t(`settings.llm.contextModes.${kind}`)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {context.kind === 'fixed' && (
              <NumberField
                label={t('settings.llm.contextSize')}
                value={context.size}
                min={1}
                step={512}
                onChange={(size) => update({ context: { kind: 'fixed', size: size ?? 4096 } })}
              />
            )}
            {context.kind === 'bounded' && (
              <div className='grid grid-cols-2 gap-2'>
                <NumberField
                  label={t('settings.llm.minimumContext')}
                  value={context.minimum ?? null}
                  min={1}
                  step={512}
                  onChange={(minimum) => update({ context: { ...context, minimum } })}
                />
                <NumberField
                  label={t('settings.llm.maximumContext')}
                  value={context.maximum ?? null}
                  min={1}
                  step={512}
                  onChange={(maximum) => update({ context: { ...context, maximum } })}
                />
              </div>
            )}
          </div>
        </PreferenceRow>
        <PreferenceRow
          title={t('settings.llm.maxOutputTokens')}
          description={t('settings.llm.maxOutputTokensDescription')}
        >
          <NumberField
            label={t('settings.llm.tokens')}
            value={value.max_output_tokens ?? null}
            min={1}
            step={128}
            onChange={(max_output_tokens) => update({ max_output_tokens })}
          />
        </PreferenceRow>
      </PreferenceSection>

      <PreferenceSection
        title={t('settings.llm.batching')}
        description={t('settings.llm.batchingDescription')}
      >
        <PreferenceRow title={t('settings.llm.batchSizes')} align='start'>
          <div className='grid grid-cols-2 gap-2'>
            <NumberField
              label='n_batch'
              value={value.n_batch ?? null}
              min={1}
              step={128}
              onChange={(n_batch) => update({ n_batch })}
            />
            <NumberField
              label='n_ubatch'
              value={value.n_ubatch ?? null}
              min={1}
              step={128}
              onChange={(n_ubatch) => update({ n_ubatch })}
            />
          </div>
        </PreferenceRow>
        <PreferenceRow title={t('settings.llm.threads')} align='start'>
          <div className='grid grid-cols-2 gap-2'>
            <NumberField
              label={t('settings.llm.generationThreads')}
              value={value.n_threads ?? null}
              min={1}
              step={1}
              onChange={(n_threads) => update({ n_threads })}
            />
            <NumberField
              label={t('settings.llm.batchThreads')}
              value={value.n_threads_batch ?? null}
              min={1}
              step={1}
              onChange={(n_threads_batch) => update({ n_threads_batch })}
            />
          </div>
        </PreferenceRow>
      </PreferenceSection>

      <PreferenceSection
        title={t('settings.llm.accelerator')}
        description={t('settings.llm.acceleratorDescription')}
      >
        <PreferenceRow
          title={t('settings.llm.gpuLayers')}
          description={deferred.get('gpu_layers')}
          align='start'
        >
          <div className='grid gap-2'>
            <Select
              value={gpuLayers.kind}
              onValueChange={(kind) =>
                kind &&
                update({
                  gpu_layers:
                    kind === 'all' ? { kind: 'all' } : { kind: 'custom', layers: layersOf(gpuLayers) },
                })
              }
            >
              <SelectTrigger
                aria-label={t('settings.llm.gpuLayers')}
                className='h-8 w-full text-[11px]'
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {gpuLayerKinds.map((kind) => (
                  <SelectItem key={kind} value={kind}>
                    {t(`settings.llm.gpuLayerModes.${kind}`)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {gpuLayers.kind === 'custom' && (
              <NumberField
                label={t('settings.llm.layers')}
                value={gpuLayers.layers}
                min={0}
                step={1}
                onChange={(layers) => update({ gpu_layers: { kind: 'custom', layers: layers ?? 0 } })}
              />
            )}
          </div>
        </PreferenceRow>
        <PreferenceRow
          title={t('settings.llm.flashAttention')}
          description={deferred.get('flash_attention')}
        >
          <Select
            value={value.flash_attention ?? 'auto'}
            onValueChange={(mode) =>
              mode && update({ flash_attention: mode as FlashAttentionMode })
            }
          >
            <SelectTrigger
              aria-label={t('settings.llm.flashAttention')}
              className='h-8 w-full text-[11px]'
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {flashAttentionModes.map((mode) => (
                <SelectItem key={mode} value={mode}>
                  {t(`settings.llm.flashAttentionModes.${mode}`)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </PreferenceRow>
        <PreferenceRow
          title={t('settings.llm.kvCache')}
          description={deferred.get('kv_cache_type')}
          align='start'
        >
          <div className='grid grid-cols-2 gap-2'>
            <KvCacheField
              label={t('settings.llm.kvCacheKey')}
              value={value.kv_cache_type_k ?? null}
              onChange={(kv_cache_type_k) => update({ kv_cache_type_k })}
            />
            <KvCacheField
              label={t('settings.llm.kvCacheValue')}
              value={value.kv_cache_type_v ?? null}
              onChange={(kv_cache_type_v) => update({ kv_cache_type_v })}
            />
          </div>
        </PreferenceRow>
      </PreferenceSection>

      {capabilities && <CapabilityNotice capabilities={capabilities} />}
    </>
  )
}

function KvCacheField({
  label,
  value,
  onChange,
}: {
  label: string
  value: KvCacheChoice | null
  onChange: (value: KvCacheChoice | null) => void
}) {
  const { t } = useTranslation()
  return (
    <label className='grid gap-1 text-[10px] text-muted-foreground'>
      {label}
      <Select
        value={value ?? DEFAULT_VALUE}
        onValueChange={(choice) =>
          choice && onChange(choice === DEFAULT_VALUE ? null : (choice as KvCacheChoice))
        }
      >
        <SelectTrigger aria-label={label} className='h-8 w-full text-[11px] text-foreground'>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={DEFAULT_VALUE}>{t('model.default')}</SelectItem>
          {kvCacheChoices.map((choice) => (
            <SelectItem key={choice} value={choice}>
              {choice.toUpperCase()}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </label>
  )
}

function CapabilityNotice({ capabilities }: { capabilities: LlmCapabilities }) {
  const { t } = useTranslation()
  return (
    <p className='flex items-start gap-2 text-[10px] leading-4 text-muted-foreground'>
      <Info className='mt-px size-3.5 shrink-0' />
      <span>
        {t('settings.llm.activeDevice', {
          device: capabilities.device,
          backend: capabilities.backend,
        })}{' '}
        {!capabilities.gpu_offload && t('settings.llm.noGpuOffload')}
      </span>
    </p>
  )
}

type ContextKind = (typeof contextKinds)[number]

function emptyContext(kind: ContextKind): ContextMode {
  switch (kind) {
    case 'dynamic':
      return { kind: 'dynamic' }
    case 'fixed':
      return { kind: 'fixed', size: 4096 }
    case 'bounded':
      return { kind: 'bounded', minimum: null, maximum: null }
  }
}

function layersOf(gpuLayers: GpuLayers): number {
  return gpuLayers.kind === 'custom' ? gpuLayers.layers : 0
}
