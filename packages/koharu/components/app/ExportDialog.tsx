'use client'

import { useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'

import type { EntityId, ExportFormat, ExportOptions } from '@koharu/bridge/protocol'
import { Button } from '@koharu/ui/components/button'
import { Checkbox } from '@koharu/ui/components/checkbox'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@koharu/ui/components/dialog'
import { Input } from '@koharu/ui/components/input'
import { Label } from '@koharu/ui/components/label'
import { RadioGroup, RadioGroupItem } from '@koharu/ui/components/radio-group'

const DEFAULT_PATTERN = '{index:04}_{label}'

/**
 * `naming::Template::parse`의 클라이언트 쪽 짝.
 *
 * 서버가 다시 검증하므로 이것은 보안 장치가 아니라, 사용자가 시작 버튼을
 * 누른 뒤가 아니라 타이핑하는 동안 오류를 보게 하려는 것이다. 규칙이
 * 어긋나면 서버가 잡는다.
 */
export function previewPattern(pattern: string): { name: string } | { error: string } {
  let out = ''
  let rest = pattern
  while (rest.includes('{')) {
    const open = rest.indexOf('{')
    out += rest.slice(0, open)
    const after = rest.slice(open + 1)
    const close = after.indexOf('}')
    if (close === -1) return { error: 'unclosed' }
    const token = after.slice(0, close)
    const [name, width] = token.includes(':') ? token.split(':', 2) : [token, undefined]
    if (width !== undefined && !/^\d+$/.test(width)) return { error: 'width' }
    if (name === 'index') out += '1'.padStart(Number(width ?? 0), '0')
    else if (name === 'label') out += 'page-01'
    else return { error: 'token' }
    rest = after.slice(close + 1)
  }
  if (rest.includes('}')) return { error: 'unclosed' }
  out += rest
  if (out.includes('/') || out.includes('\\') || out.includes('..')) return { error: 'separator' }
  if (!out.trim()) return { name: 'page' }
  return { name: out.trim() }
}

export function ExportDialog({
  open,
  onOpenChange,
  pages,
  selected,
  onStart,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  pages: EntityId[]
  selected: EntityId[]
  onStart: (pages: EntityId[], options: ExportOptions) => void
}) {
  const { t } = useTranslation()
  const [png, setPng] = useState(true)
  const [psd, setPsd] = useState(false)
  const [scope, setScope] = useState<'all' | 'selected'>(selected.length ? 'selected' : 'all')
  const [pattern, setPattern] = useState(DEFAULT_PATTERN)
  const [subfolders, setSubfolders] = useState(false)

  const formats = useMemo(() => {
    const chosen: ExportFormat[] = []
    if (png) chosen.push('png')
    if (psd) chosen.push('psd')
    return chosen
  }, [png, psd])

  const preview = useMemo(() => previewPattern(pattern), [pattern])
  const bothFormats = formats.length === 2
  const invalid = 'error' in preview
  const canStart = formats.length > 0 && !invalid && pages.length > 0

  const start = () => {
    onStart(scope === 'selected' ? selected : [], {
      formats,
      pattern,
      // 하위 폴더 체크박스가 비활성일 때는 체크 상태와 무관하게 false다.
      subfolders: bothFormats && subfolders,
    })
    onOpenChange(false)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className='sm:max-w-md'>
        <DialogHeader>
          <DialogTitle>{t('export.title')}</DialogTitle>
          <DialogDescription>{t('export.description')}</DialogDescription>
        </DialogHeader>

        <div className='space-y-4'>
          <fieldset className='space-y-2'>
            <legend className='text-xs font-medium'>{t('export.scope')}</legend>
            <RadioGroup
              value={scope}
              onValueChange={(value) => setScope(value as 'all' | 'selected')}
            >
              <div className='flex items-center gap-2'>
                <RadioGroupItem value='all' id='export-scope-all' />
                <Label htmlFor='export-scope-all'>
                  {t('export.scopeAll', { count: pages.length })}
                </Label>
              </div>
              <div className='flex items-center gap-2'>
                <RadioGroupItem
                  value='selected'
                  id='export-scope-selected'
                  disabled={selected.length === 0}
                />
                <Label htmlFor='export-scope-selected'>
                  {t('export.scopeSelected', { count: selected.length })}
                </Label>
              </div>
            </RadioGroup>
          </fieldset>

          <fieldset className='space-y-2'>
            <legend className='text-xs font-medium'>{t('export.formats')}</legend>
            <div className='flex items-center gap-2'>
              <Checkbox id='export-png' checked={png} onCheckedChange={(v) => setPng(v === true)} />
              <Label htmlFor='export-png'>PNG</Label>
            </div>
            <div className='flex items-center gap-2'>
              <Checkbox id='export-psd' checked={psd} onCheckedChange={(v) => setPsd(v === true)} />
              <Label htmlFor='export-psd'>PSD</Label>
            </div>
          </fieldset>

          <div className='space-y-1'>
            <Label htmlFor='export-pattern'>{t('export.pattern')}</Label>
            <Input
              id='export-pattern'
              value={pattern}
              onChange={(event) => setPattern(event.target.value)}
              aria-invalid={invalid}
            />
            {/* `invalid` 대신 여기서 다시 좁히는 것은 TypeScript가 별도
                boolean으로는 유니온을 좁혀 주지 않기 때문이다. 미리보기는
                텍스트 노드가 쪼개지지 않도록 템플릿 문자열 하나로 낸다 —
                테스트가 문자열 전체로 찾는다. */}
            {'error' in preview ? (
              <p role='alert' className='text-[11px] text-destructive'>
                {t(`export.patternError.${preview.error}`)}
              </p>
            ) : (
              <p className='text-[11px] text-muted-foreground'>
                {`${preview.name}.${formats[0] ?? 'png'}`}
              </p>
            )}
          </div>

          <div className='flex items-center gap-2'>
            <Checkbox
              id='export-subfolders'
              checked={bothFormats && subfolders}
              disabled={!bothFormats}
              onCheckedChange={(v) => setSubfolders(v === true)}
            />
            <Label htmlFor='export-subfolders'>{t('export.subfolders')}</Label>
          </div>
        </div>

        <DialogFooter>
          <Button variant='ghost' onClick={() => onOpenChange(false)}>
            {t('export.cancel')}
          </Button>
          <Button disabled={!canStart} onClick={start}>
            {t('export.start')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
