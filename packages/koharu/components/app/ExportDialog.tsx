'use client'

import { useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { Choice } from '@/components/controls/Choice'
import type { PageId, ExportFormat, ExportOptions, ImageFormat } from '@koharu/ipc/protocol'
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
import { Slider } from '@koharu/ui/components/slider'

const DEFAULT_PATTERN = '{index:04}_{label}'

const imageNames = { png: 'PNG', jpeg: 'JPEG', webp: 'WebP' } as const
const extensions = { png: 'png', jpeg: 'jpg', webp: 'webp' } as const

/**
 * `naming::Template::parse`의 클라이언트 쪽 짝.
 *
 * 서버가 다시 검증하므로 이것은 보안 장치가 아니라, 사용자가 시작 버튼을
 * 누른 뒤가 아니라 타이핑하는 동안 오류를 보게 하려는 것이다. 규칙이
 * 어긋나면 서버가 잡는다.
 */
export function previewPattern(
  pattern: string,
): { name: string } | { error: 'unclosed' | 'width' | 'token' | 'separator' } {
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
    if (width !== undefined && (!/^\d+$/.test(width) || Number(width) > 32))
      return { error: 'width' }
    if (name === 'index') out += '1'.padStart(Number(width ?? 0), '0')
    else if (name === 'label') out += 'page-01'
    else return { error: 'token' }
    rest = after.slice(close + 1)
  }
  if (rest.includes('}')) return { error: 'unclosed' }
  out += rest
  const control = [...out].some((character) => character.charCodeAt(0) < 0x20)
  if (control || /[<>:"/\\|?*]/.test(out) || out.includes('..')) return { error: 'separator' }
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
  pages: PageId[]
  selected: PageId[]
  onStart: (pages: PageId[], options: ExportOptions) => void
}) {
  const { t } = useTranslation()
  const [images, setImages] = useState(true)
  const [imageKind, setImageKind] = useState<keyof typeof imageNames>('png')
  const [quality, setQuality] = useState(90)
  const [psd, setPsd] = useState(false)
  const [cbz, setCbz] = useState(false)
  const [pdf, setPdf] = useState(false)
  const [scope, setScope] = useState<'all' | 'selected'>(selected.length ? 'selected' : 'all')
  const [pattern, setPattern] = useState(DEFAULT_PATTERN)
  const [subfolders, setSubfolders] = useState(false)

  const formats = useMemo(() => {
    const chosen: ExportFormat[] = []
    if (images) chosen.push('images')
    if (psd) chosen.push('psd')
    if (cbz) chosen.push('cbz')
    if (pdf) chosen.push('pdf')
    return chosen
  }, [images, psd, cbz, pdf])
  const image: ImageFormat =
    imageKind === 'png'
      ? 'png'
      : imageKind === 'jpeg'
        ? { jpeg: { quality } }
        : { webp: { quality } }

  const preview = useMemo(() => previewPattern(pattern), [pattern])
  // 하위 폴더는 페이지마다 파일을 쓰는 형식이 둘일 때만 뜻이 있다. 아카이브는
  // 대상 폴더 바로 아래에 하나만 놓이므로 세지 않는다.
  const pageFormatCount = (images ? 1 : 0) + (psd ? 1 : 0)
  const bothFormats = pageFormatCount === 2
  const invalid = 'error' in preview
  const canStart =
    formats.length > 0 && !invalid && pages.length > 0 && (scope === 'all' || selected.length > 0)
  // 아카이브 항목도 이미지 형식을 따른다. 페이지 파일 형식이 하나도 없으면
  // 미리보기도 항목 이름을 보여줘야 하므로 이미지 확장자로 떨어진다.
  const previewExtension = psd && !images ? 'psd' : extensions[imageKind]

  const start = () => {
    onStart(scope === 'selected' ? selected : pages, {
      formats,
      image,
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
              <Checkbox
                id='export-images'
                checked={images}
                onCheckedChange={(v) => setImages(v === true)}
              />
              <Label htmlFor='export-images'>{t('export.images')}</Label>
            </div>
            <div className='flex items-center gap-2'>
              <Checkbox id='export-psd' checked={psd} onCheckedChange={(v) => setPsd(v === true)} />
              <Label htmlFor='export-psd'>PSD</Label>
            </div>
            <div className='flex items-center gap-2'>
              <Checkbox id='export-cbz' checked={cbz} onCheckedChange={(v) => setCbz(v === true)} />
              <Label htmlFor='export-cbz'>CBZ</Label>
            </div>
            <div className='flex items-center gap-2'>
              <Checkbox id='export-pdf' checked={pdf} onCheckedChange={(v) => setPdf(v === true)} />
              <Label htmlFor='export-pdf'>PDF</Label>
            </div>
            <p className='text-[11px] text-muted-foreground'>{t('export.cbzHint')}</p>
          </fieldset>

          {/* 이미지 파일과 CBZ 항목이 함께 쓰는 형식, 그리고 JPEG·WebP의 품질. */}
          <div className='grid grid-cols-[auto_minmax(0,1fr)] items-center gap-x-4 gap-y-3 text-[12px]'>
            <span className='text-muted-foreground'>{t('export.imageFormat')}</span>
            <Choice
              value={imageKind}
              choices={imageNames}
              label={t('export.imageFormat')}
              disabled={!images && !cbz}
              onChange={setImageKind}
            />
            {imageKind !== 'png' && (
              <>
                <span className='text-muted-foreground'>{t('export.quality')}</span>
                <div className='flex min-w-0 items-center gap-3'>
                  <Slider
                    aria-label={t('export.quality')}
                    min={1}
                    max={100}
                    step={1}
                    value={quality}
                    disabled={!images && !cbz}
                    onValueChange={setQuality}
                  />
                  <span className='w-6 shrink-0 text-right tabular-nums'>{quality}</span>
                </div>
              </>
            )}
          </div>

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
                {`${preview.name}.${previewExtension}`}
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
