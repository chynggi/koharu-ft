'use client'

import {
  Blend,
  Brush,
  Circle,
  Eraser,
  Hand,
  Minus,
  MousePointer2,
  PaintBucket,
  Pipette,
  Plus,
  Sparkles,
  SplinePointer,
  Square,
  Type,
} from 'lucide-react'
import { useTranslation } from 'react-i18next'

import { ColorWell } from '@/components/controls/ColorWell'
import { usePage } from '@/lib/queries'
import {
  isBrushTool,
  maxBrushDiameter,
  MIN_BRUSH_DIAMETER,
  toolKeys,
  useKoharuStore,
  type CanvasTool,
} from '@/lib/store'
import type { RampAlong } from '@koharu/ipc/protocol'
import { Button } from '@koharu/ui/components/button'
import {
  NumberField,
  NumberFieldDecrement,
  NumberFieldGroup,
  NumberFieldIncrement,
  NumberFieldInput,
} from '@koharu/ui/components/number-field'
import {
  Popover,
  PopoverContent,
  PopoverTitle,
  PopoverTrigger,
} from '@koharu/ui/components/popover'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@koharu/ui/components/select'
import { Slider } from '@koharu/ui/components/slider'
import { Tooltip, TooltipContent, TooltipTrigger } from '@koharu/ui/components/tooltip'

const tools = [
  ['select', MousePointer2],
  ['direct_select', SplinePointer],
  ['text', Type],
  ['rectangle', Square],
  ['ellipse', Circle],
  ['draw', Brush],
  ['eraser', Eraser],
  ['bucket', PaintBucket],
  ['color_picker', Pipette],
  ['remove', Sparkles],
  ['pan', Hand],
] as const satisfies ReadonlyArray<readonly [CanvasTool, typeof MousePointer2]>

export function ToolBar() {
  const { t } = useTranslation()
  const page = usePage().data
  const active = useKoharuStore((state) => state.tool)
  const brush = useKoharuStore((state) => state.brush)
  const setTool = useKoharuStore((state) => state.setTool)
  const setBrush = useKoharuStore((state) => state.setBrush)
  const hasBrush = isBrushTool(active)
  // The shape tools and the bucket fill with the brush colour, as Photoshop's with the
  // foreground colour.
  const colored =
    active === 'draw' || active === 'rectangle' || active === 'ellipse' || active === 'bucket'

  return (
    <aside className='absolute top-3 left-3 z-20 flex w-11 flex-col rounded-2xl border border-border bg-[var(--surface-floating)] p-1 shadow-[var(--shadow-toolrail)]'>
      <div className='flex flex-col items-center py-0.5'>
        {tools.map(([tool, Icon], index) => (
          <div key={tool} className='contents'>
            {index === tools.length - 1 && <span className='my-1 h-px w-5 bg-border' />}
            <Tooltip>
              <TooltipTrigger
                render={
                  <Button
                    type='button'
                    variant='ghost'
                    size='icon'
                    disabled={!page}
                    aria-label={t(`tools.${tool}`)}
                    data-active={active === tool}
                    className='relative text-muted-foreground hover:bg-foreground/[0.06] hover:text-foreground disabled:opacity-30 data-[active=true]:bg-accent data-[active=true]:text-accent-foreground'
                    onClick={() => setTool(tool)}
                  />
                }
              >
                <Icon className='size-4' />
              </TooltipTrigger>
              <TooltipContent side='right'>
                {t(`tools.${tool}`)}
                <span className='ml-2 opacity-60'>{toolKeys[tool].toUpperCase()}</span>
              </TooltipContent>
            </Tooltip>
          </div>
        ))}
      </div>

      {(hasBrush || colored) && (
        <div className='flex flex-col items-center gap-1 border-t border-border/80 py-1.5'>
          {colored && (
            <ColorWell value={brush.color} onChange={(color) => setBrush({ ...brush, color })} />
          )}
          {active === 'draw' && <RampControl />}
          {hasBrush && (
            <BrushSize
              value={brush.diameter}
              onChange={(diameter) => setBrush({ ...brush, diameter })}
            />
          )}
        </div>
      )}
    </aside>
  )
}

/** Where a ramp runs, each with its label key. */
const rampAlongs: Record<RampAlong, string> = {
  stroke: 'rampStroke',
  horizontal: 'rampHorizontal',
  vertical: 'rampVertical',
}

/** The draw tool's ramp: the colour its paint fades to from the brush colour, and where. */
function RampControl() {
  const { t } = useTranslation()
  const brush = useKoharuStore((state) => state.brush)
  const setBrush = useKoharuStore((state) => state.setBrush)
  const ramp = brush.ramp
  return (
    <Popover>
      <Tooltip>
        <PopoverTrigger
          render={
            <TooltipTrigger
              render={
                <Button
                  type='button'
                  variant='ghost'
                  size='icon'
                  aria-label={t('tools.ramp')}
                  className='size-8 rounded-xl text-muted-foreground hover:bg-foreground/[0.06] hover:text-foreground'
                />
              }
            >
              {ramp ? (
                <span
                  aria-hidden='true'
                  className='size-4 rounded-[4px] border border-border'
                  style={{ background: `linear-gradient(90deg, ${brush.color}, ${ramp.to})` }}
                />
              ) : (
                <Blend className='size-4' />
              )}
            </TooltipTrigger>
          }
        />
        <TooltipContent side='right'>{t('tools.ramp')}</TooltipContent>
      </Tooltip>
      <PopoverContent
        side='right'
        align='center'
        sideOffset={8}
        className='w-52 gap-2.5 rounded-xl p-2.5'
      >
        <PopoverTitle className='text-[11px]'>{t('tools.ramp')}</PopoverTitle>
        <div className='flex items-center gap-2'>
          <ColorWell
            label={t('tools.rampColor')}
            size='sm'
            allowTransparent
            value={ramp?.to ?? null}
            onChange={(to) =>
              setBrush({
                ...brush,
                ramp: to === null ? null : { to, along: ramp?.along ?? 'stroke' },
              })
            }
          />
          <Select
            value={ramp?.along ?? 'stroke'}
            disabled={!ramp}
            onValueChange={(along) =>
              ramp && along && setBrush({ ...brush, ramp: { ...ramp, along: along as RampAlong } })
            }
          >
            <SelectTrigger size='sm' aria-label={t('tools.rampAlong')} className='min-w-0 flex-1'>
              <SelectValue>{(value: RampAlong) => t(`tools.${rampAlongs[value]}`)}</SelectValue>
            </SelectTrigger>
            <SelectContent>
              {(Object.keys(rampAlongs) as RampAlong[]).map((value) => (
                <SelectItem key={value} value={value}>
                  {t(`tools.${rampAlongs[value]}`)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </PopoverContent>
    </Popover>
  )
}

function BrushSize({ value, onChange }: { value: number; onChange: (value: number) => void }) {
  const { t } = useTranslation()
  const roundedValue = Math.round(value)
  const page = usePage().data
  const maxSize = maxBrushDiameter(page?.size ?? { width: 2048, height: 2048 })

  return (
    <Popover>
      <Tooltip>
        <PopoverTrigger
          render={
            <TooltipTrigger
              render={
                <Button
                  type='button'
                  variant='ghost'
                  size='icon'
                  aria-label={t('tools.brushSizePixels', { size: roundedValue })}
                  className='size-8 rounded-xl font-sans text-[11px] leading-none font-medium tracking-[-0.02em] text-muted-foreground tabular-nums hover:bg-foreground/[0.06] hover:text-foreground'
                />
              }
            >
              {roundedValue}
            </TooltipTrigger>
          }
        />
        <TooltipContent side='right'>
          {t('tools.brushSizeShort', { size: roundedValue })}
        </TooltipContent>
      </Tooltip>
      <PopoverContent
        side='right'
        align='center'
        sideOffset={8}
        className='w-48 gap-2.5 rounded-xl p-2.5'
      >
        <div className='flex items-center justify-between gap-3'>
          <PopoverTitle className='text-[11px]'>{t('tools.brushSize')}</PopoverTitle>
          <NumberField
            min={MIN_BRUSH_DIAMETER}
            step={1}
            value={roundedValue}
            className='w-20'
            onValueChange={(next) => {
              if (next !== null) onChange(Math.max(next, MIN_BRUSH_DIAMETER))
            }}
          >
            <NumberFieldGroup className='h-7'>
              <NumberFieldDecrement aria-label={t('tools.decreaseBrushSize')}>
                <Minus />
              </NumberFieldDecrement>
              <NumberFieldInput aria-label={t('tools.brushSizeInput')} />
              <NumberFieldIncrement aria-label={t('tools.increaseBrushSize')}>
                <Plus />
              </NumberFieldIncrement>
            </NumberFieldGroup>
          </NumberField>
        </div>
        <Slider
          aria-label={t('tools.brushSize')}
          min={MIN_BRUSH_DIAMETER}
          max={maxSize}
          step={1}
          value={value}
          className='py-1 [&_[data-slot=slider-thumb]]:size-2.5'
          onValueChange={onChange}
        />
      </PopoverContent>
    </Popover>
  )
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, Number.isFinite(value) ? value : min))
}
