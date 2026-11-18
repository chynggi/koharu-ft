'use client'

import { ImagePlus, Minus, Plus, X } from 'lucide-react'
import { useMemo } from 'react'
import { useTranslation } from 'react-i18next'

import { ColorWell } from '@/components/controls/ColorWell'
import { dispatch } from '@/lib/backend'
import { expandLayerSelection, walkLayers } from '@/lib/document'
import { usePage } from '@/lib/queries'
import { setFill, setStroke } from '@/lib/scene'
import { hexToRgba, rgbaToHex, useKoharuStore, type CanvasTool } from '@/lib/store'
import {
  commands,
  type Gradient,
  type GradientStop,
  type Layer,
  type Rgba,
  type ShapeFill,
} from '@koharu/ipc/protocol'
import { Button } from '@koharu/ui/components/button'
import {
  NumberField,
  NumberFieldDecrement,
  NumberFieldGroup,
  NumberFieldIncrement,
  NumberFieldInput,
} from '@koharu/ui/components/number-field'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@koharu/ui/components/select'

type ShapeLayer = Extract<Layer, { kind: 'shape' }>
type FillKind = 'none' | 'solid' | 'linear' | 'radial' | 'sweep' | 'pattern'

/** Each fill kind's label key, in the order the menu lists them. */
const fillKinds: Record<FillKind, string> = {
  none: 'fillNone',
  solid: 'fillSolid',
  linear: 'fillLinear',
  radial: 'fillRadial',
  sweep: 'fillSweep',
  pattern: 'fillPattern',
}
const BLACK: Rgba = [0, 0, 0, 1]
const WHITE: Rgba = [1, 1, 1, 1]
/** The scene's bounds on a gradient's stops. */
const MAX_STOPS = 32

/** The tools whose selection the bar follows. */
const TOOLS: CanvasTool[] = ['select', 'direct_select', 'rectangle', 'ellipse']

function kindOf(fill: ShapeFill | null | undefined): FillKind {
  if (!fill) return 'none'
  if (Array.isArray(fill)) return 'solid'
  if ('pattern' in fill && fill.pattern) return 'pattern'
  const kind = fill.gradient.kind
  return 'linear' in kind ? 'linear' : 'radial' in kind ? 'radial' : 'sweep'
}

/** A linear gradient's direction in degrees, clockwise from left to right. */
function angleOf(gradient: Gradient): number {
  if (!('linear' in gradient.kind) || !gradient.kind.linear) return 0
  const { start, end } = gradient.kind.linear
  return Math.round((Math.atan2(end[1] - start[1], end[0] - start[0]) * 180) / Math.PI)
}

/** A linear gradient across the shape's bounds at `degrees`, through their centre. */
function linear(degrees: number): Gradient['kind'] {
  const radians = (degrees * Math.PI) / 180
  const [x, y] = [Math.cos(radians) / 2, Math.sin(radians) / 2]
  return {
    linear: {
      start: [0.5 - x, 0.5 - y],
      end: [0.5 + x, 0.5 + y],
    },
  }
}

/** The selected shapes' fill and stroke in a bar over the canvas, as Photoshop's options bar shows
 * a shape's fill and stroke. A change goes to every shape selected. */
export function ShapeBar() {
  const { t } = useTranslation()
  const tool = useKoharuStore((state) => state.tool)
  const page = usePage().data
  const selectedIds = useKoharuStore((state) => state.selectedLayers)
  const targets = useMemo(() => {
    if (!page) return []
    const leaves = expandLayerSelection(page.layers, selectedIds)
    return walkLayers(page.layers)
      .map(({ layer }) => layer)
      .filter((layer): layer is ShapeLayer => leaves.includes(layer.id) && layer.kind === 'shape')
  }, [page, selectedIds])
  const current = targets[0]
  if (!current || !TOOLS.includes(tool)) return null

  const ids = targets.map((layer) => layer.id)
  const fill = current.fill ?? null
  const kind = kindOf(fill)
  const gradient = fill && !Array.isArray(fill) && 'gradient' in fill ? fill.gradient : null
  const stops = gradient?.stops ?? null
  const color = Array.isArray(fill) ? fill : (stops?.[0]?.color ?? BLACK)
  const stroke = current.stroke ?? null

  const fillWith = (next: ShapeFill | null) => dispatch(setFill, ids, next)
  const withStops = (next: GradientStop[]) =>
    gradient &&
    fillWith({
      gradient: { ...gradient, stops: [...next].sort((a, b) => a.offset - b.offset) },
    })

  const choose = (next: FillKind) => {
    if (next === kind) return
    if (next === 'pattern') {
      dispatch(commands.fillWithPattern, ids)
      return
    }
    if (next === 'none') return fillWith(null)
    if (next === 'solid') return fillWith(color)
    const kept = stops ?? [
      { offset: 0, color },
      { offset: 1, color: WHITE },
    ]
    const shape: Gradient['kind'] =
      next === 'linear'
        ? linear(0)
        : next === 'radial'
          ? { radial: { center: [0.5, 0.5], radius: 0.5 } }
          : { sweep: { center: [0.5, 0.5], start_degrees: 0, end_degrees: 360 } }
    fillWith({ gradient: { kind: shape, stops: kept, extend: gradient?.extend ?? 'pad' } })
  }

  const strokeWidth = stroke?.width ?? 0

  return (
    <div
      role='toolbar'
      aria-label={t('shape.title')}
      data-testid='shape-bar'
      className='absolute top-3 left-1/2 z-20 flex max-w-[calc(100%-16px)] -translate-x-1/2 flex-wrap items-center gap-1 rounded-xl border border-border bg-[var(--surface-floating)] p-1.5 shadow-[var(--shadow-toolrail)]'
    >
      <Select value={kind} onValueChange={(next) => next && choose(next as FillKind)}>
        <SelectTrigger size='sm' aria-label={t('shape.fill')} className='w-32'>
          <SelectValue>{(value: FillKind) => t(`shape.${fillKinds[value]}`)}</SelectValue>
        </SelectTrigger>
        <SelectContent>
          {(Object.keys(fillKinds) as FillKind[]).map((value) => (
            <SelectItem key={value} value={value}>
              {t(`shape.${fillKinds[value]}`)}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {kind === 'solid' && (
        <ColorWell
          label={t('shape.fillColor')}
          size='sm'
          value={rgbaToHex(color)}
          onChange={(next) => fillWith(hexToRgba(next))}
        />
      )}
      {kind === 'pattern' && (
        <Button
          type='button'
          variant='outline'
          size='sm'
          className='h-6 gap-1 px-2 text-[11px]'
          onClick={() => dispatch(commands.fillWithPattern, ids)}
        >
          <ImagePlus className='size-3' />
          {t('shape.choosePattern')}
        </Button>
      )}
      {gradient && kind === 'linear' && (
        <Field
          label={t('shape.angle')}
          value={angleOf(gradient)}
          min={-180}
          max={180}
          step={15}
          onChange={(degrees) => fillWith({ gradient: { ...gradient, kind: linear(degrees) } })}
        />
      )}
      {stops?.map((stop, index) => (
        <div key={index} className='flex items-center gap-0.5 rounded-md border border-input p-0.5'>
          <ColorWell
            label={t('shape.stopColor', { index: index + 1 })}
            size='sm'
            value={rgbaToHex(stop.color)}
            onChange={(next) =>
              withStops(
                stops.map((other, at) =>
                  at === index ? { ...other, color: hexToRgba(next) } : other,
                ),
              )
            }
          />
          <Field
            label={t('shape.stopPosition', { index: index + 1 })}
            value={Math.round(stop.offset * 100)}
            min={0}
            max={100}
            step={5}
            onChange={(percent) =>
              withStops(
                stops.map((other, at) =>
                  at === index ? { ...other, offset: percent / 100 } : other,
                ),
              )
            }
          />
          <Button
            type='button'
            variant='ghost'
            size='icon-xs'
            aria-label={t('shape.removeStop', { index: index + 1 })}
            disabled={stops.length <= 2}
            onClick={() => withStops(stops.filter((_, at) => at !== index))}
          >
            <X className='size-3' />
          </Button>
        </div>
      ))}
      {stops && (
        <Button
          type='button'
          variant='ghost'
          size='icon-xs'
          aria-label={t('shape.addStop')}
          disabled={stops.length >= MAX_STOPS}
          onClick={() => {
            // Halfway along the widest gap, in its left stop's colour.
            const gaps = stops.slice(1).map((stop, at) => [stop.offset - stops[at]!.offset, at])
            const [, at] = gaps.reduce((widest, gap) => (gap[0]! > widest[0]! ? gap : widest))
            const left = stops[at!]!
            withStops([
              ...stops,
              { offset: (left.offset + stops[at! + 1]!.offset) / 2, color: left.color },
            ])
          }}
        >
          <Plus className='size-3' />
        </Button>
      )}
      <span className='mx-0.5 h-4 w-px bg-border' />
      <ColorWell
        label={t('shape.strokeColor')}
        size='sm'
        allowTransparent
        value={stroke ? rgbaToHex(stroke.color) : null}
        onChange={(next) =>
          dispatch(
            setStroke,
            ids,
            next === null ? null : { color: hexToRgba(next), width: strokeWidth || 2 },
          )
        }
      />
      <Field
        label={t('shape.strokeWidth')}
        value={strokeWidth}
        min={0}
        max={200}
        step={1}
        onChange={(width) =>
          dispatch(setStroke, ids, width > 0 ? { color: stroke?.color ?? BLACK, width } : null)
        }
      />
    </div>
  )
}

function Field({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (value: number) => void
}) {
  const { t } = useTranslation()
  return (
    <NumberField
      className='w-20'
      value={value}
      min={min}
      max={max}
      step={step}
      onValueChange={(next) => {
        if (next !== null && next >= min && next <= max && next !== value) onChange(next)
      }}
    >
      <NumberFieldGroup>
        <NumberFieldDecrement aria-label={t('shape.decrease', { name: label })}>
          <Minus />
        </NumberFieldDecrement>
        <NumberFieldInput aria-label={label} />
        <NumberFieldIncrement aria-label={t('shape.increase', { name: label })}>
          <Plus />
        </NumberFieldIncrement>
      </NumberFieldGroup>
    </NumberField>
  )
}
