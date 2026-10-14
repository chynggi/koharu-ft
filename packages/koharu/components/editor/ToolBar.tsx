'use client'

import {
  ArrowRight,
  Blend,
  Brush,
  Circle,
  Eraser,
  Hand,
  LassoSelect,
  Minus,
  MousePointer2,
  PaintBucket,
  Pipette,
  Plus,
  ScanText,
  Sparkles,
  Square,
  Type,
} from 'lucide-react'
import { useTranslation } from 'react-i18next'

import { ColorWell } from '@/components/controls/ColorWell'
import { usePage } from '@/lib/queries'
import {
  SHAPE_KINDS,
  colorToHex,
  fillKind,
  hexToColor,
  isShapeTool,
  withFillKind,
  type ShapeKind,
} from '@/lib/shapes'
import {
  isBrushTool,
  maxBrushDiameter,
  MIN_BRUSH_DIAMETER,
  useKoharuStore,
  type CanvasTool,
} from '@/lib/store'
import type { BrushRamp, RampAxis, ShapeStyle } from '@koharu/bridge/protocol'
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
import { Slider } from '@koharu/ui/components/slider'
import { Tooltip, TooltipContent, TooltipTrigger } from '@koharu/ui/components/tooltip'

const tools = [
  ['select', MousePointer2],
  ['text', Type],
  ['shape', Square],
  ['draw', Brush],
  ['fill', PaintBucket],
  ['eraser', Eraser],
  ['color_picker', Pipette],
  ['remove', Sparkles],
  ['manual_ocr_box', ScanText],
  ['manual_ocr_lasso', LassoSelect],
  ['pan', Hand],
] as const satisfies ReadonlyArray<readonly [CanvasTool, typeof MousePointer2]>

const shapeIcons = {
  rectangle: Square,
  ellipse: Circle,
  line: Minus,
  arrow: ArrowRight,
} as const satisfies Record<ShapeKind, typeof MousePointer2>

const DEFAULT_STROKE_WIDTH = 2

type BrushRampKind = BrushRamp['kind'] | 'none'

const RAMP_KINDS: readonly BrushRampKind[] = ['none', 'along_stroke', 'fixed_axis']

// The wire tags are snake_case; the locale keys are camelCase to match the
// existing `tools.` vocabulary. This is the one place the two meet.
const RAMP_KIND_KEYS: Record<BrushRampKind, string> = {
  none: 'none',
  along_stroke: 'alongStroke',
  fixed_axis: 'fixedAxis',
}

const RAMP_AXES: readonly RampAxis[] = ['horizontal', 'vertical']

export function ToolBar() {
  const { t } = useTranslation()
  const page = usePage().data
  const active = useKoharuStore((state) => state.tool)
  const brush = useKoharuStore((state) => state.brush)
  const fillColor = useKoharuStore((state) => state.fillColor)
  const setTool = useKoharuStore((state) => state.setTool)
  const setBrush = useKoharuStore((state) => state.setBrush)
  const setFillColor = useKoharuStore((state) => state.setFillColor)
  const shortcuts = useKoharuStore((state) => state.shortcuts)
  const hasBrush = isBrushTool(active)
  const cleanupLayer = page?.layers.find(
    (layer) => layer.type === 'raster' && layer.kind === 'cleanup',
  )
  // Bucket fill is raster-only, and the server rejects anything else. The tool
  // is disabled rather than reporting that on every click. One layer is the
  // command's whole target, so a multi-layer selection is not fillable either —
  // the same rule the canvas click applies.
  const hasFillTarget = useKoharuStore((state) => {
    if (state.selectedLayers.length !== 1) return false
    const target = state.selectedLayers[0]
    return page?.layers.some((layer) => layer.id === target && layer.type === 'raster') ?? false
  })

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
                    disabled={!page || (tool === 'fill' && !hasFillTarget)}
                    aria-label={t(`tools.${tool}`)}
                    data-active={active === tool}
                    className='relative text-muted-foreground hover:bg-foreground/[0.06] hover:text-foreground disabled:opacity-30 data-[active=true]:bg-accent data-[active=true]:text-accent-foreground'
                    onClick={() => setTool(tool, cleanupLayer?.id)}
                  />
                }
              >
                <Icon className='size-4' />
              </TooltipTrigger>
              <TooltipContent side='right'>
                {t(`tools.${tool}`)}
                <span className='ml-2 opacity-60'>{shortcuts[tool].toUpperCase()}</span>
              </TooltipContent>
            </Tooltip>
          </div>
        ))}
      </div>

      {hasBrush && (
        <div className='flex flex-col items-center gap-1 border-t border-border/80 py-1.5'>
          {/* A ramp and a flat colour cannot both be live: Rust reads `ramp`
              whenever it is set and ignores `color`, so leaving the flat well
              on screen would let the user change a value nothing consumes. */}
          {active === 'draw' && brush.ramp === null && (
            <ColorWell value={brush.color} onChange={(color) => setBrush({ ...brush, color })} />
          )}
          {active === 'draw' && <BrushRampControl />}
          <BrushSize
            value={brush.diameter}
            onChange={(diameter) => setBrush({ ...brush, diameter })}
          />
        </div>
      )}

      {active === 'fill' && (
        <div className='flex flex-col items-center gap-1 border-t border-border/80 py-1.5'>
          <ColorWell
            label={t('tools.fillColor')}
            value={fillColor}
            onChange={(color) => setFillColor(color)}
          />
        </div>
      )}

      {isShapeTool(active) && <ShapeStyles />}
    </aside>
  )
}

function ShapeStyles() {
  const { t } = useTranslation()
  const kind = useKoharuStore((state) => state.shapeKind)
  const style = useKoharuStore((state) => state.shapeStyle)
  const setShapeKind = useKoharuStore((state) => state.setShapeKind)
  const setShapeStyle = useKoharuStore((state) => state.setShapeStyle)
  const stroke = style.stroke
  const KindIcon = shapeIcons[kind]

  return (
    <div className='flex flex-col items-center gap-1 border-t border-border/80 py-1.5'>
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
                    aria-label={t('inspector.shape.kindLabel', {
                      kind: t(`inspector.shape.kind.${kind}`),
                    })}
                    className='size-8 rounded-xl text-muted-foreground hover:bg-foreground/[0.06] hover:text-foreground'
                  />
                }
              >
                <KindIcon className='size-4' />
              </TooltipTrigger>
            }
          />
          <TooltipContent side='right'>
            {t('inspector.shape.kindLabel', { kind: t(`inspector.shape.kind.${kind}`) })}
          </TooltipContent>
        </Tooltip>
        <PopoverContent
          side='right'
          align='center'
          sideOffset={8}
          className='w-56 gap-2.5 rounded-xl p-2.5'
        >
          <PopoverTitle className='text-[11px]'>{t('inspector.shape.kindTitle')}</PopoverTitle>
          <div className='grid grid-cols-6 gap-1'>
            {SHAPE_KINDS.map((candidate) => {
              const Icon = shapeIcons[candidate]
              return (
                <button
                  key={candidate}
                  type='button'
                  aria-label={t(`inspector.shape.kind.${candidate}`)}
                  aria-pressed={kind === candidate}
                  className='grid h-7 place-items-center rounded-md border border-input text-muted-foreground hover:text-foreground aria-pressed:border-primary aria-pressed:bg-accent aria-pressed:text-accent-foreground'
                  onClick={() => setShapeKind(candidate)}
                >
                  <Icon className='size-3.5' />
                </button>
              )
            })}
          </div>
          <div className='flex items-center justify-between gap-3'>
            <PopoverTitle className='text-[11px]'>{t('inspector.shape.fillKindLabel')}</PopoverTitle>
            <div className='flex items-center gap-1' role='radiogroup' aria-label={t('inspector.shape.fillKindLabel')}>
              {/* Only the kind is chosen here, so a new shape can default to a
                  gradient. Editing the stops belongs to the Inspector, which
                  owns the detailed style UI; this popover stays a preset.

                  The comparison is on the fill's own tag rather than on
                  `'gradient' in fill`, because a pattern fill has neither the
                  `solid` nor the `gradient` key: testing for `gradient` alone
                  would report a pattern as a solid and light up the wrong
                  preset. Absent fill also counts as solid, matching
                  `defaultShapeStyle`. */}
              {(['solid', 'gradient'] as const).map((kind) => (
                <button
                  key={kind}
                  type='button'
                  role='radio'
                  aria-label={t(`inspector.shape.fillKind.${kind}`)}
                  aria-checked={fillKind(style.fill) === kind}
                  data-active={fillKind(style.fill) === kind}
                  className='h-6 rounded-md border border-input px-2 text-[10px] text-muted-foreground data-[active=true]:bg-accent data-[active=true]:text-accent-foreground'
                  onClick={() => setShapeStyle(withFillKind(style, kind))}
                >
                  {t(`inspector.shape.fillKind.${kind}`)}
                </button>
              ))}
            </div>
          </div>
          {style.fill !== null && 'solid' in style.fill && (
            <div className='flex items-center justify-between gap-3'>
              <PopoverTitle className='text-[11px]'>{t('inspector.shape.fill')}</PopoverTitle>
              <ColorWell
                label={t('inspector.shape.fill')}
                size='sm'
                allowTransparent
                // With the kind fixed to solid, this well can never downgrade a
                // gradient; the gradient's colours live in the Inspector's stop
                // editor, which this popover deliberately does not duplicate.
                value={colorToHex(style.fill.solid)}
                onChange={(color) =>
                  setShapeStyle({
                    ...style,
                    fill: color === null ? null : { solid: hexToColor(color) },
                  })
                }
              />
            </div>
          )}
          <div className='flex items-center justify-between gap-3'>
            <PopoverTitle className='text-[11px]'>{t('inspector.shape.stroke')}</PopoverTitle>
            <ColorWell
              label={t('inspector.shape.stroke')}
              size='sm'
              allowTransparent
              value={stroke ? colorToHex(stroke.color) : null}
              onChange={(color) =>
                setShapeStyle({
                  ...style,
                  stroke:
                    color === null
                      ? null
                      : { ...(stroke ?? defaultStroke()), color: hexToColor(color) },
                })
              }
            />
          </div>
          <div className='flex items-center justify-between gap-3'>
            <PopoverTitle className='text-[11px]'>{t('inspector.shape.strokeWidth')}</PopoverTitle>
            <NumberField
              min={0.5}
              step={0.5}
              value={stroke?.width ?? DEFAULT_STROKE_WIDTH}
              className='w-20'
              onValueChange={(next) => {
                if (next === null) return
                setShapeStyle({
                  ...style,
                  stroke: { ...(stroke ?? defaultStroke()), width: Math.max(next, 0.5) },
                })
              }}
            >
              <NumberFieldGroup className='h-7'>
                <NumberFieldInput aria-label={t('inspector.shape.strokeWidth')} />
              </NumberFieldGroup>
            </NumberField>
          </div>
        </PopoverContent>
      </Popover>
    </div>
  )
}

function BrushRampControl() {
  const { t } = useTranslation()
  const brush = useKoharuStore((state) => state.brush)
  const setBrush = useKoharuStore((state) => state.setBrush)
  const ramp = brush.ramp

  // Switching kind keeps the two colours: only the rule that turns a position
  // into a parameter changes, so re-picking the ends would be busywork.
  const setKind = (kind: BrushRampKind) => {
    if (kind === 'none') {
      setBrush({ ...brush, ramp: null })
      return
    }
    const from = ramp?.from ?? hexToColor(brush.color)
    const to = ramp?.to ?? [0, 0, 0, 255]
    setBrush({
      ...brush,
      ramp:
        kind === 'along_stroke'
          ? { kind, from, to }
          : {
              kind,
              from,
              to,
              // The server defaults nothing here; a fixed-axis ramp must always
              // name the axis it varies along.
              axis: ramp?.kind === 'fixed_axis' ? ramp.axis : 'horizontal',
            },
    })
  }

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
                  aria-label={t('tools.brushRampTitle')}
                  data-active={ramp !== null}
                  className='size-8 rounded-xl text-muted-foreground hover:bg-foreground/[0.06] hover:text-foreground data-[active=true]:bg-accent data-[active=true]:text-accent-foreground'
                />
              }
            >
              <Blend className='size-4' />
            </TooltipTrigger>
          }
        />
        <TooltipContent side='right'>{t('tools.brushRampTitle')}</TooltipContent>
      </Tooltip>
      <PopoverContent
        side='right'
        align='center'
        sideOffset={8}
        className='w-56 gap-2.5 rounded-xl p-2.5'
      >
        <PopoverTitle className='text-[11px]'>{t('tools.brushRampTitle')}</PopoverTitle>
        <div
          className='grid grid-cols-3 gap-1'
          role='radiogroup'
          aria-label={t('tools.brushRampTitle')}
        >
          {RAMP_KINDS.map((kind) => {
            const active = kind === 'none' ? ramp === null : ramp?.kind === kind
            return (
              <button
                key={kind}
                type='button'
                role='radio'
                aria-label={t(`tools.brushRamp.${RAMP_KIND_KEYS[kind]}`)}
                aria-checked={active}
                className='h-7 rounded-md border border-input px-2 text-[10px] text-muted-foreground data-[active=true]:bg-accent data-[active=true]:text-accent-foreground'
                data-active={active}
                onClick={() => setKind(kind)}
              >
                {t(`tools.brushRamp.${RAMP_KIND_KEYS[kind]}`)}
              </button>
            )
          })}
        </div>
        {ramp !== null && (
          <>
            <div className='flex items-center justify-between gap-3'>
              <PopoverTitle className='text-[11px]'>{t('tools.rampFrom')}</PopoverTitle>
              <ColorWell
                label={t('tools.rampFrom')}
                size='sm'
                value={colorToHex(ramp.from)}
                onChange={(color) => setBrush({ ...brush, ramp: { ...ramp, from: hexToColor(color) } })}
              />
            </div>
            <div className='flex items-center justify-between gap-3'>
              <PopoverTitle className='text-[11px]'>{t('tools.rampTo')}</PopoverTitle>
              <ColorWell
                label={t('tools.rampTo')}
                size='sm'
                value={colorToHex(ramp.to)}
                onChange={(color) => setBrush({ ...brush, ramp: { ...ramp, to: hexToColor(color) } })}
              />
            </div>
            {ramp.kind === 'fixed_axis' && (
              <div
                className='flex items-center justify-between gap-3'
                role='radiogroup'
                aria-label={t('tools.rampAxisTitle')}
              >
                <PopoverTitle className='text-[11px]'>{t('tools.rampAxisTitle')}</PopoverTitle>
                <div className='flex items-center gap-1'>
                  {RAMP_AXES.map((axis) => (
                    <button
                      key={axis}
                      type='button'
                      role='radio'
                      aria-label={t(`tools.rampAxis.${axis}`)}
                      aria-checked={ramp.axis === axis}
                      className='h-6 rounded-md border border-input px-2 text-[10px] text-muted-foreground data-[active=true]:bg-accent data-[active=true]:text-accent-foreground'
                      data-active={ramp.axis === axis}
                      onClick={() => setBrush({ ...brush, ramp: { ...ramp, axis } })}
                    >
                      {t(`tools.rampAxis.${axis}`)}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </PopoverContent>
    </Popover>
  )
}

function defaultStroke(): NonNullable<ShapeStyle['stroke']> {
  return {
    color: [0, 0, 0, 255],
    width: DEFAULT_STROKE_WIDTH,
    dash: null,
    cap: 'butt',
    join: 'miter',
  }
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
