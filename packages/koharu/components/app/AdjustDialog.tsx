'use client'

import { useState } from 'react'
import { useTranslation } from 'react-i18next'

import { dispatch } from '@/lib/backend'
import { adjustColors } from '@/lib/scene'
import type { ColorAdjust, LayerId } from '@koharu/ipc/protocol'
import { Button } from '@koharu/ui/components/button'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@koharu/ui/components/dialog'
import { Slider } from '@koharu/ui/components/slider'

/** The adjustments that ask for values, as Photoshop's Image > Adjustments dialogs. */
export type Adjustment = 'brightness_contrast' | 'levels' | 'hue'

/** Each value's label key, range, and starting value, by the adjustment asking for it. */
const fields: Record<Adjustment, { key: string; min: number; max: number; start: number }[]> = {
  brightness_contrast: [
    { key: 'brightness', min: -150, max: 150, start: 0 },
    { key: 'contrast', min: -100, max: 100, start: 0 },
  ],
  levels: [
    { key: 'levelsLow', min: 0, max: 254, start: 0 },
    { key: 'levelsHigh', min: 1, max: 255, start: 255 },
  ],
  hue: [{ key: 'hue', min: -180, max: 180, start: 0 }],
}

/** The adjustments the values make, in the order they run; none that would change nothing. */
function adjusts(adjustment: Adjustment, [first = 0, second = 0]: number[]): ColorAdjust[] {
  switch (adjustment) {
    case 'brightness_contrast':
      return [
        ...(first ? [{ kind: 'brightness' as const, delta: first }] : []),
        ...(second ? [{ kind: 'contrast' as const, percent: second }] : []),
      ]
    case 'levels':
      return first === 0 && second === 255 ? [] : [{ kind: 'levels', low: first, high: second }]
    case 'hue':
      return first ? [{ kind: 'hue', degrees: first }] : []
  }
}

/** Asks for an adjustment's values and adjusts `layers` with them in one undo step. */
export function AdjustDialog({
  adjustment,
  layers,
  onClose,
}: {
  adjustment: Adjustment | null
  layers: LayerId[]
  onClose: () => void
}) {
  return (
    <Dialog open={adjustment !== null} onOpenChange={(open) => !open && onClose()}>
      {adjustment && (
        <Values key={adjustment} adjustment={adjustment} layers={layers} onClose={onClose} />
      )}
    </Dialog>
  )
}

function Values({
  adjustment,
  layers,
  onClose,
}: {
  adjustment: Adjustment
  layers: LayerId[]
  onClose: () => void
}) {
  const { t } = useTranslation()
  const [values, setValues] = useState(fields[adjustment].map((field) => field.start))
  // Levels keep their low below their high, as Photoshop's input sliders do.
  const valid = adjustment !== 'levels' || values[0]! < values[1]!
  const chosen = adjusts(adjustment, values)
  return (
    <DialogContent className='sm:max-w-sm'>
      <DialogHeader>
        <DialogTitle>{t(`adjust.${adjustment}`)}</DialogTitle>
      </DialogHeader>
      <div className='flex flex-col gap-4 py-2'>
        {fields[adjustment].map((field, index) => (
          <div key={field.key} className='flex flex-col gap-1.5'>
            <div className='flex justify-between text-xs'>
              <span>{t(`adjust.${field.key}`)}</span>
              <span className='text-muted-foreground tabular-nums'>{values[index]}</span>
            </div>
            <Slider
              aria-label={t(`adjust.${field.key}`)}
              min={field.min}
              max={field.max}
              step={1}
              value={values[index]!}
              onValueChange={(next) =>
                setValues((current) => current.map((value, at) => (at === index ? next : value)))
              }
            />
          </div>
        ))}
      </div>
      <DialogFooter>
        <Button variant='ghost' onClick={onClose}>
          {t('common.cancel')}
        </Button>
        <Button
          disabled={!valid || chosen.length === 0}
          onClick={() => {
            dispatch(adjustColors, layers, chosen)
            onClose()
          }}
        >
          {t('adjust.apply')}
        </Button>
      </DialogFooter>
    </DialogContent>
  )
}
