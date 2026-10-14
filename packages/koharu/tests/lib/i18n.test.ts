import { readFileSync, readdirSync } from 'node:fs'
import path from 'node:path'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const locales = path.resolve(import.meta.dirname, '../../public/locales')

// Read the shipped files directly: under jsdom this repository's language
// detector resolves `i18n.languages` to `['en-US']`, so iterating the runtime
// instance would silently cover one locale while claiming all of them.
const languages = readdirSync(locales, { withFileTypes: true })
  .filter((entry) => entry.isDirectory())
  .map((entry) => entry.name)
  .sort()

function translation(language: string): Record<string, unknown> {
  const file = path.join(locales, language, 'translation.json')
  return JSON.parse(readFileSync(file, 'utf8')) as Record<string, unknown>
}

function lookup(source: Record<string, unknown>, key: string): unknown {
  return key.split('.').reduce<unknown>((node, segment) => {
    if (typeof node !== 'object' || node === null) return undefined
    return (node as Record<string, unknown>)[segment]
  }, source)
}

const values = new Map<string, string>()
const storage: Storage = {
  get length() {
    return values.size
  },
  clear: () => values.clear(),
  getItem: (key) => values.get(key) ?? null,
  key: (index) => [...values.keys()][index] ?? null,
  removeItem: (key) => values.delete(key),
  setItem: (key, value) => values.set(key, value),
}

describe('interface language persistence', () => {
  beforeEach(() => {
    Object.defineProperty(window, 'localStorage', { configurable: true, value: storage })
  })

  afterEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('stores an explicitly selected language', async () => {
    vi.resetModules()
    const { default: i18n } = await import('@/lib/i18n')

    await i18n.changeLanguage('ja-JP')

    expect(window.localStorage.getItem('i18nextLng')).toBe('ja-JP')
  })

  it('restores the saved language when i18n initializes', async () => {
    window.localStorage.setItem('i18nextLng', 'ja-JP')
    vi.resetModules()

    const { default: i18n } = await import('@/lib/i18n')

    expect(i18n.resolvedLanguage).toBe('ja-JP')
    expect(window.localStorage.getItem('i18nextLng')).toBe('ja-JP')
  })

  it('ships a translated string for every shape key the editor asks for', async () => {
    vi.resetModules()
    const { default: i18n } = await import('@/lib/i18n')
    const kinds = ['rectangle', 'ellipse', 'line', 'arrow']
    const fillKinds = ['solid', 'gradient']
    const gradientKinds = ['linear', 'radial']
    const rampKinds = ['none', 'alongStroke', 'fixedAxis']
    const rampAxes = ['horizontal', 'vertical']
    const keys = [
      'tools.shape',
      'tools.brushRampTitle',
      'tools.rampAxisTitle',
      'tools.rampFrom',
      'tools.rampTo',
      'tools.rampAxis.horizontal',
      'tools.rampAxis.vertical',
      'layers.kinds.shape',
      'inspector.shape.fill',
      'inspector.shape.stroke',
      'inspector.shape.strokeWidth',
      'inspector.shape.dash',
      'inspector.shape.rasterize',
      'inspector.shape.kindTitle',
      'inspector.shape.fillKindLabel',
      'inspector.shape.gradientKindLabel',
      'inspector.shape.stops',
      'inspector.shape.addStop',
      'inspector.shape.removeStop',
      'inspector.shape.stopsTooClose',
      ...kinds.map((kind) => `inspector.shape.kind.${kind}`),
      ...fillKinds.map((kind) => `inspector.shape.fillKind.${kind}`),
      ...gradientKinds.map((kind) => `inspector.shape.gradientKind.${kind}`),
      // `tools.brushRamp` is a branch key: the selector renders each leaf, and
      // the rail renders `tools.brushRampTitle`. Reaching for the branch itself
      // returns i18next's "object instead of string" diagnostic, so nothing may
      // render `tools.brushRamp` as a label.
      ...rampKinds.map((kind) => `tools.brushRamp.${kind}`),
      ...rampAxes.map((axis) => `tools.rampAxis.${axis}`),
    ]

    // The shipped translations are the source of truth; a missing key here is
    // invisible at runtime because i18next echoes the key back as a string.
    expect(languages).toHaveLength(9)
    for (const language of languages) {
      const source = translation(language)
      for (const key of keys) {
        const value = lookup(source, key)
        expect(typeof value, `${language}: ${key}`).toBe('string')
        expect(value, `${language}: ${key}`).not.toBe('')
      }
      const label = lookup(source, 'inspector.shape.kindLabel')
      // A branch key makes i18next return its "returned an object instead of
      // string" diagnostic, which is what a user would then read on screen.
      expect(typeof label, language).toBe('string')
      expect(label, language).toContain('{{kind}}')
      expect(lookup(source, 'inspector.shape.kindTitle'), language).not.toMatch(/\{\{/)
      // The stop editor names each row with its index; a translation that
      // dropped the placeholder would label every row identically.
      for (const key of ['inspector.shape.stopColor', 'inspector.shape.stopOffset']) {
        const value = lookup(source, key)
        expect(typeof value, `${language}: ${key}`).toBe('string')
        expect(value, `${language}: ${key}`).toContain('{{index}}')
      }
      // The stop limits are enforced by disabling the add and remove controls,
      // so no code path renders a message for them. These keys are deliberately
      // absent rather than shipped as dead translations; this assertion is what
      // keeps someone from re-adding them as unused strings.
      for (const key of ['inspector.shape.tooFewStops', 'inspector.shape.tooManyStops']) {
        expect(lookup(source, key), `${language}: ${key}`).toBeUndefined()
      }
    }

    // The runtime instance must also resolve each loaded language without
    // falling back to a diagnostic string.
    for (const language of languages) {
      for (const key of keys) {
        expect(i18n.t(key, { lng: language }), `${language}: ${key}`).not.toMatch(
          /returned an object instead of string/,
        )
      }
      expect(
        i18n.t('inspector.shape.kindLabel', { lng: language, kind: 'x' }),
        language,
      ).toContain('x')
      // A branch key has no string form; these two are the labels the rail
      // renders, and they must stay distinct from the branches beside them.
      for (const key of ['tools.brushRamp', 'tools.rampAxis']) {
        expect(i18n.t(key, { lng: language }), `${language}: ${key}`).toMatch(
          /returned an object instead of string/,
        )
      }
    }
  })
})
