import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

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

  it('resolves every shape key the editor asks for to a string', async () => {
    vi.resetModules()
    const { default: i18n } = await import('@/lib/i18n')
    const kinds = ['rectangle', 'ellipse', 'line', 'arrow', 'polygon', 'freeform']
    const keys = [
      'tools.shape',
      'inspector.shape.fill',
      'inspector.shape.stroke',
      'inspector.shape.strokeWidth',
      'inspector.shape.dash',
      'inspector.shape.rasterize',
      'inspector.shape.kindTitle',
      ...kinds.map((kind) => `inspector.shape.kind.${kind}`),
    ]

    for (const language of i18n.languages) {
      for (const key of keys) {
        // A branch key makes i18next return its "returned an object instead of
        // string" diagnostic, which is what a user would then read on screen.
        expect(i18n.t(key, { lng: language }), `${language}: ${key}`).not.toMatch(
          /returned an object instead of string/,
        )
      }
      expect(
        i18n.t('inspector.shape.kindLabel', { lng: language, kind: 'x' }),
        language,
      ).toContain('x')
      expect(i18n.t('inspector.shape.kindTitle', { lng: language }), language).not.toMatch(/\{\{/)
    }
  })
})
