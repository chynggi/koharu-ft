'use client'

import { create } from 'zustand'

export const FORK_PREFERENCES_STORAGE_KEY = 'koharu.fork-preferences'

/** Each id names a palette in app/themes.css; koharu is the default in globals.css. */
export const colorThemes = ['koharu', 'aniswipe', 'ocean', 'matcha'] as const
export type ColorTheme = (typeof colorThemes)[number]

export interface ForkPreferences {
  mergeDeleteOriginals: boolean
  showPageNumbers: boolean
  colorTheme: ColorTheme
  interfaceAnimations: boolean
}

export const forkPreferenceDefaults: ForkPreferences = {
  mergeDeleteOriginals: true,
  showPageNumbers: true,
  colorTheme: 'koharu',
  interfaceAnimations: true,
}

interface ForkPreferencesStore {
  preferences: ForkPreferences
  setPreference: <K extends keyof ForkPreferences>(key: K, value: ForkPreferences[K]) => void
}

export const useForkPreferences = create<ForkPreferencesStore>((set, get) => ({
  preferences: loadForkPreferences(),
  setPreference: (key, value) => {
    const preferences = { ...get().preferences, [key]: value }
    set({ preferences })
    persistForkPreferences(preferences)
  },
}))

// Palette and motion live on the root element so every surface, including the
// canvas workspace color, reads them from CSS. Applying them on load, before
// the first render, keeps the default palette from flashing at startup.
applyAppearance()
useForkPreferences.subscribe(applyAppearance)

function applyAppearance(): void {
  if (typeof document === 'undefined') return
  const { colorTheme, interfaceAnimations } = useForkPreferences.getState().preferences
  const root = document.documentElement
  const swapping = root.dataset.palette !== undefined && root.dataset.palette !== colorTheme
  root.dataset.palette = colorTheme
  if (!swapping || !interfaceAnimations) {
    root.dataset.motion = interfaceAnimations ? 'full' : 'still'
    return
  }
  // Swap the palette while still, so every surface changes at once instead of
  // cross-fading. Motion returns two frames later, after a frame has rendered
  // the new colors.
  root.dataset.motion = 'still'
  window.requestAnimationFrame(() =>
    window.requestAnimationFrame(() => {
      const animate = useForkPreferences.getState().preferences.interfaceAnimations
      root.dataset.motion = animate ? 'full' : 'still'
    }),
  )
}

function loadForkPreferences(): ForkPreferences {
  if (typeof window === 'undefined') return { ...forkPreferenceDefaults }
  try {
    const parsed: unknown = JSON.parse(
      window.localStorage.getItem(FORK_PREFERENCES_STORAGE_KEY) ?? '',
    )
    if (typeof parsed !== 'object' || parsed === null) return { ...forkPreferenceDefaults }
    const stored = parsed as Record<string, unknown>
    return {
      // Deliberate divergence from nekoharu 8af42854, which read this as
      // `=== true` and so defaulted to false: merging has always deleted the
      // originals in this tree, so an absent key must still mean "delete" or
      // existing projects get a silent behaviour change on the next merge.
      mergeDeleteOriginals: stored.mergeDeleteOriginals !== false,
      showPageNumbers: stored.showPageNumbers !== false,
      colorTheme: colorThemes.find((theme) => theme === stored.colorTheme) ?? 'koharu',
      interfaceAnimations: stored.interfaceAnimations !== false,
    }
  } catch {
    return { ...forkPreferenceDefaults }
  }
}

function persistForkPreferences(preferences: ForkPreferences): void {
  if (typeof window === 'undefined') return
  window.localStorage.setItem(FORK_PREFERENCES_STORAGE_KEY, JSON.stringify(preferences))
}
