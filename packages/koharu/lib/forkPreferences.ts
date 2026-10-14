'use client'

import { create } from 'zustand'

export const FORK_PREFERENCES_STORAGE_KEY = 'koharu.fork-preferences'

export interface ForkPreferences {
  mergeDeleteOriginals: boolean
  showPageNumbers: boolean
}

export const forkPreferenceDefaults: ForkPreferences = {
  mergeDeleteOriginals: true,
  showPageNumbers: true,
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
    }
  } catch {
    return { ...forkPreferenceDefaults }
  }
}

function persistForkPreferences(preferences: ForkPreferences): void {
  if (typeof window === 'undefined') return
  window.localStorage.setItem(FORK_PREFERENCES_STORAGE_KEY, JSON.stringify(preferences))
}
