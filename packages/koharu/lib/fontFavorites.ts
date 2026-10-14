'use client'

import { create } from 'zustand'

export const FONT_FAVORITES_STORAGE_KEY = 'koharu.font-favorites'

interface FontFavoritesStore {
  favorites: string[]
  toggle: (family: string) => void
}

export const useFontFavorites = create<FontFavoritesStore>((set, get) => ({
  favorites: loadFontFavorites(),
  toggle: (family) => {
    const favorites = get().favorites.includes(family)
      ? get().favorites.filter((entry) => entry !== family)
      : [...get().favorites, family]
    set({ favorites })
    persistFontFavorites(favorites)
  },
}))

function loadFontFavorites(): string[] {
  if (typeof window === 'undefined') return []
  try {
    const parsed: unknown = JSON.parse(
      window.localStorage.getItem(FONT_FAVORITES_STORAGE_KEY) ?? '',
    )
    if (!Array.isArray(parsed)) return []
    return [...new Set(parsed.filter((item): item is string => typeof item === 'string'))]
  } catch {
    return []
  }
}

function persistFontFavorites(favorites: string[]): void {
  if (typeof window === 'undefined') return
  window.localStorage.setItem(FONT_FAVORITES_STORAGE_KEY, JSON.stringify(favorites))
}
