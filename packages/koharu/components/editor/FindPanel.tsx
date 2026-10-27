'use client'

import { useQuery } from '@tanstack/react-query'
import { CaseSensitive, LoaderCircle, Replace, Search } from 'lucide-react'
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useTranslation } from 'react-i18next'

import { call } from '@/lib/backend'
import { activatePage } from '@/lib/navigation'
import { pageKey, pagesKey, projectKey, refresh, usePages, useProject } from '@/lib/queries'
import { useKoharuStore } from '@/lib/store'
import { commands, type TranslationMatch } from '@koharu/bridge/protocol'
import { Button } from '@koharu/ui/components/button'
import { Input } from '@koharu/ui/components/input'
import { ScrollArea } from '@koharu/ui/components/scroll-area'
import { Toggle } from '@koharu/ui/components/toggle'

const SEARCH_DELAY_MS = 200

/** Project-wide search and replace in translations. Results follow the
 * project revision, so they stay current after edits, replaces, and undo. */
export function FindPanel() {
  const { t } = useTranslation()
  const open = useKoharuStore((state) => state.sidebarPanel === 'find')
  const setSidebarPanel = useKoharuStore((state) => state.setSidebarPanel)
  const selectLayers = useKoharuStore((state) => state.selectLayers)
  const revision = useProject().data?.revision
  const pages = usePages().data
  const input = useRef<HTMLInputElement>(null)
  const [query, setQuery] = useState('')
  const [search, setSearch] = useState('')
  const [replacement, setReplacement] = useState('')
  const [matchCase, setMatchCase] = useState(false)
  const [replacing, setReplacing] = useState(false)

  useEffect(() => {
    if (open) input.current?.focus()
  }, [open])

  // Ctrl+F (Cmd+F on macOS) opens this tab from anywhere, or returns to its
  // search field when it is already open.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!(event.ctrlKey || event.metaKey) || event.altKey || event.key.toLowerCase() !== 'f')
        return
      event.preventDefault()
      setSidebarPanel('find')
      input.current?.focus()
      input.current?.select()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [setSidebarPanel])

  useEffect(() => {
    const timer = window.setTimeout(() => setSearch(query), SEARCH_DELAY_MS)
    return () => window.clearTimeout(timer)
  }, [query])

  const results = useQuery({
    queryKey: ['find-translations', search, matchCase, revision],
    queryFn: () => call(commands.findTranslations, search, matchCase),
    enabled: search.length > 0 && revision !== undefined,
  })
  const matches = useMemo(() => (search ? (results.data ?? []) : []), [search, results.data])
  const total = matches.reduce((sum, match) => sum + match.count, 0)
  const groups = useMemo(() => {
    const byPage = new Map<string, TranslationMatch[]>()
    for (const match of matches) {
      byPage.set(match.page, [...(byPage.get(match.page) ?? []), match])
    }
    return [...byPage.values()]
  }, [matches])

  const replace = (layers: string[] | null) => {
    if (!search || replacing) return
    setReplacing(true)
    void call(commands.replaceTranslations, search, replacement, matchCase, layers)
      .then(() => refresh(projectKey, pagesKey, pageKey))
      .catch(() => undefined)
      .finally(() => setReplacing(false))
  }

  const reveal = (match: TranslationMatch) => {
    void activatePage(match.page, [match.page]).then((opened) => {
      if (opened) selectLayers([match.layer])
    })
  }

  return (
    <div className='flex h-full min-h-0 flex-col gap-2 p-2.5 text-[11px]'>
      <div className='flex flex-col gap-1.5'>
        <div className='relative'>
          <Search className='pointer-events-none absolute top-1/2 left-2 size-3.5 -translate-y-1/2 text-muted-foreground' />
          <Input
            ref={input}
            value={query}
            placeholder={t('find.placeholder')}
            aria-label={t('find.placeholder')}
            className='h-8 pr-9 pl-7 text-[12px]'
            onChange={(event) => setQuery(event.currentTarget.value)}
          />
          <Toggle
            size='sm'
            pressed={matchCase}
            onPressedChange={setMatchCase}
            aria-label={t('find.matchCase')}
            title={t('find.matchCase')}
            className='absolute top-1/2 right-0.5 h-7 min-w-7 -translate-y-1/2 px-1.5'
          >
            <CaseSensitive />
          </Toggle>
        </div>
        <div className='flex items-center gap-1'>
          <Input
            value={replacement}
            placeholder={t('find.replacePlaceholder')}
            aria-label={t('find.replacePlaceholder')}
            className='h-8 min-w-0 flex-1 text-[12px]'
            onChange={(event) => setReplacement(event.currentTarget.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') replace(null)
            }}
          />
          <Button
            type='button'
            variant='secondary'
            size='sm'
            className='h-8 shrink-0 text-[11px]'
            disabled={total === 0 || replacing}
            aria-busy={replacing}
            onClick={() => replace(null)}
          >
            {replacing ? <LoaderCircle className='size-3.5 animate-spin' /> : null}
            {t('find.replaceAll')}
          </Button>
        </div>
      </div>

      {search && (
        <p className='px-0.5 text-[10px] text-muted-foreground' role='status'>
          {results.isFetching && !results.data
            ? t('find.searching')
            : total === 0
              ? t('find.noResults')
              : t('find.summary', { count: total, pages: groups.length })}
        </p>
      )}

      <ScrollArea className='min-h-0 flex-1'>
        <div className='flex flex-col gap-2 pb-2'>
          {groups.map((group) => {
            const first = group[0]!
            const label = pages?.find((page) => page.id === first.page)?.label
            return (
              <section key={first.page} className='flex flex-col gap-0.5'>
                <h3 className='truncate px-1 text-[10px] font-medium text-muted-foreground'>
                  {t('find.page', { number: first.page_index + 1 })}
                  {label ? ` · ${label}` : ''}
                </h3>
                {group.map((match) => (
                  <div
                    key={match.layer}
                    className='group flex items-start gap-1 rounded-md px-1 py-1 hover:bg-foreground/[0.05]'
                  >
                    <button
                      type='button'
                      className='min-w-0 flex-1 text-left text-[11px] leading-4 break-words text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring/25'
                      onClick={() => reveal(match)}
                    >
                      {highlight(match.text, search, matchCase)}
                    </button>
                    <Button
                      type='button'
                      variant='ghost'
                      size='icon-xs'
                      aria-label={t('find.replace')}
                      title={t('find.replace')}
                      className='shrink-0 opacity-0 group-focus-within:opacity-100 group-hover:opacity-100'
                      disabled={replacing}
                      onClick={() => replace([match.layer])}
                    >
                      <Replace />
                    </Button>
                  </div>
                ))}
              </section>
            )
          })}
        </div>
      </ScrollArea>
    </div>
  )
}

/** Marks each occurrence of `query` in `text`. Native search decides what
 * matches; this only finds the same spans again to show them. */
function highlight(text: string, query: string, matchCase: boolean): ReactNode[] {
  const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const parts = text.split(new RegExp(`(${escaped})`, matchCase ? 'gu' : 'giu'))
  return parts.map((part, index) =>
    index % 2 === 1 ? (
      <mark key={index} className='rounded-sm bg-primary/20 px-px text-foreground'>
        {part}
      </mark>
    ) : (
      part
    ),
  )
}
