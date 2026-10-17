'use client'

import { useQuery } from '@tanstack/react-query'
import { observeElementRect, useVirtualizer } from '@tanstack/react-virtual'
import {
  Columns2,
  FilePlus2,
  Flag,
  FolderOpen,
  LoaderCircle,
  MoreHorizontal,
  Search,
  Settings,
  Trash2,
} from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState, type DragEvent } from 'react'
import { useTranslation } from 'react-i18next'

import { ResourceMonitor } from '@/components/editor/ResourceMonitor'
import { usePageImportDrop } from '@/components/editor/usePageImportDrop'
import { call } from '@/lib/backend'
import { useForkPreferences } from '@/lib/forkPreferences'
import {
  pageKey,
  pagesKey,
  preparedPageKey,
  projectKey,
  queryClient,
  refresh,
  useImportDroppedPages,
  useImportPages,
  usePage,
  usePages,
} from '@/lib/queries'
import { useKoharuStore } from '@/lib/store'
import { prefetchCanvasPages, showCanvasPage } from '@koharu/bridge/canvas'
import {
  commands,
  type CanvasPagePreparation,
  type Page,
  type PageImportSource,
  type PageSummary,
  type ProjectInfo,
} from '@koharu/bridge/protocol'
import { Button } from '@koharu/ui/components/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@koharu/ui/components/dialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@koharu/ui/components/dropdown-menu'
import { Input } from '@koharu/ui/components/input'
import { ScrollArea } from '@koharu/ui/components/scroll-area'
import { Tooltip, TooltipContent, TooltipTrigger } from '@koharu/ui/components/tooltip'
import { cn } from '@koharu/ui/lib/utils'

const emptyPages: PageSummary[] = []

interface IntentPrefetchState {
  project: string
  revision: number
  pages: Set<string>
}

type DropPosition = 'before' | 'after'

export function PageRail() {
  const { t } = useTranslation()
  const pages = usePages().data ?? emptyPages
  const active = usePage().data?.id ?? null
  const selected = useKoharuStore((state) => state.selectedPages)
  const selectPages = useKoharuStore((state) => state.selectPages)
  const selectLayers = useKoharuStore((state) => state.selectLayers)
  const mergeDeleteOriginals = useForkPreferences((state) => state.preferences.mergeDeleteOriginals)
  const setSettingsOpen = useKoharuStore((state) => state.setSettingsOpen)
  const { importPages, importing } = useImportPages()
  const { importDropped } = useImportDroppedPages()
  const railRef = useRef<HTMLElement>(null)
  const { overDropZone, dropHandlers } = usePageImportDrop(railRef, importDropped)
  const anchor = useRef<number | null>(null)
  const selectionRequest = useRef(0)
  const intentPrefetch = useRef<IntentPrefetchState | null>(null)
  const keyboardPageIndex = useRef<number | null>(null)
  const draggedPage = useRef<string | null>(null)
  const dropTargetRef = useRef<{ page: string; position: DropPosition } | null>(null)
  const dropHandled = useRef(false)
  const [dragged, setDragged] = useState<string | null>(null)
  const [dropTarget, setDropTarget] = useState<{ page: string; position: DropPosition } | null>(
    null,
  )
  const [query, setQuery] = useState('')
  const [renaming, setRenaming] = useState<PageSummary | null>(null)
  const [renameValue, setRenameValue] = useState('')
  const normalized = query.trim().toLocaleLowerCase()
  const visiblePages = useMemo(
    () =>
      pages
        .map((page, index) => ({ page, index }))
        .filter(({ page }) => !normalized || page.label.toLocaleLowerCase().includes(normalized)),
    [normalized, pages],
  )
  const pageList = useRef<HTMLDivElement>(null)
  const pageVirtualizer = useVirtualizer({
    count: visiblePages.length,
    getScrollElement: () => pageList.current,
    getItemKey: (index) => visiblePages[index]?.page.id ?? index,
    estimateSize: () => 72,
    gap: 2,
    overscan: 6,
    initialRect: { width: 240, height: 600 },
    observeElementRect: (instance, callback) =>
      observeElementRect(instance, (rect) =>
        callback({ width: rect.width || 240, height: rect.height || 600 }),
      ),
  })

  useEffect(
    () => () => {
      selectionRequest.current += 1
    },
    [],
  )

  const select = useCallback(
    (index: number, additive: boolean, range: boolean) => {
      const page = pages[index]
      if (!page) return
      let next: string[]
      if (range && anchor.current !== null) {
        const start = Math.min(anchor.current, index)
        const end = Math.max(anchor.current, index)
        const rangeIds = pages.slice(start, end + 1).map((item) => item.id)
        next = additive ? [...new Set([...selected, ...rangeIds])] : rangeIds
      } else if (additive) {
        next = selected.includes(page.id)
          ? selected.filter((id) => id !== page.id)
          : [...selected, page.id]
        anchor.current = index
      } else {
        next = [page.id]
        anchor.current = index
      }
      const previousProject = queryClient.getQueryData<ProjectInfo | null>(projectKey)
      const previousPage = queryClient.getQueryData<Page | null>(pageKey)
      const prepared = queryClient.getQueryData<CanvasPagePreparation>(preparedPageKey(page.id))
      const activated = showCanvasPage(page.id, previousProject?.revision ?? null)
      const request = ++selectionRequest.current
      const synchronize = () => {
        if (selectionRequest.current !== request) return
        if (activated && previousProject && prepared?.revision === previousProject.revision) {
          queryClient.setQueryData(projectKey, { ...previousProject, active_page: page.id })
          queryClient.setQueryData(pageKey, prepared.page)
        }
        selectPages(next)
        selectLayers([])
        void call(commands.selectPage, page.id)
          .then((selection) => {
            if (selectionRequest.current !== request) return
            queryClient.setQueryData(projectKey, selection.project)
            queryClient.setQueryData(pageKey, selection.page)
          })
          .catch(() => {
            if (selectionRequest.current !== request) return
            if (queryClient.getQueryData<ProjectInfo | null>(projectKey)?.active_page === page.id) {
              queryClient.setQueryData(projectKey, previousProject)
              queryClient.setQueryData(pageKey, previousPage)
            }
          })
      }
      if (activated) {
        requestAnimationFrame(() => window.setTimeout(synchronize, 0))
      } else {
        synchronize()
      }
    },
    [pages, selected, selectLayers, selectPages],
  )

  useEffect(() => {
    keyboardPageIndex.current = pages.findIndex((page) => page.id === active)
  }, [active, pages])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (
        (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') ||
        event.defaultPrevented ||
        event.altKey ||
        event.ctrlKey ||
        event.metaKey ||
        event.shiftKey
      )
        return

      const target = event.target
      if (
        target instanceof HTMLElement &&
        (target.isContentEditable ||
          target.closest(
            'input, textarea, select, [role="dialog"], [role="menu"], [role="listbox"], [role="slider"]',
          ))
      )
        return

      const current = keyboardPageIndex.current ?? pages.findIndex((page) => page.id === active)
      const next = current + (event.key === 'ArrowLeft' ? -1 : 1)
      if (current < 0 || !pages[next]) return

      event.preventDefault()
      keyboardPageIndex.current = next
      pageVirtualizer.scrollToIndex(next, { align: 'auto' })
      select(next, false, false)
    }

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [active, pageVirtualizer, pages, select])

  const prefetchOnIntent = (page: string) => {
    const project = queryClient.getQueryData<ProjectInfo | null>(projectKey)
    if (!project || project.active_page === page) return
    let state = intentPrefetch.current
    if (!state || state.project !== project.name || state.revision !== project.revision) {
      state = { project: project.name, revision: project.revision, pages: new Set() }
      intentPrefetch.current = state
    }
    const prepared = queryClient.getQueryData<CanvasPagePreparation>(preparedPageKey(page))
    if (prepared?.revision === project.revision || state.pages.has(page)) return
    state.pages.add(page)
    void prefetchCanvasPages([page])
      .then((pages) => {
        const current = queryClient.getQueryData<ProjectInfo | null>(projectKey)
        const preparedPage = pages.find(
          (candidate) => candidate.page.id === page && candidate.revision === project.revision,
        )
        if (
          preparedPage &&
          current?.name === project.name &&
          current.revision === preparedPage.revision
        ) {
          queryClient.setQueryData(preparedPageKey(page), preparedPage)
          return
        }
        if (intentPrefetch.current === state) state.pages.delete(page)
      })
      .catch(() => {
        if (intentPrefetch.current === state) state.pages.delete(page)
      })
  }

  const deletePage = (page: string) =>
    void call(commands.deletePages, [page])
      .then(() => {
        selectPages(selected.filter((selectedPage) => selectedPage !== page))
        if (active === page) selectLayers([])
        return refresh(projectKey, pagesKey, pageKey)
      })
      .catch(() => undefined)

  const mergePages = () => {
    // `pages`는 이미 프로젝트 순서(레일의 왼쪽→오른쪽)이므로 걸러내기만 하면
    // 합칠 순서가 나온다. 서버도 인접 여부를 다시 확인한다.
    const ordered = pages.filter((page) => selected.includes(page.id)).map((page) => page.id)
    if (ordered.length < 2) return
    void call(commands.mergePages, ordered, mergeDeleteOriginals)
      .then((merged) => {
        selectPages([merged])
        selectLayers([])
        return refresh(projectKey, pagesKey, pageKey)
      })
      .catch(() => undefined)
  }

  const openRename = (page: PageSummary) => {
    setRenaming(page)
    setRenameValue(page.label)
  }

  const closeRename = () => {
    setRenaming(null)
    setRenameValue('')
  }

  const submitRename = () => {
    const page = renaming
    const label = renameValue.trim()
    if (!page || !label || label === page.label) {
      closeRename()
      return
    }
    void call(commands.renamePage, page.id, label)
      .then(() => refresh(projectKey, pagesKey, pageKey))
      .catch(() => undefined)
      .finally(closeRename)
  }

  const setChapterStart = (page: PageSummary) =>
    void call(commands.setPageChapterStart, page.id, !page.chapter_start)
      .then(() => refresh(projectKey, pagesKey))
      .catch(() => undefined)

  const handleMovePage = useCallback(
    (movingId: string | null, targetId: string | null, position: DropPosition | null) => {
      if (!movingId || !targetId || !position || movingId === targetId) return
      const movingIndex = pages.findIndex((item) => item.id === movingId)
      const targetIndexInPages = pages.findIndex((item) => item.id === targetId)
      if (movingIndex < 0 || targetIndexInPages < 0) return

      const targetIndex = targetIndexInPages - (movingIndex < targetIndexInPages ? 1 : 0)
      const destination = targetIndex + (position === 'after' ? 1 : 0)
      void call(commands.movePage, movingId, destination)
        .finally(() => refresh(projectKey, pagesKey))
        .catch(() => undefined)
    },
    [pages],
  )

  const cleanupDrag = useCallback(() => {
    draggedPage.current = null
    dropTargetRef.current = null
    dropHandled.current = false
    setDragged(null)
    setDropTarget(null)
  }, [])

  return (
    <>
      <aside
        ref={railRef}
        data-drop-active={overDropZone}
        className={cn(
          'relative flex h-full min-h-0 flex-col bg-[var(--surface-sidebar)]',
          overDropZone && 'ring-2 ring-inset ring-primary',
        )}
        {...dropHandlers}
      >
        {overDropZone && (
          <div className='pointer-events-none absolute inset-0 z-20 grid place-items-center bg-primary/[0.06] text-[11px] font-medium text-foreground'>
            {t('navigator.dropToImport')}
          </div>
        )}
        <header className='flex h-10 shrink-0 items-center px-2.5'>
          <div className='flex min-w-0 items-center gap-2'>
            <h2 className='text-[11px] font-semibold'>{t('navigator.pages')}</h2>
            <span className='rounded-full bg-primary/[0.07] px-1.5 py-0.5 text-[9px] text-muted-foreground tabular-nums'>
              {pages.length}
            </span>
          </div>
          {/* One selected page behaves exactly like acting on the active
              page, so saying so would be noise. */}
          {selected.length > 1 && (
            <span
              role='status'
              aria-live='polite'
              className='ml-auto text-[9px] text-muted-foreground tabular-nums'
            >
              {t('navigator.selected', { count: selected.length })}
            </span>
          )}
        </header>

        {pages.length > 0 && (
          <div className='border-b px-2 py-1.5'>
            <label className='flex h-6 items-center gap-1.5 rounded-md border border-input bg-background/70 px-1.5'>
              <Search className='size-3 text-muted-foreground' />
              <Input
                value={query}
                aria-label={t('navigator.filter')}
                placeholder={t('navigator.filter')}
                className='h-5 min-w-0 border-0 bg-transparent p-0 text-[11px] shadow-none focus-visible:ring-0 md:text-[11px]'
                onChange={(event) => setQuery(event.currentTarget.value)}
              />
            </label>
          </div>
        )}

        {pages.length === 0 ? (
          <div className='flex min-h-0 flex-1 flex-col items-center justify-center px-4 text-center'>
            <p className='text-[11px] font-medium'>{t('navigator.emptyTitle')}</p>
            <p className='mt-1 text-[10px] leading-4 text-muted-foreground'>
              {t('navigator.emptyDescription')}
            </p>
            <div className='mt-3'>
              <PageImportMenu importing={importing} onImport={importPages} />
            </div>
          </div>
        ) : visiblePages.length === 0 ? (
          <div className='grid flex-1 place-items-center px-4 text-center text-[10px] text-muted-foreground'>
            {t('navigator.noResults')}
          </div>
        ) : (
          <ScrollArea
            className='min-h-0 flex-1'
            viewportClassName='px-1.5 py-1.5'
            viewportRef={pageList}
          >
            <div
              className='relative w-full'
              style={{ height: pageVirtualizer.getTotalSize() }}
              onDragOver={(event) => {
                if (draggedPage.current && event.target === event.currentTarget) {
                  event.preventDefault()
                  event.dataTransfer.dropEffect = 'move'
                  const lastPage = visiblePages[visiblePages.length - 1]?.page
                  if (lastPage && lastPage.id !== draggedPage.current) {
                    dropTargetRef.current = { page: lastPage.id, position: 'after' }
                    setDropTarget({ page: lastPage.id, position: 'after' })
                  }
                }
              }}
              onDrop={(event) => {
                if (
                  event.target === event.currentTarget &&
                  !dropHandled.current &&
                  dropTargetRef.current &&
                  draggedPage.current
                ) {
                  event.preventDefault()
                  dropHandled.current = true
                  handleMovePage(
                    draggedPage.current,
                    dropTargetRef.current.page,
                    dropTargetRef.current.position,
                  )
                  cleanupDrag()
                }
              }}
            >
              {pageVirtualizer.getVirtualItems().map((virtualRow) => {
                const item = visiblePages[virtualRow.index]
                if (!item) return null
                const { page, index } = item
                return (
                  <div
                    key={virtualRow.key}
                    data-index={virtualRow.index}
                    className='absolute top-0 left-0 w-full'
                    style={{ transform: `translateY(${virtualRow.start}px)` }}
                  >
                    <PageItem
                      page={page}
                      index={index}
                      active={active === page.id}
                      selected={selected.includes(page.id)}
                      dragged={dragged === page.id}
                      isDragging={dragged !== null}
                      dropPosition={dropTarget?.page === page.id ? dropTarget.position : null}
                      onIntent={active === page.id ? undefined : () => prefetchOnIntent(page.id)}
                      onSelect={(additive, range) => select(index, additive, range)}
                      onDragStart={(event: DragEvent<HTMLElement>) => {
                        draggedPage.current = page.id
                        dropTargetRef.current = null
                        dropHandled.current = false
                        event.dataTransfer.effectAllowed = 'move'
                        event.dataTransfer.setData('text/plain', page.id)
                        setDragged(page.id)
                      }}
                      onDragEnd={() => {
                        if (!dropHandled.current && draggedPage.current && dropTargetRef.current) {
                          dropHandled.current = true
                          handleMovePage(
                            draggedPage.current,
                            dropTargetRef.current.page,
                            dropTargetRef.current.position,
                          )
                        }
                        cleanupDrag()
                      }}
                      onDragOver={(position) => {
                        if (draggedPage.current && draggedPage.current !== page.id) {
                          dropTargetRef.current = { page: page.id, position }
                          setDropTarget({ page: page.id, position })
                        }
                      }}
                      onRename={() => openRename(page)}
                      onDelete={() => deletePage(page.id)}
                      onToggleChapterStart={() => setChapterStart(page)}
                      canMerge={selected.length > 1}
                      onMerge={mergePages}
                      onDrop={(position, dataTransfer) => {
                        const moving = draggedPage.current ?? dataTransfer.getData('text/plain')
                        if (!dropHandled.current && moving && moving !== page.id) {
                          dropHandled.current = true
                          handleMovePage(moving, page.id, position)
                        }
                        cleanupDrag()
                      }}
                    />
                  </div>
                )
              })}
            </div>
            <div className='px-0.5 pt-1.5 pb-0.5'>
              <PageImportMenu
                importing={importing}
                onImport={importPages}
                className='w-full justify-center'
              />
            </div>
          </ScrollArea>
        )}

        <div className='mt-auto flex shrink-0 items-center gap-2 px-2 py-2'>
          <Tooltip>
            <TooltipTrigger
              render={
                <Button
                  type='button'
                  variant='ghost'
                  size='icon'
                  className='size-8 text-muted-foreground hover:bg-foreground/[0.06] hover:text-foreground'
                  aria-label={t('menu.settings')}
                  onClick={() => setSettingsOpen(true)}
                />
              }
            >
              <Settings className='size-4' />
            </TooltipTrigger>
            <TooltipContent side='right'>{t('menu.settings')}</TooltipContent>
          </Tooltip>
          <ResourceMonitor />
        </div>
      </aside>

      <Dialog
        open={renaming !== null}
        onOpenChange={(open) => {
          if (!open) closeRename()
        }}
      >
        <DialogContent className='sm:max-w-md'>
          <form
            onSubmit={(event) => {
              event.preventDefault()
              submitRename()
            }}
          >
            <DialogHeader>
              <DialogTitle>{t('navigator.renameTitle')}</DialogTitle>
              <DialogDescription>{t('navigator.renameDescription')}</DialogDescription>
            </DialogHeader>
            <Input
              autoFocus
              value={renameValue}
              aria-label={t('navigator.pageName')}
              className='mt-4'
              onChange={(event) => setRenameValue(event.currentTarget.value)}
            />
            <DialogFooter className='mt-4'>
              <Button type='button' variant='ghost' onClick={closeRename}>
                {t('common.cancel')}
              </Button>
              <Button type='submit' disabled={!renameValue.trim()}>
                {t('navigator.rename')}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </>
  )
}

function PageImportMenu({
  importing,
  onImport,
  className,
}: {
  importing: boolean
  onImport: (source: PageImportSource) => void
  className?: string
}) {
  const { t } = useTranslation()
  const icon = importing ? (
    <LoaderCircle className='size-3.5 animate-spin' />
  ) : (
    <FilePlus2 className='size-3.5' />
  )

  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        render={
          <Button
            size='sm'
            variant='outline'
            disabled={importing}
            aria-busy={importing}
            className={cn('rounded-lg text-[10px]', className)}
          />
        }
      >
        {icon}
        {importing ? t('navigator.importingAction') : t('navigator.importAction')}
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align='start'
        className='w-auto min-w-20 border border-border/50 p-0.5 shadow-sm ring-0'
      >
        <DropdownMenuItem
          disabled={importing}
          className='min-h-7 gap-1 px-1.5 py-0.5 text-[11px] [&_svg:not([class*="size-"])]:size-3.5'
          onClick={() => onImport('files')}
        >
          <FilePlus2 />
          {t('navigator.importFiles')}
        </DropdownMenuItem>
        <DropdownMenuItem
          disabled={importing}
          className='min-h-7 gap-1 px-1.5 py-0.5 text-[11px] [&_svg:not([class*="size-"])]:size-3.5'
          onClick={() => onImport('folder')}
        >
          <FolderOpen />
          {t('navigator.importFolder')}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function PageItem({
  page,
  index,
  active,
  selected,
  dragged,
  dropPosition,
  isDragging,
  onIntent,
  onSelect,
  onDragStart,
  onDragEnd,
  onDragOver,
  onRename,
  onDelete,
  onToggleChapterStart,
  onMerge,
  canMerge,
  onDrop,
}: {
  page: PageSummary
  index: number
  active: boolean
  selected: boolean
  dragged: boolean
  dropPosition: DropPosition | null
  isDragging?: boolean
  onIntent?: () => void
  onSelect: (additive: boolean, range: boolean) => void
  onDragStart: (event: DragEvent<HTMLElement>) => void
  onDragEnd: () => void
  onDragOver: (position: DropPosition) => void
  onRename: () => void
  onDelete: () => void
  onToggleChapterStart: () => void
  onMerge: () => void
  canMerge: boolean
  onDrop: (position: DropPosition, dataTransfer: DataTransfer) => void
}) {
  const { t } = useTranslation()
  const showPageNumbers = useForkPreferences((state) => state.preferences.showPageNumbers)

  return (
    <article
      draggable
      data-active={active}
      data-selected={selected}
      data-chapter-start={page.chapter_start}
      className={cn(
        'group relative grid cursor-grab grid-cols-[48px_minmax(0,1fr)] gap-2.5 rounded-xl p-1.5 transition-colors select-none active:cursor-grabbing',
        active
          ? 'bg-primary/[0.09] hover:bg-primary/[0.09]'
          : selected
            ? 'bg-foreground/[0.06] hover:bg-foreground/[0.08]'
            : 'hover:bg-foreground/[0.045]',
        dragged && 'opacity-50',
        dropPosition === 'before' &&
          'before:absolute before:inset-x-1 before:-top-px before:z-10 before:border-t-2 before:border-primary',
        dropPosition === 'after' &&
          'after:absolute after:inset-x-1 after:-bottom-px after:z-10 after:border-t-2 after:border-primary',
      )}
      onPointerEnter={onIntent}
      onFocus={onIntent}
      onClick={(event) => {
        if ((event.target as HTMLElement).closest('button,[role="menuitem"]')) return
        onSelect(event.ctrlKey || event.metaKey, event.shiftKey)
      }}
      onDoubleClick={onRename}
      onDragStart={onDragStart}
      onDragEnd={onDragEnd}
      onDragEnter={(event) => {
        event.preventDefault()
        event.dataTransfer.dropEffect = 'move'
      }}
      onDragOver={(event) => {
        event.preventDefault()
        event.dataTransfer.dropEffect = 'move'
        const bounds = event.currentTarget.getBoundingClientRect()
        onDragOver(event.clientY < bounds.top + bounds.height / 2 ? 'before' : 'after')
      }}
      onDrop={(event) => {
        event.preventDefault()
        const bounds = event.currentTarget.getBoundingClientRect()
        const position = event.clientY < bounds.top + bounds.height / 2 ? 'before' : 'after'
        onDrop(position, event.dataTransfer)
      }}
    >
      <div
        className={cn(
          'grid h-16 w-12 place-items-center overflow-hidden rounded-lg bg-[var(--surface-well)]',
          isDragging && 'pointer-events-none',
        )}
      >
        {page.source_asset ? (
          <PageThumbnail page={page.id} asset={page.source_asset} label={page.label} />
        ) : (
          <span className='text-[9px] text-muted-foreground'>{t('navigator.noImage')}</span>
        )}
      </div>
      <div className={cn('min-w-0 py-0.5', isDragging && 'pointer-events-none')}>
        <div className='flex items-start gap-1'>
          <span className='flex min-w-0 flex-1 items-center gap-1 truncate text-[10px] font-medium'>
            {page.chapter_start && (
              <span
                role='img'
                aria-label={t('navigator.chapterStart')}
                title={t('navigator.chapterStart')}
                className='shrink-0 text-primary'
              >
                <Flag className='size-3' />
              </span>
            )}
            <span className='truncate'>{page.label}</span>
          </span>
          <DropdownMenu>
            <DropdownMenuTrigger
              render={
                <Button
                  variant='ghost'
                  size='icon-xs'
                  aria-label={t('navigator.actionsFor', { page: page.label })}
                  className='-mt-1 shrink-0 opacity-0 shadow-none group-hover:opacity-100 aria-expanded:opacity-100'
                />
              }
            >
              <MoreHorizontal />
            </DropdownMenuTrigger>
            <DropdownMenuContent align='end'>
              <DropdownMenuItem disabled={!canMerge} onClick={onMerge}>
                <Columns2 /> {t('navigator.mergeAsSpread')}
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={onToggleChapterStart}>
                <Flag />
                {page.chapter_start
                  ? t('navigator.removeChapterStart')
                  : t('navigator.markChapterStart')}
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={onRename}>{t('navigator.rename')}</DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem variant='destructive' onClick={onDelete}>
                <Trash2 /> {t('navigator.delete')}
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
        {/* Labels can be renamed to anything; this is the page's position. */}
        <p className='mt-1 text-[9px] leading-3.5 text-muted-foreground'>
          {showPageNumbers && (
            <>
              <span
                data-page-number={index + 1}
                className='font-medium text-foreground tabular-nums'
              >
                {index + 1}
              </span>
              {' · '}
            </>
          )}
          {t('navigator.layerCount', { count: page.layer_count })}
        </p>
        <p className='mt-1 text-[9px] text-muted-foreground tabular-nums'>
          {page.size.width} × {page.size.height}
        </p>
      </div>
    </article>
  )
}

function PageThumbnail({ page, asset, label }: { page: string; asset: string; label: string }) {
  const { t } = useTranslation()
  const [source, setSource] = useState<string | null>(null)
  const [settled, setSettled] = useState(false)

  useEffect(() => {
    setSettled(false)
    const timeout = window.setTimeout(() => setSettled(true), 100)
    return () => window.clearTimeout(timeout)
  }, [asset])

  const thumbnail = useQuery({
    queryKey: ['thumbnail', asset],
    queryFn: async () => new Uint8Array(await call(commands.getThumbnail, page)),
    enabled: settled,
    staleTime: Number.POSITIVE_INFINITY,
    notifyOnChangeProps: ['data'],
  }).data

  useEffect(() => {
    if (!thumbnail) return
    const url = URL.createObjectURL(new Blob([thumbnail.buffer], { type: 'image/webp' }))
    setSource(url)
    return () => {
      URL.revokeObjectURL(url)
    }
  }, [thumbnail])

  return (
    <div className='grid size-full place-items-center'>
      {source ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={source} alt={label} draggable={false} className='size-full object-contain' />
      ) : (
        <span className='text-[9px] text-muted-foreground'>{t('common.loading')}</span>
      )}
    </div>
  )
}
