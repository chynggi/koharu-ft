'use client'

import { Bot, Search, SlidersHorizontal } from 'lucide-react'
import { useTranslation } from 'react-i18next'

import { AgentPanel } from '@/components/editor/AgentPanel'
import { FindPanel } from '@/components/editor/FindPanel'
import { Inspector } from '@/components/editor/Inspector'
import { useKoharuStore } from '@/lib/store'
import { Button } from '@koharu/ui/components/button'

export function RightSidebar() {
  const { t } = useTranslation()
  const panel = useKoharuStore((state) => state.sidebarPanel)
  const setPanel = useKoharuStore((state) => state.setSidebarPanel)
  const tabs = [
    ['properties', SlidersHorizontal, t('agent.properties')],
    ['find', Search, t('find.title')],
    ['agent', Bot, t('agent.title')],
  ] as const

  return (
    <aside className='flex h-full min-h-0 flex-col bg-[var(--surface-panel)]'>
      <div className='flex h-10 shrink-0 items-center gap-1 border-b border-border/80 px-2.5'>
        {tabs.map(([id, Icon, label]) => (
          <Button
            key={id}
            variant={panel === id ? 'secondary' : 'ghost'}
            size='sm'
            className='h-7 flex-1 text-[10px]'
            onClick={() => setPanel(id)}
          >
            <Icon className='size-3' /> {label}
          </Button>
        ))}
      </div>
      <div className={panel === 'agent' ? 'flex min-h-0 flex-1 flex-col' : 'hidden'}>
        <AgentPanel />
      </div>
      <div className={panel === 'find' ? 'min-h-0 flex-1' : 'hidden'}>
        <FindPanel />
      </div>
      <div className={panel === 'properties' ? 'min-h-0 flex-1' : 'hidden'}>
        <Inspector />
      </div>
    </aside>
  )
}
