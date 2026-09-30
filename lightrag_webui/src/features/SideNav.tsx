import { useTranslation } from 'react-i18next'
import { TabsList, TabsTrigger } from '@/components/ui/Tabs'
import Button from '@/components/ui/Button'
import { useSettingsStore } from '@/stores/settings'
import { cn } from '@/lib/utils'
import {
  BookOpenIcon,
  CpuIcon,
  DatabaseIcon,
  FileTextIcon,
  MessageSquareTextIcon,
  NetworkIcon,
  PanelLeftCloseIcon,
  PanelLeftOpenIcon,
  PlugIcon
} from 'lucide-react'

type NavItem = {
  value: string
  icon: typeof FileTextIcon
  labelKey: string
  fallback: string
}

const navGroups: { key: string; fallback: string; items: NavItem[] }[] = [
  {
    key: 'nav.workspace',
    fallback: 'Workspace',
    items: [
      { value: 'documents', icon: FileTextIcon, labelKey: 'header.documents', fallback: 'Documents' },
      { value: 'knowledge-graph', icon: NetworkIcon, labelKey: 'header.knowledgeGraph', fallback: 'Knowledge Graph' },
      { value: 'retrieval', icon: MessageSquareTextIcon, labelKey: 'header.retrieval', fallback: 'Retrieval' }
    ]
  },
  {
    key: 'nav.configuration',
    fallback: 'Configuration',
    items: [
      { value: 'knowledge-bases', icon: DatabaseIcon, labelKey: 'header.knowledgeBases', fallback: 'Knowledge Bases' },
      { value: 'models', icon: CpuIcon, labelKey: 'header.models', fallback: 'Models' },
      { value: 'external-kbs', icon: PlugIcon, labelKey: 'header.externalKbs', fallback: 'External KBs' },
      { value: 'api', icon: BookOpenIcon, labelKey: 'header.api', fallback: 'API' }
    ]
  }
]

/** Primary navigation. Kept as Radix tab triggers so every page stays mounted. */
export default function SideNav() {
  const { t } = useTranslation()
  const collapsed = useSettingsStore.use.sidebarCollapsed()
  const setCollapsed = useSettingsStore.use.setSidebarCollapsed()

  return (
    <nav className={cn('bg-card border-r py-2 shrink-0 flex flex-col gap-1', collapsed ? 'w-14' : 'w-52')}>
      {navGroups.map((group) => (
        <div key={group.key} className="flex flex-col gap-0.5">
          <span
            className={cn(
              'text-muted-foreground px-3 pt-2 pb-1 text-[10px] font-semibold uppercase tracking-wider',
              collapsed && 'text-center px-0'
            )}
          >
            {collapsed ? '·' : t(group.key, { defaultValue: group.fallback })}
          </span>
          <TabsList
            aria-orientation="vertical"
            className={cn('bg-transparent text-inherit inline-flex h-auto w-full flex-col items-stretch justify-start gap-0.5 rounded-none p-0')}
          >
            {group.items.map((item) => (
              <TabsTrigger
                key={item.value}
                value={item.value}
                className={cn(
                  'justify-start gap-2 rounded-none px-3 py-2 text-sm font-medium text-muted-foreground',
                  'data-[state=active]:bg-primary/10 data-[state=active]:text-foreground data-[state=active]:shadow-none',
                  'border-l-2 border-transparent data-[state=active]:border-primary',
                  collapsed && 'justify-center px-0'
                )}
                title={collapsed ? t(item.labelKey, { defaultValue: item.fallback }) : undefined}
              >
                <item.icon className="size-4 shrink-0" aria-hidden="true" />
                {!collapsed && <span className="truncate">{t(item.labelKey, { defaultValue: item.fallback })}</span>}
              </TabsTrigger>
            ))}
          </TabsList>
        </div>
      ))}

      <div className="mt-auto flex justify-center border-t pt-2">
        <Button
          variant="ghost"
          size="icon"
          className="size-8"
          side="right"
          tooltip={collapsed
            ? t('nav.expand', { defaultValue: 'Expand navigation' })
            : t('nav.collapse', { defaultValue: 'Collapse navigation' })}
          onClick={() => setCollapsed(!collapsed)}
        >
          {collapsed
            ? <PanelLeftOpenIcon className="size-4" aria-hidden="true" />
            : <PanelLeftCloseIcon className="size-4" aria-hidden="true" />}
        </Button>
      </div>
    </nav>
  )
}
