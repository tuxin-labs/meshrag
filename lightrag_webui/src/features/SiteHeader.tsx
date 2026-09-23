import { useEffect, useState } from 'react'
import Button from '@/components/ui/Button'
import { SiteInfo, webuiPrefix } from '@/lib/constants'
import AppSettings from '@/components/AppSettings'
import { TabsList, TabsTrigger } from '@/components/ui/Tabs'
import { useSettingsStore } from '@/stores/settings'
import { useBackendState, useAuthStore } from '@/stores/state'
import { useGraphStore } from '@/stores/graph'
import { cn, errorMessage } from '@/lib/utils'
import { useTranslation } from 'react-i18next'
import { navigationService } from '@/services/navigation'
import { createKnowledgeBase, listKnowledgeBases } from '@/api/lightrag'
import { ZapIcon, GithubIcon, LogOutIcon, PlusIcon } from 'lucide-react'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/Tooltip'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/Dialog'
import Input from '@/components/ui/Input'
import { toast } from 'sonner'

interface NavigationTabProps {
  value: string
  currentTab: string
  children: React.ReactNode
}

function NavigationTab({ value, currentTab, children }: NavigationTabProps) {
  return (
    <TabsTrigger
      value={value}
      className={cn(
        'cursor-pointer px-2 py-1 transition-all',
        currentTab === value ? '!bg-emerald-400 !text-zinc-50' : 'hover:bg-background/60'
      )}
    >
      {children}
    </TabsTrigger>
  )
}

function TabsNavigation() {
  const currentTab = useSettingsStore.use.currentTab()
  const { t } = useTranslation()

  return (
    <div className="flex h-8 self-center">
      <TabsList className="h-full gap-2">
        <NavigationTab value="documents" currentTab={currentTab}>
          {t('header.documents')}
        </NavigationTab>
        <NavigationTab value="knowledge-graph" currentTab={currentTab}>
          {t('header.knowledgeGraph')}
        </NavigationTab>
        <NavigationTab value="retrieval" currentTab={currentTab}>
          {t('header.retrieval')}
        </NavigationTab>
        <NavigationTab value="api" currentTab={currentTab}>
          {t('header.api')}
        </NavigationTab>
      </TabsList>
    </div>
  )
}

function KnowledgeBaseSwitcher() {
  const { t } = useTranslation()
  const selectedKbId = useSettingsStore.use.selectedKbId()
  const setSelectedKbId = useSettingsStore.use.setSelectedKbId()
  const availableKbIds = useSettingsStore.use.availableKbIds()
  const setAvailableKbIds = useSettingsStore.use.setAvailableKbIds()
  const setQueryLabel = useSettingsStore.use.setQueryLabel()
  const setRetrievalHistory = useSettingsStore.use.setRetrievalHistory()
  const triggerSearchLabelDropdownRefresh = useSettingsStore.use.triggerSearchLabelDropdownRefresh()
  const [isCreateOpen, setIsCreateOpen] = useState(false)
  const [isCreating, setIsCreating] = useState(false)
  const [newKbId, setNewKbId] = useState('')

  useEffect(() => {
    const loadKnowledgeBases = async () => {
      try {
        const kbIds = await listKnowledgeBases()
        setAvailableKbIds(kbIds)
      } catch (error) {
        console.error('Failed to load knowledge bases:', error)
      }
    }

    loadKnowledgeBases()
  }, [setAvailableKbIds])

  const handleKbChange = (kbId: string) => {
    if (kbId === selectedKbId) {
      return
    }

    setSelectedKbId(kbId)
    setQueryLabel('*')
    setRetrievalHistory([])
    useGraphStore.getState().reset()
    useGraphStore.getState().setGraphDataFetchAttempted(false)
    useGraphStore.getState().setLabelsFetchAttempted(false)
    useBackendState.getState().setPipelineBusy(false)
    triggerSearchLabelDropdownRefresh()
  }

  const handleCreateKnowledgeBase = async () => {
    const kbId = newKbId.trim()
    if (!kbId) {
      toast.error(t('header.kbCreateRequired', { defaultValue: 'Knowledge base ID is required' }))
      return
    }

    setIsCreating(true)
    try {
      await createKnowledgeBase(kbId)
      setAvailableKbIds(Array.from(new Set([...availableKbIds, kbId])))
      handleKbChange(kbId)
      setNewKbId('')
      setIsCreateOpen(false)
      toast.success(t('header.kbCreateSuccess', { defaultValue: 'Knowledge base created' }))
    } catch (error) {
      toast.error(errorMessage(error))
    } finally {
      setIsCreating(false)
    }
  }

  return (
    <>
      <div className="ml-4 flex items-center gap-2">
        <span className="text-xs text-gray-500 dark:text-gray-400">
          {t('header.kbLabel', { defaultValue: 'KB' })}
        </span>
        <Select value={selectedKbId} onValueChange={handleKbChange}>
          <SelectTrigger className="h-8 w-[180px]">
            <SelectValue placeholder={t('header.kbSelect', { defaultValue: 'Select knowledge base' })} />
          </SelectTrigger>
          <SelectContent>
            {availableKbIds.map((kbId) => (
              <SelectItem key={kbId} value={kbId}>
                {kbId}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Button
          variant="ghost"
          size="icon"
          side="bottom"
          tooltip={t('header.kbCreate', { defaultValue: 'Create knowledge base' })}
          onClick={() => setIsCreateOpen(true)}
        >
          <PlusIcon className="size-4" aria-hidden="true" />
        </Button>
      </div>

      <Dialog
        open={isCreateOpen}
        onOpenChange={(open) => {
          if (!isCreating) {
            setIsCreateOpen(open)
            if (!open) {
              setNewKbId('')
            }
          }
        }}
      >
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{t('header.kbCreateTitle', { defaultValue: 'Create Knowledge Base' })}</DialogTitle>
            <DialogDescription>
              {t('header.kbCreateDescription', { defaultValue: 'Create a new isolated knowledge base and switch to it.' })}
            </DialogDescription>
          </DialogHeader>
          <Input
            value={newKbId}
            onChange={(e) => setNewKbId(e.target.value)}
            placeholder={t('header.kbCreatePlaceholder', { defaultValue: 'Enter knowledge base ID' })}
            disabled={isCreating}
          />
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsCreateOpen(false)} disabled={isCreating}>
              {t('common.cancel', { defaultValue: 'Cancel' })}
            </Button>
            <Button onClick={handleCreateKnowledgeBase} disabled={isCreating}>
              {t('common.create', { defaultValue: 'Create' })}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}

export default function SiteHeader() {
  const { t } = useTranslation()
  const { isGuestMode, coreVersion, apiVersion, username, webuiTitle, webuiDescription } = useAuthStore()

  const versionDisplay = (coreVersion && apiVersion)
    ? `${coreVersion}/${apiVersion}`
    : null

  const hasWarning = apiVersion?.endsWith('鈿狅笍')
  const versionTooltip = hasWarning
    ? t('header.frontendNeedsRebuild')
    : versionDisplay ? `v${versionDisplay}` : ''

  const handleLogout = () => {
    navigationService.navigateToLogin()
  }

  return (
    <header className="border-border/40 bg-background/95 supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50 flex h-10 w-full border-b px-4 backdrop-blur">
      <div className="min-w-[200px] w-auto flex items-center">
        <a href={webuiPrefix} className="flex items-center gap-2">
          <ZapIcon className="size-4 text-emerald-400" aria-hidden="true" />
          <span className="font-bold md:inline-block">{SiteInfo.name}</span>
        </a>
        {webuiTitle && (
          <div className="flex items-center">
            <span className="mx-1 text-xs text-gray-500 dark:text-gray-400">|</span>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="font-medium text-sm cursor-default">
                    {webuiTitle}
                  </span>
                </TooltipTrigger>
                {webuiDescription && (
                  <TooltipContent side="bottom">
                    {webuiDescription}
                  </TooltipContent>
                )}
              </Tooltip>
            </TooltipProvider>
          </div>
        )}
      </div>

      <div className="flex h-10 flex-1 items-center justify-center">
        <TabsNavigation />
        <KnowledgeBaseSwitcher />
        {isGuestMode && (
          <div className="ml-2 self-center px-2 py-1 text-xs bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200 rounded-md">
            {t('login.guestMode', 'Guest Mode')}
          </div>
        )}
      </div>

      <nav className="w-[200px] flex items-center justify-end">
        <div className="flex items-center gap-2">
          {versionDisplay && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="text-xs text-gray-500 dark:text-gray-400 mr-1 cursor-default">
                    v{versionDisplay}
                  </span>
                </TooltipTrigger>
                <TooltipContent side="bottom">
                  {versionTooltip}
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
          <Button variant="ghost" size="icon" side="bottom" tooltip={t('header.projectRepository')}>
            <a href={SiteInfo.github} target="_blank" rel="noopener noreferrer">
              <GithubIcon className="size-4" aria-hidden="true" />
            </a>
          </Button>
          <AppSettings />
          {!isGuestMode && (
            <Button
              variant="ghost"
              size="icon"
              side="bottom"
              tooltip={`${t('header.logout')} (${username})`}
              onClick={handleLogout}
            >
              <LogOutIcon className="size-4" aria-hidden="true" />
            </Button>
          )}
        </div>
      </nav>
    </header>
  )
}
