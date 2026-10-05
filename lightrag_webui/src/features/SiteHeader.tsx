import { useEffect } from 'react'
import Button from '@/components/ui/Button'
import { SiteInfo, webuiPrefix } from '@/lib/constants'
import AppSettings from '@/components/AppSettings'
import { useSettingsStore } from '@/stores/settings'
import { useAuthStore } from '@/stores/state'
import { useTranslation } from 'react-i18next'
import { navigationService } from '@/services/navigation'
import { useKnowledgeBase } from '@/hooks/useKnowledgeBase'
import { GithubIcon, LogOutIcon, DatabaseIcon } from 'lucide-react'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/Tooltip'
import { Select, SelectContent, SelectItem, SelectTrigger } from '@/components/ui/Select'

function KnowledgeBaseSwitcher() {
  const { t } = useTranslation()
  const selectedKbId = useSettingsStore.use.selectedKbId()
  const availableKbIds = useSettingsStore.use.availableKbIds()
  const kbStatsById = useSettingsStore.use.kbStatsById()
  const { refreshKnowledgeBases, refreshKnowledgeBaseStats, selectKnowledgeBase } = useKnowledgeBase()

  useEffect(() => {
    refreshKnowledgeBases().catch((error) => {
      console.error('Failed to load knowledge bases:', error)
    })
  }, [refreshKnowledgeBases])

  const statsLabel = (kbId: string) => {
    const stats = kbStatsById[kbId]
    if (!stats || (stats.documents === null && stats.entities === null)) {
      return null
    }
    return t('header.kbStatsHint', {
      defaultValue: '{{documents}} docs · {{entities}} entities',
      documents: stats.documents ?? '-',
      entities: stats.entities ?? '-'
    })
  }

  return (
    <div className="ml-4 flex shrink-0 items-center gap-2">
      <DatabaseIcon className="size-3.5 shrink-0 text-gray-500 dark:text-gray-400" aria-hidden="true" />
      <Select
        value={selectedKbId}
        onValueChange={selectKnowledgeBase}
        onOpenChange={(open) => open && void refreshKnowledgeBaseStats()}
      >
        <SelectTrigger className="h-8 w-auto min-w-[140px] max-w-[300px]">
          {/* Radix echoes the selected item's children here, and those children carry the
              stats row, so the trigger renders the bare id from the store instead. */}
          <span className="truncate">
            {selectedKbId || t('header.kbSelect', { defaultValue: 'Select knowledge base' })}
          </span>
        </SelectTrigger>
        <SelectContent>
          {availableKbIds.map((kbId) => (
            <SelectItem key={kbId} value={kbId}>
              <span className="flex w-full items-center justify-between gap-3">
                <span className="truncate">{kbId}</span>
                <span className="text-muted-foreground text-xs">{statsLabel(kbId)}</span>
              </span>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}

export default function SiteHeader() {
  const { t } = useTranslation()
  const { isGuestMode, coreVersion, apiVersion, username, webuiTitle, webuiDescription } = useAuthStore()

  const versionDisplay = (coreVersion && apiVersion)
    ? `${coreVersion}/${apiVersion}`
    : null

  // The backend appends the ⚠️ marker when the built webui assets are stale.
  const hasWarning = apiVersion?.includes('⚠')
  const versionTooltip = hasWarning
    ? t('header.frontendNeedsRebuild')
    : versionDisplay ? `v${versionDisplay}` : ''

  const handleLogout = () => {
    navigationService.navigateToLogin()
  }

  return (
    <header className="border-border/40 bg-background/95 supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50 flex h-10 w-full border-b px-4 backdrop-blur">
      <div className="flex min-w-0 flex-1 items-center">
        <a href={webuiPrefix} className="flex shrink-0 items-center gap-2">
          <img src="logo.png" alt="" className="size-4" aria-hidden="true" />
          <span className="font-bold md:inline-block">{SiteInfo.name}</span>
        </a>
        {webuiTitle && (
          <div className="hidden items-center xl:flex">
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
        <KnowledgeBaseSwitcher />
      </div>

      <nav className="flex min-w-0 flex-1 items-center justify-end">
        <div className="flex items-center gap-2">
          {isGuestMode && (
            <div className="px-2 py-1 text-xs bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200 rounded-md">
              {t('login.guestMode', 'Guest Mode')}
            </div>
          )}
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
