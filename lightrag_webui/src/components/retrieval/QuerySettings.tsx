import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import { QueryMode, QueryRequest, listExternalKBs, listModelProfiles } from '@/api/lightrag'
import { cn, errorMessage } from '@/lib/utils'
// Removed unused import for Text component
import Checkbox from '@/components/ui/Checkbox'
import Input from '@/components/ui/Input'
import UserPromptInputWithHistory from '@/components/ui/UserPromptInputWithHistory'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/Select'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/Popover'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/Tooltip'
import { useSettingsStore } from '@/stores/settings'
import { useTranslation } from 'react-i18next'
import { ChevronDown, PanelRightClose, RotateCcw } from 'lucide-react'
import Button from '@/components/ui/Button'
import RetrievalDataSection from '@/components/retrieval/RetrievalDataSection'

/** Collapsible group so the ~15 controls do not all compete for attention at once. */
const Section = ({
  title,
  hint,
  defaultOpen = false,
  children
}: {
  title: string
  hint?: string
  defaultOpen?: boolean
  children: ReactNode
}) => {
  const [open, setOpen] = useState(defaultOpen)

  return (
    <div className="border-border/60 border-b pb-2 last:border-b-0">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="text-muted-foreground hover:text-foreground flex w-full items-center gap-1.5 px-1 py-0.5 text-[11px] font-semibold uppercase tracking-wide"
      >
        <ChevronDown className={cn('size-3 shrink-0 transition-transform', !open && '-rotate-90')} />
        <span className="shrink-0">{title}</span>
        {!open && hint && (
          <span className="text-muted-foreground/80 ml-auto min-w-0 truncate text-right text-[11px] font-normal normal-case">
            {hint}
          </span>
        )}
      </button>
      {open && <div className="flex flex-col gap-2 px-1 pt-1.5">{children}</div>}
    </div>
  )
}

// Module level so QuerySettings re-renders do not remount it (it is pure props anyway).
const ResetButton = ({ onClick, title }: { onClick: () => void; title: string }) => (
  <TooltipProvider>
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          onClick={onClick}
          className="mr-1 p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          title={title}
        >
          <RotateCcw className="h-3 w-3 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200" />
        </button>
      </TooltipTrigger>
      <TooltipContent side="left">
        <p>{title}</p>
      </TooltipContent>
    </Tooltip>
  </TooltipProvider>
)

interface QuerySettingsProps {
  /** When provided, the card header shows a collapse button (narrow screens). */
  onCollapse?: () => void
  /** Current query-box text, for the retrieval-data dry run. */
  getInputQuery: () => string
  /** Bump to drop the retrieval-data result when the conversation is cleared. */
  resetSignal: number
}

export default function QuerySettings({ onCollapse, getInputQuery, resetSignal }: QuerySettingsProps) {
  const { t } = useTranslation()
  const querySettings = useSettingsStore((state) => state.querySettings)
  const userPromptHistory = useSettingsStore((state) => state.userPromptHistory)
  const selectedKbId = useSettingsStore.use.selectedKbId()
  const availableKbIds = useSettingsStore.use.availableKbIds()
  const modelProfiles = useSettingsStore.use.availableModelProfiles()
  const externalKBs = useSettingsStore.use.availableExternalKBs()
  const setModelProfiles = useSettingsStore.use.setAvailableModelProfiles()
  const setExternalKBs = useSettingsStore.use.setAvailableExternalKBs()
  const currentTab = useSettingsStore.use.currentTab()

  // Pickers must show profiles/KBs registered on the management tabs without a reload.
  useEffect(() => {
    if (currentTab !== 'retrieval') return
    const refreshRegistries = async () => {
      try {
        const [profiles, kbs] = await Promise.all([listModelProfiles(), listExternalKBs()])
        setModelProfiles(profiles)
        setExternalKBs(kbs)
      } catch (error) {
        console.error('Failed to load model profiles or external knowledge bases:', errorMessage(error))
      }
    }
    refreshRegistries()
  }, [currentTab, setModelProfiles, setExternalKBs])

  const llmProfiles = useMemo(
    () => modelProfiles.filter((profile) => profile.kind === 'llm' && profile.enabled),
    [modelProfiles]
  )
  const selectableExternalKBs = useMemo(
    () => externalKBs.filter((kb) => kb.enabled),
    [externalKBs]
  )

  const handleChange = useCallback((key: keyof QueryRequest, value: any) => {
    useSettingsStore.getState().updateQuerySettings({ [key]: value })
  }, [])

  const handleSelectFromHistory = useCallback((prompt: string) => {
    handleChange('user_prompt', prompt)
  }, [handleChange])

  const handleDeleteFromHistory = useCallback((index: number) => {
    const newHistory = [...userPromptHistory]
    newHistory.splice(index, 1)
    useSettingsStore.getState().setUserPromptHistory(newHistory)
  }, [userPromptHistory])

  // Default values for reset functionality
  const defaultValues = useMemo(() => ({
    mode: 'mix' as QueryMode,
    top_k: 40,
    chunk_top_k: 20,
    max_entity_tokens: 6000,
    max_relation_tokens: 8000,
    max_total_tokens: 30000
  }), [])

  const handleReset = useCallback((key: keyof typeof defaultValues) => {
    handleChange(key, defaultValues[key])
  }, [handleChange, defaultValues])

  const externalKbCount = (querySettings.external_kb_ids ?? []).length
  // An unset kb_ids means "current knowledge base"; an explicit empty array is the user's
  // deliberate choice to query external services only.
  const kbIdsStored = querySettings.kb_ids
  const externalOnly = externalKbCount > 0 && Array.isArray(kbIdsStored) && kbIdsStored.length === 0

  const effectiveKbIds = useMemo(() => {
    const normalizedKbIds = (querySettings.kb_ids || []).filter((kbId): kbId is string => {
      return !!kbId && availableKbIds.includes(kbId)
    })
    if (normalizedKbIds.length > 0) {
      return normalizedKbIds
    }
    return Array.isArray(kbIdsStored) && kbIdsStored.length === 0 ? [] : [selectedKbId]
  }, [querySettings.kb_ids, availableKbIds, selectedKbId, kbIdsStored])

  const kbSelectionLabel = useMemo(() => {
    if (effectiveKbIds.length === 0) {
      return t('retrievePanel.querySettings.kbNoneExternalOnly', { defaultValue: 'None (external only)' })
    }
    if (effectiveKbIds.length === 1) {
      return effectiveKbIds[0]
    }
    return `${effectiveKbIds[0]} +${effectiveKbIds.length - 1}`
  }, [effectiveKbIds, t])

  const handleToggleKb = useCallback((kbId: string, checked: boolean) => {
    const nextKbIds = checked
      ? Array.from(new Set([...effectiveKbIds, kbId]))
      : effectiveKbIds.filter((item) => item !== kbId)

    handleChange('kb_ids', nextKbIds.length > 0
      ? nextKbIds
      : (querySettings.external_kb_ids?.length ? [] : [selectedKbId]))
  }, [effectiveKbIds, handleChange, selectedKbId, querySettings.external_kb_ids])

  const selectedExternalKbIds = useMemo(
    () => (querySettings.external_kb_ids ?? []).filter((id) => selectableExternalKBs.some((kb) => kb.id === id)),
    [querySettings.external_kb_ids, selectableExternalKBs]
  )

  const externalKbLabel = useMemo(() => {
    if (selectedExternalKbIds.length === 0) {
      return t('retrievePanel.querySettings.externalKbNone', { defaultValue: 'None' })
    }
    const first = selectableExternalKBs.find((kb) => kb.id === selectedExternalKbIds[0])
    return `${first?.name ?? selectedExternalKbIds[0]} +${selectedExternalKbIds.length - 1}`
  }, [selectedExternalKbIds, selectableExternalKBs, t])

  const handleToggleExternalKb = useCallback((kbId: string, checked: boolean) => {
    const nextIds = checked
      ? Array.from(new Set([...selectedExternalKbIds, kbId]))
      : selectedExternalKbIds.filter((id) => id !== kbId)
    handleChange('external_kb_ids', nextIds.length > 0 ? nextIds : undefined)
  }, [selectedExternalKbIds, handleChange])

  const selectedProfileLabel = useMemo(() => {
    const profile = llmProfiles.find((item) => item.id === querySettings.llm_profile_id)
    return profile
      ? profile.name
      : t('retrievePanel.querySettings.serverDefaultModel', { defaultValue: 'Server default' })
  }, [llmProfiles, querySettings.llm_profile_id, t])

  const handleResetKbSelection = useCallback(() => {
    handleChange('kb_ids', [selectedKbId])
  }, [handleChange, selectedKbId])

  return (
    <Card className="flex shrink-0 flex-col w-[280px]">
      <CardHeader className="flex flex-row items-center justify-between px-4 pt-4 pb-2">
        <CardTitle>{t('retrievePanel.querySettings.parametersTitle')}</CardTitle>
        {onCollapse && (
          <Button
            variant="ghost"
            size="icon"
            className="size-6"
            tooltip={t('retrievePanel.querySettings.collapse', { defaultValue: 'Collapse panel' })}
            onClick={onCollapse}
          >
            <PanelRightClose className="size-3.5" />
          </Button>
        )}
        <CardDescription className="sr-only">{t('retrievePanel.querySettings.parametersDescription')}</CardDescription>
      </CardHeader>
      <CardContent className="m-0 flex grow flex-col p-0 text-xs">
        <div className="relative size-full">
          <div className="absolute inset-0 flex flex-col gap-2 overflow-auto px-2 pr-2">
            {/* User Prompt - Moved to top for better dropdown space */}
            <>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <label htmlFor="user_prompt" className="ml-1 cursor-help">
                      {t('retrievePanel.querySettings.userPrompt')}
                    </label>
                  </TooltipTrigger>
                  <TooltipContent side="left">
                    <p>{t('retrievePanel.querySettings.userPromptTooltip')}</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <div>
                <UserPromptInputWithHistory
                  id="user_prompt"
                  value={querySettings.user_prompt || ''}
                  onChange={(value) => handleChange('user_prompt', value)}
                  onSelectFromHistory={handleSelectFromHistory}
                  onDeleteFromHistory={handleDeleteFromHistory}
                  history={userPromptHistory}
                  placeholder={t('retrievePanel.querySettings.userPromptPlaceholder')}
                  className="h-9"
                />
              </div>
            </>

            <Section
              defaultOpen
              title={t('retrievePanel.querySettings.groupScope', { defaultValue: 'Retrieval scope' })}
              hint={`${effectiveKbIds.length} · ${externalKbCount}`}
            >
              {/* KB Selection */}
              <>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label htmlFor="kb_ids" className="ml-1 cursor-help">
                        {t('retrievePanel.querySettings.kbSelection', { defaultValue: 'KB Selection' })}
                      </label>
                    </TooltipTrigger>
                    <TooltipContent side="left">
                      <p>{t('retrievePanel.querySettings.kbSelectionTooltip', { defaultValue: 'Select one or more knowledge bases for this query.' })}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <div className="flex items-center gap-1">
                  <Popover>
                    <PopoverTrigger asChild>
                      <button
                        id="kb_ids"
                        type="button"
                        className="border-input bg-background hover:bg-primary/5 flex h-9 flex-1 items-center justify-between rounded-md border px-3 text-sm"
                      >
                        <span className="truncate text-left">{kbSelectionLabel}</span>
                        <ChevronDown className="h-4 w-4 opacity-50" />
                      </button>
                    </PopoverTrigger>
                    <PopoverContent align="start" className="w-[220px] p-2">
                      <div className="flex flex-col gap-2">
                        {availableKbIds.map((kbId) => (
                          <label
                            key={kbId}
                            className="hover:bg-accent flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-sm"
                          >
                            <Checkbox
                              checked={effectiveKbIds.includes(kbId)}
                              onCheckedChange={(checked) => handleToggleKb(kbId, checked === true)}
                            />
                            <span className="truncate">{kbId}</span>
                          </label>
                        ))}
                      </div>
                    </PopoverContent>
                  </Popover>
                  <ResetButton
                    onClick={handleResetKbSelection}
                    title={t('retrievePanel.querySettings.kbSelectionReset', { defaultValue: 'Reset to current KB' })}
                  />
                </div>
              </>

              {/* External Knowledge Bases */}
              <>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label htmlFor="external_kb_ids" className="ml-1 cursor-help">
                        {t('retrievePanel.querySettings.externalKbs', { defaultValue: 'External knowledge bases' })}
                      </label>
                    </TooltipTrigger>
                    <TooltipContent side="left">
                      <p>{t('retrievePanel.querySettings.externalKbsTooltip', {
                        defaultValue: 'Registered third-party services queried together with the internal knowledge bases. Manage them on the External KBs tab.'
                      })}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <div className="flex items-center gap-1">
                  <Popover>
                    <PopoverTrigger asChild>
                      <button
                        id="external_kb_ids"
                        type="button"
                        className="border-input bg-background hover:bg-primary/5 flex h-9 flex-1 items-center justify-between rounded-md border px-3 text-sm"
                      >
                        <span className="truncate text-left">{externalKbLabel}</span>
                        <ChevronDown className="h-4 w-4 opacity-50" />
                      </button>
                    </PopoverTrigger>
                    <PopoverContent align="start" className="w-[240px] p-2">
                      {selectableExternalKBs.length === 0 ? (
                        <p className="text-muted-foreground px-2 py-1 text-sm">
                          {t('retrievePanel.querySettings.externalKbsEmpty', {
                            defaultValue: 'No external knowledge bases registered yet.'
                          })}
                        </p>
                      ) : (
                        <div className="flex flex-col gap-2">
                          {selectableExternalKBs.map((kb) => (
                            <label
                              key={kb.id}
                              className="hover:bg-accent flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-sm"
                            >
                              <Checkbox
                                checked={selectedExternalKbIds.includes(kb.id)}
                                onCheckedChange={(checked) => handleToggleExternalKb(kb.id, checked === true)}
                              />
                              <span className="truncate">{kb.name}</span>
                              <span className="text-muted-foreground ml-auto shrink-0 text-[11px]">{kb.type}</span>
                            </label>
                          ))}
                        </div>
                      )}
                    </PopoverContent>
                  </Popover>
                  <ResetButton
                    onClick={() => handleChange('external_kb_ids', undefined)}
                    title={t('retrievePanel.querySettings.externalKbsReset', { defaultValue: 'Clear external sources' })}
                  />
                </div>

                {externalKbCount > 0 && (
                  <label className="ml-1 flex cursor-pointer items-center gap-2 text-[11px]">
                    <Checkbox
                      checked={externalOnly}
                      onCheckedChange={(checked) =>
                        handleChange('kb_ids', checked === true ? [] : [selectedKbId])
                      }
                    />
                    <span>
                      {t('retrievePanel.querySettings.externalKbOnly', {
                        defaultValue: 'Skip internal knowledge bases'
                      })}
                    </span>
                  </label>
                )}
              </>

            </Section>

            <Section
              title={t('retrievePanel.querySettings.groupModel', { defaultValue: 'Model' })}
              hint={selectedProfileLabel}
            >
              {/* Model profile */}
              <>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label htmlFor="llm_profile_select" className="ml-1 cursor-help">
                        {t('retrievePanel.querySettings.modelProfile', { defaultValue: 'Model' })}
                      </label>
                    </TooltipTrigger>
                    <TooltipContent side="left">
                      <p>{t('retrievePanel.querySettings.modelProfileTooltip', {
                        defaultValue: 'Run this query with a registered model. The API key stays on the server.'
                      })}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <div className="flex items-center gap-1">
                  <Select
                    value={querySettings.llm_profile_id || '__server_default__'}
                    onValueChange={(value) =>
                      handleChange('llm_profile_id', value === '__server_default__' ? undefined : value)
                    }
                  >
                    <SelectTrigger
                      id="llm_profile_select"
                      className="hover:bg-primary/5 h-9 flex-1 cursor-pointer text-left [&>span]:break-all [&>span]:line-clamp-1"
                    >
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectGroup>
                        <SelectItem value="__server_default__">
                          {t('retrievePanel.querySettings.serverDefaultModel', { defaultValue: 'Server default' })}
                        </SelectItem>
                        {llmProfiles.map((profile) => (
                          <SelectItem key={profile.id} value={profile.id}>
                            {profile.name} ({profile.model})
                          </SelectItem>
                        ))}
                      </SelectGroup>
                    </SelectContent>
                  </Select>
                  <ResetButton
                    onClick={() => handleChange('llm_profile_id', undefined)}
                    title={t('retrievePanel.querySettings.modelProfileReset', { defaultValue: 'Use server default model' })}
                  />
                </div>
                <p className="text-muted-foreground ml-1 truncate text-[11px]">{selectedProfileLabel}</p>
              </>

            </Section>

            <Section
              title={t('retrievePanel.querySettings.groupParams', { defaultValue: 'Retrieval parameters' })}
              hint={querySettings.mode}
            >
              {/* Query Mode */}
              <>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label htmlFor="query_mode_select" className="ml-1 cursor-help">
                        {t('retrievePanel.querySettings.queryMode')}
                      </label>
                    </TooltipTrigger>
                    <TooltipContent side="left">
                      <p>{t('retrievePanel.querySettings.queryModeTooltip')}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <div className="flex items-center gap-1">
                  <Select
                    value={querySettings.mode}
                    onValueChange={(v) => handleChange('mode', v as QueryMode)}
                  >
                    <SelectTrigger
                      id="query_mode_select"
                      className="hover:bg-primary/5 h-9 cursor-pointer focus:ring-0 focus:ring-offset-0 focus:outline-0 active:right-0 flex-1 text-left [&>span]:break-all [&>span]:line-clamp-1"
                    >
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectGroup>
                        <SelectItem value="naive">{t('retrievePanel.querySettings.queryModeOptions.naive')}</SelectItem>
                        <SelectItem value="local">{t('retrievePanel.querySettings.queryModeOptions.local')}</SelectItem>
                        <SelectItem value="global">{t('retrievePanel.querySettings.queryModeOptions.global')}</SelectItem>
                        <SelectItem value="hybrid">{t('retrievePanel.querySettings.queryModeOptions.hybrid')}</SelectItem>
                        <SelectItem value="mix">{t('retrievePanel.querySettings.queryModeOptions.mix')}</SelectItem>
                        <SelectItem value="bypass">{t('retrievePanel.querySettings.queryModeOptions.bypass')}</SelectItem>
                      </SelectGroup>
                    </SelectContent>
                  </Select>
                  <ResetButton
                    onClick={() => handleReset('mode')}
                    title="Reset to default (Mix)"
                  />
                </div>
              </>

              {/* Top K */}
              <>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label htmlFor="top_k" className="ml-1 cursor-help">
                        {t('retrievePanel.querySettings.topK')}
                      </label>
                    </TooltipTrigger>
                    <TooltipContent side="left">
                      <p>{t('retrievePanel.querySettings.topKTooltip')}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <div className="flex items-center gap-1">
                  <Input
                    id="top_k"
                    type="number"
                    value={querySettings.top_k ?? ''}
                    onChange={(e) => {
                      const value = e.target.value
                      handleChange('top_k', value === '' ? '' : parseInt(value) || 0)
                    }}
                    onBlur={(e) => {
                      const value = e.target.value
                      if (value === '' || isNaN(parseInt(value))) {
                        handleChange('top_k', 40)
                      }
                    }}
                    min={1}
                    placeholder={t('retrievePanel.querySettings.topKPlaceholder')}
                    className="h-9 flex-1 pr-2 [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none [-moz-appearance:textfield]"
                  />
                  <ResetButton
                    onClick={() => handleReset('top_k')}
                    title="Reset to default"
                  />
                </div>
              </>

              {/* Chunk Top K */}
              <>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label htmlFor="chunk_top_k" className="ml-1 cursor-help">
                        {t('retrievePanel.querySettings.chunkTopK')}
                      </label>
                    </TooltipTrigger>
                    <TooltipContent side="left">
                      <p>{t('retrievePanel.querySettings.chunkTopKTooltip')}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <div className="flex items-center gap-1">
                  <Input
                    id="chunk_top_k"
                    type="number"
                    value={querySettings.chunk_top_k ?? ''}
                    onChange={(e) => {
                      const value = e.target.value
                      handleChange('chunk_top_k', value === '' ? '' : parseInt(value) || 0)
                    }}
                    onBlur={(e) => {
                      const value = e.target.value
                      if (value === '' || isNaN(parseInt(value))) {
                        handleChange('chunk_top_k', 20)
                      }
                    }}
                    min={1}
                    placeholder={t('retrievePanel.querySettings.chunkTopKPlaceholder')}
                    className="h-9 flex-1 pr-2 [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none [-moz-appearance:textfield]"
                  />
                  <ResetButton
                    onClick={() => handleReset('chunk_top_k')}
                    title="Reset to default"
                  />
                </div>
              </>

              {/* Max Entity Tokens */}
              <>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label htmlFor="max_entity_tokens" className="ml-1 cursor-help">
                        {t('retrievePanel.querySettings.maxEntityTokens')}
                      </label>
                    </TooltipTrigger>
                    <TooltipContent side="left">
                      <p>{t('retrievePanel.querySettings.maxEntityTokensTooltip')}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <div className="flex items-center gap-1">
                  <Input
                    id="max_entity_tokens"
                    type="number"
                    value={querySettings.max_entity_tokens ?? ''}
                    onChange={(e) => {
                      const value = e.target.value
                      handleChange('max_entity_tokens', value === '' ? '' : parseInt(value) || 0)
                    }}
                    onBlur={(e) => {
                      const value = e.target.value
                      if (value === '' || isNaN(parseInt(value))) {
                        handleChange('max_entity_tokens', 6000)
                      }
                    }}
                    min={1}
                    placeholder={t('retrievePanel.querySettings.maxEntityTokensPlaceholder')}
                    className="h-9 flex-1 pr-2 [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none [-moz-appearance:textfield]"
                  />
                  <ResetButton
                    onClick={() => handleReset('max_entity_tokens')}
                    title="Reset to default"
                  />
                </div>
              </>

              {/* Max Relation Tokens */}
              <>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label htmlFor="max_relation_tokens" className="ml-1 cursor-help">
                        {t('retrievePanel.querySettings.maxRelationTokens')}
                      </label>
                    </TooltipTrigger>
                    <TooltipContent side="left">
                      <p>{t('retrievePanel.querySettings.maxRelationTokensTooltip')}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <div className="flex items-center gap-1">
                  <Input
                    id="max_relation_tokens"
                    type="number"
                    value={querySettings.max_relation_tokens ?? ''}
                    onChange={(e) => {
                      const value = e.target.value
                      handleChange('max_relation_tokens', value === '' ? '' : parseInt(value) || 0)
                    }}
                    onBlur={(e) => {
                      const value = e.target.value
                      if (value === '' || isNaN(parseInt(value))) {
                        handleChange('max_relation_tokens', 8000)
                      }
                    }}
                    min={1}
                    placeholder={t('retrievePanel.querySettings.maxRelationTokensPlaceholder')}
                    className="h-9 flex-1 pr-2 [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none [-moz-appearance:textfield]"
                  />
                  <ResetButton
                    onClick={() => handleReset('max_relation_tokens')}
                    title="Reset to default"
                  />
                </div>
              </>

              {/* Max Total Tokens */}
              <>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <label htmlFor="max_total_tokens" className="ml-1 cursor-help">
                        {t('retrievePanel.querySettings.maxTotalTokens')}
                      </label>
                    </TooltipTrigger>
                    <TooltipContent side="left">
                      <p>{t('retrievePanel.querySettings.maxTotalTokensTooltip')}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <div className="flex items-center gap-1">
                  <Input
                    id="max_total_tokens"
                    type="number"
                    value={querySettings.max_total_tokens ?? ''}
                    onChange={(e) => {
                      const value = e.target.value
                      handleChange('max_total_tokens', value === '' ? '' : parseInt(value) || 0)
                    }}
                    onBlur={(e) => {
                      const value = e.target.value
                      if (value === '' || isNaN(parseInt(value))) {
                        handleChange('max_total_tokens', 30000)
                      }
                    }}
                    min={1}
                    placeholder={t('retrievePanel.querySettings.maxTotalTokensPlaceholder')}
                    className="h-9 flex-1 pr-2 [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none [-moz-appearance:textfield]"
                  />
                  <ResetButton
                    onClick={() => handleReset('max_total_tokens')}
                    title="Reset to default"
                  />
                </div>
              </>

            </Section>

            <Section
              title={t('retrievePanel.querySettings.groupOutput', { defaultValue: 'Output' })}
              hint={querySettings.stream
                ? t('retrievePanel.querySettings.streamOn', { defaultValue: 'streaming' })
                : t('retrievePanel.querySettings.streamOff', { defaultValue: 'blocking' })}
            >
              {/* Toggle Options */}
              <>
                <div className="flex items-center gap-2">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <label htmlFor="enable_rerank" className="flex-1 ml-1 cursor-help">
                          {t('retrievePanel.querySettings.enableRerank')}
                        </label>
                      </TooltipTrigger>
                      <TooltipContent side="left">
                        <p>{t('retrievePanel.querySettings.enableRerankTooltip')}</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                  <Checkbox
                    className="mr-10 cursor-pointer"
                    id="enable_rerank"
                    checked={querySettings.enable_rerank}
                    onCheckedChange={(checked) => handleChange('enable_rerank', checked)}
                  />
                </div>

                <div className="flex items-center gap-2">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <label htmlFor="include_references" className="flex-1 ml-1 cursor-help">
                          {t('retrievePanel.querySettings.includeReferences')}
                        </label>
                      </TooltipTrigger>
                      <TooltipContent side="left">
                        <p>{t('retrievePanel.querySettings.includeReferencesTooltip')}</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                  <Checkbox
                    className="mr-10 cursor-pointer"
                    id="include_references"
                    checked={querySettings.include_references ?? true}
                    onCheckedChange={(checked) => {
                      handleChange('include_references', checked)
                      if (!checked) {
                        handleChange('include_chunk_content', false)
                      }
                    }}
                  />
                </div>

                <div className="flex items-center gap-2">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <label htmlFor="include_chunk_content" className="flex-1 ml-1 cursor-help">
                          {t('retrievePanel.querySettings.includeChunkContent')}
                        </label>
                      </TooltipTrigger>
                      <TooltipContent side="left">
                        <p>{t('retrievePanel.querySettings.includeChunkContentTooltip')}</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                  <Checkbox
                    className="mr-10 cursor-pointer"
                    id="include_chunk_content"
                    disabled={querySettings.include_references === false}
                    checked={querySettings.include_chunk_content ?? false}
                    onCheckedChange={(checked) => handleChange('include_chunk_content', checked)}
                  />
                </div>

                <div className="flex items-center gap-2">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <label htmlFor="only_need_context" className="flex-1 ml-1 cursor-help">
                          {t('retrievePanel.querySettings.onlyNeedContext')}
                        </label>
                      </TooltipTrigger>
                      <TooltipContent side="left">
                        <p>{t('retrievePanel.querySettings.onlyNeedContextTooltip')}</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                  <Checkbox
                    className="mr-10 cursor-pointer"
                    id="only_need_context"
                    checked={querySettings.only_need_context}
                    onCheckedChange={(checked) => {
                      handleChange('only_need_context', checked)
                      if (checked) {
                        handleChange('only_need_prompt', false)
                      }
                    }}
                  />
                </div>

                <div className="flex items-center gap-2">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <label htmlFor="only_need_prompt" className="flex-1 ml-1 cursor-help">
                          {t('retrievePanel.querySettings.onlyNeedPrompt')}
                        </label>
                      </TooltipTrigger>
                      <TooltipContent side="left">
                        <p>{t('retrievePanel.querySettings.onlyNeedPromptTooltip')}</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                  <Checkbox
                    className="mr-10 cursor-pointer"
                    id="only_need_prompt"
                    checked={querySettings.only_need_prompt}
                    onCheckedChange={(checked) => {
                      handleChange('only_need_prompt', checked)
                      if (checked) {
                        handleChange('only_need_context', false)
                      }
                    }}
                  />
                </div>

                <div className="flex items-center gap-2">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <label htmlFor="stream" className="flex-1 ml-1 cursor-help">
                          {t('retrievePanel.querySettings.streamResponse')}
                        </label>
                      </TooltipTrigger>
                      <TooltipContent side="left">
                        <p>{t('retrievePanel.querySettings.streamResponseTooltip')}</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                  <Checkbox
                    className="mr-10 cursor-pointer"
                    id="stream"
                    checked={querySettings.stream}
                    onCheckedChange={(checked) => handleChange('stream', checked)}
                  />
                </div>
              </>
            </Section>

            <Section
              title={t('retrievePanel.retrieval.fetchData', { defaultValue: 'Retrieval data' })}
            >
              <RetrievalDataSection getInputQuery={getInputQuery} resetSignal={resetSignal} />
            </Section>

          </div>
        </div>
      </CardContent>
    </Card>
  )
}
