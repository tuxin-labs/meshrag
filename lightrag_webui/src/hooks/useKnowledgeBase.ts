import { useCallback } from 'react'
import {
  createKnowledgeBase,
  deleteKnowledgeBase,
  listKnowledgeBaseStats,
  listKnowledgeBases
} from '@/api/lightrag'
import type { KnowledgeBaseDeleteResponse, KnowledgeBaseStats } from '@/api/lightrag'
import { useBackendState } from '@/stores/state'
import { useSettingsStore } from '@/stores/settings'
import { useGraphStore } from '@/stores/graph'

/**
 * Counting a loaded base walks its graph labels, which is not free on large graphs, so the
 * header asks for stats only when the picker opens. This guard keeps the header and the
 * Knowledge Bases page from issuing the same request twice.
 */
let statsInFlight: Promise<Record<string, KnowledgeBaseStats>> | null = null

/**
 * Cached views that belong to the previously selected knowledge base. Switching or
 * deleting a base has to clear them or the UI keeps showing the old base's graph and chat.
 */
const resetKbScopedViews = () => {
  const settings = useSettingsStore.getState()
  settings.setQueryLabel('*')
  settings.setRetrievalHistory([])
  settings.triggerSearchLabelDropdownRefresh()

  const graph = useGraphStore.getState()
  graph.reset()
  graph.setGraphDataFetchAttempted(false)
  graph.setLabelsFetchAttempted(false)

  useBackendState.getState().setPipelineBusy(false)
}

/** Knowledge-base context operations shared by the header switcher and the manager page. */
export function useKnowledgeBase() {
  const refreshKnowledgeBases = useCallback(async (): Promise<string[]> => {
    const response = await listKnowledgeBases()
    const settings = useSettingsStore.getState()
    settings.setAvailableKbIds(response.knowledge_bases)
    settings.setDefaultKbId(response.default_kb ?? response.knowledge_bases[0] ?? null)
    return response.knowledge_bases
  }, [])

  const refreshKnowledgeBaseStats = useCallback(
    async (): Promise<Record<string, KnowledgeBaseStats>> => {
      if (statsInFlight) {
        return statsInFlight
      }
      statsInFlight = (async () => {
        try {
          const response = await listKnowledgeBaseStats()
          const byId = Object.fromEntries(
            response.knowledge_bases.map((stats) => [stats.kb_id, stats])
          )
          useSettingsStore.getState().setKbStatsById(byId)
          return byId
        } catch (error) {
          // An older server has no stats route; callers fall back to a list without counts.
          console.error('Failed to load knowledge base stats:', error)
          return {}
        } finally {
          statsInFlight = null
        }
      })()
      return statsInFlight
    },
    []
  )

  const selectKnowledgeBase = useCallback((kbId: string) => {
    const settings = useSettingsStore.getState()
    if (settings.selectedKbId === kbId) {
      return
    }
    settings.setSelectedKbId(kbId)
    resetKbScopedViews()
  }, [])

  const createAndSelectKnowledgeBase = useCallback(async (kbId: string) => {
    await createKnowledgeBase(kbId)
    await refreshKnowledgeBases()
    selectKnowledgeBase(kbId)
  }, [refreshKnowledgeBases, selectKnowledgeBase])

  /**
   * Deletes a base and keeps the selection valid. Deleting the currently selected
   * base first moves the selection to a surviving base, so KB-scoped views reset
   * along the same path as a normal switch instead of reacting to a base that is
   * already gone (which used to freeze the delete dialog and the table).
   */
  const removeKnowledgeBase = useCallback(async (
    kbId: string
  ): Promise<KnowledgeBaseDeleteResponse> => {
    const settings = useSettingsStore.getState()
    const wasSelected = settings.selectedKbId === kbId
    if (wasSelected) {
      const remaining = settings.availableKbIds.filter((id) => id !== kbId)
      const fallback = remaining.find((id) => id === settings.defaultKbId) || remaining[0]
      if (fallback) {
        selectKnowledgeBase(fallback)
      }
    }
    const result = await deleteKnowledgeBase(kbId)
    await refreshKnowledgeBases()
    const current = useSettingsStore.getState()
    if (current.selectedKbId !== settings.selectedKbId || wasSelected) {
      resetKbScopedViews()
    }
    return result
  }, [refreshKnowledgeBases, selectKnowledgeBase])

  return {
    refreshKnowledgeBases,
    refreshKnowledgeBaseStats,
    selectKnowledgeBase,
    createAndSelectKnowledgeBase,
    removeKnowledgeBase
  }
}
