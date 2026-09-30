import { useCallback } from 'react'
import {
  createKnowledgeBase,
  deleteKnowledgeBase,
  listKnowledgeBases
} from '@/api/lightrag'
import type { KnowledgeBaseDeleteResponse } from '@/api/lightrag'
import { useBackendState } from '@/stores/state'
import { useSettingsStore } from '@/stores/settings'
import { useGraphStore } from '@/stores/graph'

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
   * Deletes a base and keeps the selection valid. setAvailableKbIds falls back to the
   * first remaining base, so a view reset is only needed when that actually moved.
   */
  const removeKnowledgeBase = useCallback(async (
    kbId: string
  ): Promise<KnowledgeBaseDeleteResponse> => {
    const previousKbId = useSettingsStore.getState().selectedKbId
    const result = await deleteKnowledgeBase(kbId)
    await refreshKnowledgeBases()
    const current = useSettingsStore.getState()
    if (current.selectedKbId !== previousKbId) {
      resetKbScopedViews()
    }
    return result
  }, [refreshKnowledgeBases])

  return {
    refreshKnowledgeBases,
    selectKnowledgeBase,
    createAndSelectKnowledgeBase,
    removeKnowledgeBase
  }
}
