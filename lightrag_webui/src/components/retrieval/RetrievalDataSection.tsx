import { useCallback, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import Button from '@/components/ui/Button'
import { cn, errorMessage } from '@/lib/utils'
import { queryData } from '@/api/lightrag'
import type { QueryDataResponse } from '@/api/lightrag'
import { useSettingsStore } from '@/stores/settings'
import { ChevronDownIcon, DatabaseIcon, LoaderIcon } from 'lucide-react'

const summaryKeys = ['entities', 'relationships', 'chunks', 'references'] as const

/**
 * Dry-run inspector for the retrieval pipeline: posts the query-box content to
 * `/query/data` and shows the raw hits without any LLM generation. Rendered
 * inside a QuerySettings section, so it must stay narrow and self-explanatory.
 */
export default function RetrievalDataSection({
  getInputQuery,
  resetSignal
}: {
  getInputQuery: () => string
  /** Bump to drop the stale result when the conversation is cleared. */
  resetSignal: number
}) {
  const { t } = useTranslation()
  const [result, setResult] = useState<QueryDataResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [showRaw, setShowRaw] = useState(false)

  useEffect(() => {
    setResult(null)
    setError(null)
    setShowRaw(false)
  }, [resetSignal])

  const handleFetch = useCallback(async () => {
    const query = getInputQuery().trim()
    if (query.length < 3) {
      setError(t('retrievePanel.retrieval.queryTooShort', { defaultValue: 'Query is too short' }))
      return
    }

    setIsLoading(true)
    setError(null)
    try {
      // history_turns is a UI-only knob and is not part of the request contract.
      const settings = { ...useSettingsStore.getState().querySettings }
      delete settings.history_turns
      setResult(await queryData({ ...settings, query, stream: false }))
      setShowRaw(false)
    } catch (err) {
      setResult(null)
      setError(errorMessage(err))
    } finally {
      setIsLoading(false)
    }
  }, [getInputQuery, t])

  const counts = summaryKeys.map((key) => {
    const items = result?.data?.[key]
    return { key, total: Array.isArray(items) ? items.length : 0 }
  })

  const keywords = result?.metadata?.keywords as
    | { high_level?: string[]; low_level?: string[] }
    | undefined

  return (
    <div className="flex flex-col gap-2">
      <p className="text-muted-foreground text-[11px] leading-snug">
        {t('retrievePanel.retrieval.dataHint', {
          defaultValue:
            'Dry-run retrieval on the query-box content: see the entities, relationships and chunks it hits, without asking the model for an answer.'
        })}
      </p>
      <Button
        type="button"
        variant="outline"
        size="sm"
        className="self-start"
        disabled={isLoading}
        onClick={handleFetch}
      >
        {isLoading ? <LoaderIcon className="animate-spin" /> : <DatabaseIcon />}
        {t('retrievePanel.retrieval.fetchData', { defaultValue: 'Retrieval data' })}
      </Button>

      {error && <p className="text-xs text-red-500">{error}</p>}

      {result && !error && result.status !== 'success' && (
        <p className="text-xs text-amber-600 dark:text-amber-400">{result.message}</p>
      )}

      {result && !error && (
        <>
          <div className="flex flex-wrap gap-x-3 gap-y-1 text-[11px]">
            {counts.map(({ key, total }) => (
              <span key={key} className="text-muted-foreground">
                {t(`retrievePanel.retrieval.dataCounts.${key}`, { defaultValue: key })}
                <span className="ml-1 font-medium text-foreground">{total}</span>
              </span>
            ))}
          </div>

          {keywords && (
            <div className="text-muted-foreground flex flex-wrap gap-x-3 text-[11px]">
              {!!keywords.high_level?.length && (
                <span>
                  {t('retrievePanel.retrieval.highLevelKeywords', { defaultValue: 'HL' })}:{' '}
                  {keywords.high_level.join(', ')}
                </span>
              )}
              {!!keywords.low_level?.length && (
                <span>
                  {t('retrievePanel.retrieval.lowLevelKeywords', { defaultValue: 'LL' })}:{' '}
                  {keywords.low_level.join(', ')}
                </span>
              )}
            </div>
          )}

          <button
            type="button"
            className="text-muted-foreground flex items-center gap-1 self-start hover:text-foreground"
            onClick={() => setShowRaw((value) => !value)}
          >
            <ChevronDownIcon
              className={cn('size-3 transition-transform', !showRaw && '-rotate-90')}
            />
            JSON
          </button>
          {showRaw && (
            <pre className="bg-muted max-h-48 overflow-auto rounded-md p-2 text-[10px] leading-tight break-all whitespace-pre-wrap">
              {JSON.stringify(result, null, 2)}
            </pre>
          )}
        </>
      )}
    </div>
  )
}
