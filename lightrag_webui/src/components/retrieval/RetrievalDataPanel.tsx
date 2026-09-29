import { useCallback, useState } from 'react'
import { useTranslation } from 'react-i18next'
import Button from '@/components/ui/Button'
import { errorMessage } from '@/lib/utils'
import { queryData } from '@/api/lightrag'
import type { QueryDataResponse } from '@/api/lightrag'
import { useSettingsStore } from '@/stores/settings'
import { ChevronDownIcon, DatabaseIcon, LoaderIcon } from 'lucide-react'

const summaryKeys = ['entities', 'relationships', 'chunks', 'references'] as const

export default function RetrievalDataPanel({ getInputQuery }: { getInputQuery: () => string }) {
  const { t } = useTranslation()
  const [result, setResult] = useState<QueryDataResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [showRaw, setShowRaw] = useState(false)

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
    <div className="bg-primary-foreground/60 shrink-0 rounded-lg border px-2 py-1.5">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={isLoading}
          onClick={handleFetch}
        >
          {isLoading ? <LoaderIcon className="animate-spin" /> : <DatabaseIcon />}
          {t('retrievePanel.retrieval.fetchData', { defaultValue: 'Retrieval data' })}
        </Button>

        {result && (
          <>
            {counts.map(({ key, total }) => (
              <span key={key} className="text-muted-foreground">
                {t(`retrievePanel.retrieval.dataCounts.${key}`, { defaultValue: key })}
                <span className="ml-1 font-medium text-foreground">{total}</span>
              </span>
            ))}
            <button
              type="button"
              className="text-muted-foreground ml-auto flex items-center gap-1 hover:text-foreground"
              onClick={() => setShowRaw((value) => !value)}
            >
              <ChevronDownIcon
                className={`size-3 transition-transform ${showRaw ? '' : '-rotate-90'}`}
              />
              JSON
            </button>
          </>
        )}
      </div>

      {error && <p className="mt-1 text-xs text-red-500">{error}</p>}

      {result && !error && result.status !== 'success' && (
        <p className="mt-1 text-xs text-amber-600 dark:text-amber-400">{result.message}</p>
      )}

      {result && !!keywords && (
        <div className="text-muted-foreground mt-1 flex flex-wrap gap-x-3 text-[11px]">
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

      {result && showRaw && (
        <pre className="bg-muted mt-1.5 max-h-56 overflow-auto rounded-md p-2 text-[11px] leading-tight">
          {JSON.stringify(result, null, 2)}
        </pre>
      )}
    </div>
  )
}
