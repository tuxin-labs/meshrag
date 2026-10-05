import { beforeAll, describe, expect, test } from 'bun:test'

/**
 * The settings store uses zustand persist, which needs localStorage; bun has no DOM,
 * so stub it before the dynamic import below resolves.
 */
beforeAll(() => {
  const store = new Map<string, string>()
  globalThis.localStorage = {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => void store.set(key, String(value)),
    removeItem: (key: string) => void store.delete(key),
    clear: () => store.clear(),
    key: () => null,
    length: 0
  } as unknown as Storage
})

const migrate = async (state: any, version: number) => {
  const { useSettingsStore } = await import('@/stores/settings')
  const options = (useSettingsStore as any).persist.getOptions()
  return options.migrate(state, version)
}

describe('settings store persistence migrations', () => {
  test('v20/v21 backfill multi-KB defaults', async () => {
    const migrated = await migrate({ selectedKbId: '', availableKbIds: [] }, 19)
    expect(migrated.availableKbIds).toEqual(['default'])
    expect(migrated.selectedKbId).toBe('default')
  })

  test('v21 keeps a selected KB that exists in the available list', async () => {
    const migrated = await migrate(
      { selectedKbId: 'research', availableKbIds: ['default', 'research'] },
      20
    )
    expect(migrated.selectedKbId).toBe('research')
  })

  test('v22 drops inline LLM overrides in favour of server-side registries', async () => {
    const migrated = await migrate(
      {
        querySettings: {
          mode: 'mix',
          llm_binding: 'openai',
          llm_model: 'gpt-4',
          llm_binding_host: 'https://api.test/v1',
          llm_binding_api_key: 'sk-secret',
          external_kbs: [{ type: 'retrieval', url: 'https://x.test' }]
        }
      },
      21
    )
    expect(migrated.querySettings.llm_binding).toBeUndefined()
    expect(migrated.querySettings.llm_model).toBeUndefined()
    expect(migrated.querySettings.llm_binding_host).toBeUndefined()
    expect(migrated.querySettings.llm_binding_api_key).toBeUndefined()
    expect(migrated.querySettings.external_kbs).toBeUndefined()
    expect(migrated.availableModelProfiles).toEqual([])
    expect(migrated.availableExternalKBs).toEqual([])
  })

  test('v23 un-sticks a persisted empty kb_ids so queries stay internal', async () => {
    const migrated = await migrate({ querySettings: { mode: 'mix', kb_ids: [] } }, 22)
    expect(migrated.querySettings.kb_ids).toBeUndefined()
  })

  test('v23 keeps an explicit multi-KB selection intact', async () => {
    const migrated = await migrate(
      { querySettings: { mode: 'mix', kb_ids: ['a', 'b'] } },
      22
    )
    expect(migrated.querySettings.kb_ids).toEqual(['a', 'b'])
  })

  test('v24 forces sidebarCollapsed to boolean', async () => {
    const migrated = await migrate({ sidebarCollapsed: 'yes' as any }, 23)
    expect(migrated.sidebarCollapsed).toBe(false)
  })
})
