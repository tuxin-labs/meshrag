import { beforeAll, describe, expect, mock, test } from 'bun:test'

/**
 * The API module pulls in zustand persist, which needs localStorage; bun has no DOM, so stub
 * it before the dynamic import below resolves.
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

const loadApi = async () => await import('@/api/lightrag')

describe('prepareQueryRequest', () => {
  test('fills kb_ids with the selected knowledge base when none was chosen', async () => {
    const { prepareQueryRequest } = await loadApi()
    const prepared = prepareQueryRequest({ query: 'hello world', mode: 'mix' })
    expect(prepared.kb_ids).toEqual(['default'])
  })

  test('keeps an explicit multi-knowledge-base selection', async () => {
    const { prepareQueryRequest } = await loadApi()
    const prepared = prepareQueryRequest({
      query: 'hello world',
      mode: 'mix',
      kb_ids: ['alpha', 'beta']
    })
    expect(prepared.kb_ids).toEqual(['alpha', 'beta'])
  })

  test('treats an empty kb_ids as a deliberate external-only query', async () => {
    const { prepareQueryRequest } = await loadApi()
    const prepared = prepareQueryRequest({
      query: 'hello world',
      mode: 'mix',
      kb_ids: [],
      external_kb_ids: ['kb-1']
    })
    expect(prepared.kb_ids).toBeUndefined()
    expect(prepared.external_kb_ids).toEqual(['kb-1'])
  })

  test('drops external entries without a url', async () => {
    const { prepareQueryRequest } = await loadApi()
    const prepared = prepareQueryRequest({
      query: 'hello world',
      mode: 'mix',
      external_kbs: [
        { type: 'retrieval', url: '  ' },
        { type: 'rag', url: 'https://service.test/api' }
      ]
    })
    expect(prepared.external_kbs).toHaveLength(1)
    expect(prepared.external_kbs?.[0].url).toBe('https://service.test/api')
  })

  test('drops a partial LLM override, which the backend would reject', async () => {
    const { prepareQueryRequest } = await loadApi()
    const prepared = prepareQueryRequest({
      query: 'hello world',
      mode: 'mix',
      llm_model: 'only-a-model'
    })
    expect(prepared.llm_model).toBeUndefined()
    expect(prepared.llm_binding).toBeUndefined()
  })

  test('keeps a complete LLM override and its optional key', async () => {
    const { prepareQueryRequest } = await loadApi()
    const prepared = prepareQueryRequest({
      query: 'hello world',
      mode: 'mix',
      llm_binding: 'openai',
      llm_model: 'some-model',
      llm_binding_host: 'https://gw.test/v1',
      llm_binding_api_key: 'secret'
    })
    expect(prepared.llm_binding).toBe('openai')
    expect(prepared.llm_binding_api_key).toBe('secret')
  })
})

/** Build a Response whose body arrives in awkward chunks, splitting lines and multi-byte text. */
const chunkedNdjsonResponse = (chunks: string[]) => {
  const encoder = new TextEncoder()
  let index = 0
  return new Response(
    new ReadableStream<Uint8Array>({
      pull(controller) {
        if (index >= chunks.length) {
          controller.close()
          return
        }
        controller.enqueue(encoder.encode(chunks[index++]))
      }
    }),
    { status: 200, headers: { 'Content-Type': 'application/x-ndjson' } }
  )
}

describe('queryTextStream', () => {
  test('dispatches progress, references, warnings and answer deltas', async () => {
    const { queryTextStream } = await loadApi()
    const stream = chunkedNdjsonResponse([
      '{"type": "progress", "stage": "keyword_extraction", "detail": {"high_level": ["a"]}}\n',
      '{"type": "progress", "stage": "vector_search", "detail": {"entities": 9',
      ', "relations": 8}, "status": null}\n',
      '{"references": [{"reference_id": "1", "file_path": "doc.txt"}]}\n',
      '{"warnings": ["external kb unreachable"]}\n',
      '{"response": "你好"}\n{"res',
      'ponse": "世界"}\n',
      '{"error": "boom"}\n'
    ])
    globalThis.fetch = mock(async () => stream) as unknown as typeof fetch

    const stages: string[] = []
    const chunks: string[] = []
    const errors: string[] = []
    let references: unknown = null
    let warnings: unknown = null

    await queryTextStream(
      { query: 'hello world', mode: 'mix' },
      {
        onChunk: (value) => chunks.push(value),
        onProgress: (frame) => stages.push(frame.stage),
        onReferences: (value) => { references = value },
        onWarnings: (value) => { warnings = value },
        onError: (value) => { errors.push(value) }
      }
    )

    expect(stages).toEqual(['keyword_extraction', 'vector_search'])
    expect(references).toEqual([{ reference_id: '1', file_path: 'doc.txt' }])
    expect(warnings).toEqual(['external kb unreachable'])
    // the split multi-byte payload must still reassemble
    expect(chunks.join('')).toBe('你好世界')
    expect(errors).toEqual(['boom'])
  })

  test('sends registry ids rather than inline secrets', async () => {
    const { queryTextStream } = await loadApi()
    let sentBody = ''
    globalThis.fetch = mock(async (_url: string, init?: RequestInit) => {
      sentBody = String(init?.body ?? '')
      return chunkedNdjsonResponse(['{"response": "ok"}\n'])
    }) as unknown as typeof fetch

    await queryTextStream(
      {
        query: 'hello world',
        mode: 'mix',
        kb_ids: [],
        external_kb_ids: ['ext-1'],
        llm_profile_id: 'profile-1'
      },
      { onChunk: () => {} }
    )

    const payload = JSON.parse(sentBody)
    expect(payload.external_kb_ids).toEqual(['ext-1'])
    expect(payload.llm_profile_id).toBe('profile-1')
    expect(payload.kb_ids).toBeUndefined()
    expect(sentBody).not.toContain('llm_binding_api_key')
  })
})
