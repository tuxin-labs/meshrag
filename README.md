<div align="center">

# MeshRAG

**Multi-Knowledge-Base Graph RAG engine, based on [LightRAG](https://github.com/HKUDS/LightRAG)**

[中文文档](README-zh.md) · [License](#license) · Python 3.10+

</div>

> [!NOTE]
> MeshRAG is a fork of [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) (v1.4.11, upstream state as of 2026-03-08), redistributed under the MIT License with gratitude to the upstream team.
> The Python module is still `import lightrag` for ecosystem compatibility; the distribution is `pip install meshrag`.
> Inherited upstream documentation is preserved in [docs/LightRAG-upstream-usage.md](docs/LightRAG-upstream-usage.md).

## Why MeshRAG

MeshRAG keeps the LightRAG graph-RAG engine (entity/relation extraction, multi-modal retrieval, pluggable storage) and adds what production multi-team deployments need:

### 🔀 Multi-knowledge-base federated query

Query **several local knowledge bases at once**, in a single API call:

1. Per-KB parallel retrieval → raw chunks + knowledge graph context per KB
2. Global deduplication and budget control across all KBs
3. Chunk reranking (with optional reranker)
4. Reference/citation rebuilding so answers cite the right KB
5. LLM answer generation with merged context

### 🌐 External knowledge base integration

Federate **external retrieval/RAG APIs** into the same query flow via `external_kbs`:

- `retrieval` type: the external API returns similar text chunks; MeshRAG's LLM generates the answer
- `rag` type: the external API is a full RAG service and returns a final answer

### 📄 Document management APIs

Upload/status pipeline plus fast single-document lookup by file path —
`GET /documents/by_file_path?kb_id=...&file_path=...` — avoiding expensive
pagination when the document set is large.

### 🔒 Deletion/parsing concurrency pipeline

Cooperative cancellation and deletion semantics with crash recovery:
deletion requests take priority over in-flight processing, interrupted
documents resume instead of being reset, and transient failures do not get
misreported as permanent failures.

### 🧬 Everything inherited from LightRAG

- Graph-based knowledge representation with local / global / hybrid / naive / mix query modes
- Pluggable storage backends (JSON, Redis, PostgreSQL, MongoDB, Milvus, Qdrant, Neo4j, Memgraph, Faiss, NetworkX)
- WebUI, Ollama-compatible API, streaming responses
- Workspace isolation for multi-tenant deployment

## Quick Start

### Install

```bash
pip install meshrag            # core
pip install "meshrag[api]"     # with API server support
```

### A simple program

```python
import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import setup_logger

setup_logger("lightrag", level="INFO")

WORKING_DIR = "./rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )
    # IMPORTANT: Both initialization calls are required!
    await rag.initialize_storages()  # Initialize storage backends
    return rag

async def main():
    rag = await initialize_rag()
    await rag.ainsert("Your text")
    result = await rag.aquery(
        "Your question", param=QueryParam(mode="hybrid")
    )
    print(result)
    await rag.finalize_storages()  # REQUIRED for cleanup

asyncio.run(main())
```

### ⚠️ Important: Initialization Requirements

**MeshRAG requires explicit initialization before use.** You must call
`await rag.initialize_storages()` after creating a `LightRAG` instance, and
`await rag.finalize_storages()` for cleanup — otherwise you will encounter
errors such as `AttributeError: __aenter__`.

### API server

```bash
cp env.example .env    # configure LLM / embedding / storage first
meshrag-server         # production
meshrag-gunicorn       # multi-worker
```

Docker:

```bash
docker compose up -d          # uses ghcr.io/tuxin-labs/meshrag
docker build -t meshrag .     # full image (frontend + backend)
docker build -f Dockerfile.lite .  # lite image (API + offline storage only)
```

### Multi-KB query example

`POST /query` accepts `kb_ids` (local KBs) and `external_kbs` in the same request:

```json
{
  "query": "Which contracts mention data-residency requirements?",
  "mode": "mix",
  "kb_ids": ["contracts-2025", "policies"],
  "external_kbs": [
    {
      "type": "retrieval",
      "url": "https://kb.example.com/api/search",
      "api_key": "sk-...",
      "top_k": 5
    }
  ]
}
```

Results from all sources are deduplicated, reranked, and cited in one answer.
See the field descriptions in `lightrag/api/routers/query_routes.py` for the
full contract of each external KB type.

## Query Modes

| Mode | Description |
|---|---|
| `local` | Context-dependent retrieval focused on specific entities |
| `global` | Community/summary-based broad knowledge retrieval |
| `hybrid` | Combines local and global |
| `naive` | Direct vector search without graph |
| `mix` | Integrates KG and vector retrieval (recommended with a reranker) |

## Storage Backends

| Layer | Backends |
|---|---|
| KV (LLM cache, chunks, doc info) | JSON (default), Redis, PostgreSQL, MongoDB |
| Vector (embeddings) | NanoVectorDB (default), Faiss, Milvus, Qdrant, PostgreSQL, MongoDB |
| Graph (entity-relation) | NetworkX (default), Neo4j, PostgreSQL+AGE, MongoDB, Memgraph |
| Doc status | JSON (default), Redis, PostgreSQL, MongoDB |

Configure via environment variables — see [env.example](env.example) for the
full annotated list, including `LIGHTRAG_SSL_VERIFY` for deployments with
self-signed TLS certificates.

## Documentation

- [docs/LightRAG-upstream-usage.md](docs/LightRAG-upstream-usage.md) — inherited engine documentation (configuration, storage setup, API details)
- [docs/external_kb_usage.md](docs/external_kb_usage.md) — external knowledge base integration guide (Chinese)
- [docs/OfflineDeployment.md](docs/OfflineDeployment.md) — air-gapped deployment
- [docs/DockerDeployment.md](docs/DockerDeployment.md) — Docker deployment
- [docs/FrontendBuildGuide.md](docs/FrontendBuildGuide.md) — WebUI build
- [docs/InteractiveSetup.md](docs/InteractiveSetup.md) — interactive `.env` wizard
- [CONTRIBUTING.md](CONTRIBUTING.md) — development and PR guidelines

## Development

```bash
uv sync                  # core package
uv sync --extra api      # API server support
uv sync --extra test     # testing dependencies

uv run ruff check .      # lint
uv run pytest tests/     # offline test suite (no external services)
```

## License

MeshRAG is licensed under the [MIT License](LICENSE).
Copyright 2026 MeshRAG Team; original LightRAG code Copyright 2025 LightRAG Team.
See [NOTICE](NOTICE) for details.

## Acknowledgements

- [LightRAG (HKUDS)](https://github.com/HKUDS/LightRAG) — the engine this project is built on. If MeshRAG is useful to you, please also star the upstream project.
- LightRAG paper: [arXiv:2410.05779](https://arxiv.org/abs/2410.05779)
