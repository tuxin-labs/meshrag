# Changelog

All notable changes to MeshRAG are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.1.0] — 2026-10-09

Security-hardening release: clears all GitHub Dependabot and CodeQL alerts.

### Security

- **Path injection**: `DocumentManager` now validates that the workspace
  resolves inside the base input directory (`realpath` + prefix check)
  before any filesystem access.
- **Stack-trace exposure**: exception details are no longer echoed into
  HTTP/streaming error responses (7 sinks across query, document, registry
  and Ollama-compatible routes); details stay in server logs.
- **Workflow permissions**: CI workflows now declare
  `permissions: contents: read`.
- MD5 usage for content IDs/cache keys is marked `usedforsecurity=False`
  (IDs are not security-relevant; the digest must stay for data
  compatibility).

### Fixed

- `GET /documents` returned a 500 validation error for knowledge bases
  containing text-inserted documents: the pipeline stores `file_path=None`
  for inserts without a backing file, but the response model required a
  string. The field is now optional (the WebUI already handles a missing
  path).

### Changed

- Dependency floors raised so fresh installs cannot resolve into known
  vulnerable ranges: `pypdf>=6.19.0`, `PyJWT>=2.15.0`,
  `python-multipart>=0.0.31`, `docling>=2.94.0`, `ragas>=0.4.3`,
  `datasets>=5.0.1`, `pytest>=9.0.3`.
- `uv.lock` re-resolved (uv 0.12.24): 48 packages bumped to patched
  releases, clearing all 287 Dependabot alerts. Notable jumps:
  transformers 5.19, torch 2.14, starlette 1.7 (via fastapi),
  docling 2.135, langchain family 1.6.9 (via ragas 0.4.3),
  cryptography 50, aiohttp 3.14, nltk 3.10, pillow 12.3.
  GitPython left the dependency tree entirely.
- `pyjwt>=2.15` is enforced through a `[tool.uv]` override because
  `zhipuai` still pins `pyjwt<2.9` upstream while PyJWT <2.15 carries
  token-forgery and DoS advisories.
- WebUI: katex bumped to 0.18.11 via a package.json override
  (rehype-katex upstream still pins katex ^0.16).

## [1.0.0] — Initial MeshRAG release

Forked from [LightRAG](https://github.com/HKUDS/LightRAG) v1.4.11
(upstream state as of 2026-03-08), MIT licensed.

### Added

- **Multi-knowledge-base federated query**: query multiple local knowledge
  bases and external KB APIs in a single request — per-KB parallel retrieval,
  global deduplication, cross-KB budget control, chunk reranking, and
  citation/reference rebuilding.
- **External knowledge base integration**: async HTTP client for federating
  external retrieval APIs into the unified query flow.
- **Document management APIs**: upload/status pipeline plus fast lookup by
  file path (`GET /documents/by_file_path`) for large document sets.
- **Deletion/parsing concurrency pipeline**: cooperative cancellation and
  deletion semantics with crash recovery (`deletion_pending` priority,
  resumable processing states).
- **Configurable TLS verification** (`LIGHTRAG_SSL_VERIFY`) for deployments
  with self-signed certificates.

### Changed

- Distribution renamed from `lightrag-hku` to `meshrag`; CLI entry points
  renamed to `meshrag-server`, `meshrag-gunicorn`, `meshrag-download-cache`,
  `meshrag-clean-llmqc`. The Python module is still `import lightrag` for
  compatibility with the upstream ecosystem.
