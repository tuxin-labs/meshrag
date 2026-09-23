# Changelog

All notable changes to MeshRAG are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
