"""Comprehensive offline API tests against the REAL server app.

This module builds the actual ``create_app()`` FastAPI application with a REAL
``RAGManager`` whose rag factory hands out mock LightRAG instances (cached per
kb_id). That exercises the genuine manager logic (LRU, registry, KB discovery)
plus the real inline endpoints (knowledge-base management, auth-status/login/
health), the registry routers and the Ollama-compatibility router with their
authentic wiring — while remaining fully offline.

Covered surface:
- KB management: list/create/delete incl. 207 partial_success, 400 default-KB guard, stats
- Auth: /auth-status and /login guest mode, /health
- Registry: /llm_models and /external_kbs CRUD, secret masking, validation, probes
- Query integration: llm_profile_id resolution (model switching), external_kb_ids
  resolution, external-only scope
- Ollama compatibility: /api/version, /api/tags, /api/ps, /api/chat, /api/generate,
  LIGHTRAG-KB header KB selection
- Documents: text insert, paginated list, status counts, pipeline status, track status
"""

import asyncio
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

# Populated by the server_client fixture.
MOCK_MANAGER = None
MOCK_RAGS = {}


def _make_mock_rag():
    rag = MagicMock()
    rag.workspace = "default"
    rag.llm_model_kwargs = {}
    rag.initialize_storages = AsyncMock()
    rag.check_and_migrate_data = AsyncMock()
    rag.finalize_storages = AsyncMock()

    async def _llm_func(query, **kwargs):
        return "mock llm answer"

    rag.llm_model_func = AsyncMock(side_effect=_llm_func)
    rag.aquery = AsyncMock(return_value="mock rag answer")

    from lightrag.api.config import OllamaServerInfos

    rag.ollama_server_infos = OllamaServerInfos(name="meshrag", tag="latest")

    doc_status = MagicMock()
    doc_status.get_doc_by_file_path = AsyncMock(return_value=None)
    doc_status.get_by_id = AsyncMock(return_value=None)
    doc_status.get_all_status_counts = AsyncMock(
        return_value={"all": 2, "processed": 2}
    )
    rag.doc_status = doc_status

    graph = MagicMock()
    graph.get_popular_labels = AsyncMock(return_value=[])
    graph.get_all_labels = AsyncMock(return_value=[])
    rag.chunk_entity_relation_graph = graph

    rag.apipeline_enqueue_documents = AsyncMock()
    rag.apipeline_process_enqueue_documents = AsyncMock()
    rag.apipeline_enqueue_error_documents = AsyncMock()
    return rag


def _mock_rag(kb_id="default"):
    return MOCK_RAGS[kb_id]


@pytest.fixture(scope="module")
def server_client():
    """Build the real application on top of a REAL RAGManager with mock rags."""
    global MOCK_MANAGER, MOCK_RAGS
    from lightrag.api import lightrag_server
    from lightrag.api.rag_manager import RAGManager
    from lightrag.api.config import parse_args
    from lightrag.kg.shared_storage import (
        initialize_share_data,
        finalize_share_data,
        set_default_workspace,
        initialize_pipeline_status,
    )

    mp = pytest.MonkeyPatch()
    tmp = tempfile.mkdtemp(prefix="meshrag_server_api_test_")
    working_dir = os.path.join(tmp, "rag_storage")
    mp.setenv("WORKING_DIR", working_dir)
    mp.setenv("INPUT_DIR", os.path.join(tmp, "inputs"))
    mp.delenv("LIGHTRAG_API_KEY", raising=False)
    mp.delenv("AUTH_ACCOUNTS", raising=False)
    mp.setattr(sys, "argv", ["meshrag-server"])

    MOCK_RAGS = {}

    def _factory(kb_id):
        if kb_id not in MOCK_RAGS:
            MOCK_RAGS[kb_id] = _make_mock_rag()
        return MOCK_RAGS[kb_id]

    MOCK_MANAGER = RAGManager(
        rag_factory=_factory,
        default_kb="default",
        registry_path=os.path.join(working_dir, "knowledge_bases.json"),
        kb_discovery=lambda: [],
        file_stats_dir=working_dir,
    )

    # create_app constructs RAGManager(...) internally; hand back our pre-built
    # instance. Returning a base-class instance from a subclass __new__ skips
    # __init__ (isinstance check fails) while isinstance(rag, RAGManager) checks
    # elsewhere (OllamaAPI) still pass.
    class _ManagerFactory(RAGManager):
        def __new__(cls, **kwargs):
            return MOCK_MANAGER

    mp.setattr(lightrag_server, "RAGManager", _ManagerFactory)

    # The WebUI build output is gitignored, so its presence on disk must not
    # change app behavior under test (root redirect, /auth-status flags).
    # Pin the "frontend built" branch, which is how the server ships in
    # Docker and PyPI artifacts.
    mp.setattr(lightrag_server, "check_frontend_build", lambda: (True, False))

    initialize_share_data(workers=1)
    # The real server gets the default workspace from LightRAG.__post_init__; with
    # mock rags the namespace endpoints (/health, /documents/pipeline_status)
    # need it set explicitly.
    set_default_workspace("default")
    asyncio.run(initialize_pipeline_status("default"))
    args = parse_args()
    app = lightrag_server.create_app(args)

    with TestClient(app) as client:  # context manager runs the lifespan
        yield client

    finalize_share_data()
    mp.undo()
    shutil.rmtree(tmp, ignore_errors=True)


def _make_doc(doc_id="doc-1", status="processed"):
    from lightrag.base import DocStatus

    return SimpleNamespace(
        content_summary="A test document about graphs",
        content_length=32,
        status=DocStatus(status),
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:00",
        track_id="track-1",
        chunks_count=2,
        error_msg=None,
        metadata={},
        file_path="test.txt",
    )


@pytest.fixture()
def mock_doc_store():
    rag = _mock_rag()
    store = MagicMock()
    store.get_docs_paginated = AsyncMock(return_value=(([("doc-1", _make_doc())], 1)))
    store.get_all_status_counts = AsyncMock(
        return_value={"processed": 1, "pending": 0, "processing": 0, "failed": 0}
    )
    store.get_by_id = AsyncMock(return_value=None)
    store.get_doc_by_file_path = AsyncMock(return_value=None)
    rag.doc_status = store
    rag.ainsert = AsyncMock(return_value={"status": "success"})
    rag.aget_docs_by_track_id = AsyncMock(return_value={"doc-1": _make_doc()})
    # the rag object is cached module-wide, so background-pipeline call counters
    # must start from zero for every test
    rag.apipeline_enqueue_documents = AsyncMock()
    rag.apipeline_process_enqueue_documents = AsyncMock()
    rag.apipeline_enqueue_error_documents = AsyncMock()
    return store


# ---------------------------------------------------------------------------
# Knowledge base management (real inline endpoints + real RAGManager)
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_kb_list_contract(server_client):
    resp = server_client.get("/knowledge_bases")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["default_kb"] == "default"
    assert body["knowledge_bases"][0] == "default"


@pytest.mark.offline
def test_kb_create_initializes_knowledge_base(server_client):
    resp = server_client.post("/knowledge_bases", json={"kb_id": "kb_alpha"})
    assert resp.status_code == 200
    assert resp.json() == {"status": "success", "kb_id": "kb_alpha"}
    assert "kb_alpha" in MOCK_RAGS
    assert "kb_alpha" in MOCK_MANAGER.list_knowledge_bases()


@pytest.mark.offline
def test_kb_create_requires_body(server_client):
    resp = server_client.post("/knowledge_bases", json={})
    assert resp.status_code == 422


@pytest.mark.offline
def test_kb_delete_success(server_client, monkeypatch):
    monkeypatch.setattr(
        MOCK_MANAGER,
        "delete_knowledge_base",
        AsyncMock(return_value={"status": "success", "message": "deleted"}),
    )
    resp = server_client.delete("/knowledge_bases/kb_alpha")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


@pytest.mark.offline
def test_kb_delete_partial_success_returns_207(server_client, monkeypatch):
    monkeypatch.setattr(
        MOCK_MANAGER,
        "delete_knowledge_base",
        AsyncMock(
            return_value={
                "status": "partial_success",
                "message": "some storages failed",
                "storage_results": {"kv": "ok", "graph": "error"},
            }
        ),
    )
    resp = server_client.delete("/knowledge_bases/kb_partial")
    assert resp.status_code == 207
    body = resp.json()
    assert body["status"] == "partial_success"
    assert body["storage_results"]["graph"] == "error"


@pytest.mark.offline
def test_kb_delete_default_kb_rejected(server_client):
    # Real RAGManager refuses to delete the default KB.
    resp = server_client.delete("/knowledge_bases/default")
    assert resp.status_code == 400
    assert "default" in resp.json()["message"].lower()


@pytest.mark.offline
def test_kb_delete_internal_error_returns_500(server_client, monkeypatch):
    monkeypatch.setattr(
        MOCK_MANAGER,
        "delete_knowledge_base",
        AsyncMock(return_value={"status": "error", "message": "boom"}),
    )
    resp = server_client.delete("/knowledge_bases/kb_bad")
    assert resp.status_code == 500
    assert resp.json()["status"] == "error"


@pytest.mark.offline
def test_kb_stats(server_client):
    resp = server_client.get("/knowledge_bases/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["default_kb"] == "default"
    entries = {e["kb_id"]: e for e in body["knowledge_bases"]}
    assert "default" in entries
    assert entries["default"]["documents"] == 2
    assert entries["default"]["documents_by_status"] == {"processed": 2}


# ---------------------------------------------------------------------------
# Auth / health / root
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_auth_status_guest_mode(server_client):
    resp = server_client.get("/auth-status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["auth_configured"] is False
    assert body["auth_mode"] == "disabled"
    assert body["access_token"]
    assert body["token_type"] == "bearer"
    assert "webui_title" in body


@pytest.mark.offline
def test_login_guest_mode(server_client):
    resp = server_client.post(
        "/login", data={"username": "anyone", "password": "anything"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["auth_mode"] == "disabled"
    assert body["access_token"]


@pytest.mark.offline
def test_health_reports_config(server_client):
    resp = server_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert "llm_binding" in body["configuration"]
    assert "embedding_binding" in body["configuration"]
    assert body["auth_mode"] == "disabled"
    assert body["pipeline_busy"] is False


@pytest.mark.offline
def test_root_redirects_to_webui(server_client):
    resp = server_client.get("/", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert "/webui" in resp.headers["location"]


@pytest.mark.offline
def test_openapi_schema_documents_all_routers(server_client):
    resp = server_client.get("/openapi.json")
    assert resp.status_code == 200
    paths = resp.json()["paths"]
    for expected in (
        "/knowledge_bases",
        "/knowledge_bases/stats",
        "/query",
        "/query/stream",
        "/query/data",
        "/documents/upload",
        "/documents/text",
        "/documents/paginated",
        "/llm_models",
        "/external_kbs",
        "/api/chat",
        "/api/generate",
        "/graph/entity/edit",
        "/ext/retrieval",
        "/ext/rag",
    ):
        assert expected in paths, f"missing {expected} in openapi"


# ---------------------------------------------------------------------------
# Model profile registry
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_llm_models_crud_and_secret_masking(server_client):
    resp = server_client.post(
        "/llm_models",
        json={
            "name": "Doubao mini",
            "kind": "llm",
            "binding": "openai",
            "model": "doubao-seed-2.0-mini",
            "host": "https://ark.cn-beijing.volces.com/api/plan/v3",
            "api_key": "ark-test-key-1234567890abcd",
            "enabled": True,
        },
    )
    assert resp.status_code == 200
    created = resp.json()
    assert created["name"] == "Doubao mini"
    assert created["has_api_key"] is True
    assert created["api_key_tail"] == "abcd"
    assert "api_key" not in created
    profile_id = created["id"]

    listing = server_client.get("/llm_models").json()
    match = [p for p in listing if p["id"] == profile_id]
    assert match and "api_key" not in match[0]

    # update: omit api_key -> stored key preserved (tail unchanged)
    resp = server_client.put(
        f"/llm_models/{profile_id}",
        json={
            "name": "Doubao mini v2",
            "kind": "llm",
            "binding": "openai",
            "model": "doubao-seed-2.0-mini",
            "host": "https://ark.cn-beijing.volces.com/api/plan/v3",
            "enabled": False,
        },
    )
    assert resp.status_code == 200
    updated = resp.json()
    assert updated["name"] == "Doubao mini v2"
    assert updated["enabled"] is False
    assert updated["has_api_key"] is True
    assert updated["api_key_tail"] == "abcd"

    # update: empty api_key -> cleared
    resp = server_client.put(
        f"/llm_models/{profile_id}",
        json={
            "name": "Doubao mini v2",
            "kind": "llm",
            "binding": "openai",
            "model": "doubao-seed-2.0-mini",
            "host": "https://ark.cn-beijing.volces.com/api/plan/v3",
            "api_key": "",
            "enabled": True,
        },
    )
    assert resp.json()["has_api_key"] is False

    resp = server_client.delete(f"/llm_models/{profile_id}")
    assert resp.status_code == 200
    assert server_client.delete(f"/llm_models/{profile_id}").status_code == 404


@pytest.mark.offline
def test_llm_models_validation(server_client):
    bad_kind = {
        "name": "x",
        "kind": "vision",
        "binding": "openai",
        "model": "m",
        "host": "http://h",
    }
    assert server_client.post("/llm_models", json=bad_kind).status_code == 400

    bad_binding = {**bad_kind, "kind": "llm", "binding": "anthropic"}
    assert server_client.post("/llm_models", json=bad_binding).status_code == 400

    no_dim = {
        "name": "emb",
        "kind": "embedding",
        "binding": "openai",
        "model": "emb-model",
        "host": "http://h",
    }
    assert server_client.post("/llm_models", json=no_dim).status_code == 400

    good_dim = {**no_dim, "embedding_dim": 1024}
    resp = server_client.post("/llm_models", json=good_dim)
    assert resp.status_code == 200
    assert resp.json()["embedding_dim"] == 1024

    missing_name = {
        "kind": "llm",
        "binding": "openai",
        "model": "m",
        "host": "http://h",
    }
    assert server_client.post("/llm_models", json=missing_name).status_code == 422

    assert (
        server_client.put(
            "/llm_models/nonexistent",
            json={
                "name": "x",
                "kind": "llm",
                "binding": "openai",
                "model": "m",
                "host": "http://h",
            },
        ).status_code
        == 404
    )


# ---------------------------------------------------------------------------
# Model profile probe (connection test) with a fake httpx transport
# ---------------------------------------------------------------------------


class _FakeAsyncClient:
    """Minimal stand-in for httpx.AsyncClient honouring the probe call shape."""

    handler = None  # set by tests: (method, url) -> httpx.Response | raises

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, **kwargs):
        return self.handler("GET", url)

    async def post(self, url, **kwargs):
        return self.handler("POST", url)


@pytest.fixture()
def fake_probe_http(monkeypatch):
    from lightrag.api.routers import registry_routes

    monkeypatch.setattr(registry_routes.httpx, "AsyncClient", _FakeAsyncClient)
    return _FakeAsyncClient


def _response(method, url, status_code, json_body=None, text=None):
    request = httpx.Request(method, url)
    if json_body is not None:
        return httpx.Response(status_code, json=json_body, request=request)
    return httpx.Response(status_code, text=text or "", request=request)


@pytest.mark.offline
def test_probe_ollama_lists_model(server_client, fake_probe_http):
    created = server_client.post(
        "/llm_models",
        json={
            "name": "ollama local",
            "kind": "llm",
            "binding": "ollama",
            "model": "qwen3:latest",
            "host": "http://ollama.local:11434",
        },
    ).json()

    def handler(method, url):
        assert url == "http://ollama.local:11434/api/tags"
        return _response(method, url, 200, {"models": [{"name": "qwen3:latest"}]})

    fake_probe_http.handler = staticmethod(handler)
    resp = server_client.post(f"/llm_models/{created['id']}/test")
    assert resp.status_code == 200
    body = resp.json()
    assert body["reachable"] is True
    assert body["model_available"] is True
    assert "latency_ms" in body and body["id"] == created["id"]


@pytest.mark.offline
def test_probe_ollama_model_missing(server_client, fake_probe_http):
    created = server_client.post(
        "/llm_models",
        json={
            "name": "ollama missing",
            "kind": "llm",
            "binding": "ollama",
            "model": "nope",
            "host": "http://ollama.local:11434",
        },
    ).json()

    def handler(method, url):
        return _response(method, url, 200, {"models": [{"name": "other:latest"}]})

    fake_probe_http.handler = staticmethod(handler)
    body = server_client.post(f"/llm_models/{created['id']}/test").json()
    assert body["reachable"] is True
    assert body["model_available"] is False


@pytest.mark.offline
def test_probe_embedding_dimension_mismatch(server_client, fake_probe_http):
    created = server_client.post(
        "/llm_models",
        json={
            "name": "emb probe",
            "kind": "embedding",
            "binding": "openai",
            "model": "emb",
            "host": "https://emb.example.com",
            "embedding_dim": 4,
        },
    ).json()

    def handler(method, url):
        assert url == "https://emb.example.com/embeddings"
        return _response(method, url, 200, {"data": [{"embedding": [0.1, 0.2]}]})

    fake_probe_http.handler = staticmethod(handler)
    body = server_client.post(f"/llm_models/{created['id']}/test").json()
    assert body["reachable"] is True
    assert body["model_available"] is True
    assert "Dimension mismatch" in body["message"]
    assert body["detail"] == {"returned_dim": 2, "configured_dim": 4}


@pytest.mark.offline
def test_probe_embedding_ok(server_client, fake_probe_http):
    created = server_client.post(
        "/llm_models",
        json={
            "name": "emb ok",
            "kind": "embedding",
            "binding": "openai",
            "model": "emb",
            "host": "https://emb.example.com",
            "embedding_dim": 3,
        },
    ).json()

    def handler(method, url):
        return _response(method, url, 200, {"data": [{"embedding": [0.1, 0.2, 0.3]}]})

    fake_probe_http.handler = staticmethod(handler)
    body = server_client.post(f"/llm_models/{created['id']}/test").json()
    assert "Embedding OK (dim=3)" in body["message"]


@pytest.mark.offline
def test_probe_chat_upstream_error_maps_to_502(server_client, fake_probe_http):
    created = server_client.post(
        "/llm_models",
        json={
            "name": "chat bad",
            "kind": "llm",
            "binding": "openai",
            "model": "m",
            "host": "https://chat.example.com",
        },
    ).json()

    def handler(method, url):
        return _response(method, url, 401, json_body={"error": "bad key"})

    fake_probe_http.handler = staticmethod(handler)
    resp = server_client.post(f"/llm_models/{created['id']}/test")
    assert resp.status_code == 502


@pytest.mark.offline
def test_probe_connection_failure_reports_unreachable(server_client, fake_probe_http):
    created = server_client.post(
        "/llm_models",
        json={
            "name": "dead host",
            "kind": "llm",
            "binding": "openai",
            "model": "m",
            "host": "https://dead.example.com",
        },
    ).json()

    def handler(method, url):
        raise httpx.ConnectError("refused")

    fake_probe_http.handler = staticmethod(handler)
    body = server_client.post(f"/llm_models/{created['id']}/test").json()
    assert body["reachable"] is False
    assert body["model_available"] is False
    assert "Connection failed" in body["message"]


@pytest.mark.offline
def test_probe_unknown_profile_404(server_client):
    assert server_client.post("/llm_models/nonexistent/test").status_code == 404


# ---------------------------------------------------------------------------
# External KB registry
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_external_kbs_crud_and_masking(server_client):
    resp = server_client.post(
        "/external_kbs",
        json={
            "name": "Partner retrieval",
            "type": "retrieval",
            "url": "https://partner.example.com/retrieve",
            "api_key": "partner-secret-key-9999",
            "top_k": 8,
        },
    )
    assert resp.status_code == 200
    created = resp.json()
    assert created["has_api_key"] is True
    assert created["api_key_tail"] == "9999"
    assert "api_key" not in created
    kb_id = created["id"]

    listing = server_client.get("/external_kbs").json()
    assert any(k["id"] == kb_id for k in listing)

    resp = server_client.put(
        f"/external_kbs/{kb_id}",
        json={
            "name": "Partner retrieval",
            "type": "retrieval",
            "url": "https://partner.example.com/retrieve",
            "top_k": 12,
            "enabled": False,
        },
    )
    assert resp.status_code == 200
    updated = resp.json()
    assert updated["top_k"] == 12
    assert updated["enabled"] is False
    # key preserved on update without api_key field
    assert updated["has_api_key"] is True

    assert server_client.delete(f"/external_kbs/{kb_id}").status_code == 200
    assert server_client.delete(f"/external_kbs/{kb_id}").status_code == 404


@pytest.mark.offline
def test_external_kbs_validation(server_client):
    bad_type = {"name": "x", "type": "vector", "url": "https://u"}
    assert server_client.post("/external_kbs", json=bad_type).status_code == 400

    for bad_top_k in (0, 51):
        payload = {
            "name": "x",
            "type": "retrieval",
            "url": "https://u",
            "top_k": bad_top_k,
        }
        assert server_client.post("/external_kbs", json=payload).status_code in (
            400,
            422,
        )

    missing_url = {"name": "x", "type": "retrieval"}
    assert server_client.post("/external_kbs", json=missing_url).status_code == 422


@pytest.mark.offline
def test_external_kb_probes(server_client, fake_probe_http):
    retrieval = server_client.post(
        "/external_kbs",
        json={
            "name": "ret probe",
            "type": "retrieval",
            "url": "https://ret.example.com",
        },
    ).json()
    rag_kb = server_client.post(
        "/external_kbs",
        json={"name": "rag probe", "type": "rag", "url": "https://rag.example.com"},
    ).json()

    def handler(method, url):
        if "ret.example.com" in url:
            return _response(
                method,
                url,
                200,
                {"status": "success", "results": [{"content": "c"}] * 3},
            )
        if "rag.example.com" in url:
            return _response(
                method, url, 200, {"status": "success", "answer": "the answer"}
            )

    fake_probe_http.handler = staticmethod(handler)
    body = server_client.post(f"/external_kbs/{retrieval['id']}/test").json()
    assert body["reachable"] is True and body["model_available"] is True
    assert body["detail"]["result_count"] == 3

    body = server_client.post(f"/external_kbs/{rag_kb['id']}/test").json()
    assert body["reachable"] is True and body["model_available"] is True

    def failing(method, url):
        return _response(method, url, 500, text="server error")

    fake_probe_http.handler = staticmethod(failing)
    resp = server_client.post(f"/external_kbs/{retrieval['id']}/test")
    assert resp.status_code == 502

    def bad_payload(method, url):
        return _response(method, url, 200, {"status": "failure"})

    fake_probe_http.handler = staticmethod(bad_payload)
    body = server_client.post(f"/external_kbs/{rag_kb['id']}/test").json()
    assert body["reachable"] is True and body["model_available"] is False


# ---------------------------------------------------------------------------
# Query-time registry integration (model switching + external KB resolution)
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_query_resolves_llm_profile(server_client, monkeypatch):
    created = server_client.post(
        "/llm_models",
        json={
            "name": "switch target",
            "kind": "llm",
            "binding": "openai",
            "model": "doubao-seed-2.0-mini",
            "host": "https://ark.example.com/v3",
            "api_key": "ark-profile-secret-777",
        },
    ).json()

    calls = []

    async def capture(query, kb_ids, param=None, external_kbs=None):
        calls.append({"query": query, "kb_ids": kb_ids, "param": param})
        return {
            "status": "success",
            "message": "",
            "data": {
                "references": [],
                "entities": [],
                "relationships": [],
                "chunks": [],
            },
            "metadata": {},
            "llm_response": {"content": "ok", "is_streaming": False},
        }

    monkeypatch.setattr(MOCK_MANAGER, "multi_kb_query", capture)

    resp = server_client.post(
        "/query", json={"query": "what is graph rag", "llm_profile_id": created["id"]}
    )
    assert resp.status_code == 200
    assert resp.json()["response"] == "ok"

    assert len(calls) == 1
    param = calls[0]["param"]
    # the profile selection must inject a dynamic model func (model switching)
    assert param.model_func is not None
    assert "doubao-seed-2.0-mini" in param.model_func_id
    assert calls[0]["kb_ids"] == ["default"]


@pytest.mark.offline
def test_query_llm_profile_errors(server_client):
    embedding_profile = server_client.post(
        "/llm_models",
        json={
            "name": "emb not llm",
            "kind": "embedding",
            "binding": "openai",
            "model": "emb",
            "host": "https://e.example.com",
            "embedding_dim": 64,
        },
    ).json()
    resp = server_client.post(
        "/query",
        json={"query": "hello graph", "llm_profile_id": embedding_profile["id"]},
    )
    assert resp.status_code == 400
    assert "not an LLM profile" in resp.json()["detail"]

    resp = server_client.post(
        "/query", json={"query": "hello graph", "llm_profile_id": "ghost"}
    )
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


@pytest.mark.offline
def test_query_unknown_kb_id_404_without_resurrection(server_client, monkeypatch):
    """A query scoped entirely to unknown KBs must 404 and never register them."""
    registered_before = set(MOCK_MANAGER.list_knowledge_bases())

    resp = server_client.post(
        "/query", json={"query": "hello graph", "kb_ids": ["ghost_kb"], "mode": "naive"}
    )
    assert resp.status_code == 404
    assert "ghost_kb" in resp.json()["detail"]
    assert set(MOCK_MANAGER.list_knowledge_bases()) == registered_before

    # same contract on the retrieval-only endpoint
    resp = server_client.post(
        "/query/data", json={"query": "hello graph", "kb_ids": ["ghost_kb"]}
    )
    assert resp.status_code == 404
    assert set(MOCK_MANAGER.list_knowledge_bases()) == registered_before


@pytest.mark.offline
def test_query_resolves_external_kb_ids(server_client, monkeypatch):
    ext = server_client.post(
        "/external_kbs",
        json={
            "name": "registered ext",
            "type": "retrieval",
            "url": "https://ext.example.com/r",
            "api_key": "ext-key-secret-42",
            "top_k": 7,
        },
    ).json()
    disabled = server_client.post(
        "/external_kbs",
        json={
            "name": "disabled ext",
            "type": "rag",
            "url": "https://ext.example.com/g",
            "enabled": False,
        },
    ).json()

    calls = []

    async def capture(query, kb_ids, param=None, external_kbs=None):
        calls.append({"kb_ids": kb_ids, "external_kbs": external_kbs})
        return {
            "status": "success",
            "message": "",
            "data": {
                "references": [],
                "entities": [],
                "relationships": [],
                "chunks": [],
            },
            "metadata": {},
            "llm_response": {"content": "ok", "is_streaming": False},
        }

    monkeypatch.setattr(MOCK_MANAGER, "multi_kb_query", capture)

    # registered external KB expands into inline config; without explicit kb_ids
    # the query is external-only (internal KBs are skipped)
    resp = server_client.post(
        "/query", json={"query": "hybrid search", "external_kb_ids": [ext["id"]]}
    )
    assert resp.status_code == 200
    ext_kbs = calls[-1]["external_kbs"]
    assert ext_kbs and ext_kbs[0]["url"] == "https://ext.example.com/r"
    assert ext_kbs[0]["api_key"] == "ext-key-secret-42"
    assert ext_kbs[0]["top_k"] == 7
    assert calls[-1]["kb_ids"] == []

    # explicit kb_ids keep internal scope and merge the external KB (hybrid)
    resp = server_client.post(
        "/query",
        json={
            "query": "hybrid search",
            "kb_ids": ["default"],
            "external_kb_ids": [ext["id"]],
        },
    )
    assert resp.status_code == 200
    assert calls[-1]["kb_ids"] == ["default"]
    assert calls[-1]["external_kbs"][0]["url"] == "https://ext.example.com/r"

    # disabled external KB is skipped -> default kb scope
    resp = server_client.post(
        "/query", json={"query": "hybrid search", "external_kb_ids": [disabled["id"]]}
    )
    assert resp.status_code == 200
    assert calls[-1]["external_kbs"] is None
    assert calls[-1]["kb_ids"] == ["default"]

    # unknown id -> 404
    resp = server_client.post(
        "/query", json={"query": "hybrid search", "external_kb_ids": ["ghost"]}
    )
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Mock external KB endpoints (/ext/retrieval, /ext/rag)
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_ext_mock_endpoints_auth_and_shape(server_client):
    resp = server_client.post("/ext/retrieval", json={"query": "q", "top_k": 3})
    assert resp.status_code in (401, 403)

    headers = {"Authorization": "Bearer ext-kb-test-sk-2026"}
    resp = server_client.post(
        "/ext/retrieval", json={"query": "q", "top_k": 3}, headers=headers
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    assert isinstance(resp.json()["results"], list)

    resp = server_client.post("/ext/rag", json={"query": "q"}, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    assert "answer" in resp.json()


# ---------------------------------------------------------------------------
# Ollama compatibility API
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_ollama_version_tags_ps(server_client):
    assert server_client.get("/api/version").json()["version"] == "0.9.3"

    tags = server_client.get("/api/tags").json()
    assert len(tags["models"]) == 1
    assert tags["models"][0]["name"].startswith("meshrag")

    ps = server_client.get("/api/ps").json()
    assert ps["models"][0]["name"].startswith("meshrag")


@pytest.mark.offline
def test_ollama_kb_header_selects_knowledge_base(server_client):
    # an explicit header must name an existing KB: create it first
    server_client.post("/knowledge_bases", json={"kb_id": "kb_via_header"})
    server_client.get("/api/tags", headers={"LIGHTRAG-KB": "kb_via_header"})
    assert "kb_via_header" in MOCK_RAGS
    server_client.get("/api/tags")
    assert "default" in MOCK_RAGS


@pytest.mark.offline
def test_ollama_stale_kb_header_404_without_resurrection(server_client):
    registered_before = set(MOCK_MANAGER.list_knowledge_bases())
    resp = server_client.get("/api/tags", headers={"LIGHTRAG-KB": "ghost_kb"})
    assert resp.status_code == 404
    assert "ghost_kb" in resp.json()["detail"]
    assert set(MOCK_MANAGER.list_knowledge_bases()) == registered_before


@pytest.mark.offline
def test_ollama_chat_non_stream_routes_through_rag(server_client):
    rag = _mock_rag()
    rag.aquery = AsyncMock(return_value="graph-rag answer")

    resp = server_client.post(
        "/api/chat",
        json={
            "model": "meshrag:latest",
            "messages": [
                {"role": "user", "content": "/local what is entity extraction"}
            ],
            "stream": False,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["message"]["content"] == "graph-rag answer"
    assert body["done"] is True

    rag.aquery.assert_awaited_once()
    args, kwargs = rag.aquery.call_args
    assert args[0] == "what is entity extraction"  # prefix stripped
    assert kwargs["param"].mode == "local"


@pytest.mark.offline
def test_ollama_chat_requires_user_message(server_client):
    resp = server_client.post(
        "/api/chat",
        json={"model": "m", "messages": [{"role": "assistant", "content": "hi"}]},
    )
    assert resp.status_code == 400


@pytest.mark.offline
def test_ollama_generate_passthrough(server_client):
    rag = _mock_rag()

    async def _llm(query, **kwargs):
        return "direct llm output"

    rag.llm_model_func = AsyncMock(side_effect=_llm)
    rag.llm_model_kwargs = {}

    resp = server_client.post(
        "/api/generate",
        json={"model": "meshrag:latest", "prompt": "tell me a fact", "stream": False},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["response"] == "direct llm output"
    assert body["done"] is True
    rag.llm_model_func.assert_awaited_once()


# ---------------------------------------------------------------------------
# Document management endpoints (HTTP contract with a mocked doc_status store)
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_document_text_insert_requires_kb_id(server_client, mock_doc_store):
    resp = server_client.post("/documents/text", json={"text": "hello graph world"})
    assert resp.status_code == 422

    resp = server_client.post(
        "/documents/text?kb_id=default",
        json={"text": "hello graph world", "file_source": "note.txt"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["track_id"]
    _mock_rag().apipeline_enqueue_documents.assert_awaited_once()


@pytest.mark.offline
def test_document_text_insert_rejects_blank(server_client, mock_doc_store):
    resp = server_client.post("/documents/text?kb_id=default", json={"text": ""})
    assert resp.status_code in (400, 422)


@pytest.mark.offline
def test_documents_paginated(server_client, mock_doc_store):
    resp = server_client.post(
        "/documents/paginated?kb_id=default",
        json={"page": 1, "page_size": 10},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["pagination"]["total_count"] == 1
    assert body["pagination"]["page"] == 1
    assert body["documents"][0]["id"] == "doc-1"
    assert body["documents"][0]["status"] == "processed"
    assert body["status_counts"]["processed"] == 1


@pytest.mark.offline
def test_documents_status_counts(server_client, mock_doc_store):
    resp = server_client.get("/documents/status_counts?kb_id=default")
    assert resp.status_code == 200
    assert resp.json()["status_counts"]["processed"] == 1


@pytest.mark.offline
def test_documents_pipeline_status_idle(server_client, mock_doc_store):
    resp = server_client.get("/documents/pipeline_status?kb_id=default")
    assert resp.status_code == 200
    body = resp.json()
    assert body["busy"] is False
    assert "history_messages" in body
    assert "update_status" in body


@pytest.mark.offline
def test_documents_track_status(server_client, mock_doc_store):
    resp = server_client.get("/documents/track_status/track-1?kb_id=default")
    assert resp.status_code == 200
    body = resp.json()
    assert body["track_id"] == "track-1"
    assert body["total_count"] == 1
    assert body["documents"][0]["id"] == "doc-1"


@pytest.mark.offline
def test_documents_batch_texts(server_client, mock_doc_store):
    rag = _mock_rag()
    resp = server_client.post(
        "/documents/texts?kb_id=default",
        json={
            "texts": ["alpha beta gamma", "delta epsilon zeta"],
            "file_sources": ["a.txt", "b.txt"],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "success"
    assert body["track_id"]
    rag.apipeline_enqueue_documents.assert_awaited_once()


@pytest.mark.offline
def test_document_update_success(server_client, mock_doc_store):
    from lightrag.base import DocStatus

    rag = _mock_rag()

    class _AttrDict(dict):
        # handler indexes the record like a dict, the pipeline task reads .status
        __getattr__ = dict.__getitem__

    mock_doc_store.get_doc_by_file_path = AsyncMock(
        return_value=_AttrDict(id="doc-9", status=DocStatus.PROCESSED)
    )
    rag.full_docs = MagicMock()
    rag.full_docs.get_by_id = AsyncMock(return_value={"content": "old content"})
    from lightrag.base import DeletionResult

    rag.adelete_by_doc_id = AsyncMock(
        return_value=DeletionResult(status="success", doc_id="doc-9", message="deleted")
    )

    resp = server_client.put(
        "/documents/text/update?kb_id=default",
        json={"text": "brand new content", "file_source": "note.txt"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "success"
    assert body["doc_id"] == "doc-9"
    assert body["track_id"]
    # the background update task deletes the old doc and re-enqueues
    rag.adelete_by_doc_id.assert_awaited_once()
    rag.apipeline_enqueue_documents.assert_awaited_once()


@pytest.mark.offline
def test_document_update_not_found(server_client, mock_doc_store):
    resp = server_client.put(
        "/documents/text/update?kb_id=default",
        json={"text": "anything", "file_source": "missing.txt"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "not_found"


@pytest.mark.offline
def test_document_update_unchanged(server_client, mock_doc_store):
    rag = _mock_rag()
    mock_doc_store.get_doc_by_file_path = AsyncMock(
        return_value={"id": "doc-9", "status": "processed"}
    )
    rag.full_docs = MagicMock()
    rag.full_docs.get_by_id = AsyncMock(return_value={"content": "same text"})

    resp = server_client.put(
        "/documents/text/update?kb_id=default",
        json={"text": "same text", "file_source": "note.txt"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "unchanged"
    assert body["doc_id"] == "doc-9"


@pytest.mark.offline
def test_delete_entity_success_and_not_found(server_client, mock_doc_store):
    from lightrag.base import DeletionResult

    rag = _mock_rag()

    rag.adelete_by_entity = AsyncMock(
        return_value=DeletionResult(status="success", doc_id="", message="deleted")
    )
    resp = server_client.request(
        "DELETE", "/documents/delete_entity?kb_id=default", json={"entity_name": "地球"}
    )
    assert resp.status_code == 200
    rag.adelete_by_entity.assert_awaited_once_with(entity_name="地球")

    rag.adelete_by_entity = AsyncMock(
        return_value=DeletionResult(
            status="not_found", doc_id="", message="no such entity"
        )
    )
    resp = server_client.request(
        "DELETE",
        "/documents/delete_entity?kb_id=default",
        json={"entity_name": "ghost"},
    )
    assert resp.status_code == 404


@pytest.mark.offline
def test_delete_relation_success(server_client, mock_doc_store):
    from lightrag.base import DeletionResult

    rag = _mock_rag()
    rag.adelete_by_relation = AsyncMock(
        return_value=DeletionResult(status="success", doc_id="", message="deleted")
    )
    resp = server_client.request(
        "DELETE",
        "/documents/delete_relation?kb_id=default",
        json={"source_entity": "地球", "target_entity": "火星"},
    )
    assert resp.status_code == 200
    rag.adelete_by_relation.assert_awaited_once()


@pytest.mark.offline
def test_clear_cache(server_client, mock_doc_store):
    rag = _mock_rag()
    rag.aclear_cache = AsyncMock()
    resp = server_client.post("/documents/clear_cache?kb_id=default", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert "cache" in body["message"].lower()
    rag.aclear_cache.assert_awaited_once()


# ---------------------------------------------------------------------------
# Zombie-KB guard: read-only endpoints must not lazily recreate deleted KBs
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_readonly_endpoints_reject_unknown_kb(server_client):
    for method, path, kwargs in [
        ("get", "/documents/status_counts?kb_id=ghost_kb", {}),
        ("get", "/documents/pipeline_status?kb_id=ghost_kb", {}),
        (
            "post",
            "/documents/paginated?kb_id=ghost_kb",
            {"json": {"page": 1, "page_size": 10}},
        ),
        ("get", "/documents/track_status/t1?kb_id=ghost_kb", {}),
        ("get", "/graph/label/list?kb_id=ghost_kb", {}),
        ("get", "/graphs?kb_id=ghost_kb&label=x", {}),
    ]:
        resp = getattr(server_client, method)(path, **kwargs)
        assert resp.status_code == 404, (
            method,
            path,
            resp.status_code,
            resp.text[:120],
        )
        assert "not found" in resp.json()["detail"].lower()

    # the unknown id must NOT have been registered by any of the polls above
    assert "ghost_kb" not in MOCK_MANAGER.list_knowledge_bases()


@pytest.mark.offline
def test_manager_known_kb_semantics(server_client):
    # default counts as known even before any explicit use; unknown ids do not
    assert MOCK_MANAGER.has_knowledge_base("default") is True
    assert MOCK_MANAGER.has_knowledge_base("ghost_kb") is False
    assert asyncio.run(MOCK_MANAGER.get_rag_or_none("ghost_kb")) is None
