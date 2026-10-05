"""Authentication-mode tests against the REAL server app.

Covers the two protection modes that the default offline suite never exercises
(the main server fixture runs with auth fully disabled):

1. API-key mode (``LIGHTRAG_API_KEY``): 403 without/with-wrong key, whitelisted
   paths stay open, KB management endpoints stay open (no auth dependency).
2. JWT account mode (``AUTH_ACCOUNTS``): /auth-status reports configured,
   /login rejects bad credentials and issues tokens for good ones, protected
   routes need a valid Bearer token, guest tokens are refused.

Each fixture builds its own ``create_app()`` instance in-process — safe since
the route factories create per-call routers (see document/query/graph routes).
"""

import asyncio
import os
import shutil
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from test_server_app_api import MOCK_RAGS, _make_mock_rag  # noqa: F401  (reuse the mock factory)


def _shared_manager():
    from lightrag.api.rag_manager import RAGManager

    MOCK_RAGS.setdefault("default", _make_mock_rag())
    manager = RAGManager.__new__(RAGManager)
    manager.default_kb = "default"
    manager.list_knowledge_bases = MagicMock(return_value=["default"])
    manager.get_rag = AsyncMock(return_value=MOCK_RAGS["default"])
    manager.initialize_default = AsyncMock()
    manager.finalize_all = AsyncMock()
    manager.delete_knowledge_base = AsyncMock(
        return_value={"status": "success", "message": "deleted"}
    )

    async def _list_kb_stats():
        return []

    manager.list_kb_stats = _list_kb_stats

    async def _multi_kb_query(query, kb_ids, param=None, external_kbs=None):
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

    manager.multi_kb_query = _multi_kb_query

    async def _multi_kb_get_data(query, kb_ids, param=None, external_kbs=None):
        return {
            "status": "success",
            "message": "",
            "data": {
                "entities": [],
                "relationships": [],
                "chunks": [],
                "references": [],
            },
            "metadata": {},
        }

    manager.multi_kb_get_data = _multi_kb_get_data
    return manager


def _build_client(monkeypatch, *, api_key=None, accounts=None):
    """Build a fresh create_app() with the requested auth configuration."""
    from lightrag.api import lightrag_server, utils_api
    from lightrag.api.auth import auth_handler
    from lightrag.api.config import parse_args
    from lightrag.kg.shared_storage import (
        initialize_share_data,
        set_default_workspace,
        initialize_pipeline_status,
    )

    tmp = tempfile.mkdtemp(prefix="meshrag_auth_test_")
    monkeypatch.setenv("WORKING_DIR", os.path.join(tmp, "rag_storage"))
    monkeypatch.setenv("INPUT_DIR", os.path.join(tmp, "inputs"))
    monkeypatch.setattr(sys, "argv", ["meshrag-server"])

    manager = _shared_manager()

    class _ManagerFactory:
        def __new__(cls, **kwargs):
            return manager

    monkeypatch.setattr(lightrag_server, "RAGManager", _ManagerFactory)

    if api_key is not None:
        monkeypatch.setenv("LIGHTRAG_API_KEY", api_key)
    else:
        monkeypatch.delenv("LIGHTRAG_API_KEY", raising=False)

    if accounts is not None:
        monkeypatch.setattr(auth_handler, "accounts", dict(accounts))
        monkeypatch.setattr(utils_api, "auth_configured", True)
    else:
        monkeypatch.setattr(auth_handler, "accounts", {})
        monkeypatch.setattr(utils_api, "auth_configured", False)

    initialize_share_data(workers=1)
    set_default_workspace("default")
    asyncio.run(initialize_pipeline_status("default"))

    args = parse_args()
    app = lightrag_server.create_app(args)

    def teardown():
        shutil.rmtree(tmp, ignore_errors=True)

    # attach teardown to the client via finalizer handled by caller fixture
    return TestClient(app, raise_server_exceptions=False), teardown


@pytest.fixture()
def api_key_client(monkeypatch):
    client, teardown = _build_client(monkeypatch, api_key="unit-test-key-42")
    yield client
    teardown()


@pytest.fixture()
def jwt_client(monkeypatch):
    client, teardown = _build_client(
        monkeypatch, accounts={"admin": "s3cret", "alice": "wonderland"}
    )
    yield client
    teardown()


PROTECTED_PATHS = [
    ("GET", "/documents/status_counts?kb_id=default"),
    ("GET", "/graph/label/list?kb_id=default"),
]


@pytest.mark.offline
class TestApiKeyMode:
    def test_missing_key_is_rejected(self, api_key_client):
        for method, path in PROTECTED_PATHS:
            resp = getattr(api_key_client, method.lower())(path)
            assert resp.status_code == 403, (method, path, resp.status_code)
            assert "API Key" in resp.json()["detail"]

    def test_wrong_key_is_rejected(self, api_key_client):
        resp = api_key_client.get(
            "/documents/status_counts?kb_id=default", headers={"X-API-Key": "nope"}
        )
        assert resp.status_code == 403
        assert resp.json()["detail"] == "Invalid API Key"

    def test_correct_key_is_accepted(self, api_key_client):
        resp = api_key_client.get(
            "/documents/status_counts?kb_id=default",
            headers={"X-API-Key": "unit-test-key-42"},
        )
        assert resp.status_code == 200
        assert resp.json()["status_counts"]["processed"] == 2

    def test_health_stays_whitelisted(self, api_key_client):
        # /health is in the default whitelist; no key required
        resp = api_key_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_kb_management_has_no_auth_dependency(self, api_key_client):
        # The inline KB endpoints are deliberately unguarded (matches server code)
        resp = api_key_client.get("/knowledge_bases")
        assert resp.status_code == 200

    def test_query_rejects_missing_key(self, api_key_client):
        resp = api_key_client.post("/query", json={"query": "what is graph rag"})
        assert resp.status_code == 403

        resp = api_key_client.post(
            "/query",
            json={"query": "what is graph rag"},
            headers={"X-API-Key": "unit-test-key-42"},
        )
        assert resp.status_code == 200
        assert resp.json()["response"] == "ok"


@pytest.mark.offline
class TestJwtAccountMode:
    def test_auth_status_reports_configured(self, jwt_client):
        resp = jwt_client.get("/auth-status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["auth_configured"] is True
        assert body["auth_mode"] == "enabled"
        assert "access_token" not in body

    def test_login_wrong_credentials(self, jwt_client):
        resp = jwt_client.post(
            "/login", data={"username": "admin", "password": "wrong"}
        )
        assert resp.status_code == 401
        assert "Incorrect credentials" in resp.json()["detail"]

    def test_login_unknown_user(self, jwt_client):
        resp = jwt_client.post("/login", data={"username": "ghost", "password": "x"})
        assert resp.status_code == 401

    def test_login_success_and_protected_call(self, jwt_client):
        resp = jwt_client.post(
            "/login", data={"username": "admin", "password": "s3cret"}
        )
        assert resp.status_code == 200
        token = resp.json()["access_token"]
        assert resp.json()["auth_mode"] == "enabled"

        resp = jwt_client.get(
            "/documents/status_counts?kb_id=default",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_protected_route_without_token(self, jwt_client):
        resp = jwt_client.get("/documents/status_counts?kb_id=default")
        assert resp.status_code == 401
        assert "login" in resp.json()["detail"].lower()

    def test_guest_token_rejected_when_auth_configured(self, jwt_client):
        from lightrag.api.auth import auth_handler

        guest = auth_handler.create_token("guest", role="guest")
        resp = jwt_client.get(
            "/documents/status_counts?kb_id=default",
            headers={"Authorization": f"Bearer {guest}"},
        )
        assert resp.status_code == 401

    def test_malformed_token_rejected(self, jwt_client):
        resp = jwt_client.get(
            "/documents/status_counts?kb_id=default",
            headers={"Authorization": "Bearer not-a-jwt"},
        )
        assert resp.status_code == 401

    def test_login_disabled_mode_returns_guest(self, monkeypatch):
        client, teardown = _build_client(monkeypatch, accounts=None)
        try:
            resp = client.post("/login", data={"username": "a", "password": "b"})
            assert resp.status_code == 200
            body = resp.json()
            assert body["auth_mode"] == "disabled"
            assert body["access_token"]
        finally:
            teardown()
