"""
查询路由测试。

覆盖 lightrag/api/routers/query_routes.py 的核心功能：
- QueryRequest 验证（query 长度、mode、LLM override、conversation_history、external_kbs）
- POST /query 端点（非流式响应）
- POST /query/stream 端点（NDJSON 流式响应）
- POST /query/data 端点（结构化数据检索）
- POST /ext/retrieval 和 /ext/rag（外部 KB 模拟端点）
"""

import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

# /ext/* 模拟端点要求 Bearer token；须与 lightrag/api/routers/query_routes.py 中
# create_query_routes 内的 EXT_KB_TEST_API_KEY 保持一致。
_EXT_KB_TEST_API_KEY = "ext-kb-test-sk-2026"


# ──────────────────────────────────────────────
# QueryRequest 验证测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestQueryRequestValidation:
    """测试 QueryRequest 数据验证。"""

    def test_query_min_length_validation(self):
        """query 长度小于 3 应返回 422。"""
        from lightrag.api.routers.query_routes import QueryRequest

        with pytest.raises(ValueError):
            QueryRequest(query="ab")

    def test_query_mode_validation(self):
        """无效的 mode 应返回 422。"""
        from lightrag.api.routers.query_routes import QueryRequest

        with pytest.raises(ValueError):
            QueryRequest(query="valid query", mode="invalid_mode")

    def test_top_k_validation(self):
        """top_k=0 应返回 422。"""
        from lightrag.api.routers.query_routes import QueryRequest

        with pytest.raises(ValueError):
            QueryRequest(query="test", top_k=0)

    def test_llm_override_requires_binding(self):
        """仅提供 llm_model 而无 llm_binding 应返回错误。"""
        from lightrag.api.routers.query_routes import QueryRequest

        with pytest.raises(ValueError, match="llm_binding is required"):
            QueryRequest(query="test", llm_model="gpt-4")

    def test_llm_override_requires_model(self):
        """仅提供 llm_binding 而无 llm_model 应返回错误。"""
        from lightrag.api.routers.query_routes import QueryRequest

        with pytest.raises(ValueError, match="llm_model is required"):
            QueryRequest(query="test", llm_binding="openai")

    def test_llm_override_requires_host(self):
        """提供 llm_binding 和 llm_model 但无 llm_binding_host 应返回错误。"""
        from lightrag.api.routers.query_routes import QueryRequest

        with pytest.raises(ValueError, match="llm_binding_host is required"):
            QueryRequest(query="test", llm_binding="openai", llm_model="gpt-4")

    def test_invalid_llm_binding_raises(self):
        """无效的 llm_binding 应返回错误。"""
        from lightrag.api.routers.query_routes import QueryRequest

        with pytest.raises(ValueError, match="llm_binding must be one of"):
            QueryRequest(
                query="test",
                llm_binding="invalid",
                llm_model="gpt-4",
                llm_binding_host="http://api",
            )

    def test_conversation_history_role_validation(self):
        """消息缺少 role 键应返回错误。"""
        from lightrag.api.routers.query_routes import QueryRequest

        with pytest.raises(ValueError, match="must have a 'role' key"):
            QueryRequest(query="test", conversation_history=[{"content": "no role"}])

    def test_conversation_history_empty_role(self):
        """消息的 role 为空字符串应返回错误。"""
        from lightrag.api.routers.query_routes import QueryRequest

        with pytest.raises(ValueError, match="non-empty string"):
            QueryRequest(
                query="test", conversation_history=[{"role": "", "content": "test"}]
            )

    def test_valid_query_request(self):
        """有效的 QueryRequest 应成功创建。"""
        from lightrag.api.routers.query_routes import QueryRequest

        req = QueryRequest(
            query="What is AI?",
            mode="mix",
            top_k=10,
            kb_ids=["kb1"],
        )
        assert req.query == "What is AI?"
        assert req.mode == "mix"

    def test_valid_query_request_with_llm_default_headers(self):
        """LLM override 可携带自定义认证头。"""
        from lightrag.api.routers.query_routes import QueryRequest

        req = QueryRequest(
            query="What is AI?",
            llm_binding="openai",
            llm_model="qwen3_32b",
            llm_binding_host="https://example.com/v1",
            llm_default_headers={"X-Apig-AppCode": "app-code-123"},
        )
        assert req.llm_default_headers == {"X-Apig-AppCode": "app-code-123"}

    def test_external_kb_config_validation(self):
        """ExternalKBConfig 的 top_k 边界值验证。"""
        from lightrag.api.routers.query_routes import ExternalKBConfig

        # ge=1, le=50
        config = ExternalKBConfig(url="http://api", type="retrieval", top_k=5)
        assert config.top_k == 5

        with pytest.raises(ValueError):
            ExternalKBConfig(url="http://api", type="retrieval", top_k=0)

        with pytest.raises(ValueError):
            ExternalKBConfig(url="http://api", type="retrieval", top_k=100)


# ──────────────────────────────────────────────
# 辅助函数：创建挂载了 query routes 的测试客户端
# ──────────────────────────────────────────────


def _make_client(mock_mgr):
    """创建独立 FastAPI app，挂载 query routes 并返回 TestClient。

    每次调用都会替换模块级 router，避免多个测试共享同一个 APIRouter
    导致路由冲突（第一个注册的 handler 始终优先匹配）。
    """
    from unittest.mock import patch
    from fastapi import FastAPI, APIRouter
    from lightrag.api.routers import query_routes
    from lightrag.api.routers.query_routes import create_query_routes

    # 用全新的 APIRouter 替换模块级 router，确保路由隔离
    fresh_router = APIRouter(tags=["query"])
    with patch.object(query_routes, "router", fresh_router):
        app = FastAPI()
        app.include_router(create_query_routes(mock_mgr))
    return TestClient(app)


# ──────────────────────────────────────────────
# POST /query 端点测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestQueryEndpoint:
    """测试 /query 端点。"""

    @pytest.mark.asyncio
    async def test_query_endpoint_success(self, test_app):
        """成功查询应返回 200 和响应内容。"""

        async def mock_multi_kb_query(*args, **kwargs):
            return {
                "status": "success",
                "message": "Query executed successfully",
                "data": {
                    "references": [],
                    "entities": [],
                    "relationships": [],
                    "chunks": [],
                },
                "metadata": {},
                "llm_response": {"content": "AI response text", "is_streaming": False},
            }

        mock_mgr = MagicMock()
        mock_mgr.multi_kb_query = mock_multi_kb_query
        mock_mgr.default_kb = "default"

        client = _make_client(mock_mgr)
        response = client.post("/query", json={"query": "What is AI?"})

        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert data["response"] == "AI response text"

    @pytest.mark.asyncio
    async def test_query_bypass_no_references(self, test_app):
        """bypass 模式不应返回引用。"""

        async def mock_multi_kb_query(*args, **kwargs):
            return {
                "status": "success",
                "message": "Bypass mode response",
                "data": {},
                "metadata": {"query_mode": "bypass"},
                "llm_response": {"content": "bypass answer", "is_streaming": False},
            }

        mock_mgr = MagicMock()
        mock_mgr.multi_kb_query = mock_multi_kb_query
        mock_mgr.default_kb = "default"

        client = _make_client(mock_mgr)
        response = client.post("/query", json={"query": "test", "mode": "bypass"})

        assert response.status_code == 200
        data = response.json()
        assert data["references"] is None

    @pytest.mark.asyncio
    async def test_query_without_references(self, test_app):
        """include_references=False 应不返回引用。"""

        async def mock_multi_kb_query(*args, **kwargs):
            return {
                "status": "success",
                "message": "Query executed successfully",
                "data": {"references": [{"reference_id": "1", "file_path": "doc.txt"}]},
                "metadata": {},
                "llm_response": {"content": "answer", "is_streaming": False},
            }

        mock_mgr = MagicMock()
        mock_mgr.multi_kb_query = mock_multi_kb_query
        mock_mgr.default_kb = "default"

        client = _make_client(mock_mgr)
        response = client.post(
            "/query", json={"query": "test", "include_references": False}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["references"] is None

    @pytest.mark.asyncio
    async def test_query_with_chunk_content(self, test_app):
        """include_chunk_content=True 应在引用中添加内容。"""

        async def mock_multi_kb_query(*args, **kwargs):
            return {
                "status": "success",
                "message": "Query executed successfully",
                "data": {
                    "references": [{"reference_id": "1", "file_path": "doc.txt"}],
                    "chunks": [
                        {
                            "chunk_id": "c1",
                            "content": "chunk content 1",
                            "reference_id": "1",
                        },
                        {
                            "chunk_id": "c2",
                            "content": "chunk content 2",
                            "reference_id": "1",
                        },
                    ],
                },
                "metadata": {},
                "llm_response": {"content": "answer", "is_streaming": False},
            }

        mock_mgr = MagicMock()
        mock_mgr.multi_kb_query = mock_multi_kb_query
        mock_mgr.default_kb = "default"

        client = _make_client(mock_mgr)
        response = client.post(
            "/query", json={"query": "test", "include_chunk_content": True}
        )

        assert response.status_code == 200
        data = response.json()
        assert "references" in data
        assert len(data["references"]) == 1
        assert "content" in data["references"][0]
        assert len(data["references"][0]["content"]) == 2

    @pytest.mark.asyncio
    async def test_query_with_llm_override(self, test_app):
        """LLM override 参数应正确注入。"""
        captured = {"model_func": None, "model_func_id": None}

        async def mock_multi_kb_query(query, kb_ids, param, **kwargs):
            captured["model_func"] = param.model_func
            captured["model_func_id"] = getattr(param, "model_func_id", None)
            return {
                "status": "success",
                "message": "",
                "data": {},
                "metadata": {},
                "llm_response": {"content": "answer", "is_streaming": False},
            }

        mock_mgr = MagicMock()
        mock_mgr.multi_kb_query = mock_multi_kb_query
        mock_mgr.default_kb = "default"

        client = _make_client(mock_mgr)
        response = client.post(
            "/query",
            json={
                "query": "test",
                "llm_binding": "openai",
                "llm_model": "gpt-4o",
                "llm_binding_host": "http://api.openai.com",
            },
        )

        assert response.status_code == 200
        assert captured["model_func"] is not None
        assert "openai:http://api.openai.com:gpt-4o" in captured["model_func_id"]

    @pytest.mark.asyncio
    @patch("lightrag.llm.openai.openai_complete_if_cache", new_callable=AsyncMock)
    async def test_query_with_llm_override_and_default_headers(
        self, mock_complete, test_app
    ):
        """LLM override 的自定义头应传入动态 LLM 函数。"""
        mock_complete.return_value = "answer"
        captured_headers = []

        async def mock_multi_kb_query(query, kb_ids, param, **kwargs):
            await param.model_func("prompt")
            captured_headers.append(
                mock_complete.call_args.kwargs.get("default_headers")
            )
            return {
                "status": "success",
                "message": "",
                "data": {},
                "metadata": {},
                "llm_response": {"content": "answer", "is_streaming": False},
            }

        mock_mgr = MagicMock()
        mock_mgr.multi_kb_query = mock_multi_kb_query
        mock_mgr.default_kb = "default"

        client = _make_client(mock_mgr)
        response = client.post(
            "/query",
            json={
                "query": "test",
                "llm_binding": "openai",
                "llm_model": "qwen3_32b",
                "llm_binding_host": "https://example.com/v1",
                "llm_default_headers": {"X-Apig-AppCode": "app-code-123"},
            },
        )

        assert response.status_code == 200
        assert captured_headers == [{"X-Apig-AppCode": "app-code-123"}]

    @pytest.mark.asyncio
    async def test_query_error_500(self, test_app):
        """内部错误应返回 500。"""

        async def mock_multi_kb_query(*args, **kwargs):
            raise Exception("Internal error")

        mock_mgr = MagicMock()
        mock_mgr.multi_kb_query = mock_multi_kb_query
        mock_mgr.default_kb = "default"

        client = _make_client(mock_mgr)
        response = client.post("/query", json={"query": "test"})

        assert response.status_code == 500
        assert "Internal error" in response.json()["detail"]


# ──────────────────────────────────────────────
# POST /query/stream 端点测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestQueryStreamEndpoint:
    """测试 /query/stream 端点。"""

    @pytest.mark.asyncio
    async def test_stream_ndjson_format(self, test_app):
        """流式响应应返回 NDJSON 格式。"""

        async def stream_generator():
            yield "chunk1"
            yield "chunk2"

        async def mock_multi_kb_query(*args, **kwargs):
            return {
                "status": "success",
                "message": "",
                "data": {"references": []},
                "metadata": {},
                "llm_response": {
                    "content": "",
                    "is_streaming": True,
                    "response_iterator": stream_generator(),
                },
            }

        mock_mgr = MagicMock()
        mock_mgr.multi_kb_query = mock_multi_kb_query
        mock_mgr.default_kb = "default"

        client = _make_client(mock_mgr)
        response = client.post("/query/stream", json={"query": "test", "stream": True})

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/x-ndjson"

        # 验证 NDJSON 行
        lines = response.text.strip().split("\n")
        for line in lines:
            data = json.loads(line)
            assert "response" in data or "references" in data

    @pytest.mark.asyncio
    async def test_stream_with_references_first_line(self, test_app):
        """include_references=True 时首行应包含引用。"""

        async def stream_generator():
            yield "response text"

        async def mock_multi_kb_query(*args, **kwargs):
            return {
                "status": "success",
                "message": "",
                "data": {"references": [{"reference_id": "1", "file_path": "doc.txt"}]},
                "metadata": {},
                "llm_response": {
                    "content": "",
                    "is_streaming": True,
                    "response_iterator": stream_generator(),
                },
            }

        mock_mgr = MagicMock()
        mock_mgr.multi_kb_query = mock_multi_kb_query
        mock_mgr.default_kb = "default"

        client = _make_client(mock_mgr)
        response = client.post(
            "/query/stream", json={"query": "test", "include_references": True}
        )

        lines = response.text.strip().split("\n")
        first_line = json.loads(lines[0])
        assert "references" in first_line

    @pytest.mark.asyncio
    async def test_stream_error_in_ndjson(self, test_app):
        """流式过程中的错误应作为 NDJSON 行返回。"""

        async def error_stream_generator():
            yield "start"
            raise Exception("Stream error")

        async def mock_multi_kb_query(*args, **kwargs):
            return {
                "status": "success",
                "message": "",
                "data": {"references": []},
                "metadata": {},
                "llm_response": {
                    "content": "",
                    "is_streaming": True,
                    "response_iterator": error_stream_generator(),
                },
            }

        mock_mgr = MagicMock()
        mock_mgr.multi_kb_query = mock_multi_kb_query
        mock_mgr.default_kb = "default"

        client = _make_client(mock_mgr)
        response = client.post("/query/stream", json={"query": "test", "stream": True})

        lines = response.text.strip().split("\n")
        # 某一行应包含错误
        has_error = any("error" in json.loads(line) for line in lines)
        assert has_error

    @pytest.mark.asyncio
    async def test_stream_non_streaming_mode(self, test_app):
        """stream=False 应返回单行完整响应。"""

        async def mock_multi_kb_query(*args, **kwargs):
            return {
                "status": "success",
                "message": "",
                "data": {"references": []},
                "metadata": {},
                "llm_response": {"content": "complete answer", "is_streaming": False},
            }

        mock_mgr = MagicMock()
        mock_mgr.multi_kb_query = mock_multi_kb_query
        mock_mgr.default_kb = "default"

        client = _make_client(mock_mgr)
        response = client.post("/query/stream", json={"query": "test", "stream": False})

        lines = response.text.strip().split("\n")
        # 应只有一行
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["response"] == "complete answer"

    @pytest.mark.asyncio
    async def test_stream_progress_events(self, test_app):
        """流式响应应在检索阶段输出 progress 事件帧(模块 A)。"""

        async def stream_generator():
            yield "response text"

        async def mock_multi_kb_query(*args, **kwargs):
            on_progress = kwargs.get("on_progress")
            if on_progress:
                await on_progress(
                    "keyword_extraction", {"high_level": [], "low_level": ["ai"]}
                )
                await on_progress(
                    "vector_search", {"entities": 2, "relations": 1, "chunks": 3}
                )
                await on_progress(
                    "rerank",
                    {
                        "status": "failed",
                        "reason": "ConnectionTimeoutError",
                        "fallback": "original_chunks",
                    },
                )
                await on_progress("retrieval_done", {"status": "done"})

            return {
                "status": "success",
                "message": "",
                "data": {"references": []},
                "metadata": {},
                "llm_response": {
                    "content": "",
                    "is_streaming": True,
                    "response_iterator": stream_generator(),
                },
            }

        mock_mgr = MagicMock()
        mock_mgr.multi_kb_query = mock_multi_kb_query
        mock_mgr.default_kb = "default"

        client = _make_client(mock_mgr)
        response = client.post("/query/stream", json={"query": "test", "stream": True})

        assert response.status_code == 200
        lines = response.text.strip().split("\n")
        events = [json.loads(line) for line in lines]

        progress_frames = [e for e in events if e.get("type") == "progress"]
        assert len(progress_frames) == 4
        assert progress_frames[0]["stage"] == "keyword_extraction"
        assert progress_frames[1]["stage"] == "vector_search"
        assert progress_frames[2]["stage"] == "rerank"
        assert progress_frames[2]["detail"]["status"] == "failed"
        assert progress_frames[2]["status"] == "failed"  # 顶层 status 便于前端读取
        assert progress_frames[3]["stage"] == "retrieval_done"

        response_frames = [e for e in events if "response" in e]
        assert len(response_frames) == 1
        assert response_frames[0]["response"] == "response text"


# ──────────────────────────────────────────────
# POST /query/data 端点测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestQueryDataEndpoint:
    """测试 /query/data 端点。"""

    @pytest.mark.asyncio
    async def test_query_data_success(self, test_app):
        """成功请求应返回结构化数据。"""

        async def mock_multi_kb_get_data(*args, **kwargs):
            return {
                "status": "success",
                "message": "Data retrieved successfully",
                "data": {
                    "entities": [{"entity_name": "AI", "entity_type": "CONCEPT"}],
                    "relationships": [],
                    "chunks": [],
                    "references": [],
                },
                "metadata": {"query_mode": "local"},
            }

        mock_mgr = MagicMock()
        mock_mgr.multi_kb_get_data = mock_multi_kb_get_data
        mock_mgr.default_kb = "default"

        client = _make_client(mock_mgr)
        response = client.post("/query/data", json={"query": "test data query"})

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
        assert "metadata" in data
        assert data["data"]["entities"][0]["entity_name"] == "AI"

    @pytest.mark.asyncio
    async def test_query_data_structure(self, test_app):
        """返回数据应包含所有必需字段。"""

        async def mock_multi_kb_get_data(*args, **kwargs):
            return {
                "status": "success",
                "message": "",
                "data": {
                    "entities": [],
                    "relationships": [],
                    "chunks": [],
                    "references": [],
                },
                "metadata": {"query_mode": "mix"},
            }

        mock_mgr = MagicMock()
        mock_mgr.multi_kb_get_data = mock_multi_kb_get_data
        mock_mgr.default_kb = "default"

        client = _make_client(mock_mgr)
        response = client.post("/query/data", json={"query": "test"})

        assert response.status_code == 200
        data = response.json()
        # 验证所有必需字段存在
        assert "status" in data
        assert "message" in data
        assert "data" in data
        assert "metadata" in data

    @pytest.mark.asyncio
    async def test_query_data_with_external_kbs(self, test_app):
        """外部 KB 应传递给 rag_manager。"""
        captured_external_kbs = []

        async def mock_multi_kb_get_data(query, kb_ids, param, external_kbs=None):
            captured_external_kbs.append(external_kbs)
            return {
                "status": "success",
                "message": "",
                "data": {
                    "entities": [],
                    "relationships": [],
                    "chunks": [],
                    "references": [],
                },
                "metadata": {"query_mode": "mix"},
            }

        mock_mgr = MagicMock()
        mock_mgr.multi_kb_get_data = mock_multi_kb_get_data
        mock_mgr.default_kb = "default"

        client = _make_client(mock_mgr)
        response = client.post(
            "/query/data",
            json={
                "query": "test",
                "external_kbs": [
                    {"type": "retrieval", "url": "http://ext.kb", "top_k": 5}
                ],
            },
        )

        assert response.status_code == 200
        assert len(captured_external_kbs) == 1
        assert captured_external_kbs[0][0]["type"] == "retrieval"

    @pytest.mark.asyncio
    async def test_query_data_invalid_response(self, test_app):
        """非 dict 响应应触发异常（路由 handler 的 fallback 缺少 metadata 字段，
        Pydantic 校验失败后被 except 捕获并返回 500）。"""

        async def mock_multi_kb_get_data(*args, **kwargs):
            return "invalid response"

        mock_mgr = MagicMock()
        mock_mgr.multi_kb_get_data = mock_multi_kb_get_data
        mock_mgr.default_kb = "default"

        client = _make_client(mock_mgr)
        response = client.post("/query/data", json={"query": "test"})

        # 路由 handler 对非 dict 响应尝试构造 QueryDataResponse(status="failure", ...)，
        # 但缺少 metadata 必填字段，Pydantic 校验失败，被 except 捕获后返回 500
        assert response.status_code == 500


# ──────────────────────────────────────────────
# 外部 KB 模拟端点测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestExternalKBMockEndpoints:
    """测试外部 KB 模拟端点。"""

    @pytest.mark.asyncio
    async def test_ext_retrieval_endpoint(self, test_app):
        """/ext/retrieval 应返回标准格式。"""
        mock_rag = MagicMock()
        mock_rag.aquery_data = AsyncMock(
            return_value={
                "status": "success",
                "data": {
                    "chunks": [
                        {
                            "content": "test chunk",
                            "file_path": "doc.txt",
                            "chunk_id": "c1",
                        },
                    ],
                    "references": [],
                    "entities": [],
                    "relationships": [],
                },
            }
        )

        mock_mgr = MagicMock()
        mock_mgr.get_rag = AsyncMock(return_value=mock_rag)
        mock_mgr.default_kb = "default"

        client = _make_client(mock_mgr)
        response = client.post(
            "/ext/retrieval",
            json={"query": "test query", "top_k": 5},
            headers={"Authorization": f"Bearer {_EXT_KB_TEST_API_KEY}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "results" in data
        assert len(data["results"]) >= 0

    @pytest.mark.asyncio
    async def test_ext_rag_endpoint(self, test_app):
        """/ext/rag 应返回 RAG 格式。"""
        mock_rag = MagicMock()
        mock_rag.aquery_llm = AsyncMock(
            return_value={
                "status": "success",
                "data": {"references": []},
                "llm_response": {"content": "RAG answer", "is_streaming": False},
            }
        )

        mock_mgr = MagicMock()
        mock_mgr.get_rag = AsyncMock(return_value=mock_rag)
        mock_mgr.default_kb = "default"

        client = _make_client(mock_mgr)
        response = client.post(
            "/ext/rag",
            json={"query": "test query"},
            headers={"Authorization": f"Bearer {_EXT_KB_TEST_API_KEY}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "answer" in data
