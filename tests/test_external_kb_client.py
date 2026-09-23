"""
外部知识库 API 客户端测试。

覆盖 lightrag/api/external_kb_client.py 的核心功能：
- 共享 ClientSession 生命周期管理
- 检索型外部知识库 (fetch_external_kb)
- 完整 RAG 服务型外部知识库 (fetch_rag_kb)
- 重试逻辑、错误处理、数据标准化
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp


# ──────────────────────────────────────────────
# 辅助工具
# ──────────────────────────────────────────────


def _make_mock_response(status=200, json_data=None, text_data=""):
    """创建 mock aiohttp 响应对象。"""
    mock_resp = AsyncMock()
    mock_resp.status = status
    mock_resp.text = AsyncMock(return_value=text_data)
    mock_resp.json = AsyncMock(return_value=json_data or {})
    return mock_resp


def _make_mock_post(mock_resp):
    """创建 mock session.post 上下文管理器。

    返回一个模拟 aiohttp.ClientSession.post() 的 async context manager：
    - __aenter__ 返回 mock_resp
    - __aexit__ 返回 False
    """

    class MockContextManager:
        """模拟异步上下文管理器。"""
        def __init__(self, response):
            self._response = response

        async def __aenter__(self):
            return self._response

        async def __aexit__(self, *args):
            return False

    return MockContextManager(mock_resp)


# ──────────────────────────────────────────────
# Session 管理测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestSessionManagement:
    """测试共享 ClientSession 的创建、复用和关闭。"""

    @pytest.fixture(autouse=True)
    def _reset_session(self):
        """每个测试前重置模块级 session。"""
        import lightrag.api.external_kb_client as mod

        self._mod = mod
        # 关闭可能存在的旧 session
        if mod._session is not None:
            mod._session = None
            # 不需要 await close，直接设为 None
        yield
        # 测试后清理
        self._mod._session = None

    async def test_get_session_creates_new_session(self):
        """首次调用 get_session() 应创建新的 ClientSession。"""
        with patch("aiohttp.ClientSession") as MockSession:
            mock_instance = MagicMock()
            mock_instance.closed = False
            MockSession.return_value = mock_instance

            session = await self._mod.get_session()

            MockSession.assert_called_once()
            assert session is mock_instance
            assert self._mod._session is mock_instance

    async def test_get_session_returns_same_instance(self):
        """后续调用应返回同一实例（单例模式）。"""
        mock_instance = MagicMock()
        mock_instance.closed = False
        self._mod._session = mock_instance

        session = await self._mod.get_session()

        assert session is mock_instance

    async def test_get_session_recreates_after_close(self):
        """session 被关闭后，get_session() 应创建新实例。"""
        closed_session = MagicMock()
        closed_session.closed = True
        self._mod._session = closed_session

        with patch("aiohttp.ClientSession") as MockSession:
            new_instance = MagicMock()
            new_instance.closed = False
            MockSession.return_value = new_instance

            session = await self._mod.get_session()

            MockSession.assert_called_once()
            assert session is new_instance

    async def test_close_session_idempotent(self):
        """重复调用 close_session() 不应报错。"""
        mock_session = AsyncMock()
        mock_session.closed = False
        self._mod._session = mock_session

        await self._mod.close_session()
        assert self._mod._session is None

        # 再次调用不应报错
        await self._mod.close_session()
        assert self._mod._session is None

    async def test_close_session_skips_when_none(self):
        """当 session 为 None 时，close_session() 不应报错。"""
        self._mod._session = None
        await self._mod.close_session()
        assert self._mod._session is None


# ──────────────────────────────────────────────
# 检索型外部知识库测试 (fetch_external_kb)
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestFetchExternalKB:
    """测试检索型外部知识库的请求和数据标准化。"""

    @pytest.fixture(autouse=True)
    def _reset_session(self):
        import lightrag.api.external_kb_client as mod

        self._mod = mod
        mod._session = None
        yield
        mod._session = None

    @patch("lightrag.api.external_kb_client.get_session")
    async def test_fetch_success_with_chunks(self, mock_get_session):
        """成功请求应返回标准化的 chunk 列表。"""
        mock_resp = _make_mock_response(
            status=200,
            json_data={
                "status": "success",
                "results": [
                    {
                        "content": "LightRAG is a RAG framework",
                        "file_path": "doc1.txt",
                        "score": 0.95,
                    },
                    {
                        "content": "It uses knowledge graphs",
                        "file_path": "doc2.txt",
                    },
                ],
            },
        )
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_make_mock_post(mock_resp))
        mock_get_session.return_value = mock_session

        result = await self._mod.fetch_external_kb(
            query="What is LightRAG?", url="http://example.com/api"
        )

        chunks, error = result
        assert error is None
        assert len(chunks) == 2
        assert chunks[0]["content"] == "LightRAG is a RAG framework"
        assert chunks[0]["file_path"] == "doc1.txt"
        assert chunks[0]["chunk_id"]  # 自动生成
        assert chunks[1]["content"] == "It uses knowledge graphs"

    @patch("lightrag.api.external_kb_client.get_session")
    async def test_fetch_with_api_key(self, mock_get_session):
        """应正确发送 Authorization header。"""
        mock_resp = _make_mock_response(
            status=200,
            json_data={"status": "success", "results": []},
        )
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_make_mock_post(mock_resp))
        mock_get_session.return_value = mock_session

        await self._mod.fetch_external_kb(
            query="test",
            url="http://example.com/api",
            api_key="sk-test-123",
        )

        # 验证请求参数中包含 Authorization header
        call_kwargs = mock_session.post.call_args
        headers = call_kwargs.kwargs.get("headers", call_kwargs[1].get("headers", {}))
        assert headers["Authorization"] == "Bearer sk-test-123"

    @patch("lightrag.api.external_kb_client.get_session")
    async def test_fetch_empty_results(self, mock_get_session):
        """API 返回空 results 应返回空列表。"""
        mock_resp = _make_mock_response(
            status=200,
            json_data={"status": "success", "results": []},
        )
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_make_mock_post(mock_resp))
        mock_get_session.return_value = mock_session

        result = await self._mod.fetch_external_kb(
            query="test", url="http://example.com/api"
        )

        chunks, error = result
        assert chunks == []
        assert error is None

    @patch("lightrag.api.external_kb_client.get_session")
    async def test_fetch_skips_non_dict_results(self, mock_get_session):
        """results 中的非 dict 条目应被跳过。"""
        mock_resp = _make_mock_response(
            status=200,
            json_data={
                "status": "success",
                "results": [
                    "not a dict",
                    {"content": "valid chunk", "file_path": "doc.txt"},
                    42,
                ],
            },
        )
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_make_mock_post(mock_resp))
        mock_get_session.return_value = mock_session

        chunks, error = await self._mod.fetch_external_kb(
            query="test", url="http://example.com/api"
        )

        assert len(chunks) == 1
        assert chunks[0]["content"] == "valid chunk"
        assert error is None

    @patch("lightrag.api.external_kb_client.get_session")
    async def test_fetch_skips_empty_content(self, mock_get_session):
        """空 content 的条目应被跳过。"""
        mock_resp = _make_mock_response(
            status=200,
            json_data={
                "status": "success",
                "results": [
                    {"content": "", "file_path": "doc1.txt"},
                    {"content": "valid", "file_path": "doc2.txt"},
                    {"file_path": "doc3.txt"},
                ],
            },
        )
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_make_mock_post(mock_resp))
        mock_get_session.return_value = mock_session

        chunks, error = await self._mod.fetch_external_kb(
            query="test", url="http://example.com/api"
        )

        assert len(chunks) == 1
        assert chunks[0]["content"] == "valid"
        assert error is None

    @patch("lightrag.api.external_kb_client.get_session")
    async def test_fetch_http_error_returns_empty(self, mock_get_session):
        """HTTP 非 200 状态码应返回空列表和错误信息。"""
        mock_resp = _make_mock_response(
            status=500, text_data="Internal Server Error"
        )
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_make_mock_post(mock_resp))
        mock_get_session.return_value = mock_session

        chunks, error = await self._mod.fetch_external_kb(
            query="test", url="http://example.com/api"
        )

        assert chunks == []
        assert error is not None
        assert "请求失败" in error

    @patch("lightrag.api.external_kb_client.get_session")
    async def test_fetch_invalid_status_returns_empty(self, mock_get_session):
        """API 返回 status=error 应返回空列表和错误信息。"""
        mock_resp = _make_mock_response(
            status=200,
            json_data={"status": "error", "message": "Invalid query"},
        )
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_make_mock_post(mock_resp))
        mock_get_session.return_value = mock_session

        chunks, error = await self._mod.fetch_external_kb(
            query="test", url="http://example.com/api"
        )

        assert chunks == []
        assert error is not None
        assert "请求失败" in error

    @patch("lightrag.api.external_kb_client.get_session")
    async def test_fetch_retry_on_client_error(self, mock_get_session):
        """aiohttp.ClientError 应触发重试。"""
        call_count = [0]  # 使用列表以便在闭包中修改

        def _side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise aiohttp.ClientError("Connection refused")
            mock_resp = _make_mock_response(
                status=200,
                json_data={"status": "success", "results": [{"content": "retry ok"}]},
            )
            return _make_mock_post(mock_resp)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(side_effect=_side_effect)
        mock_get_session.return_value = mock_session

        chunks, error = await self._mod.fetch_external_kb(
            query="test", url="http://example.com/api"
        )

        assert len(chunks) == 1
        assert chunks[0]["content"] == "retry ok"
        assert call_count[0] == 2  # 第一次失败，第二次成功
        assert error is None

    @patch("lightrag.api.external_kb_client.get_session")
    async def test_fetch_uses_custom_chunk_id(self, mock_get_session):
        """如果结果中提供了 chunk_id，应使用该 ID 而非自动生成。"""
        mock_resp = _make_mock_response(
            status=200,
            json_data={
                "status": "success",
                "results": [
                    {"content": "test", "chunk_id": "my_custom_id_123"},
                ],
            },
        )
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_make_mock_post(mock_resp))
        mock_get_session.return_value = mock_session

        chunks, error = await self._mod.fetch_external_kb(
            query="test", url="http://example.com/api"
        )

        assert chunks[0]["chunk_id"] == "my_custom_id_123"
        assert error is None


# ──────────────────────────────────────────────
# RAG 服务型外部知识库测试 (fetch_rag_kb)
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestFetchRagKB:
    """测试完整 RAG 服务型外部知识库的请求和响应处理。"""

    @pytest.fixture(autouse=True)
    def _reset_session(self):
        import lightrag.api.external_kb_client as mod

        self._mod = mod
        mod._session = None
        yield
        mod._session = None

    @patch("lightrag.api.external_kb_client.get_session")
    async def test_fetch_rag_success(self, mock_get_session):
        """成功请求应返回正确的 answer/references/source 格式。"""
        mock_resp = _make_mock_response(
            status=200,
            json_data={
                "status": "success",
                "answer": "LightRAG 是一个 RAG 框架",
                "references": [
                    {"file_path": "doc1.txt", "content": "LightRAG description"},
                    {"file_path": "doc2.txt", "content": "KG details"},
                ],
            },
        )
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_make_mock_post(mock_resp))
        mock_get_session.return_value = mock_session

        result, error = await self._mod.fetch_rag_kb(
            query="What is LightRAG?", url="http://example.com/rag"
        )

        assert error is None
        assert result["answer"] == "LightRAG 是一个 RAG 框架"
        assert len(result["references"]) == 2
        assert result["source"] == "http://example.com/rag"

    @patch("lightrag.api.external_kb_client.get_session")
    async def test_fetch_rag_filters_refs_without_file_path(self, mock_get_session):
        """无 file_path 的引用应被过滤。"""
        mock_resp = _make_mock_response(
            status=200,
            json_data={
                "status": "success",
                "answer": "answer text",
                "references": [
                    {"file_path": "doc1.txt"},
                    {"content": "no file path"},
                    {"file_path": "", "content": "empty path"},
                    {"file_path": "doc2.txt", "content": "valid"},
                ],
            },
        )
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_make_mock_post(mock_resp))
        mock_get_session.return_value = mock_session

        result, error = await self._mod.fetch_rag_kb(
            query="test", url="http://example.com/rag"
        )

        assert error is None
        # 只有有 file_path 的引用保留
        assert len(result["references"]) == 2
        assert result["references"][0]["file_path"] == "doc1.txt"
        assert result["references"][1]["file_path"] == "doc2.txt"

    @patch("lightrag.api.external_kb_client.get_session")
    async def test_fetch_rag_failure_returns_empty_answer(self, mock_get_session):
        """请求失败时应返回空答案和错误信息（不抛异常）。"""
        mock_get_session.side_effect = Exception("Connection refused")

        result, error = await self._mod.fetch_rag_kb(
            query="test", url="http://example.com/rag"
        )

        assert result["answer"] == ""
        assert result["references"] == []
        assert result["source"] == "http://example.com/rag"
        assert error is not None
        assert "请求失败" in error

    @patch("lightrag.api.external_kb_client.get_session")
    async def test_fetch_rag_non_string_answer_returns_empty(self, mock_get_session):
        """answer 字段非字符串时应返回空答案和错误信息。"""
        mock_resp = _make_mock_response(
            status=200,
            json_data={
                "status": "success",
                "answer": 12345,  # 非 string
                "references": [],
            },
        )
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_make_mock_post(mock_resp))
        mock_get_session.return_value = mock_session

        result, error = await self._mod.fetch_rag_kb(
            query="test", url="http://example.com/rag"
        )

        assert result["answer"] == ""
        assert error is not None

    @patch("lightrag.api.external_kb_client.get_session")
    async def test_fetch_rag_empty_answer(self, mock_get_session):
        """answer 为空字符串时应正常返回。"""
        mock_resp = _make_mock_response(
            status=200,
            json_data={
                "status": "success",
                "answer": "",
                "references": [],
            },
        )
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_make_mock_post(mock_resp))
        mock_get_session.return_value = mock_session

        result, error = await self._mod.fetch_rag_kb(
            query="test", url="http://example.com/rag"
        )

        assert result["answer"] == ""
        assert result["references"] == []
        assert error is None

    @patch("lightrag.api.external_kb_client.get_session")
    async def test_fetch_rag_non_list_references_handled(self, mock_get_session):
        """references 非数组时应被替换为空列表。"""
        mock_resp = _make_mock_response(
            status=200,
            json_data={
                "status": "success",
                "answer": "answer",
                "references": "not a list",
            },
        )
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=_make_mock_post(mock_resp))
        mock_get_session.return_value = mock_session

        result, error = await self._mod.fetch_rag_kb(
            query="test", url="http://example.com/rag"
        )

        assert result["answer"] == "answer"
        assert result["references"] == []
        assert error is None
