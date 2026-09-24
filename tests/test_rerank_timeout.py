"""
Rerank 超时与重试行为测试(模块 B 验证)。

覆盖 lightrag/rerank.py 中 generic_rerank_api 的:
- 显式 10s 总超时被应用到 aiohttp ClientSession(避免默认 300s 长卡顿)
- 端点持续失败时最多尝试 2 次后抛 RetryError,缩短退避时间
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
from tenacity import RetryError


class _MockResponseCtx:
    """模拟 session.post() 返回的 async context manager。

    raise_on_enter 不为 None 时,__aenter__ 抛出该异常(用于模拟连接失败)。
    """

    def __init__(self, response=None, raise_on_enter=None):
        self._response = response
        self._raise_on_enter = raise_on_enter

    async def __aenter__(self):
        if self._raise_on_enter is not None:
            raise self._raise_on_enter
        return self._response

    async def __aexit__(self, *args):
        return False


class _MockSessionCtx:
    """模拟 aiohttp.ClientSession() 返回的 async context manager。"""

    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *args):
        return False


def _make_response(status=200, json_data=None):
    """构造一个符合 generic_rerank_api 读取方式的 mock 响应。"""
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data or {})
    resp.text = AsyncMock(return_value="")
    resp.headers = {}
    return resp


@pytest.mark.offline
class TestRerankTimeoutAndRetry:
    """rerank 超时与重试行为测试。"""

    async def test_default_timeout_applied(self):
        """aiohttp ClientSession 应被显式传入 10s 总超时。"""
        from lightrag.rerank import generic_rerank_api

        captured = {}

        response = _make_response(status=200, json_data={"results": []})
        session = MagicMock()
        session.post = MagicMock(return_value=_MockResponseCtx(response))

        def fake_client_session(**kwargs):
            captured["kwargs"] = kwargs
            return _MockSessionCtx(session)

        with patch(
            "lightrag.rerank.aiohttp.ClientSession", side_effect=fake_client_session
        ):
            result = await generic_rerank_api(
                query="q",
                documents=["d"],
                model="m",
                base_url="http://x",
                api_key="k",
            )

        # 超时应被显式传入且固定为 10s
        assert "timeout" in captured["kwargs"]
        assert isinstance(captured["kwargs"]["timeout"], aiohttp.ClientTimeout)
        assert captured["kwargs"]["timeout"].total == 10
        # 空 results 应返回空列表
        assert result == []

    async def test_persistent_failure_raises_after_two_attempts(self):
        """端点持续失败时应最多尝试 2 次(含首次)后抛 RetryError。"""
        from lightrag.rerank import generic_rerank_api

        post_count = 0

        def _post(*args, **kwargs):
            nonlocal post_count
            post_count += 1
            return _MockResponseCtx(
                raise_on_enter=aiohttp.ClientError("connection refused")
            )

        session = MagicMock()
        session.post = _post

        with patch(
            "lightrag.rerank.aiohttp.ClientSession",
            return_value=_MockSessionCtx(session),
        ):
            with pytest.raises(RetryError):
                await generic_rerank_api(
                    query="q",
                    documents=["d"],
                    model="m",
                    base_url="http://x",
                    api_key="k",
                )

        # 固定最多尝试 2 次(含首次)
        assert post_count == 2
