"""
Rerank 诊断进度事件测试(模块 A 验证)。

验证 lightrag/utils.py 中 apply_rerank_if_enabled 在不同分支下会调用 on_progress
并携带正确的 status/reason 等 detail,用于前端直观判断"rerank 是否生效"。
"""

import pytest
from unittest.mock import AsyncMock


@pytest.mark.offline
class TestRerankProgressEvents:
    """rerank 进度事件测试。"""

    async def test_rerank_success_emits_progress(self):
        """rerank 成功时应发出 success 进度事件。"""
        from lightrag.utils import apply_rerank_if_enabled

        progress = AsyncMock()

        async def fake_rerank(query, documents, top_n):
            return [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.7},
            ]

        global_config = {"rerank_model_func": fake_rerank}
        docs = [{"content": "a"}, {"content": "b"}]

        result = await apply_rerank_if_enabled(
            query="q",
            retrieved_docs=docs,
            global_config=global_config,
            enable_rerank=True,
            top_n=2,
            on_progress=progress,
        )

        assert len(result) == 2
        assert result[0]["content"] == "b"  # index 1 first
        progress.assert_awaited_once()
        args, _ = progress.call_args
        assert args[0] == "rerank"
        assert args[1]["status"] == "success"
        assert args[1]["kept"] == 2
        assert args[1]["from"] == 2

    async def test_rerank_failure_emits_failed_progress(self):
        """rerank 函数抛异常时应发出 failed 进度事件并降级返回原始 chunks。"""
        from lightrag.utils import apply_rerank_if_enabled

        progress = AsyncMock()

        async def fake_rerank(query, documents, top_n):
            raise ConnectionError("rerank service down")

        global_config = {"rerank_model_func": fake_rerank}
        docs = [{"content": "a"}, {"content": "b"}]

        result = await apply_rerank_if_enabled(
            query="q",
            retrieved_docs=docs,
            global_config=global_config,
            enable_rerank=True,
            top_n=2,
            on_progress=progress,
        )

        assert result == docs  # 降级返回原始 docs
        progress.assert_awaited_once()
        args, _ = progress.call_args
        assert args[0] == "rerank"
        assert args[1]["status"] == "failed"
        assert "rerank service down" in args[1]["reason"]
        assert args[1]["fallback"] == "original_chunks"

    async def test_rerank_disabled_emits_skipped(self):
        """enable_rerank=False 时应发出 skipped 事件。"""
        from lightrag.utils import apply_rerank_if_enabled

        progress = AsyncMock()
        docs = [{"content": "a"}]

        result = await apply_rerank_if_enabled(
            query="q",
            retrieved_docs=docs,
            global_config={},
            enable_rerank=False,
            on_progress=progress,
        )

        assert result == docs
        progress.assert_awaited_once()
        args, _ = progress.call_args
        assert args[0] == "rerank"
        assert args[1]["status"] == "skipped"
        assert args[1]["reason"] == "disabled"

    async def test_rerank_no_model_emits_skipped(self):
        """未配置 rerank_model_func 时应发出 skipped 事件。"""
        from lightrag.utils import apply_rerank_if_enabled

        progress = AsyncMock()
        docs = [{"content": "a"}]

        result = await apply_rerank_if_enabled(
            query="q",
            retrieved_docs=docs,
            global_config={},
            enable_rerank=True,
            on_progress=progress,
        )

        assert result == docs
        progress.assert_awaited_once()
        args, _ = progress.call_args
        assert args[0] == "rerank"
        assert args[1]["status"] == "skipped"
        assert args[1]["reason"] == "model_not_configured"
