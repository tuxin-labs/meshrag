"""
Ollama think 参数支持测试。

覆盖：
- OllamaLLMOptions.think 默认关闭（解析路径默认不思考）
- _ollama_model_if_cache 将 options 中的 think 提升为 chat 顶层参数
- 顶层 think 显式传入时优先于 options 中的值
- 不修改调用方传入的 options dict
"""

import pytest
from unittest.mock import AsyncMock, patch


# ──────────────────────────────────────────────
# OllamaLLMOptions.think 字段
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestOllamaLLMOptionsThink:
    """测试 OllamaLLMOptions 的 think 配置。"""

    def test_think_defaults_to_false(self):
        """解析路径默认关闭思考。"""
        from lightrag.llm.binding_options import OllamaLLMOptions

        assert OllamaLLMOptions().think is False

    def test_think_can_be_enabled(self):
        """think 可显式设置为 True。"""
        from lightrag.llm.binding_options import OllamaLLMOptions

        assert OllamaLLMOptions(think=True).think is True


# ──────────────────────────────────────────────
# _ollama_model_if_cache 的 think 提升逻辑
# ──────────────────────────────────────────────


def _mock_client(response: dict):
    """构造 ollama.AsyncClient 的 mock。"""
    client = AsyncMock()
    client.chat.return_value = response
    return client


@pytest.mark.offline
class TestOllamaThinkPromotion:
    """测试 options 中的 think 被提升为 chat 顶层参数。"""

    @pytest.mark.asyncio
    @patch("lightrag.llm.ollama.ollama.AsyncClient")
    async def test_options_think_promoted_to_toplevel(self, mock_client_cls):
        """options 里的 think 应移到 chat 顶层，并从 options 中移除。"""
        from lightrag.llm.ollama import _ollama_model_if_cache

        mock_client_cls.return_value = _mock_client(
            {"message": {"content": "ok"}}
        )

        await _ollama_model_if_cache(
            "qwen3.5:latest",
            "prompt",
            options={"num_ctx": 8192, "think": False},
            host="http://localhost:11434",
        )

        call_kwargs = mock_client_cls.return_value.chat.call_args.kwargs
        assert call_kwargs["think"] is False
        assert "think" not in call_kwargs["options"]
        assert call_kwargs["options"] == {"num_ctx": 8192}

    @pytest.mark.asyncio
    @patch("lightrag.llm.ollama.ollama.AsyncClient")
    async def test_toplevel_think_takes_precedence(self, mock_client_cls):
        """显式顶层 think 应优先于 options 中的值。"""
        from lightrag.llm.ollama import _ollama_model_if_cache

        mock_client_cls.return_value = _mock_client(
            {"message": {"content": "ok"}}
        )

        await _ollama_model_if_cache(
            "qwen3.5:latest",
            "prompt",
            options={"think": False},
            think=True,
            host="http://localhost:11434",
        )

        call_kwargs = mock_client_cls.return_value.chat.call_args.kwargs
        assert call_kwargs["think"] is True

    @pytest.mark.asyncio
    @patch("lightrag.llm.ollama.ollama.AsyncClient")
    async def test_missing_think_defaults_to_false(self, mock_client_cls):
        """未提供 think 时应默认注入 False（解析路径默认关闭思考）。"""
        from lightrag.llm.ollama import _ollama_model_if_cache

        mock_client_cls.return_value = _mock_client(
            {"message": {"content": "ok"}}
        )

        await _ollama_model_if_cache(
            "qwen3.5:latest",
            "prompt",
            options={"num_ctx": 8192},
            host="http://localhost:11434",
        )

        call_kwargs = mock_client_cls.return_value.chat.call_args.kwargs
        assert call_kwargs["think"] is False
        assert call_kwargs["options"] == {"num_ctx": 8192}

    @pytest.mark.asyncio
    @patch("lightrag.llm.ollama.ollama.AsyncClient")
    async def test_original_options_not_mutated(self, mock_client_cls):
        """提升 think 时不应修改调用方传入的 options dict。"""
        from lightrag.llm.ollama import _ollama_model_if_cache

        mock_client_cls.return_value = _mock_client(
            {"message": {"content": "ok"}}
        )
        original_options = {"num_ctx": 8192, "think": False}

        await _ollama_model_if_cache(
            "qwen3.5:latest",
            "prompt",
            options=original_options,
            host="http://localhost:11434",
        )

        assert original_options == {"num_ctx": 8192, "think": False}
