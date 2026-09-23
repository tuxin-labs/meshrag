"""
LLM 工厂模块测试。

覆盖 lightrag/api/llm_factory.py 的核心功能：
- create_dynamic_llm_func 各 provider 的函数创建
- 参数转发和清洗（hashing_kv、_priority 移除）
- 不支持的 binding 类型异常
- 各 provider 特定行为（Ollama server options、Azure env 回退等）
"""

import os
import pytest
from unittest.mock import AsyncMock, patch


# ──────────────────────────────────────────────
# create_dynamic_llm_func 入口测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestCreateDynamicLLMFunc:
    """测试动态 LLM 函数创建的入口逻辑。"""

    def test_unsupported_binding_raises(self):
        """不支持的 binding 类型应抛出 ValueError。"""
        from lightrag.api.llm_factory import create_dynamic_llm_func

        with pytest.raises(ValueError, match="Unsupported llm_binding"):
            create_dynamic_llm_func(
                binding="unknown_provider", model="test-model"
            )

    @patch("lightrag.api.llm_factory._create_openai_func")
    def test_create_openai_func_called(self, mock_create):
        """openai binding 应委托给 _create_openai_func。"""
        from lightrag.api.llm_factory import create_dynamic_llm_func

        mock_create.return_value = AsyncMock()
        create_dynamic_llm_func(
            binding="openai", model="gpt-4o-mini", host="http://api.openai.com"
        )

        mock_create.assert_called_once_with(
            "gpt-4o-mini", "http://api.openai.com", None, 120, default_headers=None
        )

    @patch("lightrag.api.llm_factory._create_azure_openai_func")
    def test_create_azure_openai_func_called(self, mock_create):
        """azure_openai binding 应委托给 _create_azure_openai_func。"""
        from lightrag.api.llm_factory import create_dynamic_llm_func

        mock_create.return_value = AsyncMock()
        create_dynamic_llm_func(
            binding="azure_openai", model="gpt-4", api_key="test-key"
        )

        mock_create.assert_called_once_with(
            "gpt-4", None, "test-key", 120, default_headers=None
        )

    @patch("lightrag.api.llm_factory._create_gemini_func")
    def test_create_gemini_func_called(self, mock_create):
        """gemini binding 应委托给 _create_gemini_func。"""
        from lightrag.api.llm_factory import create_dynamic_llm_func

        mock_create.return_value = AsyncMock()
        create_dynamic_llm_func(
            binding="gemini", model="gemini-pro", api_key="gemini-key"
        )

        mock_create.assert_called_once_with(
            "gemini-pro", None, "gemini-key", 120
        )

    @patch("lightrag.api.llm_factory._create_ollama_func")
    def test_create_ollama_func_called(self, mock_create):
        """ollama binding 应委托给 _create_ollama_func。"""
        from lightrag.api.llm_factory import create_dynamic_llm_func

        mock_create.return_value = AsyncMock()
        create_dynamic_llm_func(
            binding="ollama",
            model="qwen2",
            default_model_kwargs={"options": {"num_ctx": 32768}},
        )

        mock_create.assert_called_once_with(
            "qwen2", None, None, 120, {"options": {"num_ctx": 32768}}
        )

    @patch("lightrag.api.llm_factory._create_bedrock_func")
    def test_create_bedrock_func_called(self, mock_create):
        """aws_bedrock binding 应委托给 _create_bedrock_func。"""
        from lightrag.api.llm_factory import create_dynamic_llm_func

        mock_create.return_value = AsyncMock()
        create_dynamic_llm_func(binding="aws_bedrock", model="anthropic.claude-3")

        mock_create.assert_called_once_with("anthropic.claude-3", 120)

    @patch("lightrag.api.llm_factory._create_lollms_func")
    def test_create_lollms_func_called(self, mock_create):
        """lollms binding 应委托给 _create_lollms_func。"""
        from lightrag.api.llm_factory import create_dynamic_llm_func

        mock_create.return_value = AsyncMock()
        create_dynamic_llm_func(
            binding="lollms", model="test-model", host="http://localhost:9600"
        )

        mock_create.assert_called_once_with(
            "test-model", "http://localhost:9600", None, 120
        )


# ──────────────────────────────────────────────
# OpenAI 函数测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestOpenAIFunc:
    """测试 OpenAI 兼容 API 的动态函数。"""

    @patch("lightrag.llm.openai.openai_complete_if_cache")
    async def test_openai_passes_default_headers(self, mock_complete):
        """验证 default_headers 正确转发到底层函数。"""
        from lightrag.api.llm_factory import _create_openai_func

        mock_complete.return_value = "response text"
        func = _create_openai_func(
            model="gpt-4o-mini",
            host="http://custom.api.com",
            api_key="sk-custom",
            timeout=30,
            default_headers={"X-Apig-AppCode": "app-code-123"},
        )

        await func("test prompt")

        call_kwargs = mock_complete.call_args.kwargs
        assert call_kwargs["default_headers"] == {"X-Apig-AppCode": "app-code-123"}

    @patch("lightrag.llm.openai.openai_complete_if_cache")
    async def test_openai_passes_base_url_and_api_key(self, mock_complete):
        """验证 base_url 和 api_key 正确转发到底层函数。"""
        from lightrag.api.llm_factory import _create_openai_func

        mock_complete.return_value = "response text"
        func = _create_openai_func(
            model="gpt-4o-mini",
            host="http://custom.api.com",
            api_key="sk-custom",
            timeout=30,
        )

        result = await func("test prompt", system_prompt="system")

        assert result == "response text"
        call_kwargs = mock_complete.call_args
        assert call_kwargs.kwargs["base_url"] == "http://custom.api.com"
        assert call_kwargs.kwargs["api_key"] == "sk-custom"
        assert call_kwargs.kwargs["timeout"] == 30

    @patch("lightrag.llm.openai.openai_complete_if_cache")
    async def test_openai_strips_hashing_kv_and_priority(self, mock_complete):
        """验证 hashing_kv 和 _priority 参数被移除。"""
        from lightrag.api.llm_factory import _create_openai_func

        mock_complete.return_value = "ok"
        func = _create_openai_func(model="gpt-4o-mini", host=None, api_key=None, timeout=60)

        await func("prompt", hashing_kv={"key": "val"}, _priority=5)

        call_kwargs = mock_complete.call_args.kwargs
        assert "hashing_kv" not in call_kwargs
        assert "_priority" not in call_kwargs

    @patch("lightrag.llm.openai.openai_complete_if_cache")
    async def test_openai_keyword_extraction_format(self, mock_complete):
        """keyword_extraction=True 时 kwargs.pop 覆盖了函数参数，当前行为不设置 response_format。"""
        from lightrag.api.llm_factory import _create_openai_func

        mock_complete.return_value = "keywords"
        func = _create_openai_func(model="gpt-4o-mini", host=None, api_key=None, timeout=60)

        await func("prompt", keyword_extraction=True)

        call_kwargs = mock_complete.call_args.kwargs
        # 当前行为：keyword_extraction 通过函数参数传入，但 kwargs.pop 覆盖为 None
        # 因此 response_format 不会被设置
        assert "response_format" not in call_kwargs


# ──────────────────────────────────────────────
# Azure OpenAI 函数测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestAzureOpenAIFunc:
    """测试 Azure OpenAI 的动态函数。"""

    @patch("lightrag.llm.openai.azure_openai_complete_if_cache")
    async def test_azure_uses_env_api_key(self, mock_complete):
        """api_key 为 None 时应回退到环境变量。"""
        from lightrag.api.llm_factory import _create_azure_openai_func

        mock_complete.return_value = "azure response"

        with patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "env-key-123"}):
            func = _create_azure_openai_func(
                model="gpt-4",
                host=None,
                api_key=None,
                timeout=60,
            )
            await func("prompt")

        call_kwargs = mock_complete.call_args.kwargs
        assert call_kwargs["api_key"] == "env-key-123"

    @patch("lightrag.llm.openai.azure_openai_complete_if_cache")
    async def test_azure_explicit_api_key_overrides_env(self, mock_complete):
        """显式传入的 api_key 应优先于环境变量。"""
        from lightrag.api.llm_factory import _create_azure_openai_func

        mock_complete.return_value = "azure response"

        with patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "env-key"}):
            func = _create_azure_openai_func(
                model="gpt-4",
                host=None,
                api_key="explicit-key",
                timeout=60,
            )
            await func("prompt")

        call_kwargs = mock_complete.call_args.kwargs
        assert call_kwargs["api_key"] == "explicit-key"


# ──────────────────────────────────────────────
# Ollama 函数测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestOllamaFunc:
    """测试 Ollama 的动态函数。"""

    @patch("lightrag.llm.ollama._ollama_model_if_cache")
    async def test_ollama_applies_server_options(self, mock_model):
        """应注入服务端的 Ollama options（如 num_ctx）。"""
        from lightrag.api.llm_factory import _create_ollama_func

        mock_model.return_value = "ollama response"
        func = _create_ollama_func(
            model="qwen2",
            host=None,
            api_key=None,
            timeout=60,
            default_model_kwargs={"options": {"num_ctx": 32768, "temperature": 0.7}},
        )

        await func("prompt")

        call_kwargs = mock_model.call_args.kwargs
        assert call_kwargs["options"] == {"num_ctx": 32768, "temperature": 0.7}

    @patch("lightrag.llm.ollama._ollama_model_if_cache")
    async def test_ollama_call_kwargs_override_options(self, mock_model):
        """调用时传入的 kwargs 应覆盖服务端默认 options。"""
        from lightrag.api.llm_factory import _create_ollama_func

        mock_model.return_value = "ollama response"
        func = _create_ollama_func(
            model="qwen2",
            host=None,
            api_key=None,
            timeout=60,
            default_model_kwargs={"options": {"num_ctx": 32768}},
        )

        await func("prompt", options={"num_ctx": 65536})

        call_kwargs = mock_model.call_args.kwargs
        assert call_kwargs["options"] == {"num_ctx": 65536}

    @patch("lightrag.llm.ollama._ollama_model_if_cache")
    async def test_ollama_timeout_zero_converts_to_none(self, mock_model):
        """timeout=0 应转换为 None（无限等待）。"""
        from lightrag.api.llm_factory import _create_ollama_func

        mock_model.return_value = "response"
        func = _create_ollama_func(
            model="qwen2", host=None, api_key=None, timeout=0
        )

        await func("prompt")

        call_kwargs = mock_model.call_args.kwargs
        assert call_kwargs["timeout"] is None

    @patch("lightrag.llm.ollama._ollama_model_if_cache")
    async def test_ollama_stream_kwarg_passed(self, mock_model):
        """stream 参数应正确传递。"""
        from lightrag.api.llm_factory import _create_ollama_func

        mock_model.return_value = "response"
        func = _create_ollama_func(
            model="qwen2", host=None, api_key=None, timeout=60
        )

        await func("prompt", stream=True)

        call_kwargs = mock_model.call_args.kwargs
        assert call_kwargs["stream"] is True


# ──────────────────────────────────────────────
# LoLLMs 函数测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestLoLLMsFunc:
    """测试 LoLLMs 的动态函数。"""

    @patch("lightrag.llm.lollms.lollms_model_if_cache")
    async def test_lollms_default_host(self, mock_model):
        """host 为 None 时应使用默认的 localhost:9600。"""
        from lightrag.api.llm_factory import _create_lollms_func

        mock_model.return_value = "lollms response"
        func = _create_lollms_func(
            model="test-model", host=None, api_key=None, timeout=60
        )

        await func("prompt")

        call_kwargs = mock_model.call_args.kwargs
        assert call_kwargs["base_url"] == "http://localhost:9600"

    @patch("lightrag.llm.lollms.lollms_model_if_cache")
    async def test_lollms_custom_host(self, mock_model):
        """显式传入的 host 应覆盖默认值。"""
        from lightrag.api.llm_factory import _create_lollms_func

        mock_model.return_value = "lollms response"
        func = _create_lollms_func(
            model="test-model", host="http://custom:8080", api_key=None, timeout=60
        )

        await func("prompt")

        call_kwargs = mock_model.call_args.kwargs
        assert call_kwargs["base_url"] == "http://custom:8080"
