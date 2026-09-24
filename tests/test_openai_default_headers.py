"""
Tests for OpenAI-compatible custom default headers (e.g. ModelArts X-Apig-AppCode).
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.offline
class TestCreateOpenAIAsyncClientHeaders:
    """create_openai_async_client should merge custom headers with built-in defaults."""

    def test_merges_custom_headers_with_defaults(self):
        from lightrag.llm.openai import create_openai_async_client

        with patch("lightrag.llm.openai.AsyncOpenAI") as mock_client_cls:
            create_openai_async_client(
                api_key="test-key",
                base_url="https://example.com/v1",
                client_configs={"default_headers": {"X-Apig-AppCode": "app-code-123"}},
            )

        _, kwargs = mock_client_cls.call_args
        headers = kwargs["default_headers"]
        assert headers["Content-Type"] == "application/json"
        assert "User-Agent" in headers
        assert headers["X-Apig-AppCode"] == "app-code-123"

    def test_builtin_headers_not_overwritten_by_partial_custom(self):
        from lightrag.llm.openai import create_openai_async_client

        with patch("lightrag.llm.openai.AsyncOpenAI") as mock_client_cls:
            create_openai_async_client(
                api_key="test-key",
                client_configs={"default_headers": {"X-Custom": "value"}},
            )

        headers = mock_client_cls.call_args.kwargs["default_headers"]
        assert headers["Content-Type"] == "application/json"
        assert headers["X-Custom"] == "value"


@pytest.mark.offline
class TestOpenAICompleteDefaultHeaders:
    """openai_complete_if_cache should route headers to the client, not the API call."""

    @patch("lightrag.llm.openai.create_openai_async_client")
    async def test_default_headers_kwarg_goes_to_client_configs(
        self, mock_create_client
    ):
        from lightrag.llm.openai import openai_complete_if_cache

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_create_client.return_value = mock_client

        await openai_complete_if_cache(
            "test-model",
            "hello",
            default_headers={"X-Apig-AppCode": "app-code-123"},
        )

        client_configs = mock_create_client.call_args.kwargs["client_configs"]
        assert client_configs["default_headers"]["X-Apig-AppCode"] == "app-code-123"

        api_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "default_headers" not in api_kwargs

    @patch("lightrag.llm.openai.create_openai_async_client")
    async def test_env_openai_llm_default_headers_applied(
        self, mock_create_client, monkeypatch
    ):
        from lightrag.llm.openai import openai_complete_if_cache

        monkeypatch.setenv(
            "OPENAI_LLM_DEFAULT_HEADERS",
            json.dumps({"X-Apig-AppCode": "from-env"}),
        )

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_create_client.return_value = mock_client

        await openai_complete_if_cache("test-model", "hello")

        client_configs = mock_create_client.call_args.kwargs["client_configs"]
        assert client_configs["default_headers"]["X-Apig-AppCode"] == "from-env"

    @patch("lightrag.llm.openai.create_openai_async_client")
    async def test_query_default_headers_override_env(
        self, mock_create_client, monkeypatch
    ):
        from lightrag.llm.openai import openai_complete_if_cache

        monkeypatch.setenv(
            "OPENAI_LLM_DEFAULT_HEADERS",
            json.dumps({"X-Apig-AppCode": "from-env"}),
        )

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_create_client.return_value = mock_client

        await openai_complete_if_cache(
            "test-model",
            "hello",
            default_headers={"X-Apig-AppCode": "from-query"},
        )

        client_configs = mock_create_client.call_args.kwargs["client_configs"]
        assert client_configs["default_headers"]["X-Apig-AppCode"] == "from-query"


@pytest.mark.offline
class TestOpenAILLMOptionsDefaultHeaders:
    """OpenAILLMOptions should expose default_headers via OPENAI_LLM_DEFAULT_HEADERS."""

    def test_options_dict_includes_default_headers(self):
        from argparse import Namespace
        from lightrag.llm.binding_options import OpenAILLMOptions

        args = Namespace(openai_llm_default_headers={"X-Apig-AppCode": "app-code-123"})
        options = OpenAILLMOptions.options_dict(args)
        assert options["default_headers"] == {"X-Apig-AppCode": "app-code-123"}
