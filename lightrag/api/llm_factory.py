"""
Dynamic LLM function factory for query-time model override.

This module creates temporary LLM functions based on runtime parameters
(binding, model, host, api_key) passed via API request. The created function
conforms to the standard LightRAG LLM function signature and is injected into
QueryParam.model_func for per-query model switching.
"""

import os
from typing import AsyncIterator, Callable, Awaitable, Union

# Supported binding types
SUPPORTED_BINDINGS = [
    "openai",
    "ollama",
    "azure_openai",
    "gemini",
    "aws_bedrock",
    "lollms",
]


def create_dynamic_llm_func(
    binding: str,
    model: str,
    host: str | None = None,
    api_key: str | None = None,
    timeout: int = 120,
    default_model_kwargs: dict | None = None,
    default_headers: dict[str, str] | None = None,
) -> Callable[..., Awaitable[Union[str, AsyncIterator[str]]]]:
    """Create a dynamic LLM function for query-time model override.

    Args:
        binding: LLM provider type. One of: openai, ollama, azure_openai, gemini, aws_bedrock, lollms.
        model: Model name (e.g., 'gpt-4o-mini', 'qwen-flash').
        host: API endpoint URL. If None, provider default is used.
        api_key: API key for authentication. If None, provider default or env var is used.
        timeout: Request timeout in seconds.
        default_model_kwargs: Server's default LLM model kwargs (e.g., Ollama options with num_ctx).
        default_headers: Custom HTTP headers for OpenAI-compatible APIs (e.g. X-Apig-AppCode).

    Returns:
        Async callable with standard LightRAG LLM function signature.

    Raises:
        ValueError: If binding type is not supported.
    """
    if binding not in SUPPORTED_BINDINGS:
        raise ValueError(
            f"Unsupported llm_binding '{binding}'. Must be one of: {', '.join(SUPPORTED_BINDINGS)}"
        )

    if binding == "openai":
        return _create_openai_func(
            model, host, api_key, timeout, default_headers=default_headers
        )
    elif binding == "azure_openai":
        return _create_azure_openai_func(
            model, host, api_key, timeout, default_headers=default_headers
        )
    elif binding == "gemini":
        return _create_gemini_func(model, host, api_key, timeout)
    elif binding == "ollama":
        return _create_ollama_func(model, host, api_key, timeout, default_model_kwargs)
    elif binding == "aws_bedrock":
        return _create_bedrock_func(model, timeout)
    elif binding == "lollms":
        return _create_lollms_func(model, host, api_key, timeout)
    else:
        raise ValueError(f"Unsupported llm_binding '{binding}'")


def _create_openai_func(
    model: str,
    host: str | None,
    api_key: str | None,
    timeout: int,
    default_headers: dict[str, str] | None = None,
):
    """Create dynamic LLM function for OpenAI-compatible APIs."""
    from lightrag.llm.openai import openai_complete_if_cache

    async def dynamic_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
    ) -> str:
        from lightrag.types import GPTKeywordExtractionFormat

        # Clean up kwargs not applicable to the underlying function
        kwargs.pop("hashing_kv", None)
        kwargs.pop("_priority", None)
        keyword_extraction = kwargs.pop("keyword_extraction", None)
        if keyword_extraction:
            kwargs["response_format"] = GPTKeywordExtractionFormat
        if history_messages is None:
            history_messages = []
        kwargs["timeout"] = timeout
        if default_headers:
            kwargs["default_headers"] = default_headers

        return await openai_complete_if_cache(
            model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=host,
            api_key=api_key,
            **kwargs,
        )

    return dynamic_func


def _create_azure_openai_func(
    model: str,
    host: str | None,
    api_key: str | None,
    timeout: int,
    default_headers: dict[str, str] | None = None,
):
    """Create dynamic LLM function for Azure OpenAI."""
    # azure_openai module re-exports from openai, so import directly
    from lightrag.llm.openai import azure_openai_complete_if_cache

    async def dynamic_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
    ) -> str:
        from lightrag.types import GPTKeywordExtractionFormat

        kwargs.pop("hashing_kv", None)
        kwargs.pop("_priority", None)
        keyword_extraction = kwargs.pop("keyword_extraction", None)
        if keyword_extraction:
            kwargs["response_format"] = GPTKeywordExtractionFormat
        if history_messages is None:
            history_messages = []
        kwargs["timeout"] = timeout
        if default_headers:
            kwargs["default_headers"] = default_headers

        return await azure_openai_complete_if_cache(
            model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=host,
            api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
            **kwargs,
        )

    return dynamic_func


def _create_gemini_func(
    model: str, host: str | None, api_key: str | None, timeout: int
):
    """Create dynamic LLM function for Google Gemini."""
    from lightrag.llm.gemini import gemini_complete_if_cache

    async def dynamic_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
    ) -> str:
        kwargs.pop("hashing_kv", None)
        kwargs.pop("_priority", None)
        if history_messages is None:
            history_messages = []
        kwargs["timeout"] = timeout

        return await gemini_complete_if_cache(
            model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=host,
            api_key=api_key,
            **kwargs,
        )

    return dynamic_func


def _create_ollama_func(
    model: str,
    host: str | None,
    api_key: str | None,
    timeout: int,
    default_model_kwargs: dict | None = None,
):
    """Create dynamic LLM function for Ollama."""
    from lightrag.llm.ollama import _ollama_model_if_cache

    # Inherit server's Ollama options (num_ctx etc.) when available
    server_options = (default_model_kwargs or {}).get("options")

    async def dynamic_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
    ) -> Union[str, AsyncIterator[str]]:
        kwargs.pop("hashing_kv", None)
        kwargs.pop("_priority", None)
        keyword_extraction = kwargs.pop("keyword_extraction", None)
        if keyword_extraction:
            kwargs["format"] = "json"
        if history_messages is None:
            history_messages = []
        stream = kwargs.pop("stream", False)

        effective_timeout = timeout if timeout else None
        if effective_timeout == 0:
            effective_timeout = None

        # Apply server's Ollama options (num_ctx etc.) as defaults,
        # but allow per-call kwargs to override them
        if server_options and "options" not in kwargs:
            kwargs["options"] = server_options

        return await _ollama_model_if_cache(
            model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            stream=stream,
            host=host,
            timeout=effective_timeout,
            api_key=api_key,
            **kwargs,
        )

    return dynamic_func


def _create_bedrock_func(model: str, timeout: int):
    """Create dynamic LLM function for AWS Bedrock."""
    from lightrag.llm.bedrock import bedrock_complete_if_cache

    async def dynamic_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
    ) -> Union[str, AsyncIterator[str]]:
        kwargs.pop("hashing_kv", None)
        kwargs.pop("_priority", None)
        if history_messages is None:
            history_messages = []

        return await bedrock_complete_if_cache(
            model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )

    return dynamic_func


def _create_lollms_func(
    model: str, host: str | None, api_key: str | None, timeout: int
):
    """Create dynamic LLM function for LoLLMs."""
    from lightrag.llm.lollms import lollms_model_if_cache

    async def dynamic_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
    ) -> Union[str, AsyncIterator[str]]:
        kwargs.pop("hashing_kv", None)
        kwargs.pop("_priority", None)
        if history_messages is None:
            history_messages = []

        effective_host = host or "http://localhost:9600"
        kwargs["timeout"] = timeout
        kwargs["api_key"] = api_key

        return await lollms_model_if_cache(
            model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=effective_host,
            **kwargs,
        )

    return dynamic_func
