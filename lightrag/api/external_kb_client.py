"""
外部知识库 API HTTP 客户端模块。

提供异步 HTTP 请求能力，向外部知识库 API 发送查询并获取检索结果。
支持两种外部知识库类型：
- retrieval（检索型）：返回文本块 chunks
- rag（完整 RAG 服务型）：返回最终答案

使用 aiohttp + tenacity 实现可靠的异步请求（与 rerank.py 模式一致）。
"""

import asyncio
import uuid
from typing import Any

import aiohttp
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from lightrag.utils import logger


# ──────────────────────────────────────────────
# 共享 ClientSession（模块级单例，延迟初始化）
# ──────────────────────────────────────────────

_session: aiohttp.ClientSession | None = None
_session_lock = asyncio.Lock()


async def get_session() -> aiohttp.ClientSession:
    """获取共享 ClientSession，首次调用时延迟创建。"""
    global _session
    if _session is None or _session.closed:
        async with _session_lock:
            if _session is None or _session.closed:
                _session = aiohttp.ClientSession()
                logger.debug("外部知识库共享 ClientSession 已创建")
    return _session


async def close_session() -> None:
    """关闭共享 ClientSession，应在应用关闭时调用。

    与 get_session() 共用 _session_lock，确保创建/关闭同属一个临界区，
    避免关闭瞬间并发请求拿到已关闭 session 的竞态。
    """
    global _session
    async with _session_lock:
        if _session is not None and not _session.closed:
            await _session.close()
            _session = None
            logger.debug("外部知识库共享 ClientSession 已关闭")


# ──────────────────────────────────────────────
# 检索型外部知识库客户端
# ──────────────────────────────────────────────


async def fetch_external_kb(
    query: str,
    url: str,
    api_key: str | None = None,
    top_k: int = 5,
    timeout: int = 120,
) -> tuple[list[dict], str | None]:
    """请求检索型外部知识库 API，返回标准化的 chunk 列表。

    向指定的外部知识库 API 发送 POST 请求，获取与查询相关的文档片段。
    失败时记录警告并返回空列表和错误信息，不影响本地查询流程。

    Args:
        query: 查询文本
        url: 外部知识库 API 地址
        api_key: 认证密钥，可选
        top_k: 请求的最大结果数量，默认 5
        timeout: 请求超时时间（秒），默认 120

    Returns:
        (chunks, error_message) 元组:
        - chunks: 标准化的 chunk 字典列表
        - error_message: 失败时的错误描述，成功时为 None
    """
    try:
        return await _do_fetch_retrieval(query, url, api_key, top_k, timeout), None
    except Exception as e:
        logger.warning(f"外部知识库请求失败 [{url}]: {e}")
        return [], f"外部知识库(检索型) [{url}] 请求失败: {e}"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    reraise=True,
)
async def _do_fetch_retrieval(
    query: str,
    url: str,
    api_key: str | None,
    top_k: int,
    timeout: int,
) -> list[dict]:
    """执行检索型外部知识库 API 请求（带重试）。

    请求格式: POST {url}  Body: {"query": ..., "top_k": ...}
    期望响应: {"status": "success", "results": [{"content": ..., "file_path": ..., "score": ...}]}
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: dict[str, Any] = {"query": query, "top_k": top_k}

    session = await get_session()
    async with session.post(
        url,
        headers=headers,
        json=payload,
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as response:
        if response.status != 200:
            body = await response.text()
            raise ValueError(f"外部知识库返回 HTTP {response.status}: {body[:200]}")

        data = await response.json()

    # 验证响应格式
    if not isinstance(data, dict):
        raise ValueError(f"外部知识库返回非 JSON 对象: {type(data)}")

    status = data.get("status", "")
    if status != "success":
        msg = data.get("message", data.get("msg", ""))
        raise ValueError(f"外部知识库返回错误状态 '{status}': {msg}")

    results = data.get("results", [])
    if not isinstance(results, list):
        raise ValueError(f"外部知识库 results 字段非数组: {type(results)}")

    # 标准化为统一的 chunk 格式
    chunks: list[dict] = []
    for i, item in enumerate(results):
        if not isinstance(item, dict):
            continue
        content = item.get("content", "")
        if not content:
            continue
        chunks.append(
            {
                "content": content,
                "file_path": item.get("file_path", f"external_result_{i}"),
                "chunk_id": item.get("chunk_id", f"ext_{uuid.uuid4().hex[:12]}"),
            }
        )

    logger.info(f"外部知识库(检索型) [{url}] 返回 {len(chunks)} 条结果")
    return chunks


# ──────────────────────────────────────────────
# 完整 RAG 服务型外部知识库客户端
# ──────────────────────────────────────────────


async def fetch_rag_kb(
    query: str,
    url: str,
    api_key: str | None = None,
    timeout: int = 120,
) -> tuple[dict, str | None]:
    """请求完整 RAG 服务型外部知识库，返回最终答案。

    外部 API 内部包含向量检索和 LLM，直接返回对查询的完整答案。
    失败时记录警告并返回空答案和错误信息，不影响其他数据源。

    Args:
        query: 查询文本
        url: 外部 RAG 服务 API 地址
        api_key: 认证密钥，可选
        timeout: 请求超时时间（秒），默认 120（含 LLM 推理时间）

    Returns:
        (result, error_message) 元组:
        - result: 答案字典 {answer, references, source}
        - error_message: 失败时的错误描述，成功时为 None
    """
    try:
        return await _do_fetch_rag(query, url, api_key, timeout), None
    except Exception as e:
        logger.warning(f"外部 RAG 服务请求失败 [{url}]: {e}")
        return {
            "answer": "",
            "references": [],
            "source": url,
        }, f"外部知识库(RAG服务型) [{url}] 请求失败: {e}"


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    reraise=True,
)
async def _do_fetch_rag(
    query: str,
    url: str,
    api_key: str | None,
    timeout: int,
) -> dict:
    """执行完整 RAG 服务型外部知识库请求（带重试）。

    请求格式: POST {url}  Body: {"query": ...}
    期望响应: {"status": "success", "answer": "...", "references": [...]}
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: dict[str, Any] = {"query": query}

    session = await get_session()
    async with session.post(
        url,
        headers=headers,
        json=payload,
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as response:
        if response.status != 200:
            body = await response.text()
            raise ValueError(f"外部 RAG 服务返回 HTTP {response.status}: {body[:200]}")

        data = await response.json()

    # 验证响应格式
    if not isinstance(data, dict):
        raise ValueError(f"外部 RAG 服务返回非 JSON 对象: {type(data)}")

    status = data.get("status", "")
    if status != "success":
        msg = data.get("message", data.get("msg", ""))
        raise ValueError(f"外部 RAG 服务返回错误状态 '{status}': {msg}")

    answer = data.get("answer", "")
    if not isinstance(answer, str):
        raise ValueError(f"外部 RAG 服务 answer 字段非字符串: {type(answer)}")

    references = data.get("references", [])
    if not isinstance(references, list):
        logger.warning(f"外部 RAG 服务 references 字段非数组: {type(references)}")
        references = []

    # 仅保留 file_path，与 retrieval 型引用格式保持一致
    clean_refs = [
        {"file_path": ref.get("file_path", "")}
        for ref in references
        if isinstance(ref, dict) and ref.get("file_path")
    ]

    logger.info(
        f"外部 RAG 服务 [{url}] 返回答案（{len(answer)} 字符，{len(clean_refs)} 条引用）"
    )
    return {"answer": answer, "references": clean_refs, "source": url}
