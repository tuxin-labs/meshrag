"""
测试方案A：解析失败（FAILED）文档重新解析放行。

背景（docs/LightRAG文档解析失败后无法重新解析-根因分析与修复方案.md）：
- 解析失败后 doc_status 永久保留 status=FAILED 记录；
- 重新解析走 upload 链路时，三道判重闸只判断记录是否存在、不判断状态，
  失败文档被当作"已存在内容"永久拦截，报「内容与已存在文件重复，未解析」。
- 方案A：三道闸对 FAILED 记录放行，允许失败文档直接重新解析。
  - 闸1 文件名判重（upload 路由 get_doc_by_file_path 命中）
  - 闸2 内容 hash 同步预检（_check_content_duplicate）
  - 闸3 入队判重（apipeline_enqueue_documents 的 filter_keys）

仅放行 FAILED；pending / processing / processed 仍照常拦截（回归保护）。
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import numpy as np
import pytest

from lightrag.base import DocStatus
from lightrag.lightrag import LightRAG
from lightrag.utils import (
    EmbeddingFunc,
    Tokenizer,
    compute_mdhash_id,
    generate_track_id,
)

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# 闸3：apipeline_enqueue_documents 入队判重（真实 LightRAG 实例）
# ---------------------------------------------------------------------------


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


async def _dummy_llm(*args, **kwargs) -> str:
    return "ok"


def _deterministic_chunking(
    tokenizer,
    content: str,
    split_by_character,
    split_by_character_only: bool,
    chunk_overlap_token_size: int,
    chunk_token_size: int,
) -> list[dict]:
    return [
        {"tokens": 1, "content": f"{content}::chunk1", "chunk_order_index": 0},
        {"tokens": 1, "content": f"{content}::chunk2", "chunk_order_index": 1},
    ]


async def _build_rag(tmp_path, test_name: str) -> LightRAG:
    workspace = f"{test_name}_{uuid4().hex[:8]}"
    rag = LightRAG(
        working_dir=str(tmp_path / test_name),
        workspace=workspace,
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8,
            max_token_size=8192,
            func=_dummy_embedding,
        ),
        tokenizer=Tokenizer("test-tokenizer", _SimpleTokenizerImpl()),
        chunking_func=_deterministic_chunking,
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    return rag


async def _mark_doc_status(rag: LightRAG, doc_id: str, status: DocStatus) -> None:
    """把指定 doc_id 的 doc_status 记录置为给定状态（模拟解析失败/成功后的终态）。"""
    await rag.doc_status.upsert(
        {
            doc_id: {
                "status": status,
                "content_summary": "summary",
                "content_length": 10,
                "chunks_count": 0,
                "chunks_list": [],
                "error_msg": "LLM unavailable" if status == DocStatus.FAILED else "",
                "file_path": "first.pdf",
                "track_id": "old_track",
                "metadata": {},
            }
        }
    )


@pytest.mark.asyncio
async def test_enqueue_failed_doc_repasses_and_resets_to_pending(tmp_path):
    """FAILED 文档重新入队应放行：状态覆盖为 PENDING，不创建 dup 记录。"""
    rag = await _build_rag(tmp_path, "failed_repass")
    try:
        content = "failed document content for re-parse"
        doc_id = compute_mdhash_id(content, prefix="doc-")

        # 第一次入队后模拟解析失败
        await rag.apipeline_enqueue_documents(input=content, file_paths="first.pdf")
        await _mark_doc_status(rag, doc_id, DocStatus.FAILED)

        # 重新解析：同内容重新入队
        track_id = generate_track_id("upload")
        await rag.apipeline_enqueue_documents(
            input=content, file_paths="first.pdf", track_id=track_id
        )

        # 不应创建 dup-FAILED 记录
        dup_record_id = compute_mdhash_id(f"{doc_id}-{track_id}", prefix="dup-")
        dup_record = await rag.doc_status.get_by_id(dup_record_id)
        assert dup_record is None, "FAILED 文档重新入队不应创建 dup 记录"

        # 原记录应被覆盖为 PENDING，归属本次重新解析
        doc = await rag.doc_status.get_by_id(doc_id)
        assert doc is not None, "FAILED 文档应重新入队"
        assert doc["status"] == DocStatus.PENDING, "状态应覆盖为 PENDING"
        assert doc["track_id"] == track_id, "track_id 应为本次操作的 track_id"
        assert doc["file_path"] == "first.pdf"
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_enqueue_failed_doc_reenqueued_with_error_msg_cleared(tmp_path):
    """FAILED 文档重新入队后，旧 error_msg 不应残留在新 PENDING 记录中。"""
    rag = await _build_rag(tmp_path, "failed_errmsg")
    try:
        content = "content with stale error message"
        doc_id = compute_mdhash_id(content, prefix="doc-")

        await rag.apipeline_enqueue_documents(input=content, file_paths="a.pdf")
        await _mark_doc_status(rag, doc_id, DocStatus.FAILED)

        await rag.apipeline_enqueue_documents(input=content, file_paths="a.pdf")

        doc = await rag.doc_status.get_by_id(doc_id)
        assert doc is not None
        assert doc["status"] == DocStatus.PENDING
        assert not doc.get("error_msg"), "重新入队后不应残留旧的 error_msg"
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_enqueue_processed_doc_still_blocked(tmp_path):
    """PROCESSED 文档重新入队仍应拦截（回归保护：仅放行 FAILED）。"""
    rag = await _build_rag(tmp_path, "processed_block")
    try:
        content = "successfully processed content"
        doc_id = compute_mdhash_id(content, prefix="doc-")

        await rag.apipeline_enqueue_documents(input=content, file_paths="first.pdf")
        await _mark_doc_status(rag, doc_id, DocStatus.PROCESSED)

        track_id = generate_track_id("upload")
        await rag.apipeline_enqueue_documents(
            input=content, file_paths="second.pdf", track_id=track_id
        )

        # 应创建 dup 记录
        dup_record_id = compute_mdhash_id(f"{doc_id}-{track_id}", prefix="dup-")
        dup_record = await rag.doc_status.get_by_id(dup_record_id)
        assert dup_record is not None, "PROCESSED 文档重复入队应创建 dup 记录"
        assert dup_record["metadata"]["is_duplicate"] is True

        # 原记录不应被覆盖
        doc = await rag.doc_status.get_by_id(doc_id)
        assert doc["status"] == DocStatus.PROCESSED
        assert doc["track_id"] == "old_track"
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_enqueue_pending_doc_still_blocked(tmp_path):
    """PENDING 文档重新入队仍应拦截（幂等保护：仅放行 FAILED）。"""
    rag = await _build_rag(tmp_path, "pending_block")
    try:
        content = "pending content blocked from re-enqueue"
        doc_id = compute_mdhash_id(content, prefix="doc-")

        await rag.apipeline_enqueue_documents(input=content, file_paths="first.pdf")
        # 显式置为 PENDING + 固定 track_id，便于断言记录未被覆盖
        await _mark_doc_status(rag, doc_id, DocStatus.PENDING)

        track_id = generate_track_id("upload")
        await rag.apipeline_enqueue_documents(
            input=content, file_paths="first.pdf", track_id=track_id
        )

        dup_record_id = compute_mdhash_id(f"{doc_id}-{track_id}", prefix="dup-")
        dup_record = await rag.doc_status.get_by_id(dup_record_id)
        assert dup_record is not None, "PENDING 文档重复入队应创建 dup 记录"

        doc = await rag.doc_status.get_by_id(doc_id)
        assert doc["status"] == DocStatus.PENDING
        assert doc["track_id"] == "old_track", "PENDING 文档不应被重新入队覆盖"
    finally:
        await rag.finalize_storages()


# ---------------------------------------------------------------------------
# 闸2：_check_content_duplicate 内容 hash 同步预检（mock doc_status）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_content_check_failed_doc_returns_none(tmp_path):
    """内容 hash 命中 FAILED 记录应返回 None 放行，临时文件保留交入队。"""
    from unittest.mock import AsyncMock, MagicMock

    from lightrag.api.routers.document_routes import _check_content_duplicate
    from lightrag.utils import sanitize_text_for_encoding

    text = "failed content for re-upload"
    p = tmp_path / "failed.txt"
    p.write_text(text, encoding="utf-8")
    doc_id = compute_mdhash_id(sanitize_text_for_encoding(text), prefix="doc-")

    existing = {
        "id": doc_id,
        "status": "failed",
        "track_id": "old_track",
        "file_path": "old_name.pdf",
    }
    mock_rag = MagicMock()
    mock_rag.doc_status = AsyncMock()
    mock_rag.doc_status.get_by_id = AsyncMock(return_value=existing)

    assert await _check_content_duplicate(mock_rag, p) is None
    assert p.exists(), "FAILED 放行时不应删除临时文件"


@pytest.mark.asyncio
async def test_content_check_processed_doc_still_duplicated(tmp_path):
    """内容 hash 命中 PROCESSED 记录仍应返回 duplicated（回归保护）。"""
    from unittest.mock import AsyncMock, MagicMock

    from lightrag.api.routers.document_routes import _check_content_duplicate
    from lightrag.utils import sanitize_text_for_encoding

    text = "processed content still duplicated"
    p = tmp_path / "dup.txt"
    p.write_text(text, encoding="utf-8")
    doc_id = compute_mdhash_id(sanitize_text_for_encoding(text), prefix="doc-")

    existing = {
        "id": doc_id,
        "status": "processed",
        "track_id": "old_track",
        "file_path": "old_name.pdf",
    }
    mock_rag = MagicMock()
    mock_rag.doc_status = AsyncMock()
    mock_rag.doc_status.get_by_id = AsyncMock(return_value=existing)

    resp = await _check_content_duplicate(mock_rag, p)
    assert resp is not None
    assert resp.status == "duplicated"
    assert not p.exists(), "命中已处理内容时应清理临时文件"


# ---------------------------------------------------------------------------
# 闸1：upload 路由文件名判重（直接调用路由端点函数，绕开 HTTP 层）
# ---------------------------------------------------------------------------


@pytest.fixture()
def upload_endpoint(tmp_path):
    """构造 mock RAGManager 的 upload 端点，返回 (endpoint, mock_rag)。

    直接调用端点异步函数，BackgroundTasks 用真实实例但不触发执行
    （不经 FastAPI 响应机制），避免后台任务干扰断言。
    """
    from io import BytesIO

    from fastapi import BackgroundTasks
    from starlette.datastructures import UploadFile

    from lightrag.api.routers.document_routes import create_document_routes

    mock_rag = MagicMock()
    mock_rag.doc_status = MagicMock()
    mock_rag.doc_status.get_doc_by_file_path = AsyncMock(return_value=None)
    mock_rag.doc_status.get_doc_id_by_file_path = AsyncMock(return_value="doc-x")
    # 内容预检放行（内容不重复）
    mock_rag.doc_status.get_by_id = AsyncMock(return_value=None)
    mock_rag.adelete_by_doc_id = AsyncMock()

    mock_rag_mgr = MagicMock()
    mock_rag_mgr.get_rag = AsyncMock(return_value=mock_rag)

    mock_doc_mgr = MagicMock()
    mock_doc_mgr.base_input_dir = str(tmp_path)

    router = create_document_routes(mock_rag_mgr, mock_doc_mgr, api_key=None)
    # create_document_routes 向模块级 router 累积注册路由，须取本次
    # fixture 最新注册的一条（否则会拿到此前测试闭包捕获的旧 mock）
    route = [
        r for r in router.routes if getattr(r, "path", None) == "/documents/upload"
    ][-1]

    def _call(filename: str = "doc.txt", content: bytes = b"hello"):
        file = UploadFile(file=BytesIO(content), filename=filename)
        return route.endpoint(
            kb_id="kb1",
            background_tasks=BackgroundTasks(),
            file=file,
            overwrite=False,
        )

    return _call, mock_rag


@pytest.mark.asyncio
async def test_upload_filename_failed_doc_allows_reupload(upload_endpoint):
    """同名 FAILED 文档重新上传应放行（继续入队流程），不触发删除。"""
    call_upload, mock_rag = upload_endpoint
    mock_rag.doc_status.get_doc_by_file_path = AsyncMock(
        return_value={
            "id": "doc-x",
            "status": "failed",
            "track_id": "old_track",
            "file_path": "doc.txt",
        }
    )

    resp = await call_upload()

    assert resp.status == "success", "FAILED 文档重新上传应放行"
    # 放行不走 overwrite 删除分支（保留 LLM 缓存，由入队覆盖旧记录）
    mock_rag.adelete_by_doc_id.assert_not_called()


@pytest.mark.asyncio
async def test_upload_filename_processed_doc_still_duplicated(upload_endpoint):
    """同名 PROCESSED 文档重新上传仍应返回 duplicated（回归保护）。"""
    call_upload, mock_rag = upload_endpoint
    mock_rag.doc_status.get_doc_by_file_path = AsyncMock(
        return_value={
            "id": "doc-y",
            "status": "processed",
            "track_id": "old_track",
            "file_path": "doc.txt",
        }
    )

    resp = await call_upload()

    assert resp.status == "duplicated", "PROCESSED 同名文档仍应拦截"
    mock_rag.adelete_by_doc_id.assert_not_called()
