"""
测试 apipeline_enqueue_documents 的 overwrite 参数（内容级覆盖语义）。

背景：
- 文件名不同、内容相同的情况下，/upload 同步阶段无法检测内容重复
  （二进制文件需先解析才能算内容 MD5），只能依赖 pipeline 层检测。
- 原先 pipeline 检测到内容重复时只会创建 dup-FAILED 记录，不感知 overwrite，
  导致 Java 端"先删后传 + overwrite=true"失效。
- 本测试覆盖 apipeline_enqueue_documents(overwrite=True) 的核心行为：
  内容重复时删除旧文档并重新入队，而非创建 dup 记录。
"""

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


@pytest.mark.asyncio
async def test_overwrite_true_deletes_same_content_doc_and_skips_dup_record(tmp_path):
    """overwrite=True 时，内容重复应删除旧文档并重新入队，不创建 dup 记录。"""
    rag = await _build_rag(tmp_path, "overwrite_true")
    try:
        content = "duplicate content for overwrite test"
        original_file_path = "original_name.pdf"
        doc_id = compute_mdhash_id(content, prefix="doc-")

        # 第一次入队：建立文档
        await rag.apipeline_enqueue_documents(
            input=content, file_paths=original_file_path
        )
        existing = await rag.doc_status.get_by_id(doc_id)
        assert existing is not None
        assert existing["file_path"] == original_file_path

        # 第二次入队：相同内容、不同文件名、overwrite=True
        new_file_path = "renamed_file.pdf"
        track_id = generate_track_id("upload")
        await rag.apipeline_enqueue_documents(
            input=content,
            file_paths=new_file_path,
            track_id=track_id,
            overwrite=True,
        )

        # 不应创建 dup- 记录
        dup_record_id = compute_mdhash_id(f"{doc_id}-{track_id}", prefix="dup-")
        dup_record = await rag.doc_status.get_by_id(dup_record_id)
        assert dup_record is None, "overwrite=True 不应创建 dup-FAILED 记录"

        # 旧文档应被重新入队（file_path 更新为新文件名，状态回到 PENDING）
        doc = await rag.doc_status.get_by_id(doc_id)
        assert doc is not None, "文档应重新入队"
        assert doc["file_path"] == new_file_path, "file_path 应更新为新文件名"
        assert doc["track_id"] == track_id, "track_id 应为本次操作的 track_id"
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_overwrite_false_creates_dup_record_for_same_content(tmp_path):
    """overwrite=False 时，内容重复应创建 dup- 记录（保留原行为，回归保护）。"""
    rag = await _build_rag(tmp_path, "overwrite_false")
    try:
        content = "duplicate content default behavior test"
        original_file_path = "first.pdf"
        doc_id = compute_mdhash_id(content, prefix="doc-")

        # 第一次入队
        await rag.apipeline_enqueue_documents(
            input=content, file_paths=original_file_path
        )

        # 第二次入队：相同内容、不同文件名、overwrite=False（默认）
        new_file_path = "second.pdf"
        track_id = generate_track_id("upload")
        await rag.apipeline_enqueue_documents(
            input=content,
            file_paths=new_file_path,
            track_id=track_id,
        )

        # 应创建 dup- 记录
        dup_record_id = compute_mdhash_id(f"{doc_id}-{track_id}", prefix="dup-")
        dup_record = await rag.doc_status.get_by_id(dup_record_id)
        assert dup_record is not None, "overwrite=False 应创建 dup 记录"
        assert dup_record["status"] == DocStatus.FAILED
        assert dup_record["metadata"]["is_duplicate"] is True
        assert dup_record["metadata"]["original_doc_id"] == doc_id
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_overwrite_true_no_duplicate_keeps_default_behavior(tmp_path):
    """overwrite=True 但无内容重复时，行为与默认一致（正常入队）。"""
    rag = await _build_rag(tmp_path, "overwrite_new")
    try:
        content = "brand new content no duplicate"
        doc_id = compute_mdhash_id(content, prefix="doc-")

        await rag.apipeline_enqueue_documents(
            input=content, file_paths="new.pdf", overwrite=True
        )

        doc = await rag.doc_status.get_by_id(doc_id)
        assert doc is not None
        assert doc["status"] == DocStatus.PENDING
    finally:
        await rag.finalize_storages()
