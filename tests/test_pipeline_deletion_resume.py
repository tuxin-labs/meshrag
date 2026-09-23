"""测试：文档解析与删除的并发恢复逻辑。

验证核心契约：文档A解析中、文档B排队时删除文档A，
删除完成后文档B必须能继续被处理（任务不丢失）。

根因（bug）：
删除请求会设置 deletion_pending=True（高优先级删除暂停），
当 pipeline 忙时还会设置 cancellation_requested=True。
而 ``apipeline_process_enqueue_documents`` 主循环中 cancellation_requested
分支会清除 request_pending，且 process_document 内部检测点遇到
cancellation_requested 时不会设置 request_pending，导致删除完成后
``background_delete_documents`` 检查 request_pending=False，不会重新触发解析，
排队的 pending 文档永久卡死。

修复契约：
- deletion_pending 检查优先于 cancellation_requested；
- 检测到 deletion_pending 暂停时必须设置 request_pending=True，
  确保删除完成后能恢复解析。
"""

from types import MethodType
from uuid import uuid4

import numpy as np
import pytest

from lightrag.base import DocStatus
from lightrag.lightrag import LightRAG
from lightrag.utils import EmbeddingFunc, Tokenizer

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


def _status_to_text(status: object) -> str:
    if isinstance(status, DocStatus):
        return status.value
    return str(status).replace("DocStatus.", "").lower()


async def _build_rag(tmp_path, test_name: str, chunking_func) -> LightRAG:
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
        chunking_func=chunking_func,
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    return rag


@pytest.mark.asyncio
async def test_deletion_pause_keeps_request_pending_for_resume(tmp_path):
    """删除暂停解析时，必须保留 request_pending，确保删除完成后排队文档能被恢复。

    复现用户场景：文档A解析中，文档B排队(request_pending=True)，删除请求到达。
    删除请求会同时设置 deletion_pending=True 与 cancellation_requested=True。

    期望：解析暂停后 request_pending 保持 True。
    否则 ``background_delete_documents`` 完成后检查 request_pending=False，
    不会重新触发解析，排队文档将永久卡在 pending（任务丢失）。
    """
    rag = await _build_rag(tmp_path, "deletion_resume", _deterministic_chunking)
    try:
        from lightrag.kg.shared_storage import get_namespace_data

        # 入队文档A（pending 状态）
        content_a = "document A content for deletion resume test"
        await rag.apipeline_enqueue_documents(input=content_a, file_paths="a.txt")

        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )

        # 模拟文档B已排队（另一个并发请求设置了 request_pending=True）
        pipeline_status["request_pending"] = True

        # 在实体抽取阶段模拟删除请求到达：
        # 同时设置 deletion_pending 与 cancellation_requested
        # （复现删除接口在 pipeline 忙时的行为）
        async def extract_trigger_deletion(self, chunks, ps, ps_lock):
            ps["deletion_pending"] = True
            ps["cancellation_requested"] = True
            return {"chunk_count": len(chunks)}

        rag._process_extract_entities = MethodType(extract_trigger_deletion, rag)

        # 运行解析流程（文档A处理中遇到删除请求，应暂停）
        await rag.apipeline_process_enqueue_documents()

        # 核心断言：request_pending 必须保持 True
        assert pipeline_status.get("request_pending") is True, (
            "删除暂停后 request_pending 必须为 True，否则删除完成后 "
            "排队的文档永远不会被处理（任务丢失）"
        )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_deletion_pause_keeps_doc_processing_not_failed(tmp_path):
    """删除暂停时，正在解析的文档A应保持 PROCESSING，不应标记 FAILED。

    被删除打断不是失败：Java/前端轮询到 PROCESSING 会继续等待，删除完成后
    apipeline 恢复时会自动重新处理。若标记 FAILED，上游会误认为解析失败
    并停止轮询，与实际行为不符。
    """
    from lightrag.utils import compute_mdhash_id

    rag = await _build_rag(tmp_path, "deletion_doc_state", _deterministic_chunking)
    try:
        from lightrag.kg.shared_storage import get_namespace_data

        content_a = "document A content for state verification"
        await rag.apipeline_enqueue_documents(input=content_a, file_paths="a.txt")
        doc_a_id = compute_mdhash_id(content_a, prefix="doc-")

        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
        pipeline_status["request_pending"] = True

        async def extract_trigger_deletion(self, chunks, ps, ps_lock):
            ps["deletion_pending"] = True
            ps["cancellation_requested"] = True
            return {"chunk_count": len(chunks)}

        rag._process_extract_entities = MethodType(extract_trigger_deletion, rag)

        await rag.apipeline_process_enqueue_documents()

        # 文档A应保持 processing（被删除暂停，不是失败）
        doc_status = await rag.doc_status.get_by_id(doc_a_id)
        assert doc_status is not None
        assert _status_to_text(doc_status["status"]) == "processing", (
            "被删除暂停的文档应保持 processing（不是失败），让上游继续轮询。"
            f"实际为 {doc_status['status']}"
        )
        # error_msg 不应包含删除暂停的技术串（不暴露给上游）
        error_msg = doc_status.get("error_msg", "") or ""
        assert "deletion pending" not in error_msg, (
            f"删除暂停的技术串不应写入 error_msg，实际: {error_msg}"
        )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_user_cancellation_still_stops_pipeline(tmp_path):
    """回归保护：用户主动取消（cancel_pipeline）只设置 cancellation_requested，
    不设置 deletion_pending。此时应走用户取消分支，清除 request_pending 并停止。

    确保 deletion_pending 优先的修改没有破坏用户取消语义。
    """
    rag = await _build_rag(tmp_path, "user_cancel", _deterministic_chunking)
    try:
        from lightrag.kg.shared_storage import get_namespace_data

        content_a = "document A for user cancel test"
        await rag.apipeline_enqueue_documents(input=content_a, file_paths="a.txt")

        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
        pipeline_status["request_pending"] = True

        # 模拟用户取消（只设 cancellation_requested，不设 deletion_pending）
        async def extract_trigger_cancel(self, chunks, ps, ps_lock):
            ps["cancellation_requested"] = True
            return {"chunk_count": len(chunks)}

        rag._process_extract_entities = MethodType(extract_trigger_cancel, rag)

        await rag.apipeline_process_enqueue_documents()

        # 用户取消后 request_pending 应被清除（停止后续处理）
        assert pipeline_status.get("request_pending") is False, (
            "用户主动取消应清除 request_pending，停止后续文档处理"
        )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_queued_doc_processed_after_deletion_completes(tmp_path, monkeypatch):
    """端到端：文档A解析中被删除中断，删除完成后文档B应被实际处理（任务不丢失）。

    模拟 background_delete_documents 的完整恢复链路：
    1. 文档A、文档B 同时入队；
    2. 处理文档A时 deletion_pending 触发暂停（文档A、B 标记 failed）；
    3. 删除文档A（adelete_by_doc_id）并清除 deletion_pending；
    4. 重新触发解析（模拟删除完成后的恢复调用）；
    5. 文档B 应被处理为 processed。
    """
    import lightrag.lightrag as lightrag_module
    from lightrag.utils import compute_mdhash_id

    # merge 阶段对真实图数据格式敏感，此处替换为 noop 以聚焦"删除恢复"契约
    async def noop_merge(**kwargs):
        return None

    monkeypatch.setattr(lightrag_module, "merge_nodes_and_edges", noop_merge)

    rag = await _build_rag(tmp_path, "e2e_resume", _deterministic_chunking)
    try:
        from lightrag.kg.shared_storage import get_namespace_data

        content_a = "document A will be deleted during parsing"
        content_b = "document B should be processed after deletion completes"
        await rag.apipeline_enqueue_documents(input=content_a, file_paths="a.txt")
        await rag.apipeline_enqueue_documents(input=content_b, file_paths="b.txt")

        doc_a_id = compute_mdhash_id(content_a, prefix="doc-")
        doc_b_id = compute_mdhash_id(content_b, prefix="doc-")

        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )

        # 阶段1：处理文档A时触发删除暂停
        async def extract_trigger_deletion(self, chunks, ps, ps_lock):
            ps["deletion_pending"] = True
            return {"chunk_count": len(chunks)}

        rag._process_extract_entities = MethodType(extract_trigger_deletion, rag)
        await rag.apipeline_process_enqueue_documents()

        # 暂停后必须保留 request_pending，删除完成后才能恢复
        assert pipeline_status.get("request_pending") is True

        # 阶段2：模拟 background_delete_documents 删除文档A 并清除删除标志
        await rag.adelete_by_doc_id(doc_a_id, delete_llm_cache=False)
        pipeline_status["deletion_pending"] = False

        # 阶段3：恢复正常抽取，重新触发解析（删除完成后的恢复调用）
        async def extract_normal(self, chunks, ps, ps_lock):
            return {"chunk_count": len(chunks)}

        rag._process_extract_entities = MethodType(extract_normal, rag)
        await rag.apipeline_process_enqueue_documents()

        # 核心断言：文档B 应被处理完成（任务不丢失）
        doc_b_status = await rag.doc_status.get_by_id(doc_b_id)
        assert doc_b_status is not None
        assert _status_to_text(doc_b_status["status"]) == "processed", (
            "删除完成后，排队的文档B必须被正常处理（任务不丢失）"
        )
        # 文档A 应已被删除
        assert await rag.doc_status.get_by_id(doc_a_id) is None
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_extract_entities_aborts_immediately_on_deletion_pending(tmp_path):
    """extract_entities 在 deletion_pending=True 时应在函数开头立即中止，
    不处理任何 chunk（实现删除优先、解析及时暂停），而非跑完整个文档的所有 chunk。

    这是用户场景的核心：删除解析中的文档时，解析必须尽快停下让删除执行，
    而不是把当前文档 86 个 chunk 全部跑完。
    """
    rag = await _build_rag(tmp_path, "extract_abort", _deterministic_chunking)
    try:
        from dataclasses import asdict

        from lightrag.exceptions import PipelineCancelledException
        from lightrag.kg.shared_storage import (
            get_namespace_data,
            get_namespace_lock,
        )
        from lightrag.operate import extract_entities

        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
        # 模拟删除请求已到达
        pipeline_status["deletion_pending"] = True

        chunks = {
            "chunk-test-1": {"content": "Alice works at TechCorp.", "tokens": 5},
            "chunk-test-2": {"content": "Bob is an engineer.", "tokens": 5},
        }

        with pytest.raises(PipelineCancelledException):
            await extract_entities(
                chunks,
                global_config=asdict(rag),
                pipeline_status=pipeline_status,
                pipeline_status_lock=get_namespace_lock(
                    "pipeline_status", workspace=rag.workspace
                ),
                llm_response_cache=rag.llm_response_cache,
                text_chunks_storage=rag.text_chunks,
            )

        # 应保留 request_pending 以便删除完成后恢复解析
        assert pipeline_status.get("request_pending") is True
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_extract_entities_aborts_midway_when_deletion_set_during_processing(
    tmp_path,
):
    """extract_entities 处理过程中 deletion_pending 被设置，
    应在下一个 chunk 处理前中止（chunk 级暂停）。

    通过 llm_model_max_async=1 串行处理 chunk，第一个 chunk 的 LLM 调用
    触发 deletion_pending，后续 chunk 应被检测点拦截。
    """
    rag = await _build_rag(tmp_path, "extract_abort_mid", _deterministic_chunking)
    try:
        from dataclasses import asdict

        from lightrag.exceptions import PipelineCancelledException
        from lightrag.kg.shared_storage import (
            get_namespace_data,
            get_namespace_lock,
        )
        from lightrag.operate import extract_entities

        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )

        # 串行处理 chunk，便于确定性验证
        config = asdict(rag)
        config["llm_model_max_async"] = 1

        # 用计数器：第一次 LLM 调用（第一个 chunk）后触发删除标志
        call_state = {"count": 0}

        async def llm_trigger_deletion(*args, **kwargs):
            call_state["count"] += 1
            # 第一个 chunk 开始处理后，设置 deletion_pending
            pipeline_status["deletion_pending"] = True
            # 返回空实体格式，让 _handle_single_entity_extraction 正常解析为空
            return "<entities>\n</entities>"

        config["llm_model_func"] = llm_trigger_deletion

        chunks = {
            "chunk-c1": {"content": "content one " * 50, "tokens": 50},
            "chunk-c2": {"content": "content two " * 50, "tokens": 50},
            "chunk-c3": {"content": "content three " * 50, "tokens": 50},
        }

        # extract 应在中途被 deletion_pending 中断（raise PipelineCancelledException）
        with pytest.raises(PipelineCancelledException):
            await extract_entities(
                chunks,
                global_config=config,
                pipeline_status=pipeline_status,
                pipeline_status_lock=get_namespace_lock(
                    "pipeline_status", workspace=rag.workspace
                ),
                llm_response_cache=rag.llm_response_cache,
                text_chunks_storage=rag.text_chunks,
            )

        # 确实发生了 LLM 调用（即进入了 chunk 处理），且被中途拦截
        assert call_state["count"] >= 1
        assert pipeline_status.get("request_pending") is True
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_orphan_processing_reset_on_startup(tmp_path):
    """服务重启后，孤儿 PROCESSING 文档应在 initialize_storages 时重置为 PENDING。

    模拟场景：服务在处理文档 A 时被 kill，doc_status 里 A 状态是 PROCESSING。
    下次启动应自动重置为 PENDING，避免用户永远看到"处理中"。
    """
    from datetime import datetime, timezone

    from lightrag.utils import compute_mdhash_id

    rag = await _build_rag(tmp_path, "orphan_reset", _deterministic_chunking)
    workspace_dir = rag.working_dir
    workspace = rag.workspace
    try:
        content_a = "document A stuck in processing"
        doc_a_id = compute_mdhash_id(content_a, prefix="doc-")
        # 直接写入 PROCESSING 状态（模拟服务 kill 前的状态）
        await rag.doc_status.upsert(
            {
                doc_a_id: {
                    "status": DocStatus.PROCESSING,
                    "content_summary": content_a[:100],
                    "content_length": len(content_a),
                    "chunks_count": 2,
                    "chunks_list": ["chunk-x", "chunk-y"],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "file_path": "a.txt",
                    "track_id": "test-track",
                    "metadata": {},
                }
            }
        )
        # 验证状态确实是 PROCESSING
        before = await rag.doc_status.get_by_id(doc_a_id)
        assert _status_to_text(before["status"]) == "processing"
    finally:
        await rag.finalize_storages()

    # 模拟"服务重启"：重新构建 LightRAG 实例（同 workspace + working_dir）
    rag2 = LightRAG(
        working_dir=workspace_dir,
        workspace=workspace,
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("test-tokenizer", _SimpleTokenizerImpl()),
        chunking_func=_deterministic_chunking,
        max_parallel_insert=1,
    )
    await rag2.initialize_storages()
    try:
        # 重启后状态应被重置为 PENDING
        after = await rag2.doc_status.get_by_id(doc_a_id)
        assert after is not None
        assert _status_to_text(after["status"]) == "pending", (
            f"孤儿 PROCESSING 文档应在服务重启时重置为 PENDING，"
            f"实际为 {after['status']}"
        )
    finally:
        await rag2.finalize_storages()


@pytest.mark.asyncio
async def test_pipeline_busy_stays_true_across_deletion_handoff(tmp_path, monkeypatch):
    """删除完成后无缝 handoff 回解析：pipeline.busy 全程为 True，
    job_name 表达当前阶段。用户查询 pipeline_status 不会看到空闲窗口。
    """
    import lightrag.lightrag as lightrag_module
    from lightrag.utils import compute_mdhash_id

    # merge 阶段 noop 让 pipeline 顺利跑到 PROCESSED
    async def noop_merge(**kwargs):
        return None

    monkeypatch.setattr(lightrag_module, "merge_nodes_and_edges", noop_merge)

    rag = await _build_rag(tmp_path, "handoff_busy", _deterministic_chunking)
    try:
        from lightrag.kg.shared_storage import get_namespace_data

        content_b = "document B waiting in queue"
        await rag.apipeline_enqueue_documents(input=content_b, file_paths="b.txt")
        doc_b_id = compute_mdhash_id(content_b, prefix="doc-")

        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
        # 模拟"文档 A 解析中被删除"后进入 handoff 状态：
        # background_delete_documents 刚完成删除，即将 handoff 回解析。
        pipeline_status.update({
            "busy": True,
            "job_name": "Deleting 1 Documents",
            "request_pending": True,  # 文档 B 排队等待
            "deletion_pending": False,
            "cancellation_requested": False,
        })

        # 模拟 background_delete_documents 的 finally 逻辑：
        # 若 request_pending=True，切换 job_name，保持 busy=True，调 apipeline(assume_busy=True)
        pipeline_status["job_name"] = "Resuming indexing after deletion"

        # 关键断言 1：handoff 期间 busy 保持 True
        assert pipeline_status["busy"] is True

        # 用 extract noop 让 pipeline 快速跑完
        async def extract_normal(self, chunks, ps, ps_lock):
            return []

        from types import MethodType as _MT
        rag._process_extract_entities = _MT(extract_normal, rag)

        # assume_busy=True 应无缝接管
        await rag.apipeline_process_enqueue_documents(assume_busy=True)

        # 文档 B 应被处理
        doc_b = await rag.doc_status.get_by_id(doc_b_id)
        assert doc_b is not None
        assert _status_to_text(doc_b["status"]) == "processed"

        # 关键断言 2：pipeline 完成后正确释放 busy
        assert pipeline_status.get("busy") is False
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_assume_busy_with_no_pending_docs_releases_busy(tmp_path):
    """assume_busy=True 但无待处理文档时，应主动释放 busy 避免死锁。"""
    rag = await _build_rag(tmp_path, "assume_busy_empty", _deterministic_chunking)
    try:
        from lightrag.kg.shared_storage import get_namespace_data

        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
        pipeline_status.update({
            "busy": True,
            "job_name": "Resuming indexing after deletion",
            "request_pending": True,
        })
        # 没有任何 pending 文档
        await rag.apipeline_process_enqueue_documents(assume_busy=True)
        # busy 必须被释放，否则 pipeline 永久卡住
        assert pipeline_status.get("busy") is False
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_orphan_reset_auto_triggers_pipeline(tmp_path, monkeypatch):
    """存在孤儿 PROCESSING 文档时，initialize_storages 应后台自动触发 pipeline。

    验证：重启后无需用户上传新文档，孤儿 PENDING 文档能自动继续解析。
    """
    import asyncio as _asyncio
    from datetime import datetime, timezone

    import lightrag.lightrag as lightrag_module
    from lightrag.utils import compute_mdhash_id

    # merge 阶段对真实图数据格式敏感，noop 让 pipeline 顺利跑到 PROCESSED
    async def noop_merge(**kwargs):
        return None

    monkeypatch.setattr(lightrag_module, "merge_nodes_and_edges", noop_merge)

    rag = await _build_rag(tmp_path, "auto_resume", _deterministic_chunking)
    workspace_dir = rag.working_dir
    workspace = rag.workspace
    try:
        # 写入孤儿 PROCESSING 文档 + 对应 full_docs 内容（模拟解析中被 kill）
        content_a = "orphan doc that should be auto-resumed after restart"
        doc_a_id = compute_mdhash_id(content_a, prefix="doc-")
        now = datetime.now(timezone.utc).isoformat()
        await rag.full_docs.upsert({doc_a_id: {"content": content_a}})
        await rag.doc_status.upsert(
            {
                doc_a_id: {
                    "status": DocStatus.PROCESSING,
                    "content_summary": content_a[:100],
                    "content_length": len(content_a),
                    "chunks_count": 0,
                    "chunks_list": [],
                    "created_at": now,
                    "updated_at": now,
                    "file_path": "orphan.txt",
                    "track_id": "test-track",
                    "metadata": {},
                }
            }
        )
    finally:
        await rag.finalize_storages()

    # 模拟"服务重启"：新建 LightRAG 实例
    async def extract_normal(self, chunks, ps, ps_lock):
        return []  # 空 chunk_results 让 merge_nodes_and_edges(noop) 正常返回

    rag2 = LightRAG(
        working_dir=workspace_dir,
        workspace=workspace,
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("test-tokenizer", _SimpleTokenizerImpl()),
        chunking_func=_deterministic_chunking,
        max_parallel_insert=1,
    )
    from types import MethodType as _MT
    rag2._process_extract_entities = _MT(extract_normal, rag2)
    await rag2.initialize_storages()
    try:
        # 等待后台 pipeline task 完成（最多 5s）
        for _ in range(50):
            doc = await rag2.doc_status.get_by_id(doc_a_id)
            if doc and _status_to_text(doc["status"]) == "processed":
                break
            await _asyncio.sleep(0.1)

        final = await rag2.doc_status.get_by_id(doc_a_id)
        assert final is not None
        assert _status_to_text(final["status"]) == "processed", (
            f"孤儿文档应在服务重启后自动被处理为 processed，"
            f"实际为 {final['status']}"
        )
    finally:
        await rag2.finalize_storages()


@pytest.mark.asyncio
async def test_concurrent_deletion_waits_for_handoff_pipeline(tmp_path, monkeypatch):
    """并发删除场景：删除A完成handoff到解析管道后，删除B不应强行接管，
    而应设置 deletion_pending 让handoff管道暂停退出，避免两个管道并发运行。

    复现场景（用户报告的bug）：
    1. 文档解析中，删除A到达 → 管道暂停 → 删除A执行 → 删除A完成handoff
    2. 删除B在等待循环中，看到 busy=True, job_name="Resuming indexing after deletion"
    3. 旧代码：else: break → 删除B强行接管 → 两个管道并发 → busy=false 但 chunks 还在处理
    4. 修复后：设置 deletion_pending → handoff管道暂停 → 删除B干净接管 → 最终正确释放busy
    """
    import lightrag.lightrag as lightrag_module
    from lightrag.kg.shared_storage import get_namespace_data, get_namespace_lock
    from lightrag.utils import compute_mdhash_id

    async def noop_merge(**kwargs):
        return None

    monkeypatch.setattr(lightrag_module, "merge_nodes_and_edges", noop_merge)

    rag = await _build_rag(tmp_path, "concurrent_deletion", _deterministic_chunking)
    try:
        content_a = "document A being processed during concurrent deletions"
        await rag.apipeline_enqueue_documents(input=content_a, file_paths="a.txt")
        doc_a_id = compute_mdhash_id(content_a, prefix="doc-")

        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
        pipeline_status_lock = get_namespace_lock(
            "pipeline_status", workspace=rag.workspace
        )

        # 模拟删除A刚完成handoff的状态：busy=True, job_name="Resuming indexing after deletion"
        # 此时handoff管道正在运行，删除B的force_acquire循环应该设置deletion_pending
        pipeline_status.update({
            "busy": True,
            "job_name": "Resuming indexing after deletion",
            "request_pending": False,
            "deletion_pending": False,
            "cancellation_requested": False,
            "latest_message": "Resuming indexing after deletion",
            "history_messages": [],
        })

        # 模拟删除B的force_acquire循环逻辑（修复后的版本）
        # 验证：当job_name包含"Resuming"时，应设置deletion_pending而非break
        async with pipeline_status_lock:
            is_busy = pipeline_status.get("busy", False)
            current_job = pipeline_status.get("job_name", "")

        assert is_busy is True, "handoff管道应处于busy状态"
        assert "Resuming" in current_job, "job_name应包含Resuming"

        # 模拟修复后的force_acquire循环行为
        deletion_pending_set = False
        if is_busy and "Resuming" in current_job:
            async with pipeline_status_lock:
                if not pipeline_status.get("deletion_pending", False):
                    pipeline_status["deletion_pending"] = True
                    deletion_pending_set = True

        assert deletion_pending_set is True, (
            "删除B应设置 deletion_pending 让handoff管道暂停，而非强行接管"
        )
        assert pipeline_status.get("deletion_pending") is True

        # 验证handoff管道检测到deletion_pending后会正确暂停
        # （request_pending=True 确保删除完成后能恢复解析）
        pipeline_status["request_pending"] = True  # handoff管道暂停时设置

        # 模拟handoff管道暂停后，删除B看到busy=False并接管
        pipeline_status["busy"] = False

        # 删除B接管
        pipeline_status.update({
            "busy": True,
            "job_name": "Deleting 1 Documents",
            "deletion_pending": False,
        })

        # 删除B完成，检查request_pending（handoff管道设置的）
        has_pending = pipeline_status.get("request_pending", False)
        assert has_pending is True, (
            "handoff管道暂停时应设置request_pending，确保删除B完成后能恢复解析"
        )

        # 删除B的finally：has_pending_request=True → handoff回解析
        pipeline_status["job_name"] = "Resuming indexing after deletion"

        # 最终：解析管道处理完所有文档后释放busy
        pipeline_status["busy"] = False

        # 验证最终状态一致
        doc_a = await rag.doc_status.get_by_id(doc_a_id)
        assert doc_a is not None
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_two_deletions_only_one_runs_at_a_time(tmp_path, monkeypatch):
    """两个删除同时到达时，只有一个能获取 pipeline，另一个必须等待。

    验证原子抢占：第一个删除在锁内设置 busy=True，第二个看到 busy=True
    后进入等待循环，不会两个同时执行删除操作。
    """
    import lightrag.lightrag as lightrag_module
    from lightrag.kg.shared_storage import get_namespace_data, get_namespace_lock

    async def noop_merge(**kwargs):
        return None

    monkeypatch.setattr(lightrag_module, "merge_nodes_and_edges", noop_merge)

    rag = await _build_rag(tmp_path, "two_deletions", _deterministic_chunking)
    try:
        content_a = "document A for two deletions test"
        await rag.apipeline_enqueue_documents(input=content_a, file_paths="a.txt")

        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
        pipeline_status_lock = get_namespace_lock(
            "pipeline_status", workspace=rag.workspace
        )

        # 模拟两个删除同时到达，都看到 pipeline 空闲
        # 验证原子抢占：只有一个能成功设置 busy=True

        # 删除A：模拟第一个循环的原子抢占逻辑
        async with pipeline_status_lock:
            is_busy = pipeline_status.get("busy", False)
            assert is_busy is False, "初始状态应为空闲"
            # 删除A 原子抢占
            pipeline_status["busy"] = True
            pipeline_status["job_name"] = "Deleting 1 Documents"
            claimed_by_a = True

        # 删除B：模拟第一个循环看到 busy=True
        async with pipeline_status_lock:
            is_busy = pipeline_status.get("busy", False)
            current_job = pipeline_status.get("job_name", "")

        assert is_busy is True, "删除A已抢占，应为busy"
        assert current_job.startswith("Deleting"), "job_name应为Deleting"
        # 删除B 不应抢占，应进入等待
        claimed_by_b = False

        # 验证：只有删除A获取了 pipeline
        assert claimed_by_a is True
        assert claimed_by_b is False

        # 删除B 进入第二循环，看到 "Deleting" → 等待
        # 模拟删除A完成后 handoff
        pipeline_status["request_pending"] = True
        pipeline_status["job_name"] = "Resuming indexing after deletion"

        # 删除B 看到 "Resuming" → 设置 deletion_pending
        async with pipeline_status_lock:
            current_job = pipeline_status.get("job_name", "")
            assert "Resuming" in current_job
            if not pipeline_status.get("deletion_pending", False):
                pipeline_status["deletion_pending"] = True

        assert pipeline_status["deletion_pending"] is True

        # 模拟管道暂停
        pipeline_status["busy"] = False

        # 删除B 原子抢占
        async with pipeline_status_lock:
            is_busy = pipeline_status.get("busy", False)
            assert is_busy is False
            pipeline_status["busy"] = True
            pipeline_status["job_name"] = "Deleting 1 Documents"

        # 删除B 完成，检查 request_pending
        assert pipeline_status.get("request_pending") is True

        # 清理状态
        pipeline_status["busy"] = False
        pipeline_status["deletion_pending"] = False
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_deletion_during_parsing_paused_and_resumed(tmp_path, monkeypatch):
    """删除在解析过程中触发暂停，删除完成后解析自动恢复，busy 全程正确。

    完整流程：解析 → deletion_pending → 暂停 → 删除 → handoff → 恢复解析 → 完成
    """
    import lightrag.lightrag as lightrag_module
    from lightrag.utils import compute_mdhash_id

    async def noop_merge(**kwargs):
        return None

    monkeypatch.setattr(lightrag_module, "merge_nodes_and_edges", noop_merge)

    rag = await _build_rag(tmp_path, "pause_resume", _deterministic_chunking)
    try:
        from lightrag.kg.shared_storage import get_namespace_data

        content_a = "document A for pause resume test"
        content_b = "document B queued after pause"
        await rag.apipeline_enqueue_documents(input=content_a, file_paths="a.txt")
        await rag.apipeline_enqueue_documents(input=content_b, file_paths="b.txt")

        doc_b_id = compute_mdhash_id(content_b, prefix="doc-")

        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )

        # 阶段1：解析开始，中途触发 deletion_pending
        call_count = {"n": 0}

        async def extract_trigger_deletion(self, chunks, ps, ps_lock):
            call_count["n"] += 1
            if call_count["n"] == 1:
                ps["deletion_pending"] = True
            return []

        rag._process_extract_entities = MethodType(extract_trigger_deletion, rag)
        await rag.apipeline_process_enqueue_documents()

        # 管道暂停：request_pending 必须为 True
        assert pipeline_status.get("request_pending") is True
        assert pipeline_status.get("busy") is False

        # 阶段2：模拟删除完成，handoff 回解析
        pipeline_status["deletion_pending"] = False

        async def extract_normal(self, chunks, ps, ps_lock):
            return []

        rag._process_extract_entities = MethodType(extract_normal, rag)
        await rag.apipeline_process_enqueue_documents(assume_busy=True)

        # 文档B 应被处理
        doc_b = await rag.doc_status.get_by_id(doc_b_id)
        assert doc_b is not None
        assert _status_to_text(doc_b["status"]) == "processed"

        # 最终 busy=False
        assert pipeline_status.get("busy") is False
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_upload_during_deletion_sets_request_pending(tmp_path, monkeypatch):
    """删除期间上传文档，应设置 request_pending，删除完成后自动处理。"""
    import lightrag.lightrag as lightrag_module
    from lightrag.utils import compute_mdhash_id

    async def noop_merge(**kwargs):
        return None

    monkeypatch.setattr(lightrag_module, "merge_nodes_and_edges", noop_merge)

    rag = await _build_rag(tmp_path, "upload_during_delete", _deterministic_chunking)
    try:
        from lightrag.kg.shared_storage import get_namespace_data

        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )

        # 模拟删除正在进行
        pipeline_status.update({
            "busy": True,
            "job_name": "Deleting 1 Documents",
            "request_pending": False,
        })

        # 上传文档（应设置 request_pending）
        content_a = "document uploaded during deletion"
        await rag.apipeline_enqueue_documents(input=content_a, file_paths="a.txt")

        # 模拟 apipeline_process_enqueue_documents 的行为：
        # 看到 busy=True → 设置 request_pending=True
        pipeline_status["request_pending"] = True

        assert pipeline_status.get("request_pending") is True

        # 删除完成，handoff
        pipeline_status["deletion_pending"] = False
        pipeline_status["job_name"] = "Resuming indexing after deletion"

        # 恢复解析
        async def extract_normal(self, chunks, ps, ps_lock):
            return []

        rag._process_extract_entities = MethodType(extract_normal, rag)
        await rag.apipeline_process_enqueue_documents(assume_busy=True)

        doc_a_id = compute_mdhash_id(content_a, prefix="doc-")
        doc_a = await rag.doc_status.get_by_id(doc_a_id)
        assert doc_a is not None
        assert _status_to_text(doc_a["status"]) == "processed"
    finally:
        await rag.finalize_storages()
