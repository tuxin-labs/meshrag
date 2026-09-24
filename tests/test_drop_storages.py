"""
LightRAG.drop_storages 重试与结果收集测试。

覆盖 lightrag/lightrag.py 的 drop_storages：
- 成功存储 → results["success"]
- 抛异常的存储 → 重试 N 次后记入 failed + failed_details
- 返回逻辑失败 ({"status":"error"}) 的存储 → 同样触发重试
- None 存储 → 跳过
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.offline
class TestDropStoragesRetry:
    """测试 drop_storages 的单存储级重试与失败明细收集。"""

    @pytest.mark.asyncio
    async def test_retries_failed_storage_and_records_details(self, tmp_path):
        """失败存储应重试 3 次，最终记入 failed + failed_details；成功存储只调用一次。"""
        from lightrag.base import StoragesStatus
        from lightrag.lightrag import LightRAG

        # 用 __new__ 跳过 __init__，手工构造最小实例
        rag = LightRAG.__new__(LightRAG)
        rag._storages_status = StoragesStatus.INITIALIZED  # 跳过 initialize_storages
        rag.workspace = "test_ws"
        rag.working_dir = str(tmp_path)

        # 成功的存储
        ok_storage = MagicMock()
        ok_storage.drop = AsyncMock(return_value={"status": "success", "message": "ok"})
        # 始终抛异常的存储（应重试 3 次）
        exc_storage = MagicMock()
        exc_storage.drop = AsyncMock(side_effect=ConnectionError("connection refused"))
        # 返回逻辑失败的存储（应重试 3 次）
        err_storage = MagicMock()
        err_storage.drop = AsyncMock(
            return_value={"status": "error", "message": "auth failed"}
        )

        rag.llm_response_cache = ok_storage
        rag.text_chunks = exc_storage
        rag.full_docs = err_storage
        # 其余存储置 None（验证跳过 + 避免 AttributeError）
        for attr in (
            "full_entities",
            "full_relations",
            "entity_chunks",
            "relation_chunks",
            "entities_vdb",
            "relationships_vdb",
            "chunks_vdb",
            "chunk_entity_relation_graph",
            "doc_status",
        ):
            setattr(rag, attr, None)

        # patch initialize / finalize，避免依赖真实实现
        with (
            patch.object(LightRAG, "initialize_storages", AsyncMock()),
            patch.object(LightRAG, "finalize_storages", AsyncMock()),
        ):
            results = await rag.drop_storages()

        # 成功存储：在 success，只调用 1 次
        assert "llm_response_cache" in results["success"]
        assert ok_storage.drop.await_count == 1
        # 失败存储：在 failed + failed_details，各重试 3 次
        assert "text_chunks" in results["failed"]
        assert "full_docs" in results["failed"]
        assert exc_storage.drop.await_count == 3
        assert err_storage.drop.await_count == 3
        # failed_details 含失败原因
        assert len(results["failed_details"]) == 2
        details_map = {d["storage"]: d["error"] for d in results["failed_details"]}
        assert "connection refused" in details_map["text_chunks"]
        assert "auth failed" in details_map["full_docs"]

    @pytest.mark.asyncio
    async def test_eventual_success_after_transient_failure(self, tmp_path):
        """存储首次失败、后续成功时，重试后应计入 success（不进 failed）。"""
        from lightrag.base import StoragesStatus
        from lightrag.lightrag import LightRAG

        rag = LightRAG.__new__(LightRAG)
        rag._storages_status = StoragesStatus.INITIALIZED
        rag.workspace = "test_ws"
        rag.working_dir = str(tmp_path)

        # 首次抛异常，第二次成功
        flaky_storage = MagicMock()
        flaky_storage.drop = AsyncMock(
            side_effect=[
                ConnectionError("transient"),
                {"status": "success", "message": "ok"},
            ]
        )
        rag.llm_response_cache = flaky_storage
        for attr in (
            "text_chunks",
            "full_docs",
            "full_entities",
            "full_relations",
            "entity_chunks",
            "relation_chunks",
            "entities_vdb",
            "relationships_vdb",
            "chunks_vdb",
            "chunk_entity_relation_graph",
            "doc_status",
        ):
            setattr(rag, attr, None)

        with (
            patch.object(LightRAG, "initialize_storages", AsyncMock()),
            patch.object(LightRAG, "finalize_storages", AsyncMock()),
        ):
            results = await rag.drop_storages()

        # 重试后成功 → 进 success，不进 failed
        assert "llm_response_cache" in results["success"]
        assert "llm_response_cache" not in results["failed"]
        assert flaky_storage.drop.await_count == 2  # 首次失败 + 第二次成功
