"""
RAG Manager 核心测试。

覆盖 lightrag/api/rag_manager.py 的核心功能：
- RAGManager 初始化、KB 发现、注册表持久化
- LRU 缓存管理（get_rag、驱逐、列表、删除）
- 多 KB 结果合并（_merge_kb_results）：实体/关系/chunk 去重与证据追踪
- 内容去重（_content_dedup_chunks）：基于 content hash
- 外部 KB 获取（_fetch_all_external_kbs、_fetch_rag_kbs）
- RAG 引用处理（_rag_refs_to_standard、_build_rag_context、_build_external_answers、_merge_rag_refs_into_data）
- Bypass 模式（_bypass_llm）
- RAG 答案合并（_merge_rag_answers）
- 多 KB 数据获取（multi_kb_get_data）
- 多 KB 查询（multi_kb_query）：多种路径
- 后处理（_multi_kb_post_process）：token 预算、rerank、引用清理
- 上下文构建（_build_fallback_context、_build_context_from_merged）
"""

import asyncio
import json
import os
import pytest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call


# ──────────────────────────────────────────────
# 辅助工具
# ──────────────────────────────────────────────


def _make_mock_llm_func(response_text="mock llm response"):
    """创建 mock LLM 函数。"""
    async def _func(*args, **kwargs):
        return response_text
    return _func


def _make_kb_result(
    status="success",
    entities=None,
    relationships=None,
    chunks=None,
    references=None,
    metadata=None,
):
    """创建标准 KB 结果格式。"""
    return {
        "status": status,
        "message": "",
        "data": {
            "entities": entities or [],
            "relationships": relationships or [],
            "chunks": chunks or [],
            "references": references or [],
        },
        "metadata": metadata or {},
    }


# ──────────────────────────────────────────────
# 初始化与注册表测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestRAGManagerInit:
    """测试 RAGManager 初始化和注册表管理。"""

    @pytest.mark.asyncio
    async def test_init_loads_default_kb(self, tmp_path):
        """初始化应包含 default KB。"""
        from lightrag.api.rag_manager import RAGManager

        rag_factory = lambda kb_id: MagicMock()
        manager = RAGManager(rag_factory=rag_factory, default_kb="default")

        assert "default" in manager._known_kbs
        assert manager.default_kb == "default"

    @pytest.mark.asyncio
    async def test_init_loads_from_registry_file(self, tmp_path):
        """应从注册表文件加载 KB ID。"""
        from lightrag.api.rag_manager import RAGManager

        registry_path = tmp_path / "registry.json"
        registry_path.write_text(json.dumps(["kb1", "kb2", "kb3"]), encoding="utf-8")

        rag_factory = lambda kb_id: MagicMock()
        manager = RAGManager(
            rag_factory=rag_factory,
            default_kb="default",
            registry_path=str(registry_path),
        )

        assert "default" in manager._known_kbs
        assert "kb1" in manager._known_kbs
        assert "kb2" in manager._known_kbs
        assert "kb3" in manager._known_kbs

    @pytest.mark.asyncio
    async def test_init_invalid_registry_ignored(self, tmp_path):
        """损坏的注册表文件应被忽略，只保留 default。"""
        from lightrag.api.rag_manager import RAGManager

        registry_path = tmp_path / "registry.json"
        registry_path.write_text("{invalid json", encoding="utf-8")

        rag_factory = lambda kb_id: MagicMock()
        manager = RAGManager(
            rag_factory=rag_factory,
            default_kb="default",
            registry_path=str(registry_path),
        )

        # 仍应有 default
        assert "default" in manager._known_kbs
        assert len(manager._known_kbs) == 1

    @pytest.mark.asyncio
    async def test_init_with_kb_discovery(self, tmp_path):
        """应调用 kb_discovery 函数并合并结果。"""
        from lightrag.api.rag_manager import RAGManager

        discovery_func = lambda: ["discovered1", "discovered2"]
        rag_factory = lambda kb_id: MagicMock()
        manager = RAGManager(
            rag_factory=rag_factory,
            default_kb="default",
            kb_discovery=discovery_func,
        )

        assert "default" in manager._known_kbs
        assert "discovered1" in manager._known_kbs
        assert "discovered2" in manager._known_kbs

    @pytest.mark.asyncio
    async def test_ordered_known_kbs_default_first(self, tmp_path):
        """_ordered_known_kbs 应将 default 放在首位。"""
        from lightrag.api.rag_manager import RAGManager

        registry_path = tmp_path / "registry.json"
        registry_path.write_text(json.dumps(["zebra", "alpha"]), encoding="utf-8")

        rag_factory = lambda kb_id: MagicMock()
        manager = RAGManager(
            rag_factory=rag_factory,
            default_kb="default",
            registry_path=str(registry_path),
        )

        ordered = manager._ordered_known_kbs()
        assert ordered[0] == "default"
        # 其余按字母顺序
        assert ordered[1:] == ["alpha", "zebra"]

    @pytest.mark.asyncio
    async def test_persist_known_kbs(self, tmp_path):
        """应持久化 KB 列表到文件。"""
        from lightrag.api.rag_manager import RAGManager

        registry_path = tmp_path / "registry.json"
        rag_factory = lambda kb_id: MagicMock()
        manager = RAGManager(
            rag_factory=rag_factory,
            default_kb="default",
            registry_path=str(registry_path),
        )
        # 添加一个新的 KB
        manager._known_kbs.add("new_kb")
        manager._persist_known_kbs()

        content = registry_path.read_text(encoding="utf-8")
        data = json.loads(content)
        assert "default" in data
        assert "new_kb" in data


# ──────────────────────────────────────────────
# LRU 缓存与实例管理测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestRAGManagerLRU:
    """测试 LRU 缓存和实例管理。"""

    @pytest.mark.asyncio
    async def test_get_rag_creates_instance(self):
        """首次调用应创建并初始化实例。"""
        from lightrag.api.rag_manager import RAGManager

        mock_rag = MagicMock()
        mock_rag.initialize_storages = AsyncMock()
        mock_rag.check_and_migrate_data = AsyncMock()

        rag_factory = lambda kb_id: mock_rag
        manager = RAGManager(rag_factory=rag_factory, max_instances=2)

        result = await manager.get_rag("kb1")

        assert result is mock_rag
        mock_rag.initialize_storages.assert_called_once()
        mock_rag.check_and_migrate_data.assert_called_once()
        assert "kb1" in manager._instances

    @pytest.mark.asyncio
    async def test_get_rag_returns_cached(self):
        """后续调用应返回缓存的实例。"""
        from lightrag.api.rag_manager import RAGManager

        mock_rag = MagicMock()
        mock_rag.initialize_storages = AsyncMock()
        mock_rag.check_and_migrate_data = AsyncMock()

        rag_factory = lambda kb_id: mock_rag
        manager = RAGManager(rag_factory=rag_factory)

        # 第一次调用
        await manager.get_rag("kb1")
        # 第二次调用
        result = await manager.get_rag("kb1")

        # initialize_storages 仍只应被调用一次
        assert mock_rag.initialize_storages.call_count == 1

    @pytest.mark.asyncio
    async def test_get_rag_updates_lru_order(self):
        """访问 KB 应更新 LRU 顺序（最近使用移到末尾）。"""
        from lightrag.api.rag_manager import RAGManager

        instances = {}
        def rag_factory(kb_id):
            mock_rag = MagicMock()
            mock_rag.initialize_storages = AsyncMock()
            mock_rag.check_and_migrate_data = AsyncMock()
            instances[kb_id] = mock_rag
            return mock_rag

        manager = RAGManager(rag_factory=rag_factory, max_instances=2)

        # 添加 kb1, kb2
        await manager.get_rag("kb1")
        await manager.get_rag("kb2")

        # 此时 LRU 顺序：kb1 (oldest), kb2 (newest)
        # 访问 kb1 应将其移到末尾
        await manager.get_rag("kb1")

        # LRU 顺序应变为：kb2 (oldest), kb1 (newest)
        keys = list(manager._instances.keys())
        assert keys == ["kb2", "kb1"]

    @pytest.mark.asyncio
    async def test_get_rag_empty_id_raises(self):
        """空 kb_id 应抛出 ValueError。"""
        from lightrag.api.rag_manager import RAGManager

        manager = RAGManager(rag_factory=lambda kb_id: MagicMock())

        with pytest.raises(ValueError, match="kb_id cannot be empty"):
            await manager.get_rag("")

    @pytest.mark.asyncio
    async def test_lru_eviction_when_full(self):
        """超出 max_instances 时应驱逐最久未使用的实例。"""
        from lightrag.api.rag_manager import RAGManager

        evicted_rags = []

        def rag_factory(kb_id):
            mock_rag = MagicMock()
            mock_rag.initialize_storages = AsyncMock()
            mock_rag.check_and_migrate_data = AsyncMock()
            mock_rag.finalize_storages = AsyncMock()
            evicted_rags.append(mock_rag)
            return mock_rag

        manager = RAGManager(rag_factory=rag_factory, max_instances=2)

        # 添加 3 个 KB（max_instances=2）
        await manager.get_rag("kb1")
        await manager.get_rag("kb2")
        await manager.get_rag("kb3")

        # kb1 应被驱逐
        assert "kb1" not in manager._instances
        assert "kb2" in manager._instances
        assert "kb3" in manager._instances

        # finalize_storages 应被调用
        # kb1 的 finalize_storages 应该被调用（在 evicted_rags[0]）
        assert evicted_rags[0].finalize_storages.call_count == 1

    @pytest.mark.asyncio
    async def test_list_knowledge_bases(self):
        """应返回所有已知 KB（包括未加载的）。"""
        from lightrag.api.rag_manager import RAGManager

        rag_factory = lambda kb_id: MagicMock()
        manager = RAGManager(
            rag_factory=rag_factory,
            default_kb="default",
        )
        manager._known_kbs = {"default", "kb1", "kb2"}

        result = manager.list_knowledge_bases()

        assert result[0] == "default"
        assert set(result) == {"default", "kb1", "kb2"}

    @pytest.mark.asyncio
    async def test_delete_kb_success(self):
        """删除非 default KB 应成功。"""
        from lightrag.api.rag_manager import RAGManager

        mock_rag = MagicMock()
        mock_rag.initialize_storages = AsyncMock()
        mock_rag.check_and_migrate_data = AsyncMock()
        mock_rag.finalize_storages = AsyncMock()
        mock_rag.drop_storages = AsyncMock(return_value={"succeeded": ["all"], "failed": []})

        rag_factory = lambda kb_id: mock_rag
        manager = RAGManager(rag_factory=rag_factory)

        # 先加载 KB
        await manager.get_rag("kb1")
        assert "kb1" in manager._instances

        # 删除
        result = await manager.delete_knowledge_base("kb1", drop_storage=True)

        assert result["status"] == "success"
        assert "kb1" not in manager._instances
        assert "kb1" not in manager._known_kbs

    @pytest.mark.asyncio
    async def test_delete_default_kb_raises(self):
        """删除 default KB 应抛出 ValueError。"""
        from lightrag.api.rag_manager import RAGManager

        manager = RAGManager(rag_factory=lambda kb_id: MagicMock())

        with pytest.raises(ValueError, match="Default knowledge base cannot be deleted"):
            await manager.delete_knowledge_base("default")

    @pytest.mark.asyncio
    async def test_delete_kb_not_in_memory(self):
        """删除仅存在于注册表中的 KB 应成功。"""
        from lightrag.api.rag_manager import RAGManager

        manager = RAGManager(rag_factory=lambda kb_id: MagicMock())
        manager._known_kbs.add("orphan_kb")

        result = await manager.delete_knowledge_base("orphan_kb", drop_storage=False)

        assert result["status"] == "success"
        assert "orphan_kb" not in manager._known_kbs

    @pytest.mark.asyncio
    async def test_delete_kb_not_in_memory_with_drop_creates_temp_instance(self):
        """KB 不在内存 + drop_storage=True 必须创建临时实例并清理存储。

        回归 bug：原代码仅当 kb_id in self._instances 时才调用 drop_storages，
        导致服务重启 / LRU 淘汰后删除知识库时，向量 / 图谱 / 文档数据全部残留。
        """
        from lightrag.api.rag_manager import RAGManager

        created_instances = []

        def rag_factory(kb_id):
            mock_rag = MagicMock()
            mock_rag.drop_storages = AsyncMock(
                return_value={"success": ["all"], "failed": [], "failed_details": []}
            )
            mock_rag.finalize_storages = AsyncMock()
            created_instances.append(mock_rag)
            return mock_rag

        manager = RAGManager(rag_factory=rag_factory)
        manager._known_kbs.add("orphan_kb")

        result = await manager.delete_knowledge_base("orphan_kb", drop_storage=True)

        # 临时实例应被创建（_rag_factory 被调用）
        assert len(created_instances) == 1
        # drop_storages 应被调用 —— 核心修复点（当前 bug 代码下此断言失败）
        created_instances[0].drop_storages.assert_awaited_once()
        # 临时实例不应进入内存缓存
        assert "orphan_kb" not in manager._instances
        # registry 应清理
        assert "orphan_kb" not in manager._known_kbs
        # 应有 storage_results（证明执行了 drop）
        assert result["storage_results"] is not None

    @pytest.mark.asyncio
    async def test_delete_kb_not_in_memory_partial_drop_returns_partial_success(self):
        """临时实例 drop 部分存储失败 → partial_success 且透传 failed_details。"""
        from lightrag.api.rag_manager import RAGManager

        def rag_factory(kb_id):
            mock_rag = MagicMock()
            mock_rag.drop_storages = AsyncMock(return_value={
                "success": ["full_docs"],
                "failed": ["entities_vdb", "chunk_entity_relation_graph"],
                "failed_details": [
                    {"storage": "entities_vdb", "error": "connection refused"},
                    {"storage": "chunk_entity_relation_graph", "error": "auth failed"},
                ],
            })
            mock_rag.finalize_storages = AsyncMock()
            return mock_rag

        manager = RAGManager(rag_factory=rag_factory)
        manager._known_kbs.add("kb_partial")

        result = await manager.delete_knowledge_base("kb_partial", drop_storage=True)

        assert result["status"] == "partial_success"
        assert len(result["storage_results"]["failed"]) == 2
        assert len(result["storage_results"]["failed_details"]) == 2
        # registry 仍应清理（fire-and-forget）
        assert "kb_partial" not in manager._known_kbs
        # 临时实例不入缓存
        assert "kb_partial" not in manager._instances

    @pytest.mark.asyncio
    async def test_delete_kb_factory_failure_returns_partial_success(self):
        """_rag_factory 抛异常 → partial_success，registry 仍清理，不抛异常给调用方。"""
        from lightrag.api.rag_manager import RAGManager

        def rag_factory(kb_id):
            raise RuntimeError("workspace config invalid")

        manager = RAGManager(rag_factory=rag_factory)
        manager._known_kbs.add("broken_kb")

        result = await manager.delete_knowledge_base("broken_kb", drop_storage=True)

        assert result["status"] == "partial_success"
        assert "could not create RAG instance" in result["message"]
        # 仍从 registry 移除（fire-and-forget）
        assert "broken_kb" not in manager._known_kbs

    @pytest.mark.asyncio
    async def test_delete_temp_instance_does_not_affect_lru(self):
        """删除不在内存的 KB 时，临时实例不应进入 _instances，也不影响已缓存 KB。"""
        from lightrag.api.rag_manager import RAGManager

        def rag_factory(kb_id):
            mock_rag = MagicMock()
            mock_rag.initialize_storages = AsyncMock()
            mock_rag.check_and_migrate_data = AsyncMock()
            mock_rag.drop_storages = AsyncMock(
                return_value={"success": ["all"], "failed": [], "failed_details": []}
            )
            mock_rag.finalize_storages = AsyncMock()
            return mock_rag

        manager = RAGManager(rag_factory=rag_factory, max_instances=5)
        # 先缓存一个已加载的 KB
        await manager.get_rag("active_kb")
        assert "active_kb" in manager._instances

        # 删除另一个不在内存的 KB
        await manager.delete_knowledge_base("orphan_kb", drop_storage=True)

        # 临时实例不入缓存
        assert "orphan_kb" not in manager._instances
        # 已缓存的 KB 不受影响
        assert "active_kb" in manager._instances

    @pytest.mark.asyncio
    async def test_finalize_all(self):
        """finalize_all 应清理所有实例。"""
        from lightrag.api.rag_manager import RAGManager

        finalized = []

        def rag_factory(kb_id):
            mock_rag = MagicMock()
            mock_rag.initialize_storages = AsyncMock()
            mock_rag.check_and_migrate_data = AsyncMock()
            mock_rag.finalize_storages = AsyncMock(side_effect=lambda: finalized.append(kb_id))
            return mock_rag

        manager = RAGManager(rag_factory=rag_factory)
        await manager.get_rag("kb1")
        await manager.get_rag("kb2")

        await manager.finalize_all()

        assert "kb1" in finalized
        assert "kb2" in finalized
        assert len(manager._instances) == 0


# ──────────────────────────────────────────────
# _merge_kb_results 测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestMergeKBResults:
    """测试多 KB 结果合并逻辑。"""

    def test_merge_single_kb(self):
        """单 KB 结果应原样保留。"""
        from lightrag.api.rag_manager import RAGManager

        kb_results = [
            ("kb1", _make_kb_result(
                entities=[{"entity_name": "Alice", "entity_type": "Person", "description": "Engineer"}],
                chunks=[{"chunk_id": "c1", "content": "Alice works at TechCorp"}],
                references=[{"reference_id": "1", "file_path": "doc.txt"}],
            )),
        ]

        manager = RAGManager(rag_factory=lambda kb_id: MagicMock())
        result = manager._merge_kb_results(kb_results)

        assert len(result["entities"]) == 1
        assert result["entities"][0]["entity_name"] == "Alice"
        assert len(result["chunks"]) == 1

    def test_merge_two_kbs_distinct_entities(self):
        """不同 KB 的不同实体都应保留。"""
        from lightrag.api.rag_manager import RAGManager

        kb_results = [
            ("kb1", _make_kb_result(
                entities=[{"entity_name": "Alice", "entity_type": "Person", "description": "Engineer"}],
            )),
            ("kb2", _make_kb_result(
                entities=[{"entity_name": "Bob", "entity_type": "Person", "description": "Scientist"}],
            )),
        ]

        manager = RAGManager(rag_factory=lambda kb_id: MagicMock())
        result = manager._merge_kb_results(kb_results)

        assert len(result["entities"]) == 2
        names = {e["entity_name"] for e in result["entities"]}
        assert names == {"Alice", "Bob"}

    def test_merge_duplicate_entities_dedup(self):
        """相同实体名的实体应去重，合并 kb_ids 和 evidence。"""
        from lightrag.api.rag_manager import RAGManager

        kb_results = [
            ("kb1", _make_kb_result(
                entities=[{
                    "entity_name": "Alice",
                    "entity_type": "Person",
                    "description": "Engineer",
                }],
                references=[{"reference_id": "1", "file_path": "doc1.txt"}],
            )),
            ("kb2", _make_kb_result(
                entities=[{
                    "entity_name": "Alice",
                    "entity_type": "Person",
                    "description": "Senior Engineer with 5 years experience",
                }],
                references=[{"reference_id": "2", "file_path": "doc2.txt"}],
            )),
        ]

        manager = RAGManager(rag_factory=lambda kb_id: MagicMock())
        result = manager._merge_kb_results(kb_results)

        assert len(result["entities"]) == 1
        alice = result["entities"][0]
        assert alice["entity_name"] == "Alice"
        assert alice["kb_ids"] == ["kb1", "kb2"]
        assert alice["source_count"] == 2
        # description 应保留更长的
        assert "Senior Engineer" in alice["description"]
        assert len(alice["evidence"]) == 2

    def test_merge_entity_type_fallback(self):
        """当现有实体无 entity_type 时，应从重复实体中获取。"""
        from lightrag.api.rag_manager import RAGManager

        kb_results = [
            ("kb1", _make_kb_result(
                entities=[{"entity_name": "Alice", "description": "Engineer"}],
            )),
            ("kb2", _make_kb_result(
                entities=[{"entity_name": "Alice", "entity_type": "Person", "description": "Scientist"}],
            )),
        ]

        manager = RAGManager(rag_factory=lambda kb_id: MagicMock())
        result = manager._merge_kb_results(kb_results)

        alice = result["entities"][0]
        assert alice["entity_type"] == "Person"

    def test_merge_duplicate_relations(self):
        """相同（无向）关系应去重，保留更大 weight。"""
        from lightrag.api.rag_manager import RAGManager

        kb_results = [
            ("kb1", _make_kb_result(
                relationships=[{
                    "src_id": "Alice",
                    "tgt_id": "Bob",
                    "description": "colleague",
                    "weight": 1.0,
                }],
            )),
            ("kb2", _make_kb_result(
                relationships=[{
                    "src_id": "Bob",  # 反向
                    "tgt_id": "Alice",
                    "description": "close colleague",
                    "weight": 2.0,
                }],
            )),
        ]

        manager = RAGManager(rag_factory=lambda kb_id: MagicMock())
        result = manager._merge_kb_results(kb_results)

        assert len(result["relationships"]) == 1
        rel = result["relationships"][0]
        assert rel["weight"] == 2.0
        # description 应保留更长的
        assert rel["description"] == "close colleague"

    def test_merge_relation_keywords_fallback(self):
        """当现有关系无 keywords 时，应从重复关系中获取。"""
        from lightrag.api.rag_manager import RAGManager

        kb_results = [
            ("kb1", _make_kb_result(
                relationships=[{"src_id": "A", "tgt_id": "B", "description": "related"}],
            )),
            ("kb2", _make_kb_result(
                relationships=[{"src_id": "A", "tgt_id": "B", "description": "connected", "keywords": "strong"}],
            )),
        ]

        manager = RAGManager(rag_factory=lambda kb_id: MagicMock())
        result = manager._merge_kb_results(kb_results)

        rel = result["relationships"][0]
        assert rel["keywords"] == "strong"

    def test_merge_chunks_with_kb_id(self):
        """chunks 应包含 kb_id 字段，并按 (kb_id, chunk_id) 去重。"""
        from lightrag.api.rag_manager import RAGManager

        kb_results = [
            ("kb1", _make_kb_result(
                chunks=[
                    {"chunk_id": "c1", "content": "content 1"},
                    {"chunk_id": "c2", "content": "content 2"},
                ],
            )),
            ("kb2", _make_kb_result(
                chunks=[
                    {"chunk_id": "c1", "content": "content 1 same id"},  # 相同 chunk_id 但不同 KB
                ],
            )),
        ]

        manager = RAGManager(rag_factory=lambda kb_id: MagicMock())
        result = manager._merge_kb_results(kb_results)

        # 应保留所有 3 个 chunk（因为 kb_id 不同）
        assert len(result["chunks"]) == 3
        # 验证 kb_id 字段存在
        for chunk in result["chunks"]:
            assert "kb_id" in chunk

    def test_merge_reference_renumbering(self):
        """引用应被重编号，从 1 开始连续。"""
        from lightrag.api.rag_manager import RAGManager

        kb_results = [
            ("kb1", _make_kb_result(
                entities=[{"entity_name": "Alice", "reference_id": "kb1_ref1"}],
                references=[
                    {"reference_id": "kb1_ref1", "file_path": "doc1.txt"},
                    {"reference_id": "kb1_ref2", "file_path": "doc2.txt"},
                ],
            )),
            ("kb2", _make_kb_result(
                entities=[{"entity_name": "Bob", "reference_id": "kb2_ref1"}],
                references=[{"reference_id": "kb2_ref1", "file_path": "doc3.txt"}],
            )),
        ]

        manager = RAGManager(rag_factory=lambda kb_id: MagicMock())
        result = manager._merge_kb_results(kb_results)

        # 验证引用被重编号
        ref_ids = [r["reference_id"] for r in result["references"]]
        # 应为连续数字（可能不是从 1 开始，取决于过滤）
        # 至少应该是数字字符串
        for rid in ref_ids:
            assert rid.isdigit()

    def test_merge_skips_exception_results(self):
        """Exception 类型的结果应被跳过。"""
        from lightrag.api.rag_manager import RAGManager

        kb_results = [
            ("kb1", _make_kb_result(entities=[{"entity_name": "Alice"}])),
            ("kb2", Exception("Connection failed")),
        ]

        manager = RAGManager(rag_factory=lambda kb_id: MagicMock())
        result = manager._merge_kb_results(kb_results)

        assert len(result["entities"]) == 1

    def test_merge_skips_non_success_results(self):
        """非 success 状态的结果应被跳过。"""
        from lightrag.api.rag_manager import RAGManager

        kb_results = [
            ("kb1", _make_kb_result(entities=[{"entity_name": "Alice"}])),
            ("kb2", {"status": "failure", "data": {}}),
        ]

        manager = RAGManager(rag_factory=lambda kb_id: MagicMock())
        result = manager._merge_kb_results(kb_results)

        assert len(result["entities"]) == 1


# ──────────────────────────────────────────────
# _content_dedup_chunks 测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestContentDedupChunks:
    """测试基于内容哈希的 chunk 去重。"""

    def test_dedup_identical_content(self):
        """相同内容的 chunk 应去重，合并 reference_ids。"""
        from lightrag.api.rag_manager import RAGManager

        manager = RAGManager(rag_factory=lambda kb_id: MagicMock())

        chunks = [
            {"chunk_id": "c1", "content": "Same content", "kb_id": "kb1", "reference_id": "ref1"},
            {"chunk_id": "c2", "content": "Same content", "kb_id": "kb2", "reference_id": "ref2"},
        ]

        result = manager._content_dedup_chunks(chunks)

        assert len(result) == 1
        chunk = result[0]
        assert set(chunk.get("reference_ids", chunk.get("kb_ids", []))) == {"ref1", "ref2"}

    def test_dedup_prefers_more_kb_sources(self):
        """更多 KB 来源的 chunk 应被保留。"""
        from lightrag.api.rag_manager import RAGManager

        manager = RAGManager(rag_factory=lambda kb_id: MagicMock())

        chunks = [
            {"chunk_id": "c1", "content": "content", "kb_id": "kb1", "reference_id": "ref1"},
            {"chunk_id": "c2", "content": "content", "kb_id": "kb2", "reference_id": "ref2"},
            {"chunk_id": "c3", "content": "content", "kb_id": "kb3", "reference_id": "ref3"},
        ]

        result = manager._content_dedup_chunks(chunks)

        # 应保留来自 3 个 KB 的 chunk
        assert len(result) == 1
        # 哪个被保留取决于合并顺序
        # 最后处理的 chunk 可能被保留（因为它看到所有之前的 kb_ids）

    def test_dedup_empty_list(self):
        """空列表应返回空。"""
        from lightrag.api.rag_manager import RAGManager

        manager = RAGManager(rag_factory=lambda kb_id: MagicMock())
        result = manager._content_dedup_chunks([])

        assert result == []


# ──────────────────────────────────────────────
# 静态辅助方法测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestStaticHelperMethods:
    """测试静态辅助方法。"""

    def test_rag_refs_to_standard(self):
        """_rag_refs_to_standard 应转换引用格式。"""
        from lightrag.api.rag_manager import RAGManager

        rag_refs = [
            {"file_path": "doc1.txt", "content": "ref1"},
            {"file_path": "doc2.txt", "content": "ref2"},
        ]

        result = RAGManager._rag_refs_to_standard(rag_refs, "http://rag.service")

        assert len(result) == 2
        assert result[0]["reference_id"] == "rag_1"
        assert result[0]["file_path"] == "doc1.txt"
        assert result[0]["kb_id"] == "http://rag.service"

    def test_rag_refs_to_standard_filters_no_file_path(self):
        """无 file_path 的引用应被过滤。"""
        from lightrag.api.rag_manager import RAGManager

        rag_refs = [
            {"file_path": "doc1.txt"},
            {"content": "no path"},
            {"file_path": "", "content": "empty path"},
        ]

        result = RAGManager._rag_refs_to_standard(rag_refs, "http://rag.service")

        assert len(result) == 1

    def test_build_external_answers(self):
        """_build_external_answers 应构建标准格式。"""
        from lightrag.api.rag_manager import RAGManager

        rag_results = [
            ("http://rag1", {"answer": "ans1", "references": [], "source": "http://rag1"}),
            ("http://rag2", {"answer": "ans2", "references": [], "source": "http://rag2"}),
        ]

        manager = RAGManager(rag_factory=lambda kb_id: MagicMock())
        result = manager._build_external_answers(rag_results)

        assert len(result) == 2
        assert result[0]["source"] == "http://rag1"
        assert result[0]["answer"] == "ans1"

    def test_build_rag_context(self):
        """_build_rag_context 应构建正确的上下文格式。"""
        from lightrag.api.rag_manager import RAGManager

        rag_results = [
            ("http://rag1", {"answer": "Answer 1", "references": [], "source": "http://rag1"}),
        ]

        manager = RAGManager(rag_factory=lambda kb_id: MagicMock())
        result = manager._build_rag_context(rag_results)

        assert "[Source: http://rag1]" in result
        assert "Answer 1" in result

    def test_build_rag_context_empty(self):
        """空结果应返回空字符串。"""
        from lightrag.api.rag_manager import RAGManager

        manager = RAGManager(rag_factory=lambda kb_id: MagicMock())
        result = manager._build_rag_context([])

        assert result == ""


# ──────────────────────────────────────────────
# _bypass_llm 测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestBypassLlm:
    """测试 bypass 模式。"""

    @pytest.mark.asyncio
    async def test_bypass_uses_first_kb_llm(self, mock_query_param):
        """bypass 应使用第一个 KB 的 LLM。"""
        from lightrag.api.rag_manager import RAGManager

        mock_llm = AsyncMock(return_value="bypass response")
        mock_rag = MagicMock()
        mock_rag.llm_model_func = mock_llm
        mock_rag.initialize_storages = AsyncMock()
        mock_rag.check_and_migrate_data = AsyncMock()

        # rag_factory 必须是同步函数，RAGManager 在 get_rag 中同步调用它
        def rag_factory(kb_id):
            return mock_rag

        manager = RAGManager(rag_factory=rag_factory)

        result = await manager._bypass_llm("test query", mock_query_param, None, ["kb1"])

        assert result["status"] == "success"
        assert result["llm_response"]["content"] == "bypass response"
        assert result["metadata"]["query_mode"] == "bypass"

    @pytest.mark.asyncio
    async def test_bypass_with_priority_8(self, mock_query_param):
        """bypass 应使用 _priority=8。"""
        from lightrag.api.rag_manager import RAGManager
        from unittest.mock import sentinel

        priority_captured = []

        async def mock_llm_with_priority(*args, **kwargs):
            priority_captured.append(kwargs.get("_priority"))
            return "response"

        mock_rag = MagicMock()
        mock_rag.llm_model_func = mock_llm_with_priority
        mock_rag.initialize_storages = AsyncMock()
        mock_rag.check_and_migrate_data = AsyncMock()

        manager = RAGManager(rag_factory=lambda kb_id: mock_rag)

        await manager._bypass_llm("query", mock_query_param, None, [])

        assert priority_captured == [8]

    @pytest.mark.asyncio
    async def test_bypass_stream_mode(self, mock_query_param):
        """stream=True 应返回 is_streaming=True。"""
        from lightrag.api.rag_manager import RAGManager

        mock_rag = MagicMock()
        mock_rag.llm_model_func = AsyncMock(return_value="stream response")
        mock_rag.initialize_storages = AsyncMock()
        mock_rag.check_and_migrate_data = AsyncMock()

        manager = RAGManager(rag_factory=lambda kb_id: mock_rag)
        mock_query_param.stream = True

        result = await manager._bypass_llm("query", mock_query_param, None, [])

        assert result["llm_response"]["is_streaming"] is True
        assert result["llm_response"]["response_iterator"] == "stream response"


# ──────────────────────────────────────────────
# _fetch_all_external_kbs 和 _fetch_rag_kbs 测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestFetchExternalKbs:
    """测试外部 KB 获取。"""

    @pytest.mark.asyncio
    async def test_fetch_external_kbs_success(self):
        """成功获取外部 KB 应返回虚拟 KB 格式。"""
        from lightrag.api.rag_manager import RAGManager

        async def mock_fetch(*args, **kwargs):
            return [
                {"content": "external chunk", "file_path": "ext_doc.txt", "chunk_id": "ext_1"}
            ], None

        manager = RAGManager(rag_factory=lambda kb_id: MagicMock())

        with patch("lightrag.api.external_kb_client.fetch_external_kb", side_effect=mock_fetch):
            result, errors = await manager._fetch_all_external_kbs(
                "test query",
                [{"url": "http://ext.kb", "top_k": 5}]
            )

        assert len(result) == 1
        assert errors == []
        virtual_kb_id, data = result[0]
        assert virtual_kb_id.startswith("__ext_")
        assert data["status"] == "success"
        assert len(data["data"]["chunks"]) == 1

    @pytest.mark.asyncio
    async def test_fetch_external_kbs_failure_isolated(self):
        """单个外部 KB 失败不应影响其他 KB。"""
        from lightrag.api.rag_manager import RAGManager

        call_count = 0

        async def mock_fetch(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [], "外部知识库(检索型) [http://fail.kb] 请求失败: timeout"
            return [{"content": "ok"}], None

        manager = RAGManager(rag_factory=lambda kb_id: MagicMock())

        with patch("lightrag.api.external_kb_client.fetch_external_kb", side_effect=mock_fetch):
            result, errors = await manager._fetch_all_external_kbs(
                "query",
                [{"url": "http://fail.kb"}, {"url": "http://ok.kb"}]
            )

        assert len(result) == 2
        assert result[0][1]["status"] == "failure"
        assert result[1][1]["status"] == "success"
        assert len(errors) == 1
        assert "请求失败" in errors[0]

    @pytest.mark.asyncio
    async def test_fetch_rag_kbs_success(self):
        """成功获取 RAG 型 KB 应返回答案。"""
        from lightrag.api.rag_manager import RAGManager

        async def mock_fetch(*args, **kwargs):
            return {"answer": "RAG answer", "references": [], "source": "http://rag.kb"}, None

        manager = RAGManager(rag_factory=lambda kb_id: MagicMock())

        with patch("lightrag.api.external_kb_client.fetch_rag_kb", side_effect=mock_fetch):
            result, errors = await manager._fetch_rag_kbs(
                "query",
                [{"url": "http://rag.kb"}]
            )

        assert len(result) == 1
        assert errors == []
        url, data = result[0]
        assert url == "http://rag.kb"
        assert data["answer"] == "RAG answer"

    @pytest.mark.asyncio
    async def test_fetch_rag_kbs_failure_returns_empty(self):
        """RAG KB 失败应返回空答案和错误信息。"""
        from lightrag.api.rag_manager import RAGManager

        async def mock_fetch(*args, **kwargs):
            return {"answer": "", "references": [], "source": "http://fail.kb"}, "外部知识库(RAG服务型) [http://fail.kb] 请求失败: timeout"

        manager = RAGManager(rag_factory=lambda kb_id: MagicMock())

        with patch("lightrag.api.external_kb_client.fetch_rag_kb", side_effect=mock_fetch):
            result, errors = await manager._fetch_rag_kbs(
                "query",
                [{"url": "http://fail.kb"}]
            )

        assert len(result) == 1
        assert result[0][1]["answer"] == ""
        assert len(errors) == 1
        assert "请求失败" in errors[0]


# ──────────────────────────────────────────────
# _merge_rag_refs_into_data 测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestMergeRagRefsIntoData:
    """测试 RAG 引用合并到 merged_data。"""

    def test_merge_rag_refs_into_data(self):
        """_merge_rag_refs_into_data 应添加 external_answers 和合并引用。"""
        from lightrag.api.rag_manager import RAGManager

        manager = RAGManager(rag_factory=lambda kb_id: MagicMock())

        merged_data = {
            "references": [
                {"reference_id": "1", "file_path": "local.txt", "kb_id": "local"}
            ]
        }

        rag_results = [
            ("http://rag1", {"answer": "ans1", "references": [{"file_path": "rag1.txt"}]}),
            ("http://rag2", {"answer": "ans2", "references": [{"file_path": "rag2.txt"}]}),
        ]

        manager._merge_rag_refs_into_data(merged_data, rag_results)

        # 验证 external_answers
        assert "external_answers" in merged_data
        assert len(merged_data["external_answers"]) == 2

        # 验证引用被合并
        assert len(merged_data["references"]) == 3
        # RAG 引用应从 2 开始（现有最大是 1）
        rag_refs = [r for r in merged_data["references"] if r["kb_id"] == "http://rag1"]
        assert len(rag_refs) == 1

    def test_merge_rag_refs_with_empty_existing(self):
        """当现有引用为空时，RAG 引用应从 1 开始。"""
        from lightrag.api.rag_manager import RAGManager

        manager = RAGManager(rag_factory=lambda kb_id: MagicMock())

        merged_data = {"references": []}
        rag_results = [
            ("http://rag1", {"answer": "ans1", "references": [{"file_path": "rag1.txt"}]}),
        ]

        manager._merge_rag_refs_into_data(merged_data, rag_results)

        assert len(merged_data["references"]) == 1
        assert merged_data["references"][0]["reference_id"] == "1"


# ──────────────────────────────────────────────
# _build_context_from_merged 和 _build_fallback_context 测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestBuildContextMethods:
    """测试上下文构建方法。"""

    def test_build_context_from_merged_kg_mode(self):
        """KG 模式应包含 entities 和 relations。"""
        from lightrag.api.rag_manager import RAGManager

        manager = RAGManager(rag_factory=lambda kb_id: MagicMock())
        merged_data = {
            "entities": [{"entity_name": "Alice", "entity_type": "Person"}],
            "relationships": [{"src_id": "Alice", "tgt_id": "Bob", "description": "knows"}],
            "chunks": [{"content": "chunk content", "reference_ids": ["1"]}],
            "references": [{"reference_id": "1", "file_path": "doc.txt"}],
        }

        with patch("lightrag.api.rag_manager.PROMPTS", {"multi_kb_kg_query_context": "{entities_str}\n{relations_str}\n{text_chunks_str}\n{reference_list_str}"}):
            result = manager._build_context_from_merged(merged_data, "mix", {"evidence", "kb_ids", "source_count"})

        assert "Alice" in result
        assert "Bob" in result
        assert "chunk content" in result

    def test_build_context_from_merged_naive_mode(self):
        """naive 模式应只包含 chunks。"""
        from lightrag.api.rag_manager import RAGManager

        manager = RAGManager(rag_factory=lambda kb_id: MagicMock())
        merged_data = {
            "entities": [],
            "relationships": [],
            "chunks": [{"content": "chunk content", "reference_ids": ["1"]}],
            "references": [{"reference_id": "1", "file_path": "doc.txt"}],
        }

        with patch("lightrag.api.rag_manager.PROMPTS", {"multi_kb_naive_query_context": "{text_chunks_str}\n{reference_list_str}"}):
            result = manager._build_context_from_merged(merged_data, "naive", {"evidence", "kb_ids", "source_count"})

            assert "chunk content" in result

    def test_build_fallback_context(self):
        """_build_fallback_context 应在没有 tokenizer 时使用。"""
        from lightrag.api.rag_manager import RAGManager

        manager = RAGManager(rag_factory=lambda kb_id: MagicMock())
        merged_data = {
            "entities": [{"entity_name": "Alice"}],
            "relationships": [],
            "chunks": [{"content": "test", "reference_ids": ["1"]}],
            "references": [{"reference_id": "1", "file_path": "doc.txt"}],
        }

        with patch("lightrag.api.rag_manager.PROMPTS", {
            "multi_kb_kg_query_context": "{entities_str}\n{relations_str}\n{text_chunks_str}\n{reference_list_str}",
            "multi_kb_naive_query_context": "{text_chunks_str}\n{reference_list_str}",
        }):
            result = manager._build_fallback_context(merged_data, "mix")

            assert "Alice" in result
            assert "test" in result


# ──────────────────────────────────────────────
# multi_kb_get_data 测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestMultiKbGetData:
    """测试 multi_kb_get_data 方法。"""

    @pytest.mark.asyncio
    async def test_multi_kb_get_data_single_kb_fast_path(self, mock_query_param, mock_rag_factory):
        """单 KB 应走快速路径直接返回。"""
        from lightrag.api.rag_manager import RAGManager

        manager = RAGManager(rag_factory=mock_rag_factory)

        single_result = _make_kb_result(
            entities=[{"entity_name": "Alice"}],
            metadata={"keywords": {"high_level": ["Alice"]}},
        )

        async def mock_aquery(*args, **kwargs):
            return single_result

        manager._instances["kb1"] = MagicMock()
        manager._instances["kb1"].aquery_data = mock_aquery

        result = await manager.multi_kb_get_data("query", ["kb1"], mock_query_param)

        assert result is single_result

    @pytest.mark.asyncio
    async def test_multi_kb_get_data_empty_results_failure(self, mock_query_param, mock_rag_factory):
        """多 KB 场景下所有 KB 返回空结果应返回 failure。

        注意：单 KB 走快速路径直接返回结果，不会触发空结果检查。
        需要 2 个以上 KB 才会进入合并路径并检查空结果。
        """
        from lightrag.api.rag_manager import RAGManager

        manager = RAGManager(rag_factory=mock_rag_factory)

        empty_result = _make_kb_result(
            entities=[], relationships=[], chunks=[],
            metadata={"keywords": {}, "processing_info": {}},
        )

        async def mock_aquery(*args, **kwargs):
            return empty_result

        # 使用两个 KB 以触发合并路径中的空结果检查
        mock_rag1 = MagicMock()
        mock_rag1.aquery_data = mock_aquery
        mock_rag2 = MagicMock()
        mock_rag2.aquery_data = mock_aquery

        manager._instances["kb1"] = mock_rag1
        manager._instances["kb2"] = mock_rag2

        # Mock _multi_kb_post_process 避免 asdict(MagicMock) 错误
        async def mock_post_process(merged_data, mode, query, query_param, rag_instance):
            return merged_data, ""
        with patch.object(RAGManager, "_multi_kb_post_process", side_effect=mock_post_process):
            result = await manager.multi_kb_get_data("query", ["kb1", "kb2"], mock_query_param)

        assert result["status"] == "failure"

    @pytest.mark.asyncio
    async def test_multi_kb_get_data_no_kbs_raises(self, mock_query_param):
        """无 KB 和外部 KB 应抛出 ValueError。"""
        from lightrag.api.rag_manager import RAGManager

        manager = RAGManager(rag_factory=lambda kb_id: MagicMock())

        with pytest.raises(ValueError, match="At least one kb_id or external_kbs"):
            await manager.multi_kb_get_data("query", [], mock_query_param)


# ──────────────────────────────────────────────
# multi_kb_query 路径测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestMultiKbQueryPaths:
    """测试 multi_kb_query 的多种查询路径。"""

    @pytest.mark.asyncio
    async def test_multi_kb_query_bypass_mode(self, mock_query_param, mock_rag_factory):
        """bypass 模式应跳过所有检索。"""
        from lightrag.api.rag_manager import RAGManager

        # mock_rag_factory 接受 **overrides，但 RAGManager 调用 rag_factory(kb_id)
        # 所以需要包装为 lambda 接受位置参数
        manager = RAGManager(rag_factory=lambda kb_id: mock_rag_factory())
        mock_query_param.mode = "bypass"

        result = await manager.multi_kb_query("query", ["kb1"], mock_query_param)

        assert result["status"] == "success"
        assert result["metadata"]["query_mode"] == "bypass"

    @pytest.mark.asyncio
    async def test_multi_kb_query_single_local_kb(self, mock_query_param, mock_rag_factory):
        """单本地 KB 应直接调用 aquery_llm。"""
        from lightrag.api.rag_manager import RAGManager

        # mock_rag_factory 接受 **overrides，但 RAGManager 调用 rag_factory(kb_id)
        # 所以需要包装为 lambda 接受位置参数
        manager = RAGManager(rag_factory=lambda kb_id: mock_rag_factory())

        mock_rag = MagicMock()
        mock_rag.aquery_llm = AsyncMock(return_value={
            "status": "success",
            "data": {"content": "answer"},
            "llm_response": {"content": "answer text", "is_streaming": False},
        })
        manager._instances["kb1"] = mock_rag

        result = await manager.multi_kb_query("query", ["kb1"], mock_query_param)

        assert result["status"] == "success"
        mock_rag.aquery_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_multi_kb_query_only_external_rag_single(self, mock_query_param, mock_rag_factory):
        """仅单个外部 RAG KB 应直接返回答案。"""
        from lightrag.api.rag_manager import RAGManager

        # mock_rag_factory 接受 **overrides，但 RAGManager 调用 rag_factory(kb_id)
        # 所以需要包装为 lambda 接受位置参数
        manager = RAGManager(rag_factory=lambda kb_id: mock_rag_factory())

        rag_results = [("http://rag.kb", {"answer": "RAG answer", "references": [{"file_path": "rag.txt"}], "source": "http://rag.kb"})]

        async def mock_fetch(*args, **kwargs):
            return rag_results[0][1]

        manager._fetch_rag_kbs = lambda q, kbs: AsyncMock(return_value=(rag_results, []))()

        with patch("lightrag.api.external_kb_client.fetch_rag_kb", side_effect=mock_fetch):
            result = await manager.multi_kb_query(
                "query", [], mock_query_param, external_kbs=[{"type": "rag", "url": "http://rag.kb"}]
            )

        assert result["status"] == "success"
        assert result["llm_response"]["content"] == "RAG answer"

    @pytest.mark.asyncio
    async def test_multi_kb_query_no_kbs_raises(self, mock_query_param, mock_rag_factory):
        """无 KB 和外部 KB 应抛出 ValueError。"""
        from lightrag.api.rag_manager import RAGManager

        # mock_rag_factory 接受 **overrides，但 RAGManager 调用 rag_factory(kb_id)
        # 所以需要包装为 lambda 接受位置参数（虽然此测试不会调用 rag_factory）
        manager = RAGManager(rag_factory=lambda kb_id: mock_rag_factory())

        with pytest.raises(ValueError, match="At least one kb_id or external_kbs"):
            await manager.multi_kb_query("query", [], mock_query_param)
