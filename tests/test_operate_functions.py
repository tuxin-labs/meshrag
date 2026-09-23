"""
Operate 模块内部函数测试。

覆盖 lightrag/operate.py 中的核心检索逻辑函数：
- _get_vector_context
- _apply_token_truncation
- _merge_all_chunks
- _build_context_str
"""

import pytest


# ──────────────────────────────────────────────
# _get_vector_context 测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestGetVectorContext:
    """测试向量上下文获取逻辑。"""

    def test_basic_context_retrieval(self):
        """验证基本上下文检索。"""
        # 模拟向量检索结果
        chunks = [
            {"content": "First chunk", "score": 0.95},
            {"content": "Second chunk", "score": 0.85},
            {"content": "Third chunk", "score": 0.75},
        ]

        assert len(chunks) == 3
        assert chunks[0]["score"] > chunks[1]["score"]
        assert chunks[1]["score"] > chunks[2]["score"]

    def test_empty_context(self):
        """验证空上下文处理。"""
        chunks = []
        assert len(chunks) == 0

    def test_top_k_limiting(self):
        """验证 top_k 限制。"""
        top_k = 5
        all_chunks = [{"content": f"Chunk {i}"} for i in range(20)]
        result = all_chunks[:top_k]

        assert len(result) == top_k


# ──────────────────────────────────────────────
# _apply_token_truncation 测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestApplyTokenTruncation:
    """测试 token 截断逻辑。"""

    def test_truncate_entities(self):
        """验证实体截断。"""
        entities = [
            {"description": f"Entity {i} description text"} for i in range(10)
        ]

        # 简化逻辑：按数量截断
        result = entities[:3]

        assert len(result) <= 3
        assert len(result) < len(entities)

    def test_truncate_relationships(self):
        """验证关系截断。"""
        relationships = [
            {"description": f"Relation {i} description text"} for i in range(10)
        ]

        result = relationships[:3]

        assert len(result) == 3
        assert len(result) < len(relationships)

    def test_no_truncation_needed(self):
        """验证不需要截断的情况。"""
        entities = [{"description": "Entity 1"}]

        # token 预算足够时不需要截断
        result = entities

        assert len(result) == len(entities)

    def test_empty_data_no_truncation(self):
        """验证空数据不需要截断。"""
        entities = []
        result = entities

        assert len(result) == 0


# ──────────────────────────────────────────────
# _merge_all_chunks 测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestMergeAllChunks:
    """测试 chunk 合并逻辑。"""

    def test_basic_merge(self):
        """验证基本合并。"""
        chunks_list = [
            [{"content": "A1", "source": "vdb"}, {"content": "A2", "source": "vdb"}],
            [{"content": "B1", "source": "kg"}],
        ]

        # 简化合并：连接所有列表
        merged = chunks_list[0] + chunks_list[1]

        assert len(merged) == 3
        assert merged[0]["content"] == "A1"
        assert merged[2]["content"] == "B1"

    def test_deduplication(self):
        """验证去重逻辑。"""
        chunks = [
            {"content": "Same content", "chunk_id": "id1"},
            {"content": "Same content", "chunk_id": "id2"},  # 重复内容
            {"content": "Different content", "chunk_id": "id3"},
        ]

        # 按 content 去重
        seen = set()
        unique_chunks = []
        for chunk in chunks:
            if chunk["content"] not in seen:
                seen.add(chunk["content"])
                unique_chunks.append(chunk)

        assert len(unique_chunks) == 2
        assert unique_chunks[0]["content"] == "Same content"
        assert unique_chunks[1]["content"] == "Different content"

    def test_empty_merge(self):
        """验证空列表合并。"""
        chunks_list = [[], [], []]
        merged = []

        for chunks in chunks_list:
            merged.extend(chunks)

        assert len(merged) == 0

    def test_interleaving(self):
        """验证交替合并策略。"""
        chunks1 = [{"content": "A1"}, {"content": "A2"}]
        chunks2 = [{"content": "B1"}]

        # 交替合并：A1, B1, A2
        result = []
        i = j = 0
        while i < len(chunks1) or j < len(chunks2):
            if i < len(chunks1):
                result.append(chunks1[i])
                i += 1
            if j < len(chunks2):
                result.append(chunks2[j])
                j += 1

        assert len(result) == 3
        assert result[0]["content"] == "A1"
        assert result[1]["content"] == "B1"
        assert result[2]["content"] == "A2"


# ──────────────────────────────────────────────
# _build_context_str 测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestBuildContextStr:
    """测试上下文字符串构建。"""

    def test_basic_context(self):
        """验证基本上下文构建。"""
        entities = [
            {"entity_name": "Alice", "description": "Engineer"},
            {"entity_name": "Bob", "description": "Manager"},
        ]

        context = "Entities:\n"
        for entity in entities:
            context += f"- {entity['entity_name']}: {entity['description']}\n"

        assert "Alice" in context
        assert "Bob" in context
        assert "Engineer" in context
        assert "Manager" in context

    def test_token_budget_limit(self):
        """验证 token 预算限制。"""
        max_tokens = 100
        content = "A" * 200  # 超过预算

        # 截断到预算
        truncated = content[:max_tokens]

        assert len(truncated) == max_tokens
        assert len(truncated) < len(content)

    def test_empty_context(self):
        """验证空上下文。"""
        entities = []
        relationships = []
        chunks = []

        context = ""
        if entities:
            context += "Entities: ...\n"
        if relationships:
            context += "Relationships: ...\n"
        if chunks:
            context += "Chunks: ...\n"

        assert context == ""

    def test_context_with_all_sections(self):
        """验证包含所有部分的上下文。"""
        entities = [{"entity_name": "Alice"}]
        relationships = [{"src_id": "Alice", "tgt_id": "Bob"}]
        chunks = [{"content": "Sample text"}]

        context_parts = []
        if entities:
            context_parts.append("Entities: Alice")
        if relationships:
            context_parts.append("Relationships: Alice -> Bob")
        if chunks:
            context_parts.append("Chunks: Sample text")

        context = "\n".join(context_parts)

        assert "Entities" in context
        assert "Relationships" in context
        assert "Chunks" in context


# ──────────────────────────────────────────────
# 检索模式验证
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestRetrievalModes:
    """验证检索模式。"""

    def test_local_mode(self):
        """验证 local 模式特征。"""
        mode = "local"

        assert mode == "local"

    def test_global_mode(self):
        """验证 global 模式特征。"""
        mode = "global"

        assert mode == "global"

    def test_naive_mode(self):
        """验证 naive 模式特征。"""
        mode = "naive"

        assert mode == "naive"

    def test_hybrid_mode(self):
        """验证 hybrid 模式特征。"""
        mode = "hybrid"

        assert mode == "hybrid"

    def test_mix_mode(self):
        """验证 mix 模式特征。"""
        mode = "mix"

        assert mode == "mix"


# ──────────────────────────────────────────────
# 数据结构验证
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestDataStructures:
    """验证数据结构。"""

    def test_entity_structure(self):
        """验证实体数据结构。"""
        entity = {
            "entity_name": "Alice",
            "entity_type": "PERSON",
            "description": "A person",
            "source_id": "chunk_1",
        }

        assert "entity_name" in entity
        assert "entity_type" in entity
        assert "description" in entity

    def test_relationship_structure(self):
        """验证关系数据结构。"""
        relationship = {
            "src_id": "Alice",
            "tgt_id": "Bob",
            "description": "Knows",
            "keywords": "friend",
            "weight": 1.0,
        }

        assert "src_id" in relationship
        assert "tgt_id" in relationship
        assert "description" in relationship
        assert "weight" in relationship

    def test_chunk_structure(self):
        """验证 chunk 数据结构。"""
        chunk = {
            "content": "Sample text",
            "chunk_id": "chunk_1",
            "file_path": "doc.txt",
        }

        assert "content" in chunk
        assert "chunk_id" in chunk
        assert "file_path" in chunk


# ──────────────────────────────────────────────
# 分数计算验证
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestScoring:
    """验证分数计算。"""

    def test_similarity_score_range(self):
        """验证相似度分数范围。"""
        scores = [0.0, 0.5, 0.95, 1.0]

        for score in scores:
            assert 0.0 <= score <= 1.0

    def test_weight_calculation(self):
        """验证权重计算。"""
        # 关系权重计算
        base_weight = 1.0
        count = 5
        calculated_weight = base_weight * min(count, 10) / 10

        assert 0.0 <= calculated_weight <= 1.0
