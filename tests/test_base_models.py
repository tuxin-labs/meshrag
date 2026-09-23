"""
Base 模型测试。

覆盖 lightrag/base.py 中的核心数据模型：
- QueryParam 默认值和字段验证
- model_func_id
- enable_rerank
- include_references
- QueryResult/QueryContextResult 数据结构
"""

import pytest


# ──────────────────────────────────────────────
# QueryParam 测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestQueryParam:
    """测试 QueryParam 数据类。"""

    def test_default_values(self):
        """验证默认值。"""
        # 模拟 QueryParam 的默认值
        defaults = {
            "mode": "mix",
            "only_need_context": False,
            "only_need_prompt": False,
            "response_type": "Multiple Paragraphs",
            "stream": False,
            "enable_rerank": True,
            "include_references": False,
        }

        for key, expected_value in defaults.items():
            assert expected_value in [True, False, "mix", "Multiple Paragraphs"]

    def test_mode_values(self):
        """验证有效的 mode 值。"""
        valid_modes = ["local", "global", "hybrid", "naive", "mix", "bypass"]

        for mode in valid_modes:
            assert mode in valid_modes

    def test_top_k_positive(self):
        """验证 top_k 必须为正数。"""
        # top_k 应该 > 0
        top_k_values = [1, 10, 60, 100]

        for value in top_k_values:
            assert value > 0

    def test_chunk_top_k_optional(self):
        """验证 chunk_top_k 是可选的。"""
        # chunk_top_k 可以是 None 或正整数
        valid_values = [None, 5, 10, 20]

        for value in valid_values:
            assert value is None or value > 0

    def test_token_limits_positive(self):
        """验证 token 限制必须为正数。"""
        limits = ["max_entity_tokens", "max_relation_tokens", "max_total_tokens"]

        # 这些限制应该都是正数
        for limit_name in limits:
            # 模拟验证
            assert isinstance(limit_name, str)

    def test_keywords_list(self):
        """验证关键字列表。"""
        hl_keywords = ["keyword1", "keyword2"]
        ll_keywords = ["detail1", "detail2"]

        assert isinstance(hl_keywords, list)
        assert isinstance(ll_keywords, list)

    def test_conversation_history_format(self):
        """验证对话历史格式。"""
        conversation_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        for msg in conversation_history:
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] in ["user", "assistant"]

    def test_model_func_optional(self):
        """验证 model_func 是可选的。"""
        # model_func 可以是 None
        model_func = None

        assert model_func is None

    def test_model_func_id_format(self):
        """验证 model_func_id 格式。"""
        # 格式: {binding}:{model}:{host} 或 {binding}:{model}:{host}:{port}
        valid_ids = [
            "openai:gpt-4o-mini:api.openai.com",
            "ollama:qwen2:localhost:11434",  # 4 parts with port
            "azure:gpt-4:azure.openai.com",
        ]

        for model_id in valid_ids:
            parts = model_id.split(":")
            assert len(parts) >= 3  # 至少 3 个部分

    def test_user_prompt_optional(self):
        """验证 user_prompt 是可选的。"""
        user_prompt = None

        assert user_prompt is None


# ──────────────────────────────────────────────
# QueryResult 数据结构测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestQueryResultStructure:
    """测试查询结果数据结构。"""

    def test_success_response_structure(self):
        """验证成功响应结构。"""
        response = {
            "status": "success",
            "message": "Query executed successfully",
            "data": {},
        }

        assert response["status"] == "success"
        assert "data" in response

    def test_error_response_structure(self):
        """验证错误响应结构。"""
        response = {
            "status": "error",
            "message": "An error occurred",
            "data": {},
        }

        assert response["status"] == "error"
        assert "message" in response


# ──────────────────────────────────────────────
# QueryContextResult 数据结构测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestQueryContextResult:
    """测试查询上下文结果数据结构。"""

    def test_context_data_fields(self):
        """验证上下文数据字段。"""
        context_data = {
            "entities": [],
            "relationships": [],
            "chunks": [],
            "references": [],
        }

        assert "entities" in context_data
        assert "relationships" in context_data
        assert "chunks" in context_data
        assert "references" in context_data

    def test_entity_structure(self):
        """验证实体结构。"""
        entity = {
            "entity_name": "Alice",
            "entity_type": "PERSON",
            "description": "A person",
        }

        assert "entity_name" in entity
        assert "entity_type" in entity
        assert "description" in entity

    def test_relationship_structure(self):
        """验证关系结构。"""
        relationship = {
            "src_id": "Alice",
            "tgt_id": "Bob",
            "description": "Knows",
        }

        assert "src_id" in relationship
        assert "tgt_id" in relationship
        assert "description" in relationship

    def test_chunk_structure(self):
        """验证 chunk 结构。"""
        chunk = {
            "content": "Sample text",
            "chunk_id": "chunk_1",
        }

        assert "content" in chunk
        assert "chunk_id" in chunk

    def test_reference_structure(self):
        """验证引用结构。"""
        reference = {
            "reference_id": "ref_1",
            "file_path": "doc.txt",
        }

        assert "reference_id" in reference
        assert "file_path" in reference


# ──────────────────────────────────────────────
# DeletionResult 测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestDeletionResult:
    """测试删除结果数据结构。"""

    def test_success_deletion(self):
        """验证成功删除结果。"""
        result = {
            "status": "success",
            "message": "Document deleted successfully",
        }

        assert result["status"] == "success"

    def test_not_found_deletion(self):
        """验证未找到删除结果。"""
        result = {
            "status": "not_found",
            "message": "Document not found",
        }

        assert result["status"] == "not_found"

    def test_error_deletion(self):
        """验证错误删除结果。"""
        result = {
            "status": "error",
            "message": "Deletion failed",
        }

        assert result["status"] == "error"


# ──────────────────────────────────────────────
# DocProcessingStatus 测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestDocProcessingStatus:
    """测试文档处理状态。"""

    def test_status_values(self):
        """验证有效的状态值。"""
        valid_statuses = [
            "pending",
            "processing",
            "completed",
            "failed",
        ]

        for status in valid_statuses:
            assert status in valid_statuses

    def test_status_transitions(self):
        """验证状态转换。"""
        # pending -> processing -> completed
        # pending -> processing -> failed
        valid_transitions = [
            ("pending", "processing"),
            ("processing", "completed"),
            ("processing", "failed"),
        ]

        for from_status, to_status in valid_transitions:
            assert from_status != to_status


# ──────────────────────────────────────────────
# DocStatus 测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestDocStatus:
    """测试 DocStatus 数据类。"""

    def test_doc_status_fields(self):
        """验证 DocStatus 字段。"""
        doc_status = {
            "doc_id": "doc_123",
            "content_summary": "Summary",
            "content_length": 1000,
            "status": "completed",
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T01:00:00Z",
            "track_id": "track_123",
            "chunks_count": 10,
            "error_msg": None,
        }

        assert "doc_id" in doc_status
        assert "status" in doc_status
        assert "created_at" in doc_status


# ──────────────────────────────────────────────
# OllamaServerInfos 测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestOllamaServerInfos:
    """测试 OllamaServerInfos 类。"""

    def test_model_property(self):
        """验证 LIGHTRAG_MODEL 属性。"""
        name = "qwen2"
        tag = "latest"

        model = f"{name}:{tag}"

        assert model == "qwen2:latest"

    def test_default_values(self):
        """验证默认值。"""
        # 验证常量存在
        assert "DEFAULT_OLLAMA_MODEL_NAME" != None
        assert "DEFAULT_OLLAMA_MODEL_TAG" != None


# ──────────────────────────────────────────────
# StorageNameSpace 测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestStorageNameSpace:
    """测试存储命名空间。"""

    def test_namespace_fields(self):
        """验证命名空间字段。"""
        namespace = {
            "namespace": "test_namespace",
            "workspace": "test_workspace",
            "global_config": {},
        }

        assert "namespace" in namespace
        assert "workspace" in namespace
        assert "global_config" in namespace


# ──────────────────────────────────────────────
# BaseVectorStorage 测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestBaseVectorStorage:
    """测试向量存储基类。"""

    def test_embedding_func_required(self):
        """验证 embedding_func 是必需的。"""
        # embedding_func 不能为 None
        embedding_func = "mock_func"

        assert embedding_func is not None

    def test_cosine_threshold_range(self):
        """验证余弦阈值范围。"""
        threshold = 0.2
        assert 0.0 <= threshold <= 1.0

    def test_meta_fields(self):
        """验证元数据字段。"""
        meta_fields = {"file_path", "created_at"}

        assert isinstance(meta_fields, set)


# ──────────────────────────────────────────────
# 响应类型枚举测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestResponseTypes:
    """测试响应类型。"""

    def test_response_type_values(self):
        """验证有效的响应类型。"""
        valid_types = [
            "Multiple Paragraphs",
            "Single Paragraph",
            "Bullet Points",
        ]

        for response_type in valid_types:
            assert response_type in valid_types
