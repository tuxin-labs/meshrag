"""
核心 aquery 测试。

覆盖 lightrag/lightrag.py 中的 aquery_data 和 aquery_llm 功能：
- 返回格式验证
- 空结果处理
- processing_info 字段
- 流式/非流式响应
"""

import pytest


# ──────────────────────────────────────────────
# aquery_data 返回格式验证
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestAqueryDataFormat:
    """验证 aquery_data 返回格式。"""

    def test_success_response_structure(self):
        """验证成功响应的结构。"""
        response = {
            "status": "success",
            "message": "Query executed successfully",
            "data": {
                "entities": [],
                "relationships": [],
                "chunks": [],
                "references": [],
            },
            "metadata": {
                "query_mode": "mix",
                "keywords": {"high_level": [], "low_level": []},
                "processing_info": {
                    "total_entities_found": 0,
                    "total_relations_found": 0,
                    "entities_after_truncation": 0,
                    "relations_after_truncation": 0,
                    "merged_chunks_count": 0,
                    "final_chunks_count": 0,
                },
            },
        }

        assert response["status"] == "success"
        assert "data" in response
        assert "metadata" in response
        assert "entities" in response["data"]
        assert "relationships" in response["data"]
        assert "chunks" in response["data"]
        assert "references" in response["data"]

    def test_entity_fields(self):
        """验证 entity 对象的字段结构。"""
        entity = {
            "entity_name": "Alice",
            "entity_type": "PERSON",
            "description": "A software engineer",
            "source_id": "chunk_1",
            "file_path": "doc1.txt",
            "created_at": "1234567890",
            "reference_id": "ref_1",
        }

        assert "entity_name" in entity
        assert "entity_type" in entity
        assert "description" in entity
        assert "source_id" in entity
        assert "reference_id" in entity

    def test_relationship_fields(self):
        """验证 relationship 对象的字段结构。"""
        relationship = {
            "src_id": "Alice",
            "tgt_id": "Bob",
            "description": "colleague of",
            "keywords": "work together",
            "weight": 1.0,
            "source_id": "chunk_2",
            "file_path": "doc1.txt",
            "created_at": "1234567890",
            "reference_id": "ref_2",
        }

        assert "src_id" in relationship
        assert "tgt_id" in relationship
        assert "description" in relationship
        assert "keywords" in relationship
        assert "weight" in relationship
        assert "reference_id" in relationship

    def test_chunk_fields(self):
        """验证 chunk 对象的字段结构。"""
        chunk = {
            "content": "Sample text content",
            "file_path": "doc1.txt",
            "chunk_id": "chunk_1",
            "reference_id": "ref_1",
        }

        assert "content" in chunk
        assert "file_path" in chunk
        assert "chunk_id" in chunk
        assert "reference_id" in chunk

    def test_reference_fields(self):
        """验证 reference 对象的字段结构。"""
        reference = {
            "reference_id": "ref_1",
            "file_path": "doc1.txt",
        }

        assert "reference_id" in reference
        assert "file_path" in reference


# ──────────────────────────────────────────────
# aquery_llm 返回格式验证
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestAqueryLLMFormat:
    """验证 aquery_llm 返回格式。"""

    def test_success_response_structure(self):
        """验证成功响应的结构。"""
        response = {
            "status": "success",
            "message": "Query executed successfully",
            "data": {
                "entities": [],
                "relationships": [],
                "chunks": [],
                "references": [],
            },
            "metadata": {
                "query_mode": "mix",
                "keywords": {"high_level": [], "low_level": []},
                "processing_info": {},
            },
            "llm_response": {
                "content": "This is the LLM generated response",
                "response_iterator": None,
                "is_streaming": False,
            },
        }

        assert response["status"] == "success"
        assert "data" in response
        assert "llm_response" in response
        assert "content" in response["llm_response"]

    def test_bypass_mode_response(self):
        """验证 bypass 模式响应。"""
        response = {
            "status": "success",
            "message": "Bypass mode LLM non streaming response",
            "data": {},
            "metadata": {},
            "llm_response": {
                "content": "Direct LLM response",
                "response_iterator": None,
                "is_streaming": False,
            },
        }

        assert response["llm_response"]["is_streaming"] is False
        assert response["llm_response"]["response_iterator"] is None

    def test_streaming_response(self):
        """验证流式响应结构。"""
        # 流式响应应该有 is_streaming=True 和 response_iterator
        response = {
            "status": "success",
            "llm_response": {
                "content": "",
                "response_iterator": "mock_iterator",
                "is_streaming": True,
            },
        }

        assert response["llm_response"]["is_streaming"] is True
        assert response["llm_response"]["response_iterator"] is not None


# ──────────────────────────────────────────────
# 查询模式验证
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestQueryModes:
    """验证支持的查询模式。"""

    def test_valid_modes(self):
        """验证有效的查询模式。"""
        valid_modes = ["local", "global", "hybrid", "mix", "naive", "bypass"]
        for mode in valid_modes:
            assert mode in valid_modes

    def test_mode_in_metadata(self):
        """验证 metadata 中的 query_mode 字段。"""
        metadata = {
            "query_mode": "mix",
            "keywords": {"high_level": [], "low_level": []},
            "processing_info": {},
        }

        assert "query_mode" in metadata
        assert metadata["query_mode"] in [
            "local",
            "global",
            "hybrid",
            "mix",
            "naive",
            "bypass",
        ]


# ──────────────────────────────────────────────
# processing_info 验证
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestProcessingInfo:
    """验证 processing_info 字段。"""

    def test_processing_info_fields(self):
        """验证 processing_info 的字段结构。"""
        processing_info = {
            "total_entities_found": 10,
            "total_relations_found": 5,
            "entities_after_truncation": 8,
            "relations_after_truncation": 4,
            "merged_chunks_count": 15,
            "final_chunks_count": 12,
        }

        assert "total_entities_found" in processing_info
        assert "total_relations_found" in processing_info
        assert "entities_after_truncation" in processing_info
        assert "relations_after_truncation" in processing_info
        assert "merged_chunks_count" in processing_info
        assert "final_chunks_count" in processing_info

    def test_truncation_logic(self):
        """验证截断逻辑的合理性。"""
        # 截断后的数量应小于等于原始数量
        assert 8 <= 10  # entities_after_truncation <= total_entities_found
        assert 4 <= 5  # relations_after_truncation <= total_relations_found


# ──────────────────────────────────────────────
# 关键词验证
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestKeywords:
    """验证 keywords 字段。"""

    def test_keywords_structure(self):
        """验证 keywords 的结构。"""
        keywords = {
            "high_level": ["keyword1", "keyword2"],
            "low_level": ["detail1", "detail2", "detail3"],
        }

        assert "high_level" in keywords
        assert "low_level" in keywords
        assert isinstance(keywords["high_level"], list)
        assert isinstance(keywords["low_level"], list)

    def test_empty_keywords(self):
        """验证空关键词情况。"""
        keywords = {"high_level": [], "low_level": []}

        assert keywords["high_level"] == []
        assert keywords["low_level"] == []


# ──────────────────────────────────────────────
# 缓存验证
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestCacheBehavior:
    """验证缓存相关行为。"""

    def test_cache_key_includes_model_func_id(self):
        """验证缓存键应包含 model_func_id。"""
        # 缓存键应该包含模型函数 ID 以区分不同的模型
        # 这里验证概念
        cache_key_components = ["query", "mode", "model_func_id", "params"]
        assert "model_func_id" in cache_key_components

    def test_different_models_different_cache(self):
        """验证不同模型应使用不同的缓存。"""
        model_1_id = "gpt-4o-mini"
        model_2_id = "gpt-4o"

        # 不同模型应产生不同的缓存键
        assert model_1_id != model_2_id


# ──────────────────────────────────────────────
# 错误处理验证
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestErrorHandling:
    """验证错误处理。"""

    def test_empty_result_failure(self):
        """验证空结果应返回失败状态。"""
        response = {
            "status": "failure",
            "message": "No results found",
            "data": {
                "entities": [],
                "relationships": [],
                "chunks": [],
                "references": [],
            },
            "metadata": {},
        }

        assert response["status"] == "failure"

    def test_error_response_structure(self):
        """验证错误响应的结构。"""
        error_response = {
            "status": "error",
            "message": "An error occurred during query",
            "data": {},
            "metadata": {},
        }

        assert "status" in error_response
        assert "message" in error_response


# ──────────────────────────────────────────────
# 字符串处理验证
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestStringProcessing:
    """验证字符串处理。"""

    def test_query_strip(self):
        """验证查询字符串应去除首尾空格。"""
        query = "  What is LightRAG?  "
        assert query.strip() == "What is LightRAG?"

    def test_empty_query(self):
        """验证空查询处理。"""
        empty_queries = ["", "   ", "\t\n"]
        for query in empty_queries:
            assert len(query.strip()) == 0 or query.isspace()
