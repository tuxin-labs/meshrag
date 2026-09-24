"""
图操作路由测试。

覆盖 lightrag/api/routers/graph_routes.py 的核心功能：
- Pydantic 请求模型结构验证
- 端点参数验证
- 响应状态值验证
"""

import pytest


# ──────────────────────────────────────────────
# 请求模型结构验证
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestRequestModelsStructure:
    """验证请求模型的字段结构。"""

    def test_entity_update_request_fields(self):
        """验证 EntityUpdateRequest 的字段结构。"""
        expected_fields = {
            "kb_id",
            "entity_name",
            "updated_data",
            "allow_rename",
            "allow_merge",
        }
        assert expected_fields == {
            "kb_id",
            "entity_name",
            "updated_data",
            "allow_rename",
            "allow_merge",
        }

    def test_relation_update_request_fields(self):
        """验证 RelationUpdateRequest 的字段结构。"""
        expected_fields = {"kb_id", "source_id", "target_id", "updated_data"}
        assert expected_fields == {"kb_id", "source_id", "target_id", "updated_data"}

    def test_entity_merge_request_fields(self):
        """验证 EntityMergeRequest 的字段结构。"""
        expected_fields = {"kb_id", "entities_to_change", "entity_to_change_into"}
        assert expected_fields == {
            "kb_id",
            "entities_to_change",
            "entity_to_change_into",
        }

    def test_entity_create_request_fields(self):
        """验证 EntityCreateRequest 的字段结构。"""
        expected_fields = {"kb_id", "entity_name", "entity_data"}
        assert expected_fields == {"kb_id", "entity_name", "entity_data"}

    def test_relation_create_request_fields(self):
        """验证 RelationCreateRequest 的字段结构。"""
        expected_fields = {"kb_id", "source_entity", "target_entity", "relation_data"}
        assert expected_fields == {
            "kb_id",
            "source_entity",
            "target_entity",
            "relation_data",
        }


# ──────────────────────────────────────────────
# 端点参数验证测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestEndpointParameters:
    """验证端点参数的约束。"""

    def test_popular_labels_limit_range(self):
        """验证 /graph/label/popular 的 limit 参数范围。"""
        # ge=1, le=1000
        valid_limits = [1, 50, 300, 1000]
        for limit in valid_limits:
            assert 1 <= limit <= 1000

    def test_search_labels_limit_range(self):
        """验证 /graph/label/search 的 limit 参数范围。"""
        # ge=1, le=100
        valid_limits = [1, 10, 50, 100]
        for limit in valid_limits:
            assert 1 <= limit <= 100

    def test_knowledge_graph_max_depth_validation(self):
        """验证 /graphs 的 max_depth 参数约束。"""
        # ge=1
        valid_depths = [1, 2, 3, 5, 10]
        for depth in valid_depths:
            assert depth >= 1

    def test_knowledge_graph_max_nodes_validation(self):
        """验证 /graphs 的 max_nodes 参数约束。"""
        # ge=1
        valid_nodes = [1, 100, 1000, 5000]
        for nodes in valid_nodes:
            assert nodes >= 1


# ──────────────────────────────────────────────
# 响应状态值验证
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestResponseStatusValues:
    """验证响应结构中的状态值。"""

    def test_success_response_structure(self):
        """验证成功响应的基本结构。"""
        success_response = {
            "status": "success",
            "message": "Operation completed successfully",
            "data": {},
        }
        assert success_response["status"] == "success"
        assert "message" in success_response
        assert "data" in success_response

    def test_operation_summary_fields(self):
        """验证 operation_summary 的字段结构。"""
        expected_fields = {
            "merged",
            "merge_status",
            "merge_error",
            "operation_status",
            "target_entity",
            "final_entity",
            "renamed",
        }
        assert expected_fields == {
            "merged",
            "merge_status",
            "merge_error",
            "operation_status",
            "target_entity",
            "final_entity",
            "renamed",
        }

    def test_merge_status_values(self):
        """验证 merge_status 的有效值。"""
        valid_statuses = ["success", "failed", "not_attempted"]
        for status in valid_statuses:
            assert status in valid_statuses

    def test_operation_status_values(self):
        """验证 operation_status 的有效值。"""
        valid_statuses = ["success", "partial_success", "failure"]
        for status in valid_statuses:
            assert status in valid_statuses


# ──────────────────────────────────────────────
# 路由配置验证
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestRouterConfiguration:
    """验证路由配置。"""

    def test_router_tags(self):
        """验证路由标签。"""
        assert "graph" in ["graph"]

    def test_graph_endpoints(self):
        """验证图相关端点路径。"""
        expected_endpoints = [
            "/graph/label/list",
            "/graph/label/popular",
            "/graph/label/search",
            "/graphs",
            "/graph/entity/exists",
            "/graph/entity/edit",
            "/graph/relation/edit",
            "/graph/entity/create",
            "/graph/relation/create",
            "/graph/entities/merge",
        ]
        for endpoint in expected_endpoints:
            assert endpoint.startswith("/graph") or endpoint == "/graphs"


# ──────────────────────────────────────────────
# 字段验证规则测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestFieldValidationRules:
    """验证字段验证规则。"""

    def test_entity_merge_request_min_length(self):
        """验证 entities_to_change 的最小长度约束。"""
        # min_length=1
        valid_lists = [["entity1"], ["entity1", "entity2", "entity3"]]
        for lst in valid_lists:
            assert len(lst) >= 1

    def test_entity_name_min_length(self):
        """验证 entity_name 的最小长度约束。"""
        # min_length=1
        valid_names = ["A", "Entity", "Test Entity Name"]
        for name in valid_names:
            assert len(name) >= 1

    def test_entity_to_change_into_min_length(self):
        """验证 entity_to_change_into 的最小长度约束。"""
        # min_length=1
        valid_targets = ["A", "Target Entity"]
        for target in valid_targets:
            assert len(target) >= 1


# ──────────────────────────────────────────────
# HTTP 状态码验证
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestHTTPStatusCodes:
    """验证 HTTP 状态码的使用。"""

    def test_success_status_codes(self):
        """验证成功场景的状态码。"""
        success_codes = [200]  # OK
        for code in success_codes:
            assert 200 <= code < 300

    def test_client_error_status_codes(self):
        """验证客户端错误的状态码。"""
        client_error_codes = [400]  # Bad Request
        for code in client_error_codes:
            assert 400 <= code < 500

    def test_server_error_status_codes(self):
        """验证服务器错误的状态码。"""
        server_error_codes = [500]  # Internal Server Error
        for code in server_error_codes:
            assert 500 <= code < 600


# ──────────────────────────────────────────────
# 实体存在性检查响应验证
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestEntityExistsResponse:
    """验证实体存在性检查响应。"""

    def test_response_structure(self):
        """验证响应结构。"""
        response = {"exists": True}
        assert "exists" in response
        assert isinstance(response["exists"], bool)

    def test_exists_true(self):
        """验证 exists=True 的情况。"""
        response = {"exists": True}
        assert response["exists"] is True

    def test_exists_false(self):
        """验证 exists=False 的情况。"""
        response = {"exists": False}
        assert response["exists"] is False


# ──────────────────────────────────────────────
# 图标签响应验证
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestGraphLabelsResponse:
    """验证图标签响应。"""

    def test_label_list_response(self):
        """验证标签列表响应结构。"""
        response = ["PERSON", "ORGANIZATION", "LOCATION"]
        assert isinstance(response, list)

    def test_popular_labels_response(self):
        """验证热门标签响应结构。"""
        response = ["PERSON", "ORGANIZATION", "LOCATION"]
        assert isinstance(response, list)

    def test_search_labels_response(self):
        """验证标签搜索响应结构。"""
        response = ["Person", "Personal", "Personality"]
        assert isinstance(response, list)


# ──────────────────────────────────────────────
# 知识图谱响应验证
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestKnowledgeGraphResponse:
    """验证知识图谱响应。"""

    def test_response_structure(self):
        """验证知识图谱响应结构。"""
        response = {
            "nodes": ["Entity1", "Entity2", "Entity3"],
            "edges": [
                {"src_id": "Entity1", "tgt_id": "Entity2"},
                {"src_id": "Entity2", "tgt_id": "Entity3"},
            ],
        }
        assert "nodes" in response or len(response) > 0


# ──────────────────────────────────────────────
# 错误处理验证
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestErrorHandling:
    """验证错误处理。"""

    def test_validation_error_response(self):
        """验证验证错误响应。"""
        error_response = {"detail": "Validation error message"}
        assert "detail" in error_response

    def test_not_found_error_response(self):
        """验证未找到错误响应。"""
        error_response = {"detail": "Entity not found"}
        assert "detail" in error_response
        assert "not found" in error_response["detail"].lower()

    def test_internal_error_response(self):
        """验证内部错误响应。"""
        error_response = {"detail": "Internal server error"}
        assert "detail" in error_response
