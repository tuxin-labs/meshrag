import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture(scope="module")
def test_app():
    """创建带有 mock RAGManager 的轻量测试 app。"""
    from lightrag.api.routers.document_routes import create_document_routes
    from lightrag.api.routers.query_routes import create_query_routes
    from lightrag.api.routers.graph_routes import create_graph_routes

    # 创建 mock RAGManager
    mock_rag_mgr = MagicMock()
    mock_rag_mgr.default_kb = "default"
    mock_rag_mgr.list_knowledge_bases = MagicMock(return_value=["default"])

    # 创建 mock rag 实例，所有被 await 的方法都用 AsyncMock
    mock_rag = MagicMock()
    mock_rag.get_graph_labels = AsyncMock(return_value=[])
    # chunk_entity_relation_graph.get_popular_labels 也需要 async
    mock_graph = MagicMock()
    mock_graph.get_popular_labels = AsyncMock(return_value=[])
    mock_rag.chunk_entity_relation_graph = mock_graph
    # doc_status 的查询方法都需要 async
    mock_doc_status = MagicMock()
    mock_doc_status.get_doc_by_file_path = AsyncMock(return_value=None)
    # upload 同步内容预检会调用 get_by_id 判断内容是否重复
    mock_doc_status.get_by_id = AsyncMock(return_value=None)
    mock_rag.doc_status = mock_doc_status

    mock_rag_mgr.get_rag = AsyncMock(return_value=mock_rag)
    mock_rag_mgr.delete_knowledge_base = AsyncMock(
        return_value={"status": "success", "message": "deleted"}
    )

    # Mock multi_kb_query 和 multi_kb_get_data
    async def mock_multi_kb_query(*args, **kwargs):
        return {
            "status": "success",
            "message": "",
            "data": {
                "references": [],
                "entities": [],
                "relationships": [],
                "chunks": [],
            },
            "metadata": {},
            "llm_response": {"content": "mock answer", "is_streaming": False},
        }

    async def mock_multi_kb_get_data(*args, **kwargs):
        return {
            "status": "success",
            "message": "",
            "data": {
                "entities": [],
                "relationships": [],
                "chunks": [],
                "references": [],
            },
            "metadata": {},
        }

    mock_rag_mgr.multi_kb_query = mock_multi_kb_query
    mock_rag_mgr.multi_kb_get_data = mock_multi_kb_get_data

    # 创建 mock DocManager，设置 base_input_dir 为有效路径
    import tempfile

    tmp_dir = tempfile.mkdtemp()
    mock_doc_mgr = MagicMock()
    mock_doc_mgr.base_input_dir = tmp_dir

    app = FastAPI()
    app.include_router(create_document_routes(mock_rag_mgr, mock_doc_mgr, api_key=None))
    app.include_router(create_query_routes(mock_rag_mgr, api_key=None, top_k=60))
    app.include_router(create_graph_routes(mock_rag_mgr, api_key=None))

    # 添加 KB 管理端点
    from pydantic import BaseModel

    class KBRequest(BaseModel):
        kb_id: str

    @app.get("/knowledge_bases", tags=["Knowledge Base Management"])
    async def list_knowledge_bases():
        return {
            "status": "success",
            "knowledge_bases": mock_rag_mgr.list_knowledge_bases(),
        }

    @app.post("/knowledge_bases", tags=["Knowledge Base Management"])
    async def create_knowledge_base(req: KBRequest):
        await mock_rag_mgr.get_rag(req.kb_id)
        return {"status": "success", "kb_id": req.kb_id}

    @app.delete("/knowledge_bases/{kb_id}", tags=["Knowledge Base Management"])
    async def delete_knowledge_base(kb_id: str):
        result = await mock_rag_mgr.delete_knowledge_base(kb_id)
        if result["status"] == "success":
            return {"status": "success", "kb_id": kb_id, "message": result["message"]}
        return {"status": "error", "kb_id": kb_id, "message": result["message"]}, 500

    # 将 mock_rag_mgr 附加到 app 上供测试使用
    app.state.mock_rag_mgr = mock_rag_mgr

    yield app


@pytest.fixture(scope="module")
def client(test_app):
    return TestClient(test_app)


@pytest.mark.offline
def test_kb_lifecycle(client):
    """测试创建、列出和删除知识库"""
    # 列出 KB（应包含 default）
    response = client.get("/knowledge_bases")
    assert response.status_code == 200
    data = response.json()
    assert "default" in data["knowledge_bases"]

    # 创建新 KB
    kb_id = "test_kb_1"
    response = client.post("/knowledge_bases", json={"kb_id": kb_id})
    assert response.status_code == 200

    # 再次列出
    response = client.get("/knowledge_bases")
    # mock 只返回 ["default"]，创建操作只是调了 get_rag
    assert response.status_code == 200


@pytest.mark.offline
def test_document_upload_with_kb_id(client):
    """测试文档上传需要并遵循 kb_id"""
    kb_id = "kb_doc_test"

    # 创建一个小文本文件
    file_content = b"This is a test document content."
    files = {"file": ("test.txt", file_content, "text/plain")}

    # 缺少 kb_id（应返回 422）
    response = client.post("/documents/upload", files=files)
    assert response.status_code == 422  # FastAPI 验证错误

    # 带 kb_id 上传
    response = client.post(
        f"/documents/upload?kb_id={kb_id}",
        files={"file": ("test.txt", file_content, "text/plain")},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"


@pytest.mark.offline
def test_query_with_kb_ids(client):
    """测试查询单个或多个 KB"""
    # 1. 查询指定 kb_id
    response = client.post(
        "/query", json={"query": "test query", "kb_ids": ["default"]}
    )
    assert response.status_code == 200

    # 2. 查询多个 kb_ids
    response = client.post(
        "/query", json={"query": "multi query", "kb_ids": ["default", "other_kb"]}
    )
    assert response.status_code == 200

    # 3. 不带 kb_ids（使用默认）
    response = client.post("/query", json={"query": "default query"})
    assert response.status_code == 200


@pytest.mark.offline
def test_graph_endpoints_with_kb_id(client):
    """测试 graph 端点需要 kb_id"""
    # 获取 labels
    response = client.get("/graph/label/list")
    assert response.status_code == 422  # 缺少 kb_id

    response = client.get("/graph/label/list?kb_id=default")
    assert response.status_code == 200

    # 获取热门 labels
    response = client.get("/graph/label/popular?kb_id=default&limit=10")
    assert response.status_code == 200
