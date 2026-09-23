"""
Pytest configuration for LightRAG tests.

This file provides command-line options and fixtures for test configuration.
"""

import pytest
from fastapi import FastAPI


def pytest_configure(config):
    """Register custom markers for LightRAG tests."""
    config.addinivalue_line(
        "markers", "offline: marks tests as offline (no external dependencies)"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests requiring external services (skipped by default)",
    )
    config.addinivalue_line("markers", "requires_db: marks tests requiring database")
    config.addinivalue_line(
        "markers", "requires_api: marks tests requiring LightRAG API server"
    )


def pytest_addoption(parser):
    """Add custom command-line options for LightRAG tests."""

    parser.addoption(
        "--keep-artifacts",
        action="store_true",
        default=False,
        help="Keep test artifacts (temporary directories and files) after test completion for inspection",
    )

    parser.addoption(
        "--stress-test",
        action="store_true",
        default=False,
        help="Enable stress test mode with more intensive workloads",
    )

    parser.addoption(
        "--test-workers",
        action="store",
        default=3,
        type=int,
        help="Number of parallel workers for stress tests (default: 3)",
    )

    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require external services (database, API server, etc.)",
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip integration tests by default.

    Integration tests are skipped unless --run-integration flag is provided.
    This allows running offline tests quickly without needing external services.
    """
    if config.getoption("--run-integration"):
        # If --run-integration is specified, run all tests
        return

    skip_integration = pytest.mark.skip(
        reason="Requires external services(DB/API), use --run-integration to run"
    )

    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture(scope="session")
def keep_test_artifacts(request):
    """
    Fixture to determine whether to keep test artifacts.

    Priority: CLI option > Environment variable > Default (False)
    """
    import os

    # Check CLI option first
    if request.config.getoption("--keep-artifacts"):
        return True

    # Fall back to environment variable
    return os.getenv("LIGHTRAG_KEEP_ARTIFACTS", "false").lower() == "true"


@pytest.fixture(scope="session")
def stress_test_mode(request):
    """
    Fixture to determine whether stress test mode is enabled.

    Priority: CLI option > Environment variable > Default (False)
    """
    import os

    # Check CLI option first
    if request.config.getoption("--stress-test"):
        return True

    # Fall back to environment variable
    return os.getenv("LIGHTRAG_STRESS_TEST", "false").lower() == "true"


@pytest.fixture(scope="session")
def parallel_workers(request):
    """
    Fixture to determine the number of parallel workers for stress tests.

    Priority: CLI option > Environment variable > Default (3)
    """
    import os

    # Check CLI option first
    cli_workers = request.config.getoption("--test-workers")
    if cli_workers != 3:  # Non-default value provided
        return cli_workers

    # Fall back to environment variable
    return int(os.getenv("LIGHTRAG_TEST_WORKERS", "3"))


@pytest.fixture(scope="function")
def test_app():
    """提供用于测试的 FastAPI 应用实例。每次测试使用独立的 app。"""
    app = FastAPI()
    yield app


@pytest.fixture(scope="session")
def run_integration_tests(request):
    """
    Fixture to determine whether to run integration tests.

    Priority: CLI option > Environment variable > Default (False)
    """
    import os

    # Check CLI option first
    if request.config.getoption("--run-integration"):
        return True

    # Fall back to environment variable
    return os.getenv("LIGHTRAG_RUN_INTEGRATION", "false").lower() == "true"


# ============================
# 共享测试 Fixtures（Phase 1）
# ============================


class SharedDummyTokenizer:
    """简单字符到 token 的 1:1 映射，用于测试。"""

    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


@pytest.fixture()
def dummy_tokenizer():
    """返回一个 Tokenizer 实例，使用 DummyTokenizer。"""
    from lightrag.utils import Tokenizer

    return Tokenizer("dummy", SharedDummyTokenizer())


@pytest.fixture()
def mock_query_param():
    """返回一个默认配置的 QueryParam 实例。"""
    from lightrag.base import QueryParam

    return QueryParam(mode="mix", top_k=10, stream=False)


@pytest.fixture()
def mock_rag_factory():
    """返回一个创建 Mock LightRAG 实例的工厂函数。

    用法:
        rag = mock_rag_factory()
        result = await rag.aquery_data("test query", mock_query_param())
    """

    def _create_mock_rag(**overrides):
        from unittest.mock import AsyncMock, MagicMock

        rag = MagicMock()
        rag.initialize_storages = AsyncMock()
        rag.check_and_migrate_data = AsyncMock()
        rag.finalize_storages = AsyncMock()
        rag.drop_storages = AsyncMock()
        rag.llm_model_func = AsyncMock(return_value="mock llm response")

        # 默认的 aquery_data 返回值
        rag.aquery_data = AsyncMock(
            return_value={
                "status": "success",
                "message": "",
                "data": {
                    "entities": [],
                    "relationships": [],
                    "chunks": [],
                    "references": [],
                },
                "metadata": {"keywords": [], "processing_info": {}},
            }
        )

        # 默认的 aquery_llm 返回值
        rag.aquery_llm = AsyncMock(
            return_value={
                "status": "success",
                "data": {"content": "mock answer"},
                "metadata": {"keywords": [], "processing_info": {}},
                "llm_response": {"content": "mock answer"},
            }
        )

        rag.aquery = AsyncMock(return_value="mock query response")
        rag.ainsert = AsyncMock()
        rag.adelete_by_doc_id = AsyncMock()

        # tokenizer
        rag.tokenizer = Tokenizer("dummy", SharedDummyTokenizer())

        # 应用覆盖
        for key, value in overrides.items():
            setattr(rag, key, value)

        return rag

    from lightrag.utils import Tokenizer

    return _create_mock_rag


@pytest.fixture()
def sample_kb_results():
    """返回预构建的 KB 结果数据，用于 RAGManager 合并测试。

    返回格式: {"kb1": result_dict, "kb2": result_dict}
    """
    return {
        "kb1": {
            "status": "success",
            "message": "",
            "data": {
                "entities": [
                    {
                        "entity_name": "Alice",
                        "entity_type": "Person",
                        "description": "A software engineer at TechCorp",
                        "source_id": "chunk_1",
                        "evidence": [
                            {
                                "reference_id": "ref_1",
                                "content": "Alice works at TechCorp",
                                "kb_id": "kb1",
                            }
                        ],
                    },
                    {
                        "entity_name": "Bob",
                        "entity_type": "Person",
                        "description": "A data scientist",
                        "source_id": "chunk_2",
                        "evidence": [
                            {
                                "reference_id": "ref_2",
                                "content": "Bob is a data scientist",
                                "kb_id": "kb1",
                            }
                        ],
                    },
                ],
                "relationships": [
                    {
                        "src_id": "Alice",
                        "tgt_id": "Bob",
                        "description": "colleague of",
                        "keywords": "work together",
                        "weight": 1.0,
                        "source_id": "chunk_3",
                        "evidence": [
                            {
                                "reference_id": "ref_3",
                                "content": "Alice and Bob work together",
                                "kb_id": "kb1",
                            }
                        ],
                    }
                ],
                "chunks": [
                    {
                        "chunk_id": "chunk_1",
                        "content": "Alice works at TechCorp as a software engineer.",
                        "kb_id": "kb1",
                        "reference_ids": ["ref_1"],
                    },
                    {
                        "chunk_id": "chunk_2",
                        "content": "Bob is a data scientist at DataInc.",
                        "kb_id": "kb1",
                        "reference_ids": ["ref_2"],
                    },
                ],
                "references": [
                    {
                        "reference_id": "ref_1",
                        "content": "Alice works at TechCorp",
                        "file_path": "doc1.txt",
                    },
                    {
                        "reference_id": "ref_2",
                        "content": "Bob is a data scientist",
                        "file_path": "doc1.txt",
                    },
                    {
                        "reference_id": "ref_3",
                        "content": "Alice and Bob work together",
                        "file_path": "doc1.txt",
                    },
                ],
            },
            "metadata": {
                "keywords": ["Alice", "Bob"],
                "processing_info": {
                    "entities_count": 2,
                    "relationships_count": 1,
                    "chunks_count": 2,
                },
            },
        },
        "kb2": {
            "status": "success",
            "message": "",
            "data": {
                "entities": [
                    {
                        "entity_name": "Alice",
                        "entity_type": "Person",
                        "description": "A software engineer at TechCorp with 5 years experience",
                        "source_id": "chunk_10",
                        "evidence": [
                            {
                                "reference_id": "ref_10",
                                "content": "Alice has 5 years of experience",
                                "kb_id": "kb2",
                            }
                        ],
                    },
                    {
                        "entity_name": "Charlie",
                        "entity_type": "Person",
                        "description": "A project manager",
                        "source_id": "chunk_11",
                        "evidence": [
                            {
                                "reference_id": "ref_11",
                                "content": "Charlie manages the project",
                                "kb_id": "kb2",
                            }
                        ],
                    },
                ],
                "relationships": [
                    {
                        "src_id": "Alice",
                        "tgt_id": "Charlie",
                        "description": "reports to",
                        "keywords": "management",
                        "weight": 2.0,
                        "source_id": "chunk_12",
                        "evidence": [
                            {
                                "reference_id": "ref_12",
                                "content": "Alice reports to Charlie",
                                "kb_id": "kb2",
                            }
                        ],
                    }
                ],
                "chunks": [
                    {
                        "chunk_id": "chunk_10",
                        "content": "Alice has 5 years of experience at TechCorp.",
                        "kb_id": "kb2",
                        "reference_ids": ["ref_10"],
                    },
                    {
                        "chunk_id": "chunk_11",
                        "content": "Charlie manages the team project.",
                        "kb_id": "kb2",
                        "reference_ids": ["ref_11"],
                    },
                ],
                "references": [
                    {
                        "reference_id": "ref_10",
                        "content": "Alice has 5 years of experience",
                        "file_path": "doc2.txt",
                    },
                    {
                        "reference_id": "ref_11",
                        "content": "Charlie manages the project",
                        "file_path": "doc2.txt",
                    },
                    {
                        "reference_id": "ref_12",
                        "content": "Alice reports to Charlie",
                        "file_path": "doc2.txt",
                    },
                ],
            },
            "metadata": {
                "keywords": ["Alice", "Charlie"],
                "processing_info": {
                    "entities_count": 2,
                    "relationships_count": 1,
                    "chunks_count": 2,
                },
            },
        },
    }
