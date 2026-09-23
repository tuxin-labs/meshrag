"""
PostgreSQL 连接池 acquire 超时测试

测试覆盖：
- pool.acquire(timeout=N) 参数正确传递
- acquire 超时触发重试机制
- 默认超时值和自定义超时值配置
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightrag.kg.postgres_impl import PostgreSQLDB


def make_mock_pool():
    """创建一个兼容 asyncpg.Pool 的 mock 对象。"""
    mock_pool = MagicMock()

    async def _async_noop():
        pass

    mock_pool.close = MagicMock(return_value=_async_noop())
    return mock_pool


def make_db_config(**overrides):
    """创建测试用 PostgreSQLDB 配置。"""
    config = {
        "host": "localhost",
        "port": 5432,
        "user": "test",
        "password": "test",
        "database": "test",
        "workspace": "test",
        "max_connections": 5,
        "enable_vector": False,
        "vector_index_type": "none",
        "statement_cache_size": None,
        "ssl_mode": None,
        "ssl_root_cert": None,
        "ssl_cert": None,
        "ssl_key": None,
        "server_settings": None,
        "connection_retry_attempts": 1,
        "connection_retry_backoff": 0.0,
        "connection_retry_backoff_max": 0.0,
        "pool_close_timeout": 1.0,
        "circuit_breaker_failure_threshold": 3,
        "circuit_breaker_recovery_timeout": 5.0,
    }
    config.update(overrides)
    return config


@pytest.mark.offline
class TestPoolAcquireTimeout:
    """测试 pool.acquire timeout 参数传递"""

    @pytest.mark.asyncio
    async def test_acquire_called_with_timeout(self):
        """pool.acquire 应使用配置的 timeout 参数"""
        db = PostgreSQLDB(make_db_config(pool_acquire_timeout=10.0))
        mock_conn = AsyncMock()

        acquire_mock = MagicMock()

        class MockAcquireContext:
            async def __aenter__(self):
                return mock_conn

            async def __aexit__(self, *args):
                return False

        acquire_mock.return_value = MockAcquireContext()

        mock_pool = make_mock_pool()
        mock_pool.acquire = acquire_mock
        db.pool = mock_pool

        await db._run_with_retry(lambda conn: conn.execute("SELECT 1"))

        # 验证 acquire 被调用时传入了 timeout 参数
        acquire_mock.assert_called_once_with(timeout=10.0)

    @pytest.mark.asyncio
    async def test_acquire_with_none_timeout(self):
        """timeout=None 时 pool.acquire 应传入 timeout=None"""
        db = PostgreSQLDB(make_db_config(pool_acquire_timeout=None))
        mock_conn = AsyncMock()

        acquire_mock = MagicMock()

        class MockAcquireContext:
            async def __aenter__(self):
                return mock_conn

            async def __aexit__(self, *args):
                return False

        acquire_mock.return_value = MockAcquireContext()

        mock_pool = make_mock_pool()
        mock_pool.acquire = acquire_mock
        db.pool = mock_pool

        await db._run_with_retry(lambda conn: conn.execute("SELECT 1"))

        acquire_mock.assert_called_once_with(timeout=None)

    @pytest.mark.asyncio
    async def test_default_timeout_value(self):
        """未配置 pool_acquire_timeout 时应使用默认值 10.0"""
        config = make_db_config()
        # 不传 pool_acquire_timeout
        db = PostgreSQLDB(config)
        assert db.pool_acquire_timeout == 10.0

    @pytest.mark.asyncio
    async def test_custom_timeout_value(self):
        """配置 pool_acquire_timeout=5.0 时应使用自定义值"""
        db = PostgreSQLDB(make_db_config(pool_acquire_timeout=5.0))
        assert db.pool_acquire_timeout == 5.0

    @pytest.mark.asyncio
    async def test_acquire_timeout_triggers_retry(self):
        """acquire 超时应触发重试机制"""
        db = PostgreSQLDB(
            make_db_config(
                pool_acquire_timeout=0.1,  # 100ms 超时
                connection_retry_attempts=2,
                connection_retry_backoff=0.0,
            )
        )

        call_count = {"value": 0}

        class TimeoutThenSuccess:
            """第一次 acquire 超时，第二次成功"""

            def __init__(self):
                call_count["value"] += 1

            async def __aenter__(self):
                if call_count["value"] == 1:
                    raise asyncio.TimeoutError("pool acquire timed out")
                return AsyncMock()

            async def __aexit__(self, *args):
                return False

        mock_pool = make_mock_pool()
        mock_pool.acquire = MagicMock(side_effect=lambda **kw: TimeoutThenSuccess())
        db.pool = mock_pool

        # mock _reset_pool 避免真正关闭连接池
        db._reset_pool = AsyncMock()

        await db._run_with_retry(lambda conn: conn.execute("SELECT 1"))

        # 应该重试了 2 次
        assert call_count["value"] == 2
        assert mock_pool.acquire.call_count == 2

    @pytest.mark.asyncio
    async def test_acquire_timeout_all_retries_fail(self):
        """所有重试都超时后应抛出 asyncio.TimeoutError"""
        db = PostgreSQLDB(
            make_db_config(
                pool_acquire_timeout=0.1,
                connection_retry_attempts=2,
                connection_retry_backoff=0.0,
            )
        )

        class AlwaysTimeout:
            async def __aenter__(self):
                raise asyncio.TimeoutError("pool acquire timed out")

            async def __aexit__(self, *args):
                return False

        mock_pool = make_mock_pool()
        mock_pool.acquire = MagicMock(return_value=AlwaysTimeout())
        db.pool = mock_pool

        db._reset_pool = AsyncMock()

        with pytest.raises(asyncio.TimeoutError):
            await db._run_with_retry(lambda conn: conn.execute("SELECT 1"))

        # 应该尝试了 2 次
        assert mock_pool.acquire.call_count == 2

    @pytest.mark.asyncio
    async def test_acquire_timeout_is_transient(self):
        """asyncio.TimeoutError 应被视为瞬态异常并触发断路器记录"""
        db = PostgreSQLDB(
            make_db_config(
                pool_acquire_timeout=0.1,
                connection_retry_attempts=1,
                connection_retry_backoff=0.0,
            )
        )

        class AlwaysTimeout:
            async def __aenter__(self):
                raise asyncio.TimeoutError("pool acquire timed out")

            async def __aexit__(self, *args):
                return False

        mock_pool = make_mock_pool()
        mock_pool.acquire = MagicMock(return_value=AlwaysTimeout())
        db.pool = mock_pool

        db._reset_pool = AsyncMock()

        assert db._circuit_breaker.failure_count == 0

        with pytest.raises(asyncio.TimeoutError):
            await db._run_with_retry(lambda conn: conn.execute("SELECT 1"))

        # 断路器应记录一次失败
        assert db._circuit_breaker.failure_count == 1


@pytest.mark.offline
class TestPoolAcquireTimeoutConfig:
    """测试环境变量配置"""

    def test_config_reads_env_variable(self):
        """ClientManager.get_config 应读取 POSTGRES_POOL_ACQUIRE_TIMEOUT 环境变量"""
        from lightrag.kg.postgres_impl import ClientManager

        with patch.dict("os.environ", {"POSTGRES_POOL_ACQUIRE_TIMEOUT": "15.0"}):
            config = ClientManager.get_config()
            assert config["pool_acquire_timeout"] == 15.0

    def test_config_default_value(self):
        """未设置环境变量时应使用默认值 10.0"""
        from lightrag.kg.postgres_impl import ClientManager

        env = {k: v for k, v in __import__("os").environ.items() if k != "POSTGRES_POOL_ACQUIRE_TIMEOUT"}
        with patch.dict("os.environ", env, clear=True):
            config = ClientManager.get_config()
            assert config["pool_acquire_timeout"] == 10.0
