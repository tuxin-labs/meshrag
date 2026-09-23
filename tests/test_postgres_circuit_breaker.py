"""
PostgreSQL 断路器（Circuit Breaker）测试

测试覆盖：
- 单元测试：PostgreSQLCircuitBreaker 状态机转换
- 集成测试：断路器与 _run_with_retry 的交互
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightrag.exceptions import CircuitBreakerOpenError
from lightrag.kg.postgres_impl import CircuitState, PostgreSQLCircuitBreaker


def make_mock_pool():
    """创建一个兼容 asyncpg.Pool 的 mock 对象。

    asyncpg.Pool 的 acquire() 返回 async context manager，
    close() 是 async 方法。MagicMock 默认不满足这些约束。
    """
    mock_pool = MagicMock()

    async def _async_noop():
        pass

    mock_pool.close = MagicMock(return_value=_async_noop())
    return mock_pool


# ============================================================
# 单元测试：PostgreSQLCircuitBreaker 状态机
# ============================================================


@pytest.mark.offline
class TestCircuitBreakerStateMachine:
    """测试断路器状态机转换"""

    def test_initial_state_is_closed(self):
        """初始状态应为 CLOSED"""
        cb = PostgreSQLCircuitBreaker(failure_threshold=3, recovery_timeout=10.0)
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_allow_request_when_closed(self):
        """CLOSED 状态下 allow_request() 应始终返回 True"""
        cb = PostgreSQLCircuitBreaker(failure_threshold=3, recovery_timeout=10.0)
        assert await cb.allow_request() is True
        assert await cb.allow_request() is True  # 多次调用都返回 True

    @pytest.mark.asyncio
    async def test_opens_after_threshold_failures(self):
        """连续失败达到阈值后应转为 OPEN"""
        cb = PostgreSQLCircuitBreaker(failure_threshold=3, recovery_timeout=10.0)
        exc = ConnectionError("test connection error")

        await cb.record_failure(exc)
        assert cb.state == CircuitState.CLOSED  # 1/3，还不到阈值

        await cb.record_failure(exc)
        assert cb.state == CircuitState.CLOSED  # 2/3

        await cb.record_failure(exc)
        assert cb.state == CircuitState.OPEN  # 3/3，达到阈值

    @pytest.mark.asyncio
    async def test_opens_exactly_at_threshold(self):
        """阈值-1 次失败不应打开断路器"""
        cb = PostgreSQLCircuitBreaker(failure_threshold=5, recovery_timeout=10.0)
        exc = ConnectionError("test")

        for i in range(4):
            await cb.record_failure(exc)
        assert cb.state == CircuitState.CLOSED  # 4/5

        await cb.record_failure(exc)
        assert cb.state == CircuitState.OPEN  # 5/5

    @pytest.mark.asyncio
    async def test_fast_fail_when_open(self):
        """OPEN 状态下 allow_request() 应返回 False"""
        cb = PostgreSQLCircuitBreaker(failure_threshold=2, recovery_timeout=60.0)
        exc = ConnectionError("test")

        await cb.record_failure(exc)
        await cb.record_failure(exc)
        assert cb.state == CircuitState.OPEN

        # 在 recovery_timeout 内，所有请求都应被拒绝
        assert await cb.allow_request() is False
        assert await cb.allow_request() is False
        assert await cb.allow_request() is False

    @pytest.mark.asyncio
    async def test_half_open_after_recovery_timeout(self):
        """超过 recovery_timeout 后应转为 HALF_OPEN 并允许探测"""
        cb = PostgreSQLCircuitBreaker(failure_threshold=2, recovery_timeout=5.0)
        exc = ConnectionError("test")

        # 打开断路器
        await cb.record_failure(exc)
        await cb.record_failure(exc)
        assert cb.state == CircuitState.OPEN

        # 模拟时间经过 recovery_timeout
        with patch("time.monotonic") as mock_time:
            # 第一次调用：allow_request 内部获取当前时间
            # 第二次调用：获取 last_failure_time 对应的时间
            base_time = 1000.0
            mock_time.side_effect = [base_time, base_time - 5.0, base_time]
            # now=1000.0, last_failure_time was set to some earlier value
            # 我们需要 last_failure_time + recovery_timeout <= now
            # 直接修改内部状态更简单
            cb._last_failure_time = base_time - 10.0  # 10 秒前

        # 不需要 mock，直接设置 last_failure_time 为过去
        result = await cb.allow_request()
        assert result is True
        assert cb.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_half_open_allows_only_one_probe(self):
        """HALF_OPEN 状态下只允许一个探测请求"""
        cb = PostgreSQLCircuitBreaker(failure_threshold=2, recovery_timeout=5.0)

        # 直接设置为 HALF_OPEN 状态
        cb._state = CircuitState.HALF_OPEN
        cb._half_open_in_progress = False

        # 第一个请求通过
        assert await cb.allow_request() is True
        assert cb._half_open_in_progress is True

        # 第二个请求被拒绝
        assert await cb.allow_request() is False

    @pytest.mark.asyncio
    async def test_half_open_probe_success_closes(self):
        """探测成功后应关闭断路器"""
        cb = PostgreSQLCircuitBreaker(failure_threshold=2, recovery_timeout=5.0)

        # 模拟从 HALF_OPEN 恢复
        cb._state = CircuitState.HALF_OPEN
        cb._half_open_in_progress = True

        await cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb._half_open_in_progress is False

    @pytest.mark.asyncio
    async def test_half_open_probe_failure_reopens(self):
        """探测失败后应重新打开断路器"""
        cb = PostgreSQLCircuitBreaker(failure_threshold=2, recovery_timeout=5.0)

        # 模拟 HALF_OPEN 状态
        cb._state = CircuitState.HALF_OPEN
        cb._half_open_in_progress = True

        exc = ConnectionError("probe failed")
        await cb.record_failure(exc)
        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self):
        """成功操作应重置失败计数器"""
        cb = PostgreSQLCircuitBreaker(failure_threshold=5, recovery_timeout=10.0)
        exc = ConnectionError("test")

        # 3 次失败（低于阈值）
        for _ in range(3):
            await cb.record_failure(exc)
        assert cb.failure_count == 3

        # 1 次成功
        await cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_reset_forces_closed(self):
        """reset() 应强制回到 CLOSED 状态"""
        cb = PostgreSQLCircuitBreaker(failure_threshold=2, recovery_timeout=10.0)
        exc = ConnectionError("test")

        # 打开断路器
        await cb.record_failure(exc)
        await cb.record_failure(exc)
        assert cb.state == CircuitState.OPEN

        # 强制重置
        await cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb._last_failure_time is None

    @pytest.mark.asyncio
    async def test_failure_records_last_failure_time(self):
        """失败时应记录 last_failure_time"""
        cb = PostgreSQLCircuitBreaker(failure_threshold=5, recovery_timeout=10.0)
        assert cb._last_failure_time is None

        await cb.record_failure(ConnectionError("test"))
        assert cb._last_failure_time is not None
        assert isinstance(cb._last_failure_time, float)

    @pytest.mark.asyncio
    async def test_success_clears_last_failure_time(self):
        """成功时应清除 last_failure_time"""
        cb = PostgreSQLCircuitBreaker(failure_threshold=5, recovery_timeout=10.0)

        await cb.record_failure(ConnectionError("test"))
        assert cb._last_failure_time is not None

        await cb.record_success()
        assert cb._last_failure_time is None


# ============================================================
# 集成测试：断路器与 _run_with_retry 的交互
# ============================================================


@pytest.mark.offline
class TestCircuitBreakerWithRunWithRetry:
    """测试断路器与 PostgreSQLDB._run_with_retry 的集成"""

    @pytest.fixture
    def mock_db(self):
        """创建带有 mock 连接池的 PostgreSQLDB 实例"""
        from lightrag.kg.postgres_impl import PostgreSQLDB

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
            "connection_retry_attempts": 1,  # 不重试，每次 _run_with_retry 只尝试 1 次
            "connection_retry_backoff": 0.0,  # 无退避加速测试
            "connection_retry_backoff_max": 0.0,
            "pool_close_timeout": 1.0,
            "circuit_breaker_failure_threshold": 3,
            "circuit_breaker_recovery_timeout": 5.0,
        }
        db = PostgreSQLDB(config)
        return db

    @pytest.mark.asyncio
    async def test_fast_fails_when_circuit_open(self, mock_db):
        """断路器打开时应立即抛出 CircuitBreakerOpenError，不尝试 acquire"""
        # 强制打开断路器
        mock_db._circuit_breaker._state = CircuitState.OPEN
        mock_db._circuit_breaker._failure_count = 5
        mock_db._circuit_breaker._last_failure_time = time.monotonic() + 1000  # 远未来

        # mock pool（不应被调用）
        mock_pool = make_mock_pool()
        mock_db.pool = mock_pool

        with pytest.raises(CircuitBreakerOpenError):
            await mock_db._run_with_retry(lambda conn: conn.execute("SELECT 1"))

        # 验证 pool.acquire 没有被调用
        mock_pool.acquire.assert_not_called()

    @pytest.mark.asyncio
    async def test_records_failure_on_retry_exhaustion(self, mock_db):
        """_run_with_retry 重试耗尽后应记录一次断路器失败"""
        import asyncpg

        mock_pool = make_mock_pool()

        class FailingAcquireContext:
            async def __aenter__(self):
                raise asyncpg.exceptions.ConnectionFailureError("test")
            async def __aexit__(self, *args):
                return False

        mock_pool.acquire = MagicMock(return_value=FailingAcquireContext())
        mock_db.pool = mock_pool

        # mock _ensure_pool 和 _reset_pool，避免调用真正的 initdb
        mock_db._ensure_pool = AsyncMock()
        mock_db._reset_pool = AsyncMock()

        assert mock_db._circuit_breaker.failure_count == 0

        with pytest.raises(asyncpg.exceptions.ConnectionFailureError):
            await mock_db._run_with_retry(lambda conn: conn.execute("SELECT 1"))

        # 应记录恰好 1 次失败（不是重试次数 2 次）
        assert mock_db._circuit_breaker.failure_count == 1

    @pytest.mark.asyncio
    async def test_records_success_on_operation_success(self, mock_db):
        """成功操作后应记录成功"""
        mock_conn = AsyncMock()

        class SuccessAcquireContext:
            async def __aenter__(self):
                return mock_conn
            async def __aexit__(self, *args):
                return False

        mock_pool = make_mock_pool()
        mock_pool.acquire = MagicMock(return_value=SuccessAcquireContext())
        mock_db.pool = mock_pool

        # 先设置一些失败计数
        mock_db._circuit_breaker._failure_count = 2

        await mock_db._run_with_retry(lambda conn: conn.execute("SELECT 1"))

        # 成功后计数应重置为 0
        assert mock_db._circuit_breaker.failure_count == 0
        assert mock_db._circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_recovery_flow_e2e(self, mock_db):
        """端到端恢复流程：失败 → 打开 → 探测 → 恢复"""
        import asyncpg

        call_count = {"value": 0}
        mock_conn = AsyncMock()

        class DynamicAcquireContext:
            """根据调用次数决定成功或失败的 async context manager"""
            def __init__(self):
                call_count["value"] += 1

            async def __aenter__(self):
                if call_count["value"] <= 3:
                    raise asyncpg.exceptions.ConnectionFailureError("down")
                return mock_conn

            async def __aexit__(self, *args):
                return False

        mock_pool = make_mock_pool()
        mock_pool.acquire = MagicMock(side_effect=lambda **kw: DynamicAcquireContext())
        mock_db.pool = mock_pool

        # mock _ensure_pool 和 _reset_pool，避免调用真正的 initdb
        mock_db._ensure_pool = AsyncMock()
        mock_db._reset_pool = AsyncMock()

        # 阶段 1：连续失败触发断路器打开
        for _ in range(3):
            with pytest.raises(asyncpg.exceptions.ConnectionFailureError):
                await mock_db._run_with_retry(lambda conn: conn.execute("SELECT 1"))

        assert mock_db._circuit_breaker.state == CircuitState.OPEN
        assert mock_db._circuit_breaker.failure_count == 3

        # 阶段 2：断路器打开，快速失败
        with pytest.raises(CircuitBreakerOpenError):
            await mock_db._run_with_retry(lambda conn: conn.execute("SELECT 1"))

        # 阶段 3：模拟 recovery_timeout 已过，转为 HALF_OPEN
        mock_db._circuit_breaker._last_failure_time = time.monotonic() - 10.0

        # 阶段 4：探测成功 → 关闭断路器
        await mock_db._run_with_retry(lambda conn: conn.execute("SELECT 1"))
        assert mock_db._circuit_breaker.state == CircuitState.CLOSED
        assert mock_db._circuit_breaker.failure_count == 0
