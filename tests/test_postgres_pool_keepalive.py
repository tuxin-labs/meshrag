"""
PostgreSQL 连接池 keepalive 与空闲连接寿命测试

测试覆盖：
- TCP keepalive 参数（tcp_keepalives_idle/interval/count）配置解析与注入
- max_inactive_connection_lifetime 配置解析与传递给 create_pool
- 用户自定义 server_settings 与 keepalive 默认值的合并优先级（用户优先）
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightrag.kg.postgres_impl import PostgreSQLDB


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
class TestKeepaliveConfig:
    """测试 TCP keepalive 与空闲连接寿命的配置解析"""

    def test_default_keepalives_values(self):
        """未配置时应使用默认值 idle=60, interval=10, count=3"""
        db = PostgreSQLDB(make_db_config())
        assert db.keepalives_idle == 60
        assert db.keepalives_interval == 10
        assert db.keepalives_count == 3

    def test_custom_keepalives_values(self):
        """配置自定义 keepalive 值"""
        db = PostgreSQLDB(
            make_db_config(
                keepalives_idle=120,
                keepalives_interval=15,
                keepalives_count=5,
            )
        )
        assert db.keepalives_idle == 120
        assert db.keepalives_interval == 15
        assert db.keepalives_count == 5

    def test_keepalives_disabled_with_zero(self):
        """设为 0 表示禁用"""
        db = PostgreSQLDB(
            make_db_config(
                keepalives_idle=0,
                keepalives_interval=0,
                keepalives_count=0,
            )
        )
        assert db.keepalives_idle == 0
        assert db.keepalives_interval == 0
        assert db.keepalives_count == 0

    def test_default_max_inactive_connection_lifetime(self):
        """未配置时 max_inactive_connection_lifetime 默认 120.0"""
        db = PostgreSQLDB(make_db_config())
        assert db.max_inactive_connection_lifetime == 120.0

    def test_config_reads_keepalives_env(self):
        """ClientManager.get_config 应读取 POSTGRES_KEEPALIVES_* 环境变量"""
        from lightrag.kg.postgres_impl import ClientManager

        with patch.dict(
            "os.environ",
            {
                "POSTGRES_KEEPALIVES_IDLE": "90",
                "POSTGRES_KEEPALIVES_INTERVAL": "20",
                "POSTGRES_KEEPALIVES_COUNT": "6",
            },
        ):
            config = ClientManager.get_config()
            assert config["keepalives_idle"] == 90
            assert config["keepalives_interval"] == 20
            assert config["keepalives_count"] == 6

    def test_config_reads_max_inactive_env(self):
        """ClientManager.get_config 应读取 POSTGRES_MAX_INACTIVE_CONNECTION_LIFETIME"""
        from lightrag.kg.postgres_impl import ClientManager

        with patch.dict(
            "os.environ", {"POSTGRES_MAX_INACTIVE_CONNECTION_LIFETIME": "240"}
        ):
            config = ClientManager.get_config()
            assert config["max_inactive_connection_lifetime"] == 240.0

    def test_config_keepalives_disabled_env(self):
        """环境变量设 0 时应禁用 keepalive"""
        from lightrag.kg.postgres_impl import ClientManager

        with patch.dict(
            "os.environ",
            {
                "POSTGRES_KEEPALIVES_IDLE": "0",
                "POSTGRES_KEEPALIVES_INTERVAL": "0",
                "POSTGRES_KEEPALIVES_COUNT": "0",
            },
        ):
            config = ClientManager.get_config()
            assert config["keepalives_idle"] == 0
            assert config["keepalives_interval"] == 0
            assert config["keepalives_count"] == 0


@pytest.mark.offline
class TestKeepaliveInjection:
    """测试 keepalive 注入 server_settings"""

    @pytest.mark.asyncio
    @patch("lightrag.kg.postgres_impl.asyncpg.create_pool", new_callable=AsyncMock)
    async def test_keepalive_injected_into_server_settings(self, mock_create_pool):
        """默认应将 keepalive 参数注入 server_settings"""
        mock_create_pool.return_value = make_mock_pool()

        db = PostgreSQLDB(make_db_config(enable_vector=False))
        await db.initdb()

        _, kwargs = mock_create_pool.call_args
        server_settings = kwargs["server_settings"]
        assert server_settings["tcp_keepalives_idle"] == "60"
        assert server_settings["tcp_keepalives_interval"] == "10"
        assert server_settings["tcp_keepalives_count"] == "3"

    @pytest.mark.asyncio
    @patch("lightrag.kg.postgres_impl.asyncpg.create_pool", new_callable=AsyncMock)
    async def test_user_server_settings_takes_precedence(self, mock_create_pool):
        """用户通过 server_settings 显式设置的 keepalive key 不被默认值覆盖"""
        mock_create_pool.return_value = make_mock_pool()

        db = PostgreSQLDB(
            make_db_config(
                enable_vector=False,
                server_settings="tcp_keepalives_idle=180&extra_option=foo",
            )
        )
        await db.initdb()

        _, kwargs = mock_create_pool.call_args
        server_settings = kwargs["server_settings"]
        # 用户值优先，不被默认值覆盖
        assert server_settings["tcp_keepalives_idle"] == "180"
        # 未显式设置的仍用默认注入
        assert server_settings["tcp_keepalives_interval"] == "10"
        assert server_settings["tcp_keepalives_count"] == "3"
        # 用户的其他配置保留
        assert server_settings["extra_option"] == "foo"

    @pytest.mark.asyncio
    @patch("lightrag.kg.postgres_impl.asyncpg.create_pool", new_callable=AsyncMock)
    async def test_keepalive_disabled_not_injected(self, mock_create_pool):
        """keepalive 全设 0 时不注入 keepalive 参数"""
        mock_create_pool.return_value = make_mock_pool()

        db = PostgreSQLDB(
            make_db_config(
                enable_vector=False,
                keepalives_idle=0,
                keepalives_interval=0,
                keepalives_count=0,
            )
        )
        await db.initdb()

        _, kwargs = mock_create_pool.call_args
        server_settings = kwargs.get("server_settings", {})
        assert "tcp_keepalives_idle" not in server_settings
        assert "tcp_keepalives_interval" not in server_settings
        assert "tcp_keepalives_count" not in server_settings

    @pytest.mark.asyncio
    @patch("lightrag.kg.postgres_impl.asyncpg.create_pool", new_callable=AsyncMock)
    async def test_no_user_server_settings_still_injects_keepalive(
        self, mock_create_pool
    ):
        """用户未配置 server_settings 时，仍应注入 keepalive"""
        mock_create_pool.return_value = make_mock_pool()

        db = PostgreSQLDB(make_db_config(enable_vector=False, server_settings=None))
        await db.initdb()

        _, kwargs = mock_create_pool.call_args
        assert kwargs["server_settings"]["tcp_keepalives_idle"] == "60"


@pytest.mark.offline
class TestMaxInactiveConnectionLifetime:
    """测试 max_inactive_connection_lifetime 传递给 create_pool"""

    @pytest.mark.asyncio
    @patch("lightrag.kg.postgres_impl.asyncpg.create_pool", new_callable=AsyncMock)
    async def test_default_lifetime_passed_to_create_pool(self, mock_create_pool):
        """默认 max_inactive_connection_lifetime=120.0 应传给 create_pool"""
        mock_create_pool.return_value = make_mock_pool()

        db = PostgreSQLDB(make_db_config(enable_vector=False))
        await db.initdb()

        _, kwargs = mock_create_pool.call_args
        assert kwargs["max_inactive_connection_lifetime"] == 120.0

    @pytest.mark.asyncio
    @patch("lightrag.kg.postgres_impl.asyncpg.create_pool", new_callable=AsyncMock)
    async def test_custom_lifetime_passed(self, mock_create_pool):
        """自定义值应传给 create_pool"""
        mock_create_pool.return_value = make_mock_pool()

        db = PostgreSQLDB(
            make_db_config(enable_vector=False, max_inactive_connection_lifetime=240.0)
        )
        await db.initdb()

        _, kwargs = mock_create_pool.call_args
        assert kwargs["max_inactive_connection_lifetime"] == 240.0

    @pytest.mark.asyncio
    @patch("lightrag.kg.postgres_impl.asyncpg.create_pool", new_callable=AsyncMock)
    async def test_disabled_lifetime_zero(self, mock_create_pool):
        """设为 0 时传 0 给 create_pool（asyncpg 语义为禁用空闲连接回收）"""
        mock_create_pool.return_value = make_mock_pool()

        db = PostgreSQLDB(
            make_db_config(enable_vector=False, max_inactive_connection_lifetime=0)
        )
        await db.initdb()

        _, kwargs = mock_create_pool.call_args
        assert kwargs["max_inactive_connection_lifetime"] == 0
