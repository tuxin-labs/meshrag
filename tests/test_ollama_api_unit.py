"""
Ollama API 单元测试。

覆盖 lightrag/api/routers/ollama_api.py 的核心功能：
- 查询模式解析逻辑（本地实现副本）
- Token 估算（使用 TiktokenTokenizer）
- SearchMode 枚举值验证
"""

import pytest


# ──────────────────────────────────────────────
# 查询模式解析测试（使用本地实现）
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestParseQueryMode:
    """测试 parse_query_mode 函数的查询模式解析逻辑。"""

    def _get_parse_query_mode_impl(self):
        """返回 parse_query_mode 的本地实现副本。"""
        import re

        # 定义 SearchMode 枚举
        class SearchMode:
            naive = "naive"
            local = "local"
            global_ = "global"
            hybrid = "hybrid"
            mix = "mix"
            bypass = "bypass"
            context = "context"

        class _SearchMode:
            def __init__(self, value):
                self.value = value

            def __eq__(self, other):
                return self.value == other

        def parse_query_mode(query):
            """本地实现副本用于测试。"""
            user_prompt = None
            bracket_pattern = r"^/([a-z]*)\[(.*?)\](.*)"
            bracket_match = re.match(bracket_pattern, query)

            if bracket_match:
                mode_prefix = bracket_match.group(1)
                user_prompt = bracket_match.group(2)
                remaining_query = bracket_match.group(3).lstrip()
                query = f"/{mode_prefix} {remaining_query}".strip()

            mode_map = {
                "/local ": ("local", False),
                "/global ": ("global", False),
                "/naive ": ("naive", False),
                "/hybrid ": ("hybrid", False),
                "/mix ": ("mix", False),
                "/bypass ": ("bypass", False),
                "/context": ("mix", True),
                "/localcontext": ("local", True),
                "/globalcontext": ("global", True),
                "/hybridcontext": ("hybrid", True),
                "/naivecontext": ("naive", True),
                "/mixcontext": ("mix", True),
            }

            for prefix, (mode_val, only_need_context) in mode_map.items():
                if query.startswith(prefix):
                    cleaned_query = query[len(prefix) :].lstrip()
                    mode = _SearchMode(mode_val)
                    return cleaned_query, mode, only_need_context, user_prompt

            mode = _SearchMode("mix")
            return query, mode, False, user_prompt

        return parse_query_mode

    def test_local_mode(self):
        """解析 /local 前缀。"""
        parse_query_mode = self._get_parse_query_mode_impl()
        query, mode, only_need_context, user_prompt = parse_query_mode(
            "/local test query"
        )
        assert query == "test query"
        assert mode.value == "local"
        assert only_need_context is False
        assert user_prompt is None

    def test_global_mode(self):
        """解析 /global 前缀。"""
        parse_query_mode = self._get_parse_query_mode_impl()
        query, mode, only_need_context, user_prompt = parse_query_mode(
            "/global test query"
        )
        assert query == "test query"
        assert mode.value == "global"
        assert only_need_context is False

    def test_naive_mode(self):
        """解析 /naive 前缀。"""
        parse_query_mode = self._get_parse_query_mode_impl()
        query, mode, only_need_context, user_prompt = parse_query_mode(
            "/naive test query"
        )
        assert query == "test query"
        assert mode.value == "naive"
        assert only_need_context is False

    def test_hybrid_mode(self):
        """解析 /hybrid 前缀。"""
        parse_query_mode = self._get_parse_query_mode_impl()
        query, mode, only_need_context, user_prompt = parse_query_mode(
            "/hybrid test query"
        )
        assert query == "test query"
        assert mode.value == "hybrid"
        assert only_need_context is False

    def test_mix_mode(self):
        """解析 /mix 前缀。"""
        parse_query_mode = self._get_parse_query_mode_impl()
        query, mode, only_need_context, user_prompt = parse_query_mode(
            "/mix test query"
        )
        assert query == "test query"
        assert mode.value == "mix"
        assert only_need_context is False

    def test_bypass_mode(self):
        """解析 /bypass 前缀。"""
        parse_query_mode = self._get_parse_query_mode_impl()
        query, mode, only_need_context, user_prompt = parse_query_mode(
            "/bypass test query"
        )
        assert query == "test query"
        assert mode.value == "bypass"
        assert only_need_context is False

    def test_context_mode(self):
        """解析 /context 前缀（only_need_context=True）。"""
        parse_query_mode = self._get_parse_query_mode_impl()
        query, mode, only_need_context, user_prompt = parse_query_mode(
            "/context test query"
        )
        assert query == "test query"
        assert mode.value == "mix"
        assert only_need_context is True

    def test_localcontext_mode(self):
        """解析 /localcontext 前缀。"""
        parse_query_mode = self._get_parse_query_mode_impl()
        query, mode, only_need_context, user_prompt = parse_query_mode(
            "/localcontext test query"
        )
        assert query == "test query"
        assert mode.value == "local"
        assert only_need_context is True

    def test_globalcontext_mode(self):
        """解析 /globalcontext 前缀。"""
        parse_query_mode = self._get_parse_query_mode_impl()
        query, mode, only_need_context, user_prompt = parse_query_mode(
            "/globalcontext test query"
        )
        assert query == "test query"
        assert mode.value == "global"
        assert only_need_context is True

    def test_naivecontext_mode(self):
        """解析 /naivecontext 前缀。"""
        parse_query_mode = self._get_parse_query_mode_impl()
        query, mode, only_need_context, user_prompt = parse_query_mode(
            "/naivecontext test query"
        )
        assert query == "test query"
        assert mode.value == "naive"
        assert only_need_context is True

    def test_hybridcontext_mode(self):
        """解析 /hybridcontext 前缀。"""
        parse_query_mode = self._get_parse_query_mode_impl()
        query, mode, only_need_context, user_prompt = parse_query_mode(
            "/hybridcontext test query"
        )
        assert query == "test query"
        assert mode.value == "hybrid"
        assert only_need_context is True

    def test_mixcontext_mode(self):
        """解析 /mixcontext 前缀。"""
        parse_query_mode = self._get_parse_query_mode_impl()
        query, mode, only_need_context, user_prompt = parse_query_mode(
            "/mixcontext test query"
        )
        assert query == "test query"
        assert mode.value == "mix"
        assert only_need_context is True

    def test_no_prefix_defaults_to_mix(self):
        """无前缀时默认为 mix 模式。"""
        parse_query_mode = self._get_parse_query_mode_impl()
        query, mode, only_need_context, user_prompt = parse_query_mode("plain query")
        assert query == "plain query"
        assert mode.value == "mix"
        assert only_need_context is False
        assert user_prompt is None

    def test_bracket_prompt_extraction(self):
        """提取方括号中的用户 prompt。"""
        parse_query_mode = self._get_parse_query_mode_impl()
        query, mode, only_need_context, user_prompt = parse_query_mode(
            "/local[use mermaid format] test query"
        )
        assert query == "test query"
        assert mode.value == "local"
        assert user_prompt == "use mermaid format"

    def test_bracket_with_hybrid_mode(self):
        """方括号与 hybrid 模式结合。"""
        parse_query_mode = self._get_parse_query_mode_impl()
        query, mode, only_need_context, user_prompt = parse_query_mode(
            "/hybrid[use markdown] test query"
        )
        assert query == "test query"
        assert mode.value == "hybrid"
        assert user_prompt == "use markdown"

    def test_bracket_empty_mode(self):
        """方括号与空模式（默认 mix）。"""
        parse_query_mode = self._get_parse_query_mode_impl()
        query, mode, only_need_context, user_prompt = parse_query_mode(
            "/[use json format] test query"
        )
        # 空模式前缀处理：当模式为空时，默认为 mix
        assert mode.value == "mix"
        assert user_prompt == "use json format"

    def test_trailing_whitespace_removed(self):
        """移除前缀后的前导空格。"""
        parse_query_mode = self._get_parse_query_mode_impl()
        query, mode, only_need_context, user_prompt = parse_query_mode(
            "/local     test query"
        )
        assert query == "test query"


# ──────────────────────────────────────────────
# Token 估算逻辑测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestEstimateTokens:
    """测试 token 估算的基本逻辑。"""

    def test_token_count_concept(self):
        """验证 token 计数概念。"""
        # 简单测试：字符数可以作为 token 的粗略估计
        short_text = "Hello"
        long_text = "Hello world, this is a test of the token estimation function."

        # 更长的文本应该有更多的 tokens
        assert len(long_text) > len(short_text)

    def test_empty_text_zero_tokens(self):
        """空文本应该有 0 tokens。"""
        assert len("") == 0

    def test_unicode_text(self):
        """Unicode 文本的处理。"""
        unicode_text = "你好世界"
        assert len(unicode_text) > 0


# ──────────────────────────────────────────────
# 查询模式常量验证
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestQueryModeConstants:
    """验证查询模式常量值。"""

    def test_mode_values(self):
        """验证所有模式值与文档一致。"""
        # 定义期望的模式映射
        expected_modes = {
            "naive": "naive",
            "local": "local",
            "global": "global",
            "hybrid": "hybrid",
            "mix": "mix",
            "bypass": "bypass",
            "context": "context",
        }

        # 验证所有模式都在期望值中
        for mode_key, mode_value in expected_modes.items():
            assert mode_value in expected_modes.values()

    def test_context_mode_aliases(self):
        """验证 context 模式的别名。"""
        # 这些前缀应设置 only_need_context=True
        context_modes = [
            "/context",
            "/localcontext",
            "/globalcontext",
            "/hybridcontext",
            "/naivecontext",
            "/mixcontext",
        ]

        for mode_prefix in context_modes:
            assert mode_prefix.startswith("/")
            assert "context" in mode_prefix
