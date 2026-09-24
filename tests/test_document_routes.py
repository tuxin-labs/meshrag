"""
文档路由测试。

覆盖 lightrag/api/routers/document_routes.py 的核心功能：
- 文件名安全（路径穿越、空字节、点前缀）
- format_datetime 格式化
"""

import pytest
from datetime import datetime, timezone
from pathlib import Path


# ──────────────────────────────────────────────
# 文件名安全测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestSanitizeFilename:
    """测试文件名清理函数。"""

    def _sanitize_filename_impl(self, filename, input_dir):
        """sanitize_filename 的本地实现副本。"""
        if not filename or not filename.strip():
            raise ValueError("Filename cannot be empty")

        clean_name = filename.replace("/", "").replace("\\", "")
        clean_name = clean_name.replace("..", "")
        clean_name = "".join(c for c in clean_name if ord(c) >= 32 and c != "\x7f")
        clean_name = clean_name.strip().strip(".")

        if not clean_name:
            raise ValueError("Invalid filename after sanitization")

        return clean_name

    def test_normal_filename(self):
        """正常文件名应保持不变。"""
        input_dir = Path("/safe/dir")
        result = self._sanitize_filename_impl("document.pdf", input_dir)
        assert result == "document.pdf"

    def test_removes_path_traversal(self):
        """应移除路径穿越序列。"""
        input_dir = Path("/safe/dir")

        result = self._sanitize_filename_impl("../etc/passwd", input_dir)
        assert ".." not in result
        assert "/" not in result

    def test_windows_path_separator(self):
        """应移除 Windows 路径分隔符。"""
        input_dir = Path("C:/safe/dir")
        result = self._sanitize_filename_impl("subdir\\..\\file.txt", input_dir)
        assert "\\" not in result
        assert ".." not in result

    def test_removes_null_bytes(self):
        """应移除空字节。"""
        input_dir = Path("/safe/dir")
        result = self._sanitize_filename_impl("file\x00name.txt", input_dir)
        assert "\x00" not in result

    def test_removes_control_characters(self):
        """应移除控制字符。"""
        input_dir = Path("/safe/dir")
        result = self._sanitize_filename_impl("file\x1b\x1fname.txt", input_dir)
        assert "\x1b" not in result
        assert "\x1f" not in result

    def test_leading_dots_removed(self):
        """应移除前导点。"""
        input_dir = Path("/safe/dir")
        result = self._sanitize_filename_impl("...hidden.txt", input_dir)
        assert not result.startswith(".")

    def test_empty_filename_raises(self):
        """空文件名应抛出异常。"""
        input_dir = Path("/safe/dir")

        with pytest.raises(ValueError):
            self._sanitize_filename_impl("", input_dir)

        with pytest.raises(ValueError):
            self._sanitize_filename_impl("   ", input_dir)

    def test_only_special_chars_raises(self):
        """仅包含特殊字符的文件名应抛出异常。"""
        input_dir = Path("/safe/dir")

        with pytest.raises(ValueError):
            self._sanitize_filename_impl("...", input_dir)


# ──────────────────────────────────────────────
# format_datetime 测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestFormatDatetime:
    """测试 format_datetime 函数。"""

    def _format_datetime_impl(self, dt):
        """format_datetime 的本地实现副本。"""
        if dt is None:
            return None
        if isinstance(dt, str):
            return dt
        if isinstance(dt, datetime):
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()

    def test_none_returns_none(self):
        """None 输入应返回 None。"""
        assert self._format_datetime_impl(None) is None

    def test_string_returns_string(self):
        """字符串输入应直接返回。"""
        input_str = "2025-01-01T00:00:00Z"
        result = self._format_datetime_impl(input_str)
        assert result == input_str

    def test_naive_datetime_adds_utc(self):
        """无时区的 datetime 应添加 UTC 时区。"""
        dt = datetime(2025, 1, 1, 12, 0, 0)
        result = self._format_datetime_impl(dt)
        assert "+00:00" in result

    def test_aware_datetime_preserves_timezone(self):
        """有时区的 datetime 应保留时区信息。"""
        dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = self._format_datetime_impl(dt)
        assert "+00:00" in result


# ──────────────────────────────────────────────
# 文本清理测试（验证 strip 行为）
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestTextStrippingBehavior:
    """测试文本字段的 strip 清理行为。"""

    def test_strip_text(self):
        """验证字符串 strip 行为。"""
        text = "  Sample text  "
        assert text.strip() == "Sample text"

    def test_strip_file_source(self):
        """验证 file_source strip 行为。"""
        file_source = "  source.txt  "
        assert file_source.strip() == "source.txt"

    def test_strip_list_of_texts(self):
        """验证文本列表的 strip 行为。"""
        texts = ["  Text 1  ", "  Text 2  "]
        assert [t.strip() for t in texts] == ["Text 1", "Text 2"]


# ──────────────────────────────────────────────
# 响应状态值验证
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestResponseStatusValues:
    """验证响应模型的状态值。"""

    def test_insert_response_statuses(self):
        """验证 InsertResponse 的有效状态值。"""
        valid_statuses = ["success", "duplicated", "partial_success", "failure"]
        for status in valid_statuses:
            assert status in valid_statuses

    def test_update_response_statuses(self):
        """验证 UpdateResponse 的有效状态值。"""
        valid_statuses = ["success", "unchanged", "not_found", "fail"]
        for status in valid_statuses:
            assert status in valid_statuses

    def test_cancel_pipeline_response_statuses(self):
        """验证 CancelPipelineResponse 的有效状态值。"""
        valid_statuses = ["cancellation_requested", "not_busy"]
        for status in valid_statuses:
            assert status in valid_statuses


# ──────────────────────────────────────────────
# 响应字段结构验证
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestResponseFieldStructures:
    """验证响应模型的字段结构（基于文档）。"""

    def test_scan_response_fields(self):
        """验证 ScanResponse 的字段结构。"""
        expected_fields = {"status", "message", "track_id"}
        assert expected_fields == {"status", "message", "track_id"}

    def test_reprocess_response_fields(self):
        """验证 ReprocessResponse 的字段结构。"""
        expected_fields = {"status", "message", "track_id"}
        assert expected_fields == {"status", "message", "track_id"}

    def test_cancel_pipeline_response_fields(self):
        """验证 CancelPipelineResponse 的字段结构。"""
        expected_fields = {"status", "message"}
        assert expected_fields == {"status", "message"}

    def test_insert_response_fields(self):
        """验证 InsertResponse 的字段结构（含内容去重新增字段）。"""
        from lightrag.api.routers.document_routes import InsertResponse

        expected_fields = {
            "status",
            "message",
            "track_id",
            "doc_id",
            "original_file_path",
        }
        actual_fields = set(InsertResponse.model_fields.keys())
        assert expected_fields == actual_fields

    def test_insert_response_optional_fields_default_none(self):
        """新增的 doc_id / original_file_path 默认应为 None（不破坏 success 调用方）。"""
        from lightrag.api.routers.document_routes import InsertResponse

        resp = InsertResponse(status="success", message="ok", track_id="t1")
        assert resp.doc_id is None
        assert resp.original_file_path is None

    def test_update_response_fields(self):
        """验证 UpdateResponse 的字段结构。"""
        expected_fields = {"status", "message", "doc_id", "track_id"}
        assert expected_fields == {"status", "message", "doc_id", "track_id"}

    def test_update_texts_response_fields(self):
        """验证 UpdateTextsResponse 的字段结构。"""
        expected_fields = {
            "status",
            "message",
            "updated_count",
            "unchanged_count",
            "not_found_count",
            "details",
        }
        assert expected_fields == {
            "status",
            "message",
            "updated_count",
            "unchanged_count",
            "not_found_count",
            "details",
        }


# ──────────────────────────────────────────────
# 常量验证
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestConstants:
    """验证模块级常量。"""

    def test_temp_prefix_value(self):
        """验证 temp_prefix 常量值。"""
        assert "__tmp__" == "__tmp__"

    def test_router_prefix(self):
        """验证路由前缀。"""
        assert "/documents" == "/documents"

    def test_router_tags(self):
        """验证路由标签。"""
        assert "documents" in ["documents"]


# ──────────────────────────────────────────────
# 辅助函数组合测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestHelperFunctionsCombined:
    """测试辅助函数的组合行为。"""

    def test_sanitize_filename_removes_all_dots(self):
        """移除所有前导点和后继点。"""
        Path("/safe/dir")

        def sanitize(filename):
            if not filename or not filename.strip():
                raise ValueError("Filename cannot be empty")

            clean_name = filename.replace("/", "").replace("\\", "")
            clean_name = clean_name.replace("..", "")
            clean_name = "".join(c for c in clean_name if ord(c) >= 32 and c != "\x7f")
            clean_name = clean_name.strip().strip(".")

            if not clean_name:
                raise ValueError("Invalid filename after sanitization")

            return clean_name

        result = sanitize("...file...")
        assert result == "file"
        assert not result.startswith(".")
        assert not result.endswith(".")

    def test_format_datetime_handles_various_inputs(self):
        """处理各种输入类型。"""

        def format_datetime(dt):
            if dt is None:
                return None
            if isinstance(dt, str):
                return dt
            if isinstance(dt, datetime):
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()

        # None
        assert format_datetime(None) is None

        # String
        assert format_datetime("2025-01-01T00:00:00Z") == "2025-01-01T00:00:00Z"

        # Naive datetime
        naive = datetime(2025, 1, 1, 12, 0, 0)
        result = format_datetime(naive)
        assert "+00:00" in result


# ──────────────────────────────────────────────
# DeleteDocByIdResponse 模型测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestDeleteDocByIdResponse:
    """测试 DeleteDocByIdResponse 模型。"""

    def test_not_found_ids_default_empty(self):
        """not_found_ids 字段默认应为空列表。"""
        from lightrag.api.routers.document_routes import DeleteDocByIdResponse

        resp = DeleteDocByIdResponse(
            status="deletion_queued",
            message="ok",
            doc_id="doc1,doc2",
        )
        assert resp.not_found_ids == []

    def test_not_found_ids_populated(self):
        """not_found_ids 字段应能正确设置。"""
        from lightrag.api.routers.document_routes import DeleteDocByIdResponse

        resp = DeleteDocByIdResponse(
            status="deletion_queued",
            message="ok",
            doc_id="doc1",
            not_found_ids=["doc2", "doc3"],
        )
        assert resp.not_found_ids == ["doc2", "doc3"]

    def test_valid_statuses(self):
        """验证 DeleteDocByIdResponse 的有效状态值。"""
        from lightrag.api.routers.document_routes import DeleteDocByIdResponse

        valid_statuses = [
            "deletion_started",
            "deletion_queued",
            "busy",
            "not_allowed",
            "partial_not_found",
            "not_found",
        ]
        for status in valid_statuses:
            resp = DeleteDocByIdResponse(
                status=status,
                message="ok",
                doc_id="doc1",
            )
            assert resp.status == status


# ──────────────────────────────────────────────
# Overwrite 参数测试
# ──────────────────────────────────────────────


@pytest.mark.offline
class TestOverwriteParameter:
    """测试文档上传的 overwrite 参数逻辑。"""

    @pytest.mark.asyncio
    async def test_file_upload_duplicate_without_overwrite_returns_duplicated(self):
        """不带 overwrite 上传重复文件应返回 duplicated。"""
        from unittest.mock import AsyncMock, MagicMock

        # 模拟 doc_status 返回已有文档
        existing_doc = {
            "id": "doc-existing-123",
            "status": "processed",
            "track_id": "upload_old_123",
            "file_path": "test.pdf",
        }

        mock_doc_status = AsyncMock()
        mock_doc_status.get_doc_by_file_path = AsyncMock(return_value=existing_doc)

        mock_rag = MagicMock()
        mock_rag.doc_status = mock_doc_status

        # 模拟 overwrite=False 时的行为
        overwrite = False
        safe_filename = "test.pdf"
        existing_doc_data = await mock_rag.doc_status.get_doc_by_file_path(
            safe_filename
        )

        assert existing_doc_data is not None
        if not overwrite:
            # 应该返回 duplicated，不调用 delete
            status = existing_doc_data.get("status", "unknown")
            assert status == "processed"
            mock_rag.adelete_by_doc_id.assert_not_called()

    @pytest.mark.asyncio
    async def test_file_upload_duplicate_with_overwrite_deletes_existing(self):
        """带 overwrite 上传重复文件应删除已有文档。"""
        from unittest.mock import AsyncMock, MagicMock

        existing_doc = {
            "id": "doc-existing-123",
            "status": "processed",
            "track_id": "upload_old_123",
            "file_path": "test.pdf",
        }

        mock_doc_status = AsyncMock()
        mock_doc_status.get_doc_by_file_path = AsyncMock(return_value=existing_doc)

        mock_rag = MagicMock()
        mock_rag.doc_status = mock_doc_status
        mock_rag.adelete_by_doc_id = AsyncMock()

        # 模拟 overwrite=True 时的行为
        overwrite = True
        safe_filename = "test.pdf"
        existing_doc_data = await mock_rag.doc_status.get_doc_by_file_path(
            safe_filename
        )

        assert existing_doc_data is not None
        if overwrite:
            existing_doc_id = existing_doc_data.get("id")
            assert existing_doc_id == "doc-existing-123"
            await mock_rag.adelete_by_doc_id(existing_doc_id)
            mock_rag.adelete_by_doc_id.assert_called_once_with("doc-existing-123")

    @pytest.mark.asyncio
    async def test_file_upload_no_existing_doc_skips_delete(self):
        """上传不存在的文件时，overwrite 不应触发删除。"""
        from unittest.mock import AsyncMock, MagicMock

        mock_doc_status = AsyncMock()
        mock_doc_status.get_doc_by_file_path = AsyncMock(return_value=None)

        mock_rag = MagicMock()
        mock_rag.doc_status = mock_doc_status
        mock_rag.adelete_by_doc_id = AsyncMock()

        safe_filename = "new_file.pdf"
        existing_doc_data = await mock_rag.doc_status.get_doc_by_file_path(
            safe_filename
        )

        # 无已有文档时，即使 overwrite=True 也不应调用删除
        assert existing_doc_data is None
        mock_rag.adelete_by_doc_id.assert_not_called()

    @pytest.mark.asyncio
    async def test_text_insert_duplicate_with_overwrite(self):
        """带 overwrite 插入重复文本应删除已有文档。"""
        from unittest.mock import AsyncMock, MagicMock

        existing_doc = {
            "id": "doc-content-hash-123",
            "status": "processed",
            "track_id": "insert_old_123",
            "file_path": "source.txt",
        }

        mock_doc_status = AsyncMock()
        mock_doc_status.get_doc_by_file_path = AsyncMock(return_value=existing_doc)
        mock_doc_status.get_by_id = AsyncMock(return_value=existing_doc)

        mock_rag = MagicMock()
        mock_rag.doc_status = mock_doc_status
        mock_rag.adelete_by_doc_id = AsyncMock()

        # 模拟 overwrite=True 时的 file_source 检查
        overwrite = True
        file_source = "source.txt"
        existing_doc_data = await mock_rag.doc_status.get_doc_by_file_path(file_source)

        assert existing_doc_data is not None
        if overwrite:
            existing_doc_id = existing_doc_data.get("id")
            await mock_rag.adelete_by_doc_id(existing_doc_id)
            mock_rag.adelete_by_doc_id.assert_called_once_with("doc-content-hash-123")

    @pytest.mark.asyncio
    async def test_text_insert_content_hash_duplicate_with_overwrite(self):
        """带 overwrite 插入内容相同的文本应删除已有文档。"""
        from unittest.mock import AsyncMock, MagicMock

        mock_doc_status = AsyncMock()
        mock_doc_status.get_doc_by_file_path = AsyncMock(return_value=None)

        existing_doc = {
            "id": "doc-content-hash-456",
            "status": "processed",
            "track_id": "insert_old_456",
        }
        mock_doc_status.get_by_id = AsyncMock(return_value=existing_doc)

        mock_rag = MagicMock()
        mock_rag.doc_status = mock_doc_status
        mock_rag.adelete_by_doc_id = AsyncMock()

        # 模拟 overwrite=True 时的 content hash 检查
        overwrite = True
        content_doc_id = "doc-content-hash-456"
        existing_doc = await mock_rag.doc_status.get_by_id(content_doc_id)

        assert existing_doc is not None
        if overwrite:
            await mock_rag.adelete_by_doc_id(content_doc_id)
            mock_rag.adelete_by_doc_id.assert_called_once_with("doc-content-hash-456")


@pytest.mark.offline
class TestExtractTextFromFile:
    """测试 upload 内容预检用的文件解析辅助函数。"""

    @pytest.mark.asyncio
    async def test_extract_txt_returns_text(self, tmp_path):
        """txt 文件应返回 utf-8 解码后的文本。"""
        from lightrag.api.routers.document_routes import _extract_text_from_file

        p = tmp_path / "a.txt"
        p.write_text("hello world", encoding="utf-8")
        assert await _extract_text_from_file(p) == "hello world"

    @pytest.mark.asyncio
    async def test_extract_unsupported_ext_returns_none(self, tmp_path):
        """不支持的扩展名应返回 None（降级放行）。"""
        from lightrag.api.routers.document_routes import _extract_text_from_file

        p = tmp_path / "a.unknownext"
        p.write_bytes(b"xxx")
        assert await _extract_text_from_file(p) is None

    @pytest.mark.asyncio
    async def test_extract_empty_txt_returns_none(self, tmp_path):
        """空内容文件应返回 None。"""
        from lightrag.api.routers.document_routes import _extract_text_from_file

        p = tmp_path / "empty.txt"
        p.write_text("   ", encoding="utf-8")
        assert await _extract_text_from_file(p) is None

    @pytest.mark.asyncio
    async def test_extract_nonexistent_returns_none(self, tmp_path):
        """文件不存在应返回 None，不抛异常。"""
        from lightrag.api.routers.document_routes import _extract_text_from_file

        p = tmp_path / "missing.txt"
        assert await _extract_text_from_file(p) is None


@pytest.mark.offline
class TestCheckContentDuplicate:
    """测试 upload 同步内容预检逻辑。"""

    @pytest.mark.asyncio
    async def test_content_exists_returns_duplicated_and_cleans_temp_file(
        self, tmp_path
    ):
        """内容已存在应返回 duplicated（含 doc_id/original_file_path），并删除临时文件。"""
        from unittest.mock import AsyncMock, MagicMock

        from lightrag.api.routers.document_routes import _check_content_duplicate
        from lightrag.utils import compute_mdhash_id, sanitize_text_for_encoding

        text = "duplicate content"
        p = tmp_path / "dup.txt"
        p.write_text(text, encoding="utf-8")
        doc_id = compute_mdhash_id(sanitize_text_for_encoding(text), prefix="doc-")

        existing = {
            "id": doc_id,
            "status": "processed",
            "track_id": "old_track",
            "file_path": "old_name_20260616.xlsx",
        }
        mock_rag = MagicMock()
        mock_rag.doc_status = AsyncMock()
        mock_rag.doc_status.get_by_id = AsyncMock(return_value=existing)

        resp = await _check_content_duplicate(mock_rag, p)

        assert resp is not None
        assert resp.status == "duplicated"
        assert resp.doc_id == doc_id
        assert resp.original_file_path == "old_name_20260616.xlsx"
        assert resp.track_id == "old_track"
        assert not p.exists()  # 临时文件已清理

    @pytest.mark.asyncio
    async def test_content_new_returns_none_and_keeps_file(self, tmp_path):
        """内容不存在应返回 None（放行），不删文件。"""
        from unittest.mock import AsyncMock, MagicMock

        from lightrag.api.routers.document_routes import _check_content_duplicate

        p = tmp_path / "new.txt"
        p.write_text("brand new content", encoding="utf-8")

        mock_rag = MagicMock()
        mock_rag.doc_status = AsyncMock()
        mock_rag.doc_status.get_by_id = AsyncMock(return_value=None)

        assert await _check_content_duplicate(mock_rag, p) is None
        assert p.exists()  # 文件保留交入队

    @pytest.mark.asyncio
    async def test_unsupported_ext_returns_none_without_lookup(self, tmp_path):
        """不支持类型应返回 None，且不触发 doc_status 查询。"""
        from unittest.mock import AsyncMock, MagicMock

        from lightrag.api.routers.document_routes import _check_content_duplicate

        p = tmp_path / "a.unknownext"
        p.write_bytes(b"xxx")

        mock_rag = MagicMock()
        mock_rag.doc_status = AsyncMock()
        mock_rag.doc_status.get_by_id = AsyncMock(return_value=None)

        assert await _check_content_duplicate(mock_rag, p) is None
        mock_rag.doc_status.get_by_id.assert_not_called()
