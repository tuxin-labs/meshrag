import asyncio
import hashlib
import json
import os
from collections import OrderedDict
from typing import Dict, List, Any, Callable, Optional, Tuple

from lightrag.lightrag import LightRAG
from lightrag.base import QueryParam, ProgressCallback
from lightrag.utils import logger
from lightrag.utils import (
    truncate_list_by_token_size,
    process_chunks_unified,
)
from lightrag.prompt import PROMPTS
from lightrag.constants import (
    DEFAULT_MAX_ENTITY_TOKENS,
    DEFAULT_MAX_RELATION_TOKENS,
    DEFAULT_MAX_TOTAL_TOKENS,
)


# 外部知识库虚拟 KB ID 前缀
EXTERNAL_KB_PREFIX = "__ext_"


class RAGManager:
    """Manages multiple LightRAG instances, each representing a distinct knowledge base."""

    def __init__(
        self,
        rag_factory: Callable[[str], LightRAG],
        default_kb: str = "default",
        max_instances: int = 50,
        registry_path: Optional[str] = None,
        kb_discovery: Optional[Callable[[], List[str]]] = None,
    ):
        # LRU cache: oldest at beginning, newest at end
        self._instances: "OrderedDict[str, LightRAG]" = OrderedDict()
        self._rag_factory = rag_factory
        self.default_kb = default_kb
        self.max_instances = max_instances
        self._registry_path = registry_path
        self._kb_discovery = kb_discovery
        self._known_kbs: set[str] = set()
        self._lock = asyncio.Lock()
        self._load_known_kbs()

    def _load_known_kbs(self) -> None:
        """Load KB ids from the registry and optional discovery source."""
        known_kbs: set[str] = {self.default_kb}

        if self._registry_path and os.path.exists(self._registry_path):
            try:
                with open(self._registry_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    known_kbs.update(
                        kb_id
                        for kb_id in data
                        if isinstance(kb_id, str) and kb_id.strip()
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to load KB registry '{self._registry_path}': {e}"
                )

        if self._kb_discovery is not None:
            try:
                known_kbs.update(
                    kb_id
                    for kb_id in self._kb_discovery()
                    if isinstance(kb_id, str) and kb_id.strip()
                )
            except Exception as e:
                logger.warning(f"Failed to discover persisted knowledge bases: {e}")

        self._known_kbs = known_kbs
        self._persist_known_kbs()

    def _persist_known_kbs(self) -> None:
        """Persist known KB ids so they survive process restarts."""
        if not self._registry_path:
            return

        try:
            registry_dir = os.path.dirname(self._registry_path)
            if registry_dir:
                os.makedirs(registry_dir, exist_ok=True)
            with open(self._registry_path, "w", encoding="utf-8") as f:
                json.dump(self._ordered_known_kbs(), f, ensure_ascii=True, indent=2)
        except Exception as e:
            logger.warning(
                f"Failed to persist KB registry '{self._registry_path}': {e}"
            )

    def _ordered_known_kbs(self) -> List[str]:
        """Keep the default KB first and make the rest stable."""
        remaining = sorted(
            kb_id for kb_id in self._known_kbs if kb_id != self.default_kb
        )
        if self.default_kb in self._known_kbs:
            return [self.default_kb, *remaining]
        return remaining

    def _touch_unlocked(self, kb_id: str) -> None:
        """Mark a kb as most recently used. Caller must hold _lock."""
        try:
            self._instances.move_to_end(kb_id, last=True)
        except KeyError:
            # Not present; nothing to touch
            return

    def _evict_one_unlocked(self) -> Optional[Tuple[str, LightRAG]]:
        """Pop least-recently used instance. Caller must hold _lock."""
        if not self._instances:
            return None
        kb_id, rag = self._instances.popitem(last=False)
        return kb_id, rag

    async def get_rag(self, kb_id: str) -> LightRAG:
        """
        Get or lazily create a LightRAG instance for the specified knowledge base.
        """
        if not kb_id:
            raise ValueError("kb_id cannot be empty")

        # Fast path
        if kb_id in self._instances:
            async with self._lock:
                # Re-check under lock and update LRU
                if kb_id in self._instances:
                    self._touch_unlocked(kb_id)
                    return self._instances[kb_id]

        async with self._lock:
            # Check again under lock
            if kb_id in self._instances:
                self._touch_unlocked(kb_id)
                return self._instances[kb_id]

            logger.info(f"Initializing new knowledge base: {kb_id}")
            rag = self._rag_factory(kb_id)
            await rag.initialize_storages()
            await rag.check_and_migrate_data()
            self._instances[kb_id] = rag
            self._known_kbs.add(kb_id)
            self._persist_known_kbs()
            self._touch_unlocked(kb_id)

            # LRU eviction: keep only most recently used max_instances
            evicted: List[Tuple[str, LightRAG]] = []
            while self.max_instances and len(self._instances) > self.max_instances:
                popped = self._evict_one_unlocked()
                if popped is None:
                    break
                evicted.append(popped)

        # Finalize evicted instances outside lock to reduce contention
        for evicted_kb_id, evicted_rag in evicted:
            try:
                logger.info(
                    f"Evicting least-recently used knowledge base: {evicted_kb_id} "
                    f"(cache_limit={self.max_instances})"
                )
                await evicted_rag.finalize_storages()
            except Exception as e:
                logger.error(f"Error finalizing evicted kb {evicted_kb_id}: {e}")

        return rag

    def list_knowledge_bases(self) -> List[str]:
        """List all known knowledge bases, not only the ones loaded in memory."""
        return self._ordered_known_kbs()

    async def delete_knowledge_base(
        self, kb_id: str, drop_storage: bool = True
    ) -> dict[str, any]:
        """
        Delete a knowledge base from memory and optionally drop its storage.

        Args:
            kb_id: Knowledge base ID to delete
            drop_storage: Whether to drop the storage data (default: True)

        Returns:
            dict[str, any]: Deletion result with status and details

        Raises:
            ValueError: If kb_id is empty or is the default KB and deletion is attempted
        """
        # Prevent deletion of default knowledge base
        if kb_id == self.default_kb:
            raise ValueError("Default knowledge base cannot be deleted")

        if not kb_id:
            raise ValueError("kb_id cannot be empty")

        async with self._lock:
            logger.info(
                f"Deleting knowledge base: {kb_id}, drop_storage={drop_storage}"
            )

            # Result object to track deletion process
            result = {
                "kb_id": kb_id,
                "status": "success",
                "message": "",
                "storage_results": None,
                "workspace_dir": None,
            }

            # ── 决定使用哪个 RAG 实例 ──
            # KB 在内存则直接复用；否则（服务重启 / LRU 淘汰后）创建临时实例，
            # 以确保 drop_storage=True 时持久化数据（向量 / 图谱 / 文档）被可靠清理。
            # 不能调用 get_rag()（会再次获取 self._lock 导致死锁），改用 _rag_factory。
            rag = None
            is_temp_instance = False
            if kb_id in self._instances:
                rag = self._instances[kb_id]
            elif drop_storage:
                try:
                    logger.info(
                        f"KB {kb_id} not in memory; creating temporary instance for drop"
                    )
                    rag = self._rag_factory(kb_id)
                    is_temp_instance = True
                except Exception as e:
                    # 工厂失败：符合 fire-and-forget，记录原因后继续清理 registry
                    result["status"] = "partial_success"
                    result["message"] = (
                        f"Knowledge base registry entry removed but storage cleanup "
                        f"failed: could not create RAG instance ({e})"
                    )
                    logger.error(
                        f"Failed to create temporary RAG instance for KB {kb_id}: {e}",
                        exc_info=True,
                    )
                    rag = None

            # ── 执行 drop（内存实例与临时实例共用同一路径）──
            if drop_storage and rag is not None:
                try:
                    result["storage_results"] = await rag.drop_storages()
                    failed = result["storage_results"].get("failed", [])
                    if failed:
                        result["status"] = "partial_success"
                        result["message"] = (
                            f"Knowledge base removed but {len(failed)} storage(s) "
                            f"failed to drop: {', '.join(failed)}"
                        )
                        # 详细记录每个失败存储的原因，供后续手动干预排查
                        for detail in result["storage_results"].get(
                            "failed_details", []
                        ):
                            logger.error(
                                f"KB {kb_id} storage '{detail.get('storage')}' "
                                f"failed to drop after retries: {detail.get('error')}"
                            )
                    else:
                        result["message"] = (
                            "Knowledge base and all storage data deleted successfully"
                        )
                except Exception as e:
                    result["status"] = "partial_success"
                    result["message"] = (
                        f"Knowledge base removed but storage drop failed: {e}"
                    )
                    logger.error(
                        f"Error dropping storage for KB {kb_id}: {e}", exc_info=True
                    )

            # ── 资源清理 ──
            if rag is not None:
                # drop_storages 内部已 finalize；此处对已 FINALIZED 实例为 no-op，
                # 仅作为防御性兜底（drop 过程抛异常时确保连接释放）。
                try:
                    await rag.finalize_storages()
                except Exception as e:
                    logger.warning(f"Error finalizing storage for KB {kb_id}: {e}")

            if not is_temp_instance:
                # 内存实例：从缓存移除
                if kb_id in self._instances:
                    del self._instances[kb_id]
            else:
                logger.debug(
                    f"Temporary RAG instance for KB {kb_id} dropped; not cached"
                )

            # 从 registry 移除（无论内存 / 临时 / 工厂失败，只要在 known_kbs 就清理）
            if kb_id in self._known_kbs:
                self._known_kbs.remove(kb_id)
                self._persist_known_kbs()

            # 兜底消息（drop_storage=False 或工厂失败但 registry 已清理）
            if result["message"] == "":
                result["message"] = "Knowledge base removed from registry"

            return result

    async def initialize_default(self) -> None:
        """Optional: pre-load the default knowledge base."""
        await self.get_rag(self.default_kb)

    async def finalize_all(self) -> None:
        """Finalize all loaded knowledge bases."""
        async with self._lock:
            logger.info(f"Finalizing all {len(self._instances)} knowledge bases...")
            items = list(self._instances.items())
            self._instances.clear()

        for kb_id, rag in items:
            try:
                await rag.finalize_storages()
            except Exception as e:
                logger.error(f"Error finalizing kb {kb_id}: {e}")

    async def _fetch_all_external_kbs(
        self,
        query: str,
        external_kbs: list[dict],
    ) -> tuple[list[tuple[str, dict]], list[str]]:
        """并行请求所有外部知识库，返回虚拟 KB 结果列表和错误信息列表。

        每个 virtual_kb_id 格式为 "__ext_0__", "__ext_1__" 等。
        失败的外部 KB 返回空结果（不影响其他），同时收集错误信息。

        Args:
            query: 查询文本
            external_kbs: 外部知识库配置列表，每个包含 url, api_key?, top_k?

        Returns:
            (results, errors) 元组:
            - results: [(virtual_kb_id, result_dict), ...] 列表
            - errors: 错误信息字符串列表
        """
        from lightrag.api.external_kb_client import fetch_external_kb

        errors: list[str] = []

        async def _fetch_one(idx: int, config: dict) -> tuple[str, dict]:
            virtual_id = f"{EXTERNAL_KB_PREFIX}{idx}"
            url = config.get("url", "")
            api_key = config.get("api_key")
            top_k = config.get("top_k", 5)
            chunks, error_msg = await fetch_external_kb(
                query=query,
                url=url,
                api_key=api_key,
                top_k=top_k,
            )
            if error_msg:
                errors.append(error_msg)
                return (
                    virtual_id,
                    {
                        "status": "failure",
                        "data": {},
                        "metadata": {},
                    },
                )

            # 构建与本地 aquery_data() 返回格式一致的结果
            references = []
            seen_paths: dict[str, str] = {}
            for c in chunks:
                fp = c.get("file_path", "")
                if fp and fp not in seen_paths:
                    ref_id = str(len(seen_paths) + 1)
                    seen_paths[fp] = ref_id
                    references.append(
                        {
                            "reference_id": ref_id,
                            "file_path": fp,
                        }
                    )
                if fp:
                    c["reference_id"] = seen_paths[fp]

            return (
                virtual_id,
                {
                    "status": "success",
                    "data": {
                        "entities": [],
                        "relationships": [],
                        "chunks": chunks,
                        "references": references,
                    },
                    "metadata": {},
                },
            )

        tasks = [_fetch_one(i, cfg) for i, cfg in enumerate(external_kbs)]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return list(results), errors

    async def _fetch_rag_kbs(
        self,
        query: str,
        rag_kbs: list[dict],
    ) -> tuple[list[tuple[str, dict]], list[str]]:
        """并行请求所有 RAG 型外部知识库，返回答案列表和错误信息列表。

        Args:
            query: 查询文本
            rag_kbs: RAG 型外部知识库配置列表

        Returns:
            (results, errors) 元组:
            - results: [(source_url, {"answer": str, "references": list, "source": str}), ...]
            - errors: 错误信息字符串列表
        """
        from lightrag.api.external_kb_client import fetch_rag_kb

        errors: list[str] = []

        async def _fetch_one(config: dict) -> tuple[str, dict]:
            url = config.get("url", "")
            api_key = config.get("api_key")
            result, error_msg = await fetch_rag_kb(
                query=query,
                url=url,
                api_key=api_key,
            )
            if error_msg:
                errors.append(error_msg)
            return (url, result)

        tasks = [_fetch_one(cfg) for cfg in rag_kbs]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return list(results), errors

    @staticmethod
    def _rag_refs_to_standard(
        rag_references: list[dict], source_url: str
    ) -> list[dict]:
        """将外部 RAG 服务返回的引用转为标准 reference 格式。

        标准 reference 格式: {"reference_id": str, "file_path": str, "kb_id": str}
        与 _merge_kb_results 输出的 references 格式一致。
        """
        standard_refs = []
        for i, ref in enumerate(rag_references):
            if isinstance(ref, dict) and ref.get("file_path"):
                standard_refs.append(
                    {
                        "reference_id": f"rag_{i + 1}",
                        "file_path": ref["file_path"],
                        "kb_id": source_url,
                    }
                )
        return standard_refs

    @staticmethod
    def _build_external_answers(rag_results: list[tuple[str, dict]]) -> list[dict]:
        """从 RAG 型外部 KB 结果构建 external_answers 列表。"""
        return [
            {
                "source": s,
                "answer": a.get("answer", ""),
                "references": a.get("references", []),
            }
            for s, a in rag_results
        ]

    def _merge_rag_refs_into_data(
        self, merged_data: dict, rag_results: list[tuple[str, dict]]
    ) -> None:
        """将外部 RAG 服务的引用合并到 merged_data.references 中，分配不冲突的 reference_id。

        同时设置 merged_data["external_answers"]。
        供 multi_kb_get_data 和 multi_kb_query 复用。
        """
        # 构建 external_answers
        merged_data["external_answers"] = self._build_external_answers(rag_results)

        # 收集所有 RAG 引用
        all_rag_refs: list[dict] = []
        for source_url, answer_dict in rag_results:
            all_rag_refs.extend(
                self._rag_refs_to_standard(
                    answer_dict.get("references", []), source_url
                )
            )

        if not all_rag_refs:
            return

        # 找到现有引用的最大 reference_id，避免冲突
        existing_refs = merged_data.get("references", [])
        max_id = 0
        for ref in existing_refs:
            try:
                max_id = max(max_id, int(ref.get("reference_id", "0")))
            except (ValueError, TypeError):
                pass

        # 为 RAG 引用分配不冲突的 reference_id
        for i, ref in enumerate(all_rag_refs):
            ref["reference_id"] = str(max_id + i + 1)

        merged_data["references"] = existing_refs + all_rag_refs

    def _build_rag_context(self, rag_results: list[tuple[str, dict]]) -> str:
        """将多个 RAG 答案构建为参考上下文字符串，供 LLM prompt 使用。"""
        sections = []
        for source, answer_dict in rag_results:
            answer_text = answer_dict.get("answer", "")
            if answer_text:
                sections.append(f"[Source: {source}]\n{answer_text}")
        if not sections:
            return ""
        header = "Reference Answers from External RAG Services (use as supplementary reference):"
        return header + "\n\n" + "\n\n---\n\n".join(sections)

    async def _merge_rag_answers(
        self,
        query: str,
        rag_results: list[tuple[str, dict]],
        param: QueryParam,
        kb_ids: List[str],
    ) -> dict:
        """多个 RAG 答案通过 LLM 合并为最终答案。"""
        rag_context = self._build_rag_context(rag_results)
        if not rag_context:
            return {
                "status": "failure",
                "message": "No valid RAG answers to merge",
                "data": {},
                "llm_response": {
                    "content": PROMPTS["fail_response"],
                    "is_streaming": False,
                    "response_iterator": None,
                },
            }

        response_type = (
            param.response_type if param.response_type else "Multiple Paragraphs"
        )
        sys_prompt = PROMPTS["rag_answers_merge"].format(
            source_answers=rag_context,
            response_type=response_type,
        )

        # 获取 LLM 函数
        if kb_ids:
            first_rag = await self.get_rag(kb_ids[0])
        else:
            first_rag = await self.get_rag(self.default_kb)
        use_model_func = (
            param.model_func if param.model_func else first_rag.llm_model_func
        )

        response = await use_model_func(
            query,
            system_prompt=sys_prompt,
            history_messages=param.conversation_history,
            enable_cot=True,
            stream=param.stream,
        )

        # 构建 external_answers 数据 + 合并引用
        ext_answers = [
            {
                "source": s,
                "answer": a.get("answer", ""),
                "references": a.get("references", []),
            }
            for s, a in rag_results
        ]
        # 将所有 RAG 引用合并为标准 references，并统一重编号避免跨服务 reference_id 重复
        all_rag_refs = []
        ref_counter = 1
        for source_url, answer_dict in rag_results:
            raw_refs = self._rag_refs_to_standard(
                answer_dict.get("references", []), source_url
            )
            for ref in raw_refs:
                ref["reference_id"] = str(ref_counter)
                ref_counter += 1
                all_rag_refs.append(ref)

        if param.stream:
            return {
                "status": "success",
                "message": "Merged RAG answers via LLM",
                "data": {"external_answers": ext_answers, "references": all_rag_refs},
                "metadata": {
                    "query_mode": param.mode,
                    "sources": [s for s, _ in rag_results],
                },
                "llm_response": {
                    "content": "",
                    "is_streaming": True,
                    "response_iterator": response,
                },
            }
        else:
            return {
                "status": "success",
                "message": "Merged RAG answers via LLM",
                "data": {"external_answers": ext_answers, "references": all_rag_refs},
                "metadata": {
                    "query_mode": param.mode,
                    "sources": [s for s, _ in rag_results],
                },
                "llm_response": {
                    "content": str(response),
                    "is_streaming": False,
                    "response_iterator": None,
                },
            }

    async def _bypass_llm(
        self,
        query: str,
        param: QueryParam,
        system_prompt: str | None,
        kb_ids: List[str],
    ) -> dict:
        """Bypass 模式：跳过所有检索，直接调 LLM。"""
        from functools import partial

        if kb_ids:
            first_rag = await self.get_rag(kb_ids[0])
        else:
            first_rag = await self.get_rag(self.default_kb)
        use_llm_func = param.model_func or first_rag.llm_model_func
        # _priority=8：与单 KB bypass 路径 (lightrag.py) 保持一致，确保高优先级
        use_llm_func = partial(use_llm_func, _priority=8)

        response = await use_llm_func(
            query,
            system_prompt=system_prompt,
            history_messages=param.conversation_history,
            enable_cot=True,
            stream=param.stream,
        )

        if param.stream:
            return {
                "status": "success",
                "message": "Bypass mode response",
                "data": {},
                "metadata": {"query_mode": "bypass"},
                "llm_response": {
                    "content": "",
                    "is_streaming": True,
                    "response_iterator": response,
                },
            }
        else:
            return {
                "status": "success",
                "message": "Bypass mode response",
                "data": {},
                "metadata": {"query_mode": "bypass"},
                "llm_response": {
                    "content": str(response),
                    "is_streaming": False,
                    "response_iterator": None,
                },
            }

    async def multi_kb_get_data(
        self,
        query: str,
        kb_ids: List[str],
        param: QueryParam,
        external_kbs: list[dict] | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> Dict[str, Any]:
        """
        Concurrently retrieve structured data from multiple knowledge bases and merge them.
        Supports optional external knowledge bases that are fetched in parallel.
        仅处理 retrieval 型外部 KB。rag 型外部 KB 的答案追加到 data.external_answers。
        """
        # 按 type 分组外部 KB
        retrieval_kbs = [
            kb
            for kb in (external_kbs or [])
            if kb.get("type", "retrieval") == "retrieval"
        ]
        rag_kbs = [kb for kb in (external_kbs or []) if kb.get("type") == "rag"]

        if not kb_ids and not retrieval_kbs and not rag_kbs:
            raise ValueError("At least one kb_id or external_kbs must be specified.")

        has_external = bool(retrieval_kbs)

        # 如果只有外部知识库（无本地 KB），直接走合并流程
        if not kb_ids:
            kb_ids = []

        # 并行获取所有数据：本地 KB + retrieval 型外部 KB
        local_coros = []
        local_kb_ids = list(kb_ids)
        rags = []

        for kb_id in local_kb_ids:
            rag = await self.get_rag(kb_id)
            rags.append(rag)
            local_coros.append(rag.aquery_data(query, param, on_progress=on_progress))

        # 并行执行本地检索和 retrieval 型外部检索
        external_results: list[tuple[str, dict]] = []
        ext_errors: list[str] = []
        if has_external:
            local_results_raw, (external_results, ext_errors) = await asyncio.gather(
                asyncio.gather(*local_coros, return_exceptions=True)
                if local_coros
                else asyncio.gather(),
                self._fetch_all_external_kbs(query, retrieval_kbs),
            )
        else:
            local_results_raw = (
                await asyncio.gather(*local_coros, return_exceptions=True)
                if local_coros
                else []
            )

        # 构建合并输入
        kb_results = list(zip(local_kb_ids, local_results_raw))
        kb_results.extend(external_results)

        # 如果只有一个数据源且无外部 KB 且无 RAG 型外部 KB，走快速路径
        if len(kb_results) == 1 and not has_external and not rag_kbs:
            return local_results_raw[0]

        merged_data = self._merge_kb_results(kb_results)

        # Post-process merged data: token truncation, cross-KB rerank, reference generation
        # 当只有外部 KB（无本地 RAG 实例）时，使用默认 KB 的 RAG 实例进行后处理
        rag_for_post = rags[0] if rags else await self.get_rag(self.default_kb)
        merged_data, context_str = await self._multi_kb_post_process(
            merged_data=merged_data,
            mode=param.mode,
            query=query,
            query_param=param,
            rag_instance=rag_for_post,
        )

        # Check for empty results (consistent with single-KB aquery_data behavior)
        # 当有 RAG 型外部 KB 时，即使本地检索为空也不算失败（RAG 答案可能有效）
        has_entities = bool(merged_data.get("entities"))
        has_relations = bool(merged_data.get("relationships"))
        has_chunks = bool(merged_data.get("chunks"))
        if not has_entities and not has_relations and not has_chunks and not rag_kbs:
            return {
                "status": "failure",
                "message": "Query returned no results across all knowledge bases",
                "data": {},
                "metadata": {
                    "failure_reason": "no_results",
                    "mode": param.mode,
                    "kb_ids": kb_ids,
                },
            }

        # Aggregate keywords from each KB result
        all_hl_keywords = []
        all_ll_keywords = []
        total_entities_before_merge = 0
        total_relations_before_merge = 0
        retrieved_chunks_before_merge = 0
        for kb_id, res in kb_results:
            if (
                isinstance(res, Exception)
                or not isinstance(res, dict)
                or res.get("status") != "success"
            ):
                continue
            kb_meta = res.get("metadata", {})
            kb_keywords = kb_meta.get("keywords", {})
            if kb_keywords.get("high_level"):
                all_hl_keywords.extend(kb_keywords["high_level"])
            if kb_keywords.get("low_level"):
                all_ll_keywords.extend(kb_keywords["low_level"])
            kb_proc = kb_meta.get("processing_info", {})
            total_entities_before_merge += kb_proc.get("total_entities_found", 0)
            total_relations_before_merge += kb_proc.get("total_relations_found", 0)
            # Retrieved chunks: try multiple field names across modes, fall back to actual data
            if "merged_chunks_count" in kb_proc:
                retrieved_chunks_before_merge += kb_proc["merged_chunks_count"]
            elif "total_chunks_found" in kb_proc:
                retrieved_chunks_before_merge += kb_proc["total_chunks_found"]
            else:
                retrieved_chunks_before_merge += len(
                    res.get("data", {}).get("chunks", [])
                )

        # Deduplicate keywords
        all_hl_keywords = list(dict.fromkeys(all_hl_keywords))
        all_ll_keywords = list(dict.fromkeys(all_ll_keywords))

        # Build metadata consistent with single-KB aquery_data
        metadata = {
            "query_mode": param.mode,
            "kb_ids": kb_ids,
            "merged_context": context_str,
            "keywords": {
                "high_level": all_hl_keywords,
                "low_level": all_ll_keywords,
            },
            "processing_info": {
                "total_entities_found": total_entities_before_merge,
                "total_relations_found": total_relations_before_merge,
                "retrieved_chunks_before_merge": retrieved_chunks_before_merge,
                "entities_after_truncation": len(merged_data.get("entities", [])),
                "relations_after_truncation": len(merged_data.get("relationships", [])),
                "merged_chunks_count": len(merged_data.get("chunks", [])),
                "final_chunks_count": len(merged_data.get("chunks", [])),
            },
        }

        # 获取 RAG 型外部 KB 答案并追加到 data
        rag_errors: list[str] = []
        if rag_kbs:
            rag_results, rag_errors = await self._fetch_rag_kbs(query, rag_kbs)
            self._merge_rag_refs_into_data(merged_data, rag_results)

        # 合并所有外部 KB 错误信息
        all_ext_errors = ext_errors + rag_errors
        if all_ext_errors:
            metadata["external_kb_errors"] = all_ext_errors

        return {
            "status": "success",
            "message": "Merged query executed successfully",
            "data": merged_data,
            "metadata": metadata,
        }

    async def multi_kb_query(
        self,
        query: str,
        kb_ids: List[str],
        param: QueryParam,
        system_prompt: str | None = None,
        external_kbs: list[dict] | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> Any:
        """
        并发查询多个知识库并合并结果，支持两种外部 KB 类型：
        - retrieval 型：返回文本块，参与本地合并 rerank 流程
        - rag 型：返回完整答案，作为参考上下文或直接转发

        处理路径：
        A. Bypass → 跳过所有检索，直接调 LLM
        B. 仅内部 KB → 当前快速路径 / 合并路径
        C. 仅外部 RAG 型 → 单个直接返回，多个 LLM 合并
        D. 仅外部 retrieval 型 → 获取 chunks → LLM 回答
        E. 仅外部混合（retrieval + rag） → chunks + RAG 答案 → LLM
        F. 内部 + 外部 retrieval → 合并检索 → LLM
        G. 内部 + 外部 rag → 内部检索 + RAG 答案 → LLM
        H. 内部 + 外部 retrieval + rag → 全合并 → LLM

        返回契约（所有路径统一）：
            {
                "status": "success" | "failure",
                "message": str,
                "data": {
                    "entities": [...],
                    "relationships": [...],
                    "chunks": [...],
                    "references": [...],
                    "external_answers": [{"source": str, "answer": str, "references": list}, ...],  # rag 型外部 KB 时存在
                },
                "metadata": dict,
                "llm_response": {
                    "content": str,             # 非 stream 时为答案文本，stream 时为空
                    "is_streaming": bool,
                    "response_iterator": ... | None,
                },
            }
        注意：此返回格式与路由层（query_routes.py）约定一致，修改时需确保兼容。
        """
        if not kb_ids and not external_kbs:
            raise ValueError("At least one kb_id or external_kbs must be specified.")

        # ─── Step 1: 按 type 分组外部 KB ───
        retrieval_kbs = [
            kb
            for kb in (external_kbs or [])
            if kb.get("type", "retrieval") == "retrieval"
        ]
        rag_kbs = [kb for kb in (external_kbs or []) if kb.get("type") == "rag"]
        has_local = bool(kb_ids)
        has_ext_retrieval = bool(retrieval_kbs)
        has_ext_rag = bool(rag_kbs)
        has_any_ext = has_ext_retrieval or has_ext_rag
        has_retrieval_data = has_local or has_ext_retrieval  # 有"可检索数据源"

        # ─── 场景 A: Bypass — 跳过所有检索 ───
        if param.mode == "bypass":
            if has_any_ext:
                logger.warning(
                    "bypass mode ignores all knowledge bases including external_kbs"
                )
            return await self._bypass_llm(query, param, system_prompt, kb_ids)

        # ─── 场景 B: 仅内部 KB，无外部（当前快速路径） ───
        if has_local and not has_any_ext:
            if len(kb_ids) == 1:
                rag = await self.get_rag(kb_ids[0])
                result = await rag.aquery_llm(
                    query,
                    param=param,
                    system_prompt=system_prompt,
                    on_progress=on_progress,
                )
                if isinstance(result, dict) and result.get("status") == "failure":
                    llm_content = result.get("llm_response", {}).get("content")
                    if not llm_content:
                        raise Exception(result.get("message", "Query failed"))
                return result
            # 多内部 KB 继续走下面的通用混合路径

        # ─── 场景 C: 仅外部 RAG 型（无本地/retrieval 数据源） ───
        all_ext_errors: list[str] = []
        if has_ext_rag and not has_retrieval_data:
            rag_results, rag_errors = await self._fetch_rag_kbs(query, rag_kbs)
            all_ext_errors.extend(rag_errors)
            valid = [(s, a) for s, a in rag_results if a.get("answer")]
            if not valid:
                return {
                    "status": "failure",
                    "message": "All external RAG services returned empty results",
                    "data": {},
                    "metadata": {"external_kb_errors": all_ext_errors}
                    if all_ext_errors
                    else {},
                    "llm_response": {
                        "content": PROMPTS["fail_response"],
                        "is_streaming": False,
                        "response_iterator": None,
                    },
                }
            if len(valid) == 1:
                # 单个 RAG：直接返回外部答案
                _, answer_dict = valid[0]
                # 将 RAG 引用转为标准 reference 格式
                rag_refs = self._rag_refs_to_standard(
                    answer_dict.get("references", []), valid[0][0]
                )
                metadata = {"query_mode": param.mode, "sources": [valid[0][0]]}
                if all_ext_errors:
                    metadata["external_kb_errors"] = all_ext_errors
                return {
                    "status": "success",
                    "message": "External RAG service response",
                    "data": {"references": rag_refs},
                    "metadata": metadata,
                    "llm_response": {
                        "content": answer_dict["answer"],
                        "is_streaming": False,
                        "response_iterator": None,
                    },
                }
            # 多个 RAG：LLM 合并
            return await self._merge_rag_answers(query, valid, param, kb_ids)

        # ─── 通用混合路径（场景 D/E/F/G/H） ───
        # 并行获取 retrieval 数据 + RAG 答案
        tasks = []
        if has_retrieval_data:
            tasks.append(
                self.multi_kb_get_data(
                    query,
                    kb_ids,
                    param,
                    external_kbs=retrieval_kbs if has_ext_retrieval else None,
                    on_progress=on_progress,
                )
            )
        if has_ext_rag:
            tasks.append(self._fetch_rag_kbs(query, rag_kbs))

        results = await asyncio.gather(*tasks)

        # 解包结果
        data_res = None
        rag_results: list[tuple[str, dict]] = []
        rag_fetch_errors: list[str] = []
        result_idx = 0
        if has_retrieval_data:
            data_res = results[result_idx]
            result_idx += 1
        if has_ext_rag:
            rag_results, rag_fetch_errors = results[result_idx]
            all_ext_errors.extend(rag_fetch_errors)

        # 从 data_res 的 metadata 中提取 retrieval 型外部 KB 的错误信息
        if data_res and isinstance(data_res, dict):
            retrieval_errors = data_res.get("metadata", {}).get(
                "external_kb_errors", []
            )
            all_ext_errors.extend(retrieval_errors)

        # 检查 retrieval 数据并构建上下文
        context_str = ""
        merged_data: dict = {}
        # data_res 可能是异常对象（embedding 失败等场景），需防御性检查
        is_valid_data = isinstance(data_res, dict)
        if has_retrieval_data:
            if is_valid_data and data_res.get("status") == "success":
                merged_data = data_res.get("data", {})
                context_str = data_res.get("metadata", {}).get("merged_context", "")
            elif not has_ext_rag:
                # 无 retrieval 数据且无 RAG 兜底 → 返回失败
                fail_metadata = data_res.get("metadata", {}) if is_valid_data else {}
                if all_ext_errors:
                    fail_metadata["external_kb_errors"] = all_ext_errors
                if not is_valid_data:
                    # data_res 是异常对象，提取错误信息
                    err_msg = f"Retrieval failed: {data_res}" if data_res else "No data"
                    return {
                        "status": "failure",
                        "message": err_msg,
                        "data": {},
                        "metadata": fail_metadata,
                        "llm_response": {
                            "content": PROMPTS["fail_response"],
                            "is_streaming": False,
                            "response_iterator": None,
                        },
                    }
                return {
                    "status": data_res.get("status", "failure"),
                    "message": data_res.get("message", "Query returned no results"),
                    "data": {},
                    "metadata": fail_metadata,
                    "llm_response": {
                        "content": PROMPTS["fail_response"],
                        "is_streaming": False,
                        "response_iterator": None,
                    },
                }
            # else: retrieval 失败但有 RAG 答案兜底，继续

        # 追加 RAG 答案到上下文
        if has_ext_rag and rag_results:
            rag_context = self._build_rag_context(rag_results)
            if rag_context:
                context_str = (
                    (context_str + "\n\n" + rag_context) if context_str else rag_context
                )
            # 合并 RAG 引用到 merged_data（含 external_answers + references 重编号）
            self._merge_rag_refs_into_data(merged_data, rag_results)

        # 最终检查上下文是否为空
        if not context_str:
            fail_metadata = {}
            if all_ext_errors:
                fail_metadata["external_kb_errors"] = all_ext_errors
            return {
                "status": "failure",
                "message": "Query returned no results from any source",
                "data": merged_data,
                "metadata": fail_metadata,
                "llm_response": {
                    "content": PROMPTS["fail_response"],
                    "is_streaming": False,
                    "response_iterator": None,
                },
            }

        # ─── 调用 LLM 生成最终答案 ───
        sys_prompt_temp = (
            PROMPTS["multi_kb_rag_response"]
            if param.mode != "naive"
            else PROMPTS["multi_kb_naive_rag_response"]
        )
        response_type = (
            param.response_type if param.response_type else "Multiple Paragraphs"
        )
        user_prompt = f"\n\n{param.user_prompt}" if param.user_prompt else "n/a"

        if param.mode != "naive":
            sys_prompt = sys_prompt_temp.format(
                response_type=response_type,
                user_prompt=user_prompt,
                context_data=context_str,
            )
        else:
            sys_prompt = sys_prompt_temp.format(
                response_type=response_type,
                user_prompt=user_prompt,
                content_data=context_str,
            )

        # 获取 LLM 函数
        if kb_ids:
            first_rag = await self.get_rag(kb_ids[0])
        else:
            first_rag = await self.get_rag(self.default_kb)
        use_model_func = (
            param.model_func if param.model_func else first_rag.llm_model_func
        )

        response = await use_model_func(
            query,
            system_prompt=sys_prompt,
            history_messages=param.conversation_history,
            enable_cot=True,
            stream=param.stream,
        )

        # 透传 multi_kb_get_data() 已构建的完整 metadata
        final_metadata = (
            data_res.get("metadata", {})
            if data_res and isinstance(data_res, dict)
            else {}
        )
        final_metadata["query_mode"] = param.mode
        if all_ext_errors:
            # 合并，避免覆盖 multi_kb_get_data 已设置的 external_kb_errors
            existing_errors = final_metadata.get("external_kb_errors", [])
            final_metadata["external_kb_errors"] = existing_errors + [
                e for e in all_ext_errors if e not in existing_errors
            ]

        if param.stream:
            return {
                "status": "success",
                "message": "Query executed successfully",
                "data": merged_data,
                "metadata": final_metadata,
                "llm_response": {
                    "content": "",
                    "is_streaming": True,
                    "response_iterator": response,
                },
            }
        else:
            return {
                "status": "success",
                "message": "Query executed successfully",
                "data": merged_data,
                "metadata": final_metadata,
                "llm_response": {
                    "content": str(response),
                    "is_streaming": False,
                    "response_iterator": None,
                },
            }

    def _merge_kb_results(self, kb_results: List[tuple]) -> Dict[str, Any]:
        """Merge retrieval results from multiple KBs into structured data."""
        merged_chunks = []
        merged_references = []

        # Merge tracking ( entity_name -> merged entity, sorted(src,tgt) -> merged relation)
        entity_map: dict[str, dict] = {}
        relation_map: dict[tuple, dict] = {}
        seen_chunks = set()
        reference_key_to_id: dict[tuple[str, str], str] = {}
        next_reference_id = 1
        # Track which references are actually used
        used_reference_keys = set()

        for kb_id, res in kb_results:
            if isinstance(res, Exception):
                logger.error(f"Error querying KB {kb_id}: {res}")
                continue
            if not isinstance(res, dict) or res.get("status") != "success":
                continue

            data = res.get("data", {})
            local_reference_map: dict[str, str] = {}

            for ref in data.get("references", []):
                file_path = ref.get("file_path", "")
                local_ref_id = ref.get("reference_id", "")
                if not file_path:
                    continue

                ref_key = (kb_id, file_path)
                if ref_key not in reference_key_to_id:
                    reference_key_to_id[ref_key] = str(next_reference_id)
                    next_reference_id += 1
                    ref_copy = ref.copy()
                    ref_copy["reference_id"] = reference_key_to_id[ref_key]
                    ref_copy["kb_id"] = kb_id
                    merged_references.append(ref_copy)

                if local_ref_id:
                    local_reference_map[local_ref_id] = reference_key_to_id[ref_key]

            # Merge entities (by entity_name, preserve supplementary evidence from multiple KBs)
            for e in data.get("entities", []):
                e_name = e.get("entity_name")
                if not e_name:
                    continue

                # Track used reference (always, even for duplicates)
                ref_id = e.get("reference_id", "")
                if ref_id and ref_id in local_reference_map:
                    original_ref = next(
                        (
                            r
                            for r in data.get("references", [])
                            if r.get("reference_id") == ref_id
                        ),
                        None,
                    )
                    if original_ref:
                        used_reference_keys.add(
                            (kb_id, original_ref.get("file_path", ""))
                        )

                if e_name not in entity_map:
                    # First occurrence: keep full entity with kb_ids and evidence tracking
                    e_copy = e.copy()
                    e_copy["kb_ids"] = [kb_id]
                    if ref_id and ref_id in local_reference_map:
                        e_copy["reference_id"] = local_reference_map[ref_id]
                    e_copy["evidence"] = [
                        {
                            "kb_id": kb_id,
                            "reference_id": e_copy.get("reference_id", ""),
                            "description": e.get("description", ""),
                        }
                    ]
                    entity_map[e_name] = e_copy
                else:
                    # Duplicate: merge supplementary evidence
                    existing = entity_map[e_name]

                    # Map duplicate's local reference_id to global
                    mapped_ref_id = ""
                    if ref_id and ref_id in local_reference_map:
                        mapped_ref_id = local_reference_map[ref_id]

                    new_desc = e.get("description", "")
                    old_desc = existing.get("description", "")
                    # description: keep longer non-empty; reference_id follows description
                    if new_desc and len(new_desc) > len(old_desc):
                        existing["description"] = new_desc
                        if mapped_ref_id:
                            existing["reference_id"] = mapped_ref_id
                    # entity_type: prefer non-empty if current is empty
                    if not existing.get("entity_type") and e.get("entity_type"):
                        existing["entity_type"] = e["entity_type"]
                    # kb_ids: append unique sources
                    if kb_id not in existing["kb_ids"]:
                        existing["kb_ids"].append(kb_id)
                    # source_count
                    existing["source_count"] = existing.get("source_count", 1) + 1
                    # Append evidence entry for this duplicate
                    existing.setdefault("evidence", []).append(
                        {
                            "kb_id": kb_id,
                            "reference_id": mapped_ref_id,
                            "description": new_desc,
                        }
                    )

            # Merge relations (undirected edge dedup, preserve supplementary evidence)
            for r in data.get("relationships", []):
                src_id = r.get("src_id")
                tgt_id = r.get("tgt_id")
                if not src_id or not tgt_id:
                    continue
                r_key = tuple(sorted([str(src_id), str(tgt_id)]))

                # Track used reference (always, even for duplicates)
                ref_id = r.get("reference_id", "")
                if ref_id and ref_id in local_reference_map:
                    original_ref = next(
                        (
                            rr
                            for rr in data.get("references", [])
                            if rr.get("reference_id") == ref_id
                        ),
                        None,
                    )
                    if original_ref:
                        used_reference_keys.add(
                            (kb_id, original_ref.get("file_path", ""))
                        )

                if r_key not in relation_map:
                    # First occurrence: keep full relation with kb_ids and evidence tracking
                    r_copy = r.copy()
                    r_copy["kb_ids"] = [kb_id]
                    if ref_id and ref_id in local_reference_map:
                        r_copy["reference_id"] = local_reference_map[ref_id]
                    r_copy["evidence"] = [
                        {
                            "kb_id": kb_id,
                            "reference_id": r_copy.get("reference_id", ""),
                            "description": r.get("description", ""),
                            "weight": r.get("weight"),
                        }
                    ]
                    relation_map[r_key] = r_copy
                else:
                    # Duplicate: merge supplementary evidence
                    existing = relation_map[r_key]

                    # Map duplicate's local reference_id to global
                    mapped_ref_id = ""
                    if ref_id and ref_id in local_reference_map:
                        mapped_ref_id = local_reference_map[ref_id]

                    new_desc = r.get("description", "")
                    old_desc = existing.get("description", "")
                    # description: keep longer non-empty; reference_id follows description
                    if new_desc and len(new_desc) > len(old_desc):
                        existing["description"] = new_desc
                        if mapped_ref_id:
                            existing["reference_id"] = mapped_ref_id
                    # keywords: prefer non-empty if current is empty
                    if not existing.get("keywords") and r.get("keywords"):
                        existing["keywords"] = r["keywords"]
                    # weight: take max (stronger evidence)
                    new_weight = r.get("weight")
                    if new_weight is not None:
                        old_weight = existing.get("weight")
                        if old_weight is None or new_weight > old_weight:
                            existing["weight"] = new_weight
                    # kb_ids: append unique sources
                    if kb_id not in existing["kb_ids"]:
                        existing["kb_ids"].append(kb_id)
                    # source_count
                    existing["source_count"] = existing.get("source_count", 1) + 1
                    # Append evidence entry for this duplicate
                    existing.setdefault("evidence", []).append(
                        {
                            "kb_id": kb_id,
                            "reference_id": mapped_ref_id,
                            "description": new_desc,
                            "weight": new_weight,
                        }
                    )

            # Merge chunks (include kb_id in dedup key)
            for c in data.get("chunks", []):
                c_id = c.get("chunk_id", "")
                c_key = (kb_id, c_id)
                if c_id and c_key not in seen_chunks:
                    seen_chunks.add(c_key)
                    chunk_copy = c.copy()
                    chunk_copy["kb_id"] = kb_id
                    ref_id = chunk_copy.get("reference_id", "")
                    if ref_id in local_reference_map:
                        chunk_copy["reference_id"] = local_reference_map[ref_id]
                        # Find the reference key
                        original_ref = next(
                            (
                                r
                                for r in data.get("references", [])
                                if r.get("reference_id") == ref_id
                            ),
                            None,
                        )
                        if original_ref:
                            used_reference_keys.add(
                                (kb_id, original_ref.get("file_path", ""))
                            )
                    merged_chunks.append(chunk_copy)

        # Convert merge maps to lists
        merged_entities = list(entity_map.values())
        merged_relations = list(relation_map.values())

        # Now filter references to only keep those actually used
        # And re-number reference_ids for continuity
        filtered_references = []
        old_ref_id_to_new: dict[str, str] = {}
        new_ref_id_counter = 1

        for ref in merged_references:
            ref_key = (ref.get("kb_id", ""), ref.get("file_path", ""))
            if ref_key in used_reference_keys:
                old_ref_id = ref["reference_id"]
                new_ref_id = str(new_ref_id_counter)
                old_ref_id_to_new[old_ref_id] = new_ref_id
                ref_copy = ref.copy()
                ref_copy["reference_id"] = new_ref_id
                filtered_references.append(ref_copy)
                new_ref_id_counter += 1

        merged_references = filtered_references

        # Update reference_ids in entities, relations, and chunks
        for e in merged_entities:
            old_ref_id = e.get("reference_id", "")
            if old_ref_id and old_ref_id in old_ref_id_to_new:
                e["reference_id"] = old_ref_id_to_new[old_ref_id]

        for r in merged_relations:
            old_ref_id = r.get("reference_id", "")
            if old_ref_id and old_ref_id in old_ref_id_to_new:
                r["reference_id"] = old_ref_id_to_new[old_ref_id]

        for c in merged_chunks:
            old_ref_id = c.get("reference_id", "")
            if old_ref_id and old_ref_id in old_ref_id_to_new:
                c["reference_id"] = old_ref_id_to_new[old_ref_id]

        # Update reference_ids inside evidence entries
        for e in merged_entities:
            for ev in e.get("evidence", []):
                old_ref_id = ev.get("reference_id", "")
                if old_ref_id and old_ref_id in old_ref_id_to_new:
                    ev["reference_id"] = old_ref_id_to_new[old_ref_id]

        for r in merged_relations:
            for ev in r.get("evidence", []):
                old_ref_id = ev.get("reference_id", "")
                if old_ref_id and old_ref_id in old_ref_id_to_new:
                    ev["reference_id"] = old_ref_id_to_new[old_ref_id]

        merged_data = {
            "entities": merged_entities,
            "relationships": merged_relations,
            "chunks": merged_chunks,
            "references": merged_references,
        }

        return merged_data

    def _content_dedup_chunks(self, chunks: List[dict]) -> List[dict]:
        """Deduplicate chunks across KBs by normalized content hash.

        Handles the common case where identical documents are indexed in
        multiple KBs. Keeps the chunk with the most source information
        and merges kb_ids / reference_ids from duplicates so no KB's
        citation is silently lost.
        """
        seen: dict[str, dict] = {}
        result = []

        for chunk in chunks:
            content = chunk.get("content", "")
            content_hash = hashlib.md5(content.strip().lower().encode()).hexdigest()

            if content_hash not in seen:
                seen[content_hash] = chunk
                # Initialize reference_ids with the chunk's own reference_id
                ref_id = chunk.get("reference_id", "")
                chunk["reference_ids"] = [ref_id] if ref_id else []
                result.append(chunk)
            else:
                existing = seen[content_hash]
                dup_ref_id = chunk.get("reference_id", "")

                # Append duplicate's reference_id to surviving chunk's list (deduplicate)
                if dup_ref_id and dup_ref_id not in existing["reference_ids"]:
                    existing["reference_ids"].append(dup_ref_id)

                # Prefer chunk with more KB sources or longer content
                existing_kb_ids = existing.get("kb_ids", [existing.get("kb_id")])
                new_kb_ids = chunk.get("kb_ids", [chunk.get("kb_id")])
                if len(new_kb_ids) > len(existing_kb_ids):
                    # Swap: keep the new one, merge old's kb_ids and reference_ids
                    chunk["kb_ids"] = list(set(existing_kb_ids + new_kb_ids))
                    # Deduplicate while preserving order
                    merged_refs = existing.get("reference_ids", [])
                    if dup_ref_id and dup_ref_id not in merged_refs:
                        merged_refs.append(dup_ref_id)
                    chunk["reference_ids"] = merged_refs
                    result.remove(existing)
                    seen[content_hash] = chunk
                    result.append(chunk)
                else:
                    # Keep existing, merge new's kb_ids
                    existing["kb_ids"] = list(set(existing_kb_ids + new_kb_ids))

        return result

    async def _multi_kb_post_process(
        self,
        merged_data: Dict[str, Any],
        mode: str,
        query: str,
        query_param: QueryParam,
        rag_instance: "LightRAG",
    ) -> tuple[Dict[str, Any], str]:
        """Post-process merged multi-KB data with token budget control, rerank, and reference generation.

        Reuses the same pipeline stages as single-KB:
        1. Content-based chunk deduplication
        2. Entity/relation token truncation
        3. Dynamic chunk token budget calculation
        4. Cross-KB rerank + chunk_top_k + token truncation
        5. Frequency-based reference generation
        6. Final context string construction
        """
        from dataclasses import asdict

        # Build global_config from the first RAG instance (for tokenizer, reranker, etc.)
        global_config = asdict(rag_instance)
        global_config["tokenizer"] = rag_instance.tokenizer
        global_config["embedding_func"] = rag_instance.embedding_func
        if hasattr(rag_instance, "rerank_model_func"):
            global_config["rerank_model_func"] = rag_instance.rerank_model_func

        tokenizer = global_config.get("tokenizer")
        if not tokenizer:
            logger.warning("No tokenizer available, skipping multi-KB post-processing")
            # Fallback: build context without budget control
            return merged_data, self._build_fallback_context(merged_data, mode)

        # Internal fields to strip when building LLM context
        _multi_kb_keys = {"evidence", "kb_ids", "source_count"}

        # Step 1: Content-based chunk deduplication
        merged_data["chunks"] = self._content_dedup_chunks(merged_data["chunks"])

        # Step 2: Entity token truncation
        max_entity_tokens = getattr(
            query_param,
            "max_entity_tokens",
            global_config.get("max_entity_tokens", DEFAULT_MAX_ENTITY_TOKENS),
        )
        entities = merged_data.get("entities", [])
        if entities:
            # Use minimal schema matching single-KB's _apply_token_truncation
            # Only count entity name, type, description for token budget
            entities_for_truncation = [
                {
                    "entity": e.get("entity_name", ""),
                    "type": e.get("entity_type", "UNKNOWN"),
                    "description": e.get("description", ""),
                }
                for e in entities
            ]
            truncated_entities = truncate_list_by_token_size(
                entities_for_truncation,
                key=lambda x: json.dumps(x, ensure_ascii=False),
                max_token_size=max_entity_tokens,
                tokenizer=tokenizer,
            )
            # Filter original entities to match truncated set
            # truncated_entities uses {"entity": name} schema, not {"entity_name": name}
            truncated_names = {e.get("entity") for e in truncated_entities}
            merged_data["entities"] = [
                e for e in entities if e.get("entity_name") in truncated_names
            ]

        # Step 3: Relation token truncation
        max_relation_tokens = getattr(
            query_param,
            "max_relation_tokens",
            global_config.get("max_relation_tokens", DEFAULT_MAX_RELATION_TOKENS),
        )
        relations = merged_data.get("relationships", [])
        if relations:
            # Use minimal schema matching single-KB's _apply_token_truncation
            # Only count entity1, entity2, description for token budget
            relations_for_truncation = [
                {
                    "entity1": r.get("src_id", ""),
                    "entity2": r.get("tgt_id", ""),
                    "description": r.get("description", ""),
                }
                for r in relations
            ]
            truncated_relations = truncate_list_by_token_size(
                relations_for_truncation,
                key=lambda x: json.dumps(x, ensure_ascii=False),
                max_token_size=max_relation_tokens,
                tokenizer=tokenizer,
            )
            # Filter original relations to match truncated set
            # truncated_relations uses {"entity1": src, "entity2": tgt} schema
            truncated_keys = {
                tuple(sorted([r.get("entity1"), r.get("entity2")]))
                for r in truncated_relations
            }
            merged_data["relationships"] = [
                r
                for r in relations
                if tuple(sorted([r.get("src_id"), r.get("tgt_id")])) in truncated_keys
            ]

        # Step 4: Calculate dynamic chunk token budget
        max_total_tokens = getattr(
            query_param,
            "max_total_tokens",
            global_config.get("max_total_tokens", DEFAULT_MAX_TOTAL_TOKENS),
        )

        user_prompt = (
            f"\n\n{query_param.user_prompt}" if query_param.user_prompt else "n/a"
        )
        response_type = (
            query_param.response_type
            if query_param.response_type
            else "Multiple Paragraphs"
        )

        # Build preliminary entity/relation strings for overhead calculation
        entities_str = "\n".join(
            json.dumps(
                {k: v for k, v in e.items() if k not in _multi_kb_keys},
                ensure_ascii=False,
            )
            for e in merged_data.get("entities", [])
        )
        relations_str = "\n".join(
            json.dumps(
                {k: v for k, v in r.items() if k not in _multi_kb_keys},
                ensure_ascii=False,
            )
            for r in merged_data.get("relationships", [])
        )

        # Calculate KG context overhead (use multi-KB templates for accurate budget)
        if mode == "naive":
            kg_context_template = PROMPTS["multi_kb_naive_query_context"]
            pre_kg_context = kg_context_template.format(
                text_chunks_str="",
                reference_list_str="",
            )
        else:
            kg_context_template = PROMPTS["multi_kb_kg_query_context"]
            pre_kg_context = kg_context_template.format(
                entities_str=entities_str,
                relations_str=relations_str,
                text_chunks_str="",
                reference_list_str="",
            )
        kg_context_tokens = len(tokenizer.encode(pre_kg_context))

        # Calculate system prompt overhead (use multi-KB response templates for accurate budget)
        sys_prompt_template = (
            PROMPTS["multi_kb_naive_rag_response"]
            if mode == "naive"
            else PROMPTS["multi_kb_rag_response"]
        )
        if mode == "naive":
            pre_sys_prompt = sys_prompt_template.format(
                content_data="",
                response_type=response_type,
                user_prompt=user_prompt,
            )
        else:
            pre_sys_prompt = sys_prompt_template.format(
                context_data="",
                response_type=response_type,
                user_prompt=user_prompt,
            )
        sys_prompt_tokens = len(tokenizer.encode(pre_sys_prompt))

        # Calculate available tokens for chunks
        query_tokens = len(tokenizer.encode(query))
        buffer_tokens = 200
        available_chunk_tokens = max_total_tokens - (
            sys_prompt_tokens + kg_context_tokens + query_tokens + buffer_tokens
        )

        logger.info(
            f"Multi-KB token budget - Total: {max_total_tokens}, "
            f"SysPrompt: {sys_prompt_tokens}, Query: {query_tokens}, "
            f"KG: {kg_context_tokens}, Buffer: {buffer_tokens}, "
            f"Available for chunks: {available_chunk_tokens}"
        )

        # Step 4b: Cross-KB rerank + chunk_top_k + token truncation
        chunks = merged_data.get("chunks", [])
        if chunks:
            truncated_chunks = await process_chunks_unified(
                query=query,
                unique_chunks=chunks,
                query_param=query_param,
                global_config=global_config,
                source_type="multi_kb",
                chunk_token_limit=available_chunk_tokens,
            )
            merged_data["chunks"] = truncated_chunks

        # Step 5: Reference cleanup preserving (kb_id, file_path) dimension
        # Always run, even when chunks are empty, to sync references with surviving entities/relations.
        # Do NOT use generate_reference_list_from_chunks here 鈥?it groups only by file_path,
        # collapsing the kb_id dimension that _merge_kb_results carefully preserved.
        used_ref_ids = set()
        for e in merged_data.get("entities", []):
            rid = e.get("reference_id", "")
            if rid:
                used_ref_ids.add(rid)
        for r in merged_data.get("relationships", []):
            rid = r.get("reference_id", "")
            if rid:
                used_ref_ids.add(rid)
        for c in merged_data.get("chunks", []):
            rid = c.get("reference_id", "")
            if rid:
                used_ref_ids.add(rid)
            # Also preserve reference_ids collected during content dedup
            for extra_rid in c.get("reference_ids", []):
                if extra_rid:
                    used_ref_ids.add(extra_rid)

        # Filter references to only those still used, then re-number for continuity
        original_refs = merged_data.get("references", [])
        filtered_refs = []
        old_to_new: dict[str, str] = {}
        counter = 1
        for ref in original_refs:
            if ref.get("reference_id") in used_ref_ids:
                old_id = ref["reference_id"]
                new_id = str(counter)
                old_to_new[old_id] = new_id
                ref_copy = ref.copy()
                ref_copy["reference_id"] = new_id
                filtered_refs.append(ref_copy)
                counter += 1

        merged_data["references"] = filtered_refs

        # Update all reference_ids to new numbering (top-level + evidence entries)
        for e in merged_data.get("entities", []):
            old = e.get("reference_id", "")
            if old and old in old_to_new:
                e["reference_id"] = old_to_new[old]
            elif old:
                e["reference_id"] = ""
            for ev in e.get("evidence", []):
                old = ev.get("reference_id", "")
                if old and old in old_to_new:
                    ev["reference_id"] = old_to_new[old]
                elif old:
                    ev["reference_id"] = ""
        for r in merged_data.get("relationships", []):
            old = r.get("reference_id", "")
            if old and old in old_to_new:
                r["reference_id"] = old_to_new[old]
            elif old:
                r["reference_id"] = ""
            for ev in r.get("evidence", []):
                old = ev.get("reference_id", "")
                if old and old in old_to_new:
                    ev["reference_id"] = old_to_new[old]
                elif old:
                    ev["reference_id"] = ""
        for c in merged_data.get("chunks", []):
            old = c.get("reference_id", "")
            if old and old in old_to_new:
                c["reference_id"] = old_to_new[old]
            # Re-number all reference_ids collected during content dedup
            ref_ids = c.get("reference_ids")
            if ref_ids:
                c["reference_ids"] = [
                    old_to_new[rid] if rid in old_to_new else rid for rid in ref_ids
                ]

        # Step 6: Build final context_str
        context_str = self._build_context_from_merged(merged_data, mode, _multi_kb_keys)

        # Step 7: Clean internal reference_ids to avoid leaking into API output
        for c in merged_data.get("chunks", []):
            c.pop("reference_ids", None)

        logger.info(
            f"Multi-KB post-process complete: "
            f"{len(merged_data.get('entities', []))} entities, "
            f"{len(merged_data.get('relationships', []))} relations, "
            f"{len(merged_data.get('chunks', []))} chunks"
        )

        return merged_data, context_str

    def _build_fallback_context(self, merged_data: Dict[str, Any], mode: str) -> str:
        """Build a basic context string without token budget control (fallback when no tokenizer)."""
        _multi_kb_keys = {"evidence", "kb_ids", "source_count"}

        text_units_str = "\n".join(
            json.dumps(
                {
                    "reference_ids": c.get("reference_ids")
                    or [c.get("reference_id", "")],
                    "content": c.get("content", ""),
                },
                ensure_ascii=False,
            )
            for c in merged_data.get("chunks", [])
        )
        reference_list_str = "\n".join(
            f"[{ref.get('reference_id', '')}] {ref.get('file_path', '')}"
            for ref in merged_data.get("references", [])
            if ref.get("reference_id")
        )

        if mode == "naive":
            return PROMPTS["multi_kb_naive_query_context"].format(
                text_chunks_str=text_units_str,
                reference_list_str=reference_list_str,
            )

        entities_str = "\n".join(
            json.dumps(
                {k: v for k, v in e.items() if k not in _multi_kb_keys},
                ensure_ascii=False,
            )
            for e in merged_data.get("entities", [])
        )
        relations_str = "\n".join(
            json.dumps(
                {k: v for k, v in r.items() if k not in _multi_kb_keys},
                ensure_ascii=False,
            )
            for r in merged_data.get("relationships", [])
        )
        return PROMPTS["multi_kb_kg_query_context"].format(
            entities_str=entities_str,
            relations_str=relations_str,
            text_chunks_str=text_units_str,
            reference_list_str=reference_list_str,
        )

    def _build_context_from_merged(
        self, merged_data: Dict[str, Any], mode: str, _multi_kb_keys: set
    ) -> str:
        """Build the final context string from post-processed merged data."""
        text_units_str = "\n".join(
            json.dumps(
                {
                    "reference_ids": c.get("reference_ids")
                    or [c.get("reference_id", "")],
                    "content": c.get("content", ""),
                },
                ensure_ascii=False,
            )
            for c in merged_data.get("chunks", [])
        )
        reference_list_str = "\n".join(
            f"[{ref.get('reference_id', '')}] {ref.get('file_path', '')}"
            for ref in merged_data.get("references", [])
            if ref.get("reference_id")
        )

        if mode == "naive":
            return PROMPTS["multi_kb_naive_query_context"].format(
                text_chunks_str=text_units_str,
                reference_list_str=reference_list_str,
            )

        entities_str = "\n".join(
            json.dumps(
                {k: v for k, v in e.items() if k not in _multi_kb_keys},
                ensure_ascii=False,
            )
            for e in merged_data.get("entities", [])
        )
        relations_str = "\n".join(
            json.dumps(
                {k: v for k, v in r.items() if k not in _multi_kb_keys},
                ensure_ascii=False,
            )
            for r in merged_data.get("relationships", [])
        )
        return PROMPTS["multi_kb_kg_query_context"].format(
            entities_str=entities_str,
            relations_str=relations_str,
            text_chunks_str=text_units_str,
            reference_list_str=reference_list_str,
        )
