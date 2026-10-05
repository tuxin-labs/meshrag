"""Persistent stores for model profiles and external knowledge bases.

Both live as JSON next to ``knowledge_bases.json`` so they survive restarts.
API keys are never handed back to clients: ``public()`` strips the secret and
reports only whether one is set plus an unambiguous tail for display.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from typing import Any, Dict, List, Optional

from lightrag.utils import logger

MODEL_KINDS = ("llm", "embedding", "rerank")

# Kept in sync with QueryRequest.validate_llm_override and llm_factory.
LLM_BINDINGS = ("openai", "ollama", "azure_openai", "gemini", "aws_bedrock", "lollms")

EXTERNAL_KB_TYPES = ("retrieval", "rag")


def mask_secret(value: Optional[str]) -> Dict[str, Any]:
    """Describe a secret without revealing it."""
    text = (value or "").strip()
    if not text:
        return {"has_api_key": False, "api_key_tail": ""}
    return {
        "has_api_key": True,
        "api_key_tail": text[-4:] if len(text) >= 12 else "****",
    }


def _require(value: Optional[str], field: str) -> str:
    text = (value or "").strip()
    if not text:
        raise ValueError(f"{field} is required")
    return text


class JsonRecordStore:
    """Thread-safe list-of-objects JSON store with generated ids."""

    def __init__(self, path: str):
        self._path = path
        self._lock = threading.Lock()
        self._records: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self._records = [r for r in data if isinstance(r, dict) and r.get("id")]
        except Exception as e:
            logger.warning(f"Failed to load registry '{self._path}': {e}")

    def _flush(self) -> None:
        registry_dir = os.path.dirname(self._path)
        if registry_dir:
            os.makedirs(registry_dir, exist_ok=True)
        tmp_path = f"{self._path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._records, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self._path)

    def all(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._records]

    def get(self, record_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            for record in self._records:
                if record.get("id") == record_id:
                    return dict(record)
        return None

    def find(self, **equals: Any) -> Optional[Dict[str, Any]]:
        with self._lock:
            for record in self._records:
                if all(record.get(k) == v for k, v in equals.items()):
                    return dict(record)
        return None

    def add(self, record: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            stored = dict(record)
            stored.setdefault("id", uuid.uuid4().hex[:12])
            self._records.append(stored)
            self._flush()
            return dict(stored)

    def replace(
        self, record_id: str, patch: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            for index, record in enumerate(self._records):
                if record.get("id") != record_id:
                    continue
                updated = {**record, **patch, "id": record_id}
                self._records[index] = updated
                self._flush()
                return dict(updated)
        return None

    def remove(self, record_id: str) -> bool:
        with self._lock:
            remaining = [r for r in self._records if r.get("id") != record_id]
            if len(remaining) == len(self._records):
                return False
            self._records = remaining
            self._flush()
        return True


class ModelProfileStore(JsonRecordStore):
    """Named LLM / embedding / rerank endpoints that queries can select by id."""

    def public(self, record: Dict[str, Any]) -> Dict[str, Any]:
        safe = {k: v for k, v in record.items() if k != "api_key"}
        safe.update(mask_secret(record.get("api_key")))
        return safe

    def public_all(self) -> List[Dict[str, Any]]:
        return [self.public(r) for r in self.all()]

    def normalize(
        self, payload: Dict[str, Any], existing: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        kind = (payload.get("kind") or (existing or {}).get("kind") or "llm").strip()
        if kind not in MODEL_KINDS:
            raise ValueError(f"kind must be one of {', '.join(MODEL_KINDS)}")

        binding = (
            payload.get("binding") or (existing or {}).get("binding") or ""
        ).strip()
        if binding not in LLM_BINDINGS:
            raise ValueError(f"binding must be one of {', '.join(LLM_BINDINGS)}")

        record = {
            "name": _require(payload.get("name"), "name"),
            "kind": kind,
            "binding": binding,
            "model": _require(payload.get("model"), "model"),
            "host": _require(payload.get("host"), "host").rstrip("/"),
            "enabled": payload.get("enabled", (existing or {}).get("enabled", True)),
        }

        # The API contract is "omit to keep the stored secret, empty string to
        # clear it". Pydantic's model_dump always includes api_key=None when the
        # field was omitted, so None must mean "keep", not "clear".
        if payload.get("api_key") is not None:
            record["api_key"] = payload["api_key"].strip()
        elif existing:
            record["api_key"] = existing.get("api_key", "")

        if kind == "embedding":
            try:
                record["embedding_dim"] = int(payload.get("embedding_dim") or 0)
            except (TypeError, ValueError):
                raise ValueError("embedding_dim must be an integer")
            if record["embedding_dim"] <= 0:
                raise ValueError("embedding_dim must be a positive integer")
        return record


class ExternalKBStore(JsonRecordStore):
    """Registered third-party retrieval / RAG services."""

    def public(self, record: Dict[str, Any]) -> Dict[str, Any]:
        safe = {k: v for k, v in record.items() if k != "api_key"}
        safe.update(mask_secret(record.get("api_key")))
        return safe

    def public_all(self) -> List[Dict[str, Any]]:
        return [self.public(r) for r in self.all()]

    def normalize(
        self, payload: Dict[str, Any], existing: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        kb_type = (
            payload.get("type") or (existing or {}).get("type") or "retrieval"
        ).strip()
        if kb_type not in EXTERNAL_KB_TYPES:
            raise ValueError(f"type must be one of {', '.join(EXTERNAL_KB_TYPES)}")

        try:
            top_k = int(
                payload.get("top_k")
                if payload.get("top_k") is not None
                else (existing or {}).get("top_k", 5)
            )
        except (TypeError, ValueError):
            raise ValueError("top_k must be an integer")
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")

        record = {
            "name": _require(payload.get("name"), "name"),
            "type": kb_type,
            "url": _require(payload.get("url"), "url"),
            "top_k": top_k,
            "enabled": payload.get("enabled", (existing or {}).get("enabled", True)),
        }
        # Same contract as ModelProfileStore: None keeps the stored secret,
        # an empty string clears it.
        if payload.get("api_key") is not None:
            record["api_key"] = payload["api_key"].strip()
        elif existing:
            record["api_key"] = existing.get("api_key", "")
        return record

    def to_query_config(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Shape a stored entry into the inline ExternalKBConfig contract."""
        config: Dict[str, Any] = {"type": record["type"], "url": record["url"]}
        if record.get("api_key"):
            config["api_key"] = record["api_key"]
        if record["type"] == "retrieval":
            config["top_k"] = record.get("top_k", 5)
        return config
