"""CRUD and connection-test endpoints for model profiles and external knowledge bases.

These registries are what let the Web UI manage models and external bases instead of
re-typing endpoints on every query. Stored secrets are never returned to clients.
"""

from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from lightrag.utils import logger
from ..utils_api import get_combined_auth_dependency
from ..registry import ModelProfileStore, ExternalKBStore

PROBE_TIMEOUT_SECONDS = 20.0


class ModelProfileRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Display name of the profile")
    kind: str = Field("llm", description="llm | embedding | rerank")
    binding: str = Field(
        ...,
        description="openai | ollama | azure_openai | gemini | aws_bedrock | lollms",
    )
    model: str = Field(..., min_length=1)
    host: str = Field(..., min_length=1)
    api_key: Optional[str] = Field(
        None,
        description="Omit to keep the stored key, send an empty string to clear it.",
    )
    embedding_dim: Optional[int] = Field(
        None, description="Required for kind=embedding"
    )
    enabled: bool = True


class ExternalKBRequest(BaseModel):
    name: str = Field(..., min_length=1)
    type: str = Field("retrieval", description="retrieval | rag")
    url: str = Field(..., min_length=1)
    api_key: Optional[str] = None
    top_k: int = Field(5, ge=1, le=50)
    enabled: bool = True


def _bad_request(message: str) -> HTTPException:
    return HTTPException(status_code=400, detail=message)


async def _probe_model(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Reach out to the configured endpoint and report what happened."""
    binding = profile.get("binding", "")
    host = (profile.get("host") or "").rstrip("/")
    model = profile.get("model", "")
    api_key = profile.get("api_key") or ""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    if binding == "ollama":
        url = f"{host}/api/tags"
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_SECONDS) as client:
            response = await client.get(url, headers=headers)
        response.raise_for_status()
        names = [m.get("name") for m in response.json().get("models", [])]
        matched = any(
            name == model or (name or "").split(":")[0] == model for name in names
        )
        return {
            "reachable": True,
            "model_available": matched,
            "message": (
                f"Model '{model}' found"
                if matched
                else f"Endpoint reachable, model '{model}' not listed"
            ),
        }

    if profile.get("kind") == "embedding":
        url = f"{host}/embeddings"
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_SECONDS) as client:
            response = await client.post(
                url,
                headers={**headers, "Content-Type": "application/json"},
                json={"model": model, "input": "connection probe"},
            )
        if response.status_code >= 400:
            raise HTTPException(
                status_code=502, detail=f"{response.status_code} {response.text[:300]}"
            )
        data = response.json().get("data") or []
        dim = len(data[0].get("embedding") or []) if data else 0
        configured = profile.get("embedding_dim") or 0
        if configured and dim and configured != dim:
            return {
                "reachable": True,
                "model_available": True,
                "detail": {"returned_dim": dim, "configured_dim": configured},
                "message": f"Dimension mismatch: service returns {dim}, profile declares {configured}",
            }
        return {
            "reachable": True,
            "model_available": True,
            "detail": {"returned_dim": dim},
            "message": f"Embedding OK (dim={dim})",
        }

    # Chat-style profile: ask for a one-token completion so the model name is validated too.
    url = f"{host}/chat/completions"
    async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_SECONDS) as client:
        response = await client.post(
            url,
            headers={**headers, "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            },
        )
    if response.status_code >= 400:
        raise HTTPException(
            status_code=502, detail=f"{response.status_code} {response.text[:300]}"
        )
    return {"reachable": True, "model_available": True, "message": "Chat completion OK"}


async def _probe_external_kb(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Call the external service with the protocol the backend itself expects."""
    url = entry.get("url") or ""
    api_key = entry.get("api_key") or ""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    if entry.get("type") == "rag":
        payload: Dict[str, Any] = {"query": "connection probe"}
    else:
        payload = {"query": "connection probe", "top_k": entry.get("top_k", 5)}

    async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_SECONDS) as client:
        response = await client.post(url, headers=headers, json=payload)
    if response.status_code >= 400:
        raise HTTPException(
            status_code=502, detail=f"{response.status_code} {response.text[:300]}"
        )

    try:
        body = response.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="Response is not JSON")

    if body.get("status") != "success":
        return {
            "reachable": True,
            "model_available": False,
            "message": "Response is missing status='success'",
        }
    if entry.get("type") == "rag":
        ok = bool(body.get("answer"))
        return {
            "reachable": True,
            "model_available": ok,
            "message": "RAG answer received" if ok else "Empty answer",
        }
    results = body.get("results")
    ok = isinstance(results, list)
    return {
        "reachable": True,
        "model_available": ok,
        "detail": {"result_count": len(results) if ok else 0},
        "message": f"Retrieval OK ({len(results) if ok else 0} results)"
        if ok
        else "Missing results array",
    }


def create_registry_routes(
    model_store: ModelProfileStore,
    external_kb_store: ExternalKBStore,
    api_key: Optional[str] = None,
):
    router = APIRouter()
    combined_auth = get_combined_auth_dependency(api_key)

    @router.get(
        "/llm_models", dependencies=[Depends(combined_auth)], tags=["Model Management"]
    )
    async def list_model_profiles() -> List[Dict[str, Any]]:
        return model_store.public_all()

    @router.post(
        "/llm_models", dependencies=[Depends(combined_auth)], tags=["Model Management"]
    )
    async def create_model_profile(request: ModelProfileRequest) -> Dict[str, Any]:
        try:
            record = model_store.normalize(request.model_dump())
        except ValueError as e:
            raise _bad_request(str(e))
        return model_store.public(model_store.add(record))

    @router.put(
        "/llm_models/{profile_id}",
        dependencies=[Depends(combined_auth)],
        tags=["Model Management"],
    )
    async def update_model_profile(
        profile_id: str, request: ModelProfileRequest
    ) -> Dict[str, Any]:
        existing = model_store.get(profile_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="Model profile not found")
        try:
            patch = model_store.normalize(request.model_dump(), existing=existing)
        except ValueError as e:
            raise _bad_request(str(e))
        updated = model_store.replace(profile_id, patch)
        return model_store.public(updated)

    @router.delete(
        "/llm_models/{profile_id}",
        dependencies=[Depends(combined_auth)],
        tags=["Model Management"],
    )
    async def delete_model_profile(profile_id: str) -> Dict[str, Any]:
        if not model_store.remove(profile_id):
            raise HTTPException(status_code=404, detail="Model profile not found")
        return {"status": "success", "id": profile_id}

    @router.post(
        "/llm_models/{profile_id}/test",
        dependencies=[Depends(combined_auth)],
        tags=["Model Management"],
    )
    async def test_model_profile(profile_id: str) -> Dict[str, Any]:
        profile = model_store.get(profile_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Model profile not found")
        import time

        started = time.monotonic()
        try:
            outcome = await _probe_model(profile)
        except httpx.HTTPStatusError as e:
            outcome = {
                "reachable": True,
                "model_available": False,
                "message": f"HTTP {e.response.status_code}",
            }
        except httpx.HTTPError as e:
            logger.warning(f"External KB probe failed for profile: {e}")
            outcome = {
                "reachable": False,
                "model_available": False,
                "message": "Connection failed",
            }
        outcome["latency_ms"] = int((time.monotonic() - started) * 1000)
        outcome["id"] = profile_id
        outcome["name"] = profile.get("name")
        return outcome

    @router.get(
        "/external_kbs", dependencies=[Depends(combined_auth)], tags=["External KB"]
    )
    async def list_external_kbs() -> List[Dict[str, Any]]:
        return external_kb_store.public_all()

    @router.post(
        "/external_kbs", dependencies=[Depends(combined_auth)], tags=["External KB"]
    )
    async def create_external_kb(request: ExternalKBRequest) -> Dict[str, Any]:
        try:
            record = external_kb_store.normalize(request.model_dump())
        except ValueError as e:
            raise _bad_request(str(e))
        return external_kb_store.public(external_kb_store.add(record))

    @router.put(
        "/external_kbs/{kb_id}",
        dependencies=[Depends(combined_auth)],
        tags=["External KB"],
    )
    async def update_external_kb(
        kb_id: str, request: ExternalKBRequest
    ) -> Dict[str, Any]:
        existing = external_kb_store.get(kb_id)
        if existing is None:
            raise HTTPException(
                status_code=404, detail="External knowledge base not found"
            )
        try:
            patch = external_kb_store.normalize(request.model_dump(), existing=existing)
        except ValueError as e:
            raise _bad_request(str(e))
        return external_kb_store.public(external_kb_store.replace(kb_id, patch))

    @router.delete(
        "/external_kbs/{kb_id}",
        dependencies=[Depends(combined_auth)],
        tags=["External KB"],
    )
    async def delete_external_kb(kb_id: str) -> Dict[str, Any]:
        if not external_kb_store.remove(kb_id):
            raise HTTPException(
                status_code=404, detail="External knowledge base not found"
            )
        return {"status": "success", "id": kb_id}

    @router.post(
        "/external_kbs/{kb_id}/test",
        dependencies=[Depends(combined_auth)],
        tags=["External KB"],
    )
    async def test_external_kb(kb_id: str) -> Dict[str, Any]:
        entry = external_kb_store.get(kb_id)
        if entry is None:
            raise HTTPException(
                status_code=404, detail="External knowledge base not found"
            )
        import time

        started = time.monotonic()
        try:
            outcome = await _probe_external_kb(entry)
        except httpx.HTTPError as e:
            logger.warning(f"External KB probe failed for {kb_id}: {e}")
            outcome = {
                "reachable": False,
                "model_available": False,
                "message": "Connection failed",
            }
        outcome["latency_ms"] = int((time.monotonic() - started) * 1000)
        outcome["id"] = kb_id
        outcome["name"] = entry.get("name")
        return outcome

    return router
