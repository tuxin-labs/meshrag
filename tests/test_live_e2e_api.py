"""Live end-to-end API tests against a RUNNING MeshRAG server.

These tests exercise the full stack with the real LLM (doubao-seed-2.0-mini via
Volcano ARK) and the real embedding model (doubao-embedding-vision), so they need
network access to the configured providers and a server started from the repo root
(picks up .env): ``python -m lightrag.api.lightrag_server``.

Run with:  python -m pytest tests/test_live_e2e_api.py --run-integration -v
They are marked ``integration`` and skipped by the default offline suite.

Flow: create KBs -> ingest documents (wait for pipeline) -> query in every mode ->
multi-KB hybrid query -> streaming -> /query/data -> model profile switching ->
registered external KBs (incl. external-only) -> graph API -> Ollama compat ->
document management -> cleanup.
"""

import json
import os
import time

import pytest
import requests

BASE_URL = "http://127.0.0.1:9621"

# Volcano ARK key for the model-switching profile test; supply your own via
# the environment when running the live suite.
ARK_API_KEY = os.environ.get("ARK_API_KEY", "")

KB_A = "e2e_live_a"
KB_B = "e2e_live_b"

DOC_A = (
    "太阳系是围绕太阳运行的天体系统。地球是太阳系的第三颗行星，拥有液态水和大气层，"
    "是已知唯一存在生命的行星。火星是第四颗行星，因表面氧化铁呈红色，被称为红色星球，"
    "美国宇航局的毅力号火星车正在火星表面寻找古代生命痕迹。木星是太阳系最大的行星，"
    "是一颗气态巨行星，其大红斑是一场持续数百年的巨大风暴。"
)

DOC_B = (
    "字节跳动是一家中国科技公司，由张一鸣于2012年创立，总部位于北京。"
    "字节跳动旗下的短视频平台抖音在中国拥有数亿日活用户，其国际版 TikTok "
    "在全球市场广受欢迎。梁汝波于2021年接任字节跳动首席执行官。"
    "该公司还开发了企业协作套件飞书（Lark）。"
)

DOC_C = (
    "红烧肉是一道经典的中式菜肴，主料是五花肉。制作红烧肉需要将五花肉切块焯水，"
    "然后加入冰糖炒出糖色，再加入生抽、老抽和料酒小火慢炖一小时。"
    "上海本帮红烧肉偏甜口，湖南毛氏红烧肉则加入干辣椒。"
)

AUTH = {}  # no auth configured locally; guest requests allowed


def _wait_pipeline_done(kb_id: str, timeout: int = 300) -> bool:
    """Poll /documents/status_counts until nothing is pending/processing."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(
            f"{BASE_URL}/documents/status_counts", params={"kb_id": kb_id}, timeout=15
        )
        counts = r.json().get("status_counts", {})
        busy = counts.get("pending", 0) + counts.get("processing", 0)
        if busy == 0 and (
            counts.get("processed", 0) > 0 or counts.get("failed", 0) > 0
        ):
            return counts.get("processed", 0) > 0
        time.sleep(3)
    return False


@pytest.fixture(scope="module")
def live_server():
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    assert r.status_code == 200, "live server not reachable; start it first"
    return BASE_URL


@pytest.fixture(scope="module")
def ingested_kbs(live_server):
    """Create two KBs and ingest one document each; wait for both pipelines."""
    # Clean leftovers from a previous aborted run
    for kb in (KB_A, KB_B):
        requests.delete(f"{live_server}/knowledge_bases/{kb}", timeout=30)

    for kb in (KB_A, KB_B):
        r = requests.post(
            f"{live_server}/knowledge_bases", json={"kb_id": kb}, timeout=60
        )
        assert r.status_code == 200, r.text

    r = requests.post(
        f"{live_server}/documents/text",
        params={"kb_id": KB_A},
        json={"text": DOC_A, "file_source": "solar_system.txt"},
        timeout=30,
    )
    assert r.status_code == 200, r.text
    r = requests.post(
        f"{live_server}/documents/text",
        params={"kb_id": KB_A},
        json={"text": DOC_B, "file_source": "bytedance.txt"},
        timeout=30,
    )
    assert r.status_code == 200, r.text
    r = requests.post(
        f"{live_server}/documents/text",
        params={"kb_id": KB_B},
        json={"text": DOC_C, "file_source": "hongshaorou.txt"},
        timeout=30,
    )
    assert r.status_code == 200, r.text

    assert _wait_pipeline_done(KB_A), "KB_A pipeline did not finish"
    assert _wait_pipeline_done(KB_B), "KB_B pipeline did not finish"
    yield live_server
    for kb in (KB_A, KB_B):
        requests.delete(f"{live_server}/knowledge_bases/{kb}", timeout=60)


@pytest.mark.integration
class TestLiveKBAndDocuments:
    def test_kb_listed_and_stats(self, ingested_kbs):
        base = ingested_kbs
        r = requests.get(f"{base}/knowledge_bases", timeout=10)
        kbs = r.json()["knowledge_bases"]
        assert KB_A in kbs and KB_B in kbs

        r = requests.get(f"{base}/knowledge_bases/stats", timeout=30)
        stats = {e["kb_id"]: e for e in r.json()["knowledge_bases"]}
        assert stats[KB_A]["documents"] == 2
        assert stats[KB_B]["documents"] == 1
        assert stats[KB_A]["entities"] > 0, "entity extraction produced no entities"

    def test_document_paginated_listing(self, ingested_kbs):
        base = ingested_kbs
        r = requests.post(
            f"{base}/documents/paginated",
            params={"kb_id": KB_A},
            json={"page": 1, "page_size": 10},
            timeout=30,
        )
        body = r.json()
        assert body["pagination"]["total_count"] == 2
        names = {d["file_path"] for d in body["documents"]}
        assert "solar_system.txt" in names and "bytedance.txt" in names


@pytest.mark.integration
class TestLiveQueryModes:
    def test_query_naive(self, ingested_kbs):
        r = requests.post(
            f"{ingested_kbs}/query",
            json={
                "query": "火星为什么被称为红色星球",
                "mode": "naive",
                "kb_ids": [KB_A],
            },
            timeout=180,
        )
        assert r.status_code == 200, r.text
        assert len(r.json()["response"]) > 20

    def test_query_local(self, ingested_kbs):
        r = requests.post(
            f"{ingested_kbs}/query",
            json={"query": "木星是什么样的行星", "mode": "local", "kb_ids": [KB_A]},
            timeout=180,
        )
        assert r.status_code == 200
        assert len(r.json()["response"]) > 20

    def test_query_global(self, ingested_kbs):
        r = requests.post(
            f"{ingested_kbs}/query",
            json={
                "query": "太阳系中有哪些重要的天体",
                "mode": "global",
                "kb_ids": [KB_A],
            },
            timeout=180,
        )
        assert r.status_code == 200
        assert len(r.json()["response"]) > 20

    def test_query_hybrid_with_references(self, ingested_kbs):
        r = requests.post(
            f"{ingested_kbs}/query",
            json={
                "query": "字节跳动是谁创立的",
                "mode": "hybrid",
                "kb_ids": [KB_A],
                "include_references": True,
            },
            timeout=180,
        )
        assert r.status_code == 200
        body = r.json()
        assert "张一鸣" in body["response"], body["response"][:200]
        assert body["references"], "expected references for hybrid mode"

    def test_query_mix_default(self, ingested_kbs):
        r = requests.post(
            f"{ingested_kbs}/query",
            json={"query": "毅力号火星车在做什么"},
            timeout=180,
        )
        assert r.status_code == 200
        assert len(r.json()["response"]) > 20

    def test_query_bypass(self, ingested_kbs):
        r = requests.post(
            f"{ingested_kbs}/query",
            json={"query": "用一句话介绍地球", "mode": "bypass", "kb_ids": [KB_A]},
            timeout=120,
        )
        assert r.status_code == 200
        body = r.json()
        assert body["response"]
        assert not body.get("references")

    def test_multi_kb_hybrid_query(self, ingested_kbs):
        r = requests.post(
            f"{ingested_kbs}/query",
            json={
                "query": "分别介绍一项烹饪技术和一家科技公司",
                "mode": "mix",
                "kb_ids": [KB_A, KB_B],
                "include_references": True,
            },
            timeout=180,
        )
        assert r.status_code == 200
        body = r.json()
        assert body["response"]
        ref_files = {ref.get("file_path", "") for ref in body.get("references") or []}
        # references may be grouped; at least one should come from each KB
        assert any("hongshaorou" in f for f in ref_files), ref_files

    def test_query_data_retrieval_only(self, ingested_kbs):
        r = requests.post(
            f"{ingested_kbs}/query/data",
            json={"query": "太阳系的行星", "mode": "mix", "kb_ids": [KB_A]},
            timeout=120,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "success"
        data = body["data"]
        assert data["entities"], "no entities returned"
        assert data["chunks"], "no chunks returned"

    def test_query_stream_ndjson(self, ingested_kbs):
        r = requests.post(
            f"{ingested_kbs}/query/stream",
            json={
                "query": "抖音是什么平台",
                "mode": "mix",
                "kb_ids": [KB_A],
                "include_references": True,
            },
            stream=True,
            timeout=180,
        )
        assert r.status_code == 200
        assert "ndjson" in r.headers.get("content-type", "")

        events = []
        for line in r.iter_lines(decode_unicode=True):
            if line:
                events.append(json.loads(line))

        assert events, "no NDJSON frames received"
        first = events[0]
        assert "references" in first or "response" in first or first.get("type")
        deltas = [e["response"] for e in events if "response" in e]
        assert deltas, "no response deltas in stream"
        assert not any("error" in e for e in events), events


@pytest.mark.integration
class TestLiveModelSwitching:
    def test_model_profile_registration_and_query(self, ingested_kbs):
        if not ARK_API_KEY:
            pytest.skip("ARK_API_KEY not configured")
        base = ingested_kbs
        # register a doubao chat profile (model switching via registry)
        r = requests.post(
            f"{base}/llm_models",
            json={
                "name": "Doubao E2E profile",
                "kind": "llm",
                "binding": "openai",
                "model": "doubao-seed-2.0-mini",
                "host": "https://ark.cn-beijing.volces.com/api/plan/v3",
                "api_key": ARK_API_KEY,
                "enabled": True,
            },
            timeout=15,
        )
        assert r.status_code == 200, r.text
        profile_id = r.json()["id"]

        try:
            # connection probe against the real endpoint
            r = requests.post(f"{base}/llm_models/{profile_id}/test", timeout=60)
            assert r.status_code == 200, r.text
            probe = r.json()
            assert probe["reachable"] is True
            assert probe["model_available"] is True

            # query using the profile (per-request model switching)
            r = requests.post(
                f"{base}/query",
                json={
                    "query": "什么是火星",
                    "mode": "naive",
                    "kb_ids": [KB_A],
                    "llm_profile_id": profile_id,
                },
                timeout=180,
            )
            assert r.status_code == 200, r.text
            assert len(r.json()["response"]) > 20
        finally:
            requests.delete(f"{base}/llm_models/{profile_id}", timeout=10)


@pytest.mark.integration
class TestLiveExternalKBs:
    def test_register_probe_and_query_external(self, ingested_kbs):
        base = ingested_kbs
        # the server's own mock endpoints act as external services
        r = requests.post(
            f"{base}/external_kbs",
            json={
                "name": "E2E external retrieval",
                "type": "retrieval",
                "url": f"{base}/ext/retrieval",
                "api_key": "ext-kb-test-sk-2026",
                "top_k": 5,
                "enabled": True,
            },
            timeout=15,
        )
        assert r.status_code == 200, r.text
        ext_id = r.json()["id"]

        try:
            r = requests.post(f"{base}/external_kbs/{ext_id}/test", timeout=30)
            assert r.status_code == 200, r.text
            assert r.json()["reachable"] is True

            # hybrid: internal KB + external retrieval KB
            r = requests.post(
                f"{base}/query",
                json={
                    "query": "介绍火星",
                    "mode": "naive",
                    "kb_ids": [KB_A],
                    "external_kb_ids": [ext_id],
                },
                timeout=180,
            )
            assert r.status_code == 200, r.text
            assert r.json()["response"]

            # external-only query (no internal KB)
            r = requests.post(
                f"{base}/query",
                json={
                    "query": "介绍一下检索结果的内容",
                    "mode": "bypass",
                    "kb_ids": [],
                    "external_kb_ids": [ext_id],
                },
                timeout=120,
            )
            assert r.status_code == 200, r.text
        finally:
            requests.delete(f"{base}/external_kbs/{ext_id}", timeout=10)


@pytest.mark.integration
class TestLiveGraphAPI:
    def test_graph_labels_and_subgraph(self, ingested_kbs):
        base = ingested_kbs
        r = requests.get(f"{base}/graph/label/list", params={"kb_id": KB_A}, timeout=30)
        labels = r.json()
        assert isinstance(labels, list) and labels, "no graph labels"

        r = requests.get(
            f"{base}/graphs",
            params={"kb_id": KB_A, "label": labels[0], "max_depth": 2, "max_nodes": 50},
            timeout=30,
        )
        graph = r.json()
        assert "nodes" in graph and "edges" in graph

        r = requests.get(
            f"{base}/graph/entity/exists",
            params={"kb_id": KB_A, "name": labels[0]},
            timeout=15,
        )
        assert r.json()["exists"] is True


@pytest.mark.integration
class TestLiveOllamaCompat:
    def test_version_tags(self, ingested_kbs):
        base = ingested_kbs
        assert "version" in requests.get(f"{base}/api/version", timeout=10).json()
        tags = requests.get(f"{base}/api/tags", timeout=10).json()
        assert tags["models"], "no ollama models listed"

    def test_chat_bypass_mode(self, ingested_kbs):
        r = requests.post(
            f"{ingested_kbs}/api/chat",
            json={
                "model": "meshrag:latest",
                "messages": [
                    {"role": "user", "content": "/bypass 用一个词回答：天空是什么颜色"}
                ],
                "stream": False,
            },
            timeout=120,
        )
        assert r.status_code == 200, r.text
        assert r.json()["message"]["content"]

    def test_chat_stream(self, ingested_kbs):
        r = requests.post(
            f"{ingested_kbs}/api/chat",
            json={
                "model": "meshrag:latest",
                "messages": [{"role": "user", "content": "/bypass 数到三"}],
                "stream": True,
            },
            stream=True,
            timeout=120,
        )
        assert r.status_code == 200
        lines = [ln for ln in r.iter_lines(decode_unicode=True) if ln]
        assert lines, "no streaming output"
        assert json.loads(lines[-1])["done"] is True


@pytest.mark.integration
class TestLiveDocumentManagement:
    def test_multipart_upload_pipeline_and_duplicate(self, ingested_kbs):
        base = ingested_kbs
        content = (
            "量子计算是一种利用量子力学原理进行计算的技术。量子计算机使用量子比特，"
            "量子比特可以同时处于0和1的叠加态。量子纠缠是量子计算的另一个核心概念。"
        ).encode("utf-8")

        r = requests.post(
            f"{base}/documents/upload",
            params={"kb_id": KB_A},
            files={"file": ("quantum_live.txt", content, "text/plain")},
            timeout=30,
        )
        assert r.status_code == 200, r.text
        track_id = r.json()["track_id"]
        assert _wait_pipeline_done(KB_A, timeout=300)

        r = requests.get(
            f"{base}/documents/track_status/{track_id}",
            params={"kb_id": KB_A},
            timeout=15,
        )
        assert r.status_code == 200
        assert r.json()["total_count"] >= 1

        # uploading the same content again is detected as duplicated
        r = requests.post(
            f"{base}/documents/upload",
            params={"kb_id": KB_A},
            files={"file": ("quantum_live_copy.txt", content, "text/plain")},
            timeout=30,
        )
        assert r.status_code == 200
        assert r.json()["status"] == "duplicated"

    def test_update_text_and_delete_document(self, ingested_kbs):
        base = ingested_kbs
        # insert a disposable document then delete it
        r = requests.post(
            f"{base}/documents/text",
            params={"kb_id": KB_A},
            json={
                "text": "临时文档：冥王星曾被视为第九大行星，2006年被重新归类为矮行星。",
                "file_source": "disposable.txt",
            },
            timeout=30,
        )
        assert r.status_code == 200
        track_id = r.json().get("track_id")
        assert _wait_pipeline_done(KB_A, timeout=240)

        r = requests.get(
            f"{base}/documents/track_status/{track_id}",
            params={"kb_id": KB_A},
            timeout=15,
        )
        assert r.status_code == 200
        body = r.json()
        doc_ids = [d["id"] for d in body.get("documents", [])]
        assert doc_ids, "track status returned no documents"

        r = requests.delete(
            f"{base}/documents/delete_document",
            params={"kb_id": KB_A},
            json={"doc_ids": doc_ids},
            timeout=30,
        )
        assert r.status_code == 200, r.text

        # wait for deletion to finish
        deadline = time.time() + 120
        while time.time() < deadline:
            r = requests.post(
                f"{base}/documents/paginated",
                params={"kb_id": KB_A},
                json={"page": 1, "page_size": 50},
                timeout=15,
            )
            names = {d["file_path"] for d in r.json()["documents"]}
            if "disposable.txt" not in names:
                break
            time.sleep(3)
        assert "disposable.txt" not in names, "document not deleted"
