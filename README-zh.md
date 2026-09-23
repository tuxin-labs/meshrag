<div align="center">

# MeshRAG

**基于 [LightRAG](https://github.com/HKUDS/LightRAG) 的多知识库联邦图 RAG 引擎**

Multi-Knowledge-Base Graph RAG engine, based on LightRAG

[English](README.md) · [许可证](#许可证) · Python 3.10+

</div>

> [!NOTE]
> MeshRAG 是 [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG)（v1.4.11，上游状态截至 2026-03-08）的分支项目，在 MIT 许可证下二次分发，感谢上游团队的工作。
> Python 模块名仍为 `import lightrag`（与上游生态保持兼容），发行包名为 `pip install meshrag`。
> 上游继承的完整文档保留在 [docs/LightRAG-upstream-usage-zh.md](docs/LightRAG-upstream-usage-zh.md)。

## MeshRAG 是什么

MeshRAG 完整保留了 LightRAG 的图 RAG 引擎（实体/关系抽取、多模态检索、可插拔存储），并补充了生产级多团队部署所需的能力：

### 🔀 多知识库联邦查询

一次 API 调用，同时查询**多个本地知识库**：

1. 各知识库并行检索 → 各自返回文本块与知识图谱上下文
2. 跨库全局去重与预算控制
3. 文本块重排序（可选 reranker）
4. 引用/参考来源重建，答案可溯源到具体知识库
5. 基于合并上下文的 LLM 答案生成

### 🌐 外部知识库接入

通过 `external_kbs` 将**外部检索/RAG API** 融入同一查询流程：

- `retrieval` 类型：外部 API 返回相似文本块，由 MeshRAG 的 LLM 生成答案
- `rag` 类型：外部 API 是完整 RAG 服务，直接返回最终答案

### 📄 文档管理接口

完整的上传/状态管线，并支持按文件路径快速定位单个文档 —
`GET /documents/by_file_path?kb_id=...&file_path=...` —
在文档量很大时避免低效的分页查询。

### 🔒 删除/解析并发管线

协作式取消与删除语义，支持崩溃恢复：删除请求优先于进行中的解析任务；
被中断的文档从断点恢复而非重置；瞬态失败不会被误判为永久失败。

### 🧬 继承自 LightRAG 的全部能力

- 图结构知识表示，支持 local / global / hybrid / naive / mix 五种查询模式
- 可插拔存储后端（JSON、Redis、PostgreSQL、MongoDB、Milvus、Qdrant、Neo4j、Memgraph、Faiss、NetworkX）
- WebUI、Ollama 兼容接口、流式响应
- 面向多租户的工作区（workspace）隔离

## 快速开始

### 安装

```bash
pip install meshrag            # 核心包
pip install "meshrag[api]"     # 含 API 服务端
```

### 最小程序

```python
import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import setup_logger

setup_logger("lightrag", level="INFO")

WORKING_DIR = "./rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )
    # 重要：两步初始化缺一不可！
    await rag.initialize_storages()  # 初始化存储后端
    return rag

async def main():
    rag = await initialize_rag()
    await rag.ainsert("你的文本")
    result = await rag.aquery("你的问题", param=QueryParam(mode="hybrid"))
    print(result)
    await rag.finalize_storages()  # 清理资源（必须）

asyncio.run(main())
```

### ⚠️ 重要：初始化要求

**MeshRAG 必须显式初始化后才能使用。** 创建 `LightRAG` 实例后必须调用
`await rag.initialize_storages()`，结束时必须调用 `await rag.finalize_storages()`
— 否则会出现 `AttributeError: __aenter__` 等错误。

### API 服务端

```bash
cp env.example .env    # 先配置 LLM / embedding / 存储
meshrag-server         # 生产启动
meshrag-gunicorn       # 多 worker 启动
```

Docker：

```bash
docker compose up -d               # 使用 ghcr.io/tuxin-labs/meshrag 镜像
docker build -t meshrag .          # 完整镜像（前端 + 后端）
docker build -f Dockerfile.lite .  # 精简镜像（API + 离线存储）
```

### 多知识库查询示例

`POST /query` 在同一请求中接受 `kb_ids`（本地知识库）与 `external_kbs`（外部知识库）：

```json
{
  "query": "哪些合同提到了数据驻留要求？",
  "mode": "mix",
  "kb_ids": ["contracts-2025", "policies"],
  "external_kbs": [
    {
      "type": "retrieval",
      "url": "https://kb.example.com/api/search",
      "api_key": "sk-...",
      "top_k": 5
    }
  ]
}
```

所有来源的结果统一去重、重排并标注引用。各类型外部知识库的完整字段约定见
`lightrag/api/routers/query_routes.py` 中的字段说明。

## 查询模式

| 模式 | 说明 |
|---|---|
| `local` | 聚焦具体实体的上下文相关检索 |
| `global` | 基于社区/摘要的宽泛知识检索 |
| `hybrid` | 结合 local 与 global |
| `naive` | 不走图的直接向量检索 |
| `mix` | 图谱检索与向量检索结合（建议搭配 reranker） |

## 存储后端

| 层 | 后端 |
|---|---|
| KV（LLM 缓存、文本块、文档信息） | JSON（默认）、Redis、PostgreSQL、MongoDB |
| 向量（embedding） | NanoVectorDB（默认）、Faiss、Milvus、Qdrant、PostgreSQL、MongoDB |
| 图（实体关系） | NetworkX（默认）、Neo4j、PostgreSQL+AGE、MongoDB、Memgraph |
| 文档状态 | JSON（默认）、Redis、PostgreSQL、MongoDB |

通过环境变量配置 — 完整注释清单见 [env.example](env.example)，
包括面向自签名证书部署的 `LIGHTRAG_SSL_VERIFY`。

## 文档

- [docs/LightRAG-upstream-usage-zh.md](docs/LightRAG-upstream-usage-zh.md) — 继承的引擎文档（配置、存储部署、API 细节）
- [docs/external_kb_usage.md](docs/external_kb_usage.md) — 外部知识库接入使用说明
- [docs/OfflineDeployment.md](docs/OfflineDeployment.md) — 离线/ air-gapped 部署
- [docs/DockerDeployment.md](docs/DockerDeployment.md) — Docker 部署
- [docs/FrontendBuildGuide.md](docs/FrontendBuildGuide.md) — WebUI 构建
- [docs/InteractiveSetup.md](docs/InteractiveSetup.md) — 交互式 `.env` 配置向导
- [CONTRIBUTING.md](CONTRIBUTING.md) — 开发与 PR 规范

## 开发

```bash
uv sync                  # 核心包
uv sync --extra api      # API 服务端
uv sync --extra test     # 测试依赖

uv run ruff check .      # 代码检查
uv run pytest tests/     # 离线测试套件（无需外部服务）
```

## 许可证

MeshRAG 基于 [MIT 许可证](LICENSE) 开源。
Copyright 2026 MeshRAG Team；原始 LightRAG 代码 Copyright 2025 LightRAG Team。
详见 [NOTICE](NOTICE)。

## 致谢

- [LightRAG (HKUDS)](https://github.com/HKUDS/LightRAG) — 本项目赖以构建的引擎。如果 MeshRAG 对你有帮助，也请给上游项目点个 Star。
- LightRAG 论文：[arXiv:2410.05779](https://arxiv.org/abs/2410.05779)
