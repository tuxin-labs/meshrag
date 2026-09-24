# 外部知识库使用说明

## 概览

系统支持在查询时同时请求外部知识库 API，并将外部结果与本地知识库结果组合使用。
当前支持两类外部知识库：

| 类型 | 说明 | 外部服务返回内容 | LightRAG 中的用途 |
|------|------|------------------|-------------------|
| `retrieval` | 检索型，默认类型 | `results` / chunks | 与本地检索结果合并后，由本系统 LLM 生成最终答案 |
| `rag` | 完整 RAG 服务型 | `answer` + `references` | 作为补充答案参与最终生成，或在仅外部 RAG 场景下直接作为答案来源 |

---

## 请求参数

在 `/query`、`/query/stream`、`/query/data` 三个端点中，都可以通过 `external_kbs` 字段传入外部知识库配置：

```json
{
  "query": "你的查询文本",
  "mode": "mix",
  "kb_ids": ["kb1"],
  "external_kbs": [
    {
      "type": "retrieval",
      "url": "https://your-retrieval-api.com/search",
      "api_key": "optional-api-key",
      "top_k": 5
    },
    {
      "type": "rag",
      "url": "https://your-rag-api.com/query",
      "api_key": "optional-api-key"
    }
  ]
}
```

`ExternalKBConfig` 字段说明：

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `type` | `"retrieval"` / `"rag"` | 否 | `"retrieval"` | 外部知识库类型 |
| `url` | `string` | 是 | - | 外部知识库 API 地址 |
| `api_key` | `string` | 否 | `null` | 认证密钥，通过 `Authorization: Bearer` 传递 |
| `top_k` | `int` (1-50) | 否 | `5` | 仅 `retrieval` 类型使用，请求的最大结果数 |

注意：

- `type` 默认值是 `retrieval`。
- 只有显式传入 `"type": "rag"`，才会触发 RAG 型外部 KB 的处理逻辑。
- 当 `kb_ids` 为空但传入了 `external_kbs` 时，允许仅使用外部知识库。

---

## 外部 API 协议要求

### 1. Retrieval 型 API

请求格式：

```http
POST {url}
Content-Type: application/json
Authorization: Bearer {api_key}

{"query": "查询文本", "top_k": 5}
```

`Authorization` 头可选；未配置 `api_key` 时不会发送。

期望响应格式：

```json
{
  "status": "success",
  "results": [
    {
      "content": "文档片段文本",
      "file_path": "来源文件路径或 URL"
    },
    {
      "content": "另一个文档片段",
      "file_path": "其他来源"
    }
  ]
}
```

字段说明：

- `status`：必须为 `"success"`
- `results`：数组，每个元素至少包含 `content`
- `file_path`：可选，用于引用标注
- `chunk_id`：可选，不提供时系统会自动生成

重试策略：

- 最多重试 3 次
- 指数退避约 4s-10s
- 单次请求超时 10 秒

### 2. RAG 型 API

请求格式：

```http
POST {url}
Content-Type: application/json
Authorization: Bearer {api_key}

{"query": "查询文本"}
```

`Authorization` 头可选；未配置 `api_key` 时不会发送。

期望响应格式：

```json
{
  "status": "success",
  "answer": "完整的答案文本",
  "references": [
    {"file_path": "来源1"},
    {"file_path": "来源2"}
  ]
}
```

字段说明：

- `status`：必须为 `"success"`
- `answer`：必填，字符串，外部 RAG 服务生成的完整答案
- `references`：可选数组，每个元素包含 `file_path`

重试策略：

- 最多重试 2 次
- 指数退避约 4s-10s
- 单次请求超时 60 秒

---

## 使用示例

### 仅使用外部检索型知识库

```json
{
  "query": "什么是深度学习？",
  "external_kbs": [
    {
      "type": "retrieval",
      "url": "https://vector-db.example.com/search",
      "api_key": "sk-xxx",
      "top_k": 10
    }
  ]
}
```

### 混合本地 + 外部知识库

```json
{
  "query": "解释 Transformer 架构",
  "mode": "mix",
  "kb_ids": ["my_local_kb"],
  "external_kbs": [
    {
      "type": "retrieval",
      "url": "https://external-vdb.com/search",
      "top_k": 5
    },
    {
      "type": "rag",
      "url": "https://rag-service.com/answer"
    }
  ]
}
```

---

## 处理流程

系统会根据配置自动选择处理路径：

| 场景 | 条件 | 处理方式 |
|------|------|----------|
| A | `bypass` 模式 | 跳过所有检索，直接调用 LLM |
| B | 仅内部 KB | 使用内部单库或多库查询流程 |
| C | 仅外部 `rag` | 单个时直接返回外部答案；多个时通过 LLM 合并 |
| D | 仅外部 `retrieval` | 获取 chunks 后由本地 LLM 生成答案 |
| E | 仅外部混合型 | retrieval 结果 + RAG 答案共同参与生成 |
| F | 内部 + 外部 `retrieval` | 合并检索后统一生成 |
| G | 内部 + 外部 `rag` | 内部检索结果与 RAG 答案共同参与生成 |
| H | 内部 + 外部 `retrieval` + `rag` | 全部合并后统一生成 |

关键行为：

- 所有外部 KB 请求会并行执行
- `retrieval` 型结果会与本地检索结果统一去重、预算控制、rerank
- `rag` 型答案会作为补充上下文参与最终答案生成
- 单个外部 KB 失败不会阻断其他数据源

---

## 响应中的外部数据

当前代码实现中，不同端点的返回结构并不相同，请按下面的实际行为理解：

### `/query/data`

当请求中包含 `type: "rag"` 的外部 KB 时，`external_answers` 会出现在 `data` 字段内。

示例：

```json
{
  "status": "success",
  "message": "Merged query executed successfully",
  "data": {
    "entities": [],
    "relationships": [],
    "chunks": [],
    "references": [
      {
        "reference_id": "1",
        "file_path": "ext_doc.pdf",
        "kb_id": "https://rag-service.com/answer"
      }
    ],
    "external_answers": [
      {
        "source": "https://rag-service.com/answer",
        "answer": "外部 RAG 服务的答案",
        "references": [
          {"file_path": "ext_doc.pdf"}
        ]
      }
    ]
  },
  "metadata": {
    "query_mode": "mix"
  }
}
```

### `/query`

当前实现里，`/query` 只返回：

- `response`
- `references`

不会把 `external_answers` 单独透传给客户端。

示例：

```json
{
  "response": "最终生成的答案...",
  "references": [
    {
      "reference_id": "1",
      "file_path": "ext_doc.pdf"
    }
  ]
}
```

### `/query/stream`

当前实现里，`/query/stream` 也不会单独输出 `external_answers`。

- 首个 NDJSON 分片中最多包含 `references`
- 后续分片包含流式 `response` 内容

如果你需要获取 `external_answers` 原始数据，建议使用 `/query/data`。
@router.post(
    "/ext/retrieval",
    tags=["External KB Test"],  # 明确使用不同于默认的分类
    summary="检索型外部知识库模拟接口",
)
