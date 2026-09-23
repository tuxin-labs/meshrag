# Contributing to MeshRAG

Thanks for your interest in contributing! (中文说明见文末)

## Development setup

```bash
# Core package
uv sync

# With API server / storage backends / LLM providers / test tooling
uv sync --extra api
uv sync --extra offline-storage
uv sync --extra offline-llm
uv sync --extra test
```

Configure the environment interactively (optional):

```bash
make env-base      # LLM, embedding, reranker config
make env-storage   # Storage backends
make env-server    # Server, security, SSL
make env-validate  # Validate existing .env
```

## Running tests

The default test suite is fully offline (no external services required):

```bash
uv run pytest tests/                # Offline tests only (default)
uv run pytest tests/ --run-integration   # Include integration tests
uv run pytest tests/ --test-workers 4    # Parallel workers
```

Markers:

- `offline` — no external dependencies (default, runs in CI)
- `integration` — requires external services (skipped unless `--run-integration`)

## Linting

```bash
uv run ruff check .
```

CI runs `ruff check .` and the offline test suite on every PR. Both must pass
before review.

## Pull request guidelines

- Concise, imperative commit subjects (e.g., `Fix lock key normalization`)
- Include a summary, operational impact, and linked issues
- For user-facing changes, include screenshots or API samples
- New features must follow TDD: tests first (normal, boundary, and error
  paths), then implementation, then a full regression run

## 中文说明

- 开发环境：使用 `uv sync` 安装依赖，按需附加 `--extra api / offline-storage / offline-llm / test`
- 测试：默认离线套件 `uv run pytest tests/`，提交前必须保证 `ruff check .` 与测试全部通过
- 新功能遵循 TDD 流程：先写测试（正常/边界/异常路径），再实现，最后回归
- 提交信息使用简洁的英文祈使句（如 `Fix lock key normalization`）

## License

By contributing, you agree that your contributions will be licensed under
the MIT License that covers this project.
