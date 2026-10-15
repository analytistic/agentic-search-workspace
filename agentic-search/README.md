# Agentic Search

本目录是独立的 Python 研究项目，负责 Agent 实现及实验代码。

```text
agentic-search/
├── pyproject.toml          # 包定义、依赖及构建配置
├── benchmark/              # 评测入口、配置与评测协议
└── src/
    └── agentic_search/     # Python 源码
```

## 开发环境

需要 Python 3.10 或更高版本。在仓库根目录运行：

```bash
uv sync --project agentic-search
uv run --project agentic-search python -c "import agentic_search; print(agentic_search.__file__)"
```

虚拟环境创建在 `agentic-search/.venv/`。添加依赖时使用
`uv add --project agentic-search <package>`。构建安装包使用
`uv build --project agentic-search`。

## 目录边界

- 源码放在 `src/agentic_search/`；后续实验入口、配置和测试分别放在本目录的 `experiments/`、`configs/`、`tests/`。
- BrowseComp-Plus 等评测相关入口、配置和协议放在 `benchmark/`；生成的评测结果放在 `benchmark/results/`，不纳入 Git。`benchmark/` 共用本项目的 `pyproject.toml`，不单独建立 Python 包。
- 数据使用外层 `data/`，官方评测代码使用外层 `third_party/BrowseComp-Plus/`，运行凭据使用外层 `.agent/.env`。
- 实验输出放在本目录的 `results/`，不纳入 Git。
- 当前仅包含包骨架和 `benchmark/` 目录约定，尚未实现 Agent loop、评测入口或配置自动加载。

从仓库根目录运行命令时，共享数据路径写作 `data/...`；进入本目录后写作 `../data/...`。
