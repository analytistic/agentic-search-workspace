# Benchmark

本目录集中管理 Agentic Search 的评测协议和评测入口。它使用外层仓库准备的 BrowseComp-Plus 数据与官方评测代码（`third_party/BrowseComp-Plus/`），不复制数据，也不修改该目录。

```text
benchmark/
├── README.md       # 评测协议、输入输出格式和运行说明
├── configs/        # 可复现实验配置
├── scripts/        # 评测入口和数据适配脚本
└── results/        # 本地生成的评测结果（不提交）
```

当前只建立目录约定，尚未实现可运行的评测入口。后续每个 benchmark 配置应记录数据版本、Agent、retriever、模型、随机种子、指标和结果位置。

从仓库根目录运行时，统一使用：

```bash
uv run --project agentic-search python agentic-search/benchmark/scripts/<entrypoint>.py
```
