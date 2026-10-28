# retrieval server notes

`retrieval_server.py` 是 Search-R1 的检索服务 fork（BM25 via pyserini/Lucene，dense via faiss + e5）。
训练侧由 `quarl/tool/search_r1_like_utils.py::perform_single_search_batch()` 调用，固定
`return_scores=True`，带 semaphore 和 timeout，异常会变成 `Search execution failed: ...` 进入 prompt。

以下是相对上游的改动，按「为什么改」组织。

## 1. 语料不再全量进内存（OffsetCorpus）

上游 `load_corpus()` 用 `datasets.load_dataset(num_proc=16)` 把整份语料读进 RAM：wiki-18 是
14 GB / 21.0M 行，Arrow 展开后还要在 `~/.cache/huggingface/datasets` 再落一份拷贝，等于
**30 GB 级的内存 + 磁盘开销**，只为每次请求读几十行。

现在默认走 `OffsetCorpus`：首次启动扫一遍语料生成 int64 行偏移表
（`offsets_<corpus-stem>.npy`，21M 行 = 168 MB，wiki-18 实测扫描 15~26 s），之后
`seek + readline` 按需读单行，常驻内存恒定。偏移表用「写临时文件 + rename」落盘，
构建过程被杀不会留下半截表。

- 行偏移表默认放在语料同目录。
- 想复现上游行为：`--eager-corpus`（会打印警告，因为它就是内存/磁盘问题的来源）。

### 注意：OffsetCorpus 只管「语料原文」，不管「向量索引」

检索一次需要两份独立的数据结构，两者内存完全独立计算：

| 数据结构 | wiki-18 体积 | 是否必须驻留 |
|---|---|---|
| **向量索引**（向量 + 图） | fp32 HNSW64 ≈ **75 GB**（21.0M × 768 × 4 B = 64.6 GB 向量 + M=64 图 10.8 GB） | 基本上**是**（见下） |
| **语料原文** | 14 GB 原文件 | 否，本改动后 168 MB 偏移表 + topk 行 |

所以 **dense 检索的内存占用 ≈ 索引大小 + 0.2 GB**，不是「几十 MB」。这一点必须提前规划机器规格。

`--no-index-mmap` 控制的是索引怎么读，它**不能**把 HNSW 索引变便宜：

- `IndexFlat*` / `IVF*`：访问模式偏顺序/分簇，mmap 后生效工作集远小于文件，**确实可以少占 RAM**。
- `IndexHNSW*`：图遍历在邻居之间随机跳，触达的页会逐渐覆盖整份文件；mmap 只是把「启动时一次性拷贝」变成「运行中逐步换页」。内存吃紧时 OS 还会把页换出，之后每一跳都是缺页 → 查询慢 10~100 倍。

**结论：HNSW 索引请按 `文件大小` 预留内存**。要真的省，只有两条路：量化（`IndexHNSWPQ` / SQ8，75 GB → 12~38 GB）或改用 BM25（索引 2.2 GB，不需要向量）。

## 2. BM25 docid 与语料行号的一致性校验（fail fast）

`_search()` 用 `corpus[int(hit.docid)]` 取文档，也就是**假定 docid 就是语料行号**。
如果索引和 `--corpus_path` 不是同一份数据（或 docid 是不透明 id），取回来的是**错的文档**，
而且不报错 —— RL 会照着错的证据训练。

启动时现在会抽 5 个位置（0/1/7/n÷3/n÷2/n-1）比对 `searcher.doc(i).docid()` 与语料第 i 行的
`id` 字段，不一致直接抛错并给出可操作提示；确实知道自己在做什么可以 `--skip-docid-check`。

同时 `_check_contain_doc()` 从「只看 doc(0)」改成抽样 20 篇多数表决：单篇缺 `contents`
不该让整个服务切到「读语料」分支。

## 3. 请求侧防护

| 项 | 说明 |
|---|---|
| `--max-queries` | 单请求 query 条数上限，超出返回 413（默认 256） |
| `--max-topk` | 单 query topk 上限，超出返回 400（默认 100） |
| `--max-chars` | 响应里 `contents`/`text` 截断到 N 字符，0 = 不截断（rollout 高频调用时省带宽） |
| `--cache-entries` | `(query, topk)` LRU 缓存条数，0 = 关闭；命中时响应多返回 `cached: [bool, ...]` |

## 4. 并发与统计

- BM25 批量检索改成 `ThreadPoolExecutor`（`--max-process-num`，默认 8）。上游虽然定义了
  `max_process_num = 8`，但 `_batch_search` 是串行 for 循环，rollout 时就是吞吐瓶颈。
  结果按输入顺序回填，已用 stub searcher 验证 12 条乱序 query 顺序不串。
- 不足 topk 时不再每个 query `warnings.warn`（高频刷日志），改为计数器。
- `GET /metrics`：请求数 / query 数 / 平均与最大延迟 / 空结果数与空结果率 / 各 retriever 计数 / 缓存命中。

## 5. 就绪状态

`GET /status` 上游恒返回 `"ok"`，索引还在加载时启动脚本也会以为准备好了。现在返回：

```json
{"status":"ok","ready":true,"uptime_s":23.4,"method":"e5",
 "index_path":"...","corpus_path":"...","corpus_docs":1525,"corpus_lazy":true,
 "topk":3,"ntotal":1525,"index_mmap":true,"faiss_gpu":false}
```

## 6. 其他

- dense 侧过滤 faiss 在 k > ntotal 时补的 `-1` 索引（上游会把最后一条文档当结果返回）。
- `--host/--port`、`--pooling`、`--max-length`、`--batch-size`、`--fp32` 全部可配
  （上游把 pooling / fp16 / batch_size 硬编码在 `__main__`，换 bge 之类要改代码）。
- `faiss` 改为 dense 分支内延迟导入：只用 BM25 的部署不必装 faiss。
- `_device()` 增加 MPS（macOS 本地调试用）。

## 已知限制 / 后续

1. **BM25 query 未做 Lucene 语法转义**（`QueryParser` 对 `: ( ) " ~ ^ + - [ ]` 会抛
   `InvalidQueryException`）。LLM 生成的 query 很容易带这些字符，目前会 500。这是下一步要做的：
   要么 `QueryParser.escape()`，要么改成 analyzer + `BooleanQuery`（更贴近 BM25 本意）。
2. **不透明 docid 的语料无法用当前位置映射**：校验会 fail。可选方案是构建一次
   `id -> line` 映射（21M 条约 2 GB 内存）作为 `--id-map` 可选模式。
3. dense 侧没有 micro-batching：并发请求仍各自 encode。流量大时值得做请求合并。
4. 无索引热重载 / 优雅关闭。

## 本地冒烟（不依赖 Java / GPU）

```bash
cd agentic-search
.venv/bin/python src/agentic_search/training/examples/retrieval/retrieval_server.py \
  --retriever_name e5 \
  --index_path ../data/wiki-demo/e5_demo.index \
  --corpus_path ../data/wiki-demo/corpus.jsonl \
  --retriever_model ../data/wiki_2018/model/intfloat/e5-base-v2 \
  --topk 3 --port 8010 --cache-entries 1000 --max-chars 300 --max-topk 50 --fp32
```

实测（demo：1525 篇 / IndexFlatIP / e5-base-v2 on MPS）：起服 ~25 s，
`/status` 就绪、`/retrieve` 命中正常、缓存命中返回 `cached:[true,true]`、
`topk=999 → 400`、300 条 query → 413、`/metrics` 计数正常。
