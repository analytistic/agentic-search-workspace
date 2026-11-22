"""BrowseComp-Plus 评测：起 N 个 Ray rollout worker，分批跑题，存轨迹，算 EM。

把一批题切成 N 份并行下发（默认 4 个 worker × 每次 64 题），每个 worker 在自己的
进程里建 LLM 客户端和检索工具 —— 构造函数只传可 pickle 的 spec dict，否则闭包传不进
Ray actor。

用法：
    python -m isp.eval --limit 64 --workers 4 \
        --retriever-url http://127.0.0.1:8100 \
        --llm-kind openai --llm-base-url http://127.0.0.1:8000/v1 --llm-model Qwen3-32B \
        --out runs/eval_0001
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import logging
import math
import os
import re
import string
import time
from typing import Any, Optional, Sequence

from .agentworker import MockLLM, Trajectory
from .envfile import env, load_env
from .training import RolloutActorWorker, RolloutConfig, init_ray, summarize_trajectories

logger = logging.getLogger(__name__)

BROWSECOMP_SYSTEM = """You are a deep research agent. You answer hard multi-hop questions by searching a fixed corpus of web documents.

You have two tools. To call one, output a single JSON object and nothing else:
{"tool": "search", "arguments": {"query": "<keywords>"}}
{"tool": "browse", "arguments": {"docid": "<docid from an earlier search>"}}

How to work:
1. Read the question and list the constraints it imposes (dates, counts, roles, names, places).
2. Search one constraint at a time. Queries must be short keyword phrases, never full sentences.
3. A search returns up to 10 docids with only the first 64 tokens of each document. That snippet is a preview, not the content.
4. Before relying on a document, open it with browse. Never answer from snippets alone, and never answer before you have opened at least one document.
5. If a search is useless, change the constraint or the wording. Do not repeat a query that already failed.
6. Stop as soon as every constraint is satisfied. Do not keep searching once you can answer.

Final answer rules:
- Output one line only: the answer itself (a name, a title, a date, or a number).
- No explanation, no citation, no markdown, no quotes.
- If you are still unsure after exhausting the constraints, output your single best guess."""

DEFAULT_SYSTEM_HINT = ""


# ---------- 数据 ----------
def resolve_path(path: str) -> str:
    """相对路径先按当前目录找，找不到就逐级向上找（在 isp/ 里跑也能定位到仓库根的数据）。"""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    cur = os.path.abspath(os.getcwd())
    while True:
        cand = os.path.join(cur, path)
        if os.path.exists(cand):
            return cand
        parent = os.path.dirname(cur)
        if parent == cur:
            return path
        cur = parent


def resolve_glob(pattern: str) -> str:
    """通配符路径同样逐级向上找：只要在某一层能匹配到文件就返回绝对模式。"""
    if not pattern:
        return pattern
    if os.path.isabs(pattern) or glob.glob(pattern):
        return pattern
    cur = os.path.abspath(os.getcwd())
    while True:
        cand = os.path.join(cur, pattern)
        if glob.glob(cand):
            return cand
        parent = os.path.dirname(cur)
        if parent == cur:
            return pattern
        cur = parent


def load_queries(path: str, limit: Optional[int] = None, offset: int = 0) -> list[dict]:
    rows: list[dict] = []
    with open(resolve_path(path), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rows = rows[offset:]
    return rows[:limit] if limit else rows


def build_prompt(row: dict) -> str:
    # 题目本身作为 user 消息；答题格式由系统提示约束，这里不要再加"只给答案"之类的话，
    # 否则模型倾向不检索直接猜。
    return row["query"]


# ---------- 判分 ----------
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.I)
_PUNCT = set(string.punctuation)


def normalize_answer(text: str) -> str:
    text = (text or "").lower().strip()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> bool:
    return bool(normalize_answer(pred)) and normalize_answer(pred) == normalize_answer(gold)


def contains_gold(pred: str, gold: str) -> bool:
    g = normalize_answer(gold)
    return bool(g) and g in normalize_answer(pred)


# ---------- 免 LLM 的检索指标 ----------
def load_qrels(path: str) -> dict[str, set[str]]:
    """读 TREC 格式：qid 0 docid rel。"""
    out: dict[str, set[str]] = {}
    p = resolve_path(path)
    if not os.path.exists(p):
        return out
    with open(p, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                out.setdefault(str(parts[0]), set()).add(str(parts[2]))
    return out


def _ndcg_at_k(hits: Sequence[str], relevant: set[str], k: int = 10) -> float:
    """二值相关性 nDCG@k：命中证据记 1，未命中记 0。"""
    if not relevant:
        return 0.0
    dcg = 0.0
    for rank, docid in enumerate(hits[:k], start=1):
        if docid in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def retrieval_metrics(record: dict, evidence: set[str], gold: set[str], negative: set[str]) -> dict:
    """从一条轨迹里算检索类指标，全程不调 LLM。"""
    searched: list[str] = []
    browsed: list[str] = []
    first_search: list[str] = []
    for call in record["tool_calls"]:
        if call["name"] == "search":
            if not first_search:
                first_search = list(call["documents"])
            searched.extend(call["documents"])
        elif call["name"] == "browse":
            browsed.extend(call["documents"])

    searched_set, browsed_set = set(searched), set(browsed)
    n_ev = len(evidence)
    ev_covered = len(searched_set & evidence)

    def recall(hits: set[str], target: set[str]) -> Optional[float]:
        return len(hits & target) / len(target) if target else None

    return {
        "n_evidence": n_ev,
        "n_evidence_covered": ev_covered,
        "evidence_recall": recall(searched_set, evidence),
        "evidence_recall_browsed": recall(browsed_set, evidence),
        "evidence_hit": bool(ev_covered) if n_ev else None,
        "evidence_hit_browsed": bool(browsed_set & evidence) if n_ev else None,
        "gold_recall": recall(searched_set, gold),
        "negative_recall": recall(searched_set, negative),
        "negative_recall_browsed": recall(browsed_set, negative),
        "first_search_ndcg10": _ndcg_at_k(first_search, evidence, 10),
        "first_search_evidence": len(set(first_search) & evidence),
        "unique_docs": len(searched_set),
        "duplicate_rate": round(1 - len(searched_set) / len(searched), 4) if searched else None,
    }


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def aggregate_metrics(records: Sequence[dict]) -> dict:
    """逐题指标取平均，另给微平均召回（按证据总数加权）。"""
    m = [r["metrics"] for r in records if r.get("metrics")]
    if not m:
        return {}
    ev_tot = sum(x["n_evidence"] for x in m)
    ev_hit = sum(x["n_evidence_covered"] for x in m)
    return {
        "evidence_recall_macro": _mean([x["evidence_recall"] for x in m]),
        "evidence_recall_micro": round(ev_hit / ev_tot, 4) if ev_tot else None,
        "evidence_recall_browsed": _mean([x["evidence_recall_browsed"] for x in m]),
        "evidence_hit_rate": _mean([float(x["evidence_hit"]) for x in m if x["evidence_hit"] is not None]),
        "evidence_hit_rate_browsed": _mean(
            [float(x["evidence_hit_browsed"]) for x in m if x["evidence_hit_browsed"] is not None]
        ),
        "gold_recall": _mean([x["gold_recall"] for x in m]),
        "negative_recall": _mean([x["negative_recall"] for x in m]),
        "negative_recall_browsed": _mean([x["negative_recall_browsed"] for x in m]),
        "first_search_ndcg10": _mean([x["first_search_ndcg10"] for x in m]),
        "avg_unique_docs": _mean([float(x["unique_docs"]) for x in m]),
        "duplicate_rate": _mean([x["duplicate_rate"] for x in m]),
    }


# ---------- LLM ----------
def build_llm(spec: dict):
    """返回一个 async 可调用：messages -> 文本。spec 必须是纯 dict（要过 Ray）。"""
    kind = spec.get("kind", "mock")
    if kind == "mock":
        return MockLLM(queries=["transistor history"], answer="unknown")

    import httpx

    base_url = spec["base_url"].rstrip("/")
    model = spec["model"]
    api_key = spec.get("api_key") or os.environ.get("OPENAI_API_KEY", "EMPTY")
    temperature = float(spec.get("temperature", 0.0))
    max_tokens = int(spec.get("max_tokens", 1024))
    timeout = float(spec.get("timeout", 120))
    retries = int(spec.get("retries", 3))

    async def llm(messages: list[dict[str, Any]]) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        last: Optional[Exception] = None
        for attempt in range(retries):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(
                        f"{base_url}/chat/completions",
                        json=payload,
                        headers={"Authorization": f"Bearer {api_key}"},
                    )
                    resp.raise_for_status()
                    return resp.json()["choices"][0]["message"]["content"] or ""
            except Exception as exc:  # 网络抖动重试，最后仍然失败就抛出去
                last = exc
                await asyncio.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM 调用失败: {type(last).__name__}: {last}")

    return llm


# ---------- 工具 ----------
def build_tools(spec: dict):
    """返回 (tools, 需要保活的资源)。spec 同样是纯 dict。"""
    from .server import http_browse_tool, http_search_tool

    if spec.get("retriever_url"):
        url = spec["retriever_url"]
        tools = {
            "search": http_search_tool(url, default_topk=spec.get("topk", 10)),
            "browse": http_browse_tool(url),
        }
        return tools, None

    # 进程内检索：带索引和编码器，省掉一层 HTTP，代价是每个 worker 各占一份内存
    from .retriever import FlatRetriever

    retriever = FlatRetriever(
        index_glob=spec["index_glob"],
        corpus_path=spec.get("corpus_path"),
        model_path=spec.get("model_path"),
        device=spec.get("device", "auto"),
        preview_tokens=spec.get("preview_tokens", 64),
    )

    def search(query: str, top_k: int = spec.get("topk", 10)) -> dict:
        hits = retriever.search_text(query, topk=top_k)
        return {
            "text": "\n\n".join(f"[{h.docid}] {h.text}" for h in hits),
            "documents": [h.docid for h in hits],
        }

    def browse(docid: str) -> dict:
        return {"text": retriever.get_document(docid), "documents": [docid]}

    return {"search": search, "browse": browse}, retriever


# ---------- 本地检索服务 ----------
def start_local_server(spec: dict, port: int = 8137):
    """在主进程起一份检索服务（一个索引 + 一个编码器），actor 只连 URL。

    为什么不在 actor 里各建一份：4 个 actor 各加载 1.2G 编码器 + 0.41G 索引会直接把
    内存打爆（实测 worker 被系统杀掉）。
    """
    import threading

    import uvicorn

    from .retriever import FlatRetriever
    from .server import create_app

    retriever = FlatRetriever(
        index_glob=spec["index_glob"],
        corpus_path=spec.get("corpus_path"),
        model_path=spec.get("model_path"),
        device=spec.get("device", "auto"),
        preview_tokens=spec.get("preview_tokens", 64),
    )
    server = uvicorn.Server(uvicorn.Config(create_app(retriever), host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(400):
        if server.started:
            break
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError(f"检索服务没起来（端口 {port}）")
    return f"http://127.0.0.1:{port}", retriever


# ---------- Ray worker ----------
class EvalWorker(RolloutActorWorker):
    """在 actor 进程里按 spec 自建 LLM 与工具，避免把闭包/句柄 pickle 过去。"""

    def __init__(self, llm_spec: dict, tools_spec: dict, config: RolloutConfig) -> None:
        llm = build_llm(llm_spec)
        tools, keepalive = build_tools(tools_spec)
        super().__init__(llm=llm, tools=tools, config=config, retriever=keepalive)


# ---------- 主流程 ----------
def chunk(seq: Sequence, n: int) -> list[list]:
    if n <= 0:
        return [list(seq)]
    size = max(1, (len(seq) + n - 1) // n)
    return [list(seq[i : i + size]) for i in range(0, len(seq), size)]


def build_records(rows: Sequence[dict], trajs: Sequence[Trajectory], qrels: Optional[dict] = None) -> list[dict]:
    qrels = qrels or {}
    ev_q = qrels.get("evidence", {})
    gold_q = qrels.get("gold", {})
    neg_q = qrels.get("negative", {})
    records = []
    for row, traj in zip(rows, trajs):
        pred = traj.final_answer or ""
        gold = row.get("answer", "")
        rec = {
            "query_id": str(row.get("query_id")),
            "query": row["query"],
            "gold": gold,
            "pred": pred,
            "em": exact_match(pred, gold),
            "contains_gold": contains_gold(pred, gold),
            "turns": traj.turns,
            "error": traj.error,
            "elapsed": round(traj.elapsed, 2),
            "n_search": sum(1 for c in traj.tool_calls if c.name == "search"),
            "n_browse": sum(1 for c in traj.tool_calls if c.name == "browse"),
            "tool_calls": [
                {"name": c.name, "arguments": c.arguments, "documents": c.documents} for c in traj.tool_calls
            ],
        }
        qid = rec["query_id"]
        evidence = ev_q.get(qid, set())
        rec["metrics"] = retrieval_metrics(rec, evidence, gold_q.get(qid, set()), neg_q.get(qid, set()))
        records.append(rec)
    return records


def flush_outputs(out_dir: str, records: Sequence[dict]) -> None:
    """每批结束就把已有记录写盘：中途停止也不会白跑。"""
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "trajectories.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    partial = {
        "n": len(records),
        "em": round(sum(r["em"] for r in records) / max(len(records), 1), 4),
        "retrieval": aggregate_metrics(records),
        "partial": True,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(partial, f, ensure_ascii=False, indent=2)


async def run_eval(args) -> dict:
    rows = load_queries(args.queries, limit=args.limit, offset=args.offset)
    if not rows:
        raise SystemExit("没有题目可跑")

    batch_size = min(args.batch_size, len(rows))
    per_worker = max(1, -(-batch_size // args.workers))  # 向上取整
    concurrency = args.max_concurrency or per_worker  # 默认让分到的条全部在飞
    logger.info(
        "题目 %d 条 | 每批 %d 条 → %d 个 worker × %d 条并发（每 worker 同时 %d 条）",
        len(rows), batch_size, args.workers, per_worker, concurrency,
    )

    llm_spec = {
        "kind": args.llm_kind,
        "base_url": args.llm_base_url,
        "model": args.llm_model,
        "api_key": args.llm_api_key,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    keepalive = None
    tools_spec = {
        "retriever_url": args.retriever_url,
        "topk": args.topk,
        "preview_tokens": args.preview_tokens,
    }
    if not tools_spec["retriever_url"]:
        if not args.index_glob:
            raise SystemExit("要么给 --retriever-url，要么给 --index-glob 让主进程起一个本地检索服务")
        tools_spec["retriever_url"], keepalive = start_local_server(
            {
                "index_glob": resolve_glob(args.index_glob),
                "corpus_path": resolve_path(args.corpus_path) if args.corpus_path else None,
                "model_path": resolve_path(args.model_path) if args.model_path else None,
                "device": args.device,
                "preview_tokens": args.preview_tokens,
            },
            port=args.serve_port,
        )
        logger.info("本地检索服务已起: %s", tools_spec["retriever_url"])
    config = RolloutConfig(
        max_concurrency=concurrency, max_turns=args.max_turns, system_prompt=BROWSECOMP_SYSTEM
    )

    qrels = {
        "evidence": load_qrels(os.path.join(args.qrels_dir, "qrels_evidence.txt")),
        "gold": load_qrels(os.path.join(args.qrels_dir, "qrels_gold.txt")),
        "negative": load_qrels(os.path.join(args.qrels_dir, "qrels_negative.txt")),
    }
    logger.info(
        "qrels: evidence %d 题 / gold %d 题 / negative %d 题",
        len(qrels["evidence"]), len(qrels["gold"]), len(qrels["negative"]),
    )

    init_ray()
    import ray

    worker_cls = ray.remote(EvalWorker)
    workers = [worker_cls.remote(llm_spec, tools_spec, config) for _ in range(args.workers)]

    records: list[dict] = []
    trajs: list[Trajectory] = []
    t0 = time.time()
    n_batches = (len(rows) + batch_size - 1) // batch_size
    try:
        for bi, start in enumerate(range(0, len(rows), batch_size), start=1):
            batch_rows = rows[start : start + batch_size]
            parts = chunk([build_prompt(r) for r in batch_rows], args.workers)
            outs = await asyncio.gather(*[w.generate_batch.remote(p) for w, p in zip(workers, parts)])
            batch_trajs = [t for out in outs for t in out]
            trajs.extend(batch_trajs)

            batch_records = build_records(batch_rows, batch_trajs, qrels)
            records.extend(batch_records)
            logger.info(
                "batch %d/%d 完成: %d 条, 累计 EM %.2f%%, 累计证据召回 %.2f%%, "
                "本批平均轮数 %.1f, search %.1f, browse %.1f, 用时 %.0fs",
                bi, n_batches, len(batch_records),
                100 * sum(r["em"] for r in records) / len(records),
                100 * (aggregate_metrics(records).get("evidence_recall_macro") or 0.0),
                sum(r["turns"] for r in batch_records) / max(len(batch_records), 1),
                sum(r["n_search"] for r in batch_records) / max(len(batch_records), 1),
                sum(r["n_browse"] for r in batch_records) / max(len(batch_records), 1),
                time.time() - t0,
            )
            flush_outputs(args.out, records)
    finally:
        for w in workers:
            try:
                ray.kill(w)
            except Exception:
                pass
    elapsed = time.time() - t0

    n = len(records)
    summary = {
        "n": n,
        "workers": args.workers,
        "batch_size": batch_size,
        "per_worker": per_worker,
        "concurrency_per_worker": concurrency,
        "elapsed_sec": round(elapsed, 1),
        "em": round(sum(r["em"] for r in records) / n, 4),
        "contains_gold": round(sum(r["contains_gold"] for r in records) / n, 4),
        "failed": sum(1 for r in records if r["error"]),
        "avg_turns": round(sum(r["turns"] for r in records) / n, 2),
        "avg_search": round(sum(r["n_search"] for r in records) / n, 2),
        "avg_browse": round(sum(r["n_browse"] for r in records) / n, 2),
        "browse_per_search": round(
            sum(r["n_browse"] for r in records) / max(sum(r["n_search"] for r in records), 1), 3
        ),
        "retrieval": aggregate_metrics(records),
        "trajectory_stats": summarize_trajectories(trajs),
    }
    return {"summary": summary, "records": records}


def write_report(summary: dict, path: str) -> str:
    """把免 LLM 的指标写成 markdown，方便直接看。"""
    r = summary.get("retrieval", {})

    def pct(v):
        return "—" if v is None else f"{v * 100:.2f}%"

    def num(v):
        return "—" if v is None else f"{v}"

    lines = [
        "# BrowseComp-Plus 评测报告",
        "",
        f"- 题目数：{summary['n']}（每批 {summary['batch_size']}，{summary['workers']} 个 worker × 每批 {summary['per_worker']} 条）",
        f"- 总耗时：{summary['elapsed_sec']} 秒",
        f"- 失败轨迹：{summary['failed']}",
        "",
        "## 任务指标（字符串归一化，不调 LLM）",
        "",
        "| 指标 | 值 | 说明 |",
        "|---|---|---|",
        f"| EM | {pct(summary['em'])} | 归一化后完全匹配 |",
        f"| 含答案 | {pct(summary['contains_gold'])} | 预测里包含标准答案 |",
        "",
        "## 检索指标（qrels 比对，不调 LLM）",
        "",
        "| 指标 | 值 | 说明 |",
        "|---|---|---|",
        f"| 证据召回（宏） | {pct(r.get('evidence_recall_macro'))} | 每题命中证据数 ÷ 该题证据数，再平均 |",
        f"| 证据召回（微） | {pct(r.get('evidence_recall_micro'))} | 全部命中证据数 ÷ 全部证据数 |",
        f"| 证据召回（仅浏览） | {pct(r.get('evidence_recall_browsed'))} | 只算真正打开过的文档 |",
        f"| 证据命中率 | {pct(r.get('evidence_hit_rate'))} | 至少命中一条证据的题占比 |",
        f"| 证据命中率（仅浏览） | {pct(r.get('evidence_hit_rate_browsed'))} | 至少打开一条证据的题占比 |",
        f"| 金文档召回 | {pct(r.get('gold_recall'))} | 对照 qrels_gold |",
        f"| 难负例召回 | {pct(r.get('negative_recall'))} | 召回到的难负例占比，越低越好 |",
        f"| 难负例召回（仅浏览） | {pct(r.get('negative_recall_browsed'))} | 打开过的难负例占比 |",
        f"| 首搜 nDCG@10 | {num(r.get('first_search_ndcg10'))} | 第一次搜索按证据算 |",
        f"| 平均去重文档数 | {num(r.get('avg_unique_docs'))} | 一条轨迹里搜过的不重复文档 |",
        f"| 重复召回率 | {pct(r.get('duplicate_rate'))} | 搜索结果里重复条目的比例 |",
        "",
        "## 执行效率",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 平均轮数 | {summary['avg_turns']} |",
        f"| 平均 search 次数 | {summary['avg_search']} |",
        f"| 平均 browse 次数 | {summary['avg_browse']} |",
        f"| browse / search | {summary['browse_per_search']} |",
        "",
        "> 未包含需要 LLM 的指标（官方 Accuracy、Calibration Error、Citation Precision/Recall）。",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main() -> None:
    loaded = load_env()
    if loaded:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        logger.info(".env 已加载 %d 个键: %s", len(loaded), sorted(loaded))

    ap = argparse.ArgumentParser(description="BrowseComp-Plus 评测（Ray 多 worker）")
    ap.add_argument("--queries", default=env("ISP_QUERIES", "data/browsecomp_plus/processed/queries.jsonl"))
    ap.add_argument("--qrels-dir", default=env("ISP_QRELS_DIR", "data/browsecomp_plus/processed"))
    ap.add_argument("--limit", type=int, default=0, help="总共跑多少题，0 表示全部")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=int(env("ISP_BATCH_SIZE", 64)), help="每次下发的题数，按 worker 数均分")
    ap.add_argument("--workers", type=int, default=int(env("ISP_WORKERS", 4)))
    ap.add_argument("--out", default=env("ISP_OUT", "runs/eval"))
    ap.add_argument("--max-turns", type=int, default=int(env("ISP_MAX_TURNS", 50)))
    ap.add_argument("--max-concurrency", type=int, default=None,
                    help="每个 worker 同时在跑的条数，默认等于 batch/workers")
    ap.add_argument("--topk", type=int, default=int(env("ISP_TOPK", 10)))
    ap.add_argument("--preview-tokens", type=int, default=int(env("ISP_PREVIEW_TOKENS", 64)))

    ap.add_argument("--retriever-url", default=env("ISP_RETRIEVER_URL"), help="给了就连现成服务，否则主进程起一个")
    ap.add_argument("--serve-port", type=int, default=int(env("ISP_SERVE_PORT", 8137)))
    ap.add_argument("--index-glob", default=env("ISP_INDEX_GLOB"))
    ap.add_argument("--corpus-path", default=env("ISP_CORPUS_PATH"))
    ap.add_argument("--model-path", default=env("ISP_MODEL_PATH"))
    ap.add_argument("--device", default=env("ISP_DEVICE", "auto"))

    ap.add_argument("--llm-kind", default="openai", choices=["openai", "mock"])
    ap.add_argument("--llm-base-url", default=env("ISP_LLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--llm-model", default=env("ISP_LLM_MODEL", "Qwen3-32B"))
    ap.add_argument("--llm-api-key", default=env("ISP_LLM_API_KEY"))
    ap.add_argument("--temperature", type=float, default=float(env("ISP_TEMPERATURE", 0.0)))
    ap.add_argument("--max-tokens", type=int, default=int(env("ISP_MAX_TOKENS", 1024)))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = asyncio.run(run_eval(args))

    os.makedirs(args.out, exist_ok=True)
    traj_path = os.path.join(args.out, "trajectories.jsonl")
    with open(traj_path, "w", encoding="utf-8") as f:
        for r in result["records"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(result["summary"], f, ensure_ascii=False, indent=2)
    report_path = write_report(result["summary"], os.path.join(args.out, "report.md"))

    s = result["summary"]
    r = s.get("retrieval", {})

    def pct(v):
        return "—" if v is None else f"{v * 100:.2f}%"

    print(f"\n题目 {s['n']} | 失败 {s['failed']} | 总耗时 {s['elapsed_sec']}s")
    print(f"并行: {s['workers']} worker × 每批 {s['per_worker']} 条（每 worker 同时 {s['concurrency_per_worker']} 条）")
    print("\n[任务]")
    print(f"  EM {pct(s['em'])} | 含答案 {pct(s['contains_gold'])}")
    print("\n[检索]")
    print(f"  证据召回 宏 {pct(r.get('evidence_recall_macro'))} | 微 {pct(r.get('evidence_recall_micro'))} "
          f"| 仅浏览 {pct(r.get('evidence_recall_browsed'))}")
    print(f"  证据命中率 {pct(r.get('evidence_hit_rate'))} | 仅浏览 {pct(r.get('evidence_hit_rate_browsed'))} "
          f"| 金文档召回 {pct(r.get('gold_recall'))}")
    print(f"  难负例召回 {pct(r.get('negative_recall'))} | 仅浏览 {pct(r.get('negative_recall_browsed'))} "
          f"| 首搜 nDCG@10 {r.get('first_search_ndcg10')}")
    print("\n[效率]")
    print(f"  平均轮数 {s['avg_turns']} | search {s['avg_search']} | browse {s['avg_browse']} "
          f"| browse/search {s['browse_per_search']}")
    print(f"\n轨迹: {traj_path}\n报告: {report_path}")


if __name__ == "__main__":
    main()
