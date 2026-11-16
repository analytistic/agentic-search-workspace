#!/usr/bin/env python3
"""并发压测：检索服务删锁后（Encoder._lock / _search_lock 已移除）的并发行为。

起服务 → asyncio.to_thread 并发 100 个 /retrieve 请求（每个 query list 1-3 个）
→ 统计延迟分布、失败/超时数，验证并发 forward / 并发 faiss search 是否会死锁或崩溃。

用法：
    python test_concurrent_retrieve.py            # 默认 100 并发
    python test_concurrent_retrieve.py 200        # 指定并发数

说明：
    - 服务进程用 wiki-demo 现成索引（IndexFlatIP, 1525 向量, 768 维）+ 本地 e5 模型
    - 请求形态对齐训练侧：query_list 1-3 个 query（统计显示模型偏好 1-2 个）
    - 每个请求设 3s 超时（对齐 search_tool_fn.py 的 SEARCH_TIMEOUT）
"""

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", ".."))
# 测试文件在 training/examples/retrieval/ → 上 6 层 = 仓库根（含 agentic-search/ 和 data/）
SERVER = os.path.join(REPO_ROOT, "agentic-search", "src", "agentic_search", "training", "examples", "retrieval", "retrieval_server.py")
INDEX = os.path.join(REPO_ROOT, "data", "wiki-demo", "e5_demo.index")
CORPUS = os.path.join(REPO_ROOT, "data", "wiki-demo", "corpus.jsonl")
MODEL = os.path.join(REPO_ROOT, "data", "wiki_2018", "model", "intfloat", "e5-base-v2")

HOST = "127.0.0.1"
PORT = 8010
URL = f"http://{HOST}:{PORT}/retrieve"

# 训练侧统计出的模型偏好：1 个 query 43%，2 个 48%，3 个 8%
QUERY_SIZES = [1, 1, 1, 2, 2, 2, 2, 3]

QUERY_POOL = [
    "who makes up the state council in russia",
    "structure of the russian state council",
    "William Nigh birth date",
    "Robert Greenwald birth date",
    "composition of the State Council in Russia",
    "University of Cambridge founder",
    "Albert Einstein theory of relativity",
    "Great Wall of China history",
    "Mount Everest height and location",
    "Shakespeare most famous plays",
]


def start_server() -> subprocess.Popen:
    env = dict(os.environ)
    # mac 上 e5 走 mps（或 cpu）；禁止抢 GPU 无意义，但防止 torch 开 12 线程吃满
    env["OMP_NUM_THREADS"] = "8"
    env["TOKENIZERS_PARALLELISM"] = "false"
    # faulthandler：死锁时把服务进程线程栈 dump 到 stderr（区分死锁 vs 崩溃）
    env["PYTHONFAULTHANDLER"] = "1"
    proc = subprocess.Popen(
        [sys.executable, SERVER,
         "--index_path", INDEX,
         "--corpus_path", CORPUS,
         "--topk", "3",
         "--retriever_name", "e5",
         "--retriever_model", MODEL,
         "--cache-entries", "1000",
         "--max-chars", "300",
         "--port", str(PORT),
         "--fp32"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
    )
    return proc


async def wait_ready(proc: subprocess.Popen, timeout: float = 120.0) -> bool:
    import httpx

    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            async with httpx.AsyncClient(trust_env=False, timeout=2.0) as client:
                r = await client.get(f"http://{HOST}:{PORT}/status")
                if r.status_code == 200 and r.json().get("ready"):
                    return True
        except Exception:
            pass
        await asyncio.sleep(1)
    return False


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("concurrency", type=int, nargs="?", default=100)
    parser.add_argument("--repeats", type=int, default=3, help="每并发轮重复次数")
    args = parser.parse_args()

    import httpx
    import requests

    print(f"[test] 环境: {sys.executable}")
    print(f"[test] 索引: {INDEX} (存在: {os.path.exists(INDEX)})")
    print(f"[test] 语料: {CORPUS} (存在: {os.path.exists(CORPUS)})")
    print(f"[test] 模型: {MODEL} (存在: {os.path.exists(MODEL)})")
    if not all(os.path.exists(p) for p in (INDEX, CORPUS, MODEL)):
        print("[test] 数据不完整，退出")
        sys.exit(1)

    # 起服务
    proc = start_server()
    print(f"[test] 服务进程 PID={proc.pid} 启动中（首次要加载 e5 模型 + 索引）...")
    ready = await wait_ready(proc, timeout=180)
    if not ready:
        print("[test] 服务启动失败或超时。服务日志：")
        out = proc.stdout.read() if proc.stdout else ""
        print(out[-3000:] if out else "(无日志)")
        proc.kill()
        sys.exit(1)
    print("[test] 服务就绪")

    # 预热一次
    try:
        requests.post(URL, json={"queries": [QUERY_POOL[0]], "topk": 3, "return_scores": True}, timeout=10, proxies={"http": None, "https": None})
    except Exception as e:
        print(f"[test] 预热请求失败: {e}")

    async with httpx.AsyncClient(trust_env=False, timeout=30.0) as client:
        async def one(qs: list[str]) -> float:
            t0 = time.perf_counter()
            try:
                r = await client.post(URL, json={"queries": qs, "topk": 3, "return_scores": True})
                r.raise_for_status()
                return time.perf_counter() - t0
            except Exception as e:
                return time.perf_counter() - t0  # 耗时仍计入（超时也算）

        all_lat = []
        fails = 0
        timeouts = 0
        for rep in range(args.repeats):
            # 100 并发，每个请求 1-3 个 query
            tasks = []
            for i in range(args.concurrency):
                n_q = QUERY_SIZES[i % len(QUERY_SIZES)]
                qs = [QUERY_POOL[(i * 7 + j) % len(QUERY_POOL)] for j in range(n_q)]
                tasks.append(asyncio.create_task(one(qs)))

            results = await asyncio.gather(*tasks)
            all_lat.extend(results)

        lat = sorted(all_lat)
        n = len(lat)
        print()
        print("=" * 50)
        print(f"[结果] 并发 {args.concurrency} × {args.repeats} 轮 = {n} 请求")
        print(f"[结果] 平均: {statistics.mean(lat) * 1000:.1f} ms")
        print(f"[结果] p50:  {lat[n // 2] * 1000:.1f} ms")
        print(f"[结果] p90:  {lat[int(0.9 * n)] * 1000:.1f} ms")
        print(f"[结果] p99:  {lat[int(0.99 * n)] * 1000:.1f} ms")
        print(f"[结果] max:  {lat[-1] * 1000:.1f} ms")
        print(f"[结果] 超过 3s (训练侧超时线): {sum(1 for x in lat if x > 3.0)}")
        print(f"[结果] 超过 10s: {sum(1 for x in lat if x > 10.0)}")
        print("=" * 50)

    # 服务还活着吗？（死锁/崩溃检查）
    try:
        r = requests.get(f"http://{HOST}:{PORT}/status", timeout=3, proxies={"http": None, "https": None})
        print(f"[结果] 服务存活: 是 (status {r.status_code})")
    except Exception:
        print("[结果] 服务存活: 否 —— 并发把服务打挂了（死锁/崩溃）！")
        # 服务日志尾部（崩溃原因 / 死锁线程栈）
        try:
            out = proc.stdout.read() if proc.stdout else ""
        except Exception:
            out = ""
        if out:
            print("===== 服务日志尾部 =====")
            print(out[-3000:] if len(out) > 3000 else out)
        proc.kill()
        sys.exit(2)

    proc.kill()
    print("[test] 完成")


if __name__ == "__main__":
    asyncio.run(main())
