"""训练与采样骨架。

对齐 verl 的分工，但先合并成一个类：
    verl 侧:
      ActorRolloutRefWorker  —— 持有 FSDP 模型与优化器，负责 update_actor / update_weights
      AgentLoopWorker        —— 独立的 Ray actor，跑 agent loop、通过 HTTP 调推理服务
      AgentLoopManager       —— 切 batch、分发给多个 AgentLoopWorker、汇总
    我们单机单卡、权重就在进程里，所以把「采样」和「更新」放进同一个 worker，
    但保留 generate / update_actor 两个明确入口，将来要拆 Ray actor 时不用改调用方。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .agentworker import AgentWorker, Trajectory

logger = logging.getLogger(__name__)


@dataclass
class RolloutConfig:
    max_concurrency: int = 8
    max_turns: int = 6
    collect_trajectory: bool = True
    system_prompt: Optional[str] = None


@dataclass
class QuerySample:
    """一条训练样本：一个 query、一篇被浏览过的正例、若干被召回但没看的负例。"""

    query: str
    positive: str
    negatives: list[str] = field(default_factory=list)
    prompt: str = ""
    turn: int = -1  # 轨迹里这次搜索是第几次工具调用，排查用


class RolloutActorWorker:
    """采样侧 + 更新侧的统一入口。

    generate(prompt)          —— 单条采样，直接调 agent loop
    generate_batch(prompts)   —— 批量并发采样
    update_actor(samples)     —— 用正负样本微调检索器编码器（需要本进程持有 retriever）

    注意：微调改的是编码器本身，所以要求本进程能拿到编码器和语料；
    只走 HTTP 检索的 worker 做不了这件事（那边没有语料，也没有可反传的模型）。
    """

    def __init__(
        self,
        llm,
        tools: Optional[dict] = None,
        config: Optional[RolloutConfig] = None,
        retriever=None,
    ) -> None:
        self.config = config or RolloutConfig()
        self.agent = AgentWorker(
            llm=llm,
            tools=tools,
            max_concurrency=self.config.max_concurrency,
            max_turns=self.config.max_turns,
            system_prompt=self.config.system_prompt,
        )
        self.retriever = retriever
        self.global_steps = 0

    # ---------- 采样 ----------
    async def generate(self, prompt: str) -> Trajectory:
        """单条 prompt 跑一遍 agent，返回轨迹。"""
        return await self.agent.agent.run(prompt)

    async def generate_batch(self, prompts: Sequence[str]) -> list[Trajectory]:
        """批量采样，内部 create_task + Semaphore 限流。"""
        return await self.agent.run_batch(list(prompts))

    # ---------- 更新 ----------
    def update_actor(
        self,
        samples: Sequence[QuerySample],
        *,
        epochs: int = 1,
        lr: float = 1e-6,
        weight_decay: float = 0.0,
        batch_size: int = 256,
        micro_batch: int = 8,
        tau: float = 0.02,
        max_doc_len: int = 512,
        freeze_below: int = 0,
        doc_from_index: bool = True,
        out_model: Optional[str] = None,
    ) -> dict:
        """用正负样本微调检索器，跑 epochs 个 epoch。

        对应 verl 的 ActorRolloutRefWorker.update_actor。默认 doc_from_index=True 是
        查询侧微调：文档向量直接取索引里的现成向量，索引永远有效。
        若设成 False 则两侧都训，训完必须重建索引（见 finetune.reencode_index）。
        batch_size 是一次优化器更新的样本数，内部按 micro_batch 分块累积梯度。
        freeze_below=N 冻结底部 N 层 transformer，只训上面几层，用来压显存。
        """
        from .finetune import finetune_epoch, save_model

        if not samples:
            raise ValueError("没有样本可训")
        if self.retriever is None:
            raise ValueError(
                "微调需要编码器和语料：构造 RolloutActorWorker 时给 retriever=FlatRetriever(...)"
            )

        stats: dict = {"epochs": epochs, "doc_from_index": doc_from_index}
        if freeze_below > 0:
            from .finetune import freeze_below as _freeze

            stats["trainable"] = _freeze(self.retriever._model, freeze_below)

        for ep in range(1, epochs + 1):
            ep_stats = finetune_epoch(
                samples,
                self.retriever,
                lr=lr,
                weight_decay=weight_decay,
                batch_size=batch_size,
                micro_batch=micro_batch,
                tau=tau,
                max_doc_len=max_doc_len,
                doc_from_index=doc_from_index,
            )
            stats = {**stats, **ep_stats}

        stats["global_steps"] = self.global_steps
        if out_model:
            save_model(self.retriever, out_model)
            stats["model_dir"] = out_model
        return stats

    def set_global_steps(self, steps: int) -> None:
        self.global_steps = int(steps)


# ---------- 轨迹处理 ----------
def trajectory_as_dict(traj: Trajectory) -> dict:
    return {
        "prompt": traj.prompt,
        "final_answer": traj.final_answer,
        "turns": traj.turns,
        "error": traj.error,
        "elapsed": round(traj.elapsed, 4),
        "documents": traj.documents,
        "tool_calls": [
            {"name": c.name, "arguments": c.arguments, "result": c.result, "documents": c.documents}
            for c in traj.tool_calls
        ],
    }


def summarize_trajectories(trajs: Sequence[Trajectory]) -> dict:
    n = len(trajs)
    if n == 0:
        return {"n": 0}
    done = [t for t in trajs if t.error is None and t.final_answer]
    return {
        "n": n,
        "completed": len(done),
        "failed": n - len(done),
        "avg_turns": round(sum(t.turns for t in trajs) / n, 2),
        "avg_tool_calls": round(sum(len(t.tool_calls) for t in trajs) / n, 2),
        "avg_docs": round(sum(len(t.documents) for t in trajs) / n, 2),
        "avg_elapsed": round(sum(t.elapsed for t in trajs) / n, 3),
    }


def save_trajectories(trajs: Sequence[Trajectory], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in trajs:
            f.write(json.dumps(trajectory_as_dict(t), ensure_ascii=False) + "\n")
    return path


def collect_samples(trajs: Sequence[Trajectory]) -> list[QuerySample]:
    """从轨迹里挖正负样本：只认「搜索紧跟浏览」这一对。

    规则（用户设定）：
        - 搜索之后的那一次调用必须是浏览，才构成一对；
        - 浏览到的那篇 = 正样本；
        - 同一次搜索里被召回、但没被浏览的 = 负样本。

    搜索后面没接浏览的（直接给答案、或者继续搜），整段丢弃 —— 那是"召回但没验证"，
    拿它当负例会教检索器把好文档推远。
    """
    samples: list[QuerySample] = []
    for traj in trajs:
        calls = traj.tool_calls
        for i, call in enumerate(calls):
            if call.name != "search":
                continue
            nxt = calls[i + 1] if i + 1 < len(calls) else None
            if nxt is None or nxt.name != "browse" or not nxt.documents:
                continue
            positive = str(nxt.documents[0])
            recalled = [str(d) for d in call.documents]
            if positive not in recalled:  # 浏览的不是这次搜索召回的，配对不成立
                continue
            query = call.arguments.get("query") or call.arguments.get("q")
            if not query:
                continue
            samples.append(
                QuerySample(
                    query=str(query),
                    positive=positive,
                    negatives=[d for d in recalled if d != positive],
                    prompt=traj.prompt,
                    turn=i,
                )
            )
    return samples


def save_samples(samples: Sequence[QuerySample], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(
                json.dumps(
                    {
                        "query": s.query,
                        "positive": s.positive,
                        "negatives": s.negatives,
                        "prompt": s.prompt,
                        "turn": s.turn,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def samples_from_eval(path: str) -> list[QuerySample]:
    """从 eval 产出的 trajectories.jsonl 里挖样本，规则与 collect_samples 一致。"""
    from .agentworker import ToolCall, Trajectory

    out: list[QuerySample] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            traj = Trajectory(prompt=r.get("query", ""), final_answer=r.get("pred"))
            traj.tool_calls = [
                ToolCall(c.get("name", ""), c.get("arguments") or {}, "", c.get("documents") or [])
                for c in r.get("tool_calls", [])
            ]
            out.extend(collect_samples([traj]))
    return out


def summarize_samples(samples: Sequence[QuerySample]) -> dict:
    n = len(samples)
    if n == 0:
        return {"n": 0}
    negs = [len(s.negatives) for s in samples]
    return {
        "n": n,
        "no_negative": sum(1 for x in negs if x == 0),
        "avg_negatives": round(sum(negs) / n, 2),
    }


def push_adapter(
    retriever_url: str,
    path: str,
    global_steps: Optional[int] = None,
    version: Optional[str] = None,
    timeout: float = 30.0,
) -> dict:
    """把落盘的 adapter 路径推给检索服务。同步调用，调用方自己决定是否放线程里。"""
    import httpx

    resp = httpx.post(
        f"{retriever_url.rstrip('/')}/adapter",
        json={"path": path, "global_steps": global_steps, "version": version},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


async def train(
    worker: RolloutActorWorker,
    prompts: Sequence[str],
    *,
    num_steps: int = 1,
    output_dir: Optional[str] = None,
    update: bool = True,
    epochs: int = 1,
    lr: float = 1e-6,
    batch_size: int = 2,
    max_doc_len: int = 512,
    freeze_below: int = 0,
    reindex_to: Optional[str] = None,
) -> list[dict]:
    """闭环：采样 → 挖正负样本 → 落盘 → 微调检索器（可选重建索引）。

    微调改的是编码器权重，所以重建索引不是可选项而是代价：索引不重建，检索就会
    用旧向量对新 query 打分，训练等于白做。reindex_to 给了才重建（10 万篇要几十分钟）。

    每步产出：
        {"step", "stats", "samples", "trajectory_file", "sample_file",
         "update", "reindex", "seconds"}
    """
    history: list[dict] = []
    for step in range(1, num_steps + 1):
        t0 = time.time()
        worker.set_global_steps(step)

        trajs = await worker.generate_batch(prompts)
        stats = summarize_trajectories(trajs)

        traj_file = None
        if output_dir:
            traj_file = save_trajectories(trajs, os.path.join(output_dir, f"step{step:04d}.jsonl"))

        samples = collect_samples(trajs)
        sample_file = None
        if output_dir:
            sample_file = save_samples(samples, os.path.join(output_dir, f"step{step:04d}_samples.jsonl"))

        upd, reindex = None, None
        if update and samples:
            out_model = os.path.join(output_dir or ".", f"model_step{step:04d}")
            try:
                upd = await asyncio.to_thread(
                    worker.update_actor,
                    samples,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    max_doc_len=max_doc_len,
                    freeze_below=freeze_below,
                    out_model=out_model,
                )
                if reindex_to:
                    from .finetune import reencode_index

                    paths = await asyncio.to_thread(
                        reencode_index, worker.retriever, reindex_to, max_doc_len=max_doc_len
                    )
                    reindex = {"dir": reindex_to, "shards": len(paths)}
            except Exception as exc:  # 训练/重建失败都不该中断采样
                logger.warning("step %d 微调或重建索引失败: %s", step, exc)
                upd = {"error": f"{type(exc).__name__}: {exc}"}

        record = {
            "step": step,
            "stats": stats,
            "samples": summarize_samples(samples),
            "trajectory_file": traj_file,
            "sample_file": sample_file,
            "update": upd,
            "reindex": reindex,
            "seconds": round(time.time() - t0, 2),
        }
        history.append(record)
        logger.info("step %d | %s | 样本 %s | %.1fs", step, stats, record["samples"], record["seconds"])
    return history


# ---------- Ray 支持 ----------
def init_ray(**kwargs) -> None:
    """单机初始化 Ray：关掉 dashboard、复用已有集群，避免占 8265 端口。"""
    import ray

    if ray.is_initialized():
        return
    kwargs.setdefault("include_dashboard", False)
    kwargs.setdefault("ignore_reinit_error", True)
    kwargs.setdefault("logging_level", logging.WARNING)
    ray.init(**kwargs)


def make_remote_worker(**kwargs):
    """把 RolloutActorWorker 包成独立的 Ray actor，rollout 跑在自己的进程里。

    用法：
        init_ray()
        handle = make_remote_worker(llm=..., tools=...)
        traj = await handle.generate.remote("prompt")

    注意：构造参数必须可 pickle。Retriever 持有 parquet 文件句柄和索引张量，
    不能直接传进来 —— 传路径参数，让 actor 在自己进程里构建。
    """
    import ray

    return ray.remote(RolloutActorWorker).remote(**kwargs)
