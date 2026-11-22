"""query 侧 adapter：只改 query 向量，doc 侧和索引全程冻结。

结构刻意做到最小：
    e_q  ->  normalize(e_q + alpha * (W e_q + b))
W、b 零初始化 ⇒ 起点与原检索器完全等价；残差形式保证只做增量修正，
不会一上来就把原表征打散。alpha 用来控制 adapter 的干预强度。

训练只做一件事：同一 query 的正例（被浏览过）排在负例（被召回但没看）前面。
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class QueryAdapter(nn.Module):
    """残差式线性 adapter，输入输出同维。"""

    def __init__(self, dim: int = 1024, alpha: float = 1.0) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.dim = dim
        self.alpha = alpha

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        delta = self.proj(emb)
        return F.normalize(emb + self.alpha * delta, p=2, dim=-1)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, alpha={self.alpha}"


def save_adapter(adapter: QueryAdapter, path: str, meta: dict | None = None) -> str:
    """落盘：state_dict + 维度 + alpha，供检索服务直接 load。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
            "dim": adapter.dim,
            "alpha": adapter.alpha,
            "meta": dict(meta or {}),
        },
        path,
    )
    return path


def load_adapter(path: str) -> QueryAdapter:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    adapter = QueryAdapter(dim=int(blob["dim"]), alpha=float(blob.get("alpha", 1.0)))
    adapter.load_state_dict(blob["state_dict"])
    adapter.requires_grad_(False)  # 推理侧只要前向，别带梯度
    return adapter.eval()


def train_one_epoch(
    samples: Sequence,
    *,
    encode_queries: Callable[[Sequence[str]], torch.Tensor],
    doc_vectors: Callable[[Sequence[str]], torch.Tensor],
    adapter: QueryAdapter,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    batch_size: int = 16,
    tau: float = 0.05,
    device: str = "cpu",
) -> dict:
    """跑一个 epoch 的 listwise 对比学习。

    samples 只需有 .query / .positive / .negatives 三个属性。
    同一批里其他样本的正例会被补成本条的负例（in-batch negatives），
    这样即使某条自带的负例很少，也不会退化成零梯度。
    """
    adapter = adapter.to(device).train()
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=weight_decay)

    total_loss = 0.0
    total_correct = 0
    total = 0
    n_steps = 0

    for start in range(0, len(samples), batch_size):
        batch = list(samples[start : start + batch_size])
        q_raw = encode_queries([s.query for s in batch]).to(device)
        q = adapter(q_raw)

        cand_ids: list[list[str]] = []
        for k, s in enumerate(batch):
            cands = [str(s.positive)] + [str(d) for d in s.negatives if str(d) != str(s.positive)]
            for j, other in enumerate(batch):
                if j != k:
                    p = str(other.positive)
                    if p not in cands:
                        cands.append(p)
            cand_ids.append(cands)

        flat = list(dict.fromkeys(d for cands in cand_ids for d in cands))
        where = {d: i for i, d in enumerate(flat)}
        dv = doc_vectors(flat).to(device)
        dv = F.normalize(dv, p=2, dim=-1)

        losses = []
        for k in range(len(batch)):
            cols = [where[d] for d in cand_ids[k]]
            logits = (q[k] @ dv[cols].T) / tau
            target = torch.zeros(1, dtype=torch.long, device=device)
            losses.append(F.cross_entropy(logits.unsqueeze(0), target))
            total_correct += int(int(logits.argmax().item()) == 0)

        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += float(loss.item()) * len(batch)
        total += len(batch)
        n_steps += 1

    adapter.eval()
    return {
        "n": total,
        "steps": n_steps,
        "loss": round(total_loss / max(total, 1), 4),
        "acc": round(total_correct / max(total, 1), 4),
    }
