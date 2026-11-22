"""检索器微调：直接用轨迹里的正负样本训编码器（不再走 query 侧 adapter）。

代价必须先说清：编码器参数一改，**全库文档向量全部失效，索引必须重建**。
重建函数也在这个文件里，按 parquet 顺序流式编码，不按 docid 随机读。

训练目标照 LRAT：batch 内每个 query 只和自己的那一组（1 正 + K 负）比较，
InfoNCE 温度 0.02，学习率 1e-6 起步。
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import pickle
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def freeze_below(model, n: int) -> int:
    """冻结底部 n 层 transformer 和词嵌入，返回剩余可训参数量。

    注意 AutoModel 加载出来的是 Qwen3Model，层在 model.layers；
    若是 *ForCausalLM 则在 model.model.layers，两种都兼容。
    """
    layers = getattr(model, "layers", None)
    if layers is None:
        layers = model.model.layers
    for i, layer in enumerate(layers):
        if i < n:
            for p in layer.parameters():
                p.requires_grad_(False)
    emb = getattr(model, "embed_tokens", None)
    if emb is None and n > 0:
        emb = model.model.embed_tokens
    if emb is not None and n > 0:
        for p in emb.parameters():
            p.requires_grad_(False)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _encode(retriever, texts: Sequence[str], *, max_len: int):
    """带梯度的编码（训练不能走 no_grad）。返回已 L2 归一化的向量。"""
    tok, model, device = retriever._tokenizer, retriever._model, retriever._device
    inputs = tok(
        list(texts), max_length=max_len, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    hidden = model(**inputs).last_hidden_state
    last = inputs["attention_mask"].sum(dim=1) - 1
    emb = hidden[torch.arange(hidden.size(0), device=device), last]
    return F.normalize(emb.float(), p=2, dim=-1)


def finetune_epoch(
    samples: Sequence,
    retriever,
    *,
    lr: float = 1e-6,
    weight_decay: float = 0.0,
    batch_size: int = 256,
    micro_batch: int = 8,
    tau: float = 0.02,
    max_query_len: int = 128,
    max_doc_len: int = 512,
    max_negatives: int = 9,
    doc_from_index: bool = True,
) -> dict:
    """跑一个 epoch 的 InfoNCE 微调。samples 需有 .query/.positive/.negatives。

    batch_size 是「一次优化器更新的样本数」，内部按 micro_batch 切小块累积梯度，
    这样 256 这种大 batch 在 24G 内存上也能跑。

    doc_from_index=True 时文档向量直接取索引里的现成向量（冻结），只有 query 侧回传梯度。
    这是查询侧微调：索引永远有效，不用重编码语料。False 则两侧都训，
    训完必须重建索引。
    """
    if retriever._model is None:
        raise ValueError("微调需要编码器：构造 FlatRetriever 时给 model_path")
    if not doc_from_index and retriever.corpus is None:
        raise ValueError("文档侧也训练时需要语料：构造 FlatRetriever 时给 corpus_path")

    model = retriever._model
    device = retriever._device
    # fp16 参数上直接做梯度会下溢（实测 loss 变 nan），训练一律用 fp32
    if next(model.parameters()).dtype != torch.float32:
        model.float()
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)

    total_loss = 0.0
    total_correct = 0
    total = 0
    n_steps = 0

    for start in range(0, len(samples), batch_size):
        batch = list(samples[start : start + batch_size])
        opt.zero_grad()
        n_micro = 0
        batch_loss = 0.0

        for mstart in range(0, len(batch), micro_batch):
            micro = batch[mstart : mstart + micro_batch]
            q_texts = [f"{retriever.instruction} {s.query}" for s in micro]

            group_ids: list[list[str]] = []
            for s in micro:
                ids = [str(s.positive)] + [str(d) for d in s.negatives if str(d) != str(s.positive)]
                group_ids.append(ids[: max_negatives + 1])

            flat_ids = [d for g in group_ids for d in g]
            if not flat_ids:
                continue

            qe = _encode(retriever, q_texts, max_len=max_query_len)  # 只有 query 侧要梯度
            if doc_from_index:
                de = retriever.doc_vector(flat_ids).to(device)
                de = F.normalize(de.float(), p=2, dim=-1)
            else:
                de = _encode(retriever, [retriever.corpus.get(d) for d in flat_ids], max_len=max_doc_len)

            groups = [len(g) for g in group_ids]
            offsets, col = [], 0
            for n in groups:
                offsets.append(col)
                col += n

            logits = (qe @ de.T) / tau
            # 块对角掩码：每个 query 只和它自己那一组文档比较
            mask = torch.zeros_like(logits, dtype=torch.bool)
            for r, n in enumerate(groups):
                if n:
                    mask[r, offsets[r] : offsets[r] + n] = True
            logits = logits.masked_fill(~mask, float("-inf"))

            target = torch.tensor(offsets, dtype=torch.long, device=device)
            valid = torch.tensor([n > 0 for n in groups], device=device)
            if not bool(valid.any()):
                continue

            loss = F.cross_entropy(logits[valid], target[valid])
            (loss / max(1, len(batch) // micro_batch + 1)).backward()

            with torch.no_grad():
                pred = logits[valid].argmax(dim=1)
                total_correct += int((pred == target[valid]).sum().item())
            n_valid = int(valid.sum().item())
            total_loss += float(loss.item()) * n_valid
            total += n_valid
            batch_loss += float(loss.item())
            n_micro += 1

        if n_micro:
            opt.step()
            opt.zero_grad()
            n_steps += 1

    model.eval()
    return {
        "n": total,
        "steps": n_steps,
        "micro_batches": n_micro,
        "loss": round(total_loss / max(total, 1), 4),
        "acc": round(total_correct / max(total, 1), 4),
    }


def save_model(retriever, out_dir: str, half: bool = True) -> str:
    """把微调后的编码器落盘，目录结构和原始 Qwen3-Embedding 一致，可直接被加载。

    训练时为了梯度稳定转成了 fp32，落盘默认再转回 fp16，省一半磁盘和显存。
    """
    os.makedirs(out_dir, exist_ok=True)
    model = retriever._model
    if half and next(model.parameters()).dtype == torch.float32:
        model.half()
    model.save_pretrained(out_dir)
    retriever._tokenizer.save_pretrained(out_dir)
    logger.info("编码器已保存: %s (%s)", out_dir, next(model.parameters()).dtype)
    return out_dir


def reencode_index(
    retriever,
    out_dir: str,
    *,
    shard_size: int = 25049,
    batch_size: int = 16,
    max_doc_len: Optional[int] = None,
) -> list[str]:
    """用当前编码器重编码整个语料，写成 Tevatron 分片 pkl。

    顺序：直接流式读 parquet（不按 docid 随机读），所以顺序会变成语料文件顺序，
    docid 一起写进分片，顺序变了不影响正确性。
    """
    import pyarrow.parquet as pq

    if retriever._model is None or retriever.corpus is None:
        raise ValueError("重建索引需要编码器和语料")
    os.makedirs(out_dir, exist_ok=True)
    retriever._model.eval()

    path = os.path.abspath(retriever.corpus.path)
    files = (
        sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
        if os.path.isdir(path)
        else [path]
    )

    buf_vecs: list[np.ndarray] = []
    buf_ids: list[str] = []
    out_paths: list[str] = []
    shard = 1
    total = 0

    def flush() -> None:
        nonlocal buf_vecs, buf_ids, shard
        if not buf_ids:
            return
        p = os.path.join(out_dir, f"corpus.shard{shard}_of_X.pkl")
        with open(p, "wb") as f:
            pickle.dump((np.vstack(buf_vecs), buf_ids), f)
        out_paths.append(p)
        logger.info("写出分片 %s: %d 条", p, len(buf_ids))
        buf_vecs, buf_ids = [], []
        shard += 1

    for fp in files:
        pf = pq.ParquetFile(fp)
        for rb in pf.iter_batches(batch_size=batch_size, columns=["docid", "text"]):
            docids = [str(x) for x in rb.column("docid").to_pylist()]
            texts = [str(x) for x in rb.column("text").to_pylist()]
            with torch.no_grad():
                emb = _encode(retriever, texts, max_len=max_doc_len or retriever.max_length)
            buf_vecs.append(emb.cpu().numpy().astype(np.float32))
            buf_ids.extend(docids)
            total += len(docids)
            if len(buf_ids) >= shard_size:
                flush()
    flush()
    logger.info("索引重建完成: %d 条, %d 个分片", total, len(out_paths))
    return out_paths


def load_samples(paths: Iterable[str]) -> list:
    from .training import QuerySample

    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                out.append(
                    QuerySample(
                        query=d["query"],
                        positive=str(d["positive"]),
                        negatives=[str(x) for x in d.get("negatives", [])],
                        prompt=d.get("prompt", ""),
                        turn=int(d.get("turn", -1)),
                    )
                )
    return out


def main() -> None:
    from .retriever import FlatRetriever

    ap = argparse.ArgumentParser(description="用轨迹正负样本微调检索器")
    ap.add_argument("--samples", nargs="+", required=True, help="stepXXXX_samples.jsonl")
    ap.add_argument("--index-glob", required=True, help="现有索引，用来初始化（之后会被重建替换）")
    ap.add_argument("--corpus-path", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--max-doc-len", type=int, default=512)
    ap.add_argument("--out-model", default=None, help="微调后编码器保存目录")
    ap.add_argument("--out-index-dir", default=None, help="重建索引的输出目录")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    retriever = FlatRetriever(
        index_glob=args.index_glob,
        corpus_path=args.corpus_path,
        model_path=args.model_path,
        device=args.device,
        max_length=args.max_doc_len,
    )
    samples = load_samples(args.samples)
    logger.info("样本 %d 条", len(samples))

    for ep in range(1, args.epochs + 1):
        stats = finetune_epoch(
            samples,
            retriever,
            lr=args.lr,
            batch_size=args.batch_size,
            tau=args.tau,
            max_doc_len=args.max_doc_len,
        )
        logger.info("epoch %d/%d: %s", ep, args.epochs, stats)

    if args.out_model:
        save_model(retriever, args.out_model)
    if args.out_index_dir:
        paths = reencode_index(retriever, args.out_index_dir, max_doc_len=args.max_doc_len)
        logger.info("新索引: %s", paths)


if __name__ == "__main__":
    main()
