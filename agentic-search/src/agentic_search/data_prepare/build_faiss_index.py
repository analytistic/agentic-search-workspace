"""Build a FAISS dense index from a JSONL corpus, matching the official
Search-R1 / SSP e5_Flat.index construction.

Pipeline (mirrors Search-R1's ``search_r1/search/index_builder.py``):
    corpus rows (in file order) -> e5 "passage: ..." -> mean pooling
    -> L2 normalize -> faiss.index_factory(dim, "Flat", METRIC_INNER_PRODUCT)
    -> add -> write_index

The output is byte-compatible with the official flat index: the retrieval
server does ``faiss.read_index`` and maps ``corpus[int(idx)]``, so index row
*i* corresponds to corpus line *i*.

Usage:
    # 单进程（小语料）:
    uv run --project agentic-search python -m agentic_search.data_prepare.build_faiss_index \
        --corpus data/wiki-demo/corpus.jsonl \
        --output data/wiki-demo/e5_demo.index \
        --model data/wiki_2018/model/intfloat/e5-base-v2

    # 两段式（大语料，torch 编码与 faiss 建索引用独立进程，避开
    # macOS 上 torch/faiss 线程库冲突导致的段错误）:
    uv run --project agentic-search python -m agentic_search.data_prepare.build_faiss_index \
        --corpus data/wiki-demo/corpus.jsonl --embeddings-out /tmp/emb.npy --model <e5路径>
    uv run --project agentic-search python -m agentic_search.data_prepare.build_faiss_index \
        --embeddings-in /tmp/emb.npy --output data/wiki-demo/e5_demo.index
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

POOLING_METHOD = "mean"
PASSAGE_PREFIX = "passage: "


def load_corpus_rows(corpus_path: Path) -> list[str]:
    """Read a JSONL corpus in file order; keep the ``contents`` field only."""
    rows: list[str] = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
            contents = doc.get("contents", "")
            if not contents:
                raise ValueError(f"Missing 'contents' at line {line_no}")
            rows.append(contents)
    return rows


def encode_documents(
    model_path: Path,
    rows: list[str],
    batch_size: int,
    device: str,
) -> np.ndarray:
    # Lazy imports: keep torch out of the faiss process (see module docstring).
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModel.from_pretrained(model_path).to(device).eval()

    embeddings = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = [f"{PASSAGE_PREFIX}{doc}" for doc in rows[start : start + batch_size]]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            output = model(**inputs)
            # mean pooling over non-pad tokens
            mask = inputs["attention_mask"][..., None].bool()
            emb = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            embeddings.append(emb.detach().cpu().numpy())
            print(f"  encoded {min(start + batch_size, len(rows))}/{len(rows)}")

    return np.concatenate(embeddings, axis=0).astype(np.float32)


def build_index(embeddings: np.ndarray, output_path: Path) -> None:
    # Lazy import: keep faiss out of the torch process (see module docstring).
    import faiss

    dim = embeddings.shape[-1]
    index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    if not index.is_trained:
        index.train(embeddings)
    index.add(embeddings)
    faiss.write_index(index, str(output_path))
    print(f"index rows: {index.ntotal}, dim: {dim}, metric: IP (cosine)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, help="JSONL corpus (contents field).")
    parser.add_argument("--output", type=Path, help="Output .index file path.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("data/wiki_2018/model/intfloat/e5-base-v2"),
        help="Path to the e5 embedding model.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if _cuda_available() else "cpu")
    parser.add_argument("--embeddings-out", type=Path, help="Stage 1: encode and save embeddings (.npy), then exit.")
    parser.add_argument("--embeddings-in", type=Path, help="Stage 2: load embeddings (.npy) and build the index.")
    return parser.parse_args()


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def main() -> None:
    args = parse_args()

    if args.embeddings_in is not None:
        emb = np.load(args.embeddings_in)
        assert args.output is not None
        print(f"embeddings loaded: {emb.shape}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        build_index(emb, args.output)
        return

    assert args.corpus is not None
    rows = load_corpus_rows(args.corpus)
    print(f"corpus rows: {len(rows)}")
    emb = encode_documents(args.model, rows, args.batch_size, args.device)
    print(f"embeddings: {emb.shape}")

    if args.embeddings_out is not None:
        args.embeddings_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.embeddings_out, emb)
        print(f"saved embeddings: {args.embeddings_out}")
        return

    assert args.output is not None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_index(emb, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
