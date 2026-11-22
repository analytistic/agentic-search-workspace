"""扁平稠密检索器：numpy 内积，设备只走 cuda / mps / cpu。

对应 BrowseComp-Plus 的 *-embedding 分片索引（Tevatron 导出的 pkl: (vectors, docids)）。
不做 ANN：10 万条 × 1024 维的内积在 CPU 上也是毫秒级，先要正确，再谈快。

对外两个入口，方便把 query 侧 adapter 插进来：
    search_text(query)    —— 自己编码
    search_vector(vec)    —— 直接用外部向量，跳过编码
"""

from __future__ import annotations

import glob
import json
import logging
import os
import pickle
import threading
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

QUERY_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
)


@dataclass
class Hit:
    docid: str
    score: float
    text: str = ""


def pick_device(prefer: str = "auto"):
    """cuda > mps > cpu。"""
    import torch

    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Corpus:
    """按 docid 取原文。支持单文件 jsonl（行偏移）或 parquet（row group 缓存）。"""

    def __init__(self, path: str, cache_groups: int = 4) -> None:
        self.path = os.path.abspath(path)
        self.cache_groups = cache_groups
        self._where: dict[str, tuple] = {}
        self._jsonl_fh = None
        self._pq_files: List[Any] = []
        self._rg_cache: dict[tuple, Any] = {}
        self._meta: Optional[dict] = None
        self._lock = threading.Lock()
        if self.path.endswith(".jsonl") or self.path.endswith(".json"):
            self._index_jsonl()
        elif self.path.endswith(".parquet") or os.path.isdir(self.path):
            self._index_parquet()
        else:
            raise ValueError(f"不认识的语料格式: {self.path}")

    # ---------- jsonl ----------
    def _index_jsonl(self) -> None:
        self._jsonl_fh = open(self.path, "rb")
        offset = 0
        count = 0
        for line in self._jsonl_fh:
            if not line.strip():
                offset += len(line)
                continue
            rec = json.loads(line)
            self._where[str(rec["docid"])] = (offset,)
            self._meta = self._meta or rec
            offset += len(line)
            count += 1
        logger.info("语料索引完成(jsonl): %d 行, 字段 %s", count, list((self._meta or {}).keys()))

    # ---------- parquet ----------
    def _index_parquet(self) -> None:
        import pyarrow.parquet as pq

        files = (
            sorted(glob.glob(os.path.join(self.path, "**", "*.parquet"), recursive=True))
            if os.path.isdir(self.path)
            else [self.path]
        )
        if not files:
            raise FileNotFoundError(f"目录下没有 parquet: {self.path}")
        for fi, fp in enumerate(files):
            pf = pq.ParquetFile(fp)
            self._pq_files.append(pf)
            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(rg, columns=["docid"])
                for row_in_group, docid in enumerate(table.column("docid").to_pylist()):
                    self._where[str(docid)] = (fi, rg, row_in_group)
        logger.info("语料索引完成(parquet): %d 篇, %d 个文件", len(self._where), len(files))

    def get(self, docid: str) -> str:
        key = str(docid)
        where = self._where.get(key)
        if where is None:
            return ""

        # pyarrow 的 ParquetFile 和 jsonl 文件句柄都不是线程安全的，
        # 服务端并发取原文会直接段错误（实测崩溃在 read_row_group）。
        with self._lock:
            if self._jsonl_fh is not None:
                self._jsonl_fh.seek(where[0])
                rec = json.loads(self._jsonl_fh.readline())
            else:
                fi, rg, _ = where
                cached = self._rg_cache.get((fi, rg))
                if cached is None:
                    if len(self._rg_cache) >= self.cache_groups:
                        self._rg_cache.pop(next(iter(self._rg_cache)))
                    cached = self._pq_files[fi].read_row_group(rg).to_pylist()
                    self._rg_cache[(fi, rg)] = cached
                rec = cached[where[2]]

        # BrowseComp-Plus 原文以 "---\ntitle: ..." 
        text = str(rec.get("text", ""))
        title = ""
        if text.startswith("---"):
            parts = text.split("\n")
            for line in parts:
                if line.lower().startswith("title:"):
                    title = line.split(":", 1)[1].strip().strip('"')
                    break
            text = "\n".join(parts[2:]) if len(parts) > 2 else text
        return f"{title}\n{text}".strip() if title else text

    def __contains__(self, docid: object) -> bool:
        return str(docid) in self._where

    def __len__(self) -> int:
        return len(self._where)


class FlatRetriever:
    """加载分片向量 + 语料，提供文本/向量两种检索入口。"""

    def __init__(
        self,
        index_glob: str,
        corpus_path: Optional[str] = None,
        model_path: Optional[str] = None,
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 16,
        instruction: str = QUERY_INSTRUCTION,
        dtype: str = "float16",
        cache_groups: int = 4,
        max_chars: int = 0,
        preview_tokens: int = 64,
    ) -> None:
        self.index_glob = index_glob
        self.model_path = model_path
        self.device_pref = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.instruction = instruction
        self.dtype = dtype
        self.max_chars = max_chars
        # 检索结果里每篇文档给多少 token 的预览；浏览工具才给全文
        self.preview_tokens = preview_tokens
        self._preview_cache: dict[tuple, str] = {}
        self._preview_cache_size = 4096

        self.emb, self.docids = self._load_index(index_glob)
        self._pos = {d: i for i, d in enumerate(self.docids)}
        self.corpus = Corpus(corpus_path, cache_groups=cache_groups) if corpus_path else None
        self._torch = None
        self._device = None
        self._model = None
        self._tokenizer = None
        # query 侧 adapter：接口先就位，权重由 update_adapter 驱动加载
        self._adapter = None
        self._pending_adapter = None
        self._adapter_version = "base"
        self._pending_version = None
        self._adapter_steps = 0

        if self.model_path:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._torch = torch
            self._device = pick_device(self.device_pref)
            torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[self.dtype]
            if self._device.type == "cpu":
                torch_dtype = torch.float32
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side="left")
            self._model = AutoModel.from_pretrained(self.model_path, dtype=torch_dtype).to(self._device)
            self._model.eval()
            logger.info("编码器加载完成: %s -> %s (%s)", self.model_path, self._device, torch_dtype)

    # ---------- 索引 ----------
    @staticmethod
    def _load_index(index_glob: str):
        import torch

        paths = sorted(glob.glob(index_glob))
        if not paths:
            raise FileNotFoundError(f"没有匹配的索引分片: {index_glob}")
        embs, ids = [], []
        for p in paths:
            with open(p, "rb") as f:
                vecs, docids = pickle.load(f)
            embs.append(np.asarray(vecs, dtype=np.float32))
            ids.extend(str(d) for d in docids)
        emb = torch.from_numpy(np.vstack(embs))
        # 归一化后内积即余弦；分片自带向量未保证已归一
        emb = emb / emb.norm(dim=1, keepdim=True).clamp_min(1e-12)
        logger.info("索引加载完成: %d 分片, %d 条, 维度 %d", len(paths), emb.shape[0], emb.shape[1])
        return emb, ids

    # ---------- 编码器 ----------
    # 只喂向量时不需要它，所以只在构造时给了 model_path 才加载。
    # (原 _ensure_model 懒加载已删：加载时机不再拖到第一次 encode)

    # ---------- adapter 权重接口 ----------
    def update_adapter(
        self,
        weights=None,
        *,
        path: Optional[str] = None,
        global_steps: Optional[int] = None,
        version: Optional[str] = None,
    ) -> dict:
        """接收 query 侧 adapter 权重（形状对齐 verl 的 rollout.update_weights）。

        verl 那边权重是以 (name, tensor) 生成器逐条产出的，不一次性物化，这里保持同样的
        入口形状。**本版只做接收、校验、暂存，不写进模型**：真正生效留给后续 commit，
        这样正在跑的检索不会读到半新半旧的权重。

        参数:
            weights: 可迭代的 (name, tensor)，或 None（此时只登记 path）
            path: adapter 权重文件路径，供后续 commit 读取
            global_steps: 训练步数，用于打版本号
            version: 显式版本号，优先于 global_steps

        返回:
            dict：新版本号、待生效张量数、是否已生效、当前生效版本
        """
        pending: dict = {}
        if weights is not None:
            for name, tensor in weights:
                pending[str(name)] = tensor
        if path:
            pending["__path__"] = path
        if not pending:
            raise ValueError("update_adapter 至少需要 weights 或 path 之一")

        dim = int(self.emb.shape[1])
        for name, tensor in pending.items():
            if name == "__path__":
                continue
            shape = getattr(tensor, "shape", None)
            if shape is not None and int(shape[-1]) != dim:
                raise ValueError(f"adapter 张量 {name} 末维 {tuple(shape)} 与索引维度 {dim} 不匹配")

        if global_steps is not None:
            self._adapter_steps = int(global_steps)
        self._pending_adapter = pending
        self._pending_version = version or f"step{self._adapter_steps}"
        return {
            "version": self._pending_version,
            "pending_tensors": len([k for k in pending if k != "__path__"]),
            "applied": False,
            "current_version": self._adapter_version,
        }

    def _apply_adapter(self, emb):
        """把当前生效的 adapter 作用到 query 向量上；未加载时原样返回。"""
        if self._adapter is None:
            return emb
        import torch

        with torch.no_grad():
            return self._adapter(emb)

    def commit_adapter(
        self,
        path: Optional[str] = None,
        *,
        version: Optional[str] = None,
        global_steps: Optional[int] = None,
    ) -> dict:
        """把 update_adapter 登记的权重真正换上去（双缓冲：新对象建好再改引用）。"""
        from .adapter import load_adapter

        path = path or (self._pending_adapter or {}).get("__path__")
        if not path:
            raise ValueError("没有可提交的 adapter 路径")
        adapter = load_adapter(path)
        dim = int(self.emb.shape[1])
        if adapter.dim != dim:
            raise ValueError(f"adapter 维度 {adapter.dim} 与索引维度 {dim} 不匹配")

        self._adapter = adapter  # 引用赋值，读请求要么看到旧的要么看到新的
        if global_steps is not None:
            self._adapter_steps = int(global_steps)
        self._adapter_version = version or f"step{self._adapter_steps}"
        self._pending_adapter = None
        self._pending_version = None
        logger.info("adapter 生效: %s (%d 参数)", self._adapter_version, sum(p.numel() for p in adapter.parameters()))
        return {"applied": True, "version": self._adapter_version, "steps": self._adapter_steps}

    def encode_raw(self, texts: Sequence[str], mode: str = "query"):
        """只出底座编码器的原始向量（不套 adapter）。训练侧要靠它拿 query 原始表征。"""
        if self._model is None:
            raise ValueError("编码器未加载：构造时没给 model_path，只能用 search_vector 传向量")
        torch = self._torch
        prefix = self.instruction if mode == "query" else ""
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = [f"{prefix} {t}" if prefix else t for t in texts[i : i + self.batch_size]]
                inputs = self._tokenizer(
                    batch, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt"
                ).to(self._device)
                hidden = self._model(**inputs).last_hidden_state
                last = inputs["attention_mask"].sum(dim=1) - 1
                emb = hidden[torch.arange(hidden.size(0), device=hidden.device), last]
                out.append(torch.nn.functional.normalize(emb.float(), p=2, dim=-1).cpu())
        return torch.cat(out)

    def encode(self, queries: Sequence[str]):
        """Qwen3-Embedding：加指令前缀、取最后一个有效 token、L2 归一化，再过 adapter。"""
        return self._apply_adapter(self.encode_raw(queries, mode="query"))

    # ---------- 按 docid 取向量 ----------
    def doc_vector(self, docids: Sequence[str]):
        """直接取索引里的文档向量（训练时不必重新编码文档）。"""
        idx = []
        for d in docids:
            pos = self._pos.get(str(d))
            if pos is None:
                raise KeyError(d)
            idx.append(pos)
        return self.emb[idx]

    # ---------- 原文与预览 ----------
    def get_document(self, docid: str, max_chars: int = 0) -> str:
        """按 docid 取全文（浏览工具用）。max_chars>0 时截断，默认整篇返回。"""
        if self.corpus is None:
            raise ValueError("没有加载语料，无法取原文")
        if docid not in self.corpus:
            raise KeyError(docid)
        text = self.corpus.get(docid)
        return text[:max_chars] if max_chars > 0 else text

    def _preview(self, docid: str, text: str, n_tokens: int) -> str:
        """截成 n_tokens 个 token 的预览，按 (docid, n) 缓存。"""
        if n_tokens <= 0 or not text:
            return text
        if self._tokenizer is None:
            # 没有分词器时退化为字符近似（约 4 字符/token）
            return text[: n_tokens * 4]
        key = (docid, n_tokens)
        cached = self._preview_cache.get(key)
        if cached is not None:
            return cached
        # 先按字符粗切，避免对五万字长文整体分词；8 字符/token 是安全上界
        ids = self._tokenizer(
            text[: n_tokens * 8], max_length=n_tokens, truncation=True, add_special_tokens=False
        )["input_ids"]
        out = self._tokenizer.decode(ids, skip_special_tokens=True)
        if len(self._preview_cache) >= self._preview_cache_size:
            self._preview_cache.pop(next(iter(self._preview_cache)))
        self._preview_cache[key] = out
        return out

    # ---------- 检索 ----------
    def _topk(self, vec, topk: int, with_text: bool, preview_tokens: Optional[int] = None) -> List[Hit]:
        import torch

        n_preview = self.preview_tokens if preview_tokens is None else preview_tokens
        q = vec if isinstance(vec, torch.Tensor) else torch.as_tensor(np.asarray(vec, dtype=np.float32))
        q = q.reshape(-1).to(self.emb.device, dtype=torch.float32)
        q = q / q.norm().clamp_min(1e-12)
        scores = self.emb @ q
        k = min(topk, scores.numel())
        vals, idx = torch.topk(scores, k)
        hits = []
        for score, i in zip(vals.tolist(), idx.tolist()):
            docid = self.docids[int(i)]
            text = ""
            if with_text and self.corpus:
                full = self.corpus.get(docid)
                text = self._preview(docid, full, n_preview) if n_preview > 0 else full
                if text and self.max_chars > 0:
                    text = text[: self.max_chars]
            hits.append(Hit(docid=docid, score=float(score), text=text))
        return hits

    def search_vector(self, vec, topk: int = 10, with_text: bool = True, preview_tokens: Optional[int] = None) -> List[Hit]:
        """直接给向量：adapter 产出的向量从这里进来，绕开编码器。"""
        return self._topk(vec, topk, with_text, preview_tokens)

    def search_text(self, query: str, topk: int = 10, with_text: bool = True, preview_tokens: Optional[int] = None) -> List[Hit]:
        return self._topk(self.encode([query])[0], topk, with_text, preview_tokens)

    def search_texts(self, queries: Sequence[str], topk: int = 10, with_text: bool = True, preview_tokens: Optional[int] = None) -> List[List[Hit]]:
        if not queries:
            return []
        embs = self.encode(queries)
        return [self._topk(e, topk, with_text, preview_tokens) for e in embs]

    def __len__(self) -> int:
        return self.emb.shape[0]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    idx = os.path.join(base, "data/isp/indexes/browsecomp-plus-indexes/qwen3-embedding-0.6b/corpus.*.pkl")
    corpus = os.path.join(base, "data/browsecomp_plus/raw/corpus")
    r = FlatRetriever(index_glob=idx, corpus_path=corpus)
    print("文档数:", len(r))
    # 不加载模型，用索引里现成的一条向量做冒烟：它应该把自己排第一
    probe = r.emb[12345]
    for h in r.search_vector(probe, topk=3, with_text=False):
        print(f"  {h.docid}  {h.score:.4f}")
