# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/search/retrieval_server.py
#
# Local changes vs. upstream (see RETRIEVAL_NOTES.md):
#   1. corpus is read lazily through a line-offset table instead of being
#      materialised in RAM (wiki-18: 14 GB / 21M docs) -- OffsetCorpus.
#   2. BM25 docid is validated against the corpus line number at startup so a
#      mismatched index/corpus pair fails fast instead of silently returning
#      the wrong documents.
#   3. per-request guards (max queries / max topk), LRU query cache, request
#      metrics, readiness-aware /status and configurable host/port/encoder.

import argparse
import collections
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch  # must precede faiss: on macOS, importing faiss first breaks torch's OpenMP init (segfault)
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("retrieval_server")

_DEFAULT_TOP_LEVEL = logging.INFO


def _read_index_mmap(path):
    """Memory-map a faiss index file instead of eagerly loading it.

    Newer faiss (>= 1.12) exposes mmap either as ``read_index_mmap`` or via
    the ``IO_FLAG_MMAP`` flag on ``read_index``; older builds fall back to a
    full in-RAM load so the server never breaks on an old faiss.

    Caveat: mmap only avoids the *eager* copy. Whether the index really stays
    out of RAM depends on the access pattern:

    * ``IndexFlat*`` / ``IVF*`` -- mostly sequential or clustered reads, so
      the page cache holds a working set far smaller than the file.
    * ``IndexHNSW*`` -- graph traversal jumps at random between neighbours,
      so the touched working set converges on the *whole* index. Expect the
      file to become resident; the OS may also evict it under memory
      pressure, which turns every hop into a page fault (queries get 10-100x
      slower). For HNSW, plan for `file_size` bytes of memory (or build a
      quantized index such as IndexHNSWPQ to shrink that number).
    """
    import faiss

    if hasattr(faiss, "read_index_mmap"):
        return faiss.read_index_mmap(path)
    if hasattr(faiss, "IO_FLAG_MMAP"):
        return faiss.read_index(path, faiss.IO_FLAG_MMAP)
    return faiss.read_index(path)


def _device() -> str:
    """Return the best available device: npu > cuda > mps > cpu."""
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            return "npu"
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


#####################################
# Corpus access
#####################################


class OffsetCorpus:
    """Lazy jsonl reader keyed by **line number**.

    Search-R1's ``load_corpus`` materialises the whole corpus (wiki-18: 14 GB,
    21M docs) with ``datasets.load_dataset``, which also writes an Arrow copy
    into ``~/.cache/huggingface``. That is ~30 GB of RAM + disk for a file we
    touch a few dozen lines of per request.

    Instead we build (once) an int64 offset table (~168 MB for 21M docs) and
    ``seek``/``readline`` on demand. Resident memory stays flat regardless of
    corpus size. The table is cached next to the corpus as
    ``offsets_<stem>.npy``.
    """

    def __init__(self, corpus_path: str, offsets_path: str = None):
        self.corpus_path = corpus_path
        if offsets_path is None:
            directory, name = os.path.split(os.path.abspath(corpus_path))
            stem = name[:-6] if name.endswith(".jsonl") else name
            offsets_path = os.path.join(directory, "offsets_%s.npy" % stem)
        self.offsets_path = offsets_path
        if not os.path.exists(offsets_path):
            self._build_offsets()
        self._np = np
        self.offsets = np.load(offsets_path, mmap_mode="r")
        self._fh = open(corpus_path, "rb")
        self._lock = threading.Lock()
        logger.info(
            "[offset] corpus ready: %s (%d docs, lazy)", corpus_path, len(self.offsets) - 1
        )

    def _build_offsets(self):
        from array import array

        t0 = time.time()
        offs = array("q")
        off = 0
        with open(self.corpus_path, "rb") as fh:
            for line in fh:
                offs.append(off)
                off += len(line)
        offs.append(off)
        # atomic-ish: write to a temp name then rename, so a killed build never
        # leaves a truncated table behind. np.save() appends ".npy" when given a
        # path, hence the file handle.
        tmp = self.offsets_path + ".tmp"
        with open(tmp, "wb") as fh:
            np.save(fh, np.frombuffer(offs, dtype=np.int64))
        os.replace(tmp, self.offsets_path)
        logger.info(
            "[offset] built %s (%d lines, %.1fs)", self.offsets_path, len(offs) - 1, time.time() - t0
        )

    def __len__(self):
        return len(self.offsets) - 1

    def get_raw(self, idx: int) -> dict:
        idx = int(idx)
        if idx < 0 or idx + 1 >= len(self.offsets):
            raise IndexError(idx)
        with self._lock:
            self._fh.seek(int(self.offsets[idx]))
            raw = self._fh.readline()
        return json.loads(raw)

    def __getitem__(self, idx: int) -> dict:
        try:
            doc = self.get_raw(idx)
        except IndexError:
            return {"title": "", "text": "", "contents": ""}
        content = doc.get("contents")
        if content is None:
            # fall back to separate title/text fields if the corpus was built
            # in that style instead of the Search-R1 "contents" style.
            title = doc.get("title") or ""
            text = doc.get("text") or ""
            content = (title + "\n" + text) if title else text
        parts = content.split("\n")
        return {
            "title": doc.get("title") or parts[0].strip('"'),
            "text": doc.get("text") if doc.get("text") is not None else "\n".join(parts[1:]),
            "contents": content,
        }

    def get_many(self, idxs) -> List[dict]:
        return [self[i] for i in idxs]


def load_corpus(corpus_path: str):
    """Eager in-RAM corpus (upstream behaviour). Only used with --eager-corpus."""
    import datasets

    return datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=16)


def load_docs(corpus, doc_idxs):
    return [corpus[int(idx)] for idx in doc_idxs]


def build_corpus(config):
    if config.eager_corpus:
        logger.warning(
            "[corpus] --eager-corpus: loading the whole corpus into RAM "
            "(this is what made wiki-18 need >30GB)"
        )
        t0 = time.time()
        corpus = load_corpus(config.corpus_path)
        logger.info("[corpus] eager load done (%d docs, %.1fs)", len(corpus), time.time() - t0)
        return corpus
    return OffsetCorpus(config.corpus_path)


#####################################
# Encoder
#####################################


def load_model(model_path: str, use_fp16: bool = False):
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.to(_device())
    if use_fp16 and _device() != "cpu":
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer


def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")


class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        self.model.eval()
        # Serialize encodes: concurrent requests would otherwise split the
        # OpenMP thread pool across processes-in-flight and each encode runs
        # at a fraction of the machine's speed. One batch at a time uses all
        # cores; faiss search stays outside the lock and still overlaps.
        self._lock = threading.Lock()

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        with self._lock:
            return self._encode_locked(query_list, is_query)

    def _encode_locked(self, query_list: List[str], is_query=True) -> np.ndarray:
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [
                    f"Represent this sentence for searching relevant passages: {query}"
                    for query in query_list
                ]

        inputs = self.tokenizer(
            query_list, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.to(_device()) for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long).to(
                inputs["input_ids"].device
            )
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(
                output.pooler_output, output.last_hidden_state, inputs["attention_mask"], self.pooling_method
            )
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")

        del inputs, output
        if _device() == "cuda":
            torch.cuda.empty_cache()

        return query_emb


#####################################
# Retrievers
#####################################


class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk
        self.index_path = config.index_path
        self.corpus_path = config.corpus_path
        # counters surfaced by /metrics
        self.stat_queries = 0
        self.stat_empty = 0
        self.stat_short = 0
        self.stat_errors = 0
        self._stat_lock = threading.Lock()

    def _bump(self, **kwargs):
        with self._stat_lock:
            for key, delta in kwargs.items():
                setattr(self, key, getattr(self, key) + delta)

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: List[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)

    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        return self._batch_search(query_list, num, return_score)

    def info(self) -> Dict[str, Any]:
        return {
            "method": self.retrieval_method,
            "index_path": self.index_path,
            "corpus_path": self.corpus_path,
            "corpus_docs": len(self.corpus) if getattr(self, "corpus", None) is not None else None,
            "corpus_lazy": isinstance(getattr(self, "corpus", None), OffsetCorpus),
            "topk": self.topk,
        }


class BM25Retriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher

        t0 = time.time()
        self.searcher = LuceneSearcher(self.index_path)
        logger.info("[load] bm25 index done (%.2fs)", time.time() - t0)

        self.contain_doc = self._check_contain_doc()
        logger.info("[load] bm25 contain_doc=%s", self.contain_doc)
        self.corpus = None if self.contain_doc else build_corpus(config)

        self.max_process_num = max(1, int(config.max_process_num))
        self._pool = ThreadPoolExecutor(max_workers=self.max_process_num, thread_name_prefix="bm25")
        self.docid_is_line = None
        if not self.contain_doc and not config.skip_docid_check:
            self.docid_is_line = self._validate_docid_mapping()
        elif not self.contain_doc:
            logger.warning("[load] --skip-docid-check: docids are trusted to be corpus line numbers")

    def _check_contain_doc(self, probes: int = 20):
        """Upstream only probes doc(0).

        A single probe is fragile: one document without a stored ``contents``
        field flips the whole server into the corpus-loading branch (or, worse,
        the other way round). Probe up to ``probes`` documents and require a
        majority to carry raw content.
        """
        hit = 0
        seen = 0
        for i in range(probes):
            try:
                doc = self.searcher.doc(i)
            except Exception:
                break
            if doc is None:
                break
            seen += 1
            if doc.raw() is not None:
                hit += 1
        return seen > 0 and hit * 2 > seen

    def _validate_docid_mapping(self, probes: int = 5) -> bool:
        """Fail fast when the BM25 docid is not the corpus line number.

        ``_search`` resolves BM25 hits with ``corpus[int(hit.docid)]``. If the
        index was built from a different file than ``--corpus_path`` (or the
        docids are opaque ids), every hit silently returns the *wrong*
        document -- and RL happily trains on it. Check a spread of docids
        against the corpus ``id`` field at startup instead.
        """
        n = len(self.corpus)
        candidates = [i for i in (0, 1, 7, n // 3, n // 2, n - 1) if 0 <= i < n]
        checked = 0
        for i in candidates:
            try:
                doc = self.searcher.doc(i)
            except Exception:
                continue
            if doc is None:
                continue
            docid = doc.docid()
            try:
                corpus_id = self.corpus.get_raw(i).get("id")
            except Exception:
                corpus_id = None
            checked += 1
            if corpus_id is None or str(docid) != str(corpus_id):
                raise RuntimeError(
                    "BM25 docid does not match the corpus line index "
                    f"(docid={docid!r} at position {i}, corpus id={corpus_id!r}). "
                    "The index and --corpus_path are inconsistent, so retrieved "
                    "documents would be wrong. Rebuild the index from this corpus, "
                    "or pass --skip-docid-check if you really know what you are doing."
                )
        if checked == 0:
            raise RuntimeError(
                "could not probe any document to validate BM25 docid mapping; "
                "pass --skip-docid-check to bypass"
            )
        logger.info("[load] bm25 docid -> corpus line validated (%d probes)", checked)
        return True

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        try:
            hits = self.searcher.search(query, num)
        except Exception:
            self._bump(stat_errors=1)
            raise
        if len(hits) < 1:
            self._bump(stat_empty=1)
            if return_score:
                return [], []
            return []
        if len(hits) < num:
            self._bump(stat_short=1)
        hits = hits[:num]
        scores = [hit.score for hit in hits]

        if self.contain_doc:
            results = []
            for hit in hits:
                content = json.loads(self.searcher.doc(hit.docid).raw()).get("contents") or ""
                parts = content.split("\n")
                results.append(
                    {
                        "title": parts[0].strip('"'),
                        "text": "\n".join(parts[1:]),
                        "contents": content,
                    }
                )
        else:
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        if return_score:
            return results, scores
        return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        """Search all queries concurrently.

        Upstream loops one query at a time even though it defines
        ``max_process_num``; rollout sends hundreds of queries per second, so a
        serial loop is the throughput ceiling here.
        """
        if isinstance(query_list, str):
            query_list = [query_list]
        self._bump(stat_queries=len(query_list))
        if len(query_list) == 1:
            single = self._search(query_list[0], num, True)
            results, scores = [single[0]], [single[1]]
        else:
            results = [None] * len(query_list)
            scores = [None] * len(query_list)

            def run(i):
                results[i], scores[i] = self._search(query_list[i], num, True)

            list(self._pool.map(run, range(len(query_list))))

        if return_score:
            return results, scores
        return results

    def info(self) -> Dict[str, Any]:
        info = super().info()
        info.update({"contain_doc": self.contain_doc, "docid_is_line": self.docid_is_line})
        return info

    def close(self):
        self._pool.shutdown(wait=False)


class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        import faiss  # local import: BM25-only deployments do not need faiss installed

        t0 = time.time()
        self.corpus = build_corpus(config)
        logger.info("[load] corpus ready (%d docs, %.1fs)", len(self.corpus), time.time() - t0)

        t0 = time.time()
        if config.faiss_device == "cpu":
            self.index = _read_index_mmap(self.index_path) if config.index_mmap else faiss.read_index(self.index_path)
            logger.info("[load] index done (%d vectors, %.1fs)", self.index.ntotal, time.time() - t0)
        elif config.faiss_device == "gpu":
            self.index = faiss.read_index(self.index_path)
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
            logger.info("[load] index moved to GPU (%.1fs)", time.time() - t0)
        elif config.faiss_device == "npu":
            # Ascend faiss (https://gitcode.com/Ascend/faiss): copy the CPU
            # Flat index onto one NPU. Logical device id follows
            # ASCEND_RT_VISIBLE_DEVICES (pick an idle card in the launcher);
            # the encoder's _device() uses the same env var.
            if not hasattr(faiss, "NpuIndexFlat"):
                raise RuntimeError(
                    "--faiss-device npu requires the faiss_ascend wheel "
                    "(pip install faiss_ascend-*.whl, uninstall faiss-cpu first); "
                    "see https://gitcode.com/Ascend/faiss"
                )
            cpu_idx = faiss.read_index(self.index_path)
            devs = [int(d) for d in (config.faiss_devices or "0").split(",") if d.strip()]
            res = faiss.StandardNpuResources()
            if len(devs) == 1:
                npu_config = faiss.NpuIndexFlatConfig()
                npu_config.device = devs[0]  # logical card id (ASCEND_RT_VISIBLE_DEVICES mapped)
                self.index = faiss.NpuIndexFlat(res, cpu_idx, npu_config)  # metric follows the CPU index
            else:
                # replica mode (shard=False): full base on each card -> higher
                # throughput; search fans the batch out and merges.
                vres = faiss.NpuResourcesVector()
                vdev = faiss.Int32Vector()
                for d in devs:
                    vres.push_back(res)
                    vdev.push_back(d)
                co = faiss.NpuMultipleClonerOptions()
                co.shard = False
                co.storageType = 0  # Float16
                self.index = faiss.index_cpu_to_npu_multiple(vres, vdev, cpu_idx, co)
                logger.info("[load] index replicated across NPUs %s", devs)
            logger.info("[load] index moved to NPU (%.1fs)", time.time() - t0)
        else:
            raise ValueError(f"unknown --faiss-device: {config.faiss_device!r}")

        t0 = time.time()
        self.encoder = Encoder(
            model_name=self.retrieval_method,
            model_path=config.retrieval_model_path,
            pooling_method=config.retrieval_pooling_method,
            max_length=config.retrieval_query_max_length,
            use_fp16=config.retrieval_use_fp16,
        )
        logger.info("[load] encoder done (%s, %.1fs)", config.retrieval_model_path, time.time() - t0)
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size
        self.ntotal = self.index.ntotal if hasattr(self.index, "ntotal") else None

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        idxs, scores = self._drop_padding(idxs[0], scores[0])
        if len(idxs) == 0:
            self._bump(stat_empty=1)
            return ([], []) if return_score else []
        results = load_docs(self.corpus, idxs)
        return (results, scores.tolist()) if return_score else results

    @staticmethod
    def _drop_padding(idxs, scores):
        """faiss pads with -1 when the index holds fewer than k vectors."""
        keep = [(int(i), float(s)) for i, s in zip(idxs, scores) if int(i) >= 0]
        if not keep:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        return np.array([k[0] for k in keep]), np.array([k[1] for k in keep])

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk
        self._bump(stat_queries=len(query_list))

        results: List[List[dict]] = []
        scores: List[List[float]] = []
        for start_idx in range(0, len(query_list), self.batch_size):
            query_batch = query_list[start_idx : start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            for i in range(len(batch_idxs)):
                idxs, scs = self._drop_padding(np.array(batch_idxs[i]), np.array(batch_scores[i]))
                if len(idxs) == 0:
                    self._bump(stat_empty=1)
                    results.append([])
                    scores.append([])
                    continue
                results.append(load_docs(self.corpus, idxs))
                scores.append(scs.tolist())

            del batch_emb, batch_scores, batch_idxs, query_batch

        if return_score:
            return results, scores
        return results

    def info(self) -> Dict[str, Any]:
        info = super().info()
        info.update({"ntotal": self.ntotal, "index_mmap": self.config.index_mmap, "faiss_device": self.config.faiss_device})
        return info


def get_retriever(config):
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    return DenseRetriever(config)


#####################################
# Cache
#####################################


class QueryCache:
    """Tiny thread-safe LRU for (query, topk) -> (docs, scores)."""

    def __init__(self, max_entries: int = 0):
        self.max_entries = max(0, int(max_entries))
        self._data: "collections.OrderedDict[Tuple[str, int], Tuple[List[dict], List[float]]]" = (
            collections.OrderedDict()
        )
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0

    def get(self, key):
        if not self.enabled:
            return None
        with self._lock:
            try:
                value = self._data.pop(key)
            except KeyError:
                self.misses += 1
                return None
            self._data[key] = value  # refresh recency
            self.hits += 1
            return value

    def put(self, key, value):
        if not self.enabled:
            return
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)


#####################################
# FastAPI server
#####################################


class Config:
    def __init__(
        self,
        retrieval_method: str = "bm25",
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        faiss_device: str = "cpu",
        faiss_devices: str = "",  # npu only: comma-separated LOGICAL card ids (e.g. "0,1"); empty = device 0
        index_mmap: bool = True,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128,
        max_topk: int = 100,
        max_queries: int = 256,
        cache_entries: int = 0,
        max_chars: int = 0,
        max_process_num: int = 8,
        eager_corpus: bool = False,
        skip_docid_check: bool = False,
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.faiss_device = faiss_device
        self.faiss_devices = faiss_devices
        self.index_mmap = index_mmap
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size
        self.max_topk = max_topk
        self.max_queries = max_queries
        self.cache_entries = cache_entries
        self.max_chars = max_chars
        self.max_process_num = max_process_num
        self.eager_corpus = eager_corpus
        self.skip_docid_check = skip_docid_check


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()
retriever = None
config = None
cache = QueryCache(0)
server_stats = {
    "requests": 0,
    "queries": 0,
    "errors": 0,
    "empty_results": 0,
    "latency_ms_sum": 0.0,
    "latency_ms_max": 0.0,
    "started_at": 0.0,
}
_stats_lock = threading.Lock()


def _record_request(n_queries: int, latency_ms: float, errors: int = 0, empty: int = 0):
    with _stats_lock:
        server_stats["requests"] += 1
        server_stats["queries"] += n_queries
        server_stats["errors"] += errors
        server_stats["empty_results"] += empty
        server_stats["latency_ms_sum"] += latency_ms
        server_stats["latency_ms_max"] = max(server_stats["latency_ms_max"], latency_ms)


def _trim_docs(docs: List[dict]) -> List[dict]:
    if not config.max_chars:
        return docs
    out = []
    for doc in docs:
        contents = doc.get("contents") or ""
        if len(contents) > config.max_chars:
            trimmed = dict(doc)
            trimmed["contents"] = contents[: config.max_chars]
            trimmed["text"] = (doc.get("text") or "")[: config.max_chars]
            out.append(trimmed)
        else:
            out.append(doc)
    return out


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """Batch retrieval.

    Request:  {"queries": [...], "topk": 3, "return_scores": true}
    Response: {"result": [[{"document": {...}, "score": 0.71}, ...], ...],
               "cached": [false, ...]}   # `cached` only when caching is on
    """
    n = len(request.queries)
    if n == 0:
        return {"result": []}
    if config.max_queries and n > config.max_queries:
        raise HTTPException(
            status_code=413, detail=f"too many queries: {n} > max_queries={config.max_queries}"
        )
    topk = request.topk or config.retrieval_topk
    if topk <= 0:
        raise HTTPException(status_code=400, detail="topk must be positive")
    if config.max_topk and topk > config.max_topk:
        raise HTTPException(status_code=400, detail=f"topk={topk} exceeds max_topk={config.max_topk}")

    t0 = time.time()
    results: List[List[dict]] = [None] * n
    scores: List[List[float]] = [None] * n
    cached_flags = [False] * n
    pending = []
    for i, query in enumerate(request.queries):
        hit = cache.get((query, topk))
        if hit is not None:
            results[i], scores[i] = hit
            cached_flags[i] = True
        else:
            pending.append(i)

    errors = 0
    if pending:
        try:
            ret = retriever.batch_search(
                query_list=[request.queries[i] for i in pending],
                num=topk,
                return_score=True,
            )
            pending_results, pending_scores = ret
        except Exception as exc:  # noqa: BLE001
            _record_request(n, (time.time() - t0) * 1000, errors=1)
            logger.exception("retrieval failed")
            raise HTTPException(status_code=500, detail=f"retrieval failed: {exc}") from exc
        for slot, i in enumerate(pending):
            results[i] = _trim_docs(pending_results[slot])
            scores[i] = pending_scores[slot]
            cache.put((request.queries[i], topk), (results[i], scores[i]))

    latency_ms = (time.time() - t0) * 1000
    empty = sum(1 for r in results if not r)
    _record_request(n, latency_ms, errors=errors, empty=empty)

    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            resp.append([{"document": doc, "score": score} for doc, score in zip(single_result, scores[i] or [])])
        else:
            resp.append(single_result)

    body = {"result": resp}
    if cache.enabled:
        body["cached"] = cached_flags
    return body


@app.get("/status")
def get_status():
    """Readiness + index metadata.

    Upstream returns a bare "ok" even while the index is still loading, so a
    launcher polling /status happily starts rollout against a dead server.
    """
    info = {
        "status": "ok" if retriever is not None else "loading",
        "ready": retriever is not None,
        "uptime_s": round(time.time() - server_stats["started_at"], 1) if server_stats["started_at"] else 0,
    }
    if retriever is not None:
        info.update(retriever.info())
    return info


@app.get("/metrics")
def get_metrics():
    with _stats_lock:
        stats = dict(server_stats)
    reqs = stats["requests"] or 1
    qs = stats["queries"] or 1
    return {
        "server": {
            **stats,
            "latency_ms_avg": round(stats["latency_ms_sum"] / reqs, 2),
        },
        "retriever": {
            "queries": retriever.stat_queries if retriever else 0,
            "empty_results": retriever.stat_empty if retriever else 0,
            "short_results": retriever.stat_short if retriever else 0,
            "errors": retriever.stat_errors if retriever else 0,
            "empty_rate": round((retriever.stat_empty / qs) if retriever else 0.0, 4),
        },
        "cache": {
            "enabled": cache.enabled,
            "entries": len(cache._data),
            "max_entries": cache.max_entries,
            "hits": cache.hits,
            "misses": cache.misses,
        },
    }


def _build_parser():
    parser = argparse.ArgumentParser(description="Launch the local retriever (bm25 | dense).")
    parser.add_argument("--index_path", type=str, default="./e5_Flat.index", help="Corpus indexing file.")
    parser.add_argument("--corpus_path", type=str, default="./wiki-18.jsonl", help="Local corpus file (jsonl).")
    parser.add_argument("--topk", type=int, default=3, help="Default number of retrieved passages per query.")
    parser.add_argument("--retriever_name", type=str, default="e5", help="e5 / bge / bm25 ...")
    parser.add_argument(
        "--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the dense retriever model."
    )
    parser.add_argument(
        "--faiss-device",
        type=str,
        default="cpu",
        choices=["cpu", "npu", "gpu"],
        help=(
            "Where to run faiss search: cpu (default, mmap), npu (Ascend "
            "faiss; needs the faiss_ascend wheel + ASCEND_RT_VISIBLE_DEVICES "
            "pointing at an idle card), gpu (CUDA)."
        ),
    )
    parser.add_argument(
        "--faiss-devices",
        type=str,
        default="",
        help=(
            "npu only: comma-separated LOGICAL card ids, e.g. '0,1' (replica "
            "mode: full base on each card). Physical cards are chosen via "
            "ASCEND_RT_VISIBLE_DEVICES. Empty = device 0."
        ),
    )
    parser.add_argument(
        "--faiss_gpu",
        action="store_true",
        help=argparse.SUPPRESS,  # deprecated alias of --faiss-device gpu (run_common.sh)
    )
    parser.add_argument(
        "--no-index-mmap",
        action="store_true",
        help=(
            "Read the faiss index eagerly into anonymous RAM instead of mmap'ing it. "
            "Note: mmap does not make an HNSW index cheap -- random graph traversal makes "
            "the whole file resident anyway; mmap mainly helps Flat/IVF indexes."
        ),
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-topk", type=int, default=100, help="Reject requests asking for more than this")
    parser.add_argument("--max-queries", type=int, default=256, help="Reject batches with more queries than this")
    parser.add_argument("--cache-entries", type=int, default=0, help="LRU query cache size (0 disables)")
    parser.add_argument(
        "--max-chars", type=int, default=0, help="Truncate document contents to N chars in the response (0 = full)"
    )
    parser.add_argument("--max-process-num", type=int, default=8, help="BM25 search worker threads")
    parser.add_argument("--pooling", type=str, default="mean", help="mean | cls | pooler")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128, help="Dense encoder batch size")
    parser.add_argument("--fp32", action="store_true", help="Disable fp16 for the dense encoder")
    parser.add_argument(
        "--eager-corpus", action="store_true", help="Load the whole corpus into RAM (upstream behaviour)"
    )
    parser.add_argument(
        "--skip-docid-check", action="store_true", help="Trust BM25 docids to be corpus line numbers"
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def main(argv=None):
    global retriever, config, cache

    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), _DEFAULT_TOP_LEVEL),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = Config(
        retrieval_method=args.retriever_name,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_device=args.faiss_device if not args.faiss_gpu else "gpu",
        faiss_devices=args.faiss_devices,
        index_mmap=not args.no_index_mmap,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method=args.pooling,
        retrieval_query_max_length=args.max_length,
        retrieval_use_fp16=not args.fp32,
        retrieval_batch_size=args.batch_size,
        max_topk=args.max_topk,
        max_queries=args.max_queries,
        cache_entries=args.cache_entries,
        max_chars=args.max_chars,
        max_process_num=args.max_process_num,
        eager_corpus=args.eager_corpus,
        skip_docid_check=args.skip_docid_check,
    )
    cache = QueryCache(config.cache_entries)
    server_stats["started_at"] = time.time()

    retriever = get_retriever(config)
    logger.info("[ready] %s", retriever.info())

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
