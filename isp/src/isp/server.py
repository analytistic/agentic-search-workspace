"""retriever 的 HTTP 服务。

架构：索引和编码器只在这个进程里加载一份，agent worker（可能是多个 Ray actor）
通过 HTTP 检索。这样避免每个 actor 各背 410 MB 索引和一份 MPS 编码器。

端点：
    POST /retrieve      {query | vector, topk, with_text, preview_tokens}   返回预览，默认 512 token
    POST /document      {docid, max_chars}                                 返回全文（浏览工具用）
    POST /embed         {texts, mode, apply_adapter}                       原始表征（训练侧用）
    POST /doc_vectors   {docids}                                          按 docid 取索引向量
    POST /adapter       {path, global_steps?, version?, apply?}            权重按路径推进来并生效
    GET  /version       当前生效 / 待生效版本
    GET  /health        存活检查

启动：
    python -m isp.server --index-glob "data/isp/indexes/.../corpus.*.pkl" \
        --corpus-path data/browsecomp_plus/raw/corpus \
        --model-path data/isp/models/Qwen3-Embedding-0.6B --port 8100
"""

from __future__ import annotations

import argparse
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .retriever import FlatRetriever

logger = logging.getLogger(__name__)


class RetrieveRequest(BaseModel):
    query: Optional[str] = None
    vector: Optional[List[float]] = None
    topk: int = Field(default=10, ge=1, le=200)
    with_text: bool = True
    preview_tokens: int = Field(default=64, ge=0, le=4096)


class DocumentRequest(BaseModel):
    docid: str
    max_chars: int = Field(default=0, ge=0)


class AdapterRequest(BaseModel):
    path: str
    global_steps: Optional[int] = None
    version: Optional[str] = None
    apply: bool = True  # False 时只登记不生效


class EmbedRequest(BaseModel):
    texts: List[str] = Field(min_length=1, max_length=256)
    mode: str = "query"  # query 加指令前缀，doc 不加
    apply_adapter: bool = False  # 训练侧要原始表征，默认不套 adapter


class DocVectorRequest(BaseModel):
    docids: List[str] = Field(min_length=1, max_length=4096)


def create_app(retriever: FlatRetriever) -> FastAPI:
    app = FastAPI(title="isp retriever")

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "docs": len(retriever)}

    @app.get("/version")
    def version() -> dict:
        return {
            "current": retriever._adapter_version,
            "pending": retriever._pending_version,
            "steps": retriever._adapter_steps,
        }

    @app.post("/retrieve")
    def retrieve(req: RetrieveRequest) -> dict:
        if req.vector is not None:
            hits = retriever.search_vector(
                req.vector, topk=req.topk, with_text=req.with_text, preview_tokens=req.preview_tokens
            )
        elif req.query:
            if retriever._model is None:
                raise HTTPException(status_code=400, detail="服务未加载编码器，只能用 vector 检索")
            hits = retriever.search_text(
                req.query, topk=req.topk, with_text=req.with_text, preview_tokens=req.preview_tokens
            )
        else:
            raise HTTPException(status_code=400, detail="query 和 vector 至少给一个")
        return {
            "version": retriever._adapter_version,
            "results": [{"docid": h.docid, "score": h.score, "text": h.text} for h in hits],
        }

    @app.post("/document")
    def document(req: DocumentRequest) -> dict:
        try:
            text = retriever.get_document(req.docid, max_chars=req.max_chars)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"没有这篇文档: {req.docid}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"docid": req.docid, "chars": len(text), "text": text, "version": retriever._adapter_version}

    @app.post("/adapter")
    def update_adapter(req: AdapterRequest) -> dict:
        try:
            out = retriever.update_adapter(
                path=req.path, global_steps=req.global_steps, version=req.version
            )
            if req.apply:
                out = retriever.commit_adapter(path=req.path, version=req.version, global_steps=req.global_steps)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return out

    @app.post("/embed")
    def embed(req: EmbedRequest) -> dict:
        try:
            vecs = retriever.encode_raw(req.texts, mode=req.mode)
            if req.apply_adapter:
                vecs = retriever._apply_adapter(vecs)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"dim": int(vecs.shape[1]), "vectors": vecs.tolist()}

    @app.post("/doc_vectors")
    def doc_vectors(req: DocVectorRequest) -> dict:
        try:
            vecs = retriever.doc_vector(req.docids)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"没有这篇文档: {exc}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"dim": int(vecs.shape[1]), "vectors": vecs.tolist()}

    return app


def http_search_tool(
    base_url: str = "http://127.0.0.1:8100",
    default_topk: int = 10,
    preview_tokens: int = 64,
    timeout: float = 10.0,
    client=None,
):
    """搜索工具：返回 top-k 文档的 token 预览，不是全文。

    工具在 agent 的 to_thread 里跑，所以用同步客户端最省事。返回值遵循
    ToolAgentLoop 的约定：text 给模型看，documents 给后续挖样本用。
    """
    import httpx

    client = client or httpx.Client(base_url=base_url, timeout=timeout)

    def search(query: str, top_k: int = default_topk) -> dict:
        resp = client.post(
            "/retrieve",
            json={"query": query, "topk": top_k, "with_text": True, "preview_tokens": preview_tokens},
        )
        resp.raise_for_status()
        results = resp.json()["results"]
        return {
            "text": "\n\n".join(f"[{r['docid']}] {r['text']}" for r in results),
            "documents": [r["docid"] for r in results],
        }

    return search


def http_browse_tool(
    base_url: str = "http://127.0.0.1:8100",
    max_chars: int = 0,
    timeout: float = 10.0,
    client=None,
):
    """浏览工具：按 docid 取全文。max_chars=0 为整篇返回。

    和搜索的区别是信号强度：搜索只是召回了，浏览才是真打开看了，
    所以轨迹里这两类工具调用的正负样本权重不该一样。
    """
    import httpx

    client = client or httpx.Client(base_url=base_url, timeout=timeout)

    def browse(docid: str) -> dict:
        resp = client.post("/document", json={"docid": docid, "max_chars": max_chars})
        resp.raise_for_status()
        data = resp.json()
        return {"text": data["text"], "documents": [docid]}

    return browse


def main() -> None:
    parser = argparse.ArgumentParser(description="起 isp 的检索服务")
    parser.add_argument("--index-glob", required=True)
    parser.add_argument("--corpus-path", default=None)
    parser.add_argument("--model-path", default=None, help="不给就只能用 vector 检索")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-chars", type=int, default=0, help="预览的字符硬上限，0 不限")
    parser.add_argument("--preview-tokens", type=int, default=64, help="检索结果每篇的 token 预览长度")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    retriever = FlatRetriever(
        index_glob=args.index_glob,
        corpus_path=args.corpus_path,
        model_path=args.model_path,
        device=args.device,
        max_chars=args.max_chars,
        preview_tokens=args.preview_tokens,
    )
    app = create_app(retriever)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
