#!/usr/bin/env python3
"""Prepare document- or passage-level BrowseComp-Plus retrieval data.

This file deliberately stops at corpus preparation and selected retriever
index construction. It does not implement an agent, a runtime search command,
or evaluation. The passage path follows the segmentation convention used by
the text-ranking-in-deep-research project: spaCy sentence boundaries, a
250-token default budget, and no overlap.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


DEFAULT_CORPUS = Path("data/browsecomp_plus/raw/corpus/data")
DEFAULT_OUTPUT = Path("agentic-search/benchmark/browsecomp-plus/artifacts")
PASSAGE_SIZE = 250
SUPPORTED_RETRIEVERS = ("bm25",)


@dataclass(frozen=True)
class Document:
    docid: str
    text: str
    url: str
    title: str = ""


def iter_documents(source: Path) -> Iterator[Document]:
    """Read corpus rows from JSONL or Parquet shards."""

    if source.is_file() and source.suffix.lower() in {".jsonl", ".json"}:
        yield from _iter_jsonl(source)
        return

    jsonl_files = sorted(source.glob("*.jsonl")) if source.is_dir() else []
    if jsonl_files:
        for path in jsonl_files:
            yield from _iter_jsonl(path)
        return

    try:
        import pyarrow.dataset as ds
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "Reading BrowseComp-Plus Parquet requires pyarrow. "
            "Install it with: uv add --project agentic-search pyarrow"
        ) from exc

    dataset = ds.dataset(str(source), format="parquet")
    columns = [name for name in ("docid", "text", "url", "title") if name in dataset.schema.names]
    required = {"docid", "text"}
    missing = required - set(columns)
    if missing:
        raise ValueError(f"Corpus is missing required columns: {sorted(missing)}")

    for batch in dataset.scanner(columns=columns).to_batches():
        data = batch.to_pydict()
        for row_idx in range(batch.num_rows):
            yield Document(
                docid=str(data["docid"][row_idx]),
                text=str(data["text"][row_idx] or ""),
                url=str(data.get("url", [""] * batch.num_rows)[row_idx] or ""),
                title=str(data.get("title", [""] * batch.num_rows)[row_idx] or ""),
            )


def _iter_jsonl(path: Path) -> Iterator[Document]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                docid = row.get("docid", row.get("id"))
                text = row.get("text", row.get("contents"))
                if isinstance(text, list):
                    text = "\n".join(str(item.get("body", item)) for item in text)
                if docid is None or text is None:
                    raise KeyError("docid/id or text/contents")
            except (AttributeError, KeyError) as exc:
                raise ValueError(f"Invalid corpus row {path}:{line_number}") from exc
            yield Document(
                docid=str(docid),
                text=str(text),
                url=str(row.get("url", "") or ""),
                title=str(row.get("title", "") or ""),
            )


def load_sentence_splitter():
    """Load the upstream sentence-only pipeline, installing its model if needed."""

    try:
        import spacy
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "Passages require spaCy and en_core_web_sm. "
            f"Install the model with: {sys.executable} -m spacy download en_core_web_sm"
        ) from exc

    if not spacy.util.is_package("en_core_web_sm"):
        print(
            "en_core_web_sm is not installed; downloading the compatible spaCy model...",
            file=sys.stderr,
        )
        try:
            from spacy.cli import download

            if importlib.util.find_spec("pip") is None:
                # spaCy falls back to uv; explicitly target this interpreter.
                download("en_core_web_sm", False, False, None, "--python", sys.executable)
            else:
                # spaCy invokes sys.executable -m pip in this branch.
                download("en_core_web_sm")
            importlib.invalidate_caches()
        except (Exception, SystemExit) as exc:  # pragma: no cover - network dependent
            raise RuntimeError(
                "Automatic installation of en_core_web_sm failed. "
                f"Run manually: {sys.executable} -m spacy download en_core_web_sm"
            ) from exc

    if not spacy.util.is_package("en_core_web_sm"):
        raise RuntimeError(
            "en_core_web_sm is still unavailable after the automatic download. "
            f"Run manually: {sys.executable} -m spacy download en_core_web_sm"
        )

    try:
        nlp = spacy.load(
            "en_core_web_sm",
            exclude=[
                "parser",
                "tagger",
                "ner",
                "attribute_ruler",
                "lemmatizer",
                "tok2vec",
            ],
        )
    except OSError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("The installed en_core_web_sm model could not be loaded") from exc

    if "senter" not in nlp.pipe_names:
        nlp.enable_pipe("senter")
    return nlp


def split_passages(
    nlp: Any, document: Document, passage_size: int
) -> Iterator[dict[str, Any]]:
    """Split one document at sentence boundaries with a token budget."""

    parsed = nlp(document.text)
    current: list[str] = []
    current_tokens = 0
    passage_index = 1

    def emit(text: str, index: int) -> dict[str, Any]:
        passage_id = f"{document.docid}_{index}"
        return {
            "passage_id": passage_id,
            "docid": document.docid,
            "passage_index": index,
            "title": document.title,
            "text": text,
            "url": document.url,
        }

    for sentence in parsed.sents:
        sentence_text = sentence.text.strip()
        if not sentence_text:
            continue
        sentence_tokens = len(sentence)

        if sentence_tokens >= passage_size:
            if current:
                yield emit(" ".join(current), passage_index)
                passage_index += 1
                current = []
                current_tokens = 0
            yield emit(sentence_text, passage_index)
            passage_index += 1
            continue

        if current and current_tokens + sentence_tokens > passage_size:
            yield emit(" ".join(current), passage_index)
            passage_index += 1
            current = []
            current_tokens = 0

        current.append(sentence_text)
        current_tokens += sentence_tokens

    if current:
        yield emit(" ".join(current), passage_index)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def prepare_document_track(documents: Iterable[Document], output: Path, limit: int | None) -> int:
    pyserini_rows = []
    metadata_rows = []
    for document in documents:
        pyserini_rows.append(
            {
                "id": document.docid,
                "contents": document.text,
            }
        )
        metadata_rows.append(
            {
                "docid": document.docid,
                "url": document.url,
                "title": document.title,
            }
        )
        if limit is not None and len(pyserini_rows) >= limit:
            break
    write_jsonl(output / "pyserini" / "corpus.jsonl", pyserini_rows)
    write_jsonl(output / "metadata.jsonl", metadata_rows)
    return len(pyserini_rows)


def prepare_passage_track(
    documents: Iterable[Document],
    output: Path,
    limit: int | None,
    passage_size: int,
) -> tuple[int, int]:
    nlp = load_sentence_splitter()
    pyserini_rows: list[dict[str, Any]] = []
    tevatron_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    document_count = 0

    for document in documents:
        document_count += 1
        for passage in split_passages(nlp, document, passage_size):
            passage_id = passage["passage_id"]
            title = passage["title"]
            body = passage["text"]
            pyserini_rows.append(
                {
                    "id": passage_id,
                    "contents": f"{title} {body}".strip() if title else body,
                }
            )
            tevatron_rows.append(
                {
                    "docid": passage_id,
                    "title": title,
                    "text": body,
                }
            )
            mapping_rows.append(
                {
                    "passage_id": passage_id,
                    "docid": passage["docid"],
                    "passage_index": passage["passage_index"],
                    "url": passage["url"],
                }
            )
        if limit is not None and document_count >= limit:
            break

    passage_count = len(pyserini_rows)
    write_jsonl(output / "pyserini" / "corpus.jsonl", pyserini_rows)
    write_jsonl(output / "tevatron.jsonl", tevatron_rows)
    write_jsonl(output / "passage_to_document.jsonl", mapping_rows)
    return document_count, passage_count


def build_bm25(pyserini_dir: Path, index_dir: Path, threads: int) -> None:
    """Build the Lucene structures used by BM25."""
    command = [
        sys.executable,
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        str(pyserini_dir),
        "--index",
        str(index_dir),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        str(threads),
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]
    subprocess.run(command, check=True)


def validate_retrievers(retrievers: list[str]) -> None:
    """Validate the index types requested by the preprocessing pipeline."""

    unknown = set(retrievers) - set(SUPPORTED_RETRIEVERS)
    if unknown:
        supported = ", ".join(SUPPORTED_RETRIEVERS)
        raise ValueError(
            f"Unsupported retriever(s): {sorted(unknown)}. Supported: {supported}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("document", "passage"), required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None, help="Process at most N documents (smoke runs).")
    parser.add_argument(
        "--passage-size",
        type=int,
        default=PASSAGE_SIZE,
        help=f"Maximum target passage length in spaCy tokens (default: {PASSAGE_SIZE}).",
    )
    parser.add_argument(
        "--retrievers",
        nargs="+",
        default=[],
        metavar="NAME",
        help="Retriever indexes to build (currently: bm25). Example: --retrievers bm25",
    )
    parser.add_argument("--threads", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.passage_size <= 0:
        raise ValueError("--passage-size must be greater than zero")
    retrievers = list(dict.fromkeys(args.retrievers))
    validate_retrievers(retrievers)
    track_output = args.output / args.track
    documents = iter_documents(args.input)

    if args.track == "document":
        document_count = prepare_document_track(documents, track_output, args.limit)
        passage_count = None
    else:
        document_count, passage_count = prepare_passage_track(
            documents, track_output, args.limit, args.passage_size
        )

    pyserini_dir = track_output / "pyserini"
    index_outputs: dict[str, str] = {}
    for retriever in retrievers:
        if retriever == "bm25":
            index_dir = track_output / "indexes" / "bm25"
            build_bm25(pyserini_dir, index_dir, args.threads)
            index_outputs[retriever] = str(index_dir)

    manifest = {
        "track": args.track,
        "source": str(args.input),
        "documents": document_count,
        "passages": passage_count,
        "retrievers": retrievers,
        "passage_size_tokens": args.passage_size if args.track == "passage" else None,
        "passage_overlap_tokens": 0 if args.track == "passage" else None,
        "outputs": {
            "pyserini_corpus": str(pyserini_dir / "corpus.jsonl"),
            "metadata": str(track_output / "metadata.jsonl")
            if args.track == "document"
            else None,
            "indexes": index_outputs,
        },
    }
    track_output.mkdir(parents=True, exist_ok=True)
    (track_output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
