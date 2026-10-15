"""A small, long-running Wikidata entity fetch client.

The client is deliberately dependency-free.  Callers submit entity ids (for
example ``Q42``) to a request queue; a background worker batches pending ids,
fetches them from the Wikidata API, and resolves one future per request.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Iterable, Mapping
from urllib.parse import urlencode
from urllib.request import ProxyHandler, Request, build_opener
import os
import argparse
from pathlib import Path


API_URL = "https://www.wikidata.org/w/api.php"
DEFAULT_PROXY = "http://127.0.0.1:7890"


@dataclass(frozen=True)
class _Request:
    entity_id: str
    future: Future[dict[str, Any] | None]


class WikiKGClient:
    """Continuously consume entity requests and fetch them in API batches.

    ``start()`` launches the worker.  ``get_entity()`` is the convenient
    request/response API; ``run_forever()`` is provided for deployment
    processes that should keep the worker alive until interrupted.
    """

    def __init__(
        self,
        *,
        api_url: str = API_URL,
        language: str = "en",
        batch_size: int = 50,
        poll_interval: float = 0.05,
        timeout: float = 2.0,
        retries: int = 1,
        user_agent: str = "agentic-search/0.1 (Wikidata client)",
        proxy_url: str | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.api_url = api_url
        self.language = language
        self.batch_size = batch_size
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.retries = max(1, retries)
        self.user_agent = user_agent
        # The local Clash/V2Ray-style proxy is the default for development.
        # Set WIKI_KG_PROXY= to disable it, or pass proxy_url explicitly.
        if proxy_url is None:
            proxy_url = os.environ.get("WIKI_KG_PROXY", DEFAULT_PROXY)
        self.proxy_url = proxy_url
        self._opener = build_opener(
            ProxyHandler({"http": proxy_url, "https": proxy_url})
            if proxy_url
            else ProxyHandler({})
        )
        self._queue: queue.Queue[_Request | None] = queue.Queue()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._cache: dict[str, dict[str, Any] | None] = {}

    def start(self) -> "WikiKGClient":
        if self._worker and self._worker.is_alive():
            return self
        self._stop.clear()
        self._worker = threading.Thread(target=self._worker_loop, name="wikidata-client", daemon=True)
        self._worker.start()
        return self

    def submit(self, entity_id: str) -> Future[dict[str, Any] | None]:
        """Queue one Q-id and return a future for its entity payload."""
        entity_id = entity_id.strip()
        if not entity_id or not entity_id.startswith("Q"):
            raise ValueError("entity_id must be a Wikidata Q-id, e.g. Q42")
        if not self._worker or not self._worker.is_alive():
            self.start()
        if entity_id in self._cache:
            future: Future[dict[str, Any] | None] = Future()
            future.set_result(self._cache[entity_id])
            return future
        future = Future()
        self._queue.put(_Request(entity_id, future))
        return future

    def get_entity(self, entity_id: str, timeout: float | None = None) -> dict[str, Any] | None:
        """Queue an entity request and wait for the API response."""
        try:
            return self.submit(entity_id).result(timeout=timeout)
        except (TimeoutError, OSError):
            # Deployment callers should be able to keep consuming their own
            # queue when the upstream API/proxy is slow.  The worker remains
            # daemonized and the request future is discarded by this caller.
            return None

    def close(self) -> None:
        self._stop.set()
        self._queue.put(None)
        if self._worker:
            self._worker.join(timeout=max(self.timeout, 1.0) + 1)

    def run_forever(self) -> None:
        """Keep the process alive; requests can be submitted by other threads."""
        self.start()
        try:
            while not self._stop.wait(1.0):
                pass
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                first = self._queue.get(timeout=self.poll_interval)
            except queue.Empty:
                continue
            if first is None:
                break
            requests = [first]
            deadline = time.monotonic() + self.poll_interval
            while len(requests) < self.batch_size and time.monotonic() < deadline:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    self._stop.set()
                    break
                requests.append(item)
            ids = list(dict.fromkeys(item.entity_id for item in requests))
            try:
                entities = self._fetch(ids)
                for entity_id in ids:
                    self._cache[entity_id] = entities.get(entity_id)
                for item in requests:
                    if not item.future.done():
                        item.future.set_result(entities.get(item.entity_id))
            except Exception as exc:
                for item in requests:
                    if not item.future.done():
                        item.future.set_exception(exc)

    def _fetch(self, entity_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        params = urlencode({
            "action": "wbgetentities",
            "ids": "|".join(entity_ids),
            # Keep the response small enough for short-lived requests.  KG
            # path construction only needs labels and entity-valued claims.
            "props": "labels|claims",
            "languages": self.language,
            "languagefallback": "1",
            "format": "json",
        })
        request = Request(f"{self.api_url}?{params}", headers={"User-Agent": self.user_agent})
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    payload = json.load(response)
                break
            except Exception as exc:
                last_error = exc
                if attempt + 1 == self.retries:
                    raise
                time.sleep(0.5 * (attempt + 1))
        else:
            raise RuntimeError("Wikidata request failed") from last_error
        return {
            entity_id: entity
            for entity_id, entity in payload.get("entities", {}).items()
            if isinstance(entity, Mapping) and "missing" not in entity
        }

    def download_entities(
        self,
        count: int,
        output_dir: str | os.PathLike[str],
        seed_ids: Iterable[str] = ("Q42", "Q84", "Q90"),
    ) -> Path:
        """Download a small connected entity set and write ``items_part_0.json``."""
        if count < 1:
            raise ValueError("count must be positive")
        pending = list(dict.fromkeys(seed_ids))
        entities: dict[str, dict[str, Any]] = {}
        while pending and len(entities) < count:
            batch = pending[: self.batch_size]
            del pending[: self.batch_size]
            try:
                fetched = self._fetch(batch)
            except (OSError, TimeoutError):
                # A large API response can exceed the short request timeout;
                # retry the same ids as smaller requests.
                if len(batch) > 1:
                    pending[0:0] = batch[: max(1, len(batch) // 2)]
                continue
            for entity_id, entity in fetched.items():
                if entity_id in entities:
                    continue
                entities[entity_id] = entity
                if len(entities) >= count:
                    break
                for claims in entity.get("claims", {}).values():
                    for claim in claims:
                        value = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
                        target = value.get("id") if isinstance(value, dict) else None
                        if target and target.startswith("Q") and target not in entities and target not in pending:
                            pending.append(target)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / "items_part_0.json"
        # Match the official Wikidata JSON dump shape: an array wrapper with
        # one entity object per line (and a trailing comma on all but the last).
        rows = [json.dumps(entity, ensure_ascii=False) for entity in entities.values()]
        file_path.write_text(
            "[\n" + ",\n".join(rows) + "\n]\n",
            encoding="utf-8",
        )
        return file_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a small connected Wikidata entity set")
    parser.add_argument("--count", type=int, required=True, help="Number of entities to download")
    parser.add_argument("--output-dir", required=True, help="Directory for items_part_0.json")
    parser.add_argument("--seed", nargs="*", default=["Q42", "Q84", "Q90"])
    parser.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args()
    client = WikiKGClient(timeout=args.timeout, retries=1, batch_size=10)
    try:
        path = client.download_entities(args.count, args.output_dir, args.seed)
        print(f"Downloaded entities to {path}")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["WikiKGClient"]
