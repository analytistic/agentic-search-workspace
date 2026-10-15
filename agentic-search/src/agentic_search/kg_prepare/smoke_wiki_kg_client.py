"""Debug smoke test for :mod:`wiki_kg_client`.

Run from the repository root with::

    PYTHONPATH=agentic-search/src python \
      agentic-search/src/agentic_search/kg_prepare/smoke_wiki_kg_client.py

The script uses the live Wikidata API, so it needs network access.  It is
intentionally small and suitable for launching from an IDE debugger.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow launching this file directly from an IDE without configuring
# PYTHONPATH manually.
if __package__ in (None, ""):
    src_dir = Path(__file__).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from agentic_search.kg_prepare.wiki_kg_client import WikiKGClient


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test the Wikidata KG client")
    parser.add_argument("ids", nargs="*", default=["Q42", "Q84", "Q90"])
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=3)
    args = parser.parse_args()

    client = WikiKGClient(batch_size=args.batch_size)
    try:
        client.start()

        print("[1/3] synchronous request")
        entity = client.get_entity(args.ids[0], timeout=args.timeout)
        if entity is None:
            print(
                f"  {args.ids[0]}: no response within {args.timeout:.1f}s "
                "(check Wikidata access or local proxy)"
            )
            print("SMOKE INCONCLUSIVE")
            return 2
        print(json.dumps({
            "id": entity.get("id"),
            "type": entity.get("type"),
            "labels": entity.get("labels", {}),
            "claim_count": sum(len(v) for v in entity.get("claims", {}).values()),
        }, ensure_ascii=False, indent=2))

        print("[2/3] batched asynchronous requests")
        futures = [client.submit(entity_id) for entity_id in args.ids]
        results = []
        for future in futures:
            try:
                results.append(future.result(timeout=args.timeout))
            except TimeoutError:
                results.append(None)
        for entity_id, result in zip(args.ids, results):
            print(f"  {entity_id}: {'ok' if result else 'missing'}")

        print("[3/3] cache request")
        cached = client.get_entity(args.ids[0], timeout=args.timeout)
        if cached is not entity:
            raise RuntimeError("Expected the second request to use the in-memory cache")
        print(f"  {args.ids[0]}: cache hit")
        print("SMOKE PASS")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
