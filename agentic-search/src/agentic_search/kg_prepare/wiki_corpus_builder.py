"""Build a small Wikipedia passage corpus aligned with Wikidata Q-ids."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import ProxyHandler, Request, build_opener


WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"


class WikiCorpusBuilder:
    def __init__(self, proxy_url: str | None = None, timeout: float = 5.0) -> None:
        proxy_url = os.environ.get("WIKI_KG_PROXY", "http://127.0.0.1:7890") if proxy_url is None else proxy_url
        self.timeout = timeout
        self.opener = build_opener(ProxyHandler({"http": proxy_url, "https": proxy_url}) if proxy_url else ProxyHandler({}))
        self.headers = {"User-Agent": "agentic-search-demo/0.1"}

    def _get(self, url: str, params: dict[str, str]) -> dict:
        request = Request(f"{url}?{urlencode(params)}", headers=self.headers)
        with self.opener.open(request, timeout=self.timeout) as response:
            return json.load(response)

    def fetch_titles(self, qids: list[str]) -> dict[str, str]:
        payload = self._get(WIKIDATA_API, {
            "action": "wbgetentities", "ids": "|".join(qids),
            "props": "sitelinks|labels", "sitefilter": "enwiki",
            "languages": "en", "languagefallback": "1", "format": "json",
        })
        result = {}
        for qid, entity in payload.get("entities", {}).items():
            title = entity.get("sitelinks", {}).get("enwiki", {}).get("title")
            if not title:
                title = next(iter(entity.get("labels", {}).values()), {}).get("value")
            if title:
                result[qid] = title
        return result

    def fetch_pages(self, titles: list[str]) -> dict[str, dict]:
        payload = self._get(WIKIPEDIA_API, {
            "action": "query", "format": "json", "prop": "extracts|info",
            "explaintext": "1", "exsectionformat": "plain", "inprop": "url",
            "titles": "|".join(titles),
        })
        result = {}
        for page in payload.get("query", {}).get("pages", {}).values():
            if "missing" in page:
                continue
            result[page.get("title", "")] = page
        return result

    def build(self, input_path: Path, output_path: Path, limit: int | None = None) -> int:
        entities = json.loads(input_path.read_text(encoding="utf-8"))
        entities = entities[:limit] if limit else entities
        qids = [e["id"] for e in entities if e.get("id", "").startswith("Q")]
        records = []
        for start in range(0, len(qids), 50):
            titles = self.fetch_titles(qids[start:start + 50])
            pages = self.fetch_pages(list(titles.values()))
            for qid, title in titles.items():
                page = pages.get(title)
                if not page or not page.get("extract", "").strip():
                    continue
                records.append({
                    "id": f"{qid}-0", "qid": qid, "title": title,
                    "url": page.get("fullurl", f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"),
                    "contents": f"{title}\n{page['extract'].strip()}",
                })
            print(f"processed {min(start + 50, len(qids))}/{len(qids)} nodes; passages={len(records)}", flush=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return len(records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/wikidata/items_part_0.json"))
    parser.add_argument("--output", type=Path, default=Path("data/wiki-demo/corpus.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    n = WikiCorpusBuilder(timeout=args.timeout).build(args.input, args.output, args.limit)
    print(f"wrote {n} passages to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
