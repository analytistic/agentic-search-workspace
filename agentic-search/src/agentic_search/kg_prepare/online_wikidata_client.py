"""Online Wikidata client compatible with the KG path generator interface."""

from __future__ import annotations

import json
import os
import time
import argparse
from collections import OrderedDict
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import ProxyHandler, Request, build_opener


WIKIDATA_API = "https://www.wikidata.org/w/api.php"
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
DEFAULT_PROXY = "http://127.0.0.1:7890"


class OnlineWikidataClient:
    """Fetch only the Wikidata entities requested by an experiment.

    The client keeps a JSONL cache, so a path-generation run can be resumed
    without repeatedly querying Wikidata. It intentionally does not build a
    full-dump index or write the final KG dataset.
    """

    def __init__(
        self,
        *,
        api_url: str = WIKIDATA_API,
        sparql_endpoint: str = SPARQL_ENDPOINT,
        language: str = "en",
        proxy_url: str | None = None,
        timeout: float = 5.0,
        retries: int = 3,
        cache_size: int = 1024,
    ) -> None:
        self.api_url = api_url
        self.sparql_endpoint = sparql_endpoint
        self.language = language
        self.timeout = timeout
        self.retries = max(1, retries)
        self.cache_size = max(1, cache_size)
        if proxy_url is None:
            proxy_url = os.environ.get("WIKI_KG_PROXY", DEFAULT_PROXY)
        self.proxy_url = proxy_url
        self._opener = build_opener(
            ProxyHandler({"http": proxy_url, "https": proxy_url}) if proxy_url else ProxyHandler({})
        )
        self._lru: OrderedDict[str, dict[str, Any] | None] = OrderedDict()

    def _request(self, ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        params = urlencode({
            "action": "wbgetentities",
            "ids": "|".join(ids),
            "props": "labels|descriptions|aliases|claims|sitelinks",
            "languages": self.language,
            "languagefallback": "1",
            "format": "json",
        })
        request = Request(
            f"{self.api_url}?{params}",
            headers={"User-Agent": "agentic-search-online-kg/0.1"},
        )
        for attempt in range(self.retries):
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    payload = json.load(response)
                return {
                    qid: entity
                    for qid, entity in payload.get("entities", {}).items()
                    if isinstance(entity, dict) and "missing" not in entity
                }
            except Exception:
                if attempt + 1 >= self.retries:
                    raise
                time.sleep(attempt + 1)
        return {}

    def _random_qids(self, limit: int) -> list[str]:
        """Use Wikidata's random-item API only to obtain valid Q-IDs."""
        params = urlencode({"action": "query", "list": "random", "rnnamespace": "0", "rnlimit": str(limit), "format": "json"})
        request = Request(f"{WIKIDATA_API}?{params}", headers={"User-Agent": "agentic-search-online-kg/0.1"})
        with self._opener.open(request, timeout=self.timeout) as response:
            payload = json.load(response)
        return [item["title"] for item in payload.get("query", {}).get("random", []) if item.get("title", "").startswith("Q")]

    def _sparql(self, query: str) -> list[dict[str, Any]]:
        query = "PREFIX wd: <http://www.wikidata.org/entity/>\nPREFIX wdt: <http://www.wikidata.org/prop/direct/>\nPREFIX wikibase: <http://wikiba.se/ontology#>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\nPREFIX schema: <http://schema.org/>\n" + query
        params = urlencode({"query": query, "format": "json"})
        request = Request(
            f"{self.sparql_endpoint}?{params}",
            headers={"Accept": "application/sparql-results+json", "User-Agent": "agentic-search-online-kg/0.1"},
        )
        for attempt in range(self.retries):
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    payload = json.load(response)
                return [row for row in payload.get("results", {}).get("bindings", [])]
            except Exception:
                if attempt + 1 >= self.retries:
                    raise
                time.sleep(attempt + 1)
        return []

    @staticmethod
    def _binding(row: dict[str, Any], name: str) -> str:
        return row.get(name, {}).get("value", "")

    def _load_entity(self, qid: str) -> dict[str, Any] | None:
        """Load and assemble one complete entity, then store it in the LRU."""
        if qid in self._lru:
            entity = self._lru.pop(qid)
            self._lru[qid] = entity
            return entity
        rows = self._sparql(f'''SELECT ?label ?description WHERE {{
          VALUES ?item {{ wd:{qid} }}
          OPTIONAL {{ ?item rdfs:label ?label . FILTER(LANG(?label) = "en") }}
          OPTIONAL {{ ?item schema:description ?description . FILTER(LANG(?description) = "en") }}
        }} LIMIT 1''')
        relation_rows = self._sparql(f'''SELECT ?property ?propertyLabel ?target ?targetLabel WHERE {{
          ?property wikibase:directClaim ?directProperty .
          wd:{qid} ?directProperty ?target .
          FILTER(STRSTARTS(STR(?target), "http://www.wikidata.org/entity/Q"))
          OPTIONAL {{ ?property rdfs:label ?propertyLabel . FILTER(LANG(?propertyLabel) = "en") }}
          OPTIONAL {{ ?target rdfs:label ?targetLabel . FILTER(LANG(?targetLabel) = "en") }}
        }} LIMIT 200''')
        if not rows and not relation_rows:
            return None
        label = self._binding(rows[0], "label") if rows else qid
        description = self._binding(rows[0], "description") if rows else ""
        raw_relations = []
        display_relations = []
        for row in relation_rows:
            prop_id = self._binding(row, "property").rsplit("/", 1)[-1]
            target_id = self._binding(row, "target").rsplit("/", 1)[-1]
            if not prop_id or not target_id:
                continue
            raw_relations.append({"property": prop_id, "target": target_id})
            prop_label = self._binding(row, "propertyLabel")
            target_label = self._binding(row, "targetLabel")
            if prop_label and target_label:
                display_relations.append({"href": target_id, "text": prop_label, "title": target_label, "property_id": prop_id})
        entity = {
            "wikidata_id": qid,
            "wikipedia_title": label,
            "text": description,
            "relations": raw_relations,
            "display_relations": display_relations,
            "id": qid,
            "labels": {"en": {"value": label}},
            "descriptions": {"en": {"value": description}},
            "claims": {},
            "attributes": [],
        }
        self._lru[qid] = entity
        while len(self._lru) > self.cache_size:
            self._lru.popitem(last=False)
        return entity

    def get_entity(self, title_or_qid: str) -> dict[str, Any] | None:
        """Return one cached or online entity by Q-id or English label."""
        value = title_or_qid.strip()
        if value.startswith("Q") and value[1:].isdigit():
            return self._load_entity(value)
        for entity in self._lru.values():
            if self._label(entity).lower() == value.lower():
                return entity
        return None

    def _ensure_entity(self, title_or_qid: str) -> dict[str, Any] | None:
        value = title_or_qid.strip()
        if value.startswith("Q") and value[1:].isdigit():
            return self._load_entity(value)
        for entity in reversed(list(self._lru.values())):
            if self._label(entity).lower() == value.lower():
                return entity
        return None

    def _label(self, entity: dict[str, Any] | None) -> str:
        if not entity:
            return ""
        return entity.get("labels", {}).get(self.language, {}).get("value", entity.get("id", ""))

    def query_text(self, title_or_qid: str, **_: Any) -> str:
        entity = self.get_entity(title_or_qid)
        if not entity:
            return ""
        description = entity.get("descriptions", {}).get(self.language, {}).get("value", "")
        aliases = [a.get("value", "") for a in entity.get("aliases", {}).get(self.language, [])]
        return description + ("\n\nAlso known as: " + ", ".join(aliases) if aliases else "")

    def query_relations(self, title_or_qid: str) -> list[dict[str, Any]]:
        """Return entity-valued claims in the shape expected by path code."""
        entity = self.get_entity(title_or_qid)
        if not entity:
            return []
        if entity.get("display_relations"):
            return entity["display_relations"]
        qid = entity.get("wikidata_id", entity.get("id", ""))
        rows = self._sparql(f'''SELECT ?property ?propertyLabel ?target ?targetLabel WHERE {{
          ?property wikibase:directClaim ?directProperty .
          wd:{qid} ?directProperty ?target .
          FILTER(STRSTARTS(STR(?target), "http://www.wikidata.org/entity/Q"))
          OPTIONAL {{ ?property rdfs:label ?propertyLabel . FILTER(LANG(?propertyLabel) = "en") }}
          OPTIONAL {{ ?target rdfs:label ?targetLabel . FILTER(LANG(?targetLabel) = "en") }}
        }} LIMIT 200''')
        entity["display_relations"] = [{
            "href": self._binding(r, "target").rsplit("/", 1)[-1],
            "text": self._binding(r, "propertyLabel"),
            "title": self._binding(r, "targetLabel"),
            "property_id": self._binding(r, "property").rsplit("/", 1)[-1],
        } for r in rows if self._binding(r, "propertyLabel") and self._binding(r, "targetLabel")]
        return entity["display_relations"]

    def entity_exists(self, title_or_qid: str) -> bool:
        return self.get_entity(title_or_qid) is not None

    def get_attributes(self, title_or_qid: str, max_count: int | None = None) -> list[dict[str, Any]]:
        entity = self.get_entity(title_or_qid)
        if not entity:
            return []
        attrs = []
        for property_id, claims in entity.get("claims", {}).items():
            for claim in claims:
                value = claim.get("mainsnak", {}).get("datavalue", {}).get("value")
                if not isinstance(value, dict) or "id" not in value:
                    attrs.append({"property_id": property_id, "value": value})
        return attrs[:max_count] if max_count else attrs

    def get_entities_by_degree_range(self, min_degree: int, max_degree: int, limit: int = 100, **_: Any) -> list[dict[str, Any]]:
        """Return degree-filtered online candidates and retain them in LRU."""
        candidates = []
        seen: set[str] = set()
        attempts = 0
        while len(candidates) < limit and attempts < 10:
            attempts += 1
            qids = [qid for qid in self._random_qids(max(10, min(50, limit * 5))) if qid not in seen]
            seen.update(qids)
            if not qids:
                continue
            entities = self._request(qids)
            for qid, raw in entities.items():
                degree = sum(
                    1 for claims in raw.get("claims", {}).values()
                    for claim in claims
                    if claim.get("mainsnak", {}).get("datavalue", {}).get("type") == "wikibase-entityid"
                )
                if not min_degree <= degree <= max_degree:
                    continue
                if any(item["wikidata_id"] == qid for item in candidates):
                    continue
                label = raw.get("labels", {}).get(self.language, {}).get("value", qid)
                raw_claims = raw.get("claims", {})
                raw_relations = []
                for prop_id, claim_list in raw_claims.items():
                    for claim in claim_list:
                        value = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
                        target = value.get("id") if isinstance(value, dict) else None
                        if target and target.startswith("Q"):
                            raw_relations.append({"property": prop_id, "target": target})
                entity = {"wikidata_id": qid, "wikipedia_title": label, "text": raw.get("descriptions", {}).get(self.language, {}).get("value", ""), "relations": raw_relations, "id": qid, "labels": raw.get("labels", {}), "descriptions": raw.get("descriptions", {}), "claims": raw_claims, "attributes": []}
                self._lru[qid] = entity
                candidates.append({
                    "wikipedia_title": label,
                    "wikidata_id": qid,
                    "anchor_count": degree,
                })
                if len(candidates) >= limit:
                    break
        return candidates

    def compute_relation_entropy(self, title_or_qid: str) -> float:
        entity = self._ensure_entity(title_or_qid)
        relations = entity.get("relations", []) if entity else []
        if not relations:
            return 0.0
        counts: dict[str, int] = {}
        for relation in relations:
            key = relation.get("property", relation.get("property_id", ""))
            counts[key] = counts.get(key, 0) + 1
        total = sum(counts.values())
        return -sum((n / total) * __import__("math").log2(n / total) for n in counts.values())


__all__ = ["OnlineWikidataClient"]


def _print_entity(entity: dict[str, Any] | None, depth: int = 0) -> None:
    indent = "  " * depth
    if not entity:
        print(f"{indent}[missing entity]")
        return
    label = entity.get("labels", {}).get("en", {}).get("value", entity.get("id", ""))
    print(f"{indent}{entity.get('id')}  {label}")


def run_query_demo(client: OnlineWikidataClient, start_qid: str, hops: int) -> None:
    """Print a small, continuously queried graph walk for debugging."""
    current = start_qid
    visited = {current}
    print("Wikidata online query graph")
    for step in range(hops + 1):
        entity = client.get_entity(current)
        _print_entity(entity, step)
        if step == hops or not entity:
            break
        relations = client.query_relations(current)
        next_relation = next((r for r in relations if r["target_id"] not in visited), None)
        if not next_relation:
            print("  └─ no unvisited linked entity")
            break
        target = next_relation.get("target") or next_relation["target_id"]
        print(f"{'  ' * (step + 1)}└─[{next_relation['property_id']}]→ {target} ({next_relation['target_id']})")
        current = next_relation["target_id"]
        visited.add(current)


def interactive_shell(client: OnlineWikidataClient, start_qid: str | None = None) -> None:
    """Interactive terminal graph explorer."""
    if not start_qid:
        start_qid = input("输入起始 Wikidata Q-ID（例如 Q42，q 退出）: ").strip()
        if start_qid.lower() == "q" or not start_qid:
            return
    current = start_qid
    path: list[tuple[str, str, str]] = []
    while True:
        entity = client.get_entity(current)
        if not entity:
            print(f"无法获取实体 {current}（网络/API 超时）")
            command = input("输入 Q-ID 重试，或 q 退出: ").strip()
            if command.lower() == "q":
                return
            current = command
            continue
        label = client._label(entity)
        print("\n" + "=" * 72)
        print("当前实体:", current, label)
        print("路径:", " -> ".join([x[2] for x in path] + [label]))
        relations = client.query_relations(current)
        if not relations:
            print("没有可用关系。")
        else:
            for i, relation in enumerate(relations[:30]):
                print(f"[{i}] {relation['property_id']} -> {relation.get('target') or relation['target_id']} ({relation['target_id']})")
        raw_command = input("选择编号或 Q-ID 继续，b 返回，r 刷新，q 退出: ").strip()
        command = raw_command.lower()
        if command == "q":
            return
        if command == "r":
            continue
        if command == "b":
            if path:
                path.pop()
                current = path[-1][1] if path else start_qid
            continue
        if raw_command.startswith("Q") and raw_command[1:].isdigit():
            matches = [r for r in relations[:30] if r["target_id"] == raw_command]
            if not matches:
                print(f"当前实体的前 30 条关系中没有目标 {command}。")
                continue
            relation = matches[0]
            path.append((current, relation["target_id"], relation.get("target") or relation["target_id"]))
            current = relation["target_id"]
            continue
        if not command.isdigit() or int(command) >= len(relations[:30]):
            print("请输入列表中的编号。")
            continue
        relation = relations[int(command)]
        path.append((current, relation["target_id"], relation.get("target") or relation["target_id"]))
        current = relation["target_id"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Continuously query and display a Wikidata graph walk")
    parser.add_argument("--start", default=None, help="Starting Wikidata Q-id")
    parser.add_argument("--hops", type=int, default=3, help="Number of online query hops")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--auto", action="store_true", help="Run an automatic first-relation walk instead")
    args = parser.parse_args()
    client = OnlineWikidataClient(timeout=args.timeout)
    if args.auto:
        run_query_demo(client, args.start, args.hops)
    else:
        interactive_shell(client, args.start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
