"""Online KG path generation entry point, aligned with the third-party pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .distractor_generator import DistractorConfig, DistractorGenerator
from .llm_path_generator import LLMPathConfig, LLMPathGenerator
from .online_wikidata_client import OnlineWikidataClient
from .path_generator import PathConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-paths", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("data/kg-online/data.jsonl"))
    parser.add_argument("--min-hops", type=int, default=2)
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--min-degree", type=int, default=5)
    parser.add_argument("--max-degree", type=int, default=50)
    parser.add_argument("--min-entropy", type=float, default=1.5)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    client = OnlineWikidataClient(timeout=args.timeout, cache_size=2048)
    path_config = LLMPathConfig(
        min_hops=args.min_hops,
        max_hops=args.max_hops,
        llm_model="direct_dsr1",
        strict_path_filter=True,
        node_filter_config=__import__("agentic_search.kg_prepare.node_filter", fromlist=["NodeFilterConfig"]).NodeFilterConfig(
            min_out_degree=args.min_degree,
            max_out_degree=args.max_degree,
            min_entropy=args.min_entropy,
        ),
        fallback_to_random=True,
    )
    generator = LLMPathGenerator(client, path_config)
    distractors = DistractorGenerator(client, DistractorConfig(min_distractors=0, max_distractors=3))
    candidates = client.get_entities_by_degree_range(args.min_degree, args.max_degree, args.num_paths * 5)
    seeds = []
    for item in candidates:
        qid = item["wikidata_id"]
        entropy = client.compute_relation_entropy(qid)
        title = item["wikipedia_title"]
        if entropy >= args.min_entropy and generator.node_filter.check_label_quality(title):
            seeds.append(qid)
        if len(seeds) >= args.num_paths:
            break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for seed in seeds:
            entity = client._ensure_entity(seed)
            seed_title = entity.get("wikipedia_title", seed) if entity else seed
            path = generator.generate_path_from_seed(seed_title)
            if not path:
                continue
            qa = distractors.create_qa_with_distractors(path, 1)
            if not qa:
                continue
            handle.write(json.dumps(qa.to_dict(), ensure_ascii=False) + "\n")
            written += 1
            print(f"wrote path {written}/{args.num_paths}: {path.to_dict()['path']}", flush=True)
            if written >= args.num_paths:
                break
    print(f"wrote {written} KG paths to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
