"""Offline evaluation of the RAG retrieval layer.

Loads queries and ground-truth incident mappings from eval/queries.jsonl,
queries the vector store across multiple k values (hit@1, hit@3, hit@5),
calculates recall metrics, and saves the results to eval/results/retrieval_metrics.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aiops.config import get_settings
from aiops.retrieval.ingest_incidents import IncidentRecord
from aiops.retrieval.store import ChromaVectorStore, RetrievalResult
from scripts.build_incident_corpus import SEED_INCIDENTS, save_corpus_json

QUERIES_PATH = Path("./eval/queries.jsonl")
RESULTS_DIR = Path("./eval/results")
CORPUS_PATH = Path("./data/incidents/corpus.json")


def ensure_corpus_seeded(store: ChromaVectorStore) -> None:
    """Ensure incidents are populated in the store before querying."""
    try:
        results = store.query("test", top_k=1)
        if results:
            return
    except Exception:
        pass

    print("Seeding vector store with standard corpus...")
    records = [IncidentRecord.model_validate(inc) for inc in SEED_INCIDENTS]
    store.add_incidents(records)


def evaluate_retrieval(queries_path: Path = QUERIES_PATH) -> dict[str, Any]:
    """Calculate hit@1, hit@3, hit@5 across labeled test queries."""
    settings = get_settings()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    store = ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection,
    )
    ensure_corpus_seeded(store)

    queries: list[dict[str, Any]] = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    k_values = [1, 3, 5]
    hits_at_k = {k: 0 for k in k_values}
    detailed_results = []

    for item in queries:
        query_text = item["query"]
        expected_ids = set(item["expected_incident_ids"])

        retrieved = store.query(query_text, top_k=max(k_values))
        retrieved_ids = [r.incident_id for r in retrieved]

        query_hits = {}
        for k in k_values:
            top_k_ids = set(retrieved_ids[:k])
            is_hit = bool(top_k_ids.intersection(expected_ids))
            if is_hit:
                hits_at_k[k] += 1
            query_hits[f"hit@{k}"] = is_hit

        detailed_results.append({
            "query": query_text,
            "expected": list(expected_ids),
            "retrieved_top_5": retrieved_ids[:5],
            "hits": query_hits,
        })

    num_queries = len(queries)
    metrics = {
        "num_queries": num_queries,
        "hit@1": round(hits_at_k[1] / num_queries, 4) if num_queries else 0.0,
        "hit@3": round(hits_at_k[3] / num_queries, 4) if num_queries else 0.0,
        "hit@5": round(hits_at_k[5] / num_queries, 4) if num_queries else 0.0,
        "details": detailed_results,
    }

    out_file = RESULTS_DIR / "retrieval_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n--- Retrieval Evaluation Results ---")
    print(f"Total Queries Evaluated: {num_queries}")
    print(f"Hit@1: {metrics['hit@1'] * 100:.1f}%")
    print(f"Hit@3: {metrics['hit@3'] * 100:.1f}%")
    print(f"Hit@5: {metrics['hit@5'] * 100:.1f}%")
    print(f"Results saved to {out_file}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG incident retrieval hit rates")
    parser.add_argument("--queries", type=Path, default=QUERIES_PATH, help="Path to labeled queries")
    args = parser.parse_args()
    evaluate_retrieval(args.queries)


if __name__ == "__main__":
    main()
