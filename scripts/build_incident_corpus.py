"""Build and seed the historical incident knowledge base for RAG retrieval.

Generates realistic SRE/DevOps incident precedents and optionally indexes
them into the vector store (ChromaDB).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from aiops.config import get_settings
from aiops.retrieval.example_incidents import SEED_INCIDENTS as SEED_INCIDENTS
from aiops.retrieval.ingest_incidents import IncidentRecord

INCIDENTS_DIR = Path("./data/incidents")


def save_corpus_json(output_path: Path) -> list[IncidentRecord]:
    """Save the seed incidents to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = [IncidentRecord.model_validate(inc) for inc in SEED_INCIDENTS]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump(mode="json") for r in records], f, indent=2)
    print(f"Saved {len(records)} incident records to {output_path}")
    return records


def index_into_vector_store(records: list[IncidentRecord]) -> None:
    """Index incident records into ChromaDB."""
    from aiops.retrieval.store import ChromaVectorStore

    settings = get_settings()
    store = ChromaVectorStore(
        persist_directory=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection,
    )
    store.add_incidents(records)
    print(f"Successfully indexed {len(records)} incidents into Chroma collection '{settings.chroma_collection}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and index incident knowledge base")
    parser.add_argument("--index", action="store_true", help="Index into ChromaDB vector store")
    args = parser.parse_args()

    out_file = INCIDENTS_DIR / "corpus.json"
    records = save_corpus_json(out_file)

    if args.index:
        index_into_vector_store(records)


if __name__ == "__main__":
    main()
