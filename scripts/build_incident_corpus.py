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
from aiops.retrieval.ingest_incidents import IncidentRecord
from aiops.retrieval.store import ChromaVectorStore

INCIDENTS_DIR = Path("./data/incidents")

SEED_INCIDENTS: list[dict] = [
    {
        "id": "INC-1001",
        "title": "HDFS DataNode Connection Pool Exhaustion under Heavy Influx",
        "symptoms": (
            "DataNode threads threw Connection reset by peer and java.io.IOException. "
            "Replication monitor logs showed failed block transfers across worker nodes. "
            "Pipeline write latency spiked from 12ms to 4500ms."
        ),
        "log_templates": [
            "IOException in receiveBlock for block <*>",
            "Connection reset by peer",
            "Failed to replicate <*> to datanode <*>",
        ],
        "root_cause": (
            "Linux ephemeral port exhaustion combined with insufficient dfs.datanode.handler.count "
            "under a 4x sudden burst in batch write traffic."
        ),
        "resolution": (
            "1. Increased net.ipv4.ip_local_port_range to 1024-65535 on all worker nodes.\n"
            "2. Increased dfs.datanode.handler.count from 10 to 64 in hdfs-site.xml.\n"
            "3. Gracefully restarted DataNode service worker fleet."
        ),
        "services": ["hdfs-datanode", "hdfs-namenode"],
        "status": "active",
        "resolved_at": "2026-08-15T14:30:00Z",
    },
    {
        "id": "INC-1002",
        "title": "Corrupt Block Checksum Failure during Storage Volume Fault",
        "symptoms": (
            "DataBlockScanner reported ChecksumException and Corrupted block file found. "
            "Read jobs failed with CRC32 mismatch on storage volume /data/disk2."
        ),
        "log_templates": [
            "Block <*> verification failed: ChecksumException: Checksum error",
            "Corrupted block file found for <*>",
        ],
        "root_cause": (
            "Silent hardware block corruption on SSD backplane controller triggering bad sector reads."
        ),
        "resolution": (
            "1. Unmounted faulty volume /data/disk2 and removed from dfs.data.dir.\n"
            "2. Triggered NameSystem block recalculation to re-replicate corrupted blocks from healthy replicas.\n"
            "3. Scheduled hot-swap hardware replacement."
        ),
        "services": ["hdfs-datanode"],
        "status": "active",
        "resolved_at": "2026-09-02T09:15:00Z",
    },
    {
        "id": "INC-1003",
        "title": "NameNode Garbage Collection Stop-the-World Pause",
        "symptoms": (
            "NameNode failed heartbeats from multiple DataNodes simultaneously. "
            "Clients experienced connection timeouts to NameSystem RPC port 8020. "
            "Garbage collection pauses exceeded 45 seconds."
        ),
        "log_templates": [
            "Heartbeat monitor: timed out waiting for datanode response",
            "Call to namenode:8020 failed on connection exception",
        ],
        "root_cause": (
            "JVM heap allocation was undersized (8GB) for a filesystem containing over 40 million inodes, "
            "causing full concurrent Mark-Sweep GC loops."
        ),
        "resolution": (
            "1. Tuned JVM options: switched to G1GC with -XX:MaxGCPauseMillis=200.\n"
            "2. Increased HADOOP_NAMENODE_OPTS heap allocation to 32GB.\n"
            "3. Failed over to Standby NameNode, applied config, and failed back."
        ),
        "services": ["hdfs-namenode"],
        "status": "active",
        "resolved_at": "2026-09-18T18:40:00Z",
    },
    {
        "id": "INC-1004",
        "title": "Authentication Token Expiration in Microservice API Gateway",
        "symptoms": (
            "API Gateway returned HTTP 401 and 403 on valid user requests. "
            "Downstream auth service reported JWT signature verification failure."
        ),
        "log_templates": [
            "Authentication failed: token expired or clock skew detected",
            "JWT verification failed for issuer <*>",
        ],
        "root_cause": (
            "NTP daemon desynchronization on worker cluster nodes caused a 6-minute clock drift, "
            "violating JWT token validity window."
        ),
        "resolution": (
            "1. Resynced chrony / NTP daemon across all gateway nodes.\n"
            "2. Added clock drift alert to node-exporter Prometheus alerts.\n"
            "3. Restarted gateway auth pod instances."
        ),
        "services": ["api-gateway", "auth-service"],
        "status": "active",
        "resolved_at": "2026-10-01T11:20:00Z",
    },
    {
        "id": "INC-1005",
        "title": "Redis Cache Eviction Storm Triggering Cascading DB Saturation",
        "symptoms": (
            "Redis maxmemory reached. Key eviction rates spiked to 45,000 keys/sec. "
            "Downstream PostgreSQL database connections saturated to 100% capacity."
        ),
        "log_templates": [
            "OOM command not allowed when used memory > 'maxmemory'",
            "PostgreSQL connection limit exceeded: too many clients",
        ],
        "root_cause": (
            "Cache stampede following a cache flush caused by a deployment script, "
            "with missing mutex/lock on cache misses."
        ),
        "resolution": (
            "1. Scaled Redis cache memory limit and switched policy to allkeys-lru.\n"
            "2. Added singleflight probabilistic early expiration in client code.\n"
            "3. Cleared stalled Postgres connections via pg_terminate_backend."
        ),
        "services": ["redis-cache", "postgres-db"],
        "status": "active",
        "resolved_at": "2026-10-12T16:05:00Z",
    },
]


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
    settings = get_settings()
    store = ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
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
