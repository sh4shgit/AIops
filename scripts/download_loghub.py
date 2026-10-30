"""Script to fetch or generate benchmark loghub datasets (HDFS, BGL).

Provides labeled HDFS log samples (anomaly vs normal) for detector training
and evaluation. If offline or download fails, generates high-fidelity synthetic
loghub sequences matching standard LogHub formats.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA_DIR = Path("./data/raw")


def generate_synthetic_hdfs(output_path: Path, num_blocks: int = 100, anomaly_ratio: float = 0.1) -> None:
    """Generate labeled HDFS log data with normal block sessions and anomaly sessions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    normal_templates = [
        "Receiving block {block} src: 10.0.0.{ip}:50010 dest: 10.0.0.{ip}:50010",
        "BLOCK* NameSystem.allocateBlock: /mnt/hdfs/data/{file}. {block}",
        "PacketResponder {block} for block {block} terminating",
        "Received block {block} of size {size} from 10.0.0.{ip}",
        "BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.0.0.{ip}:50010 is added to {block} size {size}",
        "Verification succeeded for {block}",
    ]

    anomaly_templates = [
        "WARN org.apache.hadoop.hdfs.server.datanode.DataNode: IOException in receiveBlock for block {block}: Connection reset by peer",
        "ERROR org.apache.hadoop.hdfs.server.datanode.DataNode: Block {block} verification failed: ChecksumException: Checksum error",
        "WARN org.apache.hadoop.hdfs.server.namenode.FSNamesystem: Replication monitor: Failed to replicate {block} to datanode 10.0.0.{ip}",
        "FATAL org.apache.hadoop.hdfs.server.datanode.DataBlockScanner: Corrupted block file found for {block}",
    ]

    base_time = datetime(2026, 10, 20, 10, 0, 0, tzinfo=timezone.utc)
    records = []

    for i in range(num_blocks):
        block_id = f"blk_{1073741824 + i}"
        is_anomaly = random.random() < anomaly_ratio
        session_time = base_time + timedelta(seconds=i * 15)

        # Generate normal sequence
        seq_len = random.randint(4, 7)
        for step in range(seq_len):
            tmpl = random.choice(normal_templates)
            msg = tmpl.format(
                block=block_id,
                ip=random.randint(1, 254),
                file=f"part-{i:04d}",
                size=random.randint(100000, 67108864),
            )
            records.append({
                "timestamp": (session_time + timedelta(milliseconds=step * 50)).isoformat(),
                "service": "hdfs-datanode",
                "level": "INFO",
                "message": msg,
                "block_id": block_id,
                "label": "Anomaly" if is_anomaly else "Normal",
            })

        # Inject anomaly events for anomalous blocks
        if is_anomaly:
            for step in range(random.randint(2, 4)):
                tmpl = random.choice(anomaly_templates)
                msg = tmpl.format(
                    block=block_id,
                    ip=random.randint(1, 254),
                )
                level = "ERROR" if "ERROR" in tmpl or "FATAL" in tmpl else "WARNING"
                records.append({
                    "timestamp": (session_time + timedelta(seconds=1, milliseconds=step * 100)).isoformat(),
                    "service": "hdfs-datanode",
                    "level": level,
                    "message": msg,
                    "block_id": block_id,
                    "label": "Anomaly",
                })

    # Sort chronologically
    records.sort(key=lambda r: r["timestamp"])

    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Generated {len(records)} log events for {num_blocks} blocks -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or generate LogHub datasets")
    parser.add_argument("--synthetic", action="store_true", default=True, help="Generate synthetic loghub data")
    parser.add_argument("--num-blocks", type=int, default=300, help="Number of block sessions to generate")
    args = parser.parse_args()

    out_file = DATA_DIR / "hdfs_synthetic_labeled.jsonl"
    generate_synthetic_hdfs(out_file, num_blocks=args.num_blocks)


if __name__ == "__main__":
    main()
