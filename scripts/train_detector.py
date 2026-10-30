"""Offline training script for the Isolation Forest anomaly detector.

Parses training logs with Drain3, constructs session/window feature vectors,
fits the Isolation Forest model on normal-dominated data, registers drift
reference stats, and persists versioned model artifacts to data/models/.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from aiops.config import get_settings
from aiops.detection.drift import DriftDetector
from aiops.detection.features import FeatureExtractor
from aiops.detection.model import AnomalyDetector
from aiops.detection.parser import LogParser
from scripts.download_loghub import generate_synthetic_hdfs

RAW_DATA_PATH = Path("./data/raw/hdfs_synthetic_labeled.jsonl")
MODEL_DIR = Path("./data/models")


def load_or_create_dataset(path: Path) -> list[dict]:
    """Load JSON lines dataset or generate it if absent."""
    if not path.exists():
        print(f"Dataset {path} not found. Generating synthetic loghub data...")
        generate_synthetic_hdfs(path, num_blocks=400, anomaly_ratio=0.08)

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def train_pipeline(dataset_path: Path = RAW_DATA_PATH) -> None:
    """Run Drain3 -> FeatureExtractor -> Isolation Forest -> DriftDetector."""
    settings = get_settings()
    records = load_or_create_dataset(dataset_path)
    print(f"Loaded {len(records)} log events for training.")

    parser = LogParser()
    extractor = FeatureExtractor(service="hdfs-datanode")

    # Group logs by session block
    block_labels: dict[str, str] = {}
    for r in records:
        parsed_tmpl = parser.parse(r["message"])
        extractor.add_log(
            parsed_tmpl,
            timestamp=datetime.fromisoformat(r["timestamp"]),
            raw_message=r["message"],
            severity=r.get("level", "INFO"),
        )
        block_id = r.get("block_id")
        if block_id and block_id not in block_labels:
            block_labels[block_id] = r.get("label", "Normal")

    windows = extractor.close_session_windows()
    print(f"Constructed {len(windows)} session windows across {len(parser.templates)} templates.")

    # Filter to normal windows for training Isolation Forest
    normal_windows = [
        w for w in windows
        if block_labels.get(w.window_id.replace("session-", ""), "Normal") == "Normal"
    ]
    if not normal_windows:
        normal_windows = windows[: int(len(windows) * 0.9)]

    print(f"Training on {len(normal_windows)} normal session windows.")
    feature_matrix = np.array([extractor.get_feature_vector(w) for w in normal_windows])

    # Train Isolation Forest
    detector = AnomalyDetector()
    metadata = {
        "dataset": str(dataset_path),
        "total_events": len(records),
        "n_normal_windows": len(normal_windows),
        "n_templates": len(parser.templates),
        "feature_dim": feature_matrix.shape[1],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    artifact = detector.train(feature_matrix, metadata=metadata)
    print(f"Model version {artifact.version} saved to {artifact.path}")

    # Compute drift reference stats
    drift_detector = DriftDetector()
    drift_detector.fit(feature_matrix)
    drift_ref_path = artifact.path / "drift_reference.json"
    drift_detector.save(drift_ref_path)
    print(f"Drift reference profile saved to {drift_ref_path}")

    # Save latest link / pointer
    latest_meta = MODEL_DIR / "latest_model.json"
    with open(latest_meta, "w", encoding="utf-8") as f:
        json.dump(
            {
                "latest_version": artifact.version,
                "path": str(artifact.path),
                "metrics": artifact.metrics,
            },
            f,
            indent=2,
        )
    print(f"Updated latest model pointer: {latest_meta}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Isolation Forest anomaly detector")
    parser.add_argument("--data", type=Path, default=RAW_DATA_PATH, help="Path to labeled jsonl logs")
    args = parser.parse_args()
    train_pipeline(args.data)


if __name__ == "__main__":
    main()
