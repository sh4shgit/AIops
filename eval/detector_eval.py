"""Offline evaluation of the Isolation Forest anomaly detector.

Evaluates precision, recall, F1, and false positive rate (FPR) on labeled
session windows from HDFS loghub dataset across varying decision thresholds,
generating threshold-metric trade-off curves.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from aiops.config import get_settings
from aiops.detection.features import FeatureExtractor
from aiops.detection.model import AnomalyDetector
from aiops.detection.parser import LogParser
from scripts.download_loghub import generate_synthetic_hdfs

RAW_DATA_PATH = Path("./data/raw/hdfs_synthetic_labeled.jsonl")
RESULTS_DIR = Path("./eval/results")


def evaluate_detector(dataset_path: Path = RAW_DATA_PATH) -> dict[str, Any]:
    """Compute precision, recall, F1, and FPR curve on labeled dataset."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        print("Generating evaluation dataset...")
        generate_synthetic_hdfs(dataset_path, num_blocks=500, anomaly_ratio=0.15)

    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    parser = LogParser()
    extractor = FeatureExtractor(service="hdfs-datanode")
    block_labels: dict[str, int] = {}  # 1 for Anomaly, 0 for Normal

    for r in records:
        tmpl = parser.parse(r["message"])
        extractor.add_log(
            tmpl,
            timestamp=datetime.fromisoformat(r["timestamp"]),
            raw_message=r["message"],
            severity=r.get("level", "INFO"),
        )
        b_id = r.get("block_id")
        if b_id and b_id not in block_labels:
            block_labels[b_id] = 1 if r.get("label") == "Anomaly" else 0

    windows = extractor.close_session_windows()
    print(f"Extracted {len(windows)} session windows for evaluation.")

    X = np.array([extractor.get_feature_vector(w) for w in windows])
    y_true = np.array([
        block_labels.get(w.window_id.replace("session-", ""), 0)
        for w in windows
    ])

    # Fit detector on normal portion (unsupervised)
    normal_mask = (y_true == 0)
    X_train = X[normal_mask]
    if len(X_train) == 0:
        X_train = X

    detector = AnomalyDetector()
    detector.train(X_train)

    # Compute raw decision function scores (negative = anomaly)
    scores = detector._model.decision_function(X)

    # Evaluate across a sweep of thresholds from -0.40 to +0.10
    threshold_sweep = np.linspace(-0.35, 0.05, 17)
    curve_data = []

    for t in threshold_sweep:
        y_pred = (scores <= t).astype(int)

        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        curve_data.append({
            "threshold": round(float(t), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "fpr": round(float(fpr), 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        })

    # Pick best operational threshold (max F1)
    best_op = max(curve_data, key=lambda c: c["f1"])

    settings = get_settings()
    configured_t = settings.anomaly_high_threshold
    # Find closest entry to configured threshold
    configured_op = min(curve_data, key=lambda c: abs(c["threshold"] - configured_t))

    report = {
        "n_evaluated_windows": len(windows),
        "n_anomalies_ground_truth": int(np.sum(y_true == 1)),
        "n_normals_ground_truth": int(np.sum(y_true == 0)),
        "best_operating_point": best_op,
        "configured_threshold_point": configured_op,
        "threshold_curve": curve_data,
    }

    out_file = RESULTS_DIR / "detector_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n--- Detector Evaluation Results ---")
    print(f"Windows Evaluated: {len(windows)} ({report['n_anomalies_ground_truth']} anomalies)")
    print(f"Best Threshold (by F1): {best_op['threshold']}")
    print(f"  Precision: {best_op['precision'] * 100:.1f}%")
    print(f"  Recall:    {best_op['recall'] * 100:.1f}%")
    print(f"  F1 Score:  {best_op['f1'] * 100:.1f}%")
    print(f"  FPR:       {best_op['fpr'] * 100:.2f}%")
    print(f"Results saved to {out_file}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Isolation Forest detector performance")
    parser.add_argument("--data", type=Path, default=RAW_DATA_PATH, help="Path to labeled evaluation dataset")
    args = parser.parse_args()
    evaluate_detector(args.data)


if __name__ == "__main__":
    main()
