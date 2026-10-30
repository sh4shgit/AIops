"""Unit tests for confidence band routing."""

from __future__ import annotations

from aiops.detection.thresholds import AnomalyBand, classify_anomaly


def test_classify_anomaly_high() -> None:
    # High threshold default is -0.3
    band = classify_anomaly(-0.45)
    assert band == AnomalyBand.HIGH


def test_classify_anomaly_low() -> None:
    # Low threshold default is -0.1
    band = classify_anomaly(-0.20)
    assert band == AnomalyBand.LOW


def test_classify_anomaly_normal() -> None:
    band = classify_anomaly(0.15)
    assert band == AnomalyBand.NORMAL


def test_classify_anomaly_exact_boundary() -> None:
    # Score <= high_threshold -> HIGH
    band = classify_anomaly(-0.3)
    assert band == AnomalyBand.HIGH
