"""AIOps Assistant observability, metrics, and structured logging."""

from __future__ import annotations

from aiops.observability.logging import (
    bind_correlation_id,
    clear_correlation_id,
    setup_logging,
)
from aiops.observability.metrics import (
    ANOMALY_EVENTS_TOTAL,
    DETECTION_LATENCY_SECONDS,
    DLQ_ENTRIES_TOTAL,
    DRIFT_DETECTED_TOTAL,
    INCIDENT_APPROVALS_TOTAL,
    INGESTION_BATCH_SIZE,
    LLM_REQUEST_LATENCY_SECONDS,
    LLM_TOKENS_TOTAL,
    LLM_VALIDATION_FAILURES_TOTAL,
    LOGS_INGESTED_TOTAL,
    RETRIEVAL_LATENCY_SECONDS,
    RETRIEVAL_RESULTS_COUNT,
)

__all__ = [
    "ANOMALY_EVENTS_TOTAL",
    "DETECTION_LATENCY_SECONDS",
    "DLQ_ENTRIES_TOTAL",
    "DRIFT_DETECTED_TOTAL",
    "INCIDENT_APPROVALS_TOTAL",
    "INGESTION_BATCH_SIZE",
    "LLM_REQUEST_LATENCY_SECONDS",
    "LLM_TOKENS_TOTAL",
    "LLM_VALIDATION_FAILURES_TOTAL",
    "LOGS_INGESTED_TOTAL",
    "RETRIEVAL_LATENCY_SECONDS",
    "RETRIEVAL_RESULTS_COUNT",
    "bind_correlation_id",
    "clear_correlation_id",
    "setup_logging",
]
