"""Prometheus metrics definitions for AIOps Assistant.

Monitors ingestion volume/lag, detection latency, anomaly counts by band,
retrieval performance, LLM token usage and latency, and approval metrics.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# --- Ingestion Metrics ---
LOGS_INGESTED_TOTAL = Counter(
    "aiops_logs_ingested_total",
    "Total number of log entries processed at ingestion",
    ["service", "status"],  # status: accepted, rejected, duplicate
)

INGESTION_BATCH_SIZE = Histogram(
    "aiops_ingestion_batch_size",
    "Distribution of received log batch sizes",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
)

DLQ_ENTRIES_TOTAL = Counter(
    "aiops_dlq_entries_total",
    "Total entries routed to the Dead Letter Queue",
    ["reason"],
)

# --- Detection Metrics ---
DETECTION_LATENCY_SECONDS = Histogram(
    "aiops_detection_latency_seconds",
    "Time spent parsing logs and computing anomaly scores",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

ANOMALY_EVENTS_TOTAL = Counter(
    "aiops_anomaly_events_total",
    "Count of detected anomalies classified by severity band",
    ["service", "band"],  # band: HIGH, LOW, NORMAL
)

DRIFT_DETECTED_TOTAL = Counter(
    "aiops_drift_detected_total",
    "Count of feature distribution drift events detected",
    ["service"],
)

# --- Retrieval (RAG) Metrics ---
RETRIEVAL_LATENCY_SECONDS = Histogram(
    "aiops_retrieval_latency_seconds",
    "Time spent querying the vector store for incident precedents",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
)

RETRIEVAL_RESULTS_COUNT = Histogram(
    "aiops_retrieval_results_count",
    "Number of historical incident chunks retrieved per query",
    buckets=[0, 1, 2, 3, 5, 10],
)

# --- Reasoning & LLM Metrics ---
LLM_REQUEST_LATENCY_SECONDS = Histogram(
    "aiops_llm_request_latency_seconds",
    "Time spent awaiting structured fix suggestions from Claude",
    ["model"],
    buckets=[0.5, 1.0, 2.0, 3.5, 5.0, 10.0, 15.0, 30.0],
)

LLM_TOKENS_TOTAL = Counter(
    "aiops_llm_tokens_total",
    "Total tokens consumed by LLM calls",
    ["model", "type"],  # type: input, output
)

LLM_VALIDATION_FAILURES_TOTAL = Counter(
    "aiops_llm_validation_failures_total",
    "Count of LLM outputs failing schema or reference validation",
    ["reason"],
)

# --- Actions & Human-in-the-Loop Metrics ---
INCIDENT_APPROVALS_TOTAL = Counter(
    "aiops_incident_approvals_total",
    "Human approval decisions on proposed remediations",
    ["action", "decision"],  # decision: approved, rejected, auto_approved
)

ACTIVE_INCIDENTS_GAUGE = Gauge(
    "aiops_active_incidents",
    "Current number of unresolved incidents in system",
    ["severity"],
)
