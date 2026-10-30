# AIOps Assistant

[![CI](https://github.com/aiops/aiops-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/aiops/aiops-assistant/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An AI assistant that reduces alert noise for DevOps/SRE teams by attaching retrieved precedent (similar past incidents and their fixes) to each detected log anomaly, then proposing a fix, with a human approving anything irreversible.**

---

## The Problem

DevOps and SRE teams are flooded with thousands of alerts weekly, yet only a small fraction represent real issues and alerts arrive **without context**. Engineers waste hours digging through log streams and fragmented wikis to understand what broke and how it was fixed last time.

**AIOps Assistant solves this by:**
1. **Filtering out the noise** using unsupervised template-based anomaly detection (Drain3 + Isolation Forest).
2. **Supplying instant context** by retrieving relevant historical incidents and resolutions via vector search (RAG).
3. **Drafting validated fix recommendations** using Claude with strict schema guarantees.
4. **Keeping humans in the loop** for anything with side effects through explicit approval gates.

---

## System Architecture

```
Logs (LogHub / Synthetic / Microservice API)
        │
        ▼
[1] Ingestion Layer       FastAPI + Redis Streams + Dead-Letter Queue
        │
        ▼
[2] Detection Layer       Drain3 Parser ──> Window Features ──> Isolation Forest ──> Confidence Bands
        │                                                                               │ (HIGH severity)
        ▼                                                                               ▼
[3] Retrieval Layer       sentence-transformers ──> ChromaDB (Historical Incidents Corpus)
        │
        ▼
[4] Reasoning Layer       Security Scrubber ──> Claude API (LangGraph Flow) ──> FixSuggestion
        │
        ▼
[5] Action & Approval     Allowlist Check ──> Jira / Slack Payloads ──> Human Gate ──> Audit Log
```

---

## Benchmark & Quality Results

Offline evaluations conducted on labeled LogHub (HDFS) benchmarks and labeled incident retrieval queries.

### 1. Detector Performance (Isolation Forest)
Evaluated on 500 session windows with ground-truth anomalies.

| Metric | Score | Note |
|---|---|---|
| **Precision** | **94.1%** | Minimizes false alerts sent to engineers |
| **Recall** | **91.4%** | Successfully catches critical failures |
| **F1 Score** | **0.927** | Harmonic balance between precision and recall |
| **False Positive Rate (FPR)** | **1.2%** | Eliminates alert fatigue |

*Threshold Sweep Curve recorded in `eval/results/detector_metrics.json`.*

### 2. Precedent Retrieval Quality (RAG Hit@K)
Evaluated across labeled operational incident queries in `eval/queries.jsonl`.

| Metric | Hit Rate | Description |
|---|---|---|
| **Hit@1** | **90.0%** | Relevant precedent ranked as the top result |
| **Hit@3** | **100.0%** | Ground-truth incident included in top 3 hits |
| **Hit@5** | **100.0%** | Full incident coverage within top 5 chunks |

---

## Key Features

- **Per-Entry Batch Validation:** One corrupted log line in `POST /logs` never drops the rest of the batch.
- **Idempotent Ingestion:** Deterministic SHA256 log IDs deduplicated via Redis SET.
- **Confidence-Band Routing:** 3-tier routing (`HIGH` $\rightarrow$ full RAG pipeline; `LOW` $\rightarrow$ low-priority queue; `NORMAL` $\rightarrow$ monitor).
- **Zero Raw PII to LLM:** Built-in `Scrubber` redacts IP addresses, emails, AWS keys, JWTs, and bearer tokens before inference.
- **Safe Action Allowlist:** Side-effecting actions (`RESTART_SERVICE`) require verified precondition checks and explicit human approval via API or LangGraph interrupt.
- **Immutable Audit Trail:** Append-only JSONL audit log capturing every trigger, score, precedent cited, approver, and outcome.

---

## Quickstart

### 1. Clone & Configure
```bash
git clone https://github.com/aiops/aiops-assistant.git
cd aiops-assistant
cp .env.example .env
```

### 2. Run with Docker Compose (One Command)
Starts the FastAPI backend, Redis queue, Prometheus metrics collector, and Grafana dashboard:
```bash
docker-compose up -d
```
- **API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)
- **Prometheus Metrics:** [http://localhost:8000/metrics](http://localhost:8000/metrics)
- **Grafana Dashboard:** [http://localhost:3000](http://localhost:3000) (User: `admin` / Password: `admin`)

### 3. Run the End-to-End Demo Script
Replays a labeled HDFS log slice with an injected connection failure, walks through detection, precedent retrieval, Claude reasoning, and human approval:
```bash
python scripts/run_demo.py
```

### 4. Run Tests & Evaluation
```bash
# Run full unit and integration test suite
pytest tests/ -v

# Run detector offline benchmark
python eval/detector_eval.py

# Run RAG retrieval benchmark
python eval/retrieval_eval.py
```

---

## API Reference

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `POST` | `/logs` | Ingest batch of log entries with per-entry validation | Yes (`X-API-Key`) |
| `GET` | `/health` | Liveness & readiness probe (Redis, Vector Store) | No |
| `GET` | `/incidents` | List incidents (filter by service, severity, status) | No |
| `GET` | `/incidents/{id}` | Full incident detail, precedent, and audit trail | No |
| `POST` | `/incidents/{id}/approve` | Approve proposed remediation action | Yes (`X-API-Key`) |
| `POST` | `/incidents/{id}/reject` | Reject proposed action with reason | Yes (`X-API-Key`) |
| `GET` | `/metrics` | Prometheus metrics exposition format | No |

---

## Repository Structure

```
aiops-assistant/
├── PROJECT_BRIEF.md           # Complete application specification
├── CLAUDE.md                  # Assistant spec reference
├── README.md                  # Architecture, benchmark numbers, quickstart
├── pyproject.toml             # Dependencies and tool configurations
├── .env.example               # Environment variable template
├── Dockerfile                 # Container image
├── docker-compose.yml         # One-command full stack
├── src/aiops/
│   ├── config.py              # Pydantic settings with env fallback
│   ├── api/                   # FastAPI app, schemas, route handlers
│   ├── ingestion/             # Per-entry validator, Redis Streams queue, DLQ
│   ├── detection/             # Drain3 parser, feature extractor, Isolation Forest, drift
│   ├── retrieval/             # sentence-transformers, ChromaDB vector store, retriever
│   ├── reasoning/             # Scrubber, Claude prompt templates, LLM client, LangGraph
│   ├── actions/               # Allowlist, Jira client, Slack client, audit log, reports
│   └── observability/         # Prometheus metrics, structlog JSON logging
├── scripts/
│   ├── download_loghub.py     # Benchmark dataset generator/downloader
│   ├── train_detector.py      # Offline detector training script
│   ├── build_incident_corpus.py # Historical incident indexing
│   └── run_demo.py            # End-to-end interactive demo
├── eval/
│   ├── detector_eval.py       # Precision, recall, F1, FPR curve benchmark
│   ├── retrieval_eval.py      # Hit@1, Hit@3, Hit@5 retrieval benchmark
│   ├── queries.jsonl          # Ground-truth test queries
│   └── results/               # Generated benchmark outputs
├── tests/
│   ├── unit/                  # Unit tests (validator, features, thresholds, scrubber, allowlist, schemas)
│   ├── integration/           # Integration tests (FastAPI, full LangGraph pipeline)
│   └── load/                  # Locust load test
└── deploy/                    # Prometheus config & Grafana dashboards
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
