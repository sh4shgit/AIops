# AIOps Assistant: Final Application Spec

> **Purpose of this file:** a complete description of the finished application, so any Claude session (chat, Claude Code, Cowork) can understand what is being built and help design, implement, review, or document it. Read the whole file before working on the project. In Claude Code, copy it to `CLAUDE.md` in the repo root.

---

## 1. One-sentence pitch

**An AI assistant that reduces alert noise for DevOps/SRE teams by attaching retrieved precedent (similar past incidents and their fixes) to each detected log anomaly, then proposing a fix, with a human approving anything irreversible.**

Every feature must serve that sentence. Anything that doesn't is out of scope.

---

## 2. Problem

DevOps/SRE engineers are overwhelmed by alerts and spend time firefighting instead of improving reliability.

- Industry surveys report alert volumes in the thousands per week, with only a small fraction needing human action and a large share being false positives.
- Many engineers dismiss alerts without investigating, and a meaningful share of organizations report outages tied to ignored or suppressed alerts.
- Operational toil was reported to have risen in 2025 despite heavy AI tooling investment, which suggests naive "add an LLM" tools add noise without context.
- Engineers are more skeptical of AIOps tools than executives are.

**The gap targeted:** alerts arrive *without context*. The engineer must dig through logs and memory to learn what an alert means and what fixed it last time. Retrieval over past incidents supplies that context directly.

**Design stance:** humans stay in the loop for anything irreversible. This is a decision-support tool, not an autonomous fixer.

---

## 3. Scope

### In scope
- Ingest logs through an HTTP API and queue.
- Parse logs into templates and features, and detect anomalies with a single unsupervised model.
- Retrieve similar past incidents from a vector store.
- Produce a typed, validated fix suggestion with an LLM.
- Draft Jira tickets and Slack alerts, gated by an allowlist and human approval.
- Record every automated decision in an audit log.
- Expose metrics and a dashboard for the system's own health.
- Report measured quality: detector precision/recall/FPR and retrieval hit@k.

### Out of scope
- Autonomous destructive remediation (no unattended restarts, deletions, or config changes).
- Multiple anomaly models (LSTM and autoencoders were considered and deferred).
- Kafka, Kubernetes, Terraform, and Fluentd/Filebeat for v1.
- Multi-tenant auth, billing, or a polished web UI beyond a minimal review/approval view.

---

## 4. System overview

```
Logs (public dataset / synthetic / API clients)
        |
        v
[1] Ingestion         FastAPI + Redis Streams + dead-letter queue
        |
        v
[2] Detection         Drain3 parsing -> windowed features -> Isolation Forest -> confidence routing
        |
        v
[3] Retrieval (RAG)   sentence-transformers -> ChromaDB (or pgvector) via LangChain
        |
        v
[4] Reasoning         scrubber -> Claude API (structured output) in a LangGraph flow
        |
        v
[5] Actions           allowlist -> draft Jira/Slack payloads -> human approval -> audit log
```

Cross-cutting: observability (Prometheus, Grafana, structured logs), security (scrubbing, least privilege, allowlist), evaluation (offline metrics), configuration (env-based settings).

---

## 5. Components

### 5.1 Ingestion
- `POST /logs` accepts a batch (`list[dict]`) and validates **each entry individually** with the `LogEntry` model, so one malformed line never rejects the batch.
- Response: `{accepted, rejected, errors[]}`. An empty batch returns 422.
- Valid entries get a deterministic ID (hash of timestamp + service + message) and are deduplicated, since delivery is at-least-once.
- Accepted entries are written to a Redis Stream; workers consume via a consumer group and acknowledge after processing.
- Rejected or repeatedly failing entries go to a **dead-letter stream** with the failure reason.
- `GET /health` for liveness/readiness (checks Redis and vector store reachability).

**LogEntry:** `timestamp` (datetime), `service` (1-100 chars), `level` (DEBUG/INFO/WARNING/ERROR/CRITICAL), `message` (1-10,000 chars, not blank).

### 5.2 Detection
- **Parsing:** Drain3 converts each message into a template ID plus extracted parameters.
- **Features:** per-window (or per-session, for HDFS-style data) counts of each template, error-level ratios, and rate changes, assembled into a feature vector.
- **Model:** one Isolation Forest (scikit-learn), trained offline on normal-dominated data. Artifacts are versioned (model file + metadata: training data, parameters, metrics, date).
- **Confidence routing:** anomaly score maps to three bands:
  - *high*: proceed through the full retrieval and reasoning flow
  - *low*: create a low-priority incident for review, skip automation
  - *normal*: ignore
  Thresholds are configurable and chosen from the evaluation results.
- **Later hardening:** drift check comparing incoming feature distributions to training data, with a "retrain recommended" flag.

### 5.3 Retrieval (RAG)
- **Corpus:** past incidents as structured records: title, symptoms, log templates involved, root cause, resolution, services affected, resolved date, status.
- **Embedding:** `sentence-transformers` (`all-MiniLM-L6-v2`), chunked by incident section so symptoms and resolutions are retrievable separately.
- **Store:** ChromaDB for local use; pgvector as an optional production-style backend behind the same interface.
- **Query:** built from the anomaly's key templates and context; returns top-k incidents with similarity scores and metadata filters (service, status).
- **Staleness:** incidents can be marked superseded/invalid and are filtered or down-weighted; re-embedding on update is supported, so the store is not append-only.

### 5.4 Reasoning
- **Scrubber (runs first):** removes or masks secrets, tokens, emails, IPs, hostnames, and other PII/identifiers before any text goes to the LLM.
- **LLM call:** Claude API with structured output. Required response schema (`FixSuggestion`):
  - `summary`, `probable_cause`, `suggested_steps[]`, `confidence` (0-1), `risk_level` (low/medium/high), `referenced_incident_ids[]`, `proposed_action` (optional, from the allowlist enum only)
- **Validation:** output is parsed with Pydantic; invalid or schema-violating output is rejected and retried or escalated. References to incident IDs must exist in the retrieved set.
- **Orchestration:** LangGraph state machine: `detect -> retrieve -> reason -> validate -> propose -> [approval interrupt] -> act -> record`.
- **Cost/latency tracking:** tokens and latency per call are recorded as metrics.

### 5.5 Actions
- **Allowlist:** a fixed enum of permitted actions (e.g. `draft_jira_ticket`, `send_slack_alert`, and optionally a reversible `restart_service` with preconditions such as a failed health check and a not-in-freeze-window check). Anything else is refused.
- **Approval gate:** drafting is automatic; anything with side effects waits for explicit approval via the API (`POST /incidents/{id}/approve` or `/reject`) or the minimal review view.
- **Integrations:** Jira REST and Slack SDK clients with least-privilege credentials; default mode is "draft payload only."
- **Audit log:** append-only record per decision: trigger, anomaly score, retrieved incident IDs, model output, validation result, action, approver, timestamp, outcome.
- **Incident report:** a generated summary (markdown) per incident combining the above.

---

## 6. API surface

| Method | Path | Purpose |
|---|---|---|
| POST | `/logs` | Ingest a batch of log entries |
| GET | `/health` | Liveness/readiness |
| GET | `/incidents` | List incidents (filter by status, service, severity) |
| GET | `/incidents/{id}` | Incident detail: anomaly, retrieved precedent, suggestion, draft payloads, audit trail |
| POST | `/incidents/{id}/approve` | Approve the proposed action |
| POST | `/incidents/{id}/reject` | Reject with reason |
| GET | `/metrics` | Prometheus metrics |

Requests that mutate state require an API key. Basic rate limiting on `/logs`.

---

## 7. Technology stack

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python | ML/LLM ecosystem is Python-first; LLM latency dominates, so language speed isn't the bottleneck |
| API | FastAPI + Pydantic | Built-in validation, async, auto docs, easy testing |
| Queue | Redis Streams | Lightweight, supports consumer groups, acknowledgments, and DLQ patterns |
| Parsing | Drain3 | Standard log template mining |
| Detector | Isolation Forest (scikit-learn) | Simple, unsupervised, CPU-only, explainable |
| Data | loghub (HDFS, BGL) | Labeled public datasets enable honest evaluation |
| Embeddings | sentence-transformers | Local and free |
| Vector store | ChromaDB (optionally pgvector) | Zero-infra locally; production-style option |
| RAG | LangChain | Retrieval orchestration |
| Agent flow | LangGraph | Stateful flow with human-in-the-loop interrupts |
| LLM | Claude API (structured output) | Typed, validated suggestions |
| Integrations | Jira REST, Slack SDK | Ticketing and alerting |
| Observability | Prometheus, Grafana, structlog | Self-monitoring |
| Packaging | Docker, docker-compose | One-command stack |
| CI/CD | GitHub Actions | Lint, test, build on push |
| Testing | pytest, locust | Unit, integration, load |

---

## 8. Key design decisions

1. **One detector, done properly**, rather than three half-built ones.
2. **Confidence bands, not binary flags**, because false positives are what kill AIOps adoption.
3. **Per-entry validation** and a dead-letter path, so bad data degrades gracefully.
4. **Idempotent ingestion**, because at-least-once delivery produces duplicates.
5. **Structured, validated LLM output**, and actions only from a fixed allowlist.
6. **No autonomous destructive actions**; approval required for side effects.
7. **Full audit trail** of every automated decision.
8. **Scrub before the LLM**; never send raw logs externally.
9. **Evaluation as a first-class feature**, with results published in the README.
10. **Versioned model artifacts** and a documented retraining and drift story.
11. **Non-append-only vector store**, with staleness handled explicitly.
12. **Pluggable interfaces** (vector store, LLM client, action executors) so components can be swapped and mocked in tests.
