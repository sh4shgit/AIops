# AIOps Assistant: Final Application Spec

> **Purpose of this file:** a complete description of the finished application, so any Claude session (chat, Claude Code, Cowork) can understand what is being built and help design, implement, review, or document it.

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
