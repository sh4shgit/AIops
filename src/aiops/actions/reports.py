"""Markdown incident report generator for AIOps Assistant.

Synthesizes detector findings, vector store retrieval hits, LLM fix suggestions,
human-in-the-loop decisions, and audit history into a cohesive post-incident report.
"""

from __future__ import annotations

from typing import Any

from aiops.actions.audit import AuditRecord
from aiops.detection.thresholds import AnomalyEvent
from aiops.reasoning.schemas import FixSuggestion
from aiops.retrieval.store import RetrievalResult


def generate_report(
    anomaly: AnomalyEvent,
    retrieved: list[RetrievalResult],
    suggestion: FixSuggestion | None,
    audit_records: list[AuditRecord] | None = None,
) -> str:
    """Generate a clean, structured Markdown report for an incident."""
    audit_records = audit_records or []
    
    # 1. Header
    report = [
        f"# Incident Report: {anomaly.service} [{anomaly.band.value}]",
        "",
        f"- **Event ID:** `{anomaly.event_id}`",
        f"- **Service:** `{anomaly.service}`",
        f"- **Detected At:** {anomaly.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- **Anomaly Score:** `{anomaly.anomaly_score:.4f}` (Band: **{anomaly.band.value}**)",
        f"- **Model Version:** `{anomaly.model_version}`",
        "",
        "---",
        "",
        "## 1. Anomaly Overview",
        "",
    ]

    if anomaly.key_templates:
        report.append("**Key Log Templates Detected:**")
        for t in anomaly.key_templates:
            report.append(f"- `{t}`")
        report.append("")

    if anomaly.raw_log:
        report.extend([
            "**Sample Raw Log:**",
            "```text",
            anomaly.raw_log.strip(),
            "```",
            "",
        ])

    # 2. Retrieved Historical Precedent
    report.extend([
        "## 2. Retrieved Historical Precedents (RAG)",
        "",
    ])

    if not retrieved:
        report.append("_No matching historical incidents were found in the knowledge base._\n")
    else:
        for idx, item in enumerate(retrieved, start=1):
            score_pct = item.similarity_score * 100
            report.extend([
                f"### #{idx}: Incident `{item.incident_id}` (Similarity: {score_pct:.1f}%)",
                f"- **Section:** `{item.chunk_type}`",
                f"- **Title:** {item.title}",
                "",
                f"> {item.content.strip()}",
                "",
            ])

    # 3. LLM Fix Suggestion & Root Cause Analysis
    report.extend([
        "## 3. Reasoning & Fix Recommendation",
        "",
    ])

    if suggestion is None:
        report.append("_No automated fix suggestion was generated._\n")
    else:
        report.extend([
            f"**Summary:** {suggestion.summary}",
            "",
            f"**Probable Cause:** {suggestion.probable_cause}",
            "",
            f"- **Confidence:** `{suggestion.confidence * 100:.1f}%`",
            f"- **Risk Level:** `{suggestion.risk_level.value}`",
            f"- **Proposed Action:** `{suggestion.proposed_action.value if suggestion.proposed_action else 'None'}`",
            f"- **Cited Historical Incidents:** {', '.join(f'`{r}`' for r in suggestion.referenced_incident_ids) or 'None'}",
            "",
            "### Recommended Remediation Steps:",
        ])
        for idx, step in enumerate(suggestion.suggested_steps, start=1):
            report.append(f"{idx}. {step}")
        report.append("")

    # 4. Audit Trail
    report.extend([
        "## 4. Decision & Audit Trail",
        "",
    ])

    if not audit_records:
        report.append("_No audit records logged yet for this incident._\n")
    else:
        report.append("| Timestamp (UTC) | Trigger | Outcome | Action | Approver |")
        report.append("|---|---|---|---|---|")
        for rec in audit_records:
            ts = rec.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            act = rec.action_taken or "-"
            approver = rec.approver or "system"
            report.append(f"| {ts} | {rec.trigger} | {rec.outcome} | {act} | {approver} |")
        report.append("")

    report.extend([
        "---",
        "_Report compiled automatically by AIOps Assistant._",
    ])

    return "\n".join(report)
