"""Prompt templates for the Claude-based SRE reasoning engine.

All templates instruct the model to behave as a conservative SRE
assistant that:

* analyses anomalies **only** using retrieved precedent,
* cites **only** incident IDs present in the context,
* limits proposed actions to the allowlisted set, and
* is deliberately cautious in its risk assessments.
"""

from __future__ import annotations

from aiops.detection.thresholds import AnomalyEvent
from aiops.retrieval.store import RetrievalResult

# ---------------------------------------------------------------------------
# System prompt — injected as the ``system`` role in every Claude call
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = """\
You are an expert Site Reliability Engineering (SRE) assistant integrated into
an AIOps platform.  Your role is to analyse detected anomalies, correlate them
with historically similar incidents, and produce **structured** fix suggestions.

## Hard constraints
1. **Only reference incident IDs that appear in the RETRIEVED INCIDENTS section.**
   Never fabricate or guess incident identifiers.
2. **Only propose actions from the allowlist:**
   DRAFT_JIRA_TICKET | SEND_SLACK_ALERT | RESTART_SERVICE | NO_ACTION
3. **Be conservative** — when uncertain, prefer NO_ACTION and set risk_level to
   HIGH.  Human approval is always required for RESTART_SERVICE.
4. **Cite evidence** — every claim in ``probable_cause`` must trace to a
   specific retrieved incident or to data in the anomaly event.
5. Respond **only** with valid JSON matching the FixSuggestion schema.
   Do not include any text outside the JSON block.

## FixSuggestion JSON schema
```json
{
  "summary": "<one-line summary>",
  "probable_cause": "<detailed root-cause hypothesis>",
  "suggested_steps": ["<step 1>", "<step 2>", "..."],
  "confidence": <float 0-1>,
  "risk_level": "LOW | MEDIUM | HIGH",
  "referenced_incident_ids": ["<id>", ...],
  "proposed_action": "DRAFT_JIRA_TICKET | SEND_SLACK_ALERT | RESTART_SERVICE | NO_ACTION | null"
}
```
"""

# ---------------------------------------------------------------------------
# Analysis prompt builder
# ---------------------------------------------------------------------------


def build_analysis_prompt(
    anomaly_event: AnomalyEvent,
    retrieved_incidents: list[RetrievalResult],
    scrubbed_context: str,
) -> str:
    """Format all available context into a single user prompt for Claude.

    Parameters
    ----------
    anomaly_event:
        The anomaly detected by the scoring pipeline.
    retrieved_incidents:
        Top-K similar historical incidents from the vector store.
    scrubbed_context:
        Additional operational context that has already been scrubbed of
        PII/secrets.

    Returns
    -------
    str
        A fully-assembled user-role prompt ready for the LLM.
    """
    # -- anomaly section ----------------------------------------------------
    anomaly_section = (
        "## ANOMALY EVENT\n"
        f"- **Event ID:** {anomaly_event.event_id}\n"
        f"- **Timestamp:** {anomaly_event.timestamp.isoformat()}\n"
        f"- **Source:** {anomaly_event.source}\n"
        f"- **Score:** {anomaly_event.anomaly_score:.4f}\n"
        f"- **Band:** {anomaly_event.band.value}\n"
        f"- **Raw log:** {anomaly_event.raw_log}\n"
    )

    # -- retrieved incidents section ----------------------------------------
    incident_lines: list[str] = []
    for idx, inc in enumerate(retrieved_incidents, start=1):
        incident_lines.append(
            f"### Incident {idx}: {inc.incident_id}\n"
            f"- **Title:** {inc.title}\n"
            f"- **Similarity score:** {inc.score:.4f}\n"
            f"- **Content:**\n{inc.content}\n"
        )
    incidents_section = "## RETRIEVED INCIDENTS\n" + (
        "\n".join(incident_lines) if incident_lines else "_No similar incidents found._\n"
    )

    # -- additional context -------------------------------------------------
    context_section = "## ADDITIONAL CONTEXT\n" + (scrubbed_context or "_None._") + "\n"

    # -- instructions -------------------------------------------------------
    instructions = (
        "## INSTRUCTIONS\n"
        "Analyse the anomaly event above using **only** the retrieved incidents "
        "as evidence.  Produce a single JSON object conforming to the "
        "FixSuggestion schema described in your system prompt.  Do **not** "
        "include markdown fences or any text outside the JSON.\n"
    )

    return "\n".join([anomaly_section, incidents_section, context_section, instructions])
