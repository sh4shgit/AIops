"""LangGraph state machine for the AIOps reasoning-to-action pipeline.

The graph implements the following linear pipeline with an approval
interrupt before executing side-effecting actions:

    detect → retrieve → reason → validate → propose → [approval] → act → record

Each node is an async function that reads from and writes to a shared
``PipelineState`` TypedDict.  On any unhandled exception the graph routes
to an *error* node which captures the failure in an audit record.
"""

from __future__ import annotations

import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, TypedDict

import structlog
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from aiops.actions.allowlist import is_side_effect, validate_action
from aiops.actions.audit import AuditLog, AuditRecord
from aiops.detection.thresholds import AnomalyEvent
from aiops.reasoning.llm import LLMClient
from aiops.reasoning.prompts import build_analysis_prompt
from aiops.reasoning.schemas import ActionType, FixSuggestion
from aiops.reasoning.scrubber import Scrubber
from aiops.retrieval.store import RetrievalResult

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------


class PipelineState(TypedDict, total=False):
    """Mutable state bag passed through every graph node."""

    anomaly_event: AnomalyEvent
    retrieved_incidents: list[RetrievalResult]
    scrubbed_context: str
    suggestion: FixSuggestion | None
    validation_result: str
    action_proposal: dict[str, Any]
    approval_status: str  # "pending" | "approved" | "rejected"
    audit_record: AuditRecord | None
    error: str | None


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

_scrubber = Scrubber()


async def detect_node(state: PipelineState) -> PipelineState:
    """Passthrough — the anomaly event is already injected by the caller.

    In a production system this node would invoke the scoring pipeline.
    """
    log.info("pipeline_detect", event_id=state["anomaly_event"].event_id)
    return state


async def retrieve_node(state: PipelineState) -> PipelineState:
    """Passthrough — retrieved incidents are pre-populated by the caller.

    A production implementation would call the vector store here.
    """
    ids = [r.incident_id for r in state.get("retrieved_incidents", [])]
    log.info("pipeline_retrieve", retrieved_ids=ids)
    return state


async def reason_node(state: PipelineState) -> PipelineState:
    """Call the LLM to produce a ``FixSuggestion``."""
    anomaly = state["anomaly_event"]
    incidents = state.get("retrieved_incidents", [])

    # Scrub raw log and build prompt
    scrubbed_log = _scrubber.scrub(anomaly.raw_log)
    scrubbed_context = state.get("scrubbed_context", "")
    if not scrubbed_context:
        scrubbed_context = _scrubber.scrub(scrubbed_log)

    prompt = build_analysis_prompt(anomaly, incidents, scrubbed_context)

    client = LLMClient()
    suggestion = await client.analyze(prompt)

    state["suggestion"] = suggestion
    state["scrubbed_context"] = scrubbed_context
    log.info("pipeline_reason", summary=suggestion.summary)
    return state


async def validate_node(state: PipelineState) -> PipelineState:
    """Validate LLM references and action preconditions."""
    suggestion = state.get("suggestion")
    if suggestion is None:
        state["validation_result"] = "skipped:no_suggestion"
        return state

    # Strip hallucinated references
    retrieved_ids = {r.incident_id for r in state.get("retrieved_incidents", [])}
    client = LLMClient()
    suggestion = client.validate_references(suggestion, retrieved_ids)
    state["suggestion"] = suggestion

    # Validate proposed action
    action = suggestion.proposed_action
    if action and action != ActionType.NO_ACTION:
        allowed, reason = validate_action(
            action,
            {"anomaly_event": state["anomaly_event"].model_dump()},
        )
        if not allowed:
            log.warning("action_not_allowed", action=action.value, reason=reason)
            suggestion = suggestion.model_copy(update={"proposed_action": ActionType.NO_ACTION})
            state["suggestion"] = suggestion
            state["validation_result"] = f"action_blocked:{reason}"
            return state

    state["validation_result"] = "passed"
    log.info("pipeline_validate", result=state["validation_result"])
    return state


async def propose_node(state: PipelineState) -> PipelineState:
    """Build an action proposal dict for approval."""
    suggestion = state.get("suggestion")
    if suggestion is None or suggestion.proposed_action in (None, ActionType.NO_ACTION):
        state["action_proposal"] = {}
        state["approval_status"] = "not_required"
        return state

    proposal: dict[str, Any] = {
        "action": suggestion.proposed_action.value if suggestion.proposed_action else None,
        "summary": suggestion.summary,
        "risk_level": suggestion.risk_level.value,
        "confidence": suggestion.confidence,
    }
    state["action_proposal"] = proposal

    if is_side_effect(suggestion.proposed_action):
        state["approval_status"] = "pending"
    else:
        state["approval_status"] = "auto_approved"

    log.info("pipeline_propose", proposal=proposal, status=state["approval_status"])
    return state


async def approval_node(state: PipelineState) -> PipelineState:
    """Human-in-the-loop interrupt for side-effecting actions.

    Uses LangGraph's :func:`interrupt` to pause execution until a human
    approves or rejects the proposal.
    """
    if state.get("approval_status") != "pending":
        return state

    log.info("pipeline_approval_wait", proposal=state.get("action_proposal"))

    # This raises an interrupt that pauses the graph
    decision = interrupt(
        {
            "question": "Approve the proposed action?",
            "proposal": state.get("action_proposal", {}),
        }
    )

    state["approval_status"] = "approved" if decision else "rejected"
    log.info("pipeline_approval_decision", status=state["approval_status"])
    return state


async def act_node(state: PipelineState) -> PipelineState:
    """Execute the approved action (or skip if rejected)."""
    status = state.get("approval_status", "not_required")
    if status not in ("approved", "auto_approved", "not_required"):
        log.info("pipeline_act_skipped", reason=status)
        return state

    suggestion = state.get("suggestion")
    if suggestion is None or suggestion.proposed_action in (None, ActionType.NO_ACTION):
        return state

    action = suggestion.proposed_action
    log.info("pipeline_act", action=action.value if action else None)

    # Actual action dispatch is handled by the actions layer.
    # This node is intentionally kept thin — orchestration code
    # lives in the caller / API layer to keep the graph testable.
    state["action_proposal"]["executed"] = True
    return state


async def record_node(state: PipelineState) -> PipelineState:
    """Create an audit record capturing the full pipeline run."""
    anomaly = state["anomaly_event"]
    suggestion = state.get("suggestion")

    record = AuditRecord(
        id=uuid.uuid4(),
        timestamp=datetime.now(timezone.utc),
        trigger="anomaly_pipeline",
        anomaly_score=anomaly.anomaly_score,
        anomaly_band=anomaly.band.value,
        retrieved_incident_ids=[
            r.incident_id for r in state.get("retrieved_incidents", [])
        ],
        model_output=suggestion.model_dump() if suggestion else {},
        validation_result=state.get("validation_result", "unknown"),
        action_taken=(
            suggestion.proposed_action.value
            if suggestion and suggestion.proposed_action
            else None
        ),
        approver=None,
        outcome=state.get("approval_status", "not_required"),
        incident_id=anomaly.event_id,
    )

    audit_log = AuditLog()
    audit_log.record(record)

    state["audit_record"] = record
    log.info("pipeline_record", audit_id=str(record.id))
    return state


async def error_node(state: PipelineState) -> PipelineState:
    """Capture failures as an audit record with outcome ``error``."""
    anomaly = state.get("anomaly_event")
    error_msg = state.get("error", "unknown_error")

    record = AuditRecord(
        id=uuid.uuid4(),
        timestamp=datetime.now(timezone.utc),
        trigger="anomaly_pipeline",
        anomaly_score=anomaly.anomaly_score if anomaly else 0.0,
        anomaly_band=anomaly.band.value if anomaly else "UNKNOWN",
        retrieved_incident_ids=[
            r.incident_id for r in state.get("retrieved_incidents", [])
        ],
        model_output={"error": error_msg},
        validation_result="error",
        action_taken=None,
        approver=None,
        outcome=f"error:{error_msg}",
        incident_id=anomaly.event_id if anomaly else None,
    )

    audit_log = AuditLog()
    audit_log.record(record)

    state["audit_record"] = record
    log.error("pipeline_error", error=error_msg)
    return state


# ---------------------------------------------------------------------------
# Safe wrapper — catches exceptions and routes to error_node
# ---------------------------------------------------------------------------


def _make_safe(
    node_fn: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Wrap an async node so that exceptions are captured into state."""

    async def _safe(state: PipelineState) -> PipelineState:
        try:
            return await node_fn(state)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "node_exception",
                node=node_fn.__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            state["error"] = f"{node_fn.__name__}: {exc}"
            return state

    _safe.__name__ = node_fn.__name__
    return _safe


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _route_after_node(state: PipelineState) -> str:
    """Route to error node if an error was captured, else continue."""
    if state.get("error"):
        return "error"
    return "continue"


def build_pipeline() -> StateGraph:
    """Construct and compile the AIOps reasoning pipeline graph.

    Returns
    -------
    StateGraph
        The compiled LangGraph state machine, ready for ``.invoke()`` or
        ``.astream()``.
    """
    graph = StateGraph(PipelineState)

    # Register nodes
    graph.add_node("detect", _make_safe(detect_node))
    graph.add_node("retrieve", _make_safe(retrieve_node))
    graph.add_node("reason", _make_safe(reason_node))
    graph.add_node("validate", _make_safe(validate_node))
    graph.add_node("propose", _make_safe(propose_node))
    graph.add_node("approval", approval_node)  # NOT wrapped — interrupt must propagate
    graph.add_node("act", _make_safe(act_node))
    graph.add_node("record", _make_safe(record_node))
    graph.add_node("error", error_node)

    # Edges with error routing
    graph.set_entry_point("detect")

    for src, dst in [
        ("detect", "retrieve"),
        ("retrieve", "reason"),
        ("reason", "validate"),
        ("validate", "propose"),
    ]:
        graph.add_conditional_edges(
            src,
            _route_after_node,
            {"continue": dst, "error": "error"},
        )

    graph.add_edge("propose", "approval")
    graph.add_conditional_edges(
        "approval",
        _route_after_node,
        {"continue": "act", "error": "error"},
    )
    graph.add_conditional_edges(
        "act",
        _route_after_node,
        {"continue": "record", "error": "error"},
    )

    graph.add_edge("record", END)
    graph.add_edge("error", END)

    return graph


def compile_pipeline(checkpointer: Any = None) -> Any:
    """Compile the pipeline graph into an executable runnable."""
    return build_pipeline().compile(checkpointer=checkpointer)
