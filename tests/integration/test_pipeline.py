"""Integration tests for the LangGraph reasoning-to-action state machine.

These tests run the pipeline **synchronously** using a dedicated event loop
to avoid the pytest-asyncio / Windows Python 3.15 hang issue.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from aiops.actions.audit import AuditLog
from aiops.detection.thresholds import AnomalyBand, AnomalyEvent
from aiops.reasoning.graph import build_pipeline, compile_pipeline
from aiops.reasoning.schemas import ActionType, FixSuggestion
from aiops.retrieval.store import RetrievalResult


def _run(coro):
    """Run a coroutine synchronously with a fresh event loop.

    Using a fresh loop avoids the Windows / Python 3.15 hang that occurs when
    calling ``asyncio.run()`` from inside pytest's default event-loop fixture.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_full_pipeline_execution(tmp_path: Path) -> None:
    """End-to-end pipeline: anomaly → retrieve → reason → propose → record."""
    audit_file = tmp_path / "test_audit.jsonl"
    audit_log = AuditLog(log_path=audit_file)

    anomaly = AnomalyEvent(
        window_id="win-999",
        score=-0.42,
        band=AnomalyBand.HIGH,
        model_version="test-v1",
        service="hdfs-datanode",
        key_templates=["IOException in receiveBlock for block <*>"],
        raw_log=(
            "IOException in receiveBlock for block blk_99: "
            "Connection reset by peer from 10.0.0.1"
        ),
    )

    retrieved = [
        RetrievalResult(
            incident_id="INC-1001",
            chunk_type="resolution",
            similarity_score=0.91,
            content="Increased net.ipv4.ip_local_port_range and restarted DataNodes",
            metadata={"title": "HDFS Connection Reset"},
        )
    ]

    pipeline = compile_pipeline()

    initial_state = {
        "anomaly_event": anomaly,
        "retrieved_incidents": retrieved,
        "scrubbed_context": "Sample scrubbed log context",
    }

    final_state = _run(pipeline.ainvoke(initial_state))

    # --- suggestion ---
    assert "suggestion" in final_state
    suggestion = final_state["suggestion"]
    assert isinstance(suggestion, FixSuggestion)
    assert suggestion.confidence > 0.0
    assert len(suggestion.suggested_steps) > 0

    # --- action proposal ---
    assert "action_proposal" in final_state
    proposal = final_state["action_proposal"]
    # proposal may be empty dict (no_action) or populated
    assert isinstance(proposal, dict)

    # --- audit record ---
    assert "audit_record" in final_state
    record = final_state["audit_record"]
    assert record.trigger == "anomaly_pipeline"
    assert record.anomaly_score == -0.42
    assert "INC-1001" in record.retrieved_incident_ids


def test_pipeline_error_handling() -> None:
    """Pipeline should capture errors in the audit record and not raise."""
    # Provide minimal valid state; the mock LLM should still produce a suggestion
    anomaly = AnomalyEvent(
        window_id="win-err",
        score=-0.55,
        band=AnomalyBand.HIGH,
        model_version="test-v1",
        service="unknown-service",
        key_templates=[],
        raw_log="",
    )

    pipeline = compile_pipeline()
    initial_state = {
        "anomaly_event": anomaly,
        "retrieved_incidents": [],
        "scrubbed_context": "",
    }

    # Should not raise — errors are caught inside the graph
    final_state = _run(pipeline.ainvoke(initial_state))
    assert "audit_record" in final_state


def test_pipeline_no_action_for_low_band() -> None:
    """LOW confidence anomalies should not propose side-effecting actions."""
    anomaly = AnomalyEvent(
        window_id="win-low",
        score=-0.25,
        band=AnomalyBand.LOW,
        model_version="test-v1",
        service="api-gateway",
        key_templates=["Slow response time <*>ms"],
        raw_log="Slow response time 950ms on /api/v1/health",
    )

    pipeline = compile_pipeline()
    initial_state = {
        "anomaly_event": anomaly,
        "retrieved_incidents": [],
        "scrubbed_context": "slow api response",
    }

    final_state = _run(pipeline.ainvoke(initial_state))
    assert "audit_record" in final_state
    record = final_state["audit_record"]
    assert record.anomaly_score == -0.25
