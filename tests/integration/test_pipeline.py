"""Integration tests for the LangGraph reasoning-to-action state machine."""

from __future__ import annotations

import pytest

from aiops.actions.audit import AuditLog
from aiops.detection.thresholds import AnomalyBand, AnomalyEvent
from aiops.reasoning.graph import build_pipeline, compile_pipeline
from aiops.reasoning.schemas import ActionType, FixSuggestion
from aiops.retrieval.store import RetrievalResult


@pytest.mark.asyncio
async def test_full_pipeline_execution(tmp_path) -> None:
    # Set up test audit log path
    audit_file = tmp_path / "test_audit.jsonl"
    audit_log = AuditLog(log_path=audit_file)

    anomaly = AnomalyEvent(
        window_id="win-999",
        score=-0.42,
        band=AnomalyBand.HIGH,
        model_version="test-v1",
        service="hdfs-datanode",
        key_templates=["IOException in receiveBlock for block <*>"],
        raw_log="IOException in receiveBlock for block blk_99: Connection reset by peer from 10.0.0.1",
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

    # Execute graph
    final_state = await pipeline.ainvoke(initial_state)

    # Assertions
    assert "suggestion" in final_state
    suggestion = final_state["suggestion"]
    assert isinstance(suggestion, FixSuggestion)
    assert suggestion.confidence > 0.0
    assert len(suggestion.suggested_steps) > 0

    assert "action_proposal" in final_state
    proposal = final_state["action_proposal"]
    assert "summary" in proposal

    # Audit record must have been created and persisted
    assert "audit_record" in final_state
    record = final_state["audit_record"]
    assert record.trigger == "anomaly_pipeline"
    assert record.anomaly_score == -0.42
    assert "INC-1001" in record.retrieved_incident_ids
