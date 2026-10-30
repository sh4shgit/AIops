"""Shared test fixtures for the AIOps Assistant test suite."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from aiops.api.main import create_app
from aiops.config import Settings, get_settings
from aiops.detection.thresholds import AnomalyBand, AnomalyEvent
from aiops.reasoning.schemas import ActionType, FixSuggestion, RiskLevel
from aiops.retrieval.ingest_incidents import IncidentRecord
from aiops.retrieval.store import RetrievalResult

# Ensure tests run with dry run and test key
os.environ["DRY_RUN"] = "true"
os.environ["API_KEY"] = "test-secret-key"
os.environ["VECTOR_STORE_BACKEND"] = "chromadb"
os.environ["CHROMA_PERSIST_DIR"] = "./data/test_chroma"


@pytest.fixture
def test_settings() -> Settings:
    """Return test settings instance."""
    return get_settings()


@pytest.fixture
def sample_anomaly_event() -> AnomalyEvent:
    """Return a representative AnomalyEvent."""
    return AnomalyEvent(
        window_id="win-1234",
        score=-0.38,
        band=AnomalyBand.HIGH,
        model_version="test-v1",
        service="hdfs-datanode",
        key_templates=["IOException in receiveBlock for block <*>"],
        raw_log="IOException in receiveBlock for block blk_100: Connection reset by peer",
    )


@pytest.fixture
def sample_incident_record() -> IncidentRecord:
    """Return a representative IncidentRecord."""
    return IncidentRecord(
        id="INC-1001",
        title="HDFS Connection Reset",
        symptoms="Connection reset by peer in receiveBlock",
        log_templates=["IOException in receiveBlock for block <*>"],
        root_cause="Ephemeral port exhaustion",
        resolution="Increased net.ipv4.ip_local_port_range and restarted DataNodes",
        services=["hdfs-datanode"],
        status="active",
        resolved_at=datetime(2026, 8, 15, tzinfo=timezone.utc),
    )


@pytest.fixture
def sample_fix_suggestion() -> FixSuggestion:
    """Return a representative FixSuggestion."""
    return FixSuggestion(
        summary="DataNode port exhaustion and connection timeout",
        probable_cause="Exhaustion of socket buffers under burst batch writes",
        suggested_steps=[
            "Increase ephemeral port range",
            "Restart DataNode worker fleet",
        ],
        confidence=0.85,
        risk_level=RiskLevel.MEDIUM,
        referenced_incident_ids=["INC-1001"],
        proposed_action=ActionType.DRAFT_JIRA_TICKET,
    )
