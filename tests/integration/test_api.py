"""Integration tests for the FastAPI HTTP endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aiops.api.main import create_app
from aiops.api.schemas import Incident, IncidentSeverity, IncidentStatus


class MockPipeline:
    def __init__(self, redis_instance: MockRedis):
        self.redis = redis_instance
        self.ops = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def sadd(self, key: str, value: str):
        self.ops.append(("sadd", key, value))

    def xadd(self, stream: str, fields: dict):
        self.ops.append(("xadd", stream, fields))

    async def execute(self):
        results = []
        for op in self.ops:
            if op[0] == "sadd":
                _, key, val = op
                if key not in self.redis.sets:
                    self.redis.sets[key] = set()
                if val in self.redis.sets[key]:
                    results.append(0)  # already member
                else:
                    self.redis.sets[key].add(val)
                    results.append(1)  # added
            elif op[0] == "xadd":
                results.append("1000-0")
        return results


class MockRedis:
    def __init__(self):
        self.sets = {}

    async def ping(self):
        return True

    def pipeline(self, transaction: bool = False):
        return MockPipeline(self)

    async def expire(self, key: str, ttl: int):
        return True

    async def aclose(self):
        pass


@pytest.fixture
def test_client() -> TestClient:
    """Create a FastAPI TestClient with a simulated mock Redis backend."""
    app = create_app()
    app.state.redis = MockRedis()

    with TestClient(app) as client:
        yield client


def test_health_endpoint(test_client: TestClient) -> None:
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "redis_ok" in data
    assert "vector_store_ok" in data


def test_metrics_endpoint(test_client: TestClient) -> None:
    response = test_client.get("/metrics")
    assert response.status_code == 200
    assert "aiops_logs_ingested_total" in response.text


def test_ingest_empty_batch_rejected(test_client: TestClient) -> None:
    response = test_client.post(
        "/logs",
        json=[],
        headers={"X-API-Key": "test-secret-key"},
    )
    assert response.status_code == 422


def test_ingest_requires_api_key(test_client: TestClient) -> None:
    batch = [
        {
            "timestamp": "2026-10-28T12:00:00Z",
            "service": "api-gateway",
            "level": "INFO",
            "message": "Valid log entry",
        }
    ]
    # No header
    response = test_client.post("/logs", json=batch)
    assert response.status_code == 403

    # Wrong header
    response = test_client.post(
        "/logs",
        json=batch,
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 403


def test_ingest_partial_validation_and_deduplication(test_client: TestClient) -> None:
    batch = [
        # Valid entry 1
        {
            "timestamp": "2026-10-28T12:00:00Z",
            "service": "hdfs-datanode",
            "level": "INFO",
            "message": "Receiving block blk_100",
        },
        # Invalid entry (blank message)
        {
            "timestamp": "2026-10-28T12:00:01Z",
            "service": "hdfs-datanode",
            "level": "INFO",
            "message": "   ",
        },
        # Valid entry 2
        {
            "timestamp": "2026-10-28T12:00:02Z",
            "service": "hdfs-datanode",
            "level": "ERROR",
            "message": "Connection reset by peer",
        },
    ]

    response = test_client.post(
        "/logs",
        json=batch,
        headers={"X-API-Key": "test-secret-key"},
    )
    assert response.status_code == 202
    data = response.json()
    assert data["accepted"] == 2
    assert data["rejected"] == 1
    assert len(data["errors"]) == 1
    assert data["errors"][0]["index"] == 1

    # Sending the exact same batch again should deduplicate both valid entries
    resend = test_client.post(
        "/logs",
        json=batch,
        headers={"X-API-Key": "test-secret-key"},
    )
    assert resend.status_code == 202
    resend_data = resend.json()
    assert resend_data["accepted"] == 0  # 0 new accepted, 2 duplicates skipped
    assert resend_data["rejected"] == 1


def test_incident_lifecycle_approve_reject(test_client: TestClient) -> None:
    # Seed an incident in state ACTION_PROPOSED
    app = test_client.app
    store = app.state.incident_store

    import asyncio
    inc = Incident(
        title="High error rate on auth service",
        service="auth-service",
        severity=IncidentSeverity.HIGH,
        status=IncidentStatus.ACTION_PROPOSED,
        proposed_action="DRAFT_JIRA_TICKET",
    )
    asyncio.run(store.add_incident(inc))

    # 1. List incidents
    list_res = test_client.get("/incidents")
    assert list_res.status_code == 200
    assert list_res.json()["total"] >= 1

    # 2. Get detail
    detail_res = test_client.get(f"/incidents/{inc.id}")
    assert detail_res.status_code == 200
    assert detail_res.json()["status"] == "ACTION_PROPOSED"

    # 3. Approve without key fails
    bad_approve = test_client.post(f"/incidents/{inc.id}/approve")
    assert bad_approve.status_code == 403

    # 4. Approve with key succeeds
    approve_res = test_client.post(
        f"/incidents/{inc.id}/approve",
        headers={"X-API-Key": "test-secret-key"},
    )
    assert approve_res.status_code == 200
    assert approve_res.json()["status"] == "ACTION_APPROVED"
