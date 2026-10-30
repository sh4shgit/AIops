"""Incident management endpoints.

Provides CRUD-like access to incidents with an in-memory store that
satisfies an ``IncidentStore`` protocol so it can be swapped out later.
"""

from __future__ import annotations

import abc
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from aiops.api.schemas import (
    ApprovalRequest,
    Incident,
    IncidentListResponse,
    IncidentResponse,
    IncidentSeverity,
    IncidentStatus,
)
from aiops.observability.metrics import INCIDENT_APPROVALS_TOTAL

logger = structlog.get_logger()

# Two routers: read (open) and write (will be API-key protected by main.py).
read_router = APIRouter(prefix="/incidents", tags=["incidents"])
write_router = APIRouter(prefix="/incidents", tags=["incidents"])


# ---------------------------------------------------------------------------
# Store protocol / ABC
# ---------------------------------------------------------------------------


class IncidentStore(abc.ABC):
    """Abstract interface for incident persistence."""

    @abc.abstractmethod
    async def list_incidents(
        self,
        status: IncidentStatus | None = None,
        service: str | None = None,
        severity: IncidentSeverity | None = None,
    ) -> list[Incident]:
        ...

    @abc.abstractmethod
    async def get_incident(self, incident_id: UUID) -> Incident | None:
        ...

    @abc.abstractmethod
    async def update_incident(self, incident: Incident) -> Incident:
        ...

    @abc.abstractmethod
    async def add_incident(self, incident: Incident) -> Incident:
        ...


class InMemoryIncidentStore(IncidentStore):
    """Thread-safe (async-safe) in-memory dict-based store for development."""

    def __init__(self) -> None:
        self._store: dict[UUID, Incident] = {}

    async def list_incidents(
        self,
        status: IncidentStatus | None = None,
        service: str | None = None,
        severity: IncidentSeverity | None = None,
    ) -> list[Incident]:
        results = list(self._store.values())
        if status is not None:
            results = [i for i in results if i.status == status]
        if service is not None:
            results = [i for i in results if i.service == service]
        if severity is not None:
            results = [i for i in results if i.severity == severity]
        return results

    async def get_incident(self, incident_id: UUID) -> Incident | None:
        return self._store.get(incident_id)

    async def update_incident(self, incident: Incident) -> Incident:
        incident.updated_at = datetime.now(timezone.utc)
        self._store[incident.id] = incident
        return incident

    async def add_incident(self, incident: Incident) -> Incident:
        self._store[incident.id] = incident
        return incident


# ---------------------------------------------------------------------------
# Singleton store (attached to app.state in lifespan)
# ---------------------------------------------------------------------------

_default_store = InMemoryIncidentStore()


def _get_store(request: Request) -> IncidentStore:
    """Retrieve the incident store from app state, falling back to the default."""
    return getattr(request.app.state, "incident_store", _default_store)


def _incident_to_response(incident: Incident) -> IncidentResponse:
    return IncidentResponse(**incident.model_dump())


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


@read_router.get("", response_model=IncidentListResponse)
async def list_incidents(
    status: IncidentStatus | None = Query(None),
    service: str | None = Query(None),
    severity: IncidentSeverity | None = Query(None),
    store: IncidentStore = Depends(_get_store),
) -> IncidentListResponse:
    """List incidents with optional filters."""
    incidents = await store.list_incidents(status=status, service=service, severity=severity)
    return IncidentListResponse(
        incidents=[_incident_to_response(i) for i in incidents],
        total=len(incidents),
    )


@read_router.get("/{incident_id}", response_model=IncidentResponse)
async def get_incident(
    incident_id: UUID,
    store: IncidentStore = Depends(_get_store),
) -> IncidentResponse:
    """Retrieve full incident detail."""
    incident = await store.get_incident(incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    return _incident_to_response(incident)


@write_router.post("/{incident_id}/approve", response_model=IncidentResponse)
async def approve_incident(
    incident_id: UUID,
    store: IncidentStore = Depends(_get_store),
) -> IncidentResponse:
    """Approve the proposed remediation action for an incident."""
    incident = await store.get_incident(incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    if incident.status != IncidentStatus.ACTION_PROPOSED:
        raise HTTPException(
            status_code=409,
            detail=f"Incident status is '{incident.status.value}', expected ACTION_PROPOSED",
        )

    incident.status = IncidentStatus.ACTION_APPROVED
    updated = await store.update_incident(incident)
    action_name = incident.proposed_action or "UNKNOWN"
    INCIDENT_APPROVALS_TOTAL.labels(action=action_name, decision="approved").inc()
    logger.info("incident_approved", incident_id=str(incident_id))
    return _incident_to_response(updated)


@write_router.post("/{incident_id}/reject", response_model=IncidentResponse)
async def reject_incident(
    incident_id: UUID,
    body: ApprovalRequest,
    store: IncidentStore = Depends(_get_store),
) -> IncidentResponse:
    """Reject the proposed remediation action for an incident."""
    incident = await store.get_incident(incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    if incident.status != IncidentStatus.ACTION_PROPOSED:
        raise HTTPException(
            status_code=409,
            detail=f"Incident status is '{incident.status.value}', expected ACTION_PROPOSED",
        )

    incident.status = IncidentStatus.ACTION_REJECTED
    incident.rejection_reason = body.reason
    updated = await store.update_incident(incident)
    action_name = incident.proposed_action or "UNKNOWN"
    INCIDENT_APPROVALS_TOTAL.labels(action=action_name, decision="rejected").inc()
    logger.info("incident_rejected", incident_id=str(incident_id), reason=body.reason)
    return _incident_to_response(updated)
