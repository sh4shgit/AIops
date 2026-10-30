"""Request / response models for the AIOps API.

All Pydantic v2 models used by the FastAPI layer live here.
Domain models (``LogEntry``, ``ParsedLog``, ``Incident``) are also defined
here so every part of the codebase can import from a single location.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class LogLevel(str, enum.Enum):
    """Supported log severity levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class IncidentSeverity(str, enum.Enum):
    """Severity classification for incidents."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class IncidentStatus(str, enum.Enum):
    """Lifecycle status of an incident."""

    OPEN = "OPEN"
    INVESTIGATING = "INVESTIGATING"
    ACTION_PROPOSED = "ACTION_PROPOSED"
    ACTION_APPROVED = "ACTION_APPROVED"
    ACTION_REJECTED = "ACTION_REJECTED"
    RESOLVED = "RESOLVED"


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


class LogEntry(BaseModel):
    """Validated incoming log entry (before ID generation)."""

    timestamp: datetime
    service: str = Field(..., min_length=1, max_length=100)
    level: LogLevel
    message: str = Field(..., min_length=1, max_length=10_000)

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v


class ParsedLog(BaseModel):
    """A fully-validated log entry with a deterministic ID, ready for streaming."""

    log_id: str
    timestamp: datetime
    service: str
    level: LogLevel
    message: str


class Incident(BaseModel):
    """Internal incident representation."""

    id: UUID = Field(default_factory=uuid4)
    title: str
    service: str
    severity: IncidentSeverity
    status: IncidentStatus = IncidentStatus.OPEN
    description: str = ""
    proposed_action: str | None = None
    rejection_reason: str | None = None
    related_log_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class LogEntryInput(BaseModel):
    """Raw log entry as received from the caller (pre-validation)."""

    timestamp: datetime
    service: str = Field(..., min_length=1, max_length=100)
    level: LogLevel
    message: str = Field(..., min_length=1, max_length=10_000)

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v


class ApprovalRequest(BaseModel):
    """Body for the reject-incident endpoint."""

    reason: str = Field(..., min_length=1, max_length=2000, description="Reason for action")


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class IngestError(BaseModel):
    """Per-entry error detail returned inside ``IngestResponse``."""

    index: int
    error: str


class IngestResponse(BaseModel):
    """Summary of a log-ingestion batch."""

    accepted: int
    rejected: int
    errors: list[IngestError] = Field(default_factory=list)


class IncidentResponse(BaseModel):
    """Single incident detail."""

    id: UUID
    title: str
    service: str
    severity: IncidentSeverity
    status: IncidentStatus
    description: str = ""
    proposed_action: str | None = None
    rejection_reason: str | None = None
    related_log_ids: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class IncidentListResponse(BaseModel):
    """Paginated list of incidents."""

    incidents: list[IncidentResponse]
    total: int


class HealthResponse(BaseModel):
    """Health-check payload."""

    status: str
    redis_ok: bool
    vector_store_ok: bool
    timestamp: datetime
