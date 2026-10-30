"""Append-only audit log for tracking all automated decisions and actions.

Records triggers, anomaly scores, model suggestions, validation outcomes,
approval states, and final actions taken.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()

DEFAULT_AUDIT_LOG_PATH = Path("./data/audit.jsonl")


class AuditRecord(BaseModel):
    """An immutable record of a pipeline execution step or decision."""

    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    trigger: str = Field(..., description="Pipeline trigger (e.g. anomaly_pipeline, manual_review)")
    anomaly_score: float = Field(default=0.0)
    anomaly_band: str = Field(default="UNKNOWN")
    retrieved_incident_ids: list[str] = Field(default_factory=list)
    model_output: dict[str, Any] = Field(default_factory=dict)
    validation_result: str = Field(default="pending")
    action_taken: str | None = None
    approver: str | None = None
    outcome: str = Field(default="initiated")
    incident_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class AuditLog:
    """Thread-safe, append-only logger for AuditRecords."""

    def __init__(self, log_path: Path | str = DEFAULT_AUDIT_LOG_PATH) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, record: AuditRecord) -> None:
        """Append an audit record to the log file."""
        try:
            line = record.model_dump_json()
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            logger.info("audit_recorded", audit_id=str(record.id), trigger=record.trigger)
        except Exception as e:
            logger.error("audit_write_failed", error=str(e), audit_id=str(record.id))

    def get_records(self, incident_id: str | None = None, limit: int = 100) -> list[AuditRecord]:
        """Read recorded audit events, optionally filtered by incident_id."""
        if not self.log_path.exists():
            return []

        records: list[AuditRecord] = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = AuditRecord.model_validate_json(line)
                        if incident_id is None or rec.incident_id == incident_id:
                            records.append(rec)
                    except Exception:
                        continue
            # Return most recent first
            records.reverse()
            return records[:limit]
        except Exception as e:
            logger.error("audit_read_failed", error=str(e))
            return []
