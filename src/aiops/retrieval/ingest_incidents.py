"""Incident corpus ingestion utilities.

Provides the canonical :class:`IncidentRecord` model and helpers to load
incidents from JSON files or Python lists into a
:class:`~aiops.retrieval.store.VectorStoreProtocol` implementation.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import structlog
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from aiops.retrieval.store import VectorStoreProtocol

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class IncidentRecord(BaseModel):
    """A historical incident suitable for RAG retrieval."""

    id: str = Field(..., description="Unique incident identifier.")
    title: str = Field(..., description="Short human-readable title.")
    symptoms: str = Field(
        ...,
        description="Description of the symptoms observed during the incident.",
    )
    log_templates: list[str] = Field(
        default_factory=list,
        description="Log-template strings associated with this incident.",
    )
    root_cause: str = Field(
        ...,
        description="Root-cause analysis text.",
    )
    resolution: str = Field(
        ...,
        description="Steps taken or recommended to resolve the incident.",
    )
    services: list[str] = Field(
        default_factory=list,
        description="Services affected by or involved in the incident.",
    )
    status: Literal["active", "superseded"] = Field(
        "active",
        description="Whether this record is current or has been superseded.",
    )
    resolved_at: datetime | None = Field(
        None,
        description="UTC timestamp when the incident was resolved (if applicable).",
    )


# ---------------------------------------------------------------------------
# Ingestion helpers
# ---------------------------------------------------------------------------

def ingest_from_file(path: Path, store: VectorStoreProtocol) -> None:
    """Load incidents from a JSON file and ingest them into *store*.

    The JSON file must contain either a single JSON array of incident objects
    or an object with an ``"incidents"`` key mapping to such an array.

    Parameters
    ----------
    path:
        Filesystem path to the JSON incidents file.
    store:
        Vector-store backend to add the incidents to.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the JSON structure is not recognized.
    """
    log.info("ingest.file.start", path=str(path))
    if not path.exists():
        raise FileNotFoundError(f"Incident file not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(raw, list):
        records_raw = raw
    elif isinstance(raw, dict) and "incidents" in raw:
        records_raw = raw["incidents"]
    else:
        raise ValueError(
            "JSON file must be a list of incidents or an object with an 'incidents' key."
        )

    incidents: list[IncidentRecord] = []
    for idx, item in enumerate(records_raw):
        try:
            incidents.append(IncidentRecord.model_validate(item))
        except Exception:
            log.warning("ingest.file.skip_invalid", index=idx, exc_info=True)

    ingest_from_list(incidents, store)
    log.info("ingest.file.done", path=str(path), count=len(incidents))


def ingest_from_list(
    incidents: list[IncidentRecord],
    store: VectorStoreProtocol,
) -> None:
    """Validate and add a list of :class:`IncidentRecord` objects to *store*.

    Each incident is re-validated before insertion to guarantee data integrity.

    Parameters
    ----------
    incidents:
        Pre-constructed incident records.
    store:
        Vector-store backend to add the incidents to.
    """
    validated: list[IncidentRecord] = []
    for incident in incidents:
        try:
            # Re-validate to ensure consistency even if caller built models
            # manually without full validation.
            validated.append(IncidentRecord.model_validate(incident.model_dump()))
        except Exception:
            log.warning(
                "ingest.list.skip_invalid",
                incident_id=getattr(incident, "id", "unknown"),
                exc_info=True,
            )

    if not validated:
        log.warning("ingest.list.empty", message="No valid incidents to ingest.")
        return

    log.info("ingest.list.adding", count=len(validated))
    store.add_incidents(validated)
    log.info("ingest.list.done", count=len(validated))
