"""Per-entry validation for incoming log data.

Each dict is validated independently so that one bad entry never blocks an
entire batch.
"""

from __future__ import annotations

import hashlib
from typing import Any

import structlog
from pydantic import ValidationError

from aiops.api.schemas import IngestError, LogEntry, ParsedLog

logger = structlog.get_logger()


def _deterministic_id(timestamp: str, service: str, message: str) -> str:
    """Generate a deterministic SHA-256 hex ID from key fields."""
    payload = f"{timestamp}|{service}|{message}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_entry(
    data: dict[str, Any],
    index: int,
) -> tuple[ParsedLog | None, IngestError | None]:
    """Validate a single raw log dict and produce a ``ParsedLog`` or an error.

    Parameters
    ----------
    data:
        Raw dict as received from the HTTP body.
    index:
        Position inside the batch (used in error reporting).

    Returns
    -------
    tuple
        ``(ParsedLog, None)`` on success; ``(None, IngestError)`` on failure.
    """
    try:
        entry = LogEntry.model_validate(data)
    except ValidationError as exc:
        first_error = exc.errors()[0]
        msg = f"{first_error['loc']}: {first_error['msg']}"
        logger.warning("log_entry_validation_failed", index=index, error=msg)
        return None, IngestError(index=index, error=msg)
    except Exception as exc:  # noqa: BLE001
        logger.warning("log_entry_validation_unexpected", index=index, error=str(exc))
        return None, IngestError(index=index, error=str(exc))

    log_id = _deterministic_id(
        timestamp=entry.timestamp.isoformat(),
        service=entry.service,
        message=entry.message,
    )

    parsed = ParsedLog(
        log_id=log_id,
        timestamp=entry.timestamp,
        service=entry.service,
        level=entry.level,
        message=entry.message,
    )
    return parsed, None
