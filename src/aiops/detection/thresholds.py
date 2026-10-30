"""Confidence-band routing for anomaly scores.

Maps raw Isolation Forest scores to discrete severity bands that drive
downstream behaviour (full RAG pipeline, low-priority ticket, or ignore).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

import structlog
from pydantic import BaseModel, Field

from aiops.config import get_settings

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Band enum
# ---------------------------------------------------------------------------


class AnomalyBand(str, Enum):
    """Severity band derived from an anomaly score.

    * **HIGH** — strong anomaly; triggers the full RAG + remediation pipeline.
    * **LOW** — mild anomaly; creates a low-priority incident, skips
      automation.
    * **NORMAL** — within expected distribution; no action needed.
    """

    HIGH = "HIGH"
    LOW = "LOW"
    NORMAL = "NORMAL"


# ---------------------------------------------------------------------------
# Classification helper
# ---------------------------------------------------------------------------


def classify_anomaly(score: float) -> AnomalyBand:
    """Map an Isolation Forest *score* to an ``AnomalyBand``.

    Isolation Forest ``decision_function`` returns **negative** values for
    anomalies and **positive** values for inliers.  The two thresholds
    partition the real line into three regions::

        ← more anomalous ——————————————————————————→ more normal
        ───────────┬───────────────┬────────────────
           HIGH    │      LOW      │     NORMAL
                   │               │
          score ≤ high     high < score ≤ low    score > low

    Thresholds are read from ``Settings.anomaly_high_threshold`` and
    ``Settings.anomaly_low_threshold``.

    Args:
        score: Raw decision-function value (typically in [-0.5, 0.5]).

    Returns:
        The matching ``AnomalyBand``.
    """
    settings = get_settings()
    high_threshold: float = settings.anomaly_high_threshold
    low_threshold: float = settings.anomaly_low_threshold

    if score <= high_threshold:
        band = AnomalyBand.HIGH
    elif score <= low_threshold:
        band = AnomalyBand.LOW
    else:
        band = AnomalyBand.NORMAL

    logger.debug(
        "thresholds.classify",
        score=score,
        band=band.value,
        high_t=high_threshold,
        low_t=low_threshold,
    )
    return band


# ---------------------------------------------------------------------------
# Event model
# ---------------------------------------------------------------------------


class AnomalyEvent(BaseModel):
    """Structured event emitted when an anomaly is detected.

    Carries enough context for downstream consumers (RAG retrieval, Jira
    ticket creation, Slack notification) to act without re-querying the
    detection layer.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    window_id: str = ""
    score: float = 0.0
    band: AnomalyBand = AnomalyBand.NORMAL
    model_version: str = "v1.0.0"
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    service: str = ""
    key_templates: list[str] = Field(default_factory=list)
    raw_log: str = ""

    @property
    def event_id(self) -> str:
        return self.id

    @property
    def anomaly_score(self) -> float:
        return self.score

    @property
    def timestamp(self) -> datetime:
        return self.created_at

    @property
    def source(self) -> str:
        return self.service
