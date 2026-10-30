"""Feature extraction from parsed log streams.

Accumulates ``ParsedTemplate`` entries within configurable time windows and
produces numeric feature vectors suitable for scikit-learn models (e.g.
Isolation Forest).
"""

from __future__ import annotations

import re
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import structlog
from pydantic import BaseModel, Field

from aiops.config import get_settings
from aiops.detection.parser import ParsedTemplate

logger = structlog.get_logger()

# HDFS-style block ID pattern: ``blk_<digits>`` or ``blk_-<digits>``.
_BLOCK_ID_RE = re.compile(r"blk_-?\d+")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class FeatureWindow(BaseModel):
    """Aggregated feature snapshot for a single time window.

    Each window captures template occurrence counts and simple severity ratios
    that serve as the input to the anomaly-detection model.
    """

    window_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    service: str
    start: datetime
    end: datetime
    template_counts: dict[str, int] = Field(default_factory=dict)
    error_ratio: float = 0.0
    warning_ratio: float = 0.0
    total_count: int = 0
    rate_change: float | None = None


# ---------------------------------------------------------------------------
# Internal helper – a single log entry buffered inside the extractor
# ---------------------------------------------------------------------------


class _BufferedEntry:
    """Lightweight container for a parsed log inside the accumulator."""

    __slots__ = ("template", "timestamp", "raw_message", "severity")

    def __init__(
        self,
        template: ParsedTemplate,
        timestamp: datetime,
        raw_message: str = "",
        severity: str = "INFO",
    ) -> None:
        self.template = template
        self.timestamp = timestamp
        self.raw_message = raw_message
        self.severity = severity.upper()


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------


class FeatureExtractor:
    """Accumulates parsed log entries and builds feature vectors.

    Usage::

        extractor = FeatureExtractor(service="auth-service")
        for log in stream:
            extractor.add_log(parsed_template, timestamp=log.ts, severity=log.level)
        window = extractor.close_window()
        vec = extractor.get_feature_vector(window)

    The class also supports *session-based* windowing for HDFS-style datasets
    where logs are grouped by ``block_id`` rather than fixed time intervals.
    Call ``close_session_windows()`` to finalise all active sessions.
    """

    def __init__(self, service: str = "default") -> None:
        settings = get_settings()
        self._window_seconds: int = settings.feature_window_seconds
        self._service = service

        # Time-based window state
        self._buffer: list[_BufferedEntry] = []
        self._window_start: datetime | None = None
        self._prev_window_count: int | None = None

        # Session-based (HDFS block_id) state: block_id → list of entries
        self._sessions: dict[str, list[_BufferedEntry]] = defaultdict(list)

        # Registry of all template IDs seen so far — used to build a
        # consistent feature-vector dimensionality across windows.
        self._known_templates: list[str] = []

    # ------------------------------------------------------------------
    # Public API – time-based windows
    # ------------------------------------------------------------------

    def add_log(
        self,
        parsed_log: ParsedTemplate,
        *,
        timestamp: datetime | None = None,
        raw_message: str = "",
        severity: str = "INFO",
    ) -> None:
        """Add a parsed log entry to the current time window.

        If *timestamp* is ``None``, the current UTC time is used.

        The method also checks the raw message for HDFS block IDs and, when
        found, adds the entry to the corresponding session window as well.
        """
        ts = timestamp or datetime.now(tz=timezone.utc)

        if self._window_start is None:
            self._window_start = ts

        entry = _BufferedEntry(
            template=parsed_log,
            timestamp=ts,
            raw_message=raw_message,
            severity=severity,
        )
        self._buffer.append(entry)

        # Also accumulate into session window if an HDFS block_id is present.
        block_match = _BLOCK_ID_RE.search(raw_message) or _BLOCK_ID_RE.search(
            " ".join(parsed_log.parameters)
        )
        if block_match:
            self._sessions[block_match.group()].append(entry)

        # Register template id if unseen.
        if parsed_log.template_id not in self._known_templates:
            self._known_templates.append(parsed_log.template_id)

    def close_window(self) -> FeatureWindow:
        """Finalise the current time window and return its ``FeatureWindow``.

        A new (empty) window is started automatically.
        """
        now = datetime.now(tz=timezone.utc)
        start = self._window_start or now
        entries = self._buffer

        window = self._build_window(entries, start=start, end=now)

        # Track rate change (log count delta vs previous window).
        if self._prev_window_count is not None and self._prev_window_count > 0:
            window.rate_change = (window.total_count - self._prev_window_count) / self._prev_window_count
        elif self._prev_window_count == 0:
            window.rate_change = float(window.total_count) if window.total_count else 0.0

        self._prev_window_count = window.total_count

        # Reset for next window.
        self._buffer = []
        self._window_start = None

        logger.info(
            "feature_extractor.window_closed",
            window_id=window.window_id,
            total=window.total_count,
            templates=len(window.template_counts),
            service=self._service,
        )
        return window

    # ------------------------------------------------------------------
    # Public API – session-based windows (HDFS style)
    # ------------------------------------------------------------------

    def close_session_windows(self) -> list[FeatureWindow]:
        """Finalise all session-based (block_id) windows.

        Returns one ``FeatureWindow`` per unique block_id accumulated so far.
        The session buffers are cleared after this call.
        """
        windows: list[FeatureWindow] = []
        for block_id, entries in self._sessions.items():
            if not entries:
                continue
            start = min(e.timestamp for e in entries)
            end = max(e.timestamp for e in entries)
            w = self._build_window(entries, start=start, end=end)
            w.window_id = f"session-{block_id}"
            windows.append(w)
        self._sessions.clear()

        logger.info(
            "feature_extractor.sessions_closed",
            count=len(windows),
            service=self._service,
        )
        return windows

    # ------------------------------------------------------------------
    # Public API – feature vector
    # ------------------------------------------------------------------

    def get_feature_vector(self, window: FeatureWindow) -> np.ndarray:
        """Convert a ``FeatureWindow`` into a 1-D numeric array.

        The vector layout is::

            [ template_count_0, template_count_1, …,   # one per known template
              error_ratio,
              warning_ratio,
              total_count,
              rate_change ]                             # 0.0 if None

        All template slots follow the insertion order tracked by
        ``_known_templates`` so that vectors from different windows share the
        same dimensionality *within a single extractor instance*.
        """
        template_vec = [
            float(window.template_counts.get(tid, 0)) for tid in self._known_templates
        ]
        meta_vec = [
            window.error_ratio,
            window.warning_ratio,
            float(window.total_count),
            window.rate_change if window.rate_change is not None else 0.0,
        ]
        return np.array(template_vec + meta_vec, dtype=np.float64)

    @property
    def feature_names(self) -> list[str]:
        """Human-readable names matching ``get_feature_vector`` columns."""
        return [
            f"tpl_{tid}" for tid in self._known_templates
        ] + ["error_ratio", "warning_ratio", "total_count", "rate_change"]

    @property
    def known_template_ids(self) -> list[str]:
        """Return a copy of currently known template IDs (insertion order)."""
        return list(self._known_templates)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_window(
        self,
        entries: list[_BufferedEntry],
        *,
        start: datetime,
        end: datetime,
    ) -> FeatureWindow:
        """Build a ``FeatureWindow`` from a list of buffered entries."""
        template_counts: dict[str, int] = defaultdict(int)
        error_count = 0
        warn_count = 0

        for entry in entries:
            template_counts[entry.template.template_id] += 1
            if entry.severity == "ERROR" or entry.severity == "CRITICAL":
                error_count += 1
            elif entry.severity in {"WARN", "WARNING"}:
                warn_count += 1

        total = len(entries)
        return FeatureWindow(
            service=self._service,
            start=start,
            end=end,
            template_counts=dict(template_counts),
            error_ratio=error_count / total if total else 0.0,
            warning_ratio=warn_count / total if total else 0.0,
            total_count=total,
        )
