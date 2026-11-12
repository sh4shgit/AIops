"""``POST /analyze`` — synchronous end-to-end analysis endpoint for the UI.

Accepts log entries and returns template and severity-based analysis for the
dashboard. This path does not call the LLM or historical incident retrieval.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(tags=["analyze"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class LogEntry(BaseModel):
    """A single log line submitted for analysis."""

    timestamp: str = Field(default="", description="ISO-8601 timestamp (optional)")
    service: str = Field(default="unknown", description="Originating service name")
    level: str = Field(default="INFO", description="Log level")
    message: str = Field(..., min_length=1, description="Raw log message")


class AnalyzeRequest(BaseModel):
    """Batch of log lines to run through the pipeline."""

    logs: list[LogEntry] = Field(..., min_length=1, max_length=500)
    service_override: str | None = Field(
        default=None, description="Override service name for all entries"
    )


class TemplateInfo(BaseModel):
    template_id: str
    template: str
    parameters: list[str]
    count: int


class FeatureInfo(BaseModel):
    window_id: str
    service: str
    n_events: int
    n_unique_templates: int
    error_ratio: float
    warn_ratio: float
    feature_dim: int


class DetectionResult(BaseModel):
    window_id: str
    service: str
    anomaly_score: float
    band: str  # HIGH | LOW | NORMAL
    top_templates: list[str]


class ReasoningResult(BaseModel):
    window_id: str
    summary: str
    probable_cause: str
    suggested_steps: list[str]
    confidence: float
    risk_level: str
    proposed_action: str | None
    referenced_incident_ids: list[str]


class AnalyzeResponse(BaseModel):
    request_id: str
    processed_at: str
    n_logs_received: int
    templates: list[TemplateInfo]
    features: list[FeatureInfo]
    detections: list[DetectionResult]
    reasoning: list[ReasoningResult]
    summary: dict[str, Any]


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def _parse_logs(entries: list[LogEntry]) -> tuple[list[Any], dict[str, int]]:
    """Run Drain3 template miner on all entries."""
    from aiops.detection.parser import LogParser

    parser = LogParser()
    parsed: list[tuple[LogEntry, Any]] = []
    template_counts: dict[str, int] = {}

    for entry in entries:
        result = parser.parse(entry.message)
        parsed.append((entry, result))
        template_counts[result.template_id] = template_counts.get(result.template_id, 0) + 1

    return parsed, template_counts


def _extract_features(
    parsed: list[tuple[Any, Any]]
) -> list[tuple[str, Any, Any]]:
    """Group parsed logs by service, build FeatureWindows."""
    from aiops.detection.features import FeatureExtractor

    # Group by service
    service_groups: dict[str, list[tuple[Any, Any]]] = {}
    for entry, tmpl in parsed:
        svc = entry.service
        service_groups.setdefault(svc, []).append((entry, tmpl))

    all_windows: list[tuple[str, Any, FeatureExtractor]] = []
    for svc, group in service_groups.items():
        extractor = FeatureExtractor(service=svc)
        ungrouped = FeatureExtractor(service=svc)
        ungrouped_count = 0
        for entry, tmpl in group:
            try:
                ts = datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except ValueError:
                ts = datetime.now(timezone.utc)
            extractor.add_log(
                tmpl,
                timestamp=ts,
                raw_message=entry.message,
                severity=entry.level,
            )
            if not re.search(r"blk_-?\d+", entry.message):
                ungrouped.add_log(tmpl, timestamp=ts, raw_message=entry.message, severity=entry.level)
                ungrouped_count += 1
        windows = extractor.close_session_windows()
        for w in windows:
            all_windows.append((svc, w, extractor))
        if ungrouped_count:
            all_windows.append((svc, ungrouped.close_window(), ungrouped))

    return all_windows


def _detect_anomalies(
    windows: list[tuple[str, Any, Any]]
) -> list[tuple[Any, float, str]]:
    """Score severity ratios with a transparent, untrained heuristic."""
    results: list[tuple[Any, float, str]] = []
    from aiops.detection.thresholds import classify_anomaly
    for svc, window, extractor in windows:
        score = -(window.error_ratio * 0.5 + window.warning_ratio * 0.2)
        results.append((window, score, classify_anomaly(score).value))
    return results


def _run_reasoning(
    detections: list[tuple[Any, float, str]],
    windows: list[tuple[str, Any, Any]],
    parsed: list[tuple[Any, Any]],
) -> list[tuple[str, Any]]:
    """Generate fix suggestions for HIGH/LOW anomaly windows.

    Uses the LLM client's mock mode synchronously — avoids spinning up a new
    event loop (which hangs on Windows Python 3.15 with structlog in subprocess).
    """
    from aiops.detection.thresholds import AnomalyEvent, AnomalyBand
    from aiops.reasoning.llm import LLMClient
    from aiops.reasoning.prompts import build_analysis_prompt

    client = LLMClient()  # auto-activates mock mode

    results: list[tuple[str, Any]] = []

    for (window, score, band_str), (svc, win, extractor) in zip(detections, windows):
        if band_str == "NORMAL":
            continue

        band = AnomalyBand(band_str)

        raw_lines = [e.message for e, _ in parsed if e.service == svc]
        raw_log_snippet = "\n".join(raw_lines[:10])

        top_tpl_ids = sorted(
            win.template_counts.keys(),
            key=lambda tid: win.template_counts[tid],
            reverse=True,
        )[:3]

        anomaly = AnomalyEvent(
            window_id=win.window_id,
            score=score,
            band=band,
            service=svc,
            key_templates=top_tpl_ids,
            raw_log=raw_log_snippet,
        )

        # Build a minimal prompt string
        prompt = build_analysis_prompt(anomaly, [], "")

        try:
            # Call the synchronous mock directly — no event loop needed
            suggestion = client._generate_mock_suggestion(prompt)
            results.append((win.window_id, suggestion))
        except Exception:
            pass

    return results


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_logs(request: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze a batch with template mining and severity-based scoring."""
    request_id = uuid.uuid4().hex
    processed_at = datetime.now(timezone.utc).isoformat()

    # Apply service override
    entries = request.logs
    if request.service_override:
        entries = [
            LogEntry(
                timestamp=e.timestamp,
                service=request.service_override,
                level=e.level,
                message=e.message,
            )
            for e in entries
        ]

    # --- Stage 1: Parse ---
    try:
        parsed, template_counts = _parse_logs(entries)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Parsing failed: {exc}")

    # Collect unique templates
    seen_templates: dict[str, TemplateInfo] = {}
    for _entry, tmpl in parsed:
        tid = tmpl.template_id
        if tid not in seen_templates:
            seen_templates[tid] = TemplateInfo(
                template_id=tid,
                template=tmpl.template,
                parameters=tmpl.parameters[:5],  # cap for display
                count=template_counts.get(tid, 1),
            )
        else:
            seen_templates[tid].count = template_counts.get(tid, 1)
            seen_templates[tid].template = tmpl.template

    # --- Stage 2: Feature Extraction ---
    try:
        windows = _extract_features(parsed)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {exc}")

    feature_infos: list[FeatureInfo] = []
    for svc, window, extractor in windows:
        fvec = extractor.get_feature_vector(window)
        # Template counts are the first N elements; last 4 are meta
        n_tmpl = max(0, len(fvec) - 4)
        feature_infos.append(
            FeatureInfo(
                window_id=window.window_id,
                service=svc,
                n_events=window.total_count,
                n_unique_templates=len(window.template_counts),
                error_ratio=round(window.error_ratio, 4),
                warn_ratio=round(window.warning_ratio, 4),
                feature_dim=len(fvec),
            )
        )

    # --- Stage 3: Anomaly Detection ---
    try:
        detections_raw = _detect_anomalies(windows)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Detection failed: {exc}") from exc

    detection_results: list[DetectionResult] = []
    for (svc, window, extractor), (win, score, band) in zip(windows, detections_raw):
        top_tpl_ids = sorted(
            win.template_counts.keys(),
            key=lambda tid: win.template_counts[tid],
            reverse=True,
        )[:3]
        # Map template IDs → template strings for display
        top_templates = [
            seen_templates[tid].template if tid in seen_templates else tid
            for tid in top_tpl_ids
        ]
        detection_results.append(
            DetectionResult(
                window_id=win.window_id,
                service=svc,
                anomaly_score=round(score, 4),
                band=band,
                top_templates=top_templates,
            )
        )

    # The dashboard does not present mock explanations as real analysis.
    reasoning_raw = []

    reasoning_results: list[ReasoningResult] = []
    for window_id, suggestion in reasoning_raw:
        reasoning_results.append(
            ReasoningResult(
                window_id=window_id,
                summary=suggestion.summary,
                probable_cause=suggestion.probable_cause,
                suggested_steps=suggestion.suggested_steps,
                confidence=suggestion.confidence,
                risk_level=suggestion.risk_level.value
                if hasattr(suggestion.risk_level, "value")
                else str(suggestion.risk_level),
                proposed_action=(
                    suggestion.proposed_action.value
                    if suggestion.proposed_action
                    and hasattr(suggestion.proposed_action, "value")
                    else str(suggestion.proposed_action)
                    if suggestion.proposed_action
                    else None
                ),
                referenced_incident_ids=suggestion.referenced_incident_ids,
            )
        )

    # --- Summary ---
    n_high = sum(1 for d in detection_results if d.band == "HIGH")
    n_low = sum(1 for d in detection_results if d.band == "LOW")
    n_normal = sum(1 for d in detection_results if d.band == "NORMAL")

    summary: dict[str, Any] = {
        "total_logs": len(entries),
        "unique_templates": len(seen_templates),
        "windows_analyzed": len(windows),
        "anomalies_high": n_high,
        "anomalies_low": n_low,
        "normal_windows": n_normal,
        "has_critical": n_high > 0,
        "detection_method": "severity heuristic",
        "reasoning_mode": "not enabled",
    }

    return AnalyzeResponse(
        request_id=request_id,
        processed_at=processed_at,
        n_logs_received=len(entries),
        templates=list(seen_templates.values()),
        features=feature_infos,
        detections=detection_results,
        reasoning=reasoning_results,
        summary=summary,
    )
