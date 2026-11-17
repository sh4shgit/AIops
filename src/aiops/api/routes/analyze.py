"""``POST /analyze`` — synchronous end-to-end analysis endpoint for the UI.

Accepts log entries and returns severity signals, incident matches, and
rule-based investigation steps for the dashboard.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from secrets import compare_digest
from typing import TYPE_CHECKING, Annotated, Any, Literal

import httpx
import structlog
from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from aiops.config import get_settings
from aiops.detection.parser import LogParser, ParsedTemplate
from aiops.detection.thresholds import AnomalyBand, classify_anomaly
from aiops.reasoning.groq import GroqError, GroundedRecommendation, generate_recommendation
from aiops.reasoning.guidance import InvestigationGuidance, explain_checks
from aiops.retrieval.historical import IncidentMatch, IncidentSearch, load_incidents

if TYPE_CHECKING:
    from aiops.detection.features import FeatureExtractor, FeatureWindow

IncidentLookup = Literal["examples", "configured", "authentication_required", "unavailable"]

router = APIRouter(tags=["analyze"])
logger = structlog.get_logger()
ERROR_WEIGHT = 0.5
WARNING_WEIGHT = 0.2


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
    evidence_lines: list[int] = Field(default_factory=list, description="1-based input log indices")
    historical_incidents: list[IncidentMatch] = Field(default_factory=list)
    suggested_checks: list[str] = Field(default_factory=list)
    investigation_guidance: list[InvestigationGuidance] = Field(default_factory=list)


class ScoringPolicy(BaseModel):
    error_weight: float
    warning_weight: float
    high_threshold: float
    low_threshold: float


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
    incident_lookup: IncidentLookup
    scoring: ScoringPolicy
    recommendation_available: bool


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def _parse_logs(entries: list[LogEntry]) -> tuple[list[tuple[LogEntry, ParsedTemplate]], dict[str, int]]:
    """Run Drain3 template miner on all entries."""
    parser = LogParser()
    parsed: list[tuple[LogEntry, ParsedTemplate]] = []
    template_counts: dict[str, int] = {}

    for entry in entries:
        result = parser.parse(entry.message)
        parsed.append((entry, result))
        template_counts[result.template_id] = template_counts.get(result.template_id, 0) + 1

    return parsed, template_counts


def _extract_features(
    parsed: list[tuple[LogEntry, ParsedTemplate]],
) -> list[tuple[FeatureWindow, FeatureExtractor, list[int]]]:
    """Group logs by service and block, keeping their original line numbers."""
    from aiops.detection.features import FeatureExtractor

    service_groups: dict[str, list[tuple[int, LogEntry, ParsedTemplate]]] = {}
    for number, (entry, template) in enumerate(parsed, start=1):
        service_groups.setdefault(entry.service, []).append((number, entry, template))

    all_windows: list[tuple[FeatureWindow, FeatureExtractor, list[int]]] = []
    for service, group in service_groups.items():
        extractor = FeatureExtractor(service=service)
        ungrouped = FeatureExtractor(service=service)
        session_lines: dict[str, list[int]] = {}
        ungrouped_lines: list[int] = []
        for number, entry, template in group:
            try:
                ts = datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except ValueError:
                ts = datetime.now(timezone.utc)
            session_id = extractor.add_log(
                template,
                timestamp=ts,
                raw_message=entry.message,
                severity=entry.level,
            )
            if session_id:
                session_lines.setdefault(session_id, []).append(number)
            else:
                ungrouped.add_log(
                    template, timestamp=ts, raw_message=entry.message, severity=entry.level
                )
                ungrouped_lines.append(number)
        for window in extractor.close_session_windows():
            all_windows.append((window, extractor, session_lines[window.window_id]))
        if ungrouped_lines:
            all_windows.append((ungrouped.close_window(), ungrouped, ungrouped_lines))

    return all_windows


def _incident_search(api_key: str | None) -> tuple[IncidentSearch, IncidentLookup]:
    settings = get_settings()
    corpus_path = settings.incident_corpus_path
    if corpus_path:
        expected_key = settings.api_key.get_secret_value()
        if (
            not expected_key or expected_key == "changeme" or not api_key
            or not compare_digest(api_key.encode(), expected_key.encode())
        ):
            return IncidentSearch([]), "authentication_required"

    try:
        records = load_incidents(corpus_path)
    except (OSError, ValueError):
        logger.warning("incident_corpus_unavailable")
        return IncidentSearch([]), "unavailable"
    return IncidentSearch(records), "configured" if corpus_path else "examples"


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_logs(
    request: AnalyzeRequest,
    api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> AnalyzeResponse:
    """Analyze a batch and attach local investigation guidance to flagged windows."""
    entries = request.logs
    if request.service_override:
        entries = [
            entry.model_copy(update={"service": request.service_override}) for entry in entries
        ]

    try:
        parsed, template_counts = _parse_logs(entries)
        windows = _extract_features(parsed)
    except Exception as exc:
        logger.exception("log_analysis_failed")
        raise HTTPException(status_code=500, detail="Could not analyze the submitted logs") from exc

    seen_templates: dict[str, TemplateInfo] = {}
    for _, template in parsed:
        seen_templates[template.template_id] = TemplateInfo(
            template_id=template.template_id,
            template=template.template,
            parameters=template.parameters[:5],
            count=template_counts[template.template_id],
        )

    settings = get_settings()
    incident_search, incident_lookup = _incident_search(api_key)
    feature_infos: list[FeatureInfo] = []
    detection_results: list[DetectionResult] = []
    for window, extractor, line_numbers in windows:
        feature_infos.append(
            FeatureInfo(
                window_id=window.window_id,
                service=window.service,
                n_events=window.total_count,
                n_unique_templates=len(window.template_counts),
                error_ratio=round(window.error_ratio, 4),
                warn_ratio=round(window.warning_ratio, 4),
                feature_dim=len(extractor.feature_names),
            )
        )

        score = -(window.error_ratio * ERROR_WEIGHT + window.warning_ratio * WARNING_WEIGHT)
        band = classify_anomaly(score, settings=settings)
        top_template_ids = sorted(
            window.template_counts, key=window.template_counts.__getitem__, reverse=True
        )[:3]
        evidence = [(number, entries[number - 1].message) for number in line_numbers]
        triage = band != AnomalyBand.NORMAL
        guidance = explain_checks(evidence) if triage else []
        detection_results.append(
            DetectionResult(
                window_id=window.window_id,
                service=window.service,
                anomaly_score=round(score, 4),
                band=band.value,
                top_templates=[seen_templates[tid].template for tid in top_template_ids],
                evidence_lines=line_numbers,
                historical_incidents=incident_search.find(evidence) if triage else [],
                suggested_checks=[step for recommendation in guidance for step in recommendation.steps],
                investigation_guidance=guidance,
            )
        )

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
        "reasoning_mode": "rule-based",
    }

    return AnalyzeResponse(
        request_id=uuid.uuid4().hex,
        processed_at=datetime.now(timezone.utc).isoformat(),
        n_logs_received=len(entries),
        templates=list(seen_templates.values()),
        features=feature_infos,
        detections=detection_results,
        reasoning=[],
        incident_lookup=incident_lookup,
        recommendation_available=bool(settings.groq_api_key.get_secret_value().strip()),
        scoring=ScoringPolicy(
            error_weight=ERROR_WEIGHT,
            warning_weight=WARNING_WEIGHT,
            high_threshold=settings.anomaly_high_threshold,
            low_threshold=settings.anomaly_low_threshold,
        ),
        summary=summary,
    )


@router.post("/analyze/recommendation", response_model=GroundedRecommendation)
async def recommend_window(
    batch: AnalyzeRequest,
    request: Request,
    api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> GroundedRecommendation:
    """Retrieve local incident evidence and request one recommendation from Groq."""
    settings = get_settings()
    key = settings.groq_api_key.get_secret_value().strip()
    if not key:
        raise HTTPException(status_code=503, detail="Set GROQ_API_KEY on the server to enable recommendations.")

    # A loopback request from the dashboard can use the configured Groq key.
    # Remote and cross-origin callers need the application's own API key.
    origin = request.headers.get("origin")
    is_local = request.client is not None and request.client.host in {"127.0.0.1", "::1"}
    if not is_local or (origin and origin != str(request.base_url).rstrip("/")):
        expected_key = settings.api_key.get_secret_value()
        if (
            not api_key or not expected_key or expected_key == "changeme"
            or not compare_digest(api_key.encode(), expected_key.encode())
        ):
            raise HTTPException(status_code=403, detail="The server API key is required for remote recommendations.")

    analysis = await run_in_threadpool(analyze_logs, batch, api_key)
    if len(analysis.detections) != 1 or analysis.detections[0].band == AnomalyBand.NORMAL.value:
        raise HTTPException(status_code=422, detail="Select a single flagged log window.")
    window = analysis.detections[0]
    if analysis.incident_lookup == "authentication_required":
        raise HTTPException(status_code=403, detail="Enter the server API key to use the configured incident history.")
    if analysis.incident_lookup == "unavailable":
        raise HTTPException(status_code=503, detail="The configured incident history is unavailable.")
    if not window.historical_incidents:
        raise HTTPException(status_code=422, detail="No stored incident pattern matched this window. Use the investigation checks.")

    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            return await generate_recommendation(
                client,
                api_key=key,
                model=settings.groq_model,
                band=window.band,
                lines=[(number, batch.logs[number - 1].message) for number in window.evidence_lines],
                matches=window.historical_incidents,
            )
    except GroqError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from None
