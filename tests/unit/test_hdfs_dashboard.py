"""Checks for the HDFS sample and dashboard analysis path."""

import json

from aiops.api.routes.analyze import AnalyzeRequest, analyze_logs
from aiops.api.routes.datasets import hdfs_sample, parse_hdfs_line


def test_real_sample_is_paginated_and_preserves_log_content():
    sample = hdfs_sample(offset=0, limit=2)
    assert sample["total"] == 2000
    assert len(sample["logs"]) == 2
    first = sample["logs"][0]
    assert first["raw"].startswith("081109 203615 148 INFO")
    assert first["timestamp"] == "2008-11-09T20:36:15"
    assert "blk_38865049064139660" in first["message"]
    assert len(hdfs_sample(offset=1999, limit=2)["logs"]) == 1


def test_analysis_uses_real_levels_and_retains_unmatched_lines():
    block_line = parse_hdfs_line(
        "081109 203615 148 FATAL dfs.DataNode: Failed block blk_-123"
    )
    ungrouped = parse_hdfs_line("081109 203616 149 WARN dfs.DataNode: disk warning")
    result = analyze_logs(AnalyzeRequest(logs=[block_line, ungrouped]))
    assert result.summary["total_logs"] == 2
    assert result.summary["windows_analyzed"] == 2
    assert result.summary["detection_method"] == "severity heuristic"
    assert result.summary["anomalies_high"] == 1
    assert result.summary["anomalies_low"] == 1
    assert result.reasoning == []


def test_historical_match_uses_lines_from_the_same_block():
    warning = hdfs_sample(offset=77, limit=1)["logs"][0]
    unrelated = parse_hdfs_line("081109 214044 2562 ERROR dfs.DataNode: disk error for blk_123")
    result = analyze_logs(AnalyzeRequest(logs=[unrelated, warning]))
    flagged, matched = result.detections
    assert flagged.evidence_lines == [1]
    assert not flagged.historical_incidents
    assert flagged.suggested_checks
    assert matched.evidence_lines == [2]
    assert matched.historical_incidents[0].incident.id == "HDFS-EXAMPLE-01"
    assert matched.historical_incidents[0].line_number == 2
    assert "DataNode" in matched.suggested_checks[0]
    guidance = matched.investigation_guidance[0]
    assert guidance.line_number == 2
    assert guidance.matched_text == "exception while serving"
    assert guidance.steps == matched.suggested_checks
    assert flagged.investigation_guidance[0].line_number is None
    assert result.incident_lookup == "examples"


def test_scoring_metadata_reflects_configured_thresholds(monkeypatch):
    monkeypatch.setenv("ANOMALY_HIGH_THRESHOLD", "-0.15")
    monkeypatch.setenv("ANOMALY_LOW_THRESHOLD", "-0.05")
    warning = hdfs_sample(offset=77, limit=1)["logs"][0]
    result = analyze_logs(AnalyzeRequest(logs=[warning]))

    assert result.scoring.high_threshold == -0.15
    assert result.scoring.low_threshold == -0.05
    assert result.detections[0].anomaly_score == -result.scoring.warning_weight
    assert result.detections[0].band == "HIGH"


def test_configured_incidents_require_the_api_key(tmp_path, monkeypatch):
    path = tmp_path / "history.json"
    path.write_text(
        json.dumps([{
            "id": "PRIVATE-42",
            "title": "Private history",
            "symptoms": "A repeated transfer warning.",
            "log_templates": ["connection reset by peer"],
            "root_cause": "An upstream connection was interrupted.",
            "resolution": "Reviewed client logs.",
        }]),
        encoding="utf-8",
    )
    monkeypatch.setenv("INCIDENT_CORPUS_PATH", str(path))
    monkeypatch.setenv("API_KEY", "history-test-key")
    request = AnalyzeRequest(logs=[{
        "level": "ERROR",
        "message": "DataNode: Connection reset by peer for blk_999",
    }])
    restricted = analyze_logs(request)
    assert restricted.incident_lookup == "authentication_required"
    assert not restricted.detections[0].historical_incidents
    assert restricted.detections[0].suggested_checks

    authorized = analyze_logs(request, api_key="history-test-key")
    assert authorized.incident_lookup == "configured"
    assert authorized.detections[0].historical_incidents[0].incident.id == "PRIVATE-42"

    path.write_text("invalid JSON", encoding="utf-8")
    unavailable = analyze_logs(request, api_key="history-test-key")
    assert unavailable.incident_lookup == "unavailable"
    assert unavailable.detections[0].suggested_checks
