"""Checks for the HDFS sample and dashboard analysis path."""

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
