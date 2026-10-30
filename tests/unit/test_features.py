"""Unit tests for feature extraction and windowing."""

from __future__ import annotations

from datetime import datetime, timezone

from aiops.detection.features import FeatureExtractor
from aiops.detection.parser import ParsedTemplate


def test_feature_extraction_window() -> None:
    extractor = FeatureExtractor(service="auth-service")
    ts = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)

    tmpl1 = ParsedTemplate(template_id="E1", template="User <*> logged in", parameters=["user1"])
    tmpl2 = ParsedTemplate(template_id="E2", template="Failed password for <*>", parameters=["user2"])

    extractor.add_log(tmpl1, timestamp=ts, raw_message="User user1 logged in", severity="INFO")
    extractor.add_log(tmpl2, timestamp=ts, raw_message="Failed password for user2", severity="ERROR")

    window = extractor.close_window()
    assert window.total_count == 2
    assert window.template_counts == {"E1": 1, "E2": 1}
    assert window.error_ratio == 0.5
    assert window.warning_ratio == 0.0

    vec = extractor.get_feature_vector(window)
    assert len(vec) == 2 + 4  # 2 known templates + 4 meta features (error, warn, total, rate)
    assert vec[-4] == 0.5  # error_ratio
    assert vec[-2] == 2.0  # total_count


def test_hdfs_session_windowing() -> None:
    extractor = FeatureExtractor(service="hdfs-datanode")
    ts = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)

    tmpl = ParsedTemplate(template_id="E10", template="Receiving block <*>", parameters=["blk_101"])
    extractor.add_log(tmpl, timestamp=ts, raw_message="Receiving block blk_101", severity="INFO")
    extractor.add_log(tmpl, timestamp=ts, raw_message="Receiving block blk_102", severity="INFO")

    sessions = extractor.close_session_windows()
    assert len(sessions) == 2
    block_ids = {s.window_id for s in sessions}
    assert "session-blk_101" in block_ids
    assert "session-blk_102" in block_ids
