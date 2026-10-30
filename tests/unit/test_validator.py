"""Unit tests for log ingestion validation."""

from __future__ import annotations

import pytest

from aiops.ingestion.validator import validate_entry


def test_validate_valid_entry() -> None:
    data = {
        "timestamp": "2026-10-28T12:00:00Z",
        "service": "api-gateway",
        "level": "INFO",
        "message": "User login successful",
    }
    parsed, error = validate_entry(data, 0)
    assert error is None
    assert parsed is not None
    assert parsed.service == "api-gateway"
    assert parsed.level.value == "INFO"
    assert len(parsed.log_id) == 64  # SHA256 length


def test_validate_invalid_entry_blank_message() -> None:
    data = {
        "timestamp": "2026-10-28T12:00:00Z",
        "service": "api-gateway",
        "level": "INFO",
        "message": "   ",
    }
    parsed, error = validate_entry(data, 1)
    assert parsed is None
    assert error is not None
    assert error.index == 1
    assert "blank" in error.error.lower()


def test_validate_invalid_level() -> None:
    data = {
        "timestamp": "2026-10-28T12:00:00Z",
        "service": "api-gateway",
        "level": "SUPER_PANIC",
        "message": "something failed",
    }
    parsed, error = validate_entry(data, 2)
    assert parsed is None
    assert error is not None
    assert error.index == 2


def test_deterministic_id_generation() -> None:
    data = {
        "timestamp": "2026-10-28T12:00:00Z",
        "service": "api-gateway",
        "level": "INFO",
        "message": "Same exact log line",
    }
    parsed1, _ = validate_entry(data, 0)
    parsed2, _ = validate_entry(data, 1)
    assert parsed1 is not None and parsed2 is not None
    assert parsed1.log_id == parsed2.log_id
