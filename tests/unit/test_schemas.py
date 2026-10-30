"""Unit tests for Pydantic schema validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aiops.reasoning.schemas import ActionType, FixSuggestion, RiskLevel


def test_fix_suggestion_valid() -> None:
    s = FixSuggestion(
        summary="Connection reset during write",
        probable_cause="Ephemeral port saturation",
        suggested_steps=["Increase port range"],
        confidence=0.85,
        risk_level=RiskLevel.MEDIUM,
        referenced_incident_ids=["INC-1001"],
        proposed_action=ActionType.DRAFT_JIRA_TICKET,
    )
    assert s.confidence == 0.85
    assert s.referenced_incident_ids == ["INC-1001"]


def test_fix_suggestion_high_confidence_requires_references() -> None:
    # Confidence > 0.5 without any cited incidents must be rejected
    with pytest.raises(ValidationError) as exc_info:
        FixSuggestion(
            summary="Connection reset during write",
            probable_cause="Ephemeral port saturation",
            suggested_steps=["Increase port range"],
            confidence=0.85,
            risk_level=RiskLevel.MEDIUM,
            referenced_incident_ids=[],
            proposed_action=ActionType.DRAFT_JIRA_TICKET,
        )
    assert "referenced_incident_ids must not be empty" in str(exc_info.value)


def test_fix_suggestion_low_confidence_without_references_allowed() -> None:
    s = FixSuggestion(
        summary="Unclear failure mode",
        probable_cause="Unknown error pattern",
        suggested_steps=["Inspect logs manually"],
        confidence=0.35,
        risk_level=RiskLevel.HIGH,
        referenced_incident_ids=[],
        proposed_action=ActionType.NO_ACTION,
    )
    assert s.confidence == 0.35
    assert s.referenced_incident_ids == []
