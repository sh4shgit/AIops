"""Unit tests for action allowlist and precondition checks."""

from __future__ import annotations

from aiops.actions.allowlist import is_side_effect, validate_action
from aiops.reasoning.schemas import ActionType


def test_is_side_effect() -> None:
    assert is_side_effect(ActionType.RESTART_SERVICE) is True
    assert is_side_effect(ActionType.DRAFT_JIRA_TICKET) is False
    assert is_side_effect(ActionType.SEND_SLACK_ALERT) is False
    assert is_side_effect(ActionType.NO_ACTION) is False
    assert is_side_effect(None) is False


def test_validate_action_drafting() -> None:
    allowed, reason = validate_action(ActionType.DRAFT_JIRA_TICKET, {})
    assert allowed is True

    allowed, reason = validate_action(ActionType.SEND_SLACK_ALERT, {})
    assert allowed is True


def test_validate_action_restart_service_success() -> None:
    context = {
        "service": "hdfs-datanode",
        "in_freeze_window": False,
        "service_health": "degraded",
    }
    allowed, reason = validate_action(ActionType.RESTART_SERVICE, context)
    assert allowed is True


def test_validate_action_restart_during_freeze_window_refused() -> None:
    context = {
        "service": "hdfs-datanode",
        "in_freeze_window": True,
        "service_health": "degraded",
    }
    allowed, reason = validate_action(ActionType.RESTART_SERVICE, context)
    assert allowed is False
    assert "freeze window" in reason.lower()


def test_validate_action_restart_healthy_service_refused() -> None:
    context = {
        "service": "hdfs-datanode",
        "in_freeze_window": False,
        "service_health": "healthy",
    }
    allowed, reason = validate_action(ActionType.RESTART_SERVICE, context)
    assert allowed is False
    assert "healthy" in reason.lower()


def test_validate_unknown_action_refused() -> None:
    allowed, reason = validate_action("DROP_TABLE_USERS", {})
    assert allowed is False
    assert "allowlist" in reason.lower()
