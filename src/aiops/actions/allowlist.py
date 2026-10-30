"""Action allowlist and precondition validation for AIOps Assistant.

Ensures the assistant can ONLY propose and execute actions from a strictly
controlled set, and that precondition guards (health checks, freeze windows)
are satisfied before execution.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog

from aiops.reasoning.schemas import ActionType

logger = structlog.get_logger()

# Set of actions that have real-world side effects and REQUIRE human approval
SIDE_EFFECT_ACTIONS: set[ActionType] = {
    ActionType.RESTART_SERVICE,
}


def is_side_effect(action: ActionType | str | None) -> bool:
    """Return True if the action has side effects and requires explicit human approval."""
    if action is None:
        return False
    if isinstance(action, str):
        try:
            action = ActionType(action)
        except ValueError:
            return True  # Unknown actions treated as risky by default
    return action in SIDE_EFFECT_ACTIONS


def validate_action(action: ActionType | str, context: dict[str, Any]) -> tuple[bool, str]:
    """Check preconditions for proposing or executing an action.

    Args:
        action: The action to validate.
        context: Operational context (e.g., service health, freeze windows, etc.).

    Returns:
        tuple of (is_allowed: bool, reason: str)
    """
    if isinstance(action, str):
        try:
            action = ActionType(action)
        except ValueError:
            return False, f"Action '{action}' is not in the allowlist enum."

    if action == ActionType.NO_ACTION:
        return True, "No action required."

    if action in (ActionType.DRAFT_JIRA_TICKET, ActionType.SEND_SLACK_ALERT):
        # Drafting payloads is always safe
        return True, "Safe read/draft action permitted."

    if action == ActionType.RESTART_SERVICE:
        # Preconditions for restart:
        # 1. Not during production freeze window (e.g. check freeze flag or time)
        if context.get("in_freeze_window", False):
            return False, "Cannot restart service during active freeze window."

        # 2. Must have target service identified
        anomaly_event = context.get("anomaly_event", {})
        service = anomaly_event.get("service") or context.get("service")
        if not service:
            return False, "Target service is not specified in context."

        # 3. Health check confirmation (optional flag, default allow if health_failing is True or unchecked)
        health_status = context.get("service_health", "degraded")
        if health_status == "healthy":
            return False, "Service is currently reporting healthy; restart refused."

        return True, f"Service restart permitted for '{service}' pending human approval."

    return False, f"Unknown or unhandled action '{action}'."
