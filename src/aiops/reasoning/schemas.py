"""Pydantic models for LLM-generated fix suggestions.

The ``FixSuggestion`` model is the canonical output of the reasoning engine.
It encodes a structured recommendation including risk assessment, action type,
and references to the retrieved incidents that support the conclusion.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class RiskLevel(str, Enum):
    """Qualitative risk assessment for a suggested fix."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ActionType(str, Enum):
    """Allowlisted actions the assistant may propose."""

    DRAFT_JIRA_TICKET = "DRAFT_JIRA_TICKET"
    SEND_SLACK_ALERT = "SEND_SLACK_ALERT"
    RESTART_SERVICE = "RESTART_SERVICE"
    NO_ACTION = "NO_ACTION"


class FixSuggestion(BaseModel):
    """Structured output produced by the LLM reasoning engine.

    Attributes:
        summary: One-line human-readable summary of the suggestion.
        probable_cause: Detailed root-cause hypothesis.
        suggested_steps: Ordered remediation steps.
        confidence: Model confidence in [0.0, 1.0].
        risk_level: Qualitative risk of applying the suggestion.
        referenced_incident_ids: IDs of retrieved incidents supporting this.
        proposed_action: Which allowlisted action to take (if any).
    """

    summary: str = Field(..., min_length=1, description="One-line summary of the fix suggestion.")
    probable_cause: str = Field(
        ..., min_length=1, description="Detailed root-cause hypothesis."
    )
    suggested_steps: list[str] = Field(
        default_factory=list,
        description="Ordered list of remediation steps.",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model confidence score."
    )
    risk_level: RiskLevel = Field(
        ..., description="Qualitative risk assessment."
    )
    referenced_incident_ids: list[str] = Field(
        default_factory=list,
        description="Incident IDs from the retrieval layer cited by this suggestion.",
    )
    proposed_action: ActionType | None = Field(
        default=None,
        description="Proposed allowlisted action, or None / NO_ACTION.",
    )

    # -- validators ---------------------------------------------------------

    @model_validator(mode="after")
    def _require_references_when_confident(self) -> FixSuggestion:
        """If confidence > 0.5 the model *must* cite at least one incident."""
        if self.confidence > 0.5 and not self.referenced_incident_ids:
            raise ValueError(
                "referenced_incident_ids must not be empty when confidence > 0.5 "
                f"(got confidence={self.confidence})"
            )
        return self
