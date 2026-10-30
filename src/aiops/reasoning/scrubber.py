"""PII / Secret scrubber for sanitising text before LLM consumption.

Compiled regex patterns are used for performance.  Every detected
sensitive value is replaced with a descriptive placeholder such as
``[EMAIL_REDACTED]`` so the downstream model never sees raw secrets.
"""

from __future__ import annotations

import re
from typing import Any

import structlog

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Compiled patterns — order matters (more-specific rules first).
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # JWT tokens (header.payload.signature)
    (
        re.compile(
            r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
        ),
        "[JWT_REDACTED]",
    ),
    # AWS access key IDs  (AKIA…)
    (
        re.compile(r"(?<![A-Z0-9])AKIA[0-9A-Z]{16}(?![A-Z0-9])"),
        "[AWS_KEY_REDACTED]",
    ),
    # Bearer tokens in Authorization headers
    (
        re.compile(r"[Bb]earer\s+[A-Za-z0-9\-._~+/]+=*", re.ASCII),
        "[BEARER_TOKEN_REDACTED]",
    ),
    # OpenAI / Slack / generic API keys
    (
        re.compile(r"sk-[A-Za-z0-9]{20,}"),
        "[API_KEY_REDACTED]",
    ),
    (
        re.compile(r"xoxb-[A-Za-z0-9\-]{20,}"),
        "[API_KEY_REDACTED]",
    ),
    (
        re.compile(r"xoxp-[A-Za-z0-9\-]{20,}"),
        "[API_KEY_REDACTED]",
    ),
    (
        re.compile(r"xoxa-[A-Za-z0-9\-]{20,}"),
        "[API_KEY_REDACTED]",
    ),
    # Passwords embedded in URLs  (scheme://user:password@host)
    (
        re.compile(r"://[^:@/\s]+:([^@/\s]+)@"),
        "://[USER]:[PASSWORD_REDACTED]@",
    ),
    # US Social Security Numbers  (NNN-NN-NNNN)
    (
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[SSN_REDACTED]",
    ),
    # Credit card–style numbers (13–19 digits, optional separators)
    (
        re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
        "[CREDIT_CARD_REDACTED]",
    ),
    # Email addresses
    (
        re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        ),
        "[EMAIL_REDACTED]",
    ),
    # IPv6 (simplified — at least two colon-separated hex groups)
    (
        re.compile(
            r"\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b",
        ),
        "[IP_REDACTED]",
    ),
    # IPv4
    (
        re.compile(
            r"\b(?:25[0-5]|2[0-4]\d|[01]?\d\d?)"
            r"(?:\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)){3}\b",
        ),
        "[IP_REDACTED]",
    ),
    # Generic long hex / base-64 secrets (≥ 32 chars, not a common hash)
    (
        re.compile(r"(?<![A-Za-z0-9/+])[A-Za-z0-9/+]{40,}={0,2}(?![A-Za-z0-9/+=])"),
        "[SECRET_REDACTED]",
    ),
]


class Scrubber:
    """Stateless PII / secret scrubber.

    Usage::

        scrubber = Scrubber()
        clean = scrubber.scrub("Send to admin@corp.io")
        # => "Send to [EMAIL_REDACTED]"
    """

    def __init__(self) -> None:
        self._patterns = _PATTERNS

    # -- public API ---------------------------------------------------------

    def scrub(self, text: str) -> str:
        """Return *text* with all recognised secrets/PII replaced by placeholders."""
        result = text
        for pattern, placeholder in self._patterns:
            result = pattern.sub(placeholder, result)
        return result

    def scrub_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Recursively scrub all string values in *data*.

        Non-string leaf values are left unchanged.  Nested dicts and lists are
        traversed recursively.
        """
        return self._walk(data)  # type: ignore[return-value]

    # -- internals ----------------------------------------------------------

    def _walk(self, obj: Any) -> Any:  # noqa: ANN401
        if isinstance(obj, str):
            return self.scrub(obj)
        if isinstance(obj, dict):
            return {k: self._walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._walk(item) for item in obj]
        return obj
