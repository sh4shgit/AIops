"""Claude API client with retry logic and structured-output parsing.

Wraps the Anthropic SDK to produce :class:`~aiops.reasoning.schemas.FixSuggestion`
objects.  Tracks per-call token usage and latency for observability.
"""

from __future__ import annotations

import json
import time
from typing import Any

import anthropic
import structlog
from pydantic import ValidationError

from aiops.config import get_settings
from aiops.observability.metrics import (
    LLM_REQUEST_LATENCY_SECONDS,
    LLM_TOKENS_TOTAL,
    LLM_VALIDATION_FAILURES_TOTAL,
)
from aiops.reasoning.schemas import ActionType, FixSuggestion, RiskLevel

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_RETRIES: int = 3
_RETRY_BACKOFF_BASE: float = 1.0  # seconds


class LLMClient:
    """Async wrapper around the Anthropic Messages API.

    Parameters
    ----------
    api_key:
        Anthropic API key.  Falls back to ``settings.anthropic_api_key``.
    model:
        Model identifier (e.g. ``claude-sonnet-4-20250514``).
    max_tokens:
        Maximum tokens in the response.
    temperature:
        Sampling temperature.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        mock_mode: bool = False,
    ) -> None:
        settings = get_settings()
        self._api_key = api_key or settings.anthropic_api_key.get_secret_value()
        self._model = model or settings.llm_model
        self._max_tokens = max_tokens or settings.llm_max_tokens
        self._temperature = temperature if temperature is not None else settings.llm_temperature
        self.mock_mode = mock_mode or (not self._api_key) or self._api_key.startswith("sk-ant-...")

        if not self.mock_mode:
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        else:
            self._client = None

        # Observability accumulators (reset externally if needed)
        self.last_tokens_used: int = 0
        self.last_latency_ms: float = 0.0

    # -- public API ---------------------------------------------------------

    async def analyze(self, prompt: str, *, system_prompt: str = "") -> FixSuggestion:
        """Send *prompt* to Claude and parse the response into a ``FixSuggestion``.

        Retries up to ``_MAX_RETRIES`` times on API errors or JSON / Pydantic
        parse failures.

        Parameters
        ----------
        prompt:
            The fully-assembled user-role prompt.
        system_prompt:
            Optional system-role prompt.  If empty, the default
            ``SYSTEM_PROMPT`` from :mod:`aiops.reasoning.prompts` is used.

        Returns
        -------
        FixSuggestion
            Validated structured suggestion.

        Raises
        ------
        RuntimeError
            If all retries are exhausted.
        """
        from aiops.reasoning.prompts import SYSTEM_PROMPT as _DEFAULT_SYS

        sys_prompt = system_prompt or _DEFAULT_SYS
        last_error: Exception | None = None

        if self.mock_mode or self._client is None:
            return self._generate_mock_suggestion(prompt)

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                suggestion = await self._call_api(sys_prompt, prompt, attempt)
                return suggestion
            except (anthropic.APIError, json.JSONDecodeError, ValidationError) as exc:
                LLM_VALIDATION_FAILURES_TOTAL.labels(reason=type(exc).__name__).inc()
                last_error = exc
                log.warning(
                    "llm_call_failed",
                    attempt=attempt,
                    max_retries=_MAX_RETRIES,
                    error=str(exc),
                )
                if attempt < _MAX_RETRIES:
                    import asyncio

                    await asyncio.sleep(_RETRY_BACKOFF_BASE * (2 ** (attempt - 1)))

        raise RuntimeError(
            f"LLM analysis failed after {_MAX_RETRIES} attempts: {last_error}"
        ) from last_error

    def _generate_mock_suggestion(self, prompt: str) -> FixSuggestion:
        """Generate a deterministic, validated fix suggestion when running in mock/demo mode."""
        import re

        # Extract incident IDs mentioned in prompt
        inc_ids = re.findall(r"Incident \d+: ([A-Za-z0-9\-_]+)", prompt)
        if not inc_ids:
            inc_ids = re.findall(r"INC-[0-9]+", prompt)

        start = time.monotonic()
        time.sleep(0.05)  # Tiny simulated latency
        elapsed = time.monotonic() - start

        self.last_latency_ms = elapsed * 1000
        self.last_tokens_used = 420
        LLM_REQUEST_LATENCY_SECONDS.labels(model="mock-claude").observe(elapsed)
        LLM_TOKENS_TOTAL.labels(model="mock-claude", type="input").inc(350)
        LLM_TOKENS_TOTAL.labels(model="mock-claude", type="output").inc(70)

        referenced = inc_ids[:2] if inc_ids else []
        confidence = 0.88 if referenced else 0.45

        return FixSuggestion(
            summary="Investigate abnormal log template frequency and connection timeout pattern",
            probable_cause=(
                "Correlation with historical incident(s) suggests network thread pool exhaustion "
                "or downstream datanode replication timeout under heavy batch load."
            ),
            suggested_steps=[
                "Check socket buffer and connection pool metrics on active nodes",
                "Review recent config updates to block replication threshold",
                "Draft Jira ticket for platform team review",
            ],
            confidence=confidence,
            risk_level=RiskLevel.MEDIUM,
            referenced_incident_ids=referenced,
            proposed_action=ActionType.DRAFT_JIRA_TICKET,
        )

    def validate_references(
        self,
        suggestion: FixSuggestion,
        retrieved_ids: set[str],
    ) -> FixSuggestion:
        """Strip any hallucinated incident references not in *retrieved_ids*.

        If stripping references would violate the confidence > 0.5 rule, the
        confidence is clamped to 0.5.

        Parameters
        ----------
        suggestion:
            The raw LLM suggestion.
        retrieved_ids:
            The set of incident IDs actually returned by retrieval.

        Returns
        -------
        FixSuggestion
            A copy with only valid references.
        """
        valid_ids = [
            ref for ref in suggestion.referenced_incident_ids if ref in retrieved_ids
        ]
        hallucinated = set(suggestion.referenced_incident_ids) - set(valid_ids)

        if hallucinated:
            log.warning(
                "hallucinated_references_stripped",
                hallucinated=sorted(hallucinated),
                remaining=valid_ids,
            )

        # Clamp confidence when we'd otherwise violate the validator
        confidence = suggestion.confidence
        if confidence > 0.5 and not valid_ids:
            log.warning(
                "confidence_clamped",
                original=confidence,
                clamped_to=0.5,
                reason="no valid references after stripping hallucinations",
            )
            confidence = 0.5

        return suggestion.model_copy(
            update={
                "referenced_incident_ids": valid_ids,
                "confidence": confidence,
            },
        )

    # -- internals ----------------------------------------------------------

    async def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        attempt: int,
    ) -> FixSuggestion:
        """Execute one Anthropic Messages API call and parse the result."""
        start = time.monotonic()

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        elapsed = time.monotonic() - start
        elapsed_ms = elapsed * 1_000
        self.last_latency_ms = elapsed_ms
        input_tokens = response.usage.input_tokens or 0
        output_tokens = response.usage.output_tokens or 0
        self.last_tokens_used = input_tokens + output_tokens

        LLM_REQUEST_LATENCY_SECONDS.labels(model=self._model).observe(elapsed)
        LLM_TOKENS_TOTAL.labels(model=self._model, type="input").inc(input_tokens)
        LLM_TOKENS_TOTAL.labels(model=self._model, type="output").inc(output_tokens)

        log.info(
            "llm_call_completed",
            attempt=attempt,
            tokens_used=self.last_tokens_used,
            latency_ms=round(elapsed_ms, 1),
        )

        raw_text = self._extract_text(response)
        data = self._parse_json(raw_text)
        return FixSuggestion.model_validate(data)

    @staticmethod
    def _extract_text(response: Any) -> str:  # noqa: ANN401
        """Pull the text content from the first TextBlock."""
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        raise ValueError("No text content in Claude response")

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Parse JSON from the model output, tolerating markdown fences."""
        cleaned = text.strip()
        # Strip optional ```json ... ``` fences
        if cleaned.startswith("```"):
            first_newline = cleaned.index("\n")
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[: -3]
        return json.loads(cleaned.strip())
