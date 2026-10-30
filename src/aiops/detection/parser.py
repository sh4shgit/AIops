"""Drain3-based log template mining.

Wraps the Drain3 ``TemplateMiner`` with an ergonomic interface that returns
structured ``ParsedTemplate`` objects ready for downstream feature extraction.
"""

from __future__ import annotations

import dataclasses
import re
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    pass

logger = structlog.get_logger()

# Drain3 uses a masking approach — parameters replaced by ``<*>`` in templates.
_PARAM_PLACEHOLDER = "<*>"
_MAX_MESSAGE_LENGTH = 10_000  # guard against absurdly long lines


@dataclasses.dataclass(frozen=True, slots=True)
class ParsedTemplate:
    """Result of parsing a single log message with Drain3.

    Attributes:
        template_id: Opaque cluster identifier assigned by Drain3.
        template: The generalised log template with ``<*>`` placeholders.
        parameters: Concrete values extracted from the message that matched
            the ``<*>`` slots in the template.
    """

    template_id: str
    template: str
    parameters: list[str]


class LogParser:
    """Thin wrapper around Drain3's ``TemplateMiner``.

    Uses a purely in-memory persistence so the parser works without any
    filesystem side-effects — ideal for testing and ephemeral deployments.

    Example::

        parser = LogParser()
        result = parser.parse("Connection to 10.0.0.1 timed out after 30s")
        # result.template  → "Connection to <*> timed out after <*>"
        # result.parameters → ["10.0.0.1", "30s"]
    """

    def __init__(self) -> None:
        # Lazy-import so the module stays importable even when drain3 is
        # missing (e.g. during type-checking or lightweight scripting).
        from drain3 import TemplateMiner
        from drain3.template_miner_config import TemplateMinerConfig

        config = TemplateMinerConfig()
        # Sensible Drain3 defaults — snapshot_interval_minutes=0 disables
        # periodic disk snapshots (we keep everything in RAM).
        config.profiling_enabled = False
        config.snapshot_interval_minutes = 0

        self._miner = TemplateMiner(config=config)
        logger.info("log_parser.initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, message: str) -> ParsedTemplate:
        """Parse a raw log *message* and return its template + parameters.

        Args:
            message: The raw log line to parse.

        Returns:
            A ``ParsedTemplate`` with the cluster id, generalised template
            string, and extracted parameter values.

        Edge-case handling:
            * Empty / whitespace-only messages yield a fixed sentinel template.
            * Messages longer than ``_MAX_MESSAGE_LENGTH`` are truncated before
              being handed to Drain3 to avoid excessive memory usage.
        """
        if not message or not message.strip():
            logger.debug("log_parser.empty_message")
            return ParsedTemplate(
                template_id="0",
                template="<EMPTY>",
                parameters=[],
            )

        # Guard against extremely long lines (e.g. base64 blobs, stack dumps).
        truncated = message[:_MAX_MESSAGE_LENGTH] if len(message) > _MAX_MESSAGE_LENGTH else message

        result = self._miner.add_log_message(truncated)

        cluster_id: str = str(result["cluster_id"])
        template: str = result["template_mined"]

        parameters = self._extract_parameters(template, truncated)

        logger.debug(
            "log_parser.parsed",
            cluster_id=cluster_id,
            template=template,
            n_params=len(parameters),
        )

        return ParsedTemplate(
            template_id=cluster_id,
            template=template,
            parameters=parameters,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_parameters(template: str, message: str) -> list[str]:
        """Extract concrete parameter values by diffing *template* vs *message*.

        The approach splits both strings on whitespace and compares token-by-token;
        tokens that correspond to ``<*>`` in the template are collected as
        parameters.  This is a fast heuristic that works well for the
        space-delimited log lines Drain3 is designed for.
        """
        template_tokens = template.split()
        message_tokens = message.split()

        params: list[str] = []
        for t_tok, m_tok in zip(template_tokens, message_tokens):
            if t_tok == _PARAM_PLACEHOLDER:
                params.append(m_tok)

        # If the message has trailing tokens beyond the template length
        # (rare edge case with Drain3), capture them too.
        if len(message_tokens) > len(template_tokens):
            trailing = template_tokens[-1] if template_tokens else ""
            if trailing == _PARAM_PLACEHOLDER:
                params.extend(message_tokens[len(template_tokens):])

        return params

    @property
    def cluster_count(self) -> int:
        """Return the number of unique template clusters discovered so far."""
        return len(self._miner.drain.clusters)
