"""Structured logging configuration using structlog.

Provides JSON formatted logs in production with correlation IDs, timestamps,
and service context, and pretty console logs during local development.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from aiops.config import get_settings


def setup_logging() -> None:
    """Initialize structured logging based on application settings."""
    settings = get_settings()

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.log_format == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    # Silence overly verbose third-party loggers
    for quiet_logger in ("uvicorn.access", "chromadb", "httpx", "httpcore"):
        logging.getLogger(quiet_logger).setLevel(logging.WARNING)


def bind_correlation_id(correlation_id: str, **kwargs: Any) -> None:
    """Bind a request or incident correlation ID to the current context."""
    structlog.contextvars.bind_contextvars(
        correlation_id=correlation_id,
        **kwargs,
    )


def clear_correlation_id() -> None:
    """Clear all context variables for the current thread/task."""
    structlog.contextvars.clear_contextvars()
