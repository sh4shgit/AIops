"""Health-check endpoint — ``GET /health``.

Reports connectivity to Redis and the configured vector store.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Request

from aiops.api.schemas import HealthResponse
from aiops.config import get_settings

logger = structlog.get_logger()

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Return service health including Redis and vector-store status."""
    redis_ok = await _check_redis(request)
    vector_ok = await _check_vector_store()

    status = "healthy" if (redis_ok and vector_ok) else "degraded"

    return HealthResponse(
        status=status,
        redis_ok=redis_ok,
        vector_store_ok=vector_ok,
        timestamp=datetime.now(timezone.utc),
    )


async def _check_redis(request: Request) -> bool:
    """Attempt a Redis PING with timeout."""
    try:
        redis_client = getattr(request.app.state, "redis", None)
        if redis_client is None:
            return False
        pong = await asyncio.wait_for(redis_client.ping(), timeout=1.0)
        return bool(pong)
    except Exception:
        logger.warning("health_redis_unreachable", exc_info=True)
        return False


async def _check_vector_store() -> bool:
    """Check vector store reachability / persistence."""
    settings = get_settings()
    if settings.vector_store_backend.value != "chromadb":
        return True
    try:
        import chromadb  # noqa: PLC0415

        # Check local persistent directory or heartbeat
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        client.heartbeat()
        return True
    except Exception:
        logger.warning("health_vector_store_unreachable", exc_info=True)
        return False
