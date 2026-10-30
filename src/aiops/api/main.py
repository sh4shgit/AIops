"""FastAPI application factory for the AIOps Assistant.

Call :func:`create_app` to get a fully-configured ``FastAPI`` instance with:

* Redis connection managed via the ASGI lifespan
* CORS middleware
* API-key authentication dependency on mutating endpoints
* Sliding-window rate-limiting middleware for ``POST /logs``
* All route routers included
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import AsyncIterator

import redis.asyncio as aioredis
import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response

from aiops.api.routes import health, incidents, ingest, metrics
from aiops.api.routes.incidents import InMemoryIncidentStore
from aiops.config import get_settings
from aiops.observability.logging import setup_logging

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# API-Key dependency
# ---------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(
    api_key: str | None = Security(_api_key_header),
) -> str:
    """Validate the ``X-API-Key`` header against the configured secret.

    Used as a dependency for mutating endpoints.
    """
    settings = get_settings()
    expected = settings.api_key.get_secret_value()
    if api_key is None or api_key != expected:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return api_key


# ---------------------------------------------------------------------------
# Rate-limiting middleware (in-memory sliding window)
# ---------------------------------------------------------------------------


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple per-client sliding-window rate limiter for ``POST /logs``.

    Clients are identified by their originating IP address.  The window
    size is always 60 s and the maximum number of requests is read from
    ``settings.rate_limit_per_minute``.
    """

    def __init__(self, app: FastAPI, rate_limit_per_minute: int) -> None:  # type: ignore[override]
        super().__init__(app)
        self._limit = rate_limit_per_minute
        # client_ip -> list of request timestamps within the current window
        self._windows: dict[str, list[float]] = defaultdict(list)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.method == "POST" and request.url.path == "/logs":
            client_ip = request.client.host if request.client else "unknown"
            now = time.monotonic()
            window_start = now - 60.0

            # Prune old timestamps
            timestamps = self._windows[client_ip]
            self._windows[client_ip] = [t for t in timestamps if t > window_start]

            if len(self._windows[client_ip]) >= self._limit:
                logger.warning("rate_limit_exceeded", client=client_ip)
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Try again later."},
                )

            self._windows[client_ip].append(now)

        return await call_next(request)


# ---------------------------------------------------------------------------
# Lifespan — manages Redis connection
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application-level resources (Redis, stores)."""
    settings = get_settings()

    # --- Redis ---
    redis_client = aioredis.from_url(
        settings.redis_url,
        decode_responses=False,
    )
    app.state.redis = redis_client
    logger.info("redis_connected", url=settings.redis_url)

    # --- Incident store ---
    app.state.incident_store = InMemoryIncidentStore()

    try:
        yield
    finally:
        await redis_client.aclose()
        logger.info("redis_disconnected")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Build and return the fully-configured FastAPI application."""
    setup_logging()
    settings = get_settings()

    app = FastAPI(
        title="AIOps Assistant",
        version="0.1.0",
        description="AI-powered alert noise reduction for DevOps / SRE teams",
        lifespan=_lifespan,
    )

    # --- CORS ---------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Rate-limiting middleware -------------------------------------------
    app.add_middleware(RateLimitMiddleware, rate_limit_per_minute=settings.rate_limit_per_minute)

    # --- Routes (public) ---------------------------------------------------
    app.include_router(health.router)
    app.include_router(metrics.router)

    # --- Routes (authenticated mutating endpoints) --------------------------
    # Ingest uses the API-key dependency on all routes in its router
    app.include_router(
        ingest.router,
        dependencies=[Depends(require_api_key)],
    )

    # Incidents: read-only router is open; the mutations router requires the
    # API key.  See routes/incidents.py for the two-router split.
    app.include_router(incidents.read_router)
    app.include_router(
        incidents.write_router,
        dependencies=[Depends(require_api_key)],
    )

    return app
