"""FastAPI application factory for the AIOps Assistant.

Call :func:`create_app` to get a fully-configured ``FastAPI`` instance with:

* Redis connection managed via the ASGI lifespan
* CORS middleware
* API-key authentication dependency on mutating endpoints
* Sliding-window rate-limiting middleware for ``POST /logs``
* All route routers included
* Static web UI served at ``/ui``
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import redis.asyncio as aioredis
import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response

from aiops.api.routes import health, incidents, ingest, metrics
from aiops.api.routes.analyze import router as analyze_router
from aiops.api.routes.datasets import router as datasets_router
from aiops.api.routes.incidents import InMemoryIncidentStore
from aiops.config import get_settings
from aiops.observability.logging import setup_logging

# Path to the web/ directory (two levels up from this file → repo root / web)
_WEB_DIR = Path(__file__).resolve().parent.parent.parent.parent / "web"

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
    """Simple per-client sliding-window rate limiter for ingestion and model requests.

    Clients are identified by their originating IP address.  The window
    size is always 60 s and the maximum number of requests is read from
    ``settings.rate_limit_per_minute``.
    """

    def __init__(self, app: FastAPI, rate_limit_per_minute: int) -> None:  # type: ignore[override]
        super().__init__(app)
        self._limit = rate_limit_per_minute
        # client_ip -> list of request timestamps within the current window
        self._windows: dict[tuple[str, str], list[float]] = defaultdict(list)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.method == "POST" and request.url.path in {"/logs", "/analyze/recommendation"}:
            client_ip = request.client.host if request.client else "unknown"
            bucket = (client_ip, request.url.path)
            now = time.monotonic()
            window_start = now - 60.0

            # Prune old timestamps
            timestamps = self._windows[bucket]
            self._windows[bucket] = [t for t in timestamps if t > window_start]

            if len(self._windows[bucket]) >= self._limit:
                logger.warning("rate_limit_exceeded", client=client_ip)
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Try again later."},
                )

            self._windows[bucket].append(now)

        return await call_next(request)


# ---------------------------------------------------------------------------
# Lifespan — manages Redis connection
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application-level resources (Redis, stores)."""
    settings = get_settings()

    # --- Redis ---
    existing_redis = getattr(app.state, "redis", None)
    own_redis = False
    if existing_redis is None:
        redis_client = aioredis.from_url(
            settings.redis_url,
            decode_responses=False,
            socket_timeout=1.0,
            socket_connect_timeout=1.0,
        )
        app.state.redis = redis_client
        own_redis = True
        logger.info("redis_connected", url=settings.redis_url)
    else:
        redis_client = existing_redis

    # --- Incident store ---
    if not hasattr(app.state, "incident_store"):
        app.state.incident_store = InMemoryIncidentStore()

    try:
        yield
    finally:
        if own_redis and redis_client is not None:
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

    # --- Analyze endpoint (public — no auth needed for demo UI) ------------
    app.include_router(analyze_router)
    app.include_router(datasets_router)

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

    # --- Web UI -------------------------------------------------------------
    # Serve index.html at /ui and /
    @app.get("/ui", include_in_schema=False)
    @app.get("/", include_in_schema=False)
    def serve_ui() -> FileResponse:
        """Serve the web dashboard."""
        from fastapi.responses import FileResponse
        ui_path = _WEB_DIR / "index.html"
        if ui_path.exists():
            return FileResponse(str(ui_path), media_type="text/html")
        return FileResponse.__new__(FileResponse)  # fallback handled below

    # Mount static assets (CSS, JS, images if any)
    if _WEB_DIR.exists():
        app.mount("/web", StaticFiles(directory=str(_WEB_DIR)), name="web")

    return app
