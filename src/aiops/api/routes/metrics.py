"""Prometheus metrics endpoint — ``GET /metrics``.

Exposes the default ``prometheus_client`` registry in the standard
Prometheus text-exposition format.
"""

from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter(tags=["metrics"])


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Serve Prometheus metrics in text-exposition format."""
    body = generate_latest()
    return Response(content=body, media_type=CONTENT_TYPE_LATEST)
