"""Log ingestion endpoint — ``POST /logs``.

Accepts a JSON array of log entries, validates each one independently,
deduplicates via a Redis SET, publishes valid entries to a Redis Stream,
and returns a summary with per-entry error details.
"""

from __future__ import annotations

from typing import Any

import redis.asyncio as aioredis
import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from aiops.api.schemas import IngestError, IngestResponse, ParsedLog
from aiops.ingestion.queue import RedisStreamProducer
from aiops.ingestion.validator import validate_entry
from aiops.observability.metrics import INGESTION_BATCH_SIZE, LOGS_INGESTED_TOTAL

logger = structlog.get_logger()

router = APIRouter(tags=["ingestion"])

DEDUP_SET_KEY = "logs:seen_ids"
DEDUP_TTL_SECONDS = 60 * 60 * 24  # 24 h


def _get_redis(request: Request) -> aioredis.Redis:
    """Extract the Redis client stashed on ``app.state`` by the lifespan."""
    return request.app.state.redis  # type: ignore[return-value]


@router.post("/logs", response_model=IngestResponse, status_code=202)
async def ingest_logs(
    body: list[dict[str, Any]],
    request: Request,
    redis_client: aioredis.Redis = Depends(_get_redis),
) -> IngestResponse:
    """Ingest a batch of log entries.

    Each entry is validated individually — a single invalid entry does **not**
    reject the whole batch.  Duplicate entries (same deterministic ID already
    seen) are silently skipped.
    """
    if not body:
        raise HTTPException(status_code=422, detail="Batch must not be empty")

    INGESTION_BATCH_SIZE.observe(len(body))

    accepted = 0
    rejected = 0
    errors: list[IngestError] = []
    to_publish: list[ParsedLog] = []

    # --- per-entry validation -------------------------------------------
    for idx, raw in enumerate(body):
        parsed, error = validate_entry(raw, idx)
        if error is not None:
            rejected += 1
            errors.append(error)
            service_name = str(raw.get("service", "unknown"))
            LOGS_INGESTED_TOTAL.labels(service=service_name, status="rejected").inc()
            continue
        assert parsed is not None  # noqa: S101
        to_publish.append(parsed)

    # --- deduplication via Redis SET ------------------------------------
    deduplicated: list[ParsedLog] = []
    if to_publish:
        async with redis_client.pipeline(transaction=False) as pipe:
            for p in to_publish:
                pipe.sadd(DEDUP_SET_KEY, p.log_id)
            results = await pipe.execute()

        for p, was_new in zip(to_publish, results, strict=True):
            if was_new:
                deduplicated.append(p)
                LOGS_INGESTED_TOTAL.labels(service=p.service, status="accepted").inc()
            else:
                LOGS_INGESTED_TOTAL.labels(service=p.service, status="duplicate").inc()
                logger.debug("duplicate_log_skipped", log_id=p.log_id)

        # Ensure the dedup set expires eventually
        await redis_client.expire(DEDUP_SET_KEY, DEDUP_TTL_SECONDS)

    # --- publish to Redis Stream ----------------------------------------
    if deduplicated:
        producer = RedisStreamProducer(redis_client)
        try:
            await producer.publish_many(deduplicated)
        except Exception:
            logger.exception("stream_publish_failed", count=len(deduplicated))
            raise HTTPException(
                status_code=503, detail="Failed to publish logs to stream"
            ) from None

    accepted = len(deduplicated)

    logger.info(
        "ingest_complete",
        accepted=accepted,
        rejected=rejected,
        duplicates=len(to_publish) - len(deduplicated),
    )

    return IngestResponse(accepted=accepted, rejected=rejected, errors=errors)
