"""Redis Streams producer and consumer for the log ingestion pipeline.

The producer serialises ``ParsedLog`` entries into a Redis Stream.
The consumer reads from a consumer group, invokes a user-supplied callback,
and acknowledges processed messages.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import redis.asyncio as aioredis
import structlog

from aiops.api.schemas import ParsedLog
from aiops.config import get_settings

logger = structlog.get_logger()


class RedisStreamProducer:
    """Writes ``ParsedLog`` entries to a Redis Stream."""

    def __init__(self, redis_client: aioredis.Redis, stream_name: str | None = None) -> None:
        self._redis = redis_client
        self._stream = stream_name or get_settings().redis_stream_name

    async def publish(self, entry: ParsedLog) -> str:
        """Publish a single parsed log entry and return the stream message ID.

        The entry is serialised to JSON and stored under the ``data`` field.
        """
        payload = entry.model_dump_json()
        msg_id: bytes = await self._redis.xadd(self._stream, {"data": payload})
        decoded_id = msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id)
        logger.debug("stream_published", log_id=entry.log_id, stream_msg_id=decoded_id)
        return decoded_id

    async def publish_many(self, entries: list[ParsedLog]) -> list[str]:
        """Publish a batch of entries, returning their stream message IDs."""
        ids: list[str] = []
        async with self._redis.pipeline(transaction=False) as pipe:
            for entry in entries:
                pipe.xadd(self._stream, {"data": entry.model_dump_json()})
            results = await pipe.execute()
        for result in results:
            decoded = result.decode() if isinstance(result, bytes) else str(result)
            ids.append(decoded)
        logger.info("stream_published_batch", count=len(ids))
        return ids


class RedisStreamConsumer:
    """Consumes messages from a Redis Stream consumer group.

    Parameters
    ----------
    redis_client:
        An ``aioredis.Redis`` connection.
    callback:
        Async function invoked for each successfully-decoded ``ParsedLog``.
    stream_name:
        Redis stream key (defaults to config).
    group_name:
        Consumer group name (defaults to config).
    consumer_name:
        Unique consumer name inside the group.
    batch_size:
        Number of messages to read per ``XREADGROUP`` call.
    block_ms:
        Milliseconds to block when no messages are available.
    """

    def __init__(
        self,
        redis_client: aioredis.Redis,
        callback: Callable[[ParsedLog], Awaitable[None]],
        stream_name: str | None = None,
        group_name: str | None = None,
        consumer_name: str = "worker-1",
        batch_size: int = 50,
        block_ms: int = 2000,
    ) -> None:
        settings = get_settings()
        self._redis = redis_client
        self._callback = callback
        self._stream = stream_name or settings.redis_stream_name
        self._group = group_name or settings.redis_consumer_group
        self._consumer = consumer_name
        self._batch_size = batch_size
        self._block_ms = block_ms
        self._running = False

    async def ensure_group(self) -> None:
        """Create the consumer group if it does not already exist."""
        try:
            await self._redis.xgroup_create(self._stream, self._group, id="0", mkstream=True)
            logger.info(
                "consumer_group_created", stream=self._stream, group=self._group
            )
        except aioredis.ResponseError as exc:
            if "BUSYGROUP" in str(exc):
                logger.debug("consumer_group_exists", stream=self._stream, group=self._group)
            else:
                raise

    async def run(self) -> None:
        """Start the consume loop.  Call :meth:`stop` to exit gracefully."""
        await self.ensure_group()
        self._running = True
        logger.info(
            "consumer_started",
            stream=self._stream,
            group=self._group,
            consumer=self._consumer,
        )

        while self._running:
            try:
                messages: list[Any] = await self._redis.xreadgroup(
                    groupname=self._group,
                    consumername=self._consumer,
                    streams={self._stream: ">"},
                    count=self._batch_size,
                    block=self._block_ms,
                )
            except asyncio.CancelledError:
                logger.info("consumer_cancelled")
                break
            except Exception:
                logger.exception("consumer_read_error")
                await asyncio.sleep(1)
                continue

            if not messages:
                continue

            for _stream_key, entries in messages:
                for msg_id, fields in entries:
                    await self._process_message(msg_id, fields)

    async def _process_message(
        self, msg_id: bytes | str, fields: dict[bytes | str, bytes | str]
    ) -> None:
        """Decode, call the callback, and acknowledge the message."""
        decoded_id = msg_id.decode() if isinstance(msg_id, bytes) else msg_id
        try:
            raw = fields.get(b"data") or fields.get("data")
            if raw is None:
                logger.warning("consumer_missing_data_field", msg_id=decoded_id)
                await self._redis.xack(self._stream, self._group, msg_id)
                return

            data_str = raw.decode() if isinstance(raw, bytes) else raw
            parsed = ParsedLog.model_validate_json(data_str)
            await self._callback(parsed)
            await self._redis.xack(self._stream, self._group, msg_id)
            logger.debug("consumer_ack", msg_id=decoded_id, log_id=parsed.log_id)

        except Exception:
            logger.exception("consumer_processing_error", msg_id=decoded_id)
            # Not acknowledged — will be retried or moved to DLQ.

    def stop(self) -> None:
        """Signal the consumer to stop after the current iteration."""
        self._running = False
        logger.info("consumer_stop_requested")
