"""Dead-letter queue (DLQ) backed by a dedicated Redis Stream.

Failed messages are moved here with the original payload, the failure reason,
and a retry counter.  An inspection helper exposes the DLQ contents for
dashboards or admin endpoints.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as aioredis
import structlog

from aiops.config import get_settings

logger = structlog.get_logger()


class DeadLetterQueue:
    """Manages a Redis Stream that stores messages that could not be processed.

    Parameters
    ----------
    redis_client:
        An ``aioredis.Redis`` connection.
    dlq_stream:
        Stream key for the DLQ (defaults to ``settings.redis_dlq_stream``).
    max_retries:
        Maximum number of retries before a message is considered permanently
        failed (defaults to ``settings.redis_max_retries``).
    """

    def __init__(
        self,
        redis_client: aioredis.Redis,
        dlq_stream: str | None = None,
        max_retries: int | None = None,
    ) -> None:
        settings = get_settings()
        self._redis = redis_client
        self._stream = dlq_stream or settings.redis_dlq_stream
        self._max_retries = max_retries if max_retries is not None else settings.redis_max_retries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def move_to_dlq(
        self,
        original_msg_id: str,
        original_payload: dict[str, Any],
        reason: str,
        retry_count: int = 0,
        source_stream: str | None = None,
    ) -> str:
        """Add a failed message to the DLQ stream.

        Parameters
        ----------
        original_msg_id:
            Stream message ID from the source stream.
        original_payload:
            The raw message fields.
        reason:
            Human-readable failure description.
        retry_count:
            How many times the message has already been retried.
        source_stream:
            Name of the source stream (informational, defaults to config).

        Returns
        -------
        str
            The DLQ stream message ID.
        """
        settings = get_settings()
        dlq_entry: dict[str, str] = {
            "original_msg_id": original_msg_id,
            "payload": json.dumps(original_payload, default=str),
            "reason": reason,
            "retry_count": str(retry_count),
            "source_stream": source_stream or settings.redis_stream_name,
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }

        msg_id: bytes = await self._redis.xadd(self._stream, dlq_entry)
        decoded = msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id)
        logger.warning(
            "dlq_entry_added",
            dlq_msg_id=decoded,
            original_msg_id=original_msg_id,
            reason=reason,
            retry_count=retry_count,
        )
        return decoded

    async def should_retry(self, retry_count: int) -> bool:
        """Return ``True`` if the message has not yet exceeded retry limits."""
        return retry_count < self._max_retries

    async def get_dlq_entries(
        self,
        count: int = 100,
        start: str = "-",
        end: str = "+",
    ) -> list[dict[str, Any]]:
        """Read DLQ entries for inspection.

        Parameters
        ----------
        count:
            Maximum number of entries to return.
        start:
            Minimum stream ID (``-`` for earliest).
        end:
            Maximum stream ID (``+`` for latest).

        Returns
        -------
        list[dict]
            Each dict contains decoded fields from the DLQ stream.
        """
        raw: list[Any] = await self._redis.xrange(self._stream, min=start, max=end, count=count)
        entries: list[dict[str, Any]] = []
        for msg_id, fields in raw:
            decoded_id = msg_id.decode() if isinstance(msg_id, bytes) else msg_id
            decoded_fields: dict[str, Any] = {}
            for k, v in fields.items():
                key = k.decode() if isinstance(k, bytes) else k
                val = v.decode() if isinstance(v, bytes) else v
                decoded_fields[key] = val
            decoded_fields["dlq_msg_id"] = decoded_id
            # Attempt to parse retry_count back to int
            if "retry_count" in decoded_fields:
                try:
                    decoded_fields["retry_count"] = int(decoded_fields["retry_count"])
                except (ValueError, TypeError):
                    pass
            entries.append(decoded_fields)

        return entries

    async def dlq_length(self) -> int:
        """Return the current number of entries in the DLQ stream."""
        length: int = await self._redis.xlen(self._stream)
        return length
