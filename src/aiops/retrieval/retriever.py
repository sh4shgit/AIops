"""Retrieval orchestration — query builder and incident retriever.

:class:`IncidentRetriever` translates an :class:`AnomalyEvent` into a
natural-language query, calls the vector store, and returns ranked
:class:`RetrievalResult` objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from aiops.config import get_settings
from aiops.detection.thresholds import AnomalyEvent
from aiops.retrieval.store import RetrievalResult

if TYPE_CHECKING:
    from aiops.retrieval.store import VectorStoreProtocol

log = structlog.get_logger()


class IncidentRetriever:
    """Build queries from anomaly events and retrieve matching incidents.

    Parameters
    ----------
    store:
        The vector-store backend to query against.
    top_k:
        Maximum number of results to return per query.  Defaults to the
        value in ``Settings.retrieval_top_k``.
    """

    def __init__(
        self,
        store: VectorStoreProtocol,
        top_k: int | None = None,
    ) -> None:
        self._store = store
        self._top_k = top_k or get_settings().retrieval_top_k

    # -- public API -----------------------------------------------------------

    def build_query(self, anomaly_event: AnomalyEvent) -> str:
        """Construct a natural-language query from an :class:`AnomalyEvent`.

        The query concatenates available contextual signals so the embedding
        model can match it against incident symptoms, root causes, and
        resolutions stored in the vector store.

        Parameters
        ----------
        anomaly_event:
            The anomaly event to transform into a query.

        Returns
        -------
        str
            A human-readable query string.
        """
        parts: list[str] = []

        # The existing AnomalyEvent uses `source` for the service name.
        if anomaly_event.source:
            parts.append(f"Service: {anomaly_event.source}")

        # Severity band.
        parts.append(f"Severity: {anomaly_event.band.value}")

        # Raw log line (may contain the most useful textual signal).
        if anomaly_event.raw_log:
            parts.append(f"Log: {anomaly_event.raw_log}")

        # Extra metadata (e.g. namespace, pod, region).
        if anomaly_event.metadata:
            ctx_str = ", ".join(
                f"{k}={v}" for k, v in anomaly_event.metadata.items()
            )
            parts.append(f"Context: {ctx_str}")

        # Anomaly score as a textual cue.
        parts.append(f"Anomaly score: {anomaly_event.anomaly_score:.4f}")

        query = " | ".join(parts)
        log.debug(
            "retriever.build_query",
            event_id=anomaly_event.event_id,
            query=query,
        )
        return query

    def retrieve(self, anomaly_event: AnomalyEvent) -> list[RetrievalResult]:
        """Retrieve matching incidents for a given anomaly event.

        Builds a query from the event, applies metadata filters (service,
        excluding superseded incidents), and returns results sorted by
        similarity.

        Parameters
        ----------
        anomaly_event:
            The detected anomaly to find historical incidents for.

        Returns
        -------
        list[RetrievalResult]
            Up to ``top_k`` results, highest similarity first.
        """
        query = self.build_query(anomaly_event)

        # Build filters — always exclude superseded incidents.
        filters: dict[str, object] = {"include_superseded": False}

        # Restrict by service when the event provides one.
        if anomaly_event.source:
            filters["service"] = anomaly_event.source

        log.info(
            "retriever.retrieve",
            event_id=anomaly_event.event_id,
            top_k=self._top_k,
            filters=filters,
        )

        results = self._store.query(
            query_text=query,
            top_k=self._top_k,
            filters=filters,  # type: ignore[arg-type]
        )

        # Ensure descending similarity order (store should already do this).
        results.sort(key=lambda r: r.similarity_score, reverse=True)

        log.info(
            "retriever.results",
            event_id=anomaly_event.event_id,
            hits=len(results),
            top_score=results[0].similarity_score if results else None,
        )
        return results
