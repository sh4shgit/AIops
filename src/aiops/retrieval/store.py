"""Vector store interface and ChromaDB implementation.

Defines :class:`VectorStoreProtocol` — the pluggable abstraction consumed by
the retriever — and :class:`ChromaVectorStore`, a concrete implementation
backed by `chromadb <https://docs.trychroma.com>`_ with local persistence.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

try:
    import chromadb
except ImportError:
    chromadb = None

import structlog
from pydantic import BaseModel, Field

from aiops.config import get_settings
from aiops.retrieval.embeddings import EmbeddingService
from aiops.retrieval.ingest_incidents import IncidentRecord

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class RetrievalResult(BaseModel):
    """A single retrieval hit returned by the vector store."""

    incident_id: str = Field(..., description="Originating incident identifier.")
    chunk_type: str = Field(
        ...,
        description="Section type: 'symptoms', 'resolution', or 'root_cause'.",
    )
    similarity_score: float = Field(
        ...,
        description="Cosine similarity score (higher is better).",
    )
    content: str = Field(..., description="Chunk text that was matched.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Associated metadata (service, status, etc.).",
    )

    @property
    def title(self) -> str:
        return str(self.metadata.get("title", f"Incident {self.incident_id}"))

    @property
    def score(self) -> float:
        return self.similarity_score


# ---------------------------------------------------------------------------
# Protocol (interface)
# ---------------------------------------------------------------------------


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Pluggable interface for incident vector stores.

    Any concrete implementation must satisfy this protocol so that it can be
    swapped out without touching consumer code.
    """

    def add_incidents(self, incidents: list[IncidentRecord]) -> None:
        """Insert or upsert a batch of incident records."""
        ...

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve the most similar chunks for *query_text*."""
        ...

    def update_incident(self, incident_id: str, incident: IncidentRecord) -> None:
        """Replace all chunks for *incident_id* with the new record."""
        ...

    def delete_incident(self, incident_id: str) -> None:
        """Remove all chunks belonging to *incident_id*."""
        ...

    def health_check(self) -> bool:
        """Return ``True`` if the store is operational."""
        ...


# ---------------------------------------------------------------------------
# ChromaDB implementation
# ---------------------------------------------------------------------------

# Chunk types we split each incident into.
_CHUNK_TYPES: list[str] = ["symptoms", "resolution", "root_cause"]


def _chunk_id(incident_id: str, chunk_type: str) -> str:
    """Deterministic document-id for a given incident chunk."""
    return f"{incident_id}::{chunk_type}"


class ChromaVectorStore:
    """ChromaDB-backed vector store implementing :class:`VectorStoreProtocol`.

    Each incident is split into three separately retrievable chunks
    (symptoms, resolution, root_cause).  Metadata filters allow callers to
    restrict results by service or exclude superseded incidents.

    Parameters
    ----------
    embedding_service:
        The embedding wrapper to use.  When ``None`` a default
        :class:`~aiops.retrieval.embeddings.EmbeddingService` is created.
    persist_directory:
        Filesystem path for ChromaDB persistence.  Defaults to the value in
        ``Settings.chroma_persist_dir``.
    collection_name:
        Name of the ChromaDB collection.  Defaults to
        ``Settings.chroma_collection``.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        persist_directory: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        settings = get_settings()
        self._embedding = embedding_service or EmbeddingService()
        self._persist_dir = persist_directory or settings.chroma_persist_dir
        self._collection_name = collection_name or settings.chroma_collection
        self._mem_chunks: dict[str, dict[str, Any]] = {}

        log.info(
            "chroma.init",
            persist_dir=self._persist_dir,
            collection=self._collection_name,
        )

        if chromadb is not None:
            try:
                self._client = chromadb.PersistentClient(path=self._persist_dir)
                self._collection = self._client.get_or_create_collection(
                    name=self._collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception:
                log.warning("chroma.init_failed_fallback_mem", exc_info=True)
                self._client = None
                self._collection = None
        else:
            self._client = None
            self._collection = None

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _build_metadata(incident: IncidentRecord, chunk_type: str) -> dict[str, Any]:
        """Build a flat metadata dict suitable for ChromaDB storage."""
        meta: dict[str, Any] = {
            "incident_id": incident.id,
            "chunk_type": chunk_type,
            "status": incident.status,
            "services": ",".join(incident.services) if incident.services else "",
        }
        if incident.resolved_at is not None:
            meta["resolved_at"] = incident.resolved_at.isoformat()
        return meta

    @staticmethod
    def _chunk_content(incident: IncidentRecord, chunk_type: str) -> str:
        """Extract the textual content for a given chunk type."""
        if chunk_type == "symptoms":
            parts = [incident.symptoms]
            if incident.log_templates:
                parts.append("Log templates: " + " | ".join(incident.log_templates))
            return "\n".join(parts)
        if chunk_type == "resolution":
            return incident.resolution
        if chunk_type == "root_cause":
            return incident.root_cause
        raise ValueError(f"Unknown chunk_type: {chunk_type}")  # pragma: no cover

    # -- protocol methods -----------------------------------------------------

    def add_incidents(self, incidents: list[IncidentRecord]) -> None:
        """Upsert all chunks for each incident.

        Parameters
        ----------
        incidents:
            Validated incident records to store.
        """
        ids: list[str] = []
        documents: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, Any]] = []

        for inc in incidents:
            for ct in _CHUNK_TYPES:
                content = self._chunk_content(inc, ct)
                ids.append(_chunk_id(inc.id, ct))
                documents.append(content)
                metadatas.append(self._build_metadata(inc, ct))

        if not documents:
            return

        embeddings = self._embedding.embed_batch(documents)

        if self._collection is not None:
            self._collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,  # type: ignore[arg-type]
                metadatas=metadatas,  # type: ignore[arg-type]
            )
        else:
            for cid, doc, emb, meta in zip(ids, documents, embeddings, metadatas):
                self._mem_chunks[cid] = {
                    "document": doc,
                    "embedding": emb,
                    "metadata": meta,
                }
        log.info("chroma.add_incidents", count=len(incidents), chunks=len(ids))

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve the most relevant chunks for *query_text*.

        By default, superseded incidents are excluded.  Pass
        ``filters={"include_superseded": True}`` to override this behaviour.

        Parameters
        ----------
        query_text:
            Natural-language query string.
        top_k:
            Maximum number of results to return.
        filters:
            Optional filter dict.  Recognised keys:

            * ``service`` — restrict to incidents involving this service.
            * ``include_superseded`` — if ``True``, include superseded records.

        Returns
        -------
        list[RetrievalResult]
            Results sorted by descending similarity score.
        """
        filters = filters or {}

        # Build ChromaDB where-clause.
        where_clauses: list[dict[str, Any]] = []

        if not filters.get("include_superseded", False):
            where_clauses.append({"status": {"$eq": "active"}})

        service: str | None = filters.get("service")
        if service:
            # Services stored as comma-separated string; use $contains.
            where_clauses.append({"services": {"$contains": service}})

        where: dict[str, Any] | None = None
        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}

        query_embedding = self._embedding.embed(query_text)

        if self._collection is not None:
            try:
                results = self._collection.query(
                    query_embeddings=[query_embedding],  # type: ignore[arg-type]
                    n_results=top_k,
                    where=where,  # type: ignore[arg-type]
                    include=["documents", "metadatas", "distances"],
                )
            except Exception:
                log.error("chroma.query.failed", exc_info=True)
                return []

            hits: list[RetrievalResult] = []
            if not results or not results.get("ids"):
                return hits

            for doc, meta, distance in zip(
                results["documents"][0],  # type: ignore[index]
                results["metadatas"][0],  # type: ignore[index]
                results["distances"][0],  # type: ignore[index]
            ):
                similarity = 1.0 - distance
                hits.append(
                    RetrievalResult(
                        incident_id=meta.get("incident_id", ""),
                        chunk_type=meta.get("chunk_type", ""),
                        similarity_score=round(similarity, 6),
                        content=doc,
                        metadata={k: v for k, v in meta.items() if k not in ("incident_id", "chunk_type")},
                    )
                )

            hits.sort(key=lambda r: r.similarity_score, reverse=True)
            return hits
        else:
            # In-memory cosine similarity fallback
            hits = []
            for cid, item in self._mem_chunks.items():
                meta = item["metadata"]
                if not filters.get("include_superseded", False) and meta.get("status") != "active":
                    continue
                service_filt = filters.get("service")
                if service_filt and service_filt not in meta.get("services", ""):
                    continue

                emb = item["embedding"]
                sim = sum(a * b for a, b in zip(query_embedding, emb))
                hits.append(
                    RetrievalResult(
                        incident_id=meta.get("incident_id", ""),
                        chunk_type=meta.get("chunk_type", ""),
                        similarity_score=round(float(sim), 6),
                        content=item["document"],
                        metadata={k: v for k, v in meta.items() if k not in ("incident_id", "chunk_type")},
                    )
                )

            hits.sort(key=lambda r: r.similarity_score, reverse=True)
            return hits[:top_k]

    def update_incident(self, incident_id: str, incident: IncidentRecord) -> None:
        """Replace all chunks for *incident_id* with data from *incident*.

        This is **not** append-only — old embeddings are removed and new ones
        are generated.

        Parameters
        ----------
        incident_id:
            The incident whose chunks should be replaced.
        incident:
            The updated record to re-embed.
        """
        log.info("chroma.update_incident", incident_id=incident_id)
        self.delete_incident(incident_id)
        self.add_incidents([incident])

    def delete_incident(self, incident_id: str) -> None:
        """Remove all chunks associated with *incident_id*.

        Parameters
        ----------
        incident_id:
            The incident identifier whose chunks should be purged.
        """
        ids_to_delete = [_chunk_id(incident_id, ct) for ct in _CHUNK_TYPES]
        if self._collection is not None:
            try:
                self._collection.delete(ids=ids_to_delete)
            except Exception:
                log.warning(
                    "chroma.delete_incident.partial",
                    incident_id=incident_id,
                    exc_info=True,
                )
        else:
            for cid in ids_to_delete:
                self._mem_chunks.pop(cid, None)
        log.info("chroma.delete_incident", incident_id=incident_id)

    def health_check(self) -> bool:
        """Return ``True`` if ChromaDB is reachable and the collection exists."""
        if self._client is not None:
            try:
                self._client.heartbeat()
                return True
            except Exception:
                log.error("chroma.health_check.failed", exc_info=True)
                return False
        return True
