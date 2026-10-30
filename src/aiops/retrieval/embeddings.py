"""Thin wrapper around *sentence-transformers* for embedding generation.

Provides :class:`EmbeddingService` with lazy model loading and a unified
interface used by the vector store and retriever.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import structlog

from aiops.config import get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = structlog.get_logger()


class EmbeddingService:
    """Generate dense vector embeddings from text.

    The underlying ``SentenceTransformer`` model is loaded lazily on the first
    call to :meth:`embed` or :meth:`embed_batch` and cached for reuse.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.  Defaults to the value in
        :pydata:`aiops.config.Settings.embedding_model`.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name: str = model_name or get_settings().embedding_model
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()

    # -- internal helpers -----------------------------------------------------

    def _load_model(self) -> SentenceTransformer:
        """Load the model in a thread-safe manner, returning the instance."""
        if self._model is None:
            with self._lock:
                # Double-check after acquiring the lock.
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    log.info(
                        "embedding_model.loading",
                        model_name=self._model_name,
                    )
                    self._model = SentenceTransformer(self._model_name)
                    log.info(
                        "embedding_model.loaded",
                        model_name=self._model_name,
                        dimension=self._model.get_sentence_embedding_dimension(),
                    )
        return self._model  # type: ignore[return-value]

    # -- public API -----------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors produced by the model."""
        model = self._load_model()
        dim = model.get_sentence_embedding_dimension()
        if dim is None:  # pragma: no cover – defensive
            raise RuntimeError("Model did not report an embedding dimension.")
        return int(dim)

    def embed(self, text: str) -> list[float]:
        """Embed a single text string and return its vector.

        Parameters
        ----------
        text:
            The input text to embed.

        Returns
        -------
        list[float]
            Dense vector representation.
        """
        model = self._load_model()
        vector = model.encode(text, show_progress_bar=False)
        return vector.tolist()  # type: ignore[union-attr]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single call.

        Parameters
        ----------
        texts:
            List of text strings to embed.

        Returns
        -------
        list[list[float]]
            A list of dense vectors, one per input text.
        """
        if not texts:
            return []

        model = self._load_model()
        vectors = model.encode(texts, show_progress_bar=False, batch_size=64)
        return [v.tolist() for v in vectors]  # type: ignore[union-attr]
