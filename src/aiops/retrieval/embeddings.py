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

    def _load_model(self) -> Any:
        """Load the model in a thread-safe manner, returning the instance or fallback."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
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
                    except (ImportError, Exception):
                        log.info("embedding_model.fallback_hash", model_name="hash-384")
                        self._model = "fallback"
        return self._model

    def _hash_embed(self, text: str) -> list[float]:
        """Deterministic 384-dim normalized embedding when sentence-transformers is absent."""
        import hashlib
        import math

        words = text.lower().split()
        dim = 384
        vec = [0.0] * dim
        for w in words:
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if ((h >> 9) & 1) else -1.0
            vec[idx] += sign

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            return [x / norm for x in vec]
        return vec

    # -- public API -----------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors produced by the model."""
        model = self._load_model()
        if model == "fallback":
            return 384
        dim = model.get_sentence_embedding_dimension()
        if dim is None:  # pragma: no cover – defensive
            raise RuntimeError("Model did not report an embedding dimension.")
        return int(dim)

    def embed(self, text: str) -> list[float]:
        """Embed a single text string and return its vector."""
        model = self._load_model()
        if model == "fallback":
            return self._hash_embed(text)
        vector = model.encode(text, show_progress_bar=False)
        return vector.tolist()  # type: ignore[union-attr]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single call."""
        if not texts:
            return []
        model = self._load_model()
        if model == "fallback":
            return [self._hash_embed(t) for t in texts]
        vectors = model.encode(texts, show_progress_bar=False, batch_size=64)
        return [v.tolist() for v in vectors]  # type: ignore[union-attr]
