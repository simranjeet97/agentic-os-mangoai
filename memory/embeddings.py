"""
memory/embeddings.py — Shared embedding service for the Memory System.

Uses sentence-transformers/all-MiniLM-L6-v2 for local, offline embeddings.
Model is loaded lazily on first call and runs in a thread pool to avoid
blocking the async event loop.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_EMBED_DIM = 384  # all-MiniLM-L6-v2 output dimension

# Thread pool for CPU-bound embedding work
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embed")


@lru_cache(maxsize=1)
def _load_model():
    """Load the sentence-transformer model (cached singleton)."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        logger.info("Loading embedding model: %s", MODEL_NAME)
        model = SentenceTransformer(MODEL_NAME)
        logger.info("Embedding model loaded successfully")
        return model
    except ImportError:
        logger.warning(
            "sentence-transformers not installed. "
            "Falling back to zero-vector embeddings."
        )
        return None
    except Exception as exc:
        logger.error("Failed to load embedding model: %s", exc)
        return None


def _encode_sync(texts: list[str]) -> np.ndarray:
    """Synchronous encode — runs inside the thread pool."""
    model = _load_model()
    if model is None:
        # Graceful fallback: return zero vectors
        return np.zeros((len(texts), _EMBED_DIM), dtype=np.float32)
    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )


class EmbeddingService:
    """
    Async wrapper around sentence-transformers for memory components.

    Usage:
        svc = EmbeddingService()
        vectors = await svc.encode(["hello world", "another text"])
    """

    def __init__(self, executor: Optional[ThreadPoolExecutor] = None) -> None:
        self._executor = executor or _executor

    async def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of texts into embedding vectors.
        Returns shape (len(texts), 384).
        Offloads CPU work to thread pool; does not block the event loop.
        """
        if not texts:
            return np.zeros((0, _EMBED_DIM), dtype=np.float32)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            _encode_sync,
            texts,
        )

    async def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text. Returns shape (384,)."""
        vectors = await self.encode([text])
        return vectors[0]

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        Assumes pre-normalized vectors (sentence-transformers normalizes by default).
        """
        a = np.array(a, dtype=np.float32).flatten()
        b = np.array(b, dtype=np.float32).flatten()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def batch_cosine_similarity(
        query: np.ndarray,
        corpus: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between one query vector and a corpus matrix.
        Returns shape (len(corpus),).
        """
        query = query.flatten() / (np.linalg.norm(query) + 1e-9)
        norms = np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-9
        normed = corpus / norms
        return normed @ query

    async def preload(self) -> bool:
        """
        Pre-warm the model (call during startup to avoid cold-start latency).
        Returns True if model loaded successfully.
        """
        loop = asyncio.get_event_loop()
        model = await loop.run_in_executor(self._executor, _load_model)
        return model is not None

    @property
    def embedding_dim(self) -> int:
        return _EMBED_DIM


# Module-level singleton for convenient import
_default_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Return the module-level singleton EmbeddingService."""
    global _default_service
    if _default_service is None:
        _default_service = EmbeddingService()
    return _default_service
