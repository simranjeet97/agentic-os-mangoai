"""
memory/embeddings.py — Shared embedding service for the Memory System.
Uses Google Gemini API (text-embedding-004) for embeddings.
This implementation is lightweight and avoids local torch/transformers dependencies.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Standard dimension for Google's text-embedding-004
_EMBED_DIM = 768

class EmbeddingService:
    """
    Async wrapper around Google Gemini Embeddings for memory components.
    
    Usage:
        svc = EmbeddingService()
        vectors = await svc.encode(["hello world", "another text"])
    """

    def __init__(self, model_name: str = "models/gemini-embedding-001") -> None:
        self.model_name = model_name
        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        """Lazy-load the LangChain Gemini embedding client."""
        if self._client is None:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                logger.warning("GOOGLE_API_KEY not set. Embedding calls will likely fail.")
                
            self._client = GoogleGenerativeAIEmbeddings(
                model=self.model_name,
                google_api_key=api_key,
            )
        return self._client

    async def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of texts into embedding vectors via Gemini API.
        Returns shape (len(texts), 768).
        """
        if not texts:
            return np.zeros((0, _EMBED_DIM), dtype=np.float32)

        try:
            client = self._get_client()
            # LangChain's embed_documents is a sync call in the base class, 
            # but langchain-google-genai might be blocking. 
            # Wrap in to_thread just in case if not natively async.
            import asyncio
            vectors = await asyncio.to_thread(client.embed_documents, texts)
            return np.array(vectors, dtype=np.float32)
        except Exception as exc:
            logger.error("Gemini embedding failed: %s", exc)
            # Graceful fallback: return zero vectors to prevent system crash
            return np.zeros((len(texts), _EMBED_DIM), dtype=np.float32)

    async def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text. Returns shape (768,)."""
        vectors = await self.encode([text])
        if vectors.shape[0] > 0:
            return vectors[0]
        return np.zeros((_EMBED_DIM,), dtype=np.float32)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
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
        """Compute cosine similarity between one query vector and a corpus matrix."""
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return np.zeros(len(corpus))
        
        query = query.flatten() / (query_norm + 1e-9)
        norms = np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-9
        normed = corpus / norms
        return normed @ query

    async def preload(self) -> bool:
        """Validate API connectivity."""
        try:
            await self.encode(["ping"])
            return True
        except Exception:
            return False

    @property
    def embedding_dim(self) -> int:
        return _EMBED_DIM


# Module-level singleton
_default_service: Optional[EmbeddingService] = None

def get_embedding_service() -> EmbeddingService:
    global _default_service
    if _default_service is None:
        _default_service = EmbeddingService()
    return _default_service
