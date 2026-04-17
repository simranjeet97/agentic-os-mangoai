"""
memory/chroma_store.py — ChromaDB-backed long-term semantic memory.
Stores episodic memories and enables semantic retrieval.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Any, Optional

import chromadb
from chromadb import AsyncHttpClient

from core.logging_config import get_logger

logger = get_logger(__name__)

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))
COLLECTION_PREFIX = os.getenv("CHROMA_COLLECTION_PREFIX", "agentic")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


class ChromaStore:
    """
    Async ChromaDB client for long-term semantic memory.
    Uses Ollama's nomic-embed-text for local embeddings.
    """

    def __init__(self) -> None:
        self._client: Optional[chromadb.AsyncClientAPI] = None

    async def _get_client(self) -> chromadb.AsyncClientAPI:
        if self._client is None:
            self._client = await AsyncHttpClient(
                host=CHROMA_HOST,
                port=CHROMA_PORT,
            )
        return self._client

    async def _get_collection(
        self, user_id: str
    ) -> chromadb.AsyncCollection:
        """Get or create a per-user memory collection."""
        client = await self._get_client()
        collection_name = f"{COLLECTION_PREFIX}_{user_id.replace('-', '_')}"
        return await client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def add_episode(
        self,
        user_id: str,
        session_id: str,
        goal: str,
        result: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Store a completed task episode in ChromaDB."""
        try:
            collection = await self._get_collection(user_id)
            doc_id = str(uuid.uuid4())
            document = f"Goal: {goal}\nResult: {result}"
            meta = {
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                **(metadata or {}),
            }
            # Convert any non-string values for ChromaDB compatibility
            meta = {k: str(v) for k, v in meta.items()}

            await collection.add(
                ids=[doc_id],
                documents=[document],
                metadatas=[meta],
            )
            logger.debug("Episode stored in ChromaDB", doc_id=doc_id, user_id=user_id)
            return doc_id
        except Exception as exc:
            logger.warning("ChromaDB add_episode failed", error=str(exc))
            return ""

    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Semantic search over a user's episodic memory."""
        try:
            collection = await self._get_collection(user_id)
            results = await collection.query(
                query_texts=[query],
                n_results=min(top_k, 10),
                include=["documents", "metadatas", "distances"],
            )

            episodes = []
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]

            for doc, meta, dist in zip(docs, metas, dists):
                episodes.append({
                    "content": doc,
                    "metadata": meta,
                    "relevance_score": round(1.0 - float(dist), 4),
                })

            return episodes
        except Exception as exc:
            logger.warning("ChromaDB search failed", error=str(exc))
            return []

    async def delete_user_memory(self, user_id: str) -> None:
        """Delete all memory for a user (right to be forgotten)."""
        try:
            client = await self._get_client()
            collection_name = f"{COLLECTION_PREFIX}_{user_id.replace('-', '_')}"
            await client.delete_collection(name=collection_name)
            logger.info("User memory deleted", user_id=user_id)
        except Exception as exc:
            logger.warning("ChromaDB delete_user_memory failed", error=str(exc))

    async def ping(self) -> bool:
        try:
            client = await self._get_client()
            await client.heartbeat()
            return True
        except Exception:
            return False
