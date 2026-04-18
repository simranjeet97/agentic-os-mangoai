"""
memory/chroma_store.py — ChromaDB-backed Semantic Memory.

Indexes user documents, code files, and notes using sentence-transformers
embeddings (all-MiniLM-L6-v2). Supports cosine similarity search.

Auto-indexes ~/Documents and ~/Projects on first run (configurable).
Chunking: 512 tokens (~400 words) with 50-token overlap.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    import chromadb
    _HAS_CHROMA = True
except ImportError:
    import unittest.mock as mock
    chromadb = mock.MagicMock()
    _HAS_CHROMA = False

from memory.embeddings import EmbeddingService, get_embedding_service
from memory.schemas import RecallResult, MemoryType

try:
    from chromadb import AsyncHttpClient as _AsyncHttpClient
except ImportError:
    _AsyncHttpClient = None  # type: ignore

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))
COLLECTION_PREFIX = os.getenv("CHROMA_COLLECTION_PREFIX", "agentic")
CHUNK_SIZE = int(os.getenv("SEMANTIC_CHUNK_SIZE", "400"))   # words per chunk
CHUNK_OVERLAP = int(os.getenv("SEMANTIC_CHUNK_OVERLAP", "50"))  # word overlap
AUTO_INDEX_DIRS = os.getenv(
    "SEMANTIC_AUTO_INDEX_DIRS",
    f"{Path.home() / 'Documents'},{Path.home() / 'Projects'}",
)
INDEXED_MARKER = Path.home() / ".agentic_os" / ".semantic_indexed"
SUPPORTED_EXTENSIONS = {".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".rst", ".csv"}


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i: i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


def _file_id(path: str, chunk_index: int) -> str:
    h = hashlib.md5(f"{path}::{chunk_index}".encode()).hexdigest()
    return h


class SemanticMemory:
    """
    ChromaDB-based semantic memory with local sentence-transformer embeddings.

    - Indexes documents via index_document() / index_directory()
    - Performs cosine similarity search via search()
    - Auto-indexes home directories on first run
    - Embedding model: sentence-transformers/all-MiniLM-L6-v2
    """

    def __init__(
        self,
        embed_service: Optional[EmbeddingService] = None,
        chroma_client: Optional[Any] = None,
    ) -> None:
        self._embed = embed_service or get_embedding_service()
        self._chroma_client = chroma_client  # injected for testing (sync EphemeralClient OK)
        self._client: Optional[Any] = None
        self._is_sync_client: bool = False

    # ── Client & Collection ───────────────────────────────────────────────────

    async def _get_client(self) -> Any:
        """Get or create the ChromaDB client."""
        if self._chroma_client is not None:
            self._is_sync_client = True
            return self._chroma_client
        if self._client is None:
            try:
                if _AsyncHttpClient is not None:
                    self._client = await _AsyncHttpClient(
                        host=CHROMA_HOST,
                        port=CHROMA_PORT,
                    )
                    self._is_sync_client = False
                else:
                    raise RuntimeError("AsyncHttpClient not available")
            except Exception as exc:
                # Fallback to in-process ephemeral sync client
                logger.warning(
                    "Cannot connect to ChromaDB server, using ephemeral client: %s", exc
                )
                self._client = chromadb.EphemeralClient()
                self._is_sync_client = True
        return self._client

    async def _get_collection(self, namespace: str = "global") -> Any:
        """Get or create a collection, handling both sync and async clients."""
        client = await self._get_client()
        name = f"{COLLECTION_PREFIX}_{namespace}".replace("-", "_")
        if self._is_sync_client:
            return await asyncio.to_thread(
                client.get_or_create_collection,
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return await client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    async def _coll_call(self, collection: Any, method: str, **kwargs: Any) -> Any:
        """
        Dispatch a collection method call.
        Uses asyncio.to_thread for sync collections, awaits for async ones.
        """
        fn = getattr(collection, method)
        if self._is_sync_client:
            return await asyncio.to_thread(fn, **kwargs)
        result = fn(**kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    # ── Indexing ──────────────────────────────────────────────────────────────

    async def index_document(
        self,
        content: str,
        source_path: str = "unknown",
        doc_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        namespace: str = "global",
    ) -> list[str]:
        """
        Index a document (chunked) into semantic memory.
        Returns list of chunk IDs stored.
        """
        chunks = _chunk_text(content)
        if not chunks:
            return []

        vectors = await self._embed.encode(chunks)
        collection = await self._get_collection(namespace)

        ids = []
        chunk_docs = []
        chunk_metas = []
        chunk_embeddings = []

        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            cid = doc_id or _file_id(source_path, i)
            if len(chunks) > 1:
                cid = f"{cid}__chunk_{i}"
            ids.append(cid)
            chunk_docs.append(chunk)
            meta = {
                "source_path": source_path,
                "chunk_index": str(i),
                "total_chunks": str(len(chunks)),
                "indexed_at": datetime.utcnow().isoformat(),
                **(metadata or {}),
            }
            # ChromaDB requires all metadata values to be str/int/float/bool
            chunk_metas.append({k: str(v) for k, v in meta.items()})
            chunk_embeddings.append(vec.tolist())

        try:
            await self._coll_call(
                collection, "upsert",
                ids=ids,
                documents=chunk_docs,
                metadatas=chunk_metas,
                embeddings=chunk_embeddings,
            )
            logger.debug("Document indexed: source=%s chunks=%d", source_path, len(ids))
        except Exception as exc:
            logger.warning("SemanticMemory.index_document failed: %s", exc)

        return ids

    async def index_directory(
        self,
        directory: str,
        namespace: str = "global",
        recursive: bool = True,
    ) -> int:
        """
        Scan a directory and index all supported file types.
        Returns the number of files indexed.
        """
        dir_path = Path(directory).expanduser()
        if not dir_path.exists():
            logger.warning("Directory not found, skipping", path=str(dir_path))
            return 0

        pattern = "**/*" if recursive else "*"
        files = [f for f in dir_path.glob(pattern) if f.suffix in SUPPORTED_EXTENSIONS and f.is_file()]
        indexed = 0

        for file_path in files:
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                if len(text.strip()) < 20:
                    continue
                await self.index_document(
                    content=text,
                    source_path=str(file_path),
                    metadata={
                        "file_name": file_path.name,
                        "file_type": file_path.suffix,
                        "directory": str(file_path.parent),
                    },
                    namespace=namespace,
                )
                indexed += 1
            except Exception as exc:
                logger.warning("Failed to index file: %s", exc)

        logger.info("Directory indexed: directory=%s, files_indexed=%d", directory, indexed)
        return indexed

    async def auto_index_on_first_run(self, namespace: str = "global") -> None:
        """
        Index ~/Documents and ~/Projects on first run (once per installation).
        Skips if the marker file exists.
        """
        if INDEXED_MARKER.exists():
            return

        logger.info("Auto-indexing home directories on first run...")
        dirs = [d.strip() for d in AUTO_INDEX_DIRS.split(",") if d.strip()]
        total = 0
        for d in dirs:
            total += await self.index_directory(d, namespace=namespace)

        INDEXED_MARKER.parent.mkdir(parents=True, exist_ok=True)
        INDEXED_MARKER.write_text(datetime.utcnow().isoformat())
        logger.info("Auto-indexing complete: files_indexed=%d", total)

    # ── Search ────────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        top_k: int = 5,
        namespace: str = "global",
        filter_metadata: Optional[dict[str, str]] = None,
    ) -> list[RecallResult]:
        """
        Semantic search using cosine similarity.
        Returns RecallResult objects sorted by relevance descending.
        """
        try:
            collection = await self._get_collection(namespace)
            query_vector = (await self._embed.encode([query]))[0].tolist()
            extra: dict = {}
            if filter_metadata:
                extra["where"] = {k: {"$eq": v} for k, v in filter_metadata.items()}
            results = await self._coll_call(
                collection, "query",
                query_embeddings=[query_vector],
                n_results=min(top_k, 20),
                include=["documents", "metadatas", "distances"],
                **extra,
            )

            output: list[RecallResult] = []
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]
            ids = results.get("ids", [[]])[0]

            for doc, meta, dist, rid in zip(docs, metas, dists, ids):
                output.append(
                    RecallResult(
                        item_id=rid,
                        content=doc,
                        source=MemoryType.SEMANTIC,
                        relevance=round(max(0.0, 1.0 - float(dist)), 4),
                        metadata=dict(meta or {}),
                    )
                )
            return output
        except Exception as exc:
            logger.warning("SemanticMemory.search failed: %s", exc)
            return []

    # ── User-episode API (backward-compat with old ChromaStore) ──────────────

    async def add_episode(
        self,
        user_id: str,
        session_id: str,
        goal: str,
        result: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Store a completed task episode (backward-compat shim)."""
        doc = f"Goal: {goal}\nResult: {result}"
        ids = await self.index_document(
            content=doc,
            source_path=f"session:{session_id}",
            doc_id=str(uuid.uuid4()),
            metadata={
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                **(metadata or {}),
            },
            namespace=user_id,
        )
        return ids[0] if ids else ""

    async def delete_document(
        self, doc_id: str, namespace: str = "global"
    ) -> None:
        """Remove a document (and all its chunks) from semantic memory."""
        try:
            collection = await self._get_collection(namespace)
            # Fetch all chunk IDs with this prefix
            existing = await self._coll_call(
                collection, "get",
                where={"source_path": {"$eq": doc_id}},
                include=[],
            )
            ids_to_delete = existing.get("ids", [])
            if ids_to_delete:
                await self._coll_call(collection, "delete", ids=ids_to_delete)
            # Also try direct ID delete
            try:
                await self._coll_call(collection, "delete", ids=[doc_id])
            except Exception:
                pass
            logger.info("Document deleted from SemanticMemory: doc_id=%s", doc_id)
        except Exception as exc:
            logger.warning("SemanticMemory.delete_document failed: %s", exc)

    async def delete_user_memory(self, user_id: str) -> None:
        """Delete all memory for a user (GDPR right-to-be-forgotten)."""
        try:
            client = await self._get_client()
            name = f"{COLLECTION_PREFIX}_{user_id}".replace("-", "_")
            if self._is_sync_client:
                await asyncio.to_thread(client.delete_collection, name=name)
            else:
                await client.delete_collection(name=name)
            logger.info("User semantic memory deleted: user_id=%s", user_id)
        except Exception as exc:
            logger.warning("SemanticMemory.delete_user_memory failed: %s", exc)

    async def ping(self) -> bool:
        try:
            client = await self._get_client()
            if self._is_sync_client:
                await asyncio.to_thread(client.heartbeat)
            else:
                await client.heartbeat()
            return True
        except Exception:
            return False


# ── Backward-compatible alias ─────────────────────────────────────────────────
ChromaStore = SemanticMemory
