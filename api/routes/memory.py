"""
api/routes/memory.py — Memory query and management endpoints.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from core.logging_config import get_logger
from memory.memory_manager import MemoryManager

router = APIRouter()
logger = get_logger(__name__)


class MemoryQueryRequest(BaseModel):
    query: str
    user_id: str = "anonymous"
    top_k: int = 5


@router.get("/query", summary="Query long-term semantic memory")
async def query_memory(
    query: str = Query(..., description="Natural language search query"),
    user_id: str = Query(default="anonymous"),
    top_k: int = Query(default=5, ge=1, le=20),
):
    """Semantic search over a user's episodic memory in ChromaDB."""
    try:
        mm = MemoryManager()
        results = await mm.semantic.search(query=query, user_id=user_id, top_k=top_k)
        return {"query": query, "results": results, "count": len(results)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@router.get("/search", summary="Semantic search over memory")
async def search_memory(
    query: str = Query(..., description="Semantic search over user memory"),
    user_id: str = Query(default="anonymous"),
    top_k: int = Query(default=10, ge=1, le=50),
    filters: Optional[str] = Query(None, description="Optional metadata keys to filter")
):
    """Semantic search over memory endpoint (re-using query memory backing)."""
    try:
        mm = MemoryManager()
        # Filters could be parsed here in a real implementation
        filter_dict = None
        if filters:
            # Simple assumption it's a comma-separated key=value string
            try:
                filter_dict = dict(pair.split("=") for pair in filters.split(","))
            except:
                pass
        results = await mm.semantic.search(query=query, user_id=user_id, top_k=top_k, metadata_filter=filter_dict)
        return {"query": query, "results": results, "count": len(results)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))



@router.delete("/user/{user_id}", summary="Delete all memory for a user")
async def delete_user_memory(user_id: str):
    """Right-to-be-forgotten: delete all stored memory for a user."""
    try:
        mm = MemoryManager()
        await mm.semantic.delete_document(user_id)
        return {"message": f"Memory deleted for user {user_id}"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/session/{session_id}", summary="Clear working memory for a session")
async def clear_session_memory(session_id: str):
    """Clear Redis working memory for a session."""
    try:
        mm = MemoryManager()
        await mm.clear_session(session_id)
        return {"message": f"Session memory cleared: {session_id}"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
