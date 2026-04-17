"""
memory/memory_manager.py — Unified memory interface.
Combines Redis (short-term) and ChromaDB (long-term semantic) memory.
"""

from __future__ import annotations

from typing import Any, Optional

from core.logging_config import get_logger
from core.state import AgentState
from memory.redis_store import RedisStore
from memory.chroma_store import ChromaStore

logger = get_logger(__name__)


class MemoryManager:
    """
    Single interface for all memory operations.
    - Short-term: Redis (session context, working memory)
    - Long-term: ChromaDB (semantic retrieval, episode storage)
    """

    def __init__(self) -> None:
        self.redis = RedisStore()
        self.chroma = ChromaStore()

    async def load_context(
        self,
        session_id: str,
        user_id: str,
        query: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Load both short-term and long-term context for a session."""
        # Short-term: retrieve active session from Redis
        working = await self.redis.get_session(session_id) or {}

        # Long-term: semantic search for relevant episodes
        episodic = await self.chroma.search(
            query=query,
            user_id=user_id,
            top_k=top_k,
        )

        logger.debug(
            "Memory context loaded",
            session_id=session_id,
            working_keys=list(working.keys()),
            episodic_count=len(episodic),
        )

        return {
            "working": working,
            "episodic": episodic,
            "semantic": [],
        }

    async def save_context(
        self,
        session_id: str,
        user_id: str,
        state: AgentState,
    ) -> None:
        """Persist the completed task to both memory stores."""
        # Update Redis working memory
        await self.redis.set_session(
            session_id=session_id,
            data={
                "last_goal": state.get("goal"),
                "last_status": state.get("status"),
                "tool_results": state.get("tool_results", []),
                "iterations": state.get("iterations", 0),
            },
        )

        # Persist episode to ChromaDB
        if state.get("status") == "completed" and state.get("goal"):
            await self.chroma.add_episode(
                user_id=user_id,
                session_id=session_id,
                goal=state.get("goal", ""),
                result=str(state.get("tool_results", "")[:500]),
                metadata={
                    "task_id": state.get("task_id", ""),
                    "iterations": state.get("iterations", 0),
                },
            )

        logger.debug("Memory context saved", session_id=session_id)

    async def clear_session(self, session_id: str) -> None:
        """Remove all working memory for a session."""
        await self.redis.delete_session(session_id)
        logger.info("Session memory cleared", session_id=session_id)
