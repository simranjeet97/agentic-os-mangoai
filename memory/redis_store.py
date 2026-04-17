"""
memory/redis_store.py — Redis-backed short-term working memory.
Stores session state, task queues, and tool call caches.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import redis.asyncio as aioredis

from core.logging_config import get_logger

logger = get_logger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SESSION_TTL = int(os.getenv("REDIS_SESSION_TTL", "86400"))   # 24h default
CACHE_TTL = int(os.getenv("REDIS_CACHE_TTL", "3600"))        # 1h default


class RedisStore:
    """Async Redis client for working memory and caching."""

    def __init__(self) -> None:
        self._client: Optional[aioredis.Redis] = None

    async def _get_client(self) -> aioredis.Redis:
        if self._client is None:
            self._client = await aioredis.from_url(
                REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "20")),
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
            )
        return self._client

    # ── Session Management ────────────────────────────────────────────────────

    async def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        """Retrieve session data by session_id."""
        try:
            client = await self._get_client()
            raw = await client.get(f"session:{session_id}")
            return json.loads(raw) if raw else None
        except Exception as exc:
            logger.warning("Redis get_session failed", session_id=session_id, error=str(exc))
            return None

    async def set_session(
        self,
        session_id: str,
        data: dict[str, Any],
        ttl: int = SESSION_TTL,
    ) -> None:
        """Save session data with TTL."""
        try:
            client = await self._get_client()
            await client.setex(
                f"session:{session_id}",
                ttl,
                json.dumps(data, default=str),
            )
        except Exception as exc:
            logger.warning("Redis set_session failed", session_id=session_id, error=str(exc))

    async def delete_session(self, session_id: str) -> None:
        try:
            client = await self._get_client()
            await client.delete(f"session:{session_id}")
        except Exception as exc:
            logger.warning("Redis delete_session failed", error=str(exc))

    # ── Tool Call Cache ───────────────────────────────────────────────────────

    async def cache_tool_result(
        self,
        cache_key: str,
        result: Any,
        ttl: int = CACHE_TTL,
    ) -> None:
        try:
            client = await self._get_client()
            await client.setex(f"tool_cache:{cache_key}", ttl, json.dumps(result, default=str))
        except Exception as exc:
            logger.warning("Redis cache write failed", key=cache_key, error=str(exc))

    async def get_cached_tool_result(self, cache_key: str) -> Optional[Any]:
        try:
            client = await self._get_client()
            raw = await client.get(f"tool_cache:{cache_key}")
            return json.loads(raw) if raw else None
        except Exception as exc:
            logger.warning("Redis cache read failed", key=cache_key, error=str(exc))
            return None

    # ── Task Queue ────────────────────────────────────────────────────────────

    async def enqueue_task(self, queue_name: str, task: dict[str, Any]) -> None:
        try:
            client = await self._get_client()
            await client.lpush(f"queue:{queue_name}", json.dumps(task, default=str))
        except Exception as exc:
            logger.error("Redis enqueue failed", queue=queue_name, error=str(exc))

    async def dequeue_task(self, queue_name: str) -> Optional[dict[str, Any]]:
        try:
            client = await self._get_client()
            raw = await client.brpop(f"queue:{queue_name}", timeout=1)
            return json.loads(raw[1]) if raw else None
        except Exception as exc:
            logger.warning("Redis dequeue failed", queue=queue_name, error=str(exc))
            return None

    # ── Health Check ──────────────────────────────────────────────────────────

    async def ping(self) -> bool:
        try:
            client = await self._get_client()
            return await client.ping()
        except Exception:
            return False
