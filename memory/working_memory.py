"""
memory/working_memory.py — Redis-backed Working Memory.

Stores active session context, current task state, and agent working sets.
All keys are namespaced and carry configurable TTL for automatic expiry.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Optional

try:
    import redis.asyncio as aioredis
    _HAS_REDIS = True
except ImportError:
    import unittest.mock as mock
    aioredis = mock.MagicMock()
    _HAS_REDIS = False

from core.logging_config import get_logger

logger = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SESSION_TTL = int(os.getenv("REDIS_SESSION_TTL", "86400"))      # 24 h
TASK_TTL = int(os.getenv("REDIS_TASK_TTL", "3600"))             # 1 h
AGENT_WORKING_TTL = int(os.getenv("REDIS_AGENT_TTL", "1800"))   # 30 min
CACHE_TTL = int(os.getenv("REDIS_CACHE_TTL", "3600"))           # 1 h
MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "20"))


# ── Key Builders ──────────────────────────────────────────────────────────────

def _session_key(session_id: str) -> str:
    return f"wm:session:{session_id}"


def _context_key(session_id: str, key: str) -> str:
    return f"wm:ctx:{session_id}:{key}"


def _task_key(session_id: str) -> str:
    return f"wm:task:{session_id}"


def _agent_key(agent_id: str) -> str:
    return f"wm:agent:{agent_id}"


def _cache_key(cache_key: str) -> str:
    return f"wm:cache:{cache_key}"


def _queue_key(queue_name: str) -> str:
    return f"wm:queue:{queue_name}"


# ── Client Pool ───────────────────────────────────────────────────────────────

_pool: Optional[aioredis.Redis] = None


# ── Local Fallback (for when Redis is missing) ────────────────────────────────
_local_store: dict[str, Any] = {}


async def _get_client() -> aioredis.Redis:
    global _pool
    if not _HAS_REDIS:
        return mock.AsyncMock()

    if _pool is None:
        _pool = await aioredis.from_url(
            REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            max_connections=MAX_CONNECTIONS,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )
    return _pool


# ── WorkingMemory ─────────────────────────────────────────────────────────────


class WorkingMemory:
    """
    Redis-backed working memory for active sessions, tasks, and agents.

    All operations are fully async. Keys are namespaced under `wm:` to
    avoid collisions with other Redis usage in the OS.

    TTL Hierarchy:
        session  → 24 h  (full session blob)
        context  → 24 h  (per-key context within a session)
        task     → 1 h   (current task state)
        agent    → 30 m  (agent working sets)
        cache    → 1 h   (tool result cache)
    """

    def __init__(self) -> None:
        self._client: Optional[aioredis.Redis] = None

    async def _r(self) -> aioredis.Redis:
        if self._client is None:
            self._client = await _get_client()
        return self._client

    # ── Session Management ────────────────────────────────────────────────────

    async def store_context(
        self,
        session_id: str,
        key: str,
        value: Any,
        ttl: int = SESSION_TTL,
    ) -> None:
        """Store a named context value within a session."""
        try:
            if not _HAS_REDIS:
                _local_store[_context_key(session_id, key)] = json.dumps(value, default=str)
                return
            r = await self._r()
            await r.setex(
                _context_key(session_id, key),
                ttl,
                json.dumps(value, default=str),
            )
        except Exception as exc:
            logger.warning("WorkingMemory.store_context failed", key=key, error=str(exc))

    async def get_context(self, session_id: str, key: str) -> Optional[Any]:
        """Retrieve a named context value from a session."""
        try:
            if not _HAS_REDIS:
                raw = _local_store.get(_context_key(session_id, key))
                return json.loads(raw) if raw is not None else None
            r = await self._r()
            raw = await r.get(_context_key(session_id, key))
            return json.loads(raw) if raw is not None else None
        except Exception as exc:
            logger.warning("WorkingMemory.get_context failed", key=key, error=str(exc))
            return None

    async def get_working_set(self, session_id: str) -> dict[str, Any]:
        """
        Return the full working set for a session as a flat dict.
        Scans all `wm:ctx:{session_id}:*` keys.
        """
        try:
            r = await self._r()
            pattern = f"wm:ctx:{session_id}:*"
            keys = await r.keys(pattern)
            if not keys:
                return {}
            values = await r.mget(*keys)
            prefix_len = len(f"wm:ctx:{session_id}:")
            result: dict[str, Any] = {}
            for k, v in zip(keys, values):
                if v is not None:
                    short_key = k[prefix_len:]
                    result[short_key] = json.loads(v)
            return result
        except Exception as exc:
            logger.warning("WorkingMemory.get_working_set failed", error=str(exc))
            return {}

    async def set_session(
        self,
        session_id: str,
        data: dict[str, Any],
        ttl: int = SESSION_TTL,
    ) -> None:
        """Save the entire session blob."""
        try:
            data["_updated_at"] = datetime.utcnow().isoformat()
            if not _HAS_REDIS:
                _local_store[_session_key(session_id)] = json.dumps(data, default=str)
                return
            r = await self._r()
            await r.setex(
                _session_key(session_id),
                ttl,
                json.dumps(data, default=str),
            )
        except Exception as exc:
            logger.warning("WorkingMemory.set_session failed", error=str(exc))

    async def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        """Retrieve the session blob."""
        try:
            if not _HAS_REDIS:
                raw = _local_store.get(_session_key(session_id))
                return json.loads(raw) if raw else None
            r = await self._r()
            raw = await r.get(_session_key(session_id))
            return json.loads(raw) if raw else None
        except Exception as exc:
            logger.warning("WorkingMemory.get_session failed", error=str(exc))
            return None

    async def delete_session(self, session_id: str) -> None:
        """Remove all session memory (blob + individual context keys)."""
        try:
            r = await self._r()
            pipe = r.pipeline()
            pipe.delete(_session_key(session_id))
            # Delete individual context keys
            ctx_keys = await r.keys(f"wm:ctx:{session_id}:*")
            if ctx_keys:
                pipe.delete(*ctx_keys)
            await pipe.execute()
            logger.info("Session deleted from WorkingMemory", session_id=session_id)
        except Exception as exc:
            logger.warning("WorkingMemory.delete_session failed", error=str(exc))

    # ── Task State ────────────────────────────────────────────────────────────

    async def set_task_state(
        self,
        session_id: str,
        state: dict[str, Any],
        ttl: int = TASK_TTL,
    ) -> None:
        """Persist the current task's state dict."""
        try:
            r = await self._r()
            state["_updated_at"] = datetime.utcnow().isoformat()
            await r.setex(
                _task_key(session_id),
                ttl,
                json.dumps(state, default=str),
            )
        except Exception as exc:
            logger.warning("WorkingMemory.set_task_state failed", error=str(exc))

    async def get_task_state(self, session_id: str) -> Optional[dict[str, Any]]:
        """Retrieve the current task's state dict."""
        try:
            r = await self._r()
            raw = await r.get(_task_key(session_id))
            return json.loads(raw) if raw else None
        except Exception as exc:
            logger.warning("WorkingMemory.get_task_state failed", error=str(exc))
            return None

    # ── Agent Working Sets ────────────────────────────────────────────────────

    async def set_agent_working_set(
        self,
        agent_id: str,
        items: list[Any],
        ttl: int = AGENT_WORKING_TTL,
    ) -> None:
        """Store an agent's active working set (list of items)."""
        try:
            r = await self._r()
            await r.setex(
                _agent_key(agent_id),
                ttl,
                json.dumps({"items": items, "agent_id": agent_id}, default=str),
            )
        except Exception as exc:
            logger.warning("WorkingMemory.set_agent_working_set failed", error=str(exc))

    async def get_agent_working_set(self, agent_id: str) -> list[Any]:
        """Retrieve an agent's working set."""
        try:
            r = await self._r()
            raw = await r.get(_agent_key(agent_id))
            if raw:
                data = json.loads(raw)
                return data.get("items", [])
            return []
        except Exception as exc:
            logger.warning("WorkingMemory.get_agent_working_set failed", error=str(exc))
            return []

    async def expire_session(self, session_id: str, ttl: int = 60) -> None:
        """Force-expire a session after `ttl` seconds (default: 1 min)."""
        try:
            r = await self._r()
            await r.expire(_session_key(session_id), ttl)
            ctx_keys = await r.keys(f"wm:ctx:{session_id}:*")
            if ctx_keys:
                pipe = r.pipeline()
                for k in ctx_keys:
                    pipe.expire(k, ttl)
                await pipe.execute()
        except Exception as exc:
            logger.warning("WorkingMemory.expire_session failed", error=str(exc))

    # ── Tool Result Cache ─────────────────────────────────────────────────────

    async def cache_tool_result(
        self,
        cache_key: str,
        result: Any,
        ttl: int = CACHE_TTL,
    ) -> None:
        try:
            r = await self._r()
            await r.setex(_cache_key(cache_key), ttl, json.dumps(result, default=str))
        except Exception as exc:
            logger.warning("WorkingMemory.cache_tool_result failed", error=str(exc))

    async def get_cached_tool_result(self, cache_key: str) -> Optional[Any]:
        try:
            r = await self._r()
            raw = await r.get(_cache_key(cache_key))
            return json.loads(raw) if raw else None
        except Exception as exc:
            logger.warning("WorkingMemory.get_cached_tool_result failed", error=str(exc))
            return None

    # ── Task Queue ────────────────────────────────────────────────────────────

    async def enqueue_task(self, queue_name: str, task: dict[str, Any]) -> None:
        try:
            r = await self._r()
            await r.lpush(_queue_key(queue_name), json.dumps(task, default=str))
        except Exception as exc:
            logger.error("WorkingMemory.enqueue_task failed", queue=queue_name, error=str(exc))

    async def dequeue_task(
        self, queue_name: str, timeout: int = 1
    ) -> Optional[dict[str, Any]]:
        try:
            r = await self._r()
            raw = await r.brpop(_queue_key(queue_name), timeout=timeout)
            return json.loads(raw[1]) if raw else None
        except Exception as exc:
            logger.warning("WorkingMemory.dequeue_task failed", error=str(exc))
            return None

    # ── Health Check ──────────────────────────────────────────────────────────

    async def ping(self) -> bool:
        try:
            r = await self._r()
            return await r.ping()
        except Exception:
            return False


# ── Backward-compatible alias used by core/ ───────────────────────────────────
# The original RedisStore interface is preserved here so existing imports
# from memory.redis_store work unchanged.
RedisStore = WorkingMemory
