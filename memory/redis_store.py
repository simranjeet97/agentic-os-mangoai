"""
memory/redis_store.py — Backward-compatible shim.

The full RedisStore implementation now lives in memory/working_memory.py
as the WorkingMemory class. This module re-exports it under the original
name so any code importing from memory.redis_store continues to work.
"""

from memory.working_memory import WorkingMemory as RedisStore

__all__ = ["RedisStore"]
