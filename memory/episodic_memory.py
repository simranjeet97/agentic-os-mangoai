"""
memory/episodic_memory.py — SQLite-backed Episodic Memory.

Stores past actions, outcomes, and user corrections with timestamps.
Agents query it to learn from history. Uses aiosqlite for fully async I/O.

Schema:
    episodes: id, session_id, user_id, event_type, content, outcome,
              correction, is_summary, timestamp, metadata (JSON)

FTS5 virtual table on content + outcome for efficient text search.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Union

import aiosqlite

import logging
from memory.schemas import EpisodeRecord, EventType

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
DB_PATH = os.getenv(
    "EPISODIC_DB_PATH",
    str(Path.home() / ".agentic_os" / "episodic.db"),
)

# DDL ─────────────────────────────────────────────────────────────────────────

_CREATE_EPISODES_TABLE = """
CREATE TABLE IF NOT EXISTS episodes (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL DEFAULT 'default',
    user_id     TEXT NOT NULL DEFAULT 'default',
    event_type  TEXT NOT NULL DEFAULT 'custom',
    content     TEXT NOT NULL,
    outcome     TEXT,
    correction  TEXT,
    is_summary  INTEGER NOT NULL DEFAULT 0,
    timestamp   TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}'
);
"""

_CREATE_FTS_TABLE = """
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    id UNINDEXED,
    content,
    outcome
);
"""

_CREATE_FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
    INSERT INTO episodes_fts(id, content, outcome)
    VALUES (new.id, new.content, COALESCE(new.outcome, ''));
END;

CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
    DELETE FROM episodes_fts WHERE id = old.id;
END;

CREATE TRIGGER IF NOT EXISTS episodes_au AFTER UPDATE ON episodes BEGIN
    UPDATE episodes_fts SET 
        content = new.content,
        outcome = COALESCE(new.outcome, '')
    WHERE id = old.id;
END;
"""

_CREATE_INDICES = """
CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
CREATE INDEX IF NOT EXISTS idx_episodes_user    ON episodes(user_id);
CREATE INDEX IF NOT EXISTS idx_episodes_ts      ON episodes(timestamp);
CREATE INDEX IF NOT EXISTS idx_episodes_summary ON episodes(is_summary);
"""


class EpisodicMemory:
    """
    Async SQLite episodic memory store.

    Maintains a timestamped log of agent actions, outcomes, and corrections.
    Supports text search via FTS5 and time-range queries.
    Used by the MemoryAgent for consolidation (compressing old episodes).
    """

    def __init__(self, db_path: str = DB_PATH) -> None:
        self._db_path = db_path
        self._initialized = False

    async def _ensure_init(self) -> None:
        """Create the DB and tables on first access."""
        if self._initialized:
            return
        db_dir = Path(self._db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA foreign_keys=ON")
            await db.execute(_CREATE_EPISODES_TABLE)
            await db.execute(_CREATE_FTS_TABLE)
            # FTS triggers — execute one at a time
            for stmt in _CREATE_FTS_TRIGGERS.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    try:
                        await db.execute(stmt)
                    except aiosqlite.OperationalError:
                        pass  # Trigger already exists
            for stmt in _CREATE_INDICES.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    await db.execute(stmt)
            await db.commit()
        self._initialized = True
        logger.info("EpisodicMemory initialized: db=%s", self._db_path)

    # ── Write Operations ──────────────────────────────────────────────────────

    async def record(
        self,
        content: str,
        event_type: Union[EventType, str] = EventType.CUSTOM,
        outcome: Optional[str] = None,
        correction: Optional[str] = None,
        session_id: str = "default",
        user_id: str = "default",
        metadata: Optional[dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> str:
        """
        Store a new episodic event. Returns the episode_id.
        """
        # Coerce string to EventType
        if isinstance(event_type, str):
            try:
                event_type = EventType(event_type)
            except ValueError:
                # Fallback if unknown string
                event_type = EventType.CUSTOM
        await self._ensure_init()
        episode_id = str(uuid.uuid4())
        ts = (timestamp or datetime.utcnow()).isoformat()
        meta_json = json.dumps(metadata or {}, default=str)

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO episodes
                    (id, session_id, user_id, event_type, content,
                     outcome, correction, is_summary, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    episode_id,
                    session_id,
                    user_id,
                    event_type.value,
                    content,
                    outcome,
                    correction,
                    ts,
                    meta_json,
                ),
            )
            await db.commit()

        logger.debug(
            "Episode recorded: episode_id=%s, event_type=%s",
            episode_id,
            event_type.value,
        )
        return episode_id

    async def add_correction(
        self,
        episode_id: str,
        correction: str,
    ) -> None:
        """Attach a user correction to an existing episode."""
        await self._ensure_init()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "UPDATE episodes SET correction = ? WHERE id = ?",
                (correction, episode_id),
            )
            await db.commit()
        logger.info("Correction added: episode_id=%s", episode_id)

    async def mark_as_summary(
        self,
        episode_ids: list[str],
        summary_content: str,
        session_id: str = "default",
        user_id: str = "default",
    ) -> str:
        """
        Replace a batch of episodes with a single consolidated summary.

        Marks originals as `is_summary=True` (soft-delete) and inserts
        the new summary episode. Returns the summary's episode_id.
        """
        await self._ensure_init()
        summary_id = str(uuid.uuid4())
        ts = datetime.utcnow().isoformat()
        meta = json.dumps({"consolidated_from": episode_ids})

        async with aiosqlite.connect(self._db_path) as db:
            # Soft-delete originals
            placeholders = ",".join("?" * len(episode_ids))
            await db.execute(
                f"UPDATE episodes SET is_summary = 1 WHERE id IN ({placeholders})",
                episode_ids,
            )
            # Insert summary
            await db.execute(
                """
                INSERT INTO episodes
                    (id, session_id, user_id, event_type, content,
                     outcome, correction, is_summary, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, NULL, NULL, 0, ?, ?)
                """,
                (
                    summary_id,
                    session_id,
                    user_id,
                    EventType.MEMORY_CONSOLIDATION.value,
                    summary_content,
                    ts,
                    meta,
                ),
            )
            await db.commit()

        logger.info(
            "Episodes consolidated: original_count=%d, summary_id=%s",
            len(episode_ids),
            summary_id,
        )
        return summary_id

    # ── Read Operations ───────────────────────────────────────────────────────

    async def query_recent(
        self,
        hours: int = 24,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        include_summaries: bool = False,
        limit: int = 100,
    ) -> list[EpisodeRecord]:
        """Return episodes from the last `hours` hours."""
        await self._ensure_init()
        since = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

        conditions = ["timestamp >= ?", "is_summary = ?"]
        params: list[Any] = [since, 0 if not include_summaries else -1]

        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)

        where = " AND ".join(conditions)
        sql = f"""
            SELECT * FROM episodes
            WHERE {where}
            ORDER BY timestamp DESC
            LIMIT ?
        """
        params.append(limit)

        return await self._fetch_episodes(sql, params)

    async def query_similar(
        self,
        text: str,
        top_k: int = 10,
        user_id: Optional[str] = None,
    ) -> list[EpisodeRecord]:
        """Full-text search over episode content and outcome via FTS5."""
        await self._ensure_init()
        # Clean search text
        query_term = text.replace('"', '""').strip()
        if not query_term:
            return []

        base_sql = """
            SELECT e.* FROM episodes e
            JOIN episodes_fts f ON e.id = f.id
            WHERE f MATCH ?
              AND e.is_summary = 0
        """
        params: list[Any] = [query_term]

        if user_id:
            base_sql += " AND e.user_id = ?"
            params.append(user_id)

        base_sql += " ORDER BY rank LIMIT ?"
        params.append(top_k)

        try:
            return await self._fetch_episodes(base_sql, params)
        except Exception as exc:
            # Fallback to LIKE if FTS fails (e.g., very short queries)
            logger.warning("FTS5 search failed, falling back to LIKE: %s", exc)
            like_sql = """
                SELECT * FROM episodes
                WHERE (content LIKE ? OR outcome LIKE ?)
                  AND is_summary = 0
                ORDER BY timestamp DESC LIMIT ?
            """
            like_param = f"%{text}%"
            like_params: list[Any] = [like_param, like_param, top_k]
            if user_id:
                like_sql = like_sql.replace(
                    "AND is_summary = 0",
                    "AND is_summary = 0 AND user_id = ?",
                )
                like_params.insert(-1, user_id)
            return await self._fetch_episodes(like_sql, like_params)

    async def get_episode(self, episode_id: str) -> Optional[EpisodeRecord]:
        """Fetch a single episode by ID."""
        await self._ensure_init()
        rows = await self._fetch_episodes(
            "SELECT * FROM episodes WHERE id = ?", [episode_id]
        )
        return rows[0] if rows else None

    async def get_episodes_older_than(
        self,
        days: int = 7,
        include_already_summarized: bool = False,
        limit: int = 200,
    ) -> list[EpisodeRecord]:
        """Get raw (non-summary) episodes older than `days` days."""
        await self._ensure_init()
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        params: list[Any] = [cutoff]
        extra = ""
        if not include_already_summarized:
            extra = "AND is_summary = 0"
        sql = f"""
            SELECT * FROM episodes
            WHERE timestamp < ?
            {extra}
            ORDER BY timestamp ASC
            LIMIT ?
        """
        params.append(limit)
        return await self._fetch_episodes(sql, params)

    async def delete_episode(self, episode_id: str) -> None:
        """Hard-delete a single episode."""
        await self._ensure_init()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))
            await db.commit()

    async def count(self, include_summaries: bool = True) -> int:
        """Total episode count."""
        await self._ensure_init()
        where = "" if include_summaries else "WHERE is_summary = 0"
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                f"SELECT COUNT(*) FROM episodes {where}"
            ) as cursor:
                row = await cursor.fetchone()
                return row[0] if row else 0

    # ── Internal Helpers ──────────────────────────────────────────────────────

    async def _fetch_episodes(
        self, sql: str, params: list[Any]
    ) -> list[EpisodeRecord]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(sql, params) as cursor:
                rows = await cursor.fetchall()
        results = []
        for row in rows:
            try:
                results.append(
                    EpisodeRecord(
                        episode_id=row["id"],
                        session_id=row["session_id"],
                        user_id=row["user_id"],
                        event_type=row["event_type"],
                        content=row["content"],
                        outcome=row["outcome"],
                        correction=row["correction"],
                        is_summary=bool(row["is_summary"]),
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        metadata=json.loads(row["metadata"] or "{}"),
                    )
                )
            except Exception as exc:
                logger.warning("Failed to parse episode row: %s", exc)
        return results
