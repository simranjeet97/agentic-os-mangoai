"""
core/session_manager.py — Per-user session state with interruption support.

Features:
  - Session creation, lookup, and expiry
  - Interrupt / resume / cancel session mid-task
  - Context switching: user starts a new topic without finishing the old one
  - Stores session snapshots in memory (Redis-compatible interface)
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from core.logging_config import get_logger

logger = get_logger(__name__)


# ── Session models ────────────────────────────────────────────────────────────


class SessionStatus(str, Enum):
    ACTIVE = "active"
    INTERRUPTED = "interrupted"
    PAUSED = "paused"
    COMPLETED = "completed"
    EXPIRED = "expired"


STOP_PHRASES = frozenset(
    {"stop", "cancel", "abort", "halt", "pause", "hold on", "wait", "nevermind",
     "never mind", "quit", "exit", "kill it", "stop that", "stop it"}
)

RESUME_PHRASES = frozenset(
    {"resume", "continue", "go ahead", "ok", "proceed", "carry on", "keep going"}
)


class SessionSnapshot(BaseModel):
    """Point-in-time save of session state for interruption/resume."""

    snapshot_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_id: Optional[str] = None
    current_step: int = 0
    partial_results: list[Any] = Field(default_factory=list)
    user_input: str = ""
    saved_at: datetime = Field(default_factory=datetime.utcnow)


class Session(BaseModel):
    """Full per-user session state."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    status: SessionStatus = SessionStatus.ACTIVE

    # Active task tracking
    current_task_id: Optional[str] = None
    active_graph_state: Optional[dict[str, Any]] = None  # LangGraph AgentState snapshot

    # Interruption support
    interrupt_flag: bool = False
    interrupt_reason: Optional[str] = None
    snapshot: Optional[SessionSnapshot] = None

    # Context
    context: dict[str, Any] = Field(default_factory=dict)   # short-term KV store
    history: list[dict[str, Any]] = Field(default_factory=list)  # last N turns

    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)
    expires_in_seconds: int = 3600  # 1-hour TTL by default

    @property
    def is_expired(self) -> bool:
        cutoff = self.last_active + timedelta(seconds=self.expires_in_seconds)
        return datetime.utcnow() > cutoff

    def touch(self) -> None:
        self.last_active = datetime.utcnow()

    def add_turn(self, role: str, content: str) -> None:
        """Append a user/assistant turn to history (capped at 50)."""
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        })
        if len(self.history) > 50:
            self.history = self.history[-50:]

    def save_snapshot(self, task_id: str, step: int, results: list, user_input: str) -> None:
        self.snapshot = SessionSnapshot(
            task_id=task_id,
            current_step=step,
            partial_results=results,
            user_input=user_input,
        )
        self.status = SessionStatus.INTERRUPTED


# ── SessionManager ────────────────────────────────────────────────────────────


class SessionManager:
    """
    Manages the lifecycle of all user sessions.

    In-process store (dict) for dev. Swap _store for a Redis adapter in prod.

    Usage:
        sm = SessionManager()
        session = sm.get_or_create("user-42")
        sm.interrupt(session.session_id, "User said 'stop'")
        sm.resume(session.session_id)
    """

    def __init__(self, session_ttl_seconds: int = 3600, max_sessions: int = 1000) -> None:
        self._store: dict[str, Session] = {}    # session_id → Session
        self._user_index: dict[str, str] = {}   # user_id → session_id
        self._ttl = session_ttl_seconds
        self._max = max_sessions
        self._lock = asyncio.Lock()

    # ── CRUD ─────────────────────────────────────────────────────────────────

    async def create(self, user_id: str, context: Optional[dict] = None) -> Session:
        """Create a new session for the user, invalidating the previous one if any."""
        async with self._lock:
            session = Session(
                user_id=user_id,
                expires_in_seconds=self._ttl,
                context=context or {},
            )
            self._store[session.session_id] = session
            self._user_index[user_id] = session.session_id
            logger.info("Session created", session_id=session.session_id, user_id=user_id)
            return session

    async def get(self, session_id: str) -> Optional[Session]:
        """Fetch session by ID. Returns None if not found or expired."""
        session = self._store.get(session_id)
        if session is None:
            return None
        if session.is_expired:
            await self._expire(session_id)
            return None
        return session

    async def get_by_user(self, user_id: str) -> Optional[Session]:
        """Fetch active session for a user."""
        session_id = self._user_index.get(user_id)
        if session_id is None:
            return None
        return await self.get(session_id)

    async def get_or_create(self, user_id: str, context: Optional[dict] = None) -> Session:
        """Return existing active session or mint a new one."""
        session = await self.get_by_user(user_id)
        if session is not None and not session.is_expired:
            session.touch()
            return session
        return await self.create(user_id, context)

    async def update(self, session: Session) -> None:
        """Persist changes to a session object."""
        async with self._lock:
            session.touch()
            self._store[session.session_id] = session

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            session = self._store.pop(session_id, None)
            if session:
                self._user_index.pop(session.user_id, None)

    # ── Interruption ──────────────────────────────────────────────────────────

    async def interrupt(
        self,
        session_id: str,
        reason: str = "User requested stop",
        snapshot: Optional[dict] = None,
    ) -> bool:
        """
        Flag the session as interrupted so the orchestrator loop can check
        `session.interrupt_flag` and gracefully wind down.
        """
        session = await self.get(session_id)
        if session is None:
            return False
        session.interrupt_flag = True
        session.interrupt_reason = reason
        session.status = SessionStatus.INTERRUPTED
        if snapshot:
            session.snapshot = SessionSnapshot(**snapshot)
        await self.update(session)
        logger.info("Session interrupted", session_id=session_id, reason=reason)
        return True

    async def resume(self, session_id: str) -> Optional[SessionSnapshot]:
        """
        Clear the interrupt flag and return the saved snapshot (if any),
        so the orchestrator can pick up from where it left off.
        """
        session = await self.get(session_id)
        if session is None:
            return None
        snap = session.snapshot
        session.interrupt_flag = False
        session.interrupt_reason = None
        session.status = SessionStatus.ACTIVE
        await self.update(session)
        logger.info("Session resumed", session_id=session_id, has_snapshot=snap is not None)
        return snap

    # ── Context switching ─────────────────────────────────────────────────────

    async def switch_context(
        self,
        session_id: str,
        new_user_input: str,
        current_state: Optional[dict] = None,
    ) -> Session:
        """
        User starts talking about something completely different.
        Snapshot the current task and reset for a new one.
        """
        session = await self.get(session_id)
        if session is None:
            raise KeyError(f"Session {session_id} not found")

        # Save current context as snapshot
        if current_state and session.current_task_id:
            session.save_snapshot(
                task_id=session.current_task_id,
                step=current_state.get("current_step_index", 0),
                results=current_state.get("tool_results", []),
                user_input=current_state.get("user_input", ""),
            )

        # Reset for new task
        session.current_task_id = None
        session.active_graph_state = None
        session.interrupt_flag = False
        session.status = SessionStatus.ACTIVE
        session.add_turn("system", f"[Context switch] New topic: {new_user_input[:100]}")
        await self.update(session)
        logger.info("Context switched", session_id=session_id)
        return session

    # ── Signal detection ──────────────────────────────────────────────────────

    @staticmethod
    def detect_stop_signal(text: str) -> bool:
        """Return True if the user input looks like a stop command."""
        normalized = text.strip().lower().rstrip("!?.,:;")
        return normalized in STOP_PHRASES

    @staticmethod
    def detect_resume_signal(text: str) -> bool:
        """Return True if the user input is a resume command."""
        normalized = text.strip().lower().rstrip("!?.,:;")
        return normalized in RESUME_PHRASES

    # ── Housekeeping ──────────────────────────────────────────────────────────

    async def _expire(self, session_id: str) -> None:
        session = self._store.pop(session_id, None)
        if session:
            session.status = SessionStatus.EXPIRED
            self._user_index.pop(session.user_id, None)
            logger.info("Session expired", session_id=session_id, user_id=session.user_id)

    async def cleanup_expired(self) -> int:
        """Remove all expired sessions. Returns number cleaned up."""
        expired = [sid for sid, s in self._store.items() if s.is_expired]
        for sid in expired:
            await self._expire(sid)
        return len(expired)

    def active_session_count(self) -> int:
        return sum(1 for s in self._store.values() if s.status == SessionStatus.ACTIVE)


# ── Singleton ─────────────────────────────────────────────────────────────────

_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Return the global SessionManager singleton."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
