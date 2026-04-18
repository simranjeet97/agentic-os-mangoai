"""
api/routes/session.py — Session management endpoints.

Endpoints:
  GET    /                         — list all active sessions
  GET    /{session_id}             — get session details
  POST   /                         — create a new session
  DELETE /{session_id}             — delete a session
  POST   /{session_id}/interrupt   — interrupt (stop) current task
  POST   /{session_id}/resume      — resume interrupted task
  GET    /{session_id}/history     — conversation turn history
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from core.logging_config import get_logger
from core.session_manager import get_session_manager

router = APIRouter()
logger = get_logger(__name__)


# ── Models ────────────────────────────────────────────────────────────────────


class CreateSessionRequest(BaseModel):
    user_id: str
    context: dict[str, Any] = Field(default_factory=dict)


class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    status: str
    current_task_id: Optional[str]
    interrupt_flag: bool
    turn_count: int
    created_at: str
    last_active: str


class InterruptRequest(BaseModel):
    reason: str = "User requested stop"


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get(
    "/",
    summary="List all active sessions",
)
async def list_sessions():
    sm = get_session_manager()
    sessions = [s for s in sm._store.values() if not s.is_expired]
    return {
        "count": len(sessions),
        "sessions": [
            {
                "session_id": s.session_id,
                "user_id": s.user_id,
                "status": s.status.value,
                "turn_count": len(s.history),
                "interrupt_flag": s.interrupt_flag,
                "last_active": s.last_active.isoformat(),
            }
            for s in sessions
        ],
    }


@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    summary="Create a new session",
)
async def create_session(request: CreateSessionRequest):
    sm = get_session_manager()
    session = await sm.create(user_id=request.user_id, context=request.context)
    return _session_to_dict(session)


@router.get(
    "/{session_id}",
    summary="Get session details",
)
async def get_session(session_id: str):
    sm = get_session_manager()
    session = await sm.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found or expired")
    return _session_to_dict(session)


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a session",
)
async def delete_session(session_id: str):
    sm = get_session_manager()
    session = await sm.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")
    await sm.delete(session_id)
    return {"session_id": session_id, "deleted": True}


@router.post(
    "/{session_id}/interrupt",
    summary="Interrupt the current task in a session",
)
async def interrupt_session(session_id: str, request: InterruptRequest):
    """
    Sets the interrupt_flag on the session so the orchestrator
    stops at the next guardrail check.
    """
    sm = get_session_manager()
    ok = await sm.interrupt(session_id, reason=request.reason)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")
    return {"session_id": session_id, "interrupted": True, "reason": request.reason}


@router.post(
    "/{session_id}/resume",
    summary="Resume an interrupted session",
)
async def resume_session(session_id: str):
    """
    Clears the interrupt_flag and returns the saved task snapshot so the
    caller can pick up execution from where it left off.
    """
    sm = get_session_manager()
    snap = await sm.resume(session_id)
    return {
        "session_id": session_id,
        "resumed": True,
        "snapshot": snap.model_dump() if snap else None,
    }


@router.get(
    "/{session_id}/history",
    summary="Get conversation turn history",
)
async def get_history(session_id: str, limit: int = 20):
    sm = get_session_manager()
    session = await sm.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")
    turns = session.history[-limit:]
    return {"session_id": session_id, "count": len(turns), "turns": turns}


# ── Helper ────────────────────────────────────────────────────────────────────


def _session_to_dict(session) -> dict[str, Any]:
    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "status": session.status.value,
        "current_task_id": session.current_task_id,
        "interrupt_flag": session.interrupt_flag,
        "interrupt_reason": session.interrupt_reason,
        "turn_count": len(session.history),
        "created_at": session.created_at.isoformat(),
        "last_active": session.last_active.isoformat(),
        "snapshot": session.snapshot.model_dump() if session.snapshot else None,
    }
