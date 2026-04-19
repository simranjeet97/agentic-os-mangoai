"""
api/websocket/handler.py — Real-time WebSocket handler wired to OrchestratorGraph.

Protocol:
  Client → Server:  {"goal": "...", "user_id": "...", "session_id": "...", "metadata": {}}
  Server → Client:  {"event": "node_update", "node": "...", "task_id": "...", "data": {...}}
                    {"event": "complete",    "task_id": "...", "status": "..."}
                    {"event": "error",       "message": "..."}
                    {"event": "interrupted", "reason": "..."}

Stop signal from client:
  {"action": "stop"}  → interrupts the current session

Approval signal from client:
  {"action": "approve", "action_id": "...", "approved": true, "feedback": "..."}
"""

from __future__ import annotations

import json
from typing import Any, Optional

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError

from core.orchestrator_graph import get_orchestrator_graph
from core.session_manager import get_session_manager

ws_router = APIRouter()
logger = structlog.get_logger(__name__)


# ── Request model ─────────────────────────────────────────────────────────────


class WSRunRequest(BaseModel):
    goal: str
    user_id: str = "anonymous"
    session_id: Optional[str] = None
    metadata: dict[str, Any] = {}


# ── Connection manager ────────────────────────────────────────────────────────


class ConnectionManager:
    """Track active WebSocket connections by session_id."""

    def __init__(self) -> None:
        self.active: dict[str, WebSocket] = {}

    async def connect(self, session_id: str, ws: WebSocket) -> None:
        await ws.accept()
        self.active[session_id] = ws
        logger.info("WebSocket connected", session_id=session_id)

    def disconnect(self, session_id: str) -> None:
        self.active.pop(session_id, None)
        logger.info("WebSocket disconnected", session_id=session_id)

    async def send(self, session_id: str, event: dict[str, Any]) -> None:
        ws = self.active.get(session_id)
        if ws:
            try:
                await ws.send_json(event)
            except Exception as exc:
                logger.warning("WS send failed", session_id=session_id, error=str(exc))


manager = ConnectionManager()


# ── WebSocket endpoint ────────────────────────────────────────────────────────


@ws_router.websocket("/ws/{session_id}")
@ws_router.websocket("/ws/stream/{session_id}")
async def websocket_stream(ws: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time agent execution streaming.

    One connection handles multiple sequential task runs.
    Send {"action": "stop"} mid-task to interrupt.
    """
    await manager.connect(session_id, ws)

    sm = get_session_manager()
    graph = get_orchestrator_graph()

    try:
        while True:
            # ── Receive message ───────────────────────────────────────────────
            try:
                raw = await ws.receive_text()
                payload = json.loads(raw)
            except (json.JSONDecodeError, WebSocketDisconnect):
                break

            # ── Handle stop action ────────────────────────────────────────────
            if payload.get("action") == "stop":
                await sm.interrupt(session_id, reason="WebSocket stop signal")
                await ws.send_json({"event": "interrupted", "reason": "User stopped task"})
                continue
                
            # ── Handle approve action ─────────────────────────────────────────
            if payload.get("action") == "approve":
                from api.state import pending_approvals
                action_id = payload.get("action_id")
                if action_id in pending_approvals:
                    pending_approvals[action_id]["status"] = "approved" if payload.get("approved") else "rejected"
                    pending_approvals[action_id]["feedback"] = payload.get("feedback", "")
                    await ws.send_json({"event": "approval_processed", "action_id": action_id})
                else:
                    await ws.send_json({"event": "error", "message": f"Pending action {action_id} not found"})
                continue

            # ── Parse run request ─────────────────────────────────────────────
            try:
                request = WSRunRequest(**payload)
            except ValidationError as exc:
                await ws.send_json({"event": "error", "message": f"Invalid payload: {exc}"})
                continue

            # Resolve session
            session = await sm.get_or_create(request.user_id)
            effective_session_id = request.session_id or session.session_id

            logger.info(
                "WS task received",
                session_id=effective_session_id,
                goal=request.goal[:80],
            )

            # ── Stream graph updates ──────────────────────────────────────────
            try:
                async for event in graph.stream(
                    user_input=request.goal,
                    user_id=request.user_id,
                    session_id=effective_session_id,
                    metadata=request.metadata,
                ):
                    node = event.get("node", "")
                    update = event.get("update", {})

                    await ws.send_json({
                        "event": "node_update",
                        "node": node,
                        "task_id": event.get("task_id"),
                        "data": _serialize(update),
                    })

                # Final completion event
                await ws.send_json({
                    "event": "complete",
                    "session_id": effective_session_id,
                    "message": "Task execution completed",
                })

            except Exception as exc:
                logger.error("WS streaming error", error=str(exc), exc_info=True)
                await ws.send_json({"event": "error", "message": str(exc)})

    except WebSocketDisconnect:
        logger.info("WS client disconnected", session_id=session_id)
    except Exception as exc:
        logger.error("WS handler error", session_id=session_id, error=str(exc))
    finally:
        manager.disconnect(session_id)


# ── Serialiser ────────────────────────────────────────────────────────────────


def _serialize(obj: Any) -> Any:
    """Recursively make objects JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "content"):   # LangChain BaseMessage
        return {"type": type(obj).__name__, "content": str(obj.content)}
    try:
        import json as _json
        _json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)
