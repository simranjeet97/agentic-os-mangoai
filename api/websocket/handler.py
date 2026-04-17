"""
api/websocket/handler.py — Real-time WebSocket handler for agent streaming.
Streams agent state updates to the frontend as JSON events.
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError

from core.orchestrator import get_orchestrator

ws_router = APIRouter()
logger = structlog.get_logger(__name__)


class WSRunRequest(BaseModel):
    goal: str
    user_id: str = "anonymous"
    session_id: str | None = None
    metadata: dict[str, Any] = {}


class ConnectionManager:
    """Manage active WebSocket connections."""

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
                logger.warning("WebSocket send failed", session_id=session_id, error=str(exc))


manager = ConnectionManager()


@ws_router.websocket("/ws/stream/{session_id}")
async def websocket_stream(ws: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time agent execution streaming.

    Client sends:
        {"goal": "...", "user_id": "...", "session_id": "..."}

    Server streams events:
        {"event": "node_update", "node": "planner", "data": {...}}
        {"event": "complete", "status": "completed", "task_id": "..."}
        {"event": "error", "message": "..."}
    """
    await manager.connect(session_id, ws)

    try:
        while True:
            try:
                raw = await ws.receive_text()
                payload = json.loads(raw)
                request = WSRunRequest(**payload)
            except (json.JSONDecodeError, ValidationError) as exc:
                await ws.send_json({"event": "error", "message": f"Invalid request: {exc}"})
                continue
            except WebSocketDisconnect:
                break

            logger.info(
                "WebSocket task received",
                session_id=session_id,
                goal=request.goal[:80],
            )

            # Stream agent updates
            orchestrator = get_orchestrator()
            try:
                async for update in orchestrator.stream(
                    user_input=request.goal,
                    user_id=request.user_id,
                    session_id=session_id,
                    metadata=request.metadata,
                ):
                    await ws.send_json({
                        "event": "node_update",
                        "node": update.get("node"),
                        "task_id": update.get("task_id"),
                        "data": _serialize(update.get("update", {})),
                    })

                await ws.send_json({
                    "event": "complete",
                    "session_id": session_id,
                    "message": "Task execution completed",
                })

            except Exception as exc:
                logger.error("WebSocket streaming error", error=str(exc), exc_info=True)
                await ws.send_json({"event": "error", "message": str(exc)})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected", session_id=session_id)
    finally:
        manager.disconnect(session_id)


def _serialize(obj: Any) -> Any:
    """Convert LangChain messages and other non-JSON objects for serialization."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return {"type": type(obj).__name__, "content": str(obj)}
    return obj
