"""
api/routes/agent.py — Agent task endpoints wired to OrchestratorGraph.

Endpoints:
  POST   /run              — submit a task (async, returns task_id)
  GET    /status/{task_id} — poll task status / results
  DELETE /cancel/{task_id} — interrupt a running task via SessionManager
  GET    /stream/{task_id} — SSE streaming of node-by-node updates
  GET    /intent           — parse intent without running a task
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, AsyncIterator, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.logging_config import get_logger
from core.orchestrator_graph import get_orchestrator_graph
from core.session_manager import get_session_manager
from api.state import task_store, sse_queues

router = APIRouter()
logger = get_logger(__name__)


# ── Request / Response models ─────────────────────────────────────────────────


class AgentRunRequest(BaseModel):
    goal: str = Field(..., min_length=1, max_length=10_000)
    user_id: str = Field(default="anonymous")
    session_id: Optional[str] = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRunResponse(BaseModel):
    task_id: str
    session_id: str
    status: str
    message: str


class AgentStatusResponse(BaseModel):
    task_id: str
    status: str
    goal: str
    iterations: int
    plan: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    error: Optional[str]
    created_at: str
    updated_at: str


class IntentRequest(BaseModel):
    text: str


# ── POST /run ─────────────────────────────────────────────────────────────────


@router.post(
    "/run",
    response_model=AgentRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a new agent task",
)
async def run_agent(request: AgentRunRequest, background_tasks: BackgroundTasks):
    """
    Submit a goal to the OrchestratorGraph.  Returns immediately with a task_id.
    Poll GET /status/{task_id} or subscribe via GET /stream/{task_id} (SSE).
    """
    sm = get_session_manager()
    session = await sm.get_or_create(request.user_id)
    session_id = request.session_id or session.session_id

    task_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    sse_queues[task_id] = queue
    task_store[task_id] = {
        "task_id": task_id,
        "session_id": session_id,
        "status": "pending",
        "goal": request.goal,
        "iterations": 0,
        "plan": [],
        "tool_results": [],
        "error": None,
        "created_at": "",
        "updated_at": "",
    }

    logger.info("Task submitted", task_id=task_id, goal=request.goal[:80], user_id=request.user_id)

    async def _execute():
        graph = get_orchestrator_graph()
        try:
            # Track session's current task
            session.current_task_id = task_id
            await sm.update(session)

            async for event in graph.stream(
                user_input=request.goal,
                user_id=request.user_id,
                session_id=session_id,
                metadata=request.metadata,
            ):
                node = event.get("node", "")
                update = event.get("update", {})
                # Update in-memory store
                task_store[task_id].update({
                    k: v for k, v in update.items()
                    if k in ("status", "plan", "tool_results", "iterations", "error", "updated_at",
                             "created_at", "active_agent", "current_step_index")
                })
                # Push to SSE queue
                await queue.put({"event": "node_update", "node": node, "task_id": task_id,
                                 "data": _serialize(update)})

            task_store[task_id]["status"] = task_store[task_id].get("status", "completed")
            await queue.put({"event": "complete", "task_id": task_id,
                             "status": task_store[task_id]["status"]})
        except Exception as exc:
            logger.error("Background task error", task_id=task_id, error=str(exc))
            task_store[task_id]["status"] = "failed"
            task_store[task_id]["error"] = str(exc)
            await queue.put({"event": "error", "task_id": task_id, "message": str(exc)})
        finally:
            await queue.put(None)  # sentinel: close SSE

    background_tasks.add_task(_execute)

    return AgentRunResponse(
        task_id=task_id,
        session_id=session_id,
        status="pending",
        message=f"Task submitted. Stream: GET /api/v1/agent/stream/{task_id}",
    )


# ── GET /status/{task_id} ─────────────────────────────────────────────────────


@router.get(
    "/status/{task_id}",
    response_model=AgentStatusResponse,
    summary="Poll task status",
)
async def get_task_status(task_id: str):
    task = task_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found")
    return AgentStatusResponse(
        task_id=task_id,
        status=task.get("status", "unknown"),
        goal=task.get("goal", ""),
        iterations=task.get("iterations", 0),
        plan=task.get("plan", []),
        tool_results=task.get("tool_results", []),
        error=task.get("error"),
        created_at=task.get("created_at", ""),
        updated_at=task.get("updated_at", ""),
    )


# ── DELETE /cancel/{task_id} ──────────────────────────────────────────────────


@router.delete(
    "/cancel/{task_id}",
    status_code=status.HTTP_200_OK,
    summary="Cancel / interrupt a running task",
)
async def cancel_task(task_id: str):
    """
    Sets the interrupt flag on the session so the orchestrator loop
    detects it at the next execute_with_guardrails step.
    """
    task = task_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found")

    session_id = task.get("session_id")
    if session_id:
        sm = get_session_manager()
        await sm.interrupt(session_id, reason="API cancel request")

    task_store[task_id]["status"] = "cancelled"
    logger.info("Task cancel requested", task_id=task_id)
    return {"task_id": task_id, "status": "cancelled"}


# ── GET /stream/{task_id} — SSE ───────────────────────────────────────────────


@router.get(
    "/stream/{task_id}",
    summary="SSE stream of live node updates for a task",
    response_class=StreamingResponse,
)
async def stream_task(task_id: str, request: Request):
    """
    Server-Sent Events stream.  Connect with EventSource in the browser:

        const es = new EventSource('/api/v1/agent/stream/<task_id>');
        es.onmessage = e => console.log(JSON.parse(e.data));
    """
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found")

    queue = sse_queues.get(task_id)
    if queue is None:
        # Task already completed — return stored state immediately
        async def _static():
            task = task_store[task_id]
            data = json.dumps({"event": "complete", "task_id": task_id,
                               "status": task.get("status", "unknown")})
            yield f"data: {data}\n\n"

        return StreamingResponse(_static(), media_type="text/event-stream")

    async def _event_generator() -> AsyncIterator[str]:
        try:
            while True:
                # Respect client disconnect
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if item is None:  # sentinel
                    break
                yield f"data: {json.dumps(item)}\n\n"
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── POST /intent — parse only ─────────────────────────────────────────────────


@router.post(
    "/intent",
    summary="Parse intent without executing a task",
)
async def parse_intent(request: IntentRequest):
    """Classify + decompose user text into intent metadata. No task is run."""
    from core.intent_parser import IntentParser
    parser = IntentParser(use_llm=True)
    parsed = await parser.parse(request.text)
    return parsed.model_dump()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _serialize(obj: Any) -> Any:
    """Recursively make objects JSON-safe."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "content"):           # LangChain message
        return {"type": type(obj).__name__, "content": str(obj.content)}
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)
