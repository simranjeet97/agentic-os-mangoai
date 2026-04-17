"""
api/routes/agent.py — Agent task submission and status endpoints.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field

from core.logging_config import get_logger
from core.orchestrator import get_orchestrator

router = APIRouter()
logger = get_logger(__name__)

# In-memory task store (replace with Redis in production for multi-process)
_task_store: dict[str, dict[str, Any]] = {}


class AgentRunRequest(BaseModel):
    goal: str = Field(..., min_length=1, max_length=10000, description="The task or goal for the agent")
    user_id: str = Field(default="anonymous", description="User identifier for RBAC and memory")
    session_id: Optional[str] = Field(default=None, description="Session ID for memory continuity")
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRunResponse(BaseModel):
    task_id: str
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


@router.post(
    "/run",
    response_model=AgentRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a new agent task",
)
async def run_agent(request: AgentRunRequest, background_tasks: BackgroundTasks):
    """
    Submit a new goal/task to the agent orchestrator.
    Returns immediately with a task_id; use /status/{task_id} to poll.
    """
    task_id = str(uuid.uuid4())
    _task_store[task_id] = {"status": "pending", "goal": request.goal, "task_id": task_id}

    logger.info("Agent task submitted", task_id=task_id, goal=request.goal[:80])

    async def _run_task():
        try:
            orchestrator = get_orchestrator()
            final_state = await orchestrator.run(
                user_input=request.goal,
                user_id=request.user_id,
                session_id=request.session_id,
                metadata=request.metadata,
            )
            _task_store[task_id] = dict(final_state)
        except Exception as exc:
            logger.error("Background task failed", task_id=task_id, error=str(exc))
            _task_store[task_id]["status"] = "failed"
            _task_store[task_id]["error"] = str(exc)

    background_tasks.add_task(_run_task)

    return AgentRunResponse(
        task_id=task_id,
        status="pending",
        message="Task submitted. Use GET /api/v1/agent/status/{task_id} to check progress.",
    )


@router.get(
    "/status/{task_id}",
    response_model=AgentStatusResponse,
    summary="Get task status and results",
)
async def get_task_status(task_id: str):
    """Retrieve status and results for a submitted agent task."""
    task = _task_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

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


@router.delete(
    "/cancel/{task_id}",
    status_code=status.HTTP_200_OK,
    summary="Cancel a running task",
)
async def cancel_task(task_id: str):
    """Mark a task as cancelled. Note: does not interrupt running background tasks."""
    if task_id not in _task_store:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    _task_store[task_id]["status"] = "cancelled"
    return {"task_id": task_id, "status": "cancelled"}
