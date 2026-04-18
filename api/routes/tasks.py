"""
api/routes/tasks.py — Manage, list, and cancel tasks.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel

from core.logging_config import get_logger
from core.session_manager import get_session_manager
from api.state import task_store
from api.auth.users import get_current_user

router = APIRouter()
logger = get_logger(__name__)

class TaskSummary(BaseModel):
    task_id: str
    session_id: str
    status: str
    goal: str
    created_at: str

@router.get("/", response_model=List[TaskSummary], summary="List tasks")
async def list_tasks(
    status: Optional[str] = Query(None, description="Filter by status (pending, running, completed, failed)"),
    current_user=Depends(get_current_user)
):
    """List all tasks, optionally filtering by status."""
    result = []
    for t_id, task in task_store.items():
        if status and task.get("status") != status:
            continue
        result.append(TaskSummary(
            task_id=task.get("task_id", t_id),
            session_id=task.get("session_id", ""),
            status=task.get("status", "unknown"),
            goal=task.get("goal", ""),
            created_at=task.get("created_at", "")
        ))
    return result

@router.delete("/{task_id}", summary="Cancel a task")
async def cancel_task(task_id: str, current_user=Depends(get_current_user)):
    """Cancel a pending or running task."""
    task = task_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found")

    session_id = task.get("session_id")
    if session_id:
        sm = get_session_manager()
        await sm.interrupt(session_id, reason="API cancel request")

    task_store[task_id]["status"] = "cancelled"
    logger.info("Task cancel requested via tasks API", task_id=task_id)
    return {"task_id": task_id, "status": "cancelled"}
