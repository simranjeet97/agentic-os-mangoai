"""
core/router.py — Intent routing and conditional edge logic for LangGraph.
Determines which node to visit next based on current state.
"""

from __future__ import annotations

from typing import Literal

from core.logging_config import get_logger
from core.state import AgentState, TaskStatus

logger = get_logger(__name__)

MAX_ITERATIONS = int(__import__("os").getenv("MAX_AGENT_ITERATIONS", "25"))


def route_after_guardrail(
    state: AgentState,
) -> Literal["memory_load", "error"]:
    """After guardrail evaluation, decide to proceed or block."""
    if state.get("requires_approval"):
        logger.warning(
            "Task blocked by guardrails",
            task_id=state["task_id"],
            violations=state.get("guardrail_result", {}).get("violations", []),
        )
        return "error"
    return "memory_load"


def route_after_planning(
    state: AgentState,
) -> Literal["executor", "error"]:
    """After planning, validate there is a non-empty plan."""
    plan = state.get("plan", [])
    if not plan:
        logger.error("Planner produced empty plan", task_id=state["task_id"])
        return "error"
    return "executor"


def route_after_execution(
    state: AgentState,
) -> Literal["executor", "memory_save", "error"]:
    """
    After each execution step:
    - If more steps remain → continue execution loop
    - If all done → save memory
    - If too many iterations → bail out with error
    - If failed → error handler
    """
    status = state.get("status", TaskStatus.EXECUTING.value)
    iterations = state.get("iterations", 0)
    plan = state.get("plan", [])
    current_idx = state.get("current_step_index", 0)

    if status == TaskStatus.FAILED.value:
        return "error"

    if iterations >= MAX_ITERATIONS:
        logger.error(
            "Max iterations exceeded",
            task_id=state["task_id"],
            iterations=iterations,
            max=MAX_ITERATIONS,
        )
        return "error"

    if current_idx >= len(plan):
        logger.info(
            "All steps executed, saving memory",
            task_id=state["task_id"],
            total_steps=len(plan),
        )
        return "memory_save"

    return "executor"
