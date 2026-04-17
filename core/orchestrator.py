"""
core/orchestrator.py — Main LangGraph orchestration engine.
Builds and compiles the full agent graph.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from core.graph_nodes import (
    error_node,
    executor_node,
    guardrail_node,
    memory_load_node,
    memory_save_node,
    planner_node,
)
from core.logging_config import get_logger, setup_logging
from core.router import (
    route_after_execution,
    route_after_guardrail,
    route_after_planning,
)
from core.state import AgentState, TaskStatus, create_initial_state

logger = get_logger(__name__)


class AgentOrchestrator:
    """
    Builds and manages the LangGraph agent execution graph.

    Graph topology:
      START → guardrail → [memory_load | error]
                         memory_load → planner → [executor | error]
                                                executor → [executor | memory_save | error]
                                                           memory_save → END
                                                error → END
    """

    def __init__(self) -> None:
        setup_logging()
        self._graph = self._build_graph()
        logger.info("AgentOrchestrator initialized")

    def _build_graph(self) -> StateGraph:
        """Construct and compile the LangGraph StateGraph."""
        builder = StateGraph(AgentState)

        # ── Register nodes ─────────────────────────────────────────────────
        builder.add_node("guardrail", guardrail_node)
        builder.add_node("memory_load", memory_load_node)
        builder.add_node("planner", planner_node)
        builder.add_node("executor", executor_node)
        builder.add_node("memory_save", memory_save_node)
        builder.add_node("error", error_node)

        # ── Entry point ────────────────────────────────────────────────────
        builder.add_edge(START, "guardrail")

        # ── Conditional routing ────────────────────────────────────────────
        builder.add_conditional_edges(
            "guardrail",
            route_after_guardrail,
            {"memory_load": "memory_load", "error": "error"},
        )
        builder.add_edge("memory_load", "planner")
        builder.add_conditional_edges(
            "planner",
            route_after_planning,
            {"executor": "executor", "error": "error"},
        )
        builder.add_conditional_edges(
            "executor",
            route_after_execution,
            {
                "executor": "executor",
                "memory_save": "memory_save",
                "error": "error",
            },
        )
        builder.add_edge("memory_save", END)
        builder.add_edge("error", END)

        checkpointer = MemorySaver()
        return builder.compile(checkpointer=checkpointer)

    async def run(
        self,
        user_input: str,
        user_id: str = "anonymous",
        session_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> AgentState:
        """
        Execute the full agent pipeline for a given user input.
        Returns the final AgentState.
        """
        initial_state = create_initial_state(
            user_input=user_input,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
        )

        config = {"configurable": {"thread_id": initial_state["task_id"]}}

        logger.info(
            "Starting agent task",
            task_id=initial_state["task_id"],
            user_id=user_id,
            goal=user_input[:120],
        )

        try:
            final_state = await self._graph.ainvoke(initial_state, config=config)
            logger.info(
                "Task completed",
                task_id=final_state.get("task_id"),
                status=final_state.get("status"),
                iterations=final_state.get("iterations"),
            )
            return final_state
        except Exception as exc:
            logger.error("Orchestrator error", error=str(exc), exc_info=True)
            initial_state["status"] = TaskStatus.FAILED.value
            initial_state["error"] = str(exc)
            return initial_state

    async def stream(
        self,
        user_input: str,
        user_id: str = "anonymous",
        session_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Stream intermediate agent state updates as they happen.
        Yields dicts with node name and partial state update.
        """
        initial_state = create_initial_state(
            user_input=user_input,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
        )
        config = {"configurable": {"thread_id": initial_state["task_id"]}}

        logger.info(
            "Streaming agent task",
            task_id=initial_state["task_id"],
            user_id=user_id,
        )

        async for event in self._graph.astream(initial_state, config=config, stream_mode="updates"):
            for node_name, update in event.items():
                yield {
                    "node": node_name,
                    "task_id": initial_state["task_id"],
                    "update": update,
                }


# ── Singleton instance ────────────────────────────────────────────────────────
_orchestrator: Optional[AgentOrchestrator] = None


def get_orchestrator() -> AgentOrchestrator:
    """Return the global orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator
