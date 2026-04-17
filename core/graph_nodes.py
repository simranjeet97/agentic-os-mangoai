"""
core/graph_nodes.py — LangGraph node implementations.
Each node is a pure async function: AgentState → dict[str, Any]
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from core.logging_config import get_logger
from core.state import AgentState, AgentRole, TaskStatus

logger = get_logger(__name__)

# ── LLM Factory ──────────────────────────────────────────────────────────────

_llm_cache: dict[str, ChatOllama] = {}

def get_llm(model: str = "llama3.2:3b", temperature: float = 0.0) -> ChatOllama:
    """Return a cached ChatOllama instance."""
    key = f"{model}:{temperature}"
    if key not in _llm_cache:
        _llm_cache[key] = ChatOllama(
            model=model,
            temperature=temperature,
            base_url="http://localhost:11434",
        )
    return _llm_cache[key]


# ── Node: Guardrail Check ────────────────────────────────────────────────────

async def guardrail_node(state: AgentState) -> dict[str, Any]:
    """
    Safety gate — runs before any execution.
    Delegates to the GuardianEngine for policy evaluation.
    """
    logger.info("Running guardrail check", task_id=state["task_id"])

    try:
        # Import here to avoid circular dependency
        from guardrails.guardian import GuardianEngine
        guardian = GuardianEngine()
        result = await guardian.evaluate(
            user_input=state["user_input"],
            user_id=state["user_id"],
            requested_capabilities=[],
        )
        guardrail_dict = result.model_dump()
    except Exception as exc:
        logger.warning("Guardrail evaluation failed, defaulting to pass", error=str(exc))
        guardrail_dict = {"passed": True, "risk_level": "low", "violations": [], "recommendations": []}

    return {
        "guardrail_result": guardrail_dict,
        "requires_approval": not guardrail_dict.get("passed", True),
        "status": TaskStatus.PLANNING.value if guardrail_dict.get("passed") else TaskStatus.WAITING.value,
        "updated_at": datetime.utcnow().isoformat(),
    }


# ── Node: Memory Load ────────────────────────────────────────────────────────

async def memory_load_node(state: AgentState) -> dict[str, Any]:
    """Hydrate working context from Redis + ChromaDB before planning."""
    logger.info("Loading memory context", session_id=state["session_id"])

    try:
        from memory.memory_manager import MemoryManager
        mm = MemoryManager()
        context = await mm.load_context(
            session_id=state["session_id"],
            user_id=state["user_id"],
            query=state["user_input"],
        )
    except Exception as exc:
        logger.warning("Memory load failed, starting fresh", error=str(exc))
        context = {}

    return {
        "memory": context,
        "updated_at": datetime.utcnow().isoformat(),
    }


# ── Node: Planner ────────────────────────────────────────────────────────────

PLANNER_SYSTEM_PROMPT = """\
You are the Planner agent in an Agentic AI OS.
Your job is to decompose the user's goal into a precise, ordered list of steps.

Each step must specify:
- description: what needs to be done
- agent: one of [planner, executor, file, web, system, code]
- dependencies: list of step_ids this step depends on (empty if none)

Return ONLY valid JSON in this exact format:
{
  "goal_summary": "...",
  "steps": [
    {"step_id": "s1", "description": "...", "agent": "web", "dependencies": []},
    {"step_id": "s2", "description": "...", "agent": "code", "dependencies": ["s1"]}
  ]
}
"""

async def planner_node(state: AgentState) -> dict[str, Any]:
    """Decompose the user's goal into an ordered task plan."""
    logger.info("Planning task", task_id=state["task_id"], goal=state["goal"])

    llm = get_llm(temperature=0.0)

    context_str = ""
    if state.get("memory"):
        episodic = state["memory"].get("episodic", [])
        if episodic:
            context_str = f"\nRelevant past context:\n{episodic[:3]}\n"

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"Goal: {state['goal']}{context_str}"),
    ]

    try:
        response = await llm.ainvoke(messages)
        import json, re
        content = response.content.strip()
        # Extract JSON from markdown code blocks if present
        match = re.search(r"```(?:json)?\n?(.*?)```", content, re.DOTALL)
        if match:
            content = match.group(1)
        plan_data = json.loads(content)
        plan = plan_data.get("steps", [])
        logger.info("Plan created", step_count=len(plan))
    except Exception as exc:
        logger.error("Planning failed", error=str(exc))
        plan = [{"step_id": "s1", "description": state["goal"], "agent": "executor", "dependencies": []}]

    return {
        "plan": plan,
        "status": TaskStatus.EXECUTING.value,
        "messages": [AIMessage(content=f"Plan created with {len(plan)} steps.")],
        "updated_at": datetime.utcnow().isoformat(),
    }


# ── Node: Executor Router ────────────────────────────────────────────────────

async def executor_node(state: AgentState) -> dict[str, Any]:
    """
    Pick the next pending step and dispatch it to the appropriate agent.
    Returns updated tool_results and advances current_step_index.
    """
    plan = state.get("plan", [])
    idx = state.get("current_step_index", 0)

    if idx >= len(plan):
        logger.info("All steps completed", task_id=state["task_id"])
        return {
            "status": TaskStatus.COMPLETED.value,
            "updated_at": datetime.utcnow().isoformat(),
        }

    step = plan[idx]
    agent_name = step.get("agent", "executor")
    logger.info("Executing step", step_id=step.get("step_id"), agent=agent_name)

    try:
        result = await _dispatch_to_agent(agent_name, step, state)
    except Exception as exc:
        logger.error("Step execution failed", step=step, error=str(exc))
        result = {"error": str(exc), "success": False}

    tool_results = list(state.get("tool_results", []))
    tool_results.append({
        "step_id": step.get("step_id"),
        "agent": agent_name,
        "result": result,
    })

    return {
        "tool_results": tool_results,
        "current_step_index": idx + 1,
        "iterations": state.get("iterations", 0) + 1,
        "active_agent": agent_name,
        "updated_at": datetime.utcnow().isoformat(),
    }


async def _dispatch_to_agent(
    agent_name: str, step: dict, state: AgentState
) -> dict[str, Any]:
    """Route a task step to the correct agent module."""
    from agents.planner.agent import PlannerAgent
    from agents.executor.agent import ExecutorAgent
    from agents.file.agent import FileAgent
    from agents.web.agent import WebAgent
    from agents.system.agent import SystemAgent
    from agents.code.agent import CodeAgent

    agent_map = {
        AgentRole.PLANNER.value: PlannerAgent,
        AgentRole.EXECUTOR.value: ExecutorAgent,
        AgentRole.FILE.value: FileAgent,
        AgentRole.WEB.value: WebAgent,
        AgentRole.SYSTEM.value: SystemAgent,
        AgentRole.CODE.value: CodeAgent,
    }

    AgentClass = agent_map.get(agent_name, ExecutorAgent)
    agent = AgentClass()
    return await agent.execute(step=step, state=state)


# ── Node: Memory Save ────────────────────────────────────────────────────────

async def memory_save_node(state: AgentState) -> dict[str, Any]:
    """Persist session results to Redis + ChromaDB after execution."""
    logger.info("Saving memory", session_id=state["session_id"])

    try:
        from memory.memory_manager import MemoryManager
        mm = MemoryManager()
        await mm.save_context(
            session_id=state["session_id"],
            user_id=state["user_id"],
            state=state,
        )
    except Exception as exc:
        logger.warning("Memory save failed", error=str(exc))

    return {"updated_at": datetime.utcnow().isoformat()}


# ── Node: Error Handler ──────────────────────────────────────────────────────

async def error_node(state: AgentState) -> dict[str, Any]:
    """Handle unrecoverable errors and persist failure state."""
    logger.error("Task failed", task_id=state["task_id"], error=state.get("error"))
    return {
        "status": TaskStatus.FAILED.value,
        "messages": [AIMessage(content=f"Task failed: {state.get('error', 'Unknown error')}")],
        "updated_at": datetime.utcnow().isoformat(),
    }
