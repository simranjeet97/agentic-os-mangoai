"""
core/orchestrator_graph.py — Full LangGraph StateGraph orchestration engine.

Graph topology:
  START
    └─► parse_intent
          ├─► (CONVERSATION / SYSTEM_QUERY)    ──► respond_to_user ──► END
          └─► (task)
                └─► route_to_agent
                      └─► execute_with_guardrails
                            ├─► (interrupted)  ──► respond_to_user ──► END
                            ├─► (more steps)   ──► execute_with_guardrails (loop)
                            └─► (done/failed)
                                  └─► update_memory
                                        └─► respond_to_user ──► END

Streaming: every node emits partial state so the shell can print incremental output.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, AsyncIterator, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from core.logging_config import get_logger, setup_logging
from core.state import AgentRole, AgentState, TaskStatus, create_initial_state

logger = get_logger(__name__)

MAX_ITERATIONS = 25


# ═══════════════════════════════════════════════════════════════════════════════
# Node 1 — parse_intent
# ═══════════════════════════════════════════════════════════════════════════════


async def parse_intent_node(state: AgentState) -> dict[str, Any]:
    """
    Classify user input and enrich state with intent metadata.
    Populates metadata["intent_type"], metadata["required_agents"],
    metadata["urgency"], metadata["estimated_complexity"].
    """
    logger.info("parse_intent", task_id=state["task_id"])

    from core.intent_parser import IntentParser

    parser = IntentParser(use_llm=True)
    parsed = await parser.parse(state["user_input"])

    metadata = dict(state.get("metadata", {}))
    metadata.update({
        "intent_type": parsed.intent_type.value,
        "intent": parsed.intent,
        "required_agents": parsed.required_agents,
        "urgency": parsed.urgency,
        "estimated_complexity": parsed.estimated_complexity,
        "clarification_question": parsed.clarification_question,
    })

    # For CONVERSATION / SYSTEM_QUERY we can skip planning entirely
    if parsed.intent_type.value in ("CONVERSATION", "SYSTEM_QUERY", "CLARIFICATION_NEEDED"):
        return {
            "metadata": metadata,
            "goal": parsed.intent,
            "status": TaskStatus.EXECUTING.value,
            "updated_at": datetime.utcnow().isoformat(),
        }

    return {
        "metadata": metadata,
        "goal": parsed.intent,
        "status": TaskStatus.PLANNING.value,
        "updated_at": datetime.utcnow().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node 2 — route_to_agent  (plan decomposition)
# ═══════════════════════════════════════════════════════════════════════════════

_ROUTE_SYSTEM_PROMPT = """\
You are the Planner inside an Agentic AI OS.
Decompose the user's goal into an ordered list of steps.
Each step must specify:
  - step_id  (short unique id, e.g. "s1")
  - description (what to do)
  - agent  (one of: planner, executor, file, web, system, code)
  - dependencies (list of step_ids this step requires, empty list if none)

Return ONLY valid JSON:
{
  "goal_summary": "...",
  "steps": [
    {"step_id": "s1", "description": "...", "agent": "web", "dependencies": []},
    {"step_id": "s2", "description": "...", "agent": "file", "dependencies": ["s1"]}
  ]
}
"""


async def route_to_agent_node(state: AgentState) -> dict[str, Any]:
    """
    Decompose the user goal into a task plan and register subtasks with TaskQueue.
    For CONVERSATION / SYSTEM_QUERY the plan is a single synthetic step.
    """
    logger.info("route_to_agent", task_id=state["task_id"])

    intent_type = state.get("metadata", {}).get("intent_type", "SINGLE_AGENT_TASK")

    # ── Short-circuit for non-task intents ────────────────────────────────────
    if intent_type in ("CONVERSATION", "SYSTEM_QUERY"):
        plan = [{
            "step_id": "s1",
            "description": state["user_input"],
            "agent": "executor",
            "dependencies": [],
        }]
        return {
            "plan": plan,
            "status": TaskStatus.EXECUTING.value,
            "messages": [AIMessage(content="Handling your request directly.")],
            "updated_at": datetime.utcnow().isoformat(),
        }

    if intent_type == "CLARIFICATION_NEEDED":
        q = state.get("metadata", {}).get("clarification_question", "Could you clarify your request?")
        return {
            "plan": [],
            "status": TaskStatus.WAITING.value,
            "messages": [AIMessage(content=q)],
            "updated_at": datetime.utcnow().isoformat(),
        }

    # ── LLM planning ──────────────────────────────────────────────────────────
    import json, re, os
    from langchain_ollama import ChatOllama
    from langchain_core.messages import SystemMessage

    llm = ChatOllama(
        model=os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.2:3b"),
        temperature=0.0,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    memory_ctx = ""
    if state.get("memory"):
        episodic = state["memory"].get("episodic", [])[:2]
        if episodic:
            memory_ctx = f"\nRelevant context:\n{episodic}\n"

    messages = [
        SystemMessage(content=_ROUTE_SYSTEM_PROMPT),
        HumanMessage(content=f"Goal: {state['goal']}{memory_ctx}"),
    ]

    try:
        response = await llm.ainvoke(messages)
        content = response.content.strip()
        match = re.search(r"```(?:json)?\n?(.*?)```", content, re.DOTALL)
        if match:
            content = match.group(1)
        data = json.loads(content)
        plan = data.get("steps", [])
        logger.info("Plan produced", steps=len(plan))
    except Exception as exc:
        logger.error("Planning LLM failed", error=str(exc))
        plan = [{
            "step_id": "s1",
            "description": state["goal"],
            "agent": AgentRole.EXECUTOR.value,
            "dependencies": [],
        }]

    # Register with TaskQueue
    try:
        from core.task_queue import TaskQueue
        q = TaskQueue()
        for step in plan:
            await q.enqueue(
                name=str(step.get("description", "step")),
                priority=int(state.get("metadata", {}).get("urgency", 3)),
                dependencies=list(step.get("dependencies", [])),
                payload=step,
            )
    except Exception as exc:
        logger.warning("TaskQueue enqueue failed", error=str(exc))

    return {
        "plan": plan,
        "current_step_index": 0,
        "status": TaskStatus.EXECUTING.value,
        "messages": [AIMessage(content=f"Plan ready: {len(plan)} step(s).")],
        "updated_at": datetime.utcnow().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node 3 — execute_with_guardrails
# ═══════════════════════════════════════════════════════════════════════════════


async def execute_with_guardrails_node(state: AgentState) -> dict[str, Any]:
    """
    For each pending step:
      1. Run guardrail check.
      2. Check the session interrupt flag.
      3. Dispatch to the appropriate agent.
    Returns updated tool_results + advanced step index.
    """
    logger.info("execute_with_guardrails", task_id=state["task_id"])

    plan = state.get("plan", [])
    idx = state.get("current_step_index", 0)

    # ── Check interrupt ────────────────────────────────────────────────────────
    try:
        from core.session_manager import get_session_manager
        sm = get_session_manager()
        session = await sm.get(state["session_id"])
        if session and session.interrupt_flag:
            logger.info("Interrupt detected, stopping execution", session_id=state["session_id"])
            return {
                "status": TaskStatus.CANCELLED.value,
                "error": f"Interrupted: {session.interrupt_reason or 'user requested stop'}",
                "updated_at": datetime.utcnow().isoformat(),
            }
    except Exception as exc:
        logger.warning("Session interrupt check failed", error=str(exc))

    if idx >= len(plan):
        return {
            "status": TaskStatus.COMPLETED.value,
            "updated_at": datetime.utcnow().isoformat(),
        }

    step = plan[idx]
    agent_name = step.get("agent", AgentRole.EXECUTOR.value)

    # ── Guardrail ─────────────────────────────────────────────────────────────
    try:
        from guardrails.guardian import GuardianEngine
        guardian = GuardianEngine()
        gr = await guardian.evaluate(
            user_input=step.get("description", ""),
            user_id=state["user_id"],
            requested_capabilities=[agent_name],
        )
        guardrail_dict = gr.model_dump()
        if not gr.passed:
            logger.warning(
                "Step blocked by guardrails",
                step_id=step.get("step_id"),
                violations=gr.violations,
            )
            tool_results = list(state.get("tool_results", []))
            tool_results.append({
                "step_id": step.get("step_id"),
                "agent": agent_name,
                "result": {"success": False, "error": f"Guardrail violation: {gr.violations}"},
            })
            return {
                "guardrail_result": guardrail_dict,
                "tool_results": tool_results,
                "current_step_index": idx + 1,
                "iterations": state.get("iterations", 0) + 1,
                "updated_at": datetime.utcnow().isoformat(),
            }
    except Exception as exc:
        logger.warning("Guardrail check failed, allowing step", error=str(exc))
        guardrail_dict = {"passed": True, "risk_level": "low", "violations": [], "recommendations": []}

    # ── Execute step ──────────────────────────────────────────────────────────
    logger.info("Dispatching step", step_id=step.get("step_id"), agent=agent_name)
    try:
        result = await _dispatch_step(agent_name, step, state)
    except Exception as exc:
        logger.error("Step dispatch failed", error=str(exc))
        result = {"success": False, "error": str(exc)}

    tool_results = list(state.get("tool_results", []))
    tool_results.append({
        "step_id": step.get("step_id"),
        "agent": agent_name,
        "result": result,
    })

    msgs = [AIMessage(content=f"[{agent_name}] Step {step.get('step_id')}: {result.get('output', result.get('error', 'done'))}")]

    return {
        "guardrail_result": guardrail_dict,
        "tool_results": tool_results,
        "current_step_index": idx + 1,
        "iterations": state.get("iterations", 0) + 1,
        "active_agent": agent_name,
        "messages": msgs,
        "updated_at": datetime.utcnow().isoformat(),
    }


async def _dispatch_step(agent_name: str, step: dict, state: AgentState) -> dict[str, Any]:
    """Route a step to the correct agent module."""
    try:
        from agents.planner.agent import PlannerAgent
        from agents.executor.agent import ExecutorAgent
        from agents.file.agent import FileAgent
        from agents.web.agent import WebAgent
        from agents.system.agent import SystemAgent
        from agents.code.agent import CodeAgent

        agent_map = {
            AgentRole.PLANNER.value:  PlannerAgent,
            AgentRole.EXECUTOR.value: ExecutorAgent,
            AgentRole.FILE.value:     FileAgent,
            AgentRole.WEB.value:      WebAgent,
            AgentRole.SYSTEM.value:   SystemAgent,
            AgentRole.CODE.value:     CodeAgent,
        }
        AgentClass: Any = agent_map.get(agent_name, ExecutorAgent)
        agent = AgentClass()
        return await agent.execute(step=step, state=state)
    except ImportError:
        # Graceful stub for testing without all agents installed
        await asyncio.sleep(0.05)  # simulate work
        return {"success": True, "output": f"[stub] {step.get('description', 'done')}"}


# ═══════════════════════════════════════════════════════════════════════════════
# Node 4 — update_memory
# ═══════════════════════════════════════════════════════════════════════════════


async def update_memory_node(state: AgentState) -> dict[str, Any]:
    """Persist session results to Redis + ChromaDB."""
    logger.info("update_memory", session_id=state["session_id"])

    try:
        from memory.memory_manager import MemoryManager
        mm = MemoryManager()
        await mm.save_context(
            session_id=state["session_id"],
            user_id=state["user_id"],
            state=dict(state),
        )
        logger.info("Memory updated", session_id=state["session_id"])
    except Exception as exc:
        logger.warning("Memory update skipped", error=str(exc))

    # Update session last_active
    try:
        from core.session_manager import get_session_manager
        sm = get_session_manager()
        session = await sm.get(state["session_id"])
        if session:
            session.add_turn("assistant", _build_final_answer(state))
            await sm.update(session)
    except Exception as exc:
        logger.warning("Session update after memory save failed", error=str(exc))

    return {"updated_at": datetime.utcnow().isoformat()}


# ═══════════════════════════════════════════════════════════════════════════════
# Node 5 — respond_to_user
# ═══════════════════════════════════════════════════════════════════════════════


async def respond_to_user_node(state: AgentState) -> dict[str, Any]:
    """Synthesise a final natural-language answer from accumulated tool_results."""
    logger.info("respond_to_user", task_id=state["task_id"])

    answer = _build_final_answer(state)
    return {
        "status": TaskStatus.COMPLETED.value,
        "messages": [AIMessage(content=answer)],
        "updated_at": datetime.utcnow().isoformat(),
    }


def _build_final_answer(state: AgentState) -> str:
    """Build a human-readable summary from the execution results."""
    intent_type = state.get("metadata", {}).get("intent_type", "SINGLE_AGENT_TASK")
    status = state.get("status", "")

    if status == TaskStatus.CANCELLED.value:
        return f"⛔ Task stopped. {state.get('error', '')}"

    if status == TaskStatus.FAILED.value:
        return f"❌ Task failed: {state.get('error', 'unknown error')}"

    results = state.get("tool_results", [])
    if not results:
        # CONVERSATION or SYSTEM_QUERY
        if intent_type == "CONVERSATION":
            return "👋 Hello! I'm your Agentic OS. How can I help you today?"
        if intent_type == "SYSTEM_QUERY":
            plan = state.get("plan", [])
            return (
                f"🖥️  System status: running • "
                f"Session: {state['session_id'][:8]} • "
                f"Agents available: planner, executor, file, web, system, code"
            )
        return "✅ Done! No output was produced."

    # Summarise multi-step results
    lines = ["✅ Task complete!\n"]
    for r in results:
        out = r.get("result", {})
        success = out.get("success", True)
        icon = "✓" if success else "✗"
        step_id = r.get("step_id", "?")
        agent = r.get("agent", "?")
        output_text = out.get("output") or out.get("error") or "no output"
        if isinstance(output_text, (dict, list)):
            import json
            output_text = json.dumps(output_text, indent=2)[:500]
        lines.append(f"  {icon} [{agent}:{step_id}] {str(output_text)[:300]}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Conditional edge functions
# ═══════════════════════════════════════════════════════════════════════════════


def _route_after_intent(
    state: AgentState,
) -> Literal["route_to_agent", "respond_to_user"]:
    intent_type = state.get("metadata", {}).get("intent_type", "SINGLE_AGENT_TASK")
    if intent_type in ("CONVERSATION",):
        return "respond_to_user"
    return "route_to_agent"


def _route_after_route(
    state: AgentState,
) -> Literal["execute_with_guardrails", "respond_to_user"]:
    plan = state.get("plan", [])
    status = state.get("status", "")
    if not plan or status == TaskStatus.WAITING.value:
        return "respond_to_user"
    return "execute_with_guardrails"


def _route_after_execute(
    state: AgentState,
) -> Literal["execute_with_guardrails", "update_memory", "respond_to_user"]:
    status = state.get("status", "")
    iterations = state.get("iterations", 0)
    plan = state.get("plan", [])
    idx = state.get("current_step_index", 0)

    if status in (TaskStatus.CANCELLED.value, TaskStatus.FAILED.value):
        return "respond_to_user"

    if iterations >= MAX_ITERATIONS:
        return "update_memory"

    if idx >= len(plan):
        return "update_memory"

    return "execute_with_guardrails"


# ═══════════════════════════════════════════════════════════════════════════════
# OrchestratorGraph
# ═══════════════════════════════════════════════════════════════════════════════


class OrchestratorGraph:
    """
    LangGraph StateGraph with nodes:
      parse_intent → route_to_agent → execute_with_guardrails → update_memory → respond_to_user

    Supports multi-agent workflows, streaming, and async execution.

    Usage:
        graph = OrchestratorGraph()
        # Full run (returns final state)
        final_state = await graph.run("search the web for LangGraph and save a summary")

        # Streaming run (yields partial updates per node)
        async for event in graph.stream("do X then Y"):
            print(event["node"], event["update"])
    """

    def __init__(self) -> None:
        setup_logging()
        self._graph = self._build()
        logger.info("OrchestratorGraph initialised")

    def _build(self):
        builder = StateGraph(AgentState)

        # ── Nodes ─────────────────────────────────────────────────────────────
        builder.add_node("parse_intent",             parse_intent_node)
        builder.add_node("route_to_agent",           route_to_agent_node)
        builder.add_node("execute_with_guardrails",  execute_with_guardrails_node)
        builder.add_node("update_memory",            update_memory_node)
        builder.add_node("respond_to_user",          respond_to_user_node)

        # ── Entry ─────────────────────────────────────────────────────────────
        builder.add_edge(START, "parse_intent")

        # ── Conditional routing ────────────────────────────────────────────────
        builder.add_conditional_edges(
            "parse_intent",
            _route_after_intent,
            {"route_to_agent": "route_to_agent", "respond_to_user": "respond_to_user"},
        )
        builder.add_conditional_edges(
            "route_to_agent",
            _route_after_route,
            {
                "execute_with_guardrails": "execute_with_guardrails",
                "respond_to_user": "respond_to_user",
            },
        )
        builder.add_conditional_edges(
            "execute_with_guardrails",
            _route_after_execute,
            {
                "execute_with_guardrails": "execute_with_guardrails",
                "update_memory": "update_memory",
                "respond_to_user": "respond_to_user",
            },
        )
        builder.add_edge("update_memory", "respond_to_user")
        builder.add_edge("respond_to_user", END)

        checkpointer = MemorySaver()
        return builder.compile(checkpointer=checkpointer)

    # ── Run API ───────────────────────────────────────────────────────────────

    async def run(
        self,
        user_input: str,
        user_id: str = "anonymous",
        session_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> AgentState:
        """Execute the full pipeline and return the final AgentState."""
        # Ensure session exists
        from core.session_manager import get_session_manager
        sm = get_session_manager()
        session = await sm.get_or_create(user_id)
        sid = session_id or session.session_id

        # Record user turn
        session.add_turn("user", user_input)
        await sm.update(session)

        initial = create_initial_state(
            user_input=user_input,
            user_id=user_id,
            session_id=sid,
            metadata=metadata,
        )
        config = {"configurable": {"thread_id": initial["task_id"]}}

        logger.info("OrchestratorGraph.run", task_id=initial["task_id"], user_id=user_id)
        try:
            return await self._graph.ainvoke(initial, config=config)
        except Exception as exc:
            logger.error("Graph run error", error=str(exc), exc_info=True)
            initial["status"] = TaskStatus.FAILED.value
            initial["error"] = str(exc)
            return initial

    async def stream(
        self,
        user_input: str,
        user_id: str = "anonymous",
        session_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Stream partial node outputs as the graph executes.
        Yields: {"node": str, "task_id": str, "update": dict}
        """
        from core.session_manager import get_session_manager
        sm = get_session_manager()
        session = await sm.get_or_create(user_id)
        sid = session_id or session.session_id

        session.add_turn("user", user_input)
        await sm.update(session)

        initial = create_initial_state(
            user_input=user_input,
            user_id=user_id,
            session_id=sid,
            metadata=metadata,
        )
        config = {"configurable": {"thread_id": initial["task_id"]}}

        async for event in self._graph.astream(initial, config=config, stream_mode="updates"):
            for node_name, update in event.items():
                yield {"node": node_name, "task_id": initial["task_id"], "update": update}


# ── Singleton helper ──────────────────────────────────────────────────────────

_graph: Optional[OrchestratorGraph] = None


def get_orchestrator_graph() -> OrchestratorGraph:
    global _graph
    if _graph is None:
        _graph = OrchestratorGraph()
    return _graph
