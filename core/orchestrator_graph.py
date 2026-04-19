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
    # Use messages from state as history
    history = []
    for msg in state.get("messages", [])[-10:]:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        history.append({"role": role, "content": msg.content})

    parsed = await parser.parse(state["user_input"], history=history)

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

Rules:
1. **Decisiveness**: Be autonomous. If the goal is slightly ambiguous, use the conversation history to choose the most likely interpretation and proceed.
2. **Context**: Use the provided conversation history to understand references to previous tasks or results.
3. **Agent Roles**: You MUST only use the following roles for the "agent" field:
   - **web**: For searching the internet, browsing websites, and retrieving online information.
   - **file**: For reading, writing, and managing local files.
   - **code**: For generating, linting, and executing Python code.
   - **system**: For executing shell commands and system-level operations.
   - **executor**: For general-purpose tasks or summarizing results.
   - **planner**: For complex multi-step planning or sub-task decomposition.
4. **Research Rules**: For research or information retrieval tasks:
   - Always request titles and URLs.
   - Use the `web` agent first, then use the `executor` agent with the `action: "summarize"` to synthesize the results.
5. **Output**: Return ONLY valid JSON.

Schema:
{
  "goal_summary": "...",
  "steps": [
    {"step_id": "s1", "description": "Search for...", "agent": "web", "action": "search", "query": "...", "dependencies": []},
    {"step_id": "s2", "description": "Summarize findings from s1 with titles and URLs", "agent": "executor", "action": "summarize", "dependencies": ["s1"]}
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
        from agents.router import get_router
        router = get_router()
        agent_role = await router.classify(state["user_input"])
        
        plan = [{
            "step_id": "s1",
            "description": state["user_input"],
            "agent": agent_role,
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
    from core.llm_factory import get_llm
    from langchain_core.messages import SystemMessage

    llm = get_llm(temperature=0.0)

    memory_ctx = ""
    if state.get("memory"):
        episodic = state["memory"].get("episodic", [])[:2]
        if episodic:
            memory_ctx = f"\nRelevant context:\n{episodic}\n"

    messages = [SystemMessage(content=_ROUTE_SYSTEM_PROMPT)]
    # Include conversation history (state["messages"]) in the planning prompt
    messages.extend(state.get("messages", [])[-5:])
    messages.append(HumanMessage(content=f"Goal: {state['goal']}{memory_ctx}"))

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
    """Route a step to the correct agent module. Normalizes hallucinated agent names."""
    try:
        from agents.planner.agent import PlannerAgent
        from agents.executor.agent import ExecutorAgent
        from agents.file.agent import FileAgent
        from agents.web.agent import WebAgent
        from agents.system.agent import SystemAgent
        from agents.code.agent import CodeAgent

        # Normalize common hallucinations to official roles
        normalization_map = {
            "search_agent": "web",
            "web_browser": "web",
            "web_search": "web",
            "research_agent": "web",
            "file_agent": "file",
            "code_agent": "code",
            "summarizer_agent": "executor",
            "system_agent": "system",
            "planner_agent": "planner",
            "executor_agent": "executor",
        }
        
        normalized_name = normalization_map.get(agent_name.lower(), agent_name.lower())
        if normalized_name != agent_name.lower():
            logger.info("Normalized agent name", original=agent_name, normalized=normalized_name)

        agent_map = {
            AgentRole.PLANNER.value:  PlannerAgent,
            AgentRole.EXECUTOR.value: ExecutorAgent,
            AgentRole.FILE.value:     FileAgent,
            AgentRole.WEB.value:      WebAgent,
            AgentRole.SYSTEM.value:   SystemAgent,
            AgentRole.CODE.value:     CodeAgent,
        }
        
        AgentClass: Any = agent_map.get(normalized_name)
        if not AgentClass:
            logger.warning("Unknown agent role, defaulting to executor", role=agent_name)
            AgentClass = ExecutorAgent
            
        agent = AgentClass()
        return await agent.execute(step=step, state=state)
    except ImportError as exc:
        logger.error("Agent import failed", error=str(exc))
        # Graceful stub for testing without all agents installed
        await asyncio.sleep(0.05)
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
    """Build a human-readable summary or full report from the execution results."""
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
            return (
                f"🖥️  System status: running • "
                f"Session: {state['session_id'][:8]} • "
                f"Agents available: planner, executor, file, web, system, code"
            )
        return "✅ Done! No output was produced."

    # 1. Look for a "Primary Report" (Summarization or Reasoning step)
    primary_report = None
    process_log = []

    for r in results:
        out = r.get("result", {})
        success = out.get("success", True)
        icon = "✓" if success else "✗"
        step_id = r.get("step_id", "?")
        agent = r.get("agent", "?")
        
        # Determine if this step produced a major "answer" or just a log
        step_type = out.get("step_type", "")
        output_text = out.get("output") or out.get("error") or "done"
        
        # If it's a reasoning/summarization step, it's a candidate for the primary report
        if step_type in ("reason", "summarize", "synthesis"):
            primary_report = output_text
        
        # Add to the process log (checklist)
        # Increase truncation to 500 characters and keep newlines but format as a quote
        clean_text = str(output_text).strip()
        if len(clean_text) > 500:
            clean_text = clean_text[:500] + "..."
        
        # Format the log snippet cleanly
        if "\n" in clean_text:
            # If multi-line, use a blockquote/code format
            log_snippet = f"\n> {clean_text.replace(chr(10), chr(10) + '> ')}"
        else:
            log_snippet = f" {clean_text}"
            
        process_log.append(f"- **{icon} [{agent}:{step_id}]**{log_snippet}")

    # 2. Assemble the final response
    if primary_report:
        # Save to file system as a "compiled" artifact
        filepath = _save_report_to_file(primary_report, state.get("task_id", "research"))
        
        # We have a high-quality report, prioritize it
        sections = [
            primary_report,
            f"\n> 📄 Full report compiled to: `{filepath}`" if filepath else "",
            "\n---\n### 🛠️ Process Log",
            "\n".join(process_log)
        ]
        return "\n".join(sections)
    
    # 3. No single "Report" found, return the classic process log
    final_output = ["### ✅ Task Complete\n"]
    if len(results) == 1:
        # Just one step? Show its full output.
        res = results[0].get("result", {})
        output = res.get("output") or res.get("error") or "Done."
        return str(output)
    
    final_output.extend(process_log)
    return "\n".join(final_output)


def _save_report_to_file(content: str, task_id: str) -> Optional[str]:
    """Save a reasoning/synthesis report to the repo's reports/ directory."""
    try:
        import os
        from datetime import datetime
        
        # Determine the base directory (assume we're in core/ and want to go to reports/)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        reports_dir = os.path.join(base_dir, "reports")
        
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{task_id[:8]}_{timestamp}.md"
        filepath = os.path.join(reports_dir, filename)
        
        with open(filepath, "w") as f:
            f.write(content)
            
        logger.info("Report saved to file", path=filepath)
        return filepath
    except Exception as exc:
        logger.warning("Failed to save report to file", error=str(exc))
        return None


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

        # Convert session history to LangChain messages
        history = []
        for turn in session.history[-10:]:
            if turn["role"] == "user":
                history.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                history.append(AIMessage(content=turn["content"]))

        initial = create_initial_state(
            user_input=user_input,
            user_id=user_id,
            session_id=sid,
            metadata=metadata,
            history=history,
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

        # Convert session history to LangChain messages
        history = []
        for turn in session.history[-10:]:
            if turn["role"] == "user":
                history.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                history.append(AIMessage(content=turn["content"]))

        initial = create_initial_state(
            user_input=user_input,
            user_id=user_id,
            session_id=sid,
            metadata=metadata,
            history=history,
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
