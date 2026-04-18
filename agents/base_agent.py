"""
agents/base_agent.py — Abstract base class for all agent modules.

Every agent in the Agentic OS:
  - Has an agent_id, name, capabilities[], and tools[]
  - Integrates with GuardrailMiddleware (all actions pass through)
  - Integrates with MemoryAgent (stores and recalls task context)
  - Exposes run(task) → AgentResult  (high-level wrapper)
  - Exposes execute(step, state) → dict  (LangGraph node interface)
"""

from __future__ import annotations

import os
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from core.logging_config import get_logger
from core.state import AgentState
from guardrails.audit_logger import AuditLogger


# ── AgentResult — the canonical return type for run() ────────────────────────


class AgentResult(BaseModel):
    """Canonical result returned by BaseAgent.run()."""

    agent_id: str
    agent_name: str
    task_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: int = 0
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def to_step_dict(self) -> dict[str, Any]:
        """Convert to the dict format expected by LangGraph step results."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "artifacts": self.artifacts,
            "agent_id": self.agent_id,
        }


# ── BaseAgent ─────────────────────────────────────────────────────────────────


class BaseAgent(ABC):
    """
    Abstract base for all specialist agents.

    Provides:
    - Unique agent_id, name, description, capabilities, tools
    - Lazy-loaded GuardrailMiddleware (shared across the process)
    - Lazy-loaded MemoryAgent for recall and remember
    - run(task) → AgentResult  high-level entry point
    - execute(step, state) → dict  LangGraph node entry point
    - _run_with_audit()  timing + error handling + audit trail
    """

    # ── Class-level identity (override in subclasses) ─────────────────────────
    name: str = "base"
    description: str = "Base agent"
    capabilities: list[str] = []   # e.g. ["file_read", "file_write"]
    tools: list[str] = []          # e.g. ["read_file", "write_file"]

    # ── Shared singletons (one per process) ───────────────────────────────────
    _shared_guardrail: Optional[Any] = None
    _shared_memory: Optional[Any] = None

    def __init__(self, agent_id: Optional[str] = None) -> None:
        self.agent_id: str = agent_id or f"{self.name}-{uuid.uuid4().hex[:8]}"
        self.logger = get_logger(f"agent.{self.name}")
        self.auditor = AuditLogger()

    # ── Lazy singletons ───────────────────────────────────────────────────────

    @property
    def guardrail(self) -> Any:
        """Lazy-loaded shared GuardrailMiddleware instance."""
        if BaseAgent._shared_guardrail is None:
            from guardrails.middleware import GuardrailMiddleware
            BaseAgent._shared_guardrail = GuardrailMiddleware()
        return BaseAgent._shared_guardrail

    @property
    def memory(self) -> Any:
        """Lazy-loaded shared MemoryAgent instance."""
        if BaseAgent._shared_memory is None:
            from memory.memory_manager import MemoryAgent
            BaseAgent._shared_memory = MemoryAgent()
        return BaseAgent._shared_memory

    # ── High-level run() API ──────────────────────────────────────────────────

    async def run(
        self,
        task: dict[str, Any],
        user_id: str = "system",
        session_id: Optional[str] = None,
    ) -> AgentResult:
        """
        High-level entry point called externally (AgentRouter, tests, API).

        Wraps execute() with a synthetic AgentState and returns AgentResult.
        Stores task outcome in memory after completion.

        Args:
            task: Dict containing at minimum 'description' and 'action'.
            user_id: Who is invoking the agent.
            session_id: Optional session for memory continuity.

        Returns:
            AgentResult with success/failure, output, and metadata.
        """
        task_id = task.get("task_id") or str(uuid.uuid4())
        sid = session_id or str(uuid.uuid4())
        start = datetime.utcnow()

        # Recall relevant memory context
        memory_context: dict[str, Any] = {}
        try:
            recall = await self.memory.recall(
                query=task.get("description", ""),
                session_id=sid,
                user_id=user_id,
                top_k=3,
            )
            memory_context = {
                "results": [r.model_dump() for r in recall.results[:3]],
                "sources": [s.value for s in recall.sources_queried],
            }
        except Exception as exc:
            self.logger.warning("Memory recall failed", error=str(exc))

        # Build a minimal AgentState for execute()
        from core.state import AgentState, TaskStatus
        state: AgentState = {  # type: ignore[assignment]
            "task_id": task_id,
            "session_id": sid,
            "user_id": user_id,
            "messages": [],
            "user_input": task.get("description", ""),
            "goal": task.get("description", ""),
            "plan": [],
            "current_step_index": 0,
            "active_agent": self.name,
            "tool_calls": [],
            "tool_results": [],
            "artifacts": [],
            "status": TaskStatus.EXECUTING.value,
            "error": None,
            "iterations": 0,
            "memory": memory_context,
            "guardrail_result": None,
            "requires_approval": False,
            "metadata": task.get("metadata", {}),
            "created_at": start.isoformat(),
            "updated_at": start.isoformat(),
        }

        try:
            result_dict = await self.execute(step=task, state=state)
            duration_ms = int((datetime.utcnow() - start).total_seconds() * 1000)

            result = AgentResult(
                agent_id=self.agent_id,
                agent_name=self.name,
                task_id=task_id,
                success=result_dict.get("success", True),
                output=result_dict.get("output"),
                error=result_dict.get("error"),
                duration_ms=result_dict.get("duration_ms", duration_ms),
                artifacts=result_dict.get("artifacts", []),
                metadata={**result_dict, "session_id": sid},
            )
        except Exception as exc:
            duration_ms = int((datetime.utcnow() - start).total_seconds() * 1000)
            self.logger.error("Agent run failed", error=str(exc), exc_info=True)
            result = AgentResult(
                agent_id=self.agent_id,
                agent_name=self.name,
                task_id=task_id,
                success=False,
                error=str(exc),
                duration_ms=duration_ms,
            )

        # Store result in memory
        try:
            from memory.schemas import EventType, MemoryEvent
            await self.memory.remember(
                MemoryEvent(
                    event_type=EventType.TASK_COMPLETED if result.success else EventType.TASK_FAILED,
                    content=f"[{self.name}] {task.get('description', '')}",
                    outcome=str(result.output)[:500] if result.output else result.error or "no output",
                    session_id=sid,
                    user_id=user_id,
                    agent_id=self.agent_id,
                    metadata={"task_id": task_id, "duration_ms": result.duration_ms},
                )
            )
        except Exception as exc:
            self.logger.warning("Memory store after run failed", error=str(exc))

        return result

    # ── LangGraph node interface ───────────────────────────────────────────────

    @abstractmethod
    async def execute(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        """Execute a task step. Must be overridden by subclasses."""
        ...

    # ── Shared helpers ────────────────────────────────────────────────────────

    async def _run_with_audit(
        self,
        step: dict[str, Any],
        state: AgentState,
        func,
    ) -> dict[str, Any]:
        """Wrap execution with timing, error handling, and audit logging."""
        start = datetime.utcnow()
        try:
            result = await func()
            duration_ms = int((datetime.utcnow() - start).total_seconds() * 1000)
            result["duration_ms"] = duration_ms
            result["success"] = result.get("success", True)

            await self.auditor.log(
                agent_id=self.agent_id,
                action_type=f"agent.{self.name}.execute",
                risk_score=0.0,
                outcome="success" if result["success"] else "error",
                details={"step": step, "duration_ms": duration_ms, "task_id": state.get("task_id", "unknown")},
                approved_by=state.get("user_id", "unknown"),
            )
            return result

        except Exception as exc:
            duration_ms = int((datetime.utcnow() - start).total_seconds() * 1000)
            self.logger.error(
                f"{self.name} agent error",
                error=str(exc),
                step=step,
                exc_info=True,
            )
            await self.auditor.log(
                agent_id=self.agent_id,
                action_type=f"agent.{self.name}.error",
                risk_score=0.0,
                outcome="error",
                details={"step": step, "error": str(exc), "duration_ms": duration_ms, "task_id": state.get("task_id", "unknown")},
                approved_by=state.get("user_id", "unknown"),
            )
            return {"success": False, "error": str(exc), "duration_ms": duration_ms}

    def _get_llm(
        self,
        model_env: str = "OLLAMA_DEFAULT_MODEL",
        temperature: float = 0.0,
    ) -> Any:
        """Return a shared ChatOllama instance."""
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv(model_env, os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.2:3b")),
            temperature=temperature,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
