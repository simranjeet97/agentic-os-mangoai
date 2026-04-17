"""
agents/base_agent.py — Abstract base class for all agent modules.
All agents must implement the `execute` method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from core.logging_config import get_logger
from core.state import AgentState
from guardrails.audit_logger import AuditLogger


class BaseAgent(ABC):
    """
    Abstract base for all specialist agents.
    Provides logging, audit trail, and error handling scaffolding.
    """

    name: str = "base"
    description: str = "Base agent"

    def __init__(self) -> None:
        self.logger = get_logger(f"agent.{self.name}")
        self.auditor = AuditLogger()

    @abstractmethod
    async def execute(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        """Execute a task step. Must be overridden by subclasses."""
        ...

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

            await self.auditor.log_action(
                event_type=f"agent.{self.name}.execute",
                user_id=state.get("user_id", "unknown"),
                task_id=state.get("task_id", "unknown"),
                agent=self.name,
                details={"step": step, "duration_ms": duration_ms},
                outcome="success" if result["success"] else "failure",
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
            await self.auditor.log_action(
                event_type=f"agent.{self.name}.error",
                user_id=state.get("user_id", "unknown"),
                task_id=state.get("task_id", "unknown"),
                agent=self.name,
                details={"step": step, "error": str(exc), "duration_ms": duration_ms},
                outcome="error",
            )
            return {"success": False, "error": str(exc), "duration_ms": duration_ms}
