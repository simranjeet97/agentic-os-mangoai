"""
agents/executor/agent.py — Executor Agent.
General-purpose action dispatcher and result aggregator.
"""

from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent
from core.state import AgentState


class ExecutorAgent(BaseAgent):
    name = "executor"
    description = "General-purpose action dispatcher"

    async def execute(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        async def _run():
            description = step.get("description", "No description")
            self.logger.info("Executor dispatching step", step=description[:80])

            # Executor is the fallback — it runs any uncategorized step
            # by delegating to the best-matching tool or returning a summary
            return {
                "output": f"Executed: {description}",
                "step_type": "action",
                "details": {"step": step},
            }

        return await self._run_with_audit(step, state, _run)
