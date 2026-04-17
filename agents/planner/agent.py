"""
agents/planner/agent.py — Planner Agent.
Handles sub-goal decomposition within a running task graph.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from agents.base_agent import BaseAgent
from core.state import AgentState

SYSTEM_PROMPT = """\
You are the Planner sub-agent. Given a high-level task step, break it down into
concrete, actionable sub-tasks or provide a clarified execution plan.
Be precise and structured. Output plain text with numbered steps.
"""


class PlannerAgent(BaseAgent):
    name = "planner"
    description = "Decomposes complex steps into fine-grained sub-tasks"

    def __init__(self) -> None:
        super().__init__()
        import os
        self.llm = ChatOllama(
            model=os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.2:3b"),
            temperature=0.0,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    async def execute(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        async def _run():
            description = step.get("description", "")
            self.logger.info("Planner sub-decomposition", step=description[:80])

            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"Task step: {description}\nContext: {state.get('goal', '')}"),
            ]
            response = await self.llm.ainvoke(messages)
            return {
                "output": response.content,
                "step_type": "plan",
            }

        return await self._run_with_audit(step, state, _run)
