"""
agents/planner/agent.py — PlannerAgent

Receives a high-level goal, decomposes it into subtasks using chain-of-thought
reasoning, assigns each subtask to the most appropriate specialist agent, monitors
completion, and handles failures with replanning.

Chain-of-thought decomposition:
  1. Understand the goal
  2. Identify dependencies and constraints
  3. Decompose into ordered, atomic subtasks
  4. Assign an agent role to each subtask
  5. Validate the plan for completeness

Replanning:
  - Triggered when a subtask fails (max 2 replan attempts per step)
  - Uses failure context to generate a revised plan
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import BaseAgent
from core.state import AgentRole, AgentState, TaskStatus


# ── Prompts ───────────────────────────────────────────────────────────────────

DECOMPOSE_SYSTEM_PROMPT = """\
You are the PlannerAgent — the master orchestrator for a multi-agent AI OS.
Your job is to receive a high-level goal and produce a structured execution plan.

RULES:
1. Use chain-of-thought: think step-by-step before writing the plan.
2. Each subtask must be atomic (one clear action only).
3. Assign each subtask to exactly one agent role from:
   [planner, executor, file, web, system, code]
4. Order subtasks by dependency (earlier subtasks first).
5. Output ONLY the JSON plan — no explanation outside the JSON.

AGENT ROLE GUIDE:
- planner:  sub-planning, goal decomposition
- executor: running shell commands, scripts, executables
- file:     reading, writing, searching, converting files
- web:      browsing URLs, scraping, form filling, screenshots
- system:   checking CPU/RAM/disk, monitoring, managing processes
- code:     writing, debugging, refactoring, testing code

OUTPUT FORMAT (strict JSON array):
[
  {
    "step_id": "1",
    "description": "<clear action description>",
    "agent": "<agent_role>",
    "dependencies": [],
    "priority": 1
  },
  ...
]
"""

REPLAN_SYSTEM_PROMPT = """\
You are the PlannerAgent in REPLAN mode.
A subtask has FAILED. Your job is to produce a revised plan to recover.

Given:
- Original goal
- Failed step details
- Error message
- Remaining steps

Produce a revised JSON plan for the remaining steps, incorporating the failure context.
Output ONLY the JSON array (same format as before).
"""

COT_SYSTEM_PROMPT = """\
You are a reasoning engine. Think through this goal step by step.
Identify: what needs to happen, what tools are needed, what order makes sense.
Be thorough but concise.
"""


class PlannerAgent(BaseAgent):
    name = "planner"
    description = "Decomposes high-level goals into subtasks and orchestrates agents"
    capabilities = ["plan_decompose", "task_assign", "replan", "monitor"]
    tools = ["chain_of_thought", "assign_agent", "replan_on_failure"]

    MAX_REPLAN_ATTEMPTS = 2
    MAX_SUBTASKS = 15

    def __init__(self, agent_id: Optional[str] = None) -> None:
        super().__init__(agent_id)
        self._llm = None
        self._replan_counts: dict[str, int] = {}  # step_id → attempt count

    @property
    def llm(self) -> Any:
        if self._llm is None:
            self._llm = self._get_llm("OLLAMA_DEFAULT_MODEL", temperature=0.1)
        return self._llm

    async def execute(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        async def _run() -> dict[str, Any]:
            action = step.get("action", "decompose")

            if action == "replan":
                return await self._replan(step, state)
            else:
                return await self._decompose(step, state)

        return await self._run_with_audit(step, state, _run)

    # ── Decompose goal into subtasks ──────────────────────────────────────────

    async def _decompose(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        """Chain-of-thought goal decomposition."""
        goal = step.get("description") or state.get("goal", "")
        context = state.get("memory", {})

        self.logger.info("PlannerAgent decomposing goal", goal=goal[:120])

        # Step 1: CoT reasoning
        cot_messages = [
            SystemMessage(content=COT_SYSTEM_PROMPT),
            HumanMessage(content=f"Goal: {goal}\n\nMemory context: {json.dumps(context, default=str)[:500]}"),
        ]
        try:
            cot_response = await self.llm.ainvoke(cot_messages)
            reasoning = cot_response.content.strip()
        except Exception as exc:
            reasoning = f"CoT failed: {exc}"

        self.logger.debug("PlannerAgent CoT complete", reasoning_length=len(reasoning))

        # Step 2: Generate structured plan
        plan_messages = [
            SystemMessage(content=DECOMPOSE_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Goal: {goal}\n\n"
                    f"Reasoning:\n{reasoning}\n\n"
                    f"Now produce the JSON plan:"
                )
            ),
        ]
        try:
            plan_response = await self.llm.ainvoke(plan_messages)
            raw = plan_response.content.strip()
            subtasks = self._parse_plan(raw)
        except Exception as exc:
            self.logger.error("Plan generation failed", error=str(exc))
            # Fallback: single executor step
            subtasks = [
                {
                    "step_id": "1",
                    "description": goal,
                    "agent": "executor",
                    "dependencies": [],
                    "priority": 1,
                    "status": TaskStatus.PENDING.value,
                }
            ]

        # Limit subtask count
        subtasks = subtasks[: self.MAX_SUBTASKS]
        for i, s in enumerate(subtasks):
            s["status"] = TaskStatus.PENDING.value
            s["step_id"] = s.get("step_id", str(i + 1))

        self.logger.info(
            "Plan created",
            goal=goal[:80],
            subtask_count=len(subtasks),
        )

        return {
            "output": f"Plan decomposed: {len(subtasks)} subtasks",
            "plan": subtasks,
            "reasoning": reasoning,
            "step_type": "plan",
        }

    # ── Replan after failure ──────────────────────────────────────────────────

    async def _replan(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        """Replan after a subtask failure."""
        failed_step = step.get("failed_step", {})
        error = step.get("error", "Unknown error")
        goal = state.get("goal", "")
        remaining = step.get("remaining_steps", [])

        step_id = str(failed_step.get("step_id", "?"))
        self._replan_counts[step_id] = self._replan_counts.get(step_id, 0) + 1

        if self._replan_counts[step_id] > self.MAX_REPLAN_ATTEMPTS:
            return {
                "success": False,
                "error": f"Max replan attempts ({self.MAX_REPLAN_ATTEMPTS}) exceeded for step {step_id}",
                "step_type": "replan",
            }

        self.logger.warning(
            "PlannerAgent replanning",
            failed_step=step_id,
            attempt=self._replan_counts[step_id],
            error=error[:200],
        )

        replan_messages = [
            SystemMessage(content=REPLAN_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Original goal: {goal}\n\n"
                    f"Failed step: {json.dumps(failed_step, default=str)}\n"
                    f"Error: {error}\n\n"
                    f"Remaining steps: {json.dumps(remaining, default=str)}\n\n"
                    f"Attempt {self._replan_counts[step_id]} of {self.MAX_REPLAN_ATTEMPTS}.\n"
                    f"Produce a revised plan for the remaining steps:"
                )
            ),
        ]

        try:
            response = await self.llm.ainvoke(replan_messages)
            revised = self._parse_plan(response.content.strip())
        except Exception as exc:
            return {
                "success": False,
                "error": f"Replan LLM call failed: {exc}",
                "step_type": "replan",
            }

        for s in revised:
            s["status"] = TaskStatus.PENDING.value

        return {
            "output": f"Replanned: {len(revised)} steps",
            "revised_plan": revised,
            "step_type": "replan",
            "replan_attempt": self._replan_counts[step_id],
        }

    # ── Plan monitoring ───────────────────────────────────────────────────────

    async def monitor_plan(
        self,
        plan: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Check plan completion status and surface any failures.

        Returns summary dict: {total, completed, failed, pending, progress_pct}
        """
        total = len(plan)
        completed = sum(1 for s in plan if s.get("status") == TaskStatus.COMPLETED.value)
        failed = [s for s in plan if s.get("status") == TaskStatus.FAILED.value]
        pending = sum(1 for s in plan if s.get("status") == TaskStatus.PENDING.value)

        return {
            "total": total,
            "completed": completed,
            "failed_steps": failed,
            "pending": pending,
            "progress_pct": round(100 * completed / total, 1) if total else 0,
            "is_complete": completed == total,
            "has_failures": len(failed) > 0,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _parse_plan(self, raw: str) -> list[dict[str, Any]]:
        """Extract and parse JSON array from LLM output."""
        # Try to extract JSON array from the raw text
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            raw = match.group(0)

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [self._normalise_step(s) for s in parsed if isinstance(s, dict)]
        except json.JSONDecodeError:
            pass

        # Last resort: try to find individual JSON objects
        objects = re.findall(r"\{[^{}]+\}", raw, re.DOTALL)
        steps = []
        for i, obj in enumerate(objects):
            try:
                d = json.loads(obj)
                steps.append(self._normalise_step(d))
            except Exception:
                pass
        return steps or []

    @staticmethod
    def _normalise_step(s: dict[str, Any]) -> dict[str, Any]:
        """Ensure a step dict has the required fields with safe defaults."""
        valid_agents = {r.value for r in AgentRole}
        agent = str(s.get("agent", "executor")).lower()
        if agent not in valid_agents:
            agent = "executor"
        return {
            "step_id": str(s.get("step_id", "1")),
            "description": str(s.get("description", "Task step")),
            "agent": agent,
            "dependencies": s.get("dependencies", []),
            "priority": int(s.get("priority", 1)),
            "status": TaskStatus.PENDING.value,
            "result": None,
            "error": None,
            "started_at": None,
            "completed_at": None,
        }
