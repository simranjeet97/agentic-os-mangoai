"""
agents/executor/agent.py — ExecutorAgent

Runs code and scripts safely. Uses safe_execute() from GuardrailMiddleware's
SandboxEnforcer. Supports Python, Bash, JavaScript, and Go.

Returns:
    stdout, stderr, exit_code, explanation, duration_ms

Every command is gated through GuardrailMiddleware before execution:
- Command classified for risk (0-10)
- Blocklist enforced
- Audit record written
- Undo snapshot taken for destructive ops
"""

from __future__ import annotations

import shlex
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import BaseAgent
from core.state import AgentState
from guardrails.models import ActionType, AgentAction


# ── Supported languages and their execution templates ────────────────────────

LANGUAGE_MAP: dict[str, dict[str, str]] = {
    "python": {
        "runner": "python3",
        "flag": "-c",
        "extension": "py",
        "boilerplate": "#!/usr/bin/env python3\n",
    },
    "bash": {
        "runner": "bash",
        "flag": "-c",
        "extension": "sh",
        "boilerplate": "#!/bin/bash\nset -euo pipefail\n",
    },
    "shell": {
        "runner": "sh",
        "flag": "-c",
        "extension": "sh",
        "boilerplate": "#!/bin/sh\n",
    },
    "javascript": {
        "runner": "node",
        "flag": "-e",
        "extension": "js",
        "boilerplate": "'use strict';\n",
    },
    "go": {
        "runner": "go",
        "flag": "run",
        "extension": "go",
        "boilerplate": "package main\nimport \"fmt\"\nfunc main() {\n",
    },
}

SUPPORTED_LANGUAGES = set(LANGUAGE_MAP.keys())

EXPLAIN_PROMPT = """\
You are a technical explainer. Given the execution result of a script, write a
concise 1-3 sentence explanation of what happened. Focus on the key outcome,
any errors, and what the output means.
"""


class ExecutorAgent(BaseAgent):
    name = "executor"
    description = "Executes code and scripts in a safe sandbox across multiple languages"
    capabilities = ["code_execute", "shell_exec", "process_spawn"]
    tools = ["safe_execute", "explain_output", "sandbox_run"]

    def __init__(self, agent_id: Optional[str] = None) -> None:
        super().__init__(agent_id)
        self._llm: Optional[Any] = None
        self._sandbox: Optional[Any] = None

    @property
    def llm(self) -> Any:
        if self._llm is None:
            self._llm = self._get_llm("OLLAMA_CODE_MODEL", temperature=0.0)
        return self._llm

    @property
    def sandbox(self) -> Any:
        """Lazy-loaded SandboxEnforcer from the guardrails stack."""
        if self._sandbox is None:
            from guardrails.sandbox_enforcer import SandboxEnforcer
            self._sandbox = SandboxEnforcer()
        return self._sandbox

    async def execute(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        async def _run() -> dict[str, Any]:
            action = step.get("action", "execute")

            if action in ("execute", "run"):
                return await self._safe_execute(step, state)
            elif action == "explain":
                return await self._explain_execution(step)
            else:
                # Default: treat description as a command to run
                return await self._safe_execute(step, state)

        return await self._run_with_audit(step, state, _run)

    # ── Primary: safe_execute ─────────────────────────────────────────────────

    async def safe_execute(
        self,
        code: str,
        language: str = "python",
        timeout: int = 30,
        env: Optional[dict[str, str]] = None,
        user_id: str = "system",
    ) -> dict[str, Any]:
        """
        Public API: execute code through the guardrail + sandbox stack.

        Args:
            code:       The script or command to execute.
            language:   One of python | bash | shell | javascript | go.
            timeout:    Execution timeout in seconds.
            env:        Additional environment variables.
            user_id:    Requesting user (for audit trail).

        Returns:
            dict with stdout, stderr, exit_code, explanation, success.
        """
        language = language.lower()
        if language not in SUPPORTED_LANGUAGES:
            return {
                "success": False,
                "error": f"Unsupported language: {language}. Supported: {sorted(SUPPORTED_LANGUAGES)}",
                "stdout": "",
                "stderr": "",
                "exit_code": -1,
            }

        # Build an AgentAction so the guardrail chain can evaluate it
        action = AgentAction(
            agent_id=self.agent_id,
            agent_type=self.name,
            action_type=ActionType.CODE_EXECUTION,
            code=code,
            language=language,
            command=self._build_command(code, language),
            raw_input=code,
            metadata={"timeout": timeout, "env": env or {}},
        )

        # Run through guardrails (raises BlockedActionError or PendingApprovalException)
        from guardrails.exceptions import BlockedActionError, PendingApprovalException
        try:
            guardrail_result = await self.guardrail.safe_execute_code(action)
        except BlockedActionError as exc:
            return {
                "success": False,
                "blocked": True,
                "error": f"Blocked by guardrail: {exc.reason}",
                "violations": exc.violations,
                "stdout": "",
                "stderr": "",
                "exit_code": -1,
            }
        except PendingApprovalException as exc:
            return {
                "success": False,
                "pending_approval": True,
                "error": f"Requires approval: {exc.explanation}",
                "stdout": "",
                "stderr": "",
                "exit_code": -1,
            }

        # If sandbox_result is attached to the guardrail result, use it
        if guardrail_result.sandbox_result:
            sr = guardrail_result.sandbox_result
            result = {
                "stdout": sr.stdout,
                "stderr": sr.stderr,
                "exit_code": sr.exit_code,
                "success": sr.exit_code == 0,
                "duration_ms": sr.duration_ms,
                "language": language,
                "timed_out": sr.timed_out,
            }
        else:
            # Fallback: run directly through SandboxEnforcer
            sr = await self.sandbox.safe_execute(
                code=code,
                language=language,
                timeout=timeout,
                allow_network=False,
            )
            result = {
                "stdout": sr.stdout,
                "stderr": sr.stderr,
                "exit_code": sr.exit_code,
                "success": sr.exit_code == 0,
                "duration_ms": sr.duration_ms,
                "language": language,
                "timed_out": sr.timed_out,
            }

        # Add LLM explanation
        explanation = await self._generate_explanation(code, result)
        result["explanation"] = explanation

        self.logger.info(
            "Code executed",
            language=language,
            exit_code=result["exit_code"],
            duration_ms=result.get("duration_ms"),
        )
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _safe_execute(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        """Execute code from a step dict."""
        code = step.get("code") or step.get("command") or step.get("description", "")
        language = step.get("language", "bash").lower()
        timeout = int(step.get("timeout", 30))

        return await self.safe_execute(
            code=code,
            language=language,
            timeout=timeout,
            user_id=state.get("user_id", "system"),
        )

    async def _explain_execution(self, step: dict[str, Any]) -> dict[str, Any]:
        """Ask the LLM to explain an already-captured execution result."""
        stdout = step.get("stdout", "")
        stderr = step.get("stderr", "")
        exit_code = step.get("exit_code", 0)
        code = step.get("code", "")

        explanation = await self._generate_explanation(
            code, {"stdout": stdout, "stderr": stderr, "exit_code": exit_code}
        )
        return {"explanation": explanation, "step_type": "explain", "output": explanation}

    async def _generate_explanation(
        self,
        code: str,
        result: dict[str, Any],
    ) -> str:
        """Generate a human-readable explanation of the execution result."""
        try:
            messages = [
                SystemMessage(content=EXPLAIN_PROMPT),
                HumanMessage(
                    content=(
                        f"Code:\n```\n{code[:500]}\n```\n\n"
                        f"Exit code: {result.get('exit_code')}\n"
                        f"Stdout:\n{result.get('stdout', '')[:800]}\n"
                        f"Stderr:\n{result.get('stderr', '')[:400]}"
                    )
                ),
            ]
            response = await self.llm.ainvoke(messages)
            return response.content.strip()
        except Exception as exc:
            return f"(Explanation unavailable: {exc})"

    @staticmethod
    def _build_command(code: str, language: str) -> str:
        """Build the shell command that would run this code snippet."""
        lang = LANGUAGE_MAP.get(language, {})
        runner = lang.get("runner", "sh")
        flag = lang.get("flag", "-c")

        if language == "go":
            return f"echo {shlex.quote(code)} > /tmp/_exec.go && go run /tmp/_exec.go"

        return f"{runner} {flag} {shlex.quote(code)}"
