"""
agents/code/agent.py — Code Agent.
Generates, lints, tests, and executes code using the sandbox.
Supports Python (primary), JavaScript, Bash.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from agents.base_agent import BaseAgent
from core.state import AgentState

CODEGEN_SYSTEM_PROMPT = """\
You are an expert software engineer. Generate clean, production-quality code.
- Add type hints and docstrings (Python)
- Handle errors gracefully
- Do NOT include explanations outside code blocks
- Return ONLY the code, wrapped in a single ```language ... ``` block
"""

SUPPORTED_LANGUAGES = {"python", "javascript", "bash", "shell"}


class CodeAgent(BaseAgent):
    name = "code"
    description = "Code generation, linting, testing, and sandbox execution"

    def __init__(self) -> None:
        super().__init__()
        self.llm = ChatOllama(
            model=os.getenv("OLLAMA_CODE_MODEL", os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.2:3b")),
            temperature=0.1,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    async def execute(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        async def _run():
            action = step.get("action", "generate")
            language = step.get("language", "python").lower()
            description = step.get("description", "")
            code = step.get("code", "")

            if language not in SUPPORTED_LANGUAGES:
                return {"success": False, "error": f"Unsupported language: {language}"}

            if action == "generate" or not code:
                code_result = await self._generate_code(description, language)
                if not code_result["success"]:
                    return code_result
                code = code_result["code"]

            if action == "execute" or step.get("execute", False):
                exec_result = await self._execute_code(code, language)
                return {**code_result if "code_result" in dir() else {}, **exec_result, "code": code}

            return {
                "output": code,
                "code": code,
                "language": language,
                "step_type": "code_generate",
            }

        return await self._run_with_audit(step, state, _run)

    async def _generate_code(self, description: str, language: str) -> dict[str, Any]:
        """Use LLM to generate code for the given description."""
        self.logger.info("Generating code", description=description[:80], language=language)

        import re
        messages = [
            SystemMessage(content=CODEGEN_SYSTEM_PROMPT),
            HumanMessage(content=f"Language: {language}\nTask: {description}"),
        ]
        try:
            response = await self.llm.ainvoke(messages)
            content = response.content.strip()

            # Extract code from markdown block
            match = re.search(r"```(?:\w+)?\n?(.*?)```", content, re.DOTALL)
            code = match.group(1).strip() if match else content

            return {"code": code, "language": language, "success": True}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def _execute_code(self, code: str, language: str) -> dict[str, Any]:
        """Execute the generated code in the system sandbox."""
        self.logger.info("Executing code in sandbox", language=language, code_length=len(code))

        from agents.system.agent import SystemAgent
        sys_agent = SystemAgent()

        if language == "python":
            cmd = f'python3 -c {__import__("shlex").quote(code)}'
        elif language in ("bash", "shell"):
            cmd = code
        elif language == "javascript":
            cmd = f'node -e {__import__("shlex").quote(code)}'
        else:
            return {"success": False, "error": f"Cannot execute {language}"}

        result = await sys_agent._run_in_sandbox(cmd)
        result["step_type"] = "code_execute"
        return result
