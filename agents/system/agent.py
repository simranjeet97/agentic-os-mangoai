"""
agents/system/agent.py — System Agent.
Executes OS-level commands inside the Docker sandbox.
All commands run in an isolated container with resource limits.
"""

from __future__ import annotations

import asyncio
import os
import shlex
from typing import Any

from agents.base_agent import BaseAgent
from core.state import AgentState

# Commands that are always blocked — even in sandbox
BLOCKED_COMMANDS = {
    "rm -rf /",
    "dd if=/dev/zero",
    ":(){ :|:& };:",  # fork bomb
    "mkfs",
    "fdisk",
    "chmod 777 /",
}

COMMAND_TIMEOUT = int(os.getenv("SANDBOX_TIMEOUT_SECONDS", "30"))


class SystemAgent(BaseAgent):
    name = "system"
    description = "Sandboxed OS command and process management"

    async def execute(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        async def _run():
            command = step.get("command") or step.get("description", "echo hello")
            use_sandbox = step.get("sandboxed", True)

            # Block obviously dangerous commands
            for blocked in BLOCKED_COMMANDS:
                if blocked.lower() in command.lower():
                    return {
                        "success": False,
                        "error": f"Command blocked by system agent policy: '{blocked}'",
                        "step_type": "system_blocked",
                    }

            if use_sandbox:
                return await self._run_in_sandbox(command)
            else:
                return await self._run_local(command)

        return await self._run_with_audit(step, state, _run)

    async def _run_in_sandbox(self, command: str) -> dict[str, Any]:
        """Run command inside the Docker sandbox container."""
        self.logger.info("Running command in Docker sandbox", command=command[:120])
        try:
            import docker as docker_sdk
            client = docker_sdk.from_env()
            container = client.containers.run(
                image="agentic-sandbox:latest",
                command=["sh", "-c", command],
                remove=True,
                mem_limit="256m",
                cpu_period=100_000,
                cpu_quota=50_000,       # 50% of one CPU
                network_mode="none",    # No network access in sandbox
                read_only=False,
                timeout=COMMAND_TIMEOUT,
                stdout=True,
                stderr=True,
            )
            output = container.decode("utf-8", errors="replace") if isinstance(container, bytes) else str(container)
            return {"output": output[:5000], "command": command, "step_type": "system_exec", "sandboxed": True}
        except Exception as exc:
            self.logger.warning("Docker sandbox execution failed, running local", error=str(exc))
            return await self._run_local(command)

    async def _run_local(self, command: str) -> dict[str, Any]:
        """Run command locally with timeout (less safe — for trusted ops)."""
        self.logger.info("Running command locally", command=command[:120])
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=1024 * 1024,  # 1MB output limit
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=COMMAND_TIMEOUT,
                )
            except asyncio.TimeoutError:
                proc.kill()
                return {"success": False, "error": f"Command timed out after {COMMAND_TIMEOUT}s"}

            output = stdout.decode("utf-8", errors="replace")
            err = stderr.decode("utf-8", errors="replace")
            return_code = proc.returncode

            return {
                "output": output[:5000],
                "stderr": err[:1000],
                "return_code": return_code,
                "success": return_code == 0,
                "command": command,
                "step_type": "system_exec",
                "sandboxed": False,
            }
        except Exception as exc:
            return {"success": False, "error": str(exc), "step_type": "system_exec"}
