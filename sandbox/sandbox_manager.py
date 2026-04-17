"""
sandbox/sandbox_manager.py — Docker-based sandbox orchestration.
Creates, manages, and destroys isolated execution containers.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any, Optional

from core.logging_config import get_logger

logger = get_logger(__name__)

SANDBOX_IMAGE = os.getenv("SANDBOX_IMAGE", "agentic-sandbox:latest")
SANDBOX_MEMORY = os.getenv("SANDBOX_MAX_MEMORY", "256m")
SANDBOX_CPU_QUOTA = int(os.getenv("SANDBOX_CPU_QUOTA", "50000"))   # 50% of 1 CPU
SANDBOX_TIMEOUT = int(os.getenv("SANDBOX_TIMEOUT_SECONDS", "30"))
SANDBOX_TMP_SIZE = os.getenv("SANDBOX_TMP_SIZE", "64m")


class SandboxManager:
    """
    Manages Docker-based code execution sandboxes.
    Each task gets a fresh, isolated container that is removed after execution.
    """

    def __init__(self) -> None:
        import docker
        self.client = docker.from_env()
        logger.info("SandboxManager initialized", image=SANDBOX_IMAGE)

    async def run(
        self,
        command: str,
        language: str = "shell",
        env: Optional[dict[str, str]] = None,
        files: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """
        Execute a command in an isolated Docker container.

        Args:
            command: Shell command or script to execute
            language: Hint for execution context (python, javascript, shell)
            env: Environment variables to inject
            files: Dict of filename → content to write before execution

        Returns:
            Dict with stdout, stderr, exit_code, duration_ms
        """
        container_id = f"agentic-sandbox-{uuid.uuid4().hex[:8]}"
        logger.info("Launching sandbox container", container=container_id, language=language)

        import time
        start = time.monotonic()

        try:
            output = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._run_sync(command, container_id, env or {}, files or {}),
            )
            duration_ms = int((time.monotonic() - start) * 1000)
            output["duration_ms"] = duration_ms
            return output

        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.error("Sandbox execution failed", container=container_id, error=str(exc))
            return {
                "stdout": "",
                "stderr": str(exc),
                "exit_code": -1,
                "success": False,
                "duration_ms": duration_ms,
            }

    def _run_sync(
        self,
        command: str,
        container_name: str,
        env: dict[str, str],
        files: dict[str, str],
    ) -> dict[str, Any]:
        """Synchronous Docker run (called from executor)."""
        # Inject files into the command using heredocs
        setup_commands = ""
        for fname, content in files.items():
            safe_content = content.replace("'", "'\\''")
            setup_commands += f"cat > /sandbox/workspace/{fname} << 'SANDBOXEOF'\n{content}\nSANDBOXEOF\n"

        full_command = f"{setup_commands}{command}" if setup_commands else command

        try:
            logs = self.client.containers.run(
                image=SANDBOX_IMAGE,
                command=["sh", "-c", full_command],
                name=container_name,
                remove=True,
                mem_limit=SANDBOX_MEMORY,
                cpu_period=100_000,
                cpu_quota=SANDBOX_CPU_QUOTA,
                network_mode="none",
                read_only=False,
                environment=env,
                tmpfs={"/tmp": f"noexec,nosuid,size={SANDBOX_TMP_SIZE}"},
                security_opt=["no-new-privileges:true"],
                cap_drop=["ALL"],
                stdout=True,
                stderr=True,
                timeout=SANDBOX_TIMEOUT,
            )
            stdout = logs.decode("utf-8", errors="replace") if isinstance(logs, bytes) else str(logs)
            return {"stdout": stdout[:10000], "stderr": "", "exit_code": 0, "success": True}

        except Exception as exc:
            return {"stdout": "", "stderr": str(exc)[:2000], "exit_code": 1, "success": False}

    def build_image(self) -> bool:
        """Build the sandbox image from Dockerfile.sandbox."""
        import subprocess
        sandbox_dir = os.path.dirname(os.path.abspath(__file__))
        result = subprocess.run(
            ["docker", "build", "-f", "Dockerfile.sandbox", "-t", SANDBOX_IMAGE, "."],
            cwd=sandbox_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info("Sandbox image built successfully", image=SANDBOX_IMAGE)
            return True
        logger.error("Sandbox image build failed", stderr=result.stderr[-500:])
        return False
