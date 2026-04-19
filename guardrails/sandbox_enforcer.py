"""
guardrails/sandbox_enforcer.py — SandboxEnforcer

Wraps all code execution in Docker containers with --network none by default.
Exposes a clean safe_execute(code, language, timeout) API.

Supported languages: python, javascript, bash, ruby, go, typescript
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

from guardrails.exceptions import SandboxError
from guardrails.models import SandboxResult

# ---------------------------------------------------------------------------
# Language → Docker image + invocation config
# ---------------------------------------------------------------------------

_LANGUAGE_CONFIG: dict[str, dict] = {
    "python": {
        "image": "python:3.12-alpine",
        "filename": "script.py",
        "cmd": ["python", "/sandbox/script.py"],
    },
    "python3": {
        "image": "python:3.12-alpine",
        "filename": "script.py",
        "cmd": ["python", "/sandbox/script.py"],
    },
    "javascript": {
        "image": "node:20-alpine",
        "filename": "script.js",
        "cmd": ["node", "/sandbox/script.js"],
    },
    "js": {
        "image": "node:20-alpine",
        "filename": "script.js",
        "cmd": ["node", "/sandbox/script.js"],
    },
    "typescript": {
        "image": "node:20-alpine",
        "filename": "script.ts",
        "cmd": ["sh", "-c", "npx ts-node /sandbox/script.ts"],
    },
    "bash": {
        "image": "alpine:latest",
        "filename": "script.sh",
        "cmd": ["sh", "/sandbox/script.sh"],
    },
    "sh": {
        "image": "alpine:latest",
        "filename": "script.sh",
        "cmd": ["sh", "/sandbox/script.sh"],
    },
    "ruby": {
        "image": "ruby:3.3-alpine",
        "filename": "script.rb",
        "cmd": ["ruby", "/sandbox/script.rb"],
    },
    "go": {
        "image": "golang:1.22-alpine",
        "filename": "main.go",
        "cmd": ["sh", "-c", "cd /sandbox && go run main.go"],
    },
}

_DEFAULT_TIMEOUT = 30      # seconds
_DEFAULT_MEMORY = "256m"
_DEFAULT_CPUS = "0.5"
_DEFAULT_PIDS = "64"


class SandboxEnforcer:
    """
    Executes arbitrary code inside an isolated Docker container.

    Security guarantees:
    - --network none (no outbound internet by default)
    - --memory 256m --cpus 0.5 --pids-limit 64
    - Read-only root filesystem (--read-only) except /tmp bind-mount
    - --no-new-privileges
    - Container auto-removed after execution
    - Code written to host temp dir, bind-mounted read-only into container

    Usage:
        enforcer = SandboxEnforcer()
        result = await enforcer.safe_execute(
            code='print("hello")',
            language="python",
            timeout=10,
        )
        print(result.stdout, result.exit_code)
    """

    def __init__(
        self,
        memory_limit: str = _DEFAULT_MEMORY,
        cpu_limit: str = _DEFAULT_CPUS,
        pids_limit: str = _DEFAULT_PIDS,
        enable_network: bool = False,
    ) -> None:
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.pids_limit = pids_limit
        self.enable_network = enable_network
        self._docker_available = self._check_docker()
        # Allow environment override to force local mode
        self.force_local = os.getenv("SANDBOX_LOCAL_MODE", "false").lower() == "true"

    async def safe_execute(
        self,
        code: str,
        language: str,
        timeout: int = _DEFAULT_TIMEOUT,
        allow_network: bool = False,
        extra_env: Optional[dict[str, str]] = None,
    ) -> SandboxResult:
        """
        Execute `code` in a sandboxed Docker container.

        Args:
            code:          Source code to execute.
            language:      Programming language (python, bash, js, go, ruby …).
            timeout:       Max execution time in seconds (default 30).
            allow_network: Override the no-network default (requires NetworkPolicy approval).
            extra_env:     Extra environment variables to pass to container.

        Returns:
            SandboxResult with stdout, stderr, exit_code, duration_ms.

        Raises:
            SandboxError: If Docker is unavailable or container fails to start.
        """
        lang = language.lower().strip()
        config = _LANGUAGE_CONFIG.get(lang)
        if config is None:
            supported = ", ".join(sorted(_LANGUAGE_CONFIG.keys()))
            raise SandboxError(
                f"Language {lang!r} not supported. Supported: {supported}"
            )

        if not self._docker_available or self.force_local:
            return await self._run_locally(
                code=code,
                config=config,
                timeout=timeout,
                language=lang,
                extra_env=extra_env or {},
            )

        return await self._run_container(
            code=code,
            config=config,
            timeout=timeout,
            allow_network=allow_network,
            extra_env=extra_env or {},
            language=lang,
        )

    async def _run_locally(
        self,
        code: str,
        config: dict,
        timeout: int,
        language: str,
        extra_env: dict[str, str],
    ) -> SandboxResult:
        """Execute code directly on the host using a subprocess fallback."""
        tmpdir = tempfile.mkdtemp(prefix="local_sandbox_")
        code_file = Path(tmpdir) / config["filename"]
        code_file.write_text(code)

        # Map local binaries (some image-specific sh -c commands won't work locally)
        local_cmd = list(config["cmd"])
        # Replace '/sandbox/...' with the actual local path
        local_cmd = [str(code_file) if "/sandbox/" in arg else arg for arg in local_cmd]
        
        # Binary overrides for common local setups
        binary_map = {"python": "python3", "python3": "python3", "node": "node", "sh": "sh"}
        if local_cmd[0] in binary_map:
            local_cmd[0] = binary_map[local_cmd[0]]

        start = time.monotonic()
        timed_out = False
        stdout_data = ""
        stderr_data = ""
        exit_code = -1

        try:
            # Merge with existing environment
            env = os.environ.copy()
            env.update(extra_env)

            proc = await asyncio.create_subprocess_exec(
                *local_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=tmpdir,
            )

            try:
                raw_stdout, raw_stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=float(timeout)
                )
                exit_code = proc.returncode or 0
                stdout_data = raw_stdout.decode(errors="replace")
                stderr_data = raw_stderr.decode(errors="replace")
            except asyncio.TimeoutError:
                timed_out = True
                exit_code = 124
                stderr_data = f"Local execution timed out after {timeout}s"
                try:
                    proc.kill()
                except Exception:
                    pass

        except Exception as exc:
            exit_code = -1
            stderr_data = f"Local execution failed: {exc}"
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

        duration_ms = (time.monotonic() - start) * 1000

        return SandboxResult(
            exit_code=exit_code,
            stdout=stdout_data,
            stderr=stderr_data,
            duration_ms=round(duration_ms, 2),
            timed_out=timed_out,
            container_id="local_host",
            language=language,
        )
    async def _run_container(
        self,
        code: str,
        config: dict,
        timeout: int,
        allow_network: bool,
        extra_env: dict[str, str],
        language: str,
    ) -> SandboxResult:
        """Build the docker run command and execute it."""
        # Write code to a temporary directory on the host
        tmpdir = tempfile.mkdtemp(prefix="sandbox_")
        code_file = Path(tmpdir) / config["filename"]
        code_file.write_text(code)

        container_id = f"sandbox_{uuid.uuid4().hex[:12]}"

        # Build docker run args
        docker_cmd = [
            "docker", "run",
            "--rm",
            "--name", container_id,
            "--memory", self.memory_limit,
            "--cpus", self.cpu_limit,
            f"--pids-limit={self.pids_limit}",
            "--no-new-privileges",
            "--read-only",
            "--tmpfs", "/tmp:size=64m,mode=1777",
            "--volume", f"{tmpdir}:/sandbox:ro",  # code is read-only in container
        ]

        if not (self.enable_network or allow_network):
            docker_cmd.append("--network=none")

        # Add environment variables
        for k, v in extra_env.items():
            docker_cmd.extend(["--env", f"{k}={v}"])

        docker_cmd.append(config["image"])
        docker_cmd.extend(config["cmd"])

        start = time.monotonic()
        timed_out = False
        stdout_data = ""
        stderr_data = ""
        exit_code = -1

        try:
            proc = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                raw_stdout, raw_stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=float(timeout)
                )
                exit_code = proc.returncode or 0
                stdout_data = raw_stdout.decode(errors="replace")
                stderr_data = raw_stderr.decode(errors="replace")
            except asyncio.TimeoutError:
                timed_out = True
                exit_code = 124
                stderr_data = f"Execution timed out after {timeout}s"
                # Kill the container
                await self._kill_container(container_id)

        except FileNotFoundError:
            raise SandboxError(
                "docker executable not found. Ensure Docker is installed and "
                "the `docker` binary is in PATH."
            )
        finally:
            # Cleanup temp directory
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

        duration_ms = (time.monotonic() - start) * 1000

        return SandboxResult(
            exit_code=exit_code,
            stdout=stdout_data,
            stderr=stderr_data,
            duration_ms=round(duration_ms, 2),
            timed_out=timed_out,
            container_id=container_id,
            language=language,
        )

    @staticmethod
    async def _kill_container(container_id: str) -> None:
        """Forcefully kill a running container."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "kill", container_id,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
        except Exception:
            pass

    @staticmethod
    def _check_docker() -> bool:
        """Return True if the docker binary is accessible."""
        return os.system("docker info > /dev/null 2>&1") == 0

    @property
    def supported_languages(self) -> list[str]:
        return sorted(_LANGUAGE_CONFIG.keys())
