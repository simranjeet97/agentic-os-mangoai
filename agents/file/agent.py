"""
agents/file/agent.py — File Agent.
Sandboxed file system operations: read, write, search, list.
All paths are validated against allowed directories.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles

from agents.base_agent import BaseAgent
from core.state import AgentState

# Restrict file operations to this base directory
SANDBOX_ROOT = Path(os.getenv("FILE_SANDBOX_ROOT", "/tmp/agentic-sandbox"))
ALLOWED_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml",
    ".csv", ".html", ".css", ".sh", ".log", ".xml", ".toml",
}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB


class FileAgent(BaseAgent):
    name = "file"
    description = "Sandboxed file read/write/search operations"

    def _safe_path(self, path: str) -> Path:
        """Resolve and validate path stays within sandbox."""
        SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)
        resolved = (SANDBOX_ROOT / path).resolve()
        if not str(resolved).startswith(str(SANDBOX_ROOT.resolve())):
            raise PermissionError(f"Path escape attempt detected: {path}")
        return resolved

    async def execute(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        async def _run():
            action = step.get("action", "read")
            path = step.get("path", "")
            content = step.get("content", "")

            if action == "read":
                return await self._read(path)
            elif action == "write":
                return await self._write(path, content)
            elif action == "list":
                return await self._list(path)
            elif action == "search":
                return await self._search(path, step.get("query", ""))
            else:
                self.logger.info(
                    "File agent executing step",
                    description=step.get("description", "")[:80],
                )
                return {"output": f"File action '{action}' acknowledged", "step_type": "file"}

        return await self._run_with_audit(step, state, _run)

    async def _read(self, path: str) -> dict[str, Any]:
        safe = self._safe_path(path)
        if not safe.exists():
            return {"success": False, "error": f"File not found: {path}"}
        if safe.stat().st_size > MAX_FILE_SIZE_BYTES:
            return {"success": False, "error": "File too large (>10MB)"}
        if safe.suffix not in ALLOWED_EXTENSIONS:
            return {"success": False, "error": f"Extension not allowed: {safe.suffix}"}
        async with aiofiles.open(safe, "r", errors="replace") as f:
            content = await f.read()
        return {"output": content, "path": str(safe), "step_type": "file_read"}

    async def _write(self, path: str, content: str) -> dict[str, Any]:
        safe = self._safe_path(path)
        if safe.suffix not in ALLOWED_EXTENSIONS:
            return {"success": False, "error": f"Extension not allowed: {safe.suffix}"}
        safe.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(safe, "w") as f:
            await f.write(content)
        return {"output": f"Written {len(content)} bytes to {path}", "path": str(safe), "step_type": "file_write"}

    async def _list(self, path: str) -> dict[str, Any]:
        safe = self._safe_path(path)
        if not safe.is_dir():
            return {"success": False, "error": f"Not a directory: {path}"}
        entries = [
            {"name": e.name, "type": "dir" if e.is_dir() else "file", "size": e.stat().st_size if e.is_file() else None}
            for e in sorted(safe.iterdir())[:200]
        ]
        return {"output": entries, "path": str(safe), "step_type": "file_list"}

    async def _search(self, path: str, query: str) -> dict[str, Any]:
        safe = self._safe_path(path)
        matches = []
        for p in safe.rglob("*"):
            if p.is_file() and query.lower() in p.name.lower():
                matches.append(str(p.relative_to(SANDBOX_ROOT)))
            if len(matches) >= 50:
                break
        return {"output": matches, "query": query, "step_type": "file_search"}
