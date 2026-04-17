"""
guardrails/permission_checker.py — PermissionChecker

Three-tier file zone system (READONLY, AGENT_WRITE, USER_CONFIRM) and
RBAC for agent types. Loads configuration from config/permissions.yaml.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

from guardrails.models import FileZone

# ---------------------------------------------------------------------------
# Default configuration (used if permissions.yaml is absent)
# ---------------------------------------------------------------------------

_DEFAULT_FILE_ZONES: dict[str, list[str]] = {
    "READONLY": [
        "/etc", "/usr", "/bin", "/sbin", "/lib", "/lib64",
        "/proc", "/sys", "/boot", "/dev",
    ],
    "AGENT_WRITE": [
        "/tmp/agent_workspace",
        "/home/agent/workspace",
        "/var/agent/scratch",
    ],
    "USER_CONFIRM": [
        "/home", "/root", "/var", "/opt",
    ],
}

_DEFAULT_AGENT_ROLES: dict[str, list[str]] = {
    "planner_agent":  ["memory_read", "memory_write", "task_decompose", "web_search"],
    "executor_agent": ["shell_exec", "docker_run", "file_write", "code_execute"],
    "file_agent":     ["file_read", "file_write", "file_delete"],
    "web_agent":      ["web_browse", "web_search"],
    "system_agent":   ["shell_exec", "process_list", "process_kill", "config_read"],
    "code_agent":     ["code_generate", "code_execute", "code_lint", "code_test"],
    "default":        ["file_read", "web_search", "code_generate", "memory_read"],
}

_PERMISSIONS_CONFIG = Path(__file__).parent.parent / "config" / "permissions.yaml"


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class FileZoneResult(BaseModel):
    path: str
    zone: FileZone
    access_allowed: bool
    requires_user_confirm: bool
    reason: str


class AgentPermissionResult(BaseModel):
    agent_id: str
    agent_type: str
    authorized: bool
    granted_capabilities: list[str] = Field(default_factory=list)
    denied_capabilities: list[str] = Field(default_factory=list)
    denied_paths: list[FileZoneResult] = Field(default_factory=list)
    requires_confirmation: list[FileZoneResult] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# PermissionChecker
# ---------------------------------------------------------------------------

class PermissionChecker:
    """
    Evaluates per-agent-type RBAC and path-based file zone restrictions.

    Usage:
        checker = PermissionChecker()
        result = await checker.check(
            agent_id="agent-001",
            agent_type="file_agent",
            requested_capabilities=["file_write"],
            target_paths=["/etc/hosts"],
        )
    """

    def __init__(self) -> None:
        cfg = self._load_config()
        self._file_zones: dict[str, list[str]] = cfg.get("file_zones", _DEFAULT_FILE_ZONES)
        self._agent_roles: dict[str, list[str]] = cfg.get("agent_roles", _DEFAULT_AGENT_ROLES)
        self._user_roles: dict[str, Any] = cfg.get("roles", {})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def check(
        self,
        agent_id: str,
        agent_type: str,
        requested_capabilities: list[str],
        target_paths: Optional[list[str]] = None,
        user_role: str = "user",
    ) -> AgentPermissionResult:
        """
        Full permission check: capabilities + file path zones.

        Returns AgentPermissionResult with authorized=True only if ALL
        requested capabilities are granted AND no READONLY paths are targeted.
        """
        # 1. Capability check
        allowed_caps = self._get_allowed_capabilities(agent_type, user_role)
        granted = [c for c in requested_capabilities if c in allowed_caps]
        denied_caps = [c for c in requested_capabilities if c not in allowed_caps]

        # 2. Path zone check
        path_results: list[FileZoneResult] = []
        if target_paths:
            for path in target_paths:
                path_results.append(self._classify_path(path, requested_capabilities))

        denied_paths = [p for p in path_results if not p.access_allowed]
        confirm_paths = [p for p in path_results if p.requires_user_confirm and p.access_allowed]

        # Authorized if: no denied capabilities AND no denied paths
        authorized = not denied_caps and not denied_paths

        return AgentPermissionResult(
            agent_id=agent_id,
            agent_type=agent_type,
            authorized=authorized,
            granted_capabilities=granted,
            denied_capabilities=denied_caps,
            denied_paths=denied_paths,
            requires_confirmation=confirm_paths,
        )

    def classify_path(self, path: str) -> FileZone:
        """Classify a filesystem path into a FileZone."""
        return self._classify_path(path, []).zone

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_allowed_capabilities(
        self, agent_type: str, user_role: str
    ) -> set[str]:
        """
        Union of capabilities from agent_type role and user role.
        Agent type caps constrain what the agent CAN do; user role further
        intersects with what the user is ALLOWED to request.
        """
        agent_caps = set(
            self._agent_roles.get(agent_type, self._agent_roles.get("default", []))
        )
        user_caps = set(
            self._user_roles.get(user_role, {}).get("capabilities", [])
        )
        # If a user role is configured, take the intersection; otherwise trust agent role
        if user_caps:
            return agent_caps & user_caps
        return agent_caps

    def _classify_path(
        self, path: str, requested_capabilities: list[str]
    ) -> FileZoneResult:
        """Determine the file zone for a path and whether access is allowed."""
        is_write = any(
            c in requested_capabilities
            for c in {"file_write", "file_delete", "shell_exec"}
        )

        # Check READONLY
        for prefix in self._file_zones.get("READONLY", []):
            if self._path_under(path, prefix):
                return FileZoneResult(
                    path=path,
                    zone=FileZone.READONLY,
                    access_allowed=False,
                    requires_user_confirm=False,
                    reason=f"Path is in READONLY zone (prefix: {prefix})",
                )

        # Check AGENT_WRITE
        for prefix in self._file_zones.get("AGENT_WRITE", []):
            if self._path_under(path, prefix):
                return FileZoneResult(
                    path=path,
                    zone=FileZone.AGENT_WRITE,
                    access_allowed=True,
                    requires_user_confirm=False,
                    reason=f"Path is in AGENT_WRITE zone (prefix: {prefix})",
                )

        # Check USER_CONFIRM
        for prefix in self._file_zones.get("USER_CONFIRM", []):
            if self._path_under(path, prefix):
                # Read access OK without confirm; write requires confirm
                return FileZoneResult(
                    path=path,
                    zone=FileZone.USER_CONFIRM,
                    access_allowed=not is_write,
                    requires_user_confirm=is_write,
                    reason=(
                        f"Path is in USER_CONFIRM zone (prefix: {prefix})"
                        + (" — human approval required for write" if is_write else "")
                    ),
                )

        # Unknown zone — default allow with audit
        return FileZoneResult(
            path=path,
            zone=FileZone.UNKNOWN,
            access_allowed=True,
            requires_user_confirm=False,
            reason="Path does not match any defined file zone — allowed by default",
        )

    @staticmethod
    def _path_under(path: str, prefix: str) -> bool:
        """Check if path is under a zone prefix (resolves symlinks conceptually)."""
        try:
            return Path(path).resolve().is_relative_to(Path(prefix).resolve())
        except (ValueError, RuntimeError):
            # Fallback to string matching
            norm_path = path.rstrip("/")
            norm_prefix = prefix.rstrip("/")
            return norm_path == norm_prefix or norm_path.startswith(norm_prefix + "/")

    @staticmethod
    def _load_config() -> dict[str, Any]:
        if _PERMISSIONS_CONFIG.exists():
            with open(_PERMISSIONS_CONFIG) as f:
                return yaml.safe_load(f) or {}
        return {}
