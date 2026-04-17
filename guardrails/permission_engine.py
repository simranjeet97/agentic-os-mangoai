"""
guardrails/permission_engine.py — RBAC-based capability permission system.
Loads role definitions from config/permissions.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from core.logging_config import get_logger

logger = get_logger(__name__)

PERMISSIONS_CONFIG = Path(__file__).parent.parent / "config" / "permissions.yaml"


class PermissionResult(BaseModel):
    authorized: bool
    granted_capabilities: list[str] = Field(default_factory=list)
    denied_capabilities: list[str] = Field(default_factory=list)


class PermissionEngine:
    """RBAC engine loaded from config/permissions.yaml."""

    def __init__(self) -> None:
        self._config = self._load_config()
        logger.info("PermissionEngine loaded", roles=list(self._config.get("roles", {}).keys()))

    def _load_config(self) -> dict[str, Any]:
        if PERMISSIONS_CONFIG.exists():
            with open(PERMISSIONS_CONFIG) as f:
                return yaml.safe_load(f) or {}
        logger.warning("permissions.yaml not found, using permissive defaults")
        return {"roles": {"default": {"capabilities": ["file_read", "web_browse", "code_generate"]}}}

    async def check(
        self,
        user_id: str,
        capabilities: list[str],
        role: str = "default",
    ) -> PermissionResult:
        """Check if a role has the requested capabilities."""
        roles = self._config.get("roles", {})
        role_config = roles.get(role, roles.get("default", {}))
        allowed = set(role_config.get("capabilities", []))

        granted = [c for c in capabilities if c in allowed]
        denied = [c for c in capabilities if c not in allowed]

        return PermissionResult(
            authorized=len(denied) == 0,
            granted_capabilities=granted,
            denied_capabilities=denied,
        )
