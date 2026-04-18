"""
agents/registry.py — AgentRegistry

Discovers, registers, and provides access to all agent instances.

Features:
  - Auto-discovery: scans agents/ subpackages for agent.py files
  - Health checks: verifies each agent is properly initialised
  - Metadata: exposes name, capabilities, description per agent
  - Singleton: one registry per process
  - Thread-safe lazy init

Usage:
    registry = AgentRegistry.get_instance()
    registry.register(MyAgent())
    agent = registry.get("code")
    all_agents = registry.list_all()
"""

from __future__ import annotations

import importlib
import threading
from pathlib import Path
from typing import Any, Optional

from agents.base_agent import BaseAgent
from core.logging_config import get_logger

logger = get_logger("agent.registry")


class AgentInfo:
    """Lightweight metadata record for a registered agent."""

    __slots__ = ("agent", "name", "description", "capabilities", "tools", "agent_id")

    def __init__(self, agent: BaseAgent) -> None:
        self.agent = agent
        self.name = agent.name
        self.description = agent.description
        self.capabilities: list[str] = agent.capabilities
        self.tools: list[str] = agent.tools
        self.agent_id: str = agent.agent_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "tools": self.tools,
            "agent_id": self.agent_id,
        }


class AgentRegistry:
    """
    Singleton registry of all specialist agents.

    Agents are discovered automatically from the agents/ directory.
    Additional agents can be registered at runtime.
    """

    _instance: Optional["AgentRegistry"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._agents: dict[str, AgentInfo] = {}

    # ── Singleton ─────────────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> "AgentRegistry":
        """Return or create the process-level singleton."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    cls._instance._auto_discover()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (useful for tests)."""
        with cls._lock:
            cls._instance = None

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, agent: BaseAgent) -> None:
        """Register a single agent instance."""
        info = AgentInfo(agent)
        self._agents[info.name] = info
        logger.info(
            "Agent registered",
            name=info.name,
            capabilities=info.capabilities,
            agent_id=info.agent_id,
        )

    def unregister(self, name: str) -> bool:
        """Remove an agent by name. Returns True if found and removed."""
        if name in self._agents:
            del self._agents[name]
            logger.info("Agent unregistered", name=name)
            return True
        return False

    # ── Lookup ────────────────────────────────────────────────────────────────

    def get(self, name: str) -> Optional[BaseAgent]:
        """Return the agent instance for a given name, or None."""
        info = self._agents.get(name)
        return info.agent if info else None

    def get_info(self, name: str) -> Optional[AgentInfo]:
        """Return the full AgentInfo record."""
        return self._agents.get(name)

    def get_by_capability(self, capability: str) -> list[BaseAgent]:
        """Return all agents that have a specific capability."""
        return [
            info.agent
            for info in self._agents.values()
            if capability in info.capabilities
        ]

    def get_by_tool(self, tool: str) -> list[BaseAgent]:
        """Return all agents that expose a specific tool."""
        return [
            info.agent
            for info in self._agents.values()
            if tool in info.tools
        ]

    def list_all(self) -> list[dict[str, Any]]:
        """Return metadata for all registered agents."""
        return [info.to_dict() for info in self._agents.values()]

    def list_names(self) -> list[str]:
        """Return registered agent names."""
        return list(self._agents.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._agents

    def __len__(self) -> int:
        return len(self._agents)

    # ── Auto-Discovery ────────────────────────────────────────────────────────

    def _auto_discover(self) -> None:
        """
        Scan agents/ subdirectories for agent.py files and import them.
        Each submodule must define exactly one class named <Name>Agent
        that extends BaseAgent.
        """
        agents_dir = Path(__file__).parent  # agents/
        SKIP_DIRS = {"__pycache__", ".git"}

        for sub in sorted(agents_dir.iterdir()):
            if not sub.is_dir() or sub.name in SKIP_DIRS:
                continue
            agent_module = sub / "agent.py"
            if not agent_module.exists():
                continue

            module_path = f"agents.{sub.name}.agent"
            try:
                mod = importlib.import_module(module_path)
            except Exception as exc:
                logger.warning(
                    "Auto-discovery: could not import module",
                    module=module_path,
                    error=str(exc),
                )
                continue

            # Find the BaseAgent subclass in the module
            for attr_name in dir(mod):
                obj = getattr(mod, attr_name)
                try:
                    if (
                        isinstance(obj, type)
                        and issubclass(obj, BaseAgent)
                        and obj is not BaseAgent
                        and attr_name.endswith("Agent")
                    ):
                        instance = obj()
                        self.register(instance)
                        break  # Only one agent class per module
                except Exception as exc:
                    logger.warning(
                        "Auto-discovery: could not instantiate",
                        class_name=attr_name,
                        error=str(exc),
                    )

        logger.info(
            "Auto-discovery complete",
            registered=list(self._agents.keys()),
        )

    # ── Health Check ──────────────────────────────────────────────────────────

    async def health_check(self) -> dict[str, Any]:
        """
        Verify all registered agents are functional.
        Returns {name: {healthy, agent_id, capabilities}}.
        """
        results: dict[str, Any] = {}
        for name, info in self._agents.items():
            try:
                # Just verify the agent object has the expected attributes
                assert hasattr(info.agent, "execute")
                assert hasattr(info.agent, "run")
                assert hasattr(info.agent, "agent_id")
                results[name] = {
                    "healthy": True,
                    "agent_id": info.agent_id,
                    "capabilities": info.capabilities,
                }
            except Exception as exc:
                results[name] = {
                    "healthy": False,
                    "error": str(exc),
                }
        return results
