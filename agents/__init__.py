"""
agents/ — Agentic OS Agent Package

Public API:
  from agents import (
      BaseAgent, AgentResult,
      PlannerAgent, ExecutorAgent, FileAgent,
      WebAgent, SystemAgent, CodeAgent,
      AgentRegistry, AgentRouter, get_router,
  )

  # Auto-discover and get the global registry
  registry = AgentRegistry.get_instance()

  # Route a task to the best agent
  router = get_router()
  agent = await router.route("Write a Python script to parse JSON")
  result = await agent.run({"description": "..."})
"""

from agents.base_agent import AgentResult, BaseAgent
from agents.code.agent import CodeAgent
from agents.executor.agent import ExecutorAgent
from agents.file.agent import FileAgent
from agents.planner.agent import PlannerAgent
from agents.registry import AgentRegistry
from agents.router import AgentRouter, get_router
from agents.system.agent import SystemAgent
from agents.web.agent import WebAgent

__all__ = [
    # Base
    "BaseAgent",
    "AgentResult",
    # Specialists
    "PlannerAgent",
    "ExecutorAgent",
    "FileAgent",
    "WebAgent",
    "SystemAgent",
    "CodeAgent",
    # Infrastructure
    "AgentRegistry",
    "AgentRouter",
    "get_router",
]
