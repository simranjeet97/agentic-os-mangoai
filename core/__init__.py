"""
core/ — Agent Orchestration Engine for the Agentic AI OS.

Public surface:
  OrchestratorGraph      — LangGraph StateGraph (5-node pipeline)
  IntentParser           — classifies user input into intent types
  TaskQueue              — async priority queue with dependency DAG
  AgentCoordinator       — A2A message bus for agent delegation
  SessionManager         — per-user session state + interruption handling
  NaturalLanguageShell   — main entry point / interactive shell
"""

from core.orchestrator_graph import OrchestratorGraph, get_orchestrator_graph
from core.intent_parser import IntentParser, IntentType, ParsedIntent
from core.task_queue import TaskQueue, QueuedTask, QueuedTaskStatus
from core.agent_coordinator import AgentCoordinator, A2AMessage, A2AMessageType, get_coordinator
from core.session_manager import SessionManager, Session, get_session_manager
from core.shell import NaturalLanguageShell
from core.state import AgentState, AgentRole, TaskStatus, create_initial_state
from core.logging_config import get_logger

__all__ = [
    # ── Orchestration ──────────────────────────────────────────────────────────
    "OrchestratorGraph",
    "get_orchestrator_graph",
    # ── Intent ────────────────────────────────────────────────────────────────
    "IntentParser",
    "IntentType",
    "ParsedIntent",
    # ── Task Queue ────────────────────────────────────────────────────────────
    "TaskQueue",
    "QueuedTask",
    "QueuedTaskStatus",
    # ── A2A Coordination ──────────────────────────────────────────────────────
    "AgentCoordinator",
    "A2AMessage",
    "A2AMessageType",
    "get_coordinator",
    # ── Session ───────────────────────────────────────────────────────────────
    "SessionManager",
    "Session",
    "get_session_manager",
    # ── Shell ─────────────────────────────────────────────────────────────────
    "NaturalLanguageShell",
    # ── State ─────────────────────────────────────────────────────────────────
    "AgentState",
    "AgentRole",
    "TaskStatus",
    "create_initial_state",
    # ── Logging ───────────────────────────────────────────────────────────────
    "get_logger",
]
