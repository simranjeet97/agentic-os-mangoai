"""
core/state.py — Shared AgentState definition using TypedDict + Pydantic.
This is the single source of truth for LangGraph graph state.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentRole(str, Enum):
    PLANNER = "planner"
    EXECUTOR = "executor"
    FILE = "file"
    WEB = "web"
    SYSTEM = "system"
    CODE = "code"
    GUARDIAN = "guardian"
    ORCHESTRATOR = "orchestrator"


class TaskStep(BaseModel):
    """A single decomposed task step."""

    step_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str
    agent: AgentRole
    dependencies: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        use_enum_values = True


class MemoryContext(BaseModel):
    """Memory context attached to the current task."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "system"
    episodic: list[dict[str, Any]] = Field(default_factory=list)
    semantic: list[dict[str, Any]] = Field(default_factory=list)
    working: dict[str, Any] = Field(default_factory=dict)


class GuardrailResult(BaseModel):
    """Result of guardrail evaluation."""

    passed: bool
    risk_level: str = "low"   # low | medium | high | critical
    violations: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class AgentState(TypedDict):
    """
    The central state object passed through all LangGraph nodes.
    Uses add_messages reducer for safe message list management.
    """

    # ── Identity ─────────────────────────────────────────────────────────────
    task_id: str
    session_id: str
    user_id: str

    # ── Conversation ─────────────────────────────────────────────────────────
    messages: Annotated[list[AnyMessage], add_messages]
    user_input: str

    # ── Planning ─────────────────────────────────────────────────────────────
    goal: str
    plan: list[dict[str, Any]]           # List of TaskStep dicts
    current_step_index: int
    active_agent: Optional[str]

    # ── Execution context ─────────────────────────────────────────────────────
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    artifacts: list[dict[str, Any]]      # Files, code outputs, etc.

    # ── Status ────────────────────────────────────────────────────────────────
    status: str                          # TaskStatus value
    error: Optional[str]
    iterations: int

    # ── Memory ────────────────────────────────────────────────────────────────
    memory: dict[str, Any]              # MemoryContext dict

    # ── Guardrails ────────────────────────────────────────────────────────────
    guardrail_result: Optional[dict[str, Any]]
    requires_approval: bool

    # ── Metadata ─────────────────────────────────────────────────────────────
    metadata: dict[str, Any]
    created_at: str
    updated_at: str


def create_initial_state(
    user_input: str,
    user_id: str = "anonymous",
    session_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    history: Optional[list[AnyMessage]] = None,
) -> AgentState:
    """Create a fresh AgentState for a new task."""
    now = datetime.utcnow().isoformat()
    return AgentState(
        task_id=str(uuid.uuid4()),
        session_id=session_id or str(uuid.uuid4()),
        user_id=user_id,
        messages=history or [],
        user_input=user_input,
        goal=user_input,
        plan=[],
        current_step_index=0,
        active_agent=None,
        tool_calls=[],
        tool_results=[],
        artifacts=[],
        status=TaskStatus.PENDING.value,
        error=None,
        iterations=0,
        memory={},
        guardrail_result=None,
        requires_approval=False,
        metadata=metadata or {},
        created_at=now,
        updated_at=now,
    )
