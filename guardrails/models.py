"""
guardrails/models.py — Shared data models for the Guardrail Engine.

All inter-component data flows through these Pydantic models, providing
type safety, validation, and JSON serialisation throughout the chain.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ActionType(str, Enum):
    SHELL_COMMAND = "shell_command"
    CODE_EXECUTION = "code_execution"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    NETWORK_REQUEST = "network_request"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    PROCESS_SPAWN = "process_spawn"
    CONFIG_CHANGE = "config_change"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    SAFE = "safe"            # 0-2
    LOW = "low"              # 3-4
    MEDIUM = "medium"        # 5-6
    HIGH = "high"            # 7-8  → PendingApprovalException
    CRITICAL = "critical"    # 9-10 → BlockedActionError


class Outcome(str, Enum):
    SUCCESS = "success"
    BLOCKED = "blocked"
    PENDING = "pending"
    ERROR = "error"
    ROLLED_BACK = "rolled_back"


class FileZone(str, Enum):
    READONLY = "READONLY"
    AGENT_WRITE = "AGENT_WRITE"
    USER_CONFIRM = "USER_CONFIRM"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Input / Action models
# ---------------------------------------------------------------------------


class AgentAction(BaseModel):
    """Represents a single action an agent wants to perform."""

    action_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    agent_type: str = "unknown"           # planner_agent, executor_agent, …
    action_type: ActionType = ActionType.UNKNOWN
    command: Optional[str] = None         # shell command string if applicable
    code: Optional[str] = None            # code snippet if applicable
    language: Optional[str] = None        # python | bash | javascript …
    target_paths: list[str] = Field(default_factory=list)
    url: Optional[str] = None             # target URL if network action
    raw_input: str = ""                   # original user/agent prompt
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def is_destructive(self) -> bool:
        """True if this action could overwrite or delete data."""
        return self.action_type in {
            ActionType.FILE_WRITE,
            ActionType.FILE_DELETE,
            ActionType.SHELL_COMMAND,
            ActionType.CODE_EXECUTION,
        }


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


def _risk_level_from_score(score: float) -> RiskLevel:
    if score <= 2:
        return RiskLevel.SAFE
    if score <= 4:
        return RiskLevel.LOW
    if score <= 6:
        return RiskLevel.MEDIUM
    if score <= 8:
        return RiskLevel.HIGH
    return RiskLevel.CRITICAL


class SafeActionResult(BaseModel):
    """Returned when the action passes all guardrail checks."""

    action_id: str
    agent_id: str
    action_type: ActionType
    risk_score: float
    risk_level: RiskLevel
    sanitized_input: Optional[str] = None
    audit_id: Optional[str] = None
    sandbox_result: Optional["SandboxResult"] = None
    approved_by: Optional[str] = None      # None = auto-approved
    message: str = "Action approved"
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @classmethod
    def build(
        cls,
        action: AgentAction,
        risk_score: float,
        audit_id: Optional[str] = None,
        sandbox_result: Optional["SandboxResult"] = None,
        approved_by: Optional[str] = None,
        sanitized_input: Optional[str] = None,
    ) -> "SafeActionResult":
        return cls(
            action_id=action.action_id,
            agent_id=action.agent_id,
            action_type=action.action_type,
            risk_score=risk_score,
            risk_level=_risk_level_from_score(risk_score),
            sanitized_input=sanitized_input,
            audit_id=audit_id,
            sandbox_result=sandbox_result,
            approved_by=approved_by,
        )


class BlockedActionResult(BaseModel):
    """Returned when the action is unconditionally blocked."""

    action_id: str
    agent_id: str
    action_type: ActionType
    risk_score: float
    risk_level: RiskLevel
    reason: str
    violations: list[str] = Field(default_factory=list)
    audit_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @classmethod
    def build(
        cls,
        action: AgentAction,
        risk_score: float,
        reason: str,
        violations: list[str],
        audit_id: Optional[str] = None,
    ) -> "BlockedActionResult":
        return cls(
            action_id=action.action_id,
            agent_id=action.agent_id,
            action_type=action.action_type,
            risk_score=risk_score,
            risk_level=_risk_level_from_score(risk_score),
            reason=reason,
            violations=violations,
            audit_id=audit_id,
        )


# ---------------------------------------------------------------------------
# Sandbox models
# ---------------------------------------------------------------------------


class SandboxResult(BaseModel):
    """Output from SandboxEnforcer.safe_execute()."""

    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False
    container_id: Optional[str] = None
    language: str = "unknown"


# ---------------------------------------------------------------------------
# Prompt injection models
# ---------------------------------------------------------------------------


class SanitizedInput(BaseModel):
    """Result of PromptInjectionDefender.sanitize()."""

    original: str
    sanitized: str
    was_modified: bool
    threats_detected: list[str] = Field(default_factory=list)
    is_safe: bool = True


# ---------------------------------------------------------------------------
# Network policy models
# ---------------------------------------------------------------------------


class NetworkPolicyResult(BaseModel):
    """Result of NetworkPolicy.check_url()."""

    url: str
    domain: str
    allowed: bool
    reason: str
    matched_rule: Optional[str] = None


# ---------------------------------------------------------------------------
# Undo buffer models
# ---------------------------------------------------------------------------


class SnapshotMetadata(BaseModel):
    """Metadata for a single UndoBuffer snapshot."""

    snapshot_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    operation_id: str
    agent_id: str
    action_type: ActionType
    paths: list[str]
    archive_path: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    restored: bool = False


# ---------------------------------------------------------------------------
# Command classification models
# ---------------------------------------------------------------------------


class CommandRiskResult(BaseModel):
    """Risk assessment for a shell command."""

    command: str
    risk_score: float          # 0.0 – 10.0
    risk_level: RiskLevel
    is_blocked: bool           # True = unconditional blocklist hit
    rule_score: float
    llm_score: Optional[float] = None
    explanation: str
    matched_rules: list[str] = Field(default_factory=list)
    suggested_alternatives: list[str] = Field(default_factory=list)
