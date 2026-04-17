"""
guardrails/__init__.py — Public API surface for the Guardrail Engine.

Import everything you need from `guardrails`:

    from guardrails import GuardrailMiddleware, AgentAction, ActionType
    from guardrails import (
        PendingApprovalException, BlockedActionError,
        SafeActionResult, BlockedActionResult,
    )
"""

from guardrails.audit_logger import AuditLogger
from guardrails.command_classifier import CommandClassifier
from guardrails.exceptions import (
    BlockedActionError,
    NetworkPolicyViolation,
    PendingApprovalException,
    SandboxError,
    UndoBufferError,
)
from guardrails.middleware import GuardrailMiddleware
from guardrails.models import (
    ActionType,
    AgentAction,
    BlockedActionResult,
    CommandRiskResult,
    FileZone,
    NetworkPolicyResult,
    Outcome,
    RiskLevel,
    SafeActionResult,
    SandboxResult,
    SanitizedInput,
    SnapshotMetadata,
)
from guardrails.network_policy import NetworkPolicy
from guardrails.permission_checker import PermissionChecker
from guardrails.prompt_injection_defender import PromptInjectionDefender
from guardrails.sandbox_enforcer import SandboxEnforcer
from guardrails.undo_buffer import UndoBuffer

__all__ = [
    # Middleware (primary entry point)
    "GuardrailMiddleware",
    # Components
    "CommandClassifier",
    "PermissionChecker",
    "SandboxEnforcer",
    "AuditLogger",
    "UndoBuffer",
    "PromptInjectionDefender",
    "NetworkPolicy",
    # Models
    "AgentAction",
    "ActionType",
    "RiskLevel",
    "FileZone",
    "Outcome",
    "SafeActionResult",
    "BlockedActionResult",
    "SandboxResult",
    "SanitizedInput",
    "NetworkPolicyResult",
    "CommandRiskResult",
    "SnapshotMetadata",
    # Exceptions
    "PendingApprovalException",
    "BlockedActionError",
    "SandboxError",
    "NetworkPolicyViolation",
    "UndoBufferError",
]
