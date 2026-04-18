"""
guardrails/exceptions.py — Custom exceptions for the Guardrail Engine.

Raised by GuardrailMiddleware to signal blocked or pending-approval actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PendingApprovalException(Exception):
    """
    Raised when an action's risk_score exceeds the auto-approval threshold (>6).

    Human review is required before the action may proceed. The caller should
    surface `explanation` to the end-user / operator dashboard.

    Attributes:
        action_id:              UUID of the blocked AgentAction.
        risk_score:             Computed risk score (0-10).
        explanation:            Human-readable reason for requiring approval.
        component:              Which guardrail component triggered the hold.
        suggested_alternatives: Safer alternative commands / approaches.
    """

    action_id: str
    risk_score: float
    explanation: str
    component: str = "GuardrailMiddleware"
    suggested_alternatives: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        alts = (
            "\n  Suggested alternatives:\n  - "
            + "\n  - ".join(self.suggested_alternatives)
            if self.suggested_alternatives
            else ""
        )
        return (
            f"[PendingApproval] action_id={self.action_id!r} "
            f"risk_score={self.risk_score:.1f}/10 "
            f"component={self.component!r}\n"
            f"  {self.explanation}{alts}"
        )

    def to_dict(self) -> dict:
        return {
            "error": "pending_approval",
            "action_id": self.action_id,
            "risk_score": self.risk_score,
            "explanation": self.explanation,
            "component": self.component,
            "suggested_alternatives": self.suggested_alternatives,
        }


@dataclass
class BlockedActionError(Exception):
    """
    Raised when an action is unconditionally blocked (blocklist hit, critical violation).

    Unlike PendingApprovalException, BlockedActionError cannot be overridden by
    human approval — the action is permanently forbidden in this context.

    Attributes:
        action_id:  UUID of the blocked AgentAction.
        reason:     Short machine-readable code (e.g. "BLOCKLIST_HIT").
        violations: Detailed list of rule violations that triggered the block.
    """

    action_id: str
    reason: str
    violations: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        viol = "\n  - ".join(self.violations) if self.violations else "unspecified"
        return (
            f"[BlockedAction] action_id={self.action_id!r} reason={self.reason!r}\n"
            f"  Violations:\n  - {viol}"
        )

    def to_dict(self) -> dict:
        return {
            "error": "blocked_action",
            "action_id": self.action_id,
            "reason": self.reason,
            "violations": self.violations,
        }


class SandboxError(Exception):
    """Raised when the sandbox container fails to start or times out."""


class NetworkPolicyViolation(Exception):
    """Raised when an agent attempts a network call that violates policy."""


class UndoBufferError(Exception):
    """Raised when a snapshot or rollback operation fails."""
