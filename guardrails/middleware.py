"""
guardrails/middleware.py — GuardrailMiddleware

The central chain that every agent action must pass through.
Composes all 7 guardrail components in a deterministic order and returns
SafeActionResult or BlockedActionResult. Raises PendingApprovalException
when risk_score > 6.

Chain order:
  1. PromptInjectionDefender  — sanitise raw input, detect jailbreaks
  2. CommandClassifier        — score shell commands 0-10
  3. PermissionChecker        — file zones + RBAC
  4. NetworkPolicy            — whitelist outbound URLs
  5. UndoBuffer.snapshot      — save FS state before destructive ops
  6. AuditLogger.log          — always write an immutable audit record
  7. Return result            — SafeActionResult or BlockedActionResult
"""

from __future__ import annotations

import uuid
from typing import Optional

from guardrails.audit_logger import AuditLogger
from guardrails.command_classifier import CommandClassifier
from guardrails.exceptions import BlockedActionError, PendingApprovalException
from guardrails.models import (
    ActionType,
    AgentAction,
    BlockedActionResult,
    Outcome,
    SafeActionResult,
)
from guardrails.network_policy import NetworkPolicy
from guardrails.permission_checker import PermissionChecker
from guardrails.prompt_injection_defender import PromptInjectionDefender
from guardrails.sandbox_enforcer import SandboxEnforcer
from guardrails.undo_buffer import UndoBuffer

# Risk score threshold above which human approval is required
_APPROVAL_THRESHOLD = 6.0
# Risk score at which action is unconditionally blocked (regardless of approval)
_BLOCK_THRESHOLD = 9.5


class GuardrailMiddleware:
    """
    The single entry point for all agent actions.

    Instantiate once and share across the agent stack. Each component is
    lazy-initialised the first time evaluate_action() is called.

    Usage:
        middleware = GuardrailMiddleware()

        try:
            result = await middleware.evaluate_action(action)
            # result is SafeActionResult
        except PendingApprovalException as e:
            # surface e.explanation to operator dashboard
            dashboard.request_approval(e)
        except BlockedActionError as e:
            # log and reject unconditionally
            logger.error(str(e))
    """

    def __init__(
        self,
        llm_model: Optional[str] = None,
        approval_threshold: float = _APPROVAL_THRESHOLD,
        block_threshold: float = _BLOCK_THRESHOLD,
        audit_db_path: Optional[str] = None,
        undo_dir: Optional[str] = None,
        extra_allowed_domains: Optional[list[str]] = None,
    ) -> None:
        self._approval_threshold = approval_threshold
        self._block_threshold = block_threshold

        # Initialise all components
        self.defender = PromptInjectionDefender(strict_mode=True)
        self.classifier = CommandClassifier(llm_model=llm_model)
        self.checker = PermissionChecker()
        self.network = NetworkPolicy(extra_allowed_domains=extra_allowed_domains)
        self.sandbox = SandboxEnforcer()
        self.audit = AuditLogger(db_path=audit_db_path)
        self.undo = UndoBuffer(undo_dir=undo_dir)

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    async def evaluate_action(
        self,
        action: AgentAction,
        approved_by: Optional[str] = None,
        skip_snapshot: bool = False,
        user_role: str = "user",
    ) -> SafeActionResult:
        """
        Run all guardrail checks on `action`.

        Args:
            action:       The AgentAction to evaluate.
            approved_by:  If a human has pre-approved this action, pass their ID.
                          Pre-approval bypasses PendingApprovalException but still
                          audits and enforces blocklist.
            skip_snapshot: If True, skip UndoBuffer snapshot (for read-only ops).
            user_role:    The RBAC role of the triggering user (defaults to "user").

        Returns:
            SafeActionResult if all checks pass.

        Raises:
            PendingApprovalException: risk_score > approval_threshold (default 6).
            BlockedActionError:       blocklist hit or risk_score > block_threshold.
        """
        violations: list[str] = []
        risk_score = 0.0
        audit_id: Optional[str] = None
        sanitized_text: Optional[str] = None

        # ── Step 1: Prompt Injection Defence ──────────────────────────────────
        if action.raw_input:
            san = await self.defender.sanitize(action.raw_input)
            sanitized_text = san.sanitized
            if not san.is_safe:
                violations.extend(san.threats_detected)
                risk_score = max(risk_score, 8.0)
                # Immediate block for confirmed jailbreak
                audit_id = await self._log(
                    action, risk_score, Outcome.BLOCKED,
                    details={
                        "step": "prompt_injection_defender",
                        "threats": san.threats_detected,
                    },
                )
                result = BlockedActionResult.build(
                    action=action,
                    risk_score=risk_score,
                    reason="PROMPT_INJECTION_DETECTED",
                    violations=violations,
                    audit_id=audit_id,
                )
                raise BlockedActionError(
                    action_id=action.action_id,
                    reason="PROMPT_INJECTION_DETECTED",
                    violations=violations,
                )

        # ── Step 2: Command Classification ────────────────────────────────────
        cmd_score = 0.0
        cmd_alternatives: list[str] = []
        if action.command:
            cmd_result = await self.classifier.classify(action.command)
            cmd_score = cmd_result.risk_score
            cmd_alternatives = cmd_result.suggested_alternatives

            if cmd_result.is_blocked:
                # Unconditional block — no amount of approval can override
                audit_id = await self._log(
                    action, 10.0, Outcome.BLOCKED,
                    details={
                        "step": "command_classifier",
                        "command": action.command,
                        "reason": cmd_result.explanation,
                    },
                )
                raise BlockedActionError(
                    action_id=action.action_id,
                    reason="BLOCKLIST_HIT",
                    violations=[cmd_result.explanation],
                )

            risk_score = max(risk_score, cmd_score)
            violations.extend(cmd_result.matched_rules)

        # ── Step 3: Permission Check ───────────────────────────────────────────
        perm_result = await self.checker.check(
            agent_id=action.agent_id,
            agent_type=action.agent_type,
            requested_capabilities=self._infer_capabilities(action),
            target_paths=action.target_paths,
            user_role=user_role,
        )

        if not perm_result.authorized:
            denied_caps = perm_result.denied_capabilities
            denied_paths = [p.path for p in perm_result.denied_paths]
            deny_violations = [
                *(f"Capability denied: {c}" for c in denied_caps),
                *(f"Path in READONLY zone: {p}" for p in denied_paths),
            ]
            violations.extend(deny_violations)
            risk_score = max(risk_score, 7.0)
            audit_id = await self._log(
                action, risk_score, Outcome.BLOCKED,
                details={
                    "step": "permission_checker",
                    "denied_capabilities": denied_caps,
                    "denied_paths": denied_paths,
                },
            )
            raise BlockedActionError(
                action_id=action.action_id,
                reason="PERMISSION_DENIED",
                violations=deny_violations,
            )

        # Paths requiring user confirmation raise PendingApprovalException
        if perm_result.requires_confirmation and not approved_by:
            confirm_paths = [p.path for p in perm_result.requires_confirmation]
            risk_score = max(risk_score, self._approval_threshold + 1)
            audit_id = await self._log(
                action, risk_score, Outcome.PENDING,
                details={
                    "step": "permission_checker",
                    "requires_confirmation": confirm_paths,
                },
            )
            raise PendingApprovalException(
                action_id=action.action_id,
                risk_score=risk_score,
                explanation=(
                    f"Action targets USER_CONFIRM zone paths: {confirm_paths}. "
                    "Human approval is required before proceeding."
                ),
                component="PermissionChecker",
                suggested_alternatives=[
                    "Move files to /tmp/agent_workspace instead",
                    "Request operator approval via the dashboard",
                ],
            )

        # ── Step 4: Network Policy ─────────────────────────────────────────────
        if action.url:
            net_result = self.network.check_url(action.url)
            self.network.log_external_call(
                url=action.url,
                agent_id=action.agent_id,
                outcome="allowed" if net_result.allowed else "blocked",
            )
            if not net_result.allowed:
                violations.append(f"Network policy violation: {net_result.reason}")
                risk_score = max(risk_score, 7.0)
                audit_id = await self._log(
                    action, risk_score, Outcome.BLOCKED,
                    details={
                        "step": "network_policy",
                        "url": action.url,
                        "domain": net_result.domain,
                        "reason": net_result.reason,
                    },
                )
                raise BlockedActionError(
                    action_id=action.action_id,
                    reason="NETWORK_POLICY_VIOLATION",
                    violations=[net_result.reason],
                )

        # ── Step 5: Risk threshold check (approval gate) ───────────────────────
        if risk_score > self._block_threshold and not approved_by:
            # Scores above block_threshold are always blocked
            audit_id = await self._log(
                action, risk_score, Outcome.BLOCKED,
                details={"step": "risk_threshold", "risk_score": risk_score, "violations": violations},
            )
            raise BlockedActionError(
                action_id=action.action_id,
                reason="RISK_SCORE_TOO_HIGH",
                violations=violations + [f"Risk score {risk_score:.1f} exceeds block threshold {self._block_threshold}"],
            )

        if risk_score > self._approval_threshold and not approved_by:
            audit_id = await self._log(
                action, risk_score, Outcome.PENDING,
                details={"step": "risk_threshold", "risk_score": risk_score, "violations": violations},
            )
            raise PendingApprovalException(
                action_id=action.action_id,
                risk_score=risk_score,
                explanation=(
                    f"Action risk score {risk_score:.1f}/10 exceeds the automatic "
                    f"approval threshold of {self._approval_threshold}. "
                    f"Triggered by: {'; '.join(violations) or 'multiple heuristics'}."
                ),
                component="GuardrailMiddleware",
                suggested_alternatives=cmd_alternatives,
            )

        # ── Step 5.5: Action-based approval (Explicitly for deletes) ───────────
        if action.action_type == ActionType.FILE_DELETE and not approved_by:
            risk_score = max(risk_score, self._approval_threshold + 0.5)
            audit_id = await self._log(
                action, risk_score, Outcome.PENDING,
                details={"step": "action_policy", "action_type": "file_delete"},
            )
            raise PendingApprovalException(
                action_id=action.action_id,
                risk_score=risk_score,
                explanation="File deletion requested. Human approval is required for all destructive file operations.",
                component="GuardrailPolicy",
            )

        # ── Step 6: UndoBuffer snapshot (before destructive ops) ──────────────
        if action.is_destructive and action.target_paths and not skip_snapshot:
            try:
                await self.undo.snapshot(
                    paths=action.target_paths,
                    operation_id=action.action_id,
                    agent_id=action.agent_id,
                    action_type=action.action_type,
                )
            except Exception:
                pass  # Snapshot failure should not block the action; just log

        # ── Step 7: Audit log (always) ─────────────────────────────────────────
        audit_id = await self._log(
            action, risk_score, Outcome.SUCCESS,
            details={
                "command": action.command,
                "target_paths": action.target_paths,
                "url": action.url,
                "violations": violations,
                "approved_by": approved_by,
            },
            approved_by=approved_by,
        )

        return SafeActionResult.build(
            action=action,
            risk_score=risk_score,
            audit_id=audit_id,
            approved_by=approved_by,
            sanitized_input=sanitized_text,
        )

    # ------------------------------------------------------------------
    # Convenience: safe code execution via sandbox
    # ------------------------------------------------------------------

    async def safe_execute_code(
        self,
        action: AgentAction,
        allow_network: bool = False,
    ) -> SafeActionResult:
        """
        Evaluate the action then execute code inside the sandbox.

        This is the correct entry point for CodeAgent — it chains the full
        guardrail check + sandbox execution in one call.

        Returns SafeActionResult with sandbox_result populated.
        """
        result = await self.evaluate_action(action)

        if action.code and action.language:
            sandbox_result = await self.sandbox.safe_execute(
                code=action.code,
                language=action.language,
                timeout=action.metadata.get("timeout", 30),
                allow_network=allow_network,
            )
            # Re-build result with sandbox output attached
            return SafeActionResult(
                **result.model_dump(exclude={"sandbox_result"}),
                sandbox_result=sandbox_result,
            )

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _log(
        self,
        action: AgentAction,
        risk_score: float,
        outcome: Outcome,
        details: Optional[dict] = None,
        approved_by: Optional[str] = None,
    ) -> str:
        """Write an audit record and return the audit_id."""
        try:
            return await self.audit.log(
                agent_id=action.agent_id,
                action_type=action.action_type.value,
                risk_score=risk_score,
                outcome=outcome.value,
                details=details,
                approved_by=approved_by,
            )
        except Exception:
            return str(uuid.uuid4())  # Return a dummy ID if audit fails

    @staticmethod
    def _infer_capabilities(action: AgentAction) -> list[str]:
        """Map ActionType to the required capability strings."""
        mapping: dict[ActionType, list[str]] = {
            ActionType.SHELL_COMMAND:  ["shell_exec"],
            ActionType.CODE_EXECUTION: ["code_execute"],
            ActionType.FILE_READ:      ["file_read"],
            ActionType.FILE_WRITE:     ["file_write"],
            ActionType.FILE_DELETE:    ["file_delete"],
            ActionType.NETWORK_REQUEST:["web_browse"],
            ActionType.MEMORY_READ:    ["memory_read"],
            ActionType.MEMORY_WRITE:   ["memory_write"],
            ActionType.PROCESS_SPAWN:  ["process_list"],
            ActionType.CONFIG_CHANGE:  ["config_write"],
        }
        return mapping.get(action.action_type, [])
