"""
guardrails/guardian.py — Central guardrail enforcement engine.
Evaluates safety, permissions, and policy compliance before any agent action.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from core.logging_config import get_logger
from guardrails.content_filter import ContentFilter
from guardrails.permission_engine import PermissionEngine

logger = get_logger(__name__)


class GuardrailResult(BaseModel):
    """Structured output from guardrail evaluation."""

    passed: bool
    risk_level: str = "low"    # low | medium | high | critical
    violations: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    blocked_reason: Optional[str] = None


class GuardianEngine:
    """
    The central safety enforcement layer.
    Combines content filtering, permission checks, and policy evaluation.
    """

    # Risk escalation thresholds
    BLOCK_RISK_LEVEL = "high"

    def __init__(self) -> None:
        self.content_filter = ContentFilter()
        self.permission_engine = PermissionEngine()
        logger.info("GuardianEngine initialized")

    async def evaluate(
        self,
        user_input: str,
        user_id: str,
        requested_capabilities: list[str],
        context: Optional[dict[str, Any]] = None,
    ) -> GuardrailResult:
        """
        Run all guardrail checks on a user request.
        Returns a GuardrailResult indicating pass/fail + details.
        """
        violations: list[str] = []
        recommendations: list[str] = []
        risk_level = "low"

        # ── 1. Content Filter ──────────────────────────────────────────────
        content_result = await self.content_filter.check(user_input)
        if content_result.is_harmful:
            violations.extend(content_result.reasons)
            risk_level = self._escalate_risk(risk_level, "critical")
            logger.warning(
                "Content filter triggered",
                user_id=user_id,
                reasons=content_result.reasons,
                audit=True,
            )

        # ── 2. Permission Check ────────────────────────────────────────────
        if requested_capabilities:
            perm_result = await self.permission_engine.check(
                user_id=user_id,
                capabilities=requested_capabilities,
            )
            if not perm_result.authorized:
                violations.extend(perm_result.denied_capabilities)
                risk_level = self._escalate_risk(risk_level, "high")
                recommendations.append(
                    "Request elevated permissions from administrator."
                )

        # ── 3. Input Length Guard ──────────────────────────────────────────
        if len(user_input) > 50_000:
            violations.append("Input exceeds maximum allowed length (50k chars)")
            risk_level = self._escalate_risk(risk_level, "medium")

        # ── 4. Prompt Injection Detection ─────────────────────────────────
        injection_patterns = [
            r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions",
            r"you\s+are\s+(?:now|a)\s+(?:different|new)\s+(?:ai|assistant|model)",
            r"jailbreak",
            r"act\s+as\s+(?:if\s+you\s+(?:are|were)|an?)\s+",
            r"pretend\s+(?:you\s+are|to\s+be)",
            r"<\s*system\s*>",
            r"\[INST\]|\[\/INST\]",
        ]
        for pattern in injection_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                violations.append(f"Potential prompt injection detected: pattern '{pattern}'")
                risk_level = self._escalate_risk(risk_level, "high")
                break

        passed = risk_level not in ("high", "critical") or len(violations) == 0

        result = GuardrailResult(
            passed=passed,
            risk_level=risk_level,
            violations=violations,
            recommendations=recommendations,
            blocked_reason=violations[0] if violations and not passed else None,
        )

        logger.bind(audit=True).info(
            "Guardrail evaluation complete",
            user_id=user_id,
            passed=passed,
            risk_level=risk_level,
            violation_count=len(violations),
        )

        return result

    @staticmethod
    def _escalate_risk(current: str, new: str) -> str:
        """Return the higher of two risk levels."""
        order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        return new if order.get(new, 0) > order.get(current, 0) else current
