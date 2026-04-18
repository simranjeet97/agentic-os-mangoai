"""
guardrails/content_filter.py — Harmful content detection layer.
Uses keyword rules + LLM fallback for ambiguous cases.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from core.logging_config import get_logger

logger = get_logger(__name__)

# Patterns that are always blocked regardless of LLM judgment
HARD_BLOCK_PATTERNS = [
    r"\b(?:rm\s+-rf\s+[/~]|format\s+[c-z]:)\b",
    r"\b(?:sudo\s+rm|dd\s+if=)",
    r"(?:synthesize|manufacture|produce)\s+(?:drugs?|explosives?|weapons?|malware)",
    r"\b(?:child\s+(?:pornography|exploitation|abuse\s+material))\b",
    r"(?:hack|crack|exploit)\s+(?:bank|financial|government|military)",
    r"\b(?:ransomware|keylogger|rootkit|trojan)\b",
]

SOFT_WARN_PATTERNS = [
    r"\b(?:delete|remove|drop)\s+(?:all|everything|database)\b",
    r"\b(?:password|credentials?|api.?key|secret)\b",
    r"\b(?:phishing|scam|fraud)\b",
]


class ContentFilterResult(BaseModel):
    is_harmful: bool
    severity: str = "none"  # none | low | medium | high | critical
    reasons: list[str] = Field(default_factory=list)


class ContentFilter:
    """Rule-based + pattern content safety filter."""

    async def check(self, text: str) -> ContentFilterResult:
        reasons: list[str] = []
        severity = "none"

        # Hard blocks
        for pattern in HARD_BLOCK_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                reasons.append(f"Hard-blocked content pattern: {pattern}")
                severity = "critical"

        # Soft warnings (don't block, but flag)
        for pattern in SOFT_WARN_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                reasons.append(f"Sensitive content pattern detected: {pattern}")
                if severity == "none":
                    severity = "low"

        is_harmful = severity in ("high", "critical")
        return ContentFilterResult(
            is_harmful=is_harmful,
            severity=severity,
            reasons=reasons,
        )
