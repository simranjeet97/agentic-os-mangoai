"""
guardrails/prompt_injection_defender.py — PromptInjectionDefender

Sanitizes external input before it enters agent context.
Detects and blocks 40+ common jailbreak patterns across multiple categories:
  - Role override attacks
  - Instruction override attacks
  - Token/encoding manipulation
  - Context escaping
  - Many-shot jailbreaks
  - Indirect injection via URLs
"""

from __future__ import annotations

import re
import unicodedata
import urllib.parse
from typing import Optional

from guardrails.models import SanitizedInput

# ---------------------------------------------------------------------------
# Jailbreak pattern registry (category → list of (pattern, description))
# ---------------------------------------------------------------------------

_JAILBREAK_PATTERNS: list[tuple[str, str, str]] = [
    # Category: Role Override
    ("role_override", r"you\s+are\s+now\s+(?:dan|jailbroken|unfiltered|evil|uncensored)", "DAN / jailbroken mode activation"),
    ("role_override", r"act\s+as\s+(?:if\s+you\s+(?:are|were)|an?\s+)(?:different|unrestricted|evil|uncensored|unethical)", "act-as identity override"),
    ("role_override", r"pretend\s+(?:you\s+are|to\s+be)\s+(?:a\s+)?(?:different|unrestricted|evil|uncensored)", "pretend identity override"),
    ("role_override", r"you\s+(?:are|were|must\s+be)\s+(?:now\s+)?(?:a\s+)?(?:different|new|unrestricted)\s+(?:ai|assistant|model|bot)", "identity replacement"),
    ("role_override", r"switch\s+(?:to|into)\s+(?:developer|jailbreak|evil|uncensored|unrestricted)\s+mode", "mode switch attack"),
    ("role_override", r"(?:enable|activate|turn\s+on)\s+(?:developer|jailbreak|god|unrestricted)\s+mode", "mode activation attack"),
    ("role_override", r"(?:from\s+now\s+on|henceforth|starting\s+(?:now|today)).*(?:ignore|disregard|forget)\s+(?:your|all)\s+(?:previous\s+)?(?:training|instructions|guidelines|rules|ethics)", "persistent override attempt"),

    # Category: Instruction Override
    ("instruction_override", r"ignore\s+(?:all\s+)?(?:previous|prior|above|your)\s+instructions", "ignore previous instructions"),
    ("instruction_override", r"disregard\s+(?:all\s+)?(?:previous|prior|your)\s+(?:instructions|training|guidelines)", "disregard instructions"),
    ("instruction_override", r"forget\s+(?:everything|all)\s+(?:you\s+(?:know|were\s+told)|(?:your\s+)?(?:previous\s+)?instructions)", "forget instructions"),
    ("instruction_override", r"override\s+(?:your\s+)?(?:safety|ethical|content)\s+(?:filters?|guidelines?|restrictions?)", "safety filter override"),
    ("instruction_override", r"bypass\s+(?:your\s+)?(?:safety|ethical|content|security)\s+(?:filters?|guidelines?|restrictions?)", "safety filter bypass"),
    ("instruction_override", r"(?:new|updated|corrected)\s+system\s+(?:prompt|instruction|message|directive)", "system prompt injection"),
    ("instruction_override", r"your\s+(?:true|real|actual|hidden)\s+(?:instructions?|purpose|goal|objective)", "hidden instruction elicitation"),

    # Category: Token / Encoding Manipulation
    ("token_manipulation", r"<\s*system\s*>", "embedded <system> tag"),
    ("token_manipulation", r"<\s*/?\s*(?:instructions?|context|prompt|sys)\s*>", "embedded instruction tag"),
    ("token_manipulation", r"\[INST\]|\[/INST\]", "Llama INST token injection"),
    ("token_manipulation", r"<\|(?:im_start|im_end|system|user|assistant)\|>", "ChatML token injection"),
    ("token_manipulation", r"###\s*(?:System|Instruction|Human|Assistant)\s*:", "markdown role injection"),
    ("token_manipulation", r"Human:\s*\n.*\nAssistant:", "conversation injection pattern"),
    ("token_manipulation", r"(?:[A-Za-z0-9+/]{40,}={0,2})", "potential base64 payload (long string)"),  # flagged, not blocked

    # Category: Context Escaping
    ("context_escape", r"</?\s*(?:instructions?|system|context|prompt)\s*>", "context closing tag escape"),
    ("context_escape", r"###\s*END\s*(?:OF)?\s*(?:SYSTEM\s*)?(?:PROMPT|INSTRUCTIONS?)", "end-of-prompt marker injection"),
    ("context_escape", r"---+\s*(?:END|STOP|IGNORE\s+ABOVE)", "separator context escape"),
    ("context_escape", r"\[(?:END|STOP|IGNORE)\s*(?:ABOVE|PREVIOUS)?\]", "bracket context escape"),
    ("context_escape", r"(?:^|\n)>\s*(?:SYSTEM|INSTRUCTION)\s*:", "quoted instruction injection"),

    # Category: Many-Shot / Repetitive Override
    ("many_shot", r"(?:(?:ignore|disregard|bypass|override)\s+(?:your\s+)?(?:safety|instructions?|rules?|ethics)\s*[.!]?\s*){3,}", "many-shot safety override"),
    ("many_shot", r"(?:you\s+(?:must|should|will|shall)\s+(?:not\s+)?(?:follow|obey|comply)\s*){3,}", "repetitive compliance manipulation"),

    # Category: Jailbreak Keywords
    ("jailbreak_keyword", r"\bjailbreak\b", "explicit jailbreak keyword"),
    ("jailbreak_keyword", r"\bdan\b.*(?:mode|version|prompt)", "DAN mode keyword"),
    ("jailbreak_keyword", r"\bunfiltered\b.*(?:mode|version|response|output)", "unfiltered mode keyword"),
    ("jailbreak_keyword", r"\buncensored\b.*(?:mode|version|response|output)", "uncensored mode keyword"),
    ("jailbreak_keyword", r"(?:evil|malicious|harmful|unethical)\s+(?:ai|assistant|mode|version)", "evil AI keyword"),
    ("jailbreak_keyword", r"\bdo\s+anything\s+now\b", "DAN 'do anything now' pattern"),

    # Category: Prompt Leaking
    ("prompt_leak", r"(?:show|reveal|print|output|tell\s+me)\s+(?:your\s+|the\s+)?(?:system\s+)?(?:prompt|instructions?|context)", "prompt leak attempt"),
    ("prompt_leak", r"what\s+(?:are|were)\s+(?:your\s+)?(?:original\s+)?(?:instructions?|system\s+prompt|training)", "instruction leak attempt"),
    ("prompt_leak", r"repeat\s+(?:back\s+)?(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)\s+verbatim", "verbatim prompt repeat request"),

    # Category: Hypothetical Framing
    ("hypothetical", r"hypothetically\s+(?:speaking\s+)?(?:if\s+you\s+(?:had\s+no|could\s+ignore|were\s+without)\s+(?:restrictions?|safety|filters?))", "hypothetical framing attack"),
    ("hypothetical", r"(?:in\s+a\s+)?fictional\s+(?:world|scenario|story|setting)\s+(?:where|in\s+which)\s+(?:ai\s+)?(?:have\s+no\s+)?(?:safety|ethics|restrictions?)", "fictional framing attack"),
    ("hypothetical", r"(?:as\s+an?\s+)?(?:exercise|thought\s+experiment|roleplay)\s+.*(?:ignore|bypass|pretend)\s+(?:safety|ethics|rules)", "roleplay framing attack"),
]

# Patterns that cause the input to be BLOCKED outright (not just flagged)
_BLOCKING_CATEGORIES = {
    "role_override",
    "instruction_override",
    "context_escape",
    "many_shot",
    "jailbreak_keyword",
}

# Control characters to strip (except common whitespace)
_CONTROL_CHAR_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\x80-\x9f]"  # C0/C1 controls
)

# Embedded URL pattern (for indirect injection detection)
_URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)


class PromptInjectionDefender:
    """
    Multi-layer prompt injection and jailbreak defense.

    Pipeline:
    1. Strip null bytes and control characters
    2. Normalize Unicode (NFKC)
    3. Remove embedded <system>/<INST> tags
    4. Scan 40+ jailbreak patterns across 7 categories
    5. Flag suspicious URLs (potential indirect injection)
    6. Return SanitizedInput (possibly modified, threats listed)

    Usage:
        defender = PromptInjectionDefender()
        result = await defender.sanitize("Ignore all previous instructions and ...")
        if not result.is_safe:
            raise BlockedActionError(...)
        use(result.sanitized)
    """

    def __init__(self, strict_mode: bool = True) -> None:
        """
        Args:
            strict_mode: If True, any blocking-category match is_safe=False.
                         If False, all threats are flagged but is_safe is only
                         False for multi-category hits.
        """
        self._strict = strict_mode
        # Pre-compile all patterns
        self._compiled: list[tuple[str, re.Pattern, str]] = [
            (cat, re.compile(pat, re.IGNORECASE | re.MULTILINE | re.DOTALL), desc)
            for cat, pat, desc in _JAILBREAK_PATTERNS
        ]

    async def sanitize(self, text: str) -> SanitizedInput:
        """
        Sanitize and analyse `text` for prompt injection / jailbreak attempts.

        Returns:
            SanitizedInput with sanitized text, modification flag, and threats.
        """
        original = text
        threats: list[str] = []

        # Step 1: Strip control characters (keep \t \n \r)
        cleaned = _CONTROL_CHAR_RE.sub("", text)

        # Step 2: Unicode normalization (NFKC collapses lookalike chars)
        cleaned = unicodedata.normalize("NFKC", cleaned)

        # Step 3: Remove embedded template tags
        cleaned, tag_threats = self._strip_template_tags(cleaned)
        threats.extend(tag_threats)

        # Step 4: Scan jailbreak patterns
        matched_categories: set[str] = set()
        pattern_threats: list[str] = []

        for category, compiled, description in self._compiled:
            if compiled.search(cleaned):
                pattern_threats.append(f"[{category}] {description}")
                matched_categories.add(category)

        threats.extend(pattern_threats)

        # Step 5: Indirect injection via URLs
        urls = _URL_RE.findall(cleaned)
        for url in urls:
            if self._is_suspicious_url(url):
                threats.append(f"[indirect_injection] Suspicious URL detected: {url[:80]}")

        # Determine safety
        blocking_hits = matched_categories & _BLOCKING_CATEGORIES
        is_safe: bool
        if self._strict:
            is_safe = len(blocking_hits) == 0
        else:
            is_safe = len(blocking_hits) < 2  # require 2+ blocking categories

        was_modified = cleaned != original

        return SanitizedInput(
            original=original,
            sanitized=cleaned,
            was_modified=was_modified,
            threats_detected=threats,
            is_safe=is_safe,
        )

    def quick_check(self, text: str) -> bool:
        """
        Synchronous fast-path check: returns True if text appears safe.
        Does not perform Unicode normalization — use for hot-path pre-screening.
        """
        for category, compiled, _ in self._compiled:
            if category in _BLOCKING_CATEGORIES and compiled.search(text):
                return False
        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_template_tags(text: str) -> tuple[str, list[str]]:
        """Remove common LLM template injection tags and report what was stripped."""
        threats: list[str] = []
        tags_removed = []

        # Remove <system>…</system> blocks
        if re.search(r"<\s*system\s*>", text, re.IGNORECASE):
            text = re.sub(r"<\s*system\s*>.*?<\s*/\s*system\s*>", "", text, flags=re.IGNORECASE | re.DOTALL)
            tags_removed.append("<system> block")

        # Remove [INST]…[/INST] tokens
        if re.search(r"\[INST\]", text, re.IGNORECASE):
            text = re.sub(r"\[/?INST\]", "", text, flags=re.IGNORECASE)
            tags_removed.append("[INST] tokens")

        # Remove <|im_start|> / <|im_end|> tokens
        if re.search(r"<\|im_(?:start|end)\|>", text):
            text = re.sub(r"<\|im_(?:start|end)\|>", "", text)
            tags_removed.append("<|im_start|>/<|im_end|> tokens")

        # Remove embedded instruction tags
        for tag in ("instructions", "context", "prompt", "sys"):
            if re.search(rf"<\s*/?{tag}\s*>", text, re.IGNORECASE):
                text = re.sub(rf"<\s*/?{tag}\s*>", "", text, flags=re.IGNORECASE)
                tags_removed.append(f"<{tag}> tags")

        if tags_removed:
            threats.append(f"[template_tag] Stripped: {', '.join(tags_removed)}")

        return text, threats

    @staticmethod
    def _is_suspicious_url(url: str) -> bool:
        """
        Heuristically detect URLs that might carry injection payloads.
        Flags: overly long query strings, data: URIs, encoded payloads.
        """
        try:
            parsed = urllib.parse.urlparse(url)
            # data: URI — can embed executable content
            if parsed.scheme == "data":
                return True
            # Suspiciously long query string (potential injection via redirect)
            if len(parsed.query) > 200:
                return True
            # URL-encoded brackets/tags in path/query (common for web injection)
            decoded = urllib.parse.unquote(url).lower()
            if any(kw in decoded for kw in ("<system>", "[inst]", "ignore all", "jailbreak")):
                return True
        except Exception:
            pass
        return False
