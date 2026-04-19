"""
guardrails/command_classifier.py — CommandClassifier

Scores any shell command 0-10 for risk using rule-based heuristics and an
optional LLM fallback. Commands matching the catastrophic blocklist receive
an instant score of 10 and are unconditionally blocked.

Risk score bands:
  0-2  → SAFE    (auto-approved)
  3-4  → LOW     (auto-approved with audit)
  5-6  → MEDIUM  (auto-approved, extra monitoring)
  7-8  → HIGH    (PendingApprovalException raised by middleware)
  9-10 → CRITICAL / BLOCKED (BlockedActionError raised by middleware)
"""

from __future__ import annotations

import json
import re
from typing import Optional


from guardrails.models import CommandRiskResult, RiskLevel

# ---------------------------------------------------------------------------
# Blocklist — catastrophic commands, INSTANT score=10
# ---------------------------------------------------------------------------

_BLOCKLIST: list[tuple[str, str]] = [
    # Filesystem annihilation
    (r"rm\s+-[^\s]*f[^\s]*\s+/\s*$", "rm -rf / (root deletion)"),
    (r"rm\s+-[^\s]*f[^\s]*\s+/\s*\*", "rm -rf /* (root glob deletion)"),
    (r"rm\s+-[^\s]*f[^\s]*\s+~\s*$", "rm -rf ~ (home deletion)"),
    (r"rm\s+-[^\s]*f[^\s]*\s+~/\s*\*", "rm -rf ~/* (home glob deletion)"),
    (r"rm\s+-[^\s]*f[^\s]*\s+--no-preserve-root", "rm --no-preserve-root"),
    # Fork bomb
    (r":\(\)\s*\{", "Fork bomb pattern :(){ :|:& };:"),
    (r"\(\)\s*\{\s*[^}]*\|\s*[^}]*&", "Fork bomb variant"),
    # Disk overwrite
    (r"dd\s+if=/dev/(?:zero|random|urandom)\s+of=/dev/(?:sd|hd|nvme|vd)", "dd disk overwrite"),
    (r"dd\s+if=/dev/\w+\s+of=/dev/\w+", "dd device-to-device copy"),
    (r">\s*/dev/sd[a-z]", "direct write to block device"),
    (r">\s*/dev/hd[a-z]", "direct write to block device"),
    # Filesystem format
    (r"mkfs\.\w+\s+/dev/", "mkfs format command"),
    (r"mkswap\s+/dev/", "mkswap on raw device"),
    (r"fdisk\s+/dev/", "fdisk partition tool"),
    (r"parted\s+/dev/", "parted partition tool"),
    # Remote execution pipelines (eval from internet)
    (r"curl\s+[^|]+\|\s*(?:bash|sh|zsh|fish|python|ruby|perl)", "curl | shell pipe"),
    (r"wget\s+-[^\s]*O\s*-[^\s]*\s+[^|]+\|\s*(?:bash|sh|zsh|fish)", "wget | shell pipe"),
    (r"fetch\s+[^|]+\|\s*(?:bash|sh|zsh|fish)", "fetch | shell pipe"),
    # Kernel / boot record wipes
    (r"dd\s+if=/dev/zero\s+of=/boot", "dd wipe boot partition"),
    (r"shred\s+/dev/", "shred raw device"),
    # chmod 777 on system paths
    (r"chmod\s+-R\s+777\s+/(?:etc|usr|bin|sbin|lib)", "chmod 777 on system path"),
    # Dangerous cron / system file manipulation
    (r"echo\s+.+\s*>+\s*/etc/(?:passwd|shadow|sudoers|hosts|crontab)", "overwrite critical system file"),
    (r"truncate\s+.*\s*/etc/(?:passwd|shadow|sudoers)", "truncate critical system file"),
]

# ---------------------------------------------------------------------------
# Rule-based scoring heuristics (additive)
# ---------------------------------------------------------------------------

_HEURISTICS: list[tuple[str, float, str]] = [
    # sudo elevation
    (r"\bsudo\b", 2.5, "sudo elevation"),
    # rm with force flags
    (r"\brm\b.*-[^\s]*f", 2.0, "rm with force flag"),
    # rm on important directories
    (r"\brm\b.*/(?:home|root|var|etc|usr|bin|boot)", 3.0, "rm targeting system directory"),
    # Wildcard removal
    (r"\brm\b.*\*", 2.0, "rm with wildcard"),
    # Redirection to important paths
    (r">\s*/(?:etc|usr|bin|sbin|boot|lib|root)", 3.5, "redirect to system path"),
    # Shell injection via eval/exec
    (r"\beval\b", 2.5, "use of eval"),
    (r"\bexec\b", 1.5, "use of exec"),
    # Network download
    (r"\b(?:curl|wget|fetch)\b", 1.0, "network download tool"),
    # Python/bash inline execution
    (r"\bpython3?\b.*-c\b", 1.5, "python -c inline exec"),
    (r"\bbash\b.*-c\b", 1.5, "bash -c inline exec"),
    # Dangerous shell sequences
    (r";\s*rm\b", 2.0, "rm after command chain"),
    (r"&&\s*rm\b", 2.0, "rm after && chain"),
    # Process kill-all
    (r"\bkillall\b", 1.5, "killall"),
    (r"\bkill\s+-9\b", 1.0, "kill -9 SIGKILL"),
    # iptables / network rule flush
    (r"\biptables\b.*-F\b", 2.5, "iptables flush"),
    (r"\bufw\b.*(?:disable|reset)\b", 2.0, "ufw disable/reset"),
    # Cron modifications
    (r"\bcrontab\b.*-[er]?\b", 1.5, "crontab edit/remove"),
    # su / privilege switch
    (r"\bsu\b\s+-?\s*(?:root|\s*$)", 2.0, "su to root"),
    # sshd / auth config changes
    (r"\bsshd_config\b", 1.5, "sshd config modification"),
    # Environment manipulation
    (r"\bLD_PRELOAD\b", 3.0, "LD_PRELOAD injection"),
    (r"\bLD_LIBRARY_PATH\b", 1.5, "LD_LIBRARY_PATH manipulation"),
    # systemctl disable / mask
    (r"\bsystemctl\b.*(?:disable|mask|stop|kill)\b", 1.5, "systemctl disable/stop"),
    # Package manager abuse
    (r"\bapt\b.*(?:remove|purge)\b.*-y\b", 1.5, "apt purge -y"),
    (r"\byum\b.*remove\b.*-y\b", 1.5, "yum remove -y"),
    # chmod on passwd/shadow
    (r"chmod\b.*/etc/shadow\b", 3.0, "chmod on /etc/shadow"),
    # write to /proc or /sys
    (r">\s*/proc/", 2.5, "write to /proc"),
    (r">\s*/sys/", 2.5, "write to /sys"),
    # General destructive commands (force approval)
    (r"\brm\b", 6.5, "general file removal"),
    (r"\b(?:mkfs|mke2fs|fdisk|parted)\b", 7.0, "disk partition/format tool"),
]

_SAFE_COMMANDS = {
    "ls", "pwd", "echo", "cat", "head", "tail", "grep", "find", "wc",
    "date", "whoami", "uname", "df", "du", "ps", "top", "history",
    "man", "help", "which", "type", "alias", "env", "set", "export",
    "cd", "mkdir", "touch", "cp", "mv", "ln", "chmod", "chown",
    "git", "pip", "python", "python3", "node", "npm", "pytest",
}


def _extract_base_command(cmd: str) -> str:
    """Extract the first word of a command (ignoring leading whitespace/pipes)."""
    stripped = cmd.strip().lstrip("|&; ")
    return stripped.split()[0] if stripped.split() else ""


class CommandClassifier:
    """
    Scores shell commands for risk using rule heuristics + optional LLM.

    Usage:
        classifier = CommandClassifier()
        result = await classifier.classify("rm -rf /")
        print(result.risk_score, result.is_blocked)
    """

    # Threshold above which LLM is invoked (saves latency for clearly safe cmds)
    LLM_INVOKE_THRESHOLD = 3.0
    MAX_RULE_SCORE = 10.0

    def __init__(self, llm_model: Optional[str] = None) -> None:
        self._llm_model = llm_model  # e.g. "llama3.3" or None to skip LLM

    async def classify(self, command: str) -> CommandRiskResult:
        """Return a CommandRiskResult for the given shell command string."""
        if not command or not command.strip():
            return self._safe_result(command or "")

        # 1. Blocklist check (O(1) per pattern, instant block)
        blocked_reason = self._check_blocklist(command)
        if blocked_reason:
            return CommandRiskResult(
                command=command,
                risk_score=10.0,
                risk_level=RiskLevel.CRITICAL,
                is_blocked=True,
                rule_score=10.0,
                llm_score=None,
                explanation=f"BLOCKLIST HIT: {blocked_reason}",
                matched_rules=[blocked_reason],
                suggested_alternatives=[
                    "Use a more targeted command with explicit paths",
                    "Consider using a file manager for this operation",
                    "Break the operation into smaller, reversible steps",
                ],
            )

        # 2. Rule-based heuristic scoring
        rule_score, matched_rules = self._compute_rule_score(command)
        rule_score = min(rule_score, self.MAX_RULE_SCORE)

        # 3. LLM scoring (only when rule score is borderline or above threshold)
        llm_score: Optional[float] = None
        if self._llm_model and rule_score >= self.LLM_INVOKE_THRESHOLD:
            llm_score = await self._llm_score(command)

        final_score = max(rule_score, llm_score or 0.0)
        risk_level = self._score_to_level(final_score)

        explanation = self._build_explanation(final_score, matched_rules, llm_score)
        alternatives = self._suggest_alternatives(command, final_score)

        return CommandRiskResult(
            command=command,
            risk_score=final_score,
            risk_level=risk_level,
            is_blocked=False,
            rule_score=rule_score,
            llm_score=llm_score,
            explanation=explanation,
            matched_rules=matched_rules,
            suggested_alternatives=alternatives,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_blocklist(self, command: str) -> Optional[str]:
        """Return blocklist description if matched, else None."""
        for pattern, description in _BLOCKLIST:
            if re.search(pattern, command, re.IGNORECASE | re.MULTILINE):
                return description
        return None

    def _compute_rule_score(self, command: str) -> tuple[float, list[str]]:
        """Apply heuristics and return (accumulated_score, matched_rule_names)."""
        score = 0.0
        matched: list[str] = []

        # Bonus for safe base commands (reduce score)
        base = _extract_base_command(command)
        if base in _SAFE_COMMANDS and len(command.split()) == 1:
            return 0.0, []

        for pattern, weight, label in _HEURISTICS:
            if re.search(pattern, command, re.IGNORECASE):
                score += weight
                matched.append(f"{label} (+{weight})")

        return score, matched

    async def _llm_score(self, command: str) -> Optional[float]:
        """
        Ask the local Ollama LLM to score the command.

        Returns a float 0-10 or None if LLM is unavailable.
        The LLM response is expected in JSON: {"score": <float>, "reason": "<str>"}
        """
        try:
            import httpx  # already in requirements

            prompt = (
                "You are a security expert evaluating shell command risk.\n"
                "Score the following shell command from 0 (completely safe) to 10 (catastrophic).\n"
                "Consider: data loss, privilege escalation, system damage, network exfiltration.\n"
                f"Command: {command!r}\n"
                "Respond ONLY with valid JSON: {\"score\": <0-10 float>, \"reason\": \"<brief reason>\"}"
            )

            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self._llm_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0, "num_predict": 128},
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    text = data.get("response", "")
                    # Extract JSON from response
                    match = re.search(r'\{[^}]+\}', text)
                    if match:
                        parsed = json.loads(match.group())
                        score = float(parsed.get("score", 0))
                        return max(0.0, min(10.0, score))
        except Exception:
            # LLM unavailable — fall back to rule score only
            pass
        return None

    @staticmethod
    def _score_to_level(score: float) -> RiskLevel:
        if score <= 2:
            return RiskLevel.SAFE
        if score <= 4:
            return RiskLevel.LOW
        if score <= 6:
            return RiskLevel.MEDIUM
        if score <= 8:
            return RiskLevel.HIGH
        return RiskLevel.CRITICAL

    @staticmethod
    def _build_explanation(
        score: float, matched_rules: list[str], llm_score: Optional[float]
    ) -> str:
        if not matched_rules and score == 0:
            return "No risk indicators found. Command appears safe."
        rules_text = "; ".join(matched_rules) if matched_rules else "none"
        llm_text = f" LLM score={llm_score:.1f}." if llm_score is not None else ""
        return (
            f"Risk score {score:.1f}/10. Rule triggers: {rules_text}.{llm_text}"
        )

    @staticmethod
    def _suggest_alternatives(command: str, score: float) -> list[str]:
        alts: list[str] = []
        if score <= 3:
            return alts
        if re.search(r"\brm\b", command):
            alts.append("Use 'trash-cli' or move files to /tmp instead of deleting")
            alts.append("Specify exact file paths rather than wildcards")
        if re.search(r"\bsudo\b", command):
            alts.append("Run without sudo if possible; use scoped capabilities")
        if re.search(r"\beval\b|\bexec\b", command):
            alts.append("Avoid eval/exec; use explicit function calls")
        if re.search(r"curl|wget", command) and re.search(r"\|.*(?:sh|bash)", command):
            alts.append("Download first, inspect, then execute separately")
        return alts

    @staticmethod
    def _safe_result(command: str) -> CommandRiskResult:
        return CommandRiskResult(
            command=command,
            risk_score=0.0,
            risk_level=RiskLevel.SAFE,
            is_blocked=False,
            rule_score=0.0,
            explanation="Empty or whitespace command — no risk.",
            matched_rules=[],
        )
