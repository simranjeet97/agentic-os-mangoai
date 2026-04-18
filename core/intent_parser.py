"""
core/intent_parser.py — Classifies user intent and extracts routing metadata.

IntentTypes:
  SINGLE_AGENT_TASK  — one agent can handle it end-to-end
  MULTI_AGENT_TASK   — requires coordination across multiple agents
  CLARIFICATION_NEEDED — ambiguous, must ask the user
  SYSTEM_QUERY       — introspection/status request (no LLM needed)
  CONVERSATION       — casual chat, no task execution

Extraction:
  intent, required_agents[], urgency (1-5), estimated_complexity (1-10)
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from core.logging_config import get_logger
from core.state import AgentRole

logger = get_logger(__name__)


# ── Intent types ─────────────────────────────────────────────────────────────


class IntentType(str, Enum):
    SINGLE_AGENT_TASK = "SINGLE_AGENT_TASK"
    MULTI_AGENT_TASK = "MULTI_AGENT_TASK"
    CLARIFICATION_NEEDED = "CLARIFICATION_NEEDED"
    SYSTEM_QUERY = "SYSTEM_QUERY"
    CONVERSATION = "CONVERSATION"


# ── Result model ──────────────────────────────────────────────────────────────


class ParsedIntent(BaseModel):
    """Structured output of IntentParser."""

    raw_input: str
    intent_type: IntentType
    intent: str                                   # one-line summary
    required_agents: list[str] = Field(default_factory=list)  # AgentRole values
    urgency: int = Field(default=3, ge=1, le=5)   # 1=low … 5=critical
    estimated_complexity: int = Field(default=3, ge=1, le=10)
    clarification_question: Optional[str] = None  # set when CLARIFICATION_NEEDED
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def needs_multiple_agents(self) -> bool:
        return len(self.required_agents) > 1

    @property
    def is_executable(self) -> bool:
        return self.intent_type in (
            IntentType.SINGLE_AGENT_TASK,
            IntentType.MULTI_AGENT_TASK,
        )


# ── Heuristic fast-path patterns ─────────────────────────────────────────────

_SYSTEM_QUERY_PATTERNS = re.compile(
    r"\b(status|health|ping|list agents?|show tasks?|what are you|who are you"
    r"|what can you do|help|version|uptime|running tasks?)\b",
    re.IGNORECASE,
)

_CONVERSATION_PATTERNS = re.compile(
    r"^(hi|hello|hey|thanks|thank you|bye|goodbye|how are you"
    r"|what's up|good morning|good evening|nice to meet you)[.!?]?$",
    re.IGNORECASE,
)

_MULTI_AGENT_SIGNALS = re.compile(
    r"\b(then|after that|next|followed by|also|and then|pipeline|workflow"
    r"|step by step|multiple|several|chain)\b",
    re.IGNORECASE,
)

# Map keywords → AgentRole
_AGENT_KEYWORD_MAP: dict[str, str] = {
    "file": AgentRole.FILE.value,
    "read": AgentRole.FILE.value,
    "write": AgentRole.FILE.value,
    "create file": AgentRole.FILE.value,
    "delete file": AgentRole.FILE.value,
    "search": AgentRole.WEB.value,
    "browse": AgentRole.WEB.value,
    "web": AgentRole.WEB.value,
    "internet": AgentRole.WEB.value,
    "google": AgentRole.WEB.value,
    "run": AgentRole.SYSTEM.value,
    "execute": AgentRole.EXECUTOR.value,
    "shell": AgentRole.SYSTEM.value,
    "command": AgentRole.SYSTEM.value,
    "code": AgentRole.CODE.value,
    "python": AgentRole.CODE.value,
    "script": AgentRole.CODE.value,
    "program": AgentRole.CODE.value,
    "plan": AgentRole.PLANNER.value,
    "schedule": AgentRole.PLANNER.value,
    "organise": AgentRole.PLANNER.value,
    "organize": AgentRole.PLANNER.value,
}


def _heuristic_agents(text: str) -> list[str]:
    """Quickly infer required agents from keywords — no LLM needed."""
    lower = text.lower()
    found: set[str] = set()
    for kw, role in _AGENT_KEYWORD_MAP.items():
        if kw in lower:
            found.add(role)
    return list(found) if found else [AgentRole.EXECUTOR.value]


def _heuristic_urgency(text: str) -> int:
    if re.search(r"\b(asap|urgent|immediately|now|critical|emergency)\b", text, re.I):
        return 5
    if re.search(r"\b(soon|quickly|fast|hurry)\b", text, re.I):
        return 4
    if re.search(r"\b(when you can|no rush|low priority|eventually)\b", text, re.I):
        return 1
    return 3


def _heuristic_complexity(text: str, agent_count: int) -> int:
    words = len(text.split())
    base = min(words // 10 + 1, 5)
    return min(base + agent_count - 1, 10)


# ── LLM-powered parsing ───────────────────────────────────────────────────────

_LLM_SYSTEM_PROMPT = """\
You are an intent classifier inside an Agentic AI OS.
Analyse the user's input and return ONLY valid JSON (no markdown, no explanation).

Schema:
{
  "intent_type": "SINGLE_AGENT_TASK | MULTI_AGENT_TASK | CLARIFICATION_NEEDED | SYSTEM_QUERY | CONVERSATION",
  "intent": "<one-sentence summary of what the user wants>",
  "required_agents": ["<agent-role>"],   // subset of: planner, executor, file, web, system, code
  "urgency": <1-5>,
  "estimated_complexity": <1-10>,
  "clarification_question": "<question to ask if CLARIFICATION_NEEDED, else null>"
}

Rules:
- Use MULTI_AGENT_TASK only when two or more distinct skills are genuinely required.
- Use CLARIFICATION_NEEDED when the request is ambiguous and cannot proceed safely.
- urgency 5 = critical/blocking; 1 = whenever.
- estimated_complexity 1 = trivial; 10 = research-grade multi-day project.
"""


async def _llm_parse(text: str) -> Optional[dict[str, Any]]:
    """Call the LLM to produce a structured intent dict. Returns None on failure."""
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage, SystemMessage
        import os

        llm = ChatOllama(
            model=os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.2:3b"),
            temperature=0.0,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
        response = await llm.ainvoke([
            SystemMessage(content=_LLM_SYSTEM_PROMPT),
            HumanMessage(content=f"User input: {text}"),
        ])
        raw = response.content.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        return json.loads(raw)
    except Exception as exc:
        logger.warning("LLM intent parse failed", error=str(exc))
        return None


# ── IntentParser ──────────────────────────────────────────────────────────────


class IntentParser:
    """
    Two-tier intent classification:
      1. Fast heuristic pass (instant, no LLM).
      2. LLM pass for ambiguous inputs (async, best-effort).

    Usage:
        parser = IntentParser()
        parsed = await parser.parse("Search the web for LangGraph and write a summary to report.md")
    """

    def __init__(self, use_llm: bool = True) -> None:
        self.use_llm = use_llm

    async def parse(self, text: str) -> ParsedIntent:
        """Parse user input and return a rich ParsedIntent."""
        text = text.strip()
        logger.debug("Parsing intent", input_preview=text[:80])

        # ── Fast heuristics ───────────────────────────────────────────────────
        if _CONVERSATION_PATTERNS.match(text):
            return ParsedIntent(
                raw_input=text,
                intent_type=IntentType.CONVERSATION,
                intent="Casual greeting / small-talk",
                required_agents=[],
                urgency=1,
                estimated_complexity=1,
            )

        if _SYSTEM_QUERY_PATTERNS.search(text):
            return ParsedIntent(
                raw_input=text,
                intent_type=IntentType.SYSTEM_QUERY,
                intent="System status / introspection query",
                required_agents=[],
                urgency=2,
                estimated_complexity=1,
            )

        # ── LLM pass ─────────────────────────────────────────────────────────
        llm_result: Optional[dict[str, Any]] = None
        if self.use_llm:
            llm_result = await _llm_parse(text)

        if llm_result:
            try:
                return ParsedIntent(
                    raw_input=text,
                    intent_type=IntentType(llm_result.get("intent_type", "SINGLE_AGENT_TASK")),
                    intent=llm_result.get("intent", text[:120]),
                    required_agents=llm_result.get("required_agents", [AgentRole.EXECUTOR.value]),
                    urgency=int(llm_result.get("urgency", 3)),
                    estimated_complexity=int(llm_result.get("estimated_complexity", 3)),
                    clarification_question=llm_result.get("clarification_question"),
                )
            except Exception as exc:
                logger.warning("LLM result validation failed, falling back to heuristics", error=str(exc))

        # ── Pure heuristic fallback ────────────────────────────────────────────
        agents = _heuristic_agents(text)
        is_multi = len(agents) > 1 or bool(_MULTI_AGENT_SIGNALS.search(text))
        urgency = _heuristic_urgency(text)
        complexity = _heuristic_complexity(text, len(agents))

        return ParsedIntent(
            raw_input=text,
            intent_type=IntentType.MULTI_AGENT_TASK if is_multi else IntentType.SINGLE_AGENT_TASK,
            intent=text[:120],
            required_agents=agents,
            urgency=urgency,
            estimated_complexity=complexity,
        )
