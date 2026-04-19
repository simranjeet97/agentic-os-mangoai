"""
agents/router.py — AgentRouter

Selects the best agent for a given task using multi-stage classification:

Stage 1: Rule-based keyword matching (instant, zero-cost)
Stage 2: Capability-based filtering (structural)
Stage 3: LLM classification (when ambiguous)

The router consults the AgentRegistry to resolve agent names to instances.
Falls back to the ExecutorAgent when no match is found.

Usage:
    router = AgentRouter()
    agent = await router.route(task_description)
    result = await agent.run(task)
"""

from __future__ import annotations

import re
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import BaseAgent
from core.logging_config import get_logger

logger = get_logger("agent.router")


# ── Keyword → agent role mapping ──────────────────────────────────────────────

KEYWORD_RULES: list[tuple[list[str], str]] = [
    # PlannerAgent
    (
        ["plan", "decompose", "strategy", "roadmap", "subtasks", "breakdown",
         "orchestrate", "schedule", "prioritize", "sequence"],
        "planner",
    ),
    # ExecutorAgent
    (
        ["code", "python", "script", "program", "execute", "run", "refactor", "debug", "test", "generate code",
         "subprocess", "execute command", "run this", "execute this"],
        "executor",
    ),
    # FileAgent
    (
        ["file", "folder", "directory", "rename", "move", "copy", "delete",
         "search file", "find file", "summarize file", "read file", "write file",
         "organize", "convert file", "batch rename", "pdf", "docx"],
        "file",
    ),
    # WebAgent
    (
        ["browse", "url", "website", "webpage", "http", "https", "scrape",
         "crawl", "extract data", "form", "screenshot", "search google",
         "search duckduckgo", "duckduckgo", "web search", "search web",
         "research", "google", "download page", "click", "navigate to", "latest", "news"],
        "web",
    ),
    # SystemAgent
    (
        ["cpu", "ram", "memory", "disk", "process", "monitor", "system",
         "uptime", "service", "restart service", "kill process", "clean temp",
         "network stats", "anomaly", "alert", "performance"],
        "system",
    ),
    # CodeAgent
    (
        ["code", "debug", "refactor", "write function", "fix bug", "lint",
         "test", "unit test", "git", "commit", "push", "review", "explain code",
         "generate code", "python", "javascript", "typescript", "golang",
         "class", "function", "syntax", "import"],
        "code",
    ),
]

LLM_ROUTE_PROMPT = """\
You are an AI agent router for a multi-agent operating system.
Given a task description, choose the SINGLE best agent to handle it.

Available agents:
- planner   : Decompose goals, create plans, orchestrate multi-step tasks
- executor  : Run code/scripts/shell commands in a sandbox
- file      : File operations — read, write, search, summarize, organize, convert
- web       : Browse URLs, scrape data, fill forms, search Google/DuckDuckGo
- system    : Monitor CPU/RAM/disk/network, manage processes and services
- code      : Write, debug, refactor, explain, lint, test code; git integration

Rules:
- Respond with EXACTLY ONE word: the agent name (planner/executor/file/web/system/code)
- No explanation, no punctuation
- If genuinely ambiguous between code and executor, choose executor
- If the task requires decomposition of a complex multi-step goal, choose planner
"""


class AgentRouter:
    """
    Routes tasks to the most appropriate registered agent.

    Classification order:
    1. Keyword rules (zero-latency)
    2. Capability match (structural)
    3. LLM classification (slow path, only when ambiguous)
    """

    VALID_AGENTS = {"planner", "executor", "file", "web", "system", "code"}
    DEFAULT_AGENT = "executor"

    def __init__(
        self,
        use_llm: bool = True,
        registry: Optional[Any] = None,
    ) -> None:
        self._use_llm = use_llm
        self._registry = registry
        self._llm = None
        self._route_cache: dict[str, str] = {}  # query hash → agent name

    @property
    def registry(self) -> Any:
        if self._registry is None:
            from agents.registry import AgentRegistry
            self._registry = AgentRegistry.get_instance()
        return self._registry

    @property
    def llm(self) -> Any:
        if self._llm is None:
            from core.llm_factory import get_llm
            self._llm = get_llm(temperature=0.0)
        return self._llm

    # ── Primary API ───────────────────────────────────────────────────────────

    async def route(
        self,
        task: str | dict[str, Any],
        context: Optional[dict[str, Any]] = None,
    ) -> BaseAgent:
        """
        Route a task to the best agent.

        Args:
            task: Either a string description or a task dict with 'description'.
            context: Optional additional context (e.g. file paths, current state).

        Returns:
            The best matching BaseAgent instance.
        """
        if isinstance(task, dict):
            description = str(task.get("description", task.get("goal", "")))
        else:
            description = str(task)

        agent_name = await self.classify(description, context)
        agent = self.registry.get(agent_name)

        if agent is None:
            logger.warning(
                "Routed agent not in registry — falling back",
                requested=agent_name,
                fallback=self.DEFAULT_AGENT,
            )
            agent = self.registry.get(self.DEFAULT_AGENT)

        if agent is None:
            # Very last resort: try any registered agent
            names = self.registry.list_names()
            if names:
                agent = self.registry.get(names[0])
            else:
                raise RuntimeError("No agents registered in registry.")

        logger.info("Task routed", agent=agent.name, task_snippet=description[:80])
        return agent

    async def classify(
        self,
        description: str,
        context: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Classify a task description into an agent name.

        Returns one of: planner | executor | file | web | system | code
        """
        lower = description.lower().strip()

        # Stage 1: cache lookup
        cache_key = lower[:200]
        if cache_key in self._route_cache:
            cached = self._route_cache[cache_key]
            logger.debug("Router cache hit", agent=cached)
            return cached

        # Stage 2: keyword rules
        keyword_match = self._keyword_classify(lower)
        if keyword_match:
            self._route_cache[cache_key] = keyword_match
            logger.debug("Keyword route", agent=keyword_match, description=lower[:60])
            return keyword_match

        # Stage 3: LLM classification
        if self._use_llm:
            llm_match = await self._llm_classify(description, context)
            if llm_match:
                self._route_cache[cache_key] = llm_match
                logger.debug("LLM route", agent=llm_match, description=lower[:60])
                return llm_match

        logger.debug("No route found — using default", agent=self.DEFAULT_AGENT)
        return self.DEFAULT_AGENT

    def classify_sync(self, description: str) -> str:
        """
        Synchronous keyword-only classification (no LLM).
        Useful in non-async contexts.
        """
        return self._keyword_classify(description.lower()) or self.DEFAULT_AGENT

    # ── Routing table API ─────────────────────────────────────────────────────

    def explain_routing(self, description: str) -> dict[str, Any]:
        """
        Return scoring details for each agent for debugging/transparency.
        """
        lower = description.lower()
        scores: dict[str, int] = {a: 0 for a in self.VALID_AGENTS}

        for keywords, agent_name in KEYWORD_RULES:
            score = sum(1 for kw in keywords if kw in lower)
            scores[agent_name] = scores.get(agent_name, 0) + score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return {
            "description": description[:100],
            "keyword_scores": dict(ranked),
            "recommended": ranked[0][0] if ranked[0][1] > 0 else self.DEFAULT_AGENT,
            "registered_agents": self.registry.list_names(),
        }

    def get_routing_table(self) -> list[dict[str, Any]]:
        """Return human-readable routing table with all keyword rules."""
        table = []
        for keywords, agent in KEYWORD_RULES:
            table.append({
                "agent": agent,
                "keywords": keywords,
                "example_triggers": keywords[:3],
            })
        return table

    def add_rule(self, keywords: list[str], agent_name: str) -> None:
        """Add a custom keyword rule at runtime."""
        if agent_name not in self.VALID_AGENTS:
            raise ValueError(f"Unknown agent: {agent_name}")
        KEYWORD_RULES.insert(0, (keywords, agent_name))
        self._route_cache.clear()
        logger.info("Custom route rule added", keywords=keywords, agent=agent_name)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _keyword_classify(self, lower: str) -> Optional[str]:
        """
        Vote-based keyword classification.
        Returns the agent with the most keyword matches, or None if tied at 0.
        """
        scores: dict[str, int] = {}

        for keywords, agent_name in KEYWORD_RULES:
            score = sum(1 for kw in keywords if kw in lower)
            if score > 0:
                scores[agent_name] = scores.get(agent_name, 0) + score

        if not scores:
            return None

        # Break ties by rule order (earlier rules win)
        best_score = max(scores.values())
        for _, agent_name in KEYWORD_RULES:
            if scores.get(agent_name, 0) == best_score:
                return agent_name

        return None

    async def _llm_classify(
        self,
        description: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Optional[str]:
        """Call the LLM to classify the task. Returns agent name or None."""
        context_str = ""
        if context:
            ctx_items = [
                f"- {k}: {str(v)[:100]}"
                for k, v in list(context.items())[:5]
            ]
            context_str = "\nContext:\n" + "\n".join(ctx_items)

        try:
            messages = [
                SystemMessage(content=LLM_ROUTE_PROMPT),
                HumanMessage(content=f"Task: {description[:500]}{context_str}"),
            ]
            response = await self.llm.ainvoke(messages)
            raw = response.content.strip().lower()

            # Extract the first word
            word = re.match(r"([a-z]+)", raw)
            agent_name = word.group(1) if word else ""

            if agent_name in self.VALID_AGENTS:
                return agent_name

            logger.warning("LLM returned invalid agent name", raw=raw)
            return None
        except Exception as exc:
            logger.warning("LLM routing failed", error=str(exc))
            return None


# ── Convenience factory ───────────────────────────────────────────────────────

_default_router: Optional[AgentRouter] = None


def get_router() -> AgentRouter:
    """Return or create the process-level default AgentRouter."""
    global _default_router
    if _default_router is None:
        _default_router = AgentRouter()
    return _default_router
