"""
memory/memory_manager.py — MemoryAgent: coordinator for all memory tiers.

Exposes a unified async API:
    remember(event)          → store in Working + Episodic + Semantic + Graph
    recall(query, type)      → fan-out search across tiers
    forget(item_id, type)    → remove from specific tier
    summarize_recent(hours)  → human-readable recent-activity summary

Background consolidation task runs every hour:
    Compresses episodic memories older than 7 days into summaries.

Backward-compat: MemoryManager alias preserved for existing core/ imports.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

from memory.chroma_store import SemanticMemory
from memory.episodic_memory import EpisodicMemory
from memory.knowledge_graph import KnowledgeGraph
from memory.schemas import (
    ConsolidationReport,
    MemoryEvent,
    MemoryType,
    RecallResponse,
    RecallResult,
    NodeType,
    EventType,
)
from memory.working_memory import WorkingMemory

logger = logging.getLogger(__name__)

# Local AgentState TypedDict — avoids importing langgraph in the memory module
AgentState = Dict[str, Any]

# Consolidation: sessions with more than this many old episodes get compressed
CONSOLIDATION_BATCH_SIZE = 20
# How often the consolidation loop fires (seconds)
CONSOLIDATION_INTERVAL = int(3600)


class MemoryAgent:
    """
    Central coordinator for the four-tier memory system.

    Tier routing:
    ┌─────────────┬──────────────────────────────────────────────────────┐
    │ Working     │ Redis  — active sessions, task state, agent sets     │
    │ Episodic    │ SQLite — past events, outcomes, corrections          │
    │ Semantic    │ Chroma — document/code embeddings, cosine search     │
    │ Graph       │ NetworkX — entity/relationship graph                 │
    └─────────────┴──────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        working: Optional[WorkingMemory] = None,
        episodic: Optional[EpisodicMemory] = None,
        semantic: Optional[SemanticMemory] = None,
        graph: Optional[KnowledgeGraph] = None,
    ) -> None:
        self.working = working or WorkingMemory()
        self.episodic = episodic or EpisodicMemory()
        self.semantic = semantic or SemanticMemory()
        self.graph = graph or KnowledgeGraph()

        self._consolidation_task: Optional[asyncio.Task] = None  # type: ignore

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Start the background consolidation loop.
        Call this once during agent/app startup.
        """
        if self._consolidation_task is None or self._consolidation_task.done():
            self._consolidation_task = asyncio.create_task(
                self._consolidation_loop(),
                name="memory_consolidation",
            )
            logger.info("Memory consolidation loop started")

    async def stop(self) -> None:
        """Stop the background consolidation loop."""
        if self._consolidation_task and not self._consolidation_task.done():
            self._consolidation_task.cancel()
            try:
                await self._consolidation_task
            except asyncio.CancelledError:
                pass
        logger.info("Memory consolidation loop stopped")

    # ── Core API ──────────────────────────────────────────────────────────────

    async def remember(self, event: MemoryEvent) -> dict[str, str]:
        """
        Store an event across the appropriate memory tiers.

        - Always writes to Episodic (permanent record)
        - Writes to Working memory if session context is meaningful
        - Indexes content in Semantic memory (documents/long content)
        - Extracts entities into Knowledge Graph
        
        Returns dict of {tier: item_id} for each tier written.
        """
        result: dict[str, str] = {}
        text = event.content
        if event.outcome:
            text += f"\n{event.outcome}"

        # 1. Episodic — always record
        try:
            ep_id = await self.episodic.record(
                content=event.content,
                event_type=event.event_type,
                outcome=event.outcome,
                correction=event.correction,
                session_id=event.session_id,
                user_id=event.user_id,
                metadata=event.metadata,
                timestamp=event.timestamp,
            )
            result["episodic"] = ep_id
        except Exception as exc:
            logger.warning("remember: episodic write failed: %s", exc)

        # 2. Working memory — update session context
        try:
            await self.working.store_context(
                session_id=event.session_id,
                key=f"last_{event.event_type.value}",
                value={
                    "content": event.content[:200],
                    "outcome": event.outcome,
                    "timestamp": event.timestamp.isoformat(),
                },
            )
            result["working"] = event.session_id
        except Exception as exc:
            logger.warning("remember: working write failed: %s", exc)

        # 3. Semantic — index if content is substantial (>50 chars)
        if len(event.content) > 50:
            try:
                ids = await self.semantic.index_document(
                    content=text,
                    source_path=f"event:{event.event_type.value}:{event.session_id}",
                    metadata={
                        "user_id": event.user_id,
                        "session_id": event.session_id,
                        "event_type": event.event_type.value,
                        "timestamp": event.timestamp.isoformat(),
                    },
                    namespace=event.user_id,
                )
                result["semantic"] = ids[0] if ids else ""
            except Exception as exc:
                logger.warning("remember: semantic write failed: %s", exc)

        # 4. Knowledge Graph — extract entities from event text
        try:
            node_ids = await self.graph.extract_and_add(
                text=text,
                source_entity=event.agent_id or event.event_type.value,
                source_type=NodeType.CONCEPT,
            )
            result["graph"] = ",".join(node_ids[:5])
        except Exception as exc:
            logger.warning("remember: graph extraction failed: %s", exc)

        logger.debug("Event remembered: event_type=%s, tiers=%s", event.event_type.value, list(result.keys()))
        return result

    async def recall(
        self,
        query: str,
        memory_type: MemoryType = MemoryType.ALL,
        top_k: int = 5,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> RecallResponse:
        """
        Search across memory tiers and return ranked results.

        memory_type=ALL fans out to episodic + semantic simultaneously.
        memory_type=WORKING returns the current session's working set.
        memory_type=GRAPH returns related entities to the query term.
        """
        t0 = time.monotonic()
        results: list[RecallResult] = []
        sources: list[MemoryType] = []

        if memory_type in (MemoryType.WORKING, MemoryType.ALL):
            try:
                sess = session_id or "default"
                ws = await self.working.get_working_set(sess)
                if ws:
                    # Working memory doesn't do similarity; return full context
                    results.append(
                        RecallResult(
                            item_id=f"wm:{sess}",
                            content=str(ws),
                            source=MemoryType.WORKING,
                            relevance=1.0,
                            timestamp=datetime.utcnow(),
                            metadata={"session_id": sess},
                        )
                    )
                    sources.append(MemoryType.WORKING)
            except Exception as exc:
                logger.warning("recall: working read failed: %s", exc)

        # Episodic + Semantic can run in parallel
        ep_task = sem_task = None

        if memory_type in (MemoryType.EPISODIC, MemoryType.ALL):
            ep_task = asyncio.create_task(
                self.episodic.query_similar(query, top_k=top_k, user_id=user_id)
            )

        if memory_type in (MemoryType.SEMANTIC, MemoryType.ALL):
            sem_task = asyncio.create_task(
                self.semantic.search(
                    query=query,
                    top_k=top_k,
                    namespace=user_id or "global",
                )
            )

        if ep_task:
            try:
                episodes = await ep_task
                sources.append(MemoryType.EPISODIC)
                for ep in episodes:
                    results.append(
                        RecallResult(
                            item_id=ep.episode_id,
                            content=ep.content,
                            source=MemoryType.EPISODIC,
                            relevance=0.9,  # FTS doesn't return scores
                            timestamp=ep.timestamp,
                            metadata={
                                "event_type": ep.event_type,
                                "session_id": ep.session_id,
                                "correction": ep.correction,
                            },
                        )
                    )
            except Exception as exc:
                logger.warning("recall: episodic read failed: %s", exc)

        if sem_task:
            try:
                sem_results = await sem_task
                sources.append(MemoryType.SEMANTIC)
                results.extend(sem_results)
            except Exception as exc:
                logger.warning("recall: semantic read failed: %s", exc)

        if memory_type in (MemoryType.GRAPH, MemoryType.ALL):
            try:
                subgraph = await self.graph.get_related(query, depth=2)
                sources.append(MemoryType.GRAPH)
                for node in subgraph.nodes[:top_k]:
                    results.append(
                        RecallResult(
                            item_id=node.node_id,
                            content=f"{node.node_type.value}: {node.name}",
                            source=MemoryType.GRAPH,
                            relevance=min(1.0, node.mention_count / 10.0),
                            timestamp=node.created_at,
                            metadata={
                                "node_type": node.node_type.value,
                                "mention_count": node.mention_count,
                                "attributes": node.attributes,
                            },
                        )
                    )
            except Exception as exc:
                logger.warning("recall: graph read failed: %s", exc)

        # Sort by relevance descending
        results.sort(key=lambda r: r.relevance, reverse=True)

        elapsed = (time.monotonic() - t0) * 1000
        return RecallResponse(
            query=query,
            results=results[:top_k * 2],
            sources_queried=list(set(sources)),
            total_found=len(results),
            elapsed_ms=round(elapsed, 2),
        )

    async def forget(
        self,
        item_id: str,
        memory_type: MemoryType = MemoryType.ALL,
        user_id: Optional[str] = None,
    ) -> dict[str, bool]:
        """
        Remove an item from the specified memory tier(s).
        Returns dict of {tier: success} for each attempted deletion.
        """
        success: dict[str, bool] = {}

        if memory_type in (MemoryType.EPISODIC, MemoryType.ALL):
            try:
                await self.episodic.delete_episode(item_id)
                success["episodic"] = True
            except Exception as exc:
                logger.warning("forget: episodic failed: %s", exc)
                success["episodic"] = False

        if memory_type in (MemoryType.SEMANTIC, MemoryType.ALL):
            try:
                await self.semantic.delete_document(
                    item_id, namespace=user_id or "global"
                )
                success["semantic"] = True
            except Exception as exc:
                logger.warning("forget: semantic failed: %s", exc)
                success["semantic"] = False

        if memory_type in (MemoryType.GRAPH, MemoryType.ALL):
            try:
                await self.graph.remove_entity(item_id)
                success["graph"] = True
            except Exception as exc:
                logger.warning("forget: graph failed: %s", exc)
                success["graph"] = False

        logger.info("Forget operation completed: item_id=%s, results=%s", item_id, success)
        return success

    async def summarize_recent(
        self,
        hours: int = 24,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Compile a human-readable summary of recent activity.

        Aggregates episodic events from the last `hours` hours and
        augments with working memory context.
        """
        episodes = await self.episodic.query_recent(
            hours=hours,
            user_id=user_id,
            session_id=session_id,
            limit=50,
        )

        if not episodes:
            return f"No activity recorded in the last {hours} hours."

        lines = [f"## Activity Summary — Last {hours} Hours\n"]
        lines.append(f"Total events: {len(episodes)}\n")

        # Group by event type
        by_type: dict[str, list[Any]] = {}
        for ep in episodes:
            by_type.setdefault(ep.event_type, []).append(ep)

        for event_type, eps in by_type.items():
            lines.append(f"\n### {event_type.replace('_', ' ').title()} ({len(eps)})")
            for ep in eps[:5]:  # Show at most 5 per type
                ts = ep.timestamp.strftime("%H:%M") if ep.timestamp else "?"
                snippet = ep.content[:120].replace("\n", " ")
                lines.append(f"- [{ts}] {snippet}")
            if len(eps) > 5:
                lines.append(f"  ... and {len(eps) - 5} more")

        # Append graph stats
        try:
            stats = await self.graph.stats()
            lines.append(
                f"\n### Knowledge Graph\n"
                f"- Nodes: {stats['nodes']} | Edges: {stats['edges']}\n"
                f"- Types: {stats['node_types']}"
            )
        except Exception:
            pass

        return "\n".join(lines)

    # ── Consolidation ─────────────────────────────────────────────────────────

    async def consolidate(self) -> ConsolidationReport:
        """
        Memory consolidation: compress episodic memories older than 7 days
        into summaries. Runs in the background every hour.

        Algorithm:
        1. Fetch all raw episodes older than 7 days (limit 200)
        2. Group by session_id (batches of ≤ CONSOLIDATION_BATCH_SIZE)
        3. For each batch: build summary → store → soft-delete originals
        """
        t0 = time.monotonic()
        report = ConsolidationReport()

        try:
            old_episodes = await self.episodic.get_episodes_older_than(days=7, limit=200)
            report.episodes_scanned = len(old_episodes)

            if not old_episodes:
                logger.info("Consolidation: no old episodes to compress")
                return report

            # Group by session_id
            by_session: dict[str, list[Any]] = {}
            for ep in old_episodes:
                by_session.setdefault(ep.session_id, []).append(ep)

            for session_id, eps in by_session.items():
                # Process in batches
                for i in range(0, len(eps), CONSOLIDATION_BATCH_SIZE):
                    batch = eps[i: i + CONSOLIDATION_BATCH_SIZE]
                    if not batch:
                        continue

                    # Build summary text
                    summary_lines = [
                        f"Consolidated memory: {len(batch)} events from "
                        f"{batch[0].timestamp.date() if batch[0].timestamp else '?'} "
                        f"to {batch[-1].timestamp.date() if batch[-1].timestamp else '?'}"
                    ]
                    for ep in batch:
                        ts = ep.timestamp.strftime("%Y-%m-%d") if ep.timestamp else "?"
                        snippet = ep.content[:200].replace("\n", " ")
                        summary_lines.append(f"[{ts}] [{ep.event_type}] {snippet}")
                        if ep.correction:
                            summary_lines.append(f"  ↳ Correction: {ep.correction}")

                    summary_text = "\n".join(summary_lines)
                    episode_ids = [ep.episode_id for ep in batch]

                    try:
                        await self.episodic.mark_as_summary(
                            episode_ids=episode_ids,
                            summary_content=summary_text,
                            session_id=session_id,
                            user_id=batch[0].user_id,
                        )
                        report.episodes_consolidated += len(batch)
                        report.summaries_created += 1
                    except Exception as exc:
                        err_msg = f"Consolidation batch failed: {exc}"
                        logger.warning(err_msg)
                        report.errors.append(err_msg)

        except Exception as exc:
            err_msg = f"Consolidation run failed: {exc}"
            logger.error(err_msg)
            report.errors.append(err_msg)

        report.elapsed_seconds = round(time.monotonic() - t0, 3)
        logger.info(
            "Memory consolidation complete: scanned=%d, consolidated=%d, summaries=%d, elapsed_s=%.3f",
            report.episodes_scanned,
            report.episodes_consolidated,
            report.summaries_created,
            report.elapsed_seconds,
        )
        return report

    async def _consolidation_loop(self) -> None:
        """Background task: run consolidation every CONSOLIDATION_INTERVAL seconds."""
        logger.info(
            "Consolidation loop running: interval_s=%d",
            CONSOLIDATION_INTERVAL,
        )
        while True:
            await asyncio.sleep(CONSOLIDATION_INTERVAL)
            try:
                await self.consolidate()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Consolidation loop error: %s", exc)

    # ── Backward-Compatible MemoryManager API ─────────────────────────────────

    async def load_context(
        self,
        session_id: str,
        user_id: str,
        query: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Legacy load_context — used by core/graph_nodes.py etc."""
        working = await self.working.get_session(session_id) or {}
        episodic_results = await self.episodic.query_similar(
            query, top_k=top_k, user_id=user_id
        )
        return {
            "working": working,
            "episodic": [
                {"content": ep.content, "metadata": ep.metadata}
                for ep in episodic_results
            ],
            "semantic": [],
        }

    async def save_context(
        self,
        session_id: str,
        user_id: str,
        state: AgentState,
    ) -> None:
        """Legacy save_context — used by core/graph_nodes.py etc."""
        await self.working.set_session(
            session_id=session_id,
            data={
                "last_goal": state.get("goal"),
                "last_status": state.get("status"),
                "tool_results": state.get("tool_results", []),
                "iterations": state.get("iterations", 0),
            },
        )
        if state.get("status") == "completed" and state.get("goal"):
            event = MemoryEvent(
                event_type=EventType.TASK_COMPLETED,
                content=str(state.get("goal", "")),
                outcome=str(state.get("tool_results", ""))[:500],
                session_id=session_id,
                user_id=user_id,
                metadata={
                    "task_id": str(state.get("task_id", "")),
                    "iterations": state.get("iterations", 0),
                },
            )
            await self.remember(event)

    async def clear_session(self, session_id: str) -> None:
        """Legacy clear_session."""
        await self.working.delete_session(session_id)
        logger.info("Session memory cleared: session_id=%s", session_id)


# ── Backward-compatible alias ─────────────────────────────────────────────────
MemoryManager = MemoryAgent
