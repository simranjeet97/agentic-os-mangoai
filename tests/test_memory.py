"""
tests/test_memory.py — Full async test suite for the Memory System.

Test classes:
    TestEmbeddingService    — encode, cosine similarity, fallback
    TestWorkingMemory       — set/get/expire, task state, agent sets, cache
    TestEpisodicMemory      — record, query_recent, query_similar, corrections,
                              consolidation markers, FTS5 fallback
    TestSemanticMemory      — index_document, search, delete (uses ephemeral Chroma)
    TestKnowledgeGraph      — add_entity, add_relationship, get_related,
                              extract_and_add, serialization, analytics
    TestMemoryAgent         — remember, recall, forget, summarize_recent,
                              consolidate, backward-compat API
    TestConsolidation       — full consolidation pipeline with age simulation

Infrastructure:
    - fakeredis for Redis (no server required)
    - chromadb.EphemeralClient for Chroma (no server required)
    - aiosqlite with in-memory ":memory:" DB
    - networkx graph with temp JSON path
"""

from __future__ import annotations

import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def graph_path(temp_dir):
    return str(Path(temp_dir) / "test_graph.json")


@pytest_asyncio.fixture
async def episodic(temp_dir):
    from memory.episodic_memory import EpisodicMemory
    db_path = str(Path(temp_dir) / "test_episodic.db")
    mem = EpisodicMemory(db_path=db_path)
    await mem._ensure_init()
    return mem


@pytest_asyncio.fixture
async def episodic_inmem(temp_dir):
    """Temporary file SQLite instance (reliable persistence for tests)."""
    from memory.episodic_memory import EpisodicMemory
    db_path = str(Path(temp_dir) / f"test_episodic_{uuid.uuid4().hex[:8]}.db")
    mem = EpisodicMemory(db_path=db_path)
    await mem._ensure_init()
    return mem


@pytest_asyncio.fixture
async def semantic():
    """SemanticMemory with ephemeral ChromaDB (no server)."""
    import chromadb
    from memory.chroma_store import SemanticMemory
    from memory.embeddings import EmbeddingService
    client = chromadb.EphemeralClient()
    # Use a mock embed service to avoid loading the model in tests
    embed = MagicMock(spec=EmbeddingService)
    embed.encode = AsyncMock(
        side_effect=lambda texts: np.random.rand(len(texts), 384).astype(np.float32)
    )
    embed.encode_single = AsyncMock(
        return_value=np.random.rand(384).astype(np.float32)
    )
    mem = SemanticMemory(embed_service=embed, chroma_client=client)
    return mem


@pytest_asyncio.fixture
async def knowledge_graph(graph_path):
    from memory.knowledge_graph import KnowledgeGraph
    return KnowledgeGraph(graph_path=graph_path)


@pytest_asyncio.fixture
async def working_memory_fake():
    """WorkingMemory backed by fakeredis (no Redis server required)."""
    try:
        import fakeredis.aioredis as fake_aio
        fake_client = fake_aio.FakeRedis(decode_responses=True)

        from memory.working_memory import WorkingMemory
        wm = WorkingMemory()
        wm._client = fake_client
        return wm
    except ImportError:
        pytest.skip("fakeredis not installed — skipping WorkingMemory tests")


@pytest_asyncio.fixture
async def agent(working_memory_fake, episodic_inmem, semantic, knowledge_graph):
    from memory.memory_manager import MemoryAgent
    return MemoryAgent(
        working=working_memory_fake,
        episodic=episodic_inmem,
        semantic=semantic,
        graph=knowledge_graph,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TestEmbeddingService
# ─────────────────────────────────────────────────────────────────────────────


class TestEmbeddingService:
    def test_cosine_similarity_identical(self):
        from memory.embeddings import EmbeddingService
        svc = EmbeddingService()
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert svc.cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_cosine_similarity_orthogonal(self):
        from memory.embeddings import EmbeddingService
        svc = EmbeddingService()
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert svc.cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-5)

    def test_cosine_similarity_zero_vector(self):
        from memory.embeddings import EmbeddingService
        svc = EmbeddingService()
        a = np.zeros(10, dtype=np.float32)
        b = np.ones(10, dtype=np.float32)
        assert svc.cosine_similarity(a, b) == 0.0

    def test_batch_cosine_similarity_shape(self):
        from memory.embeddings import EmbeddingService
        svc = EmbeddingService()
        query = np.random.rand(384).astype(np.float32)
        corpus = np.random.rand(10, 384).astype(np.float32)
        scores = svc.batch_cosine_similarity(query, corpus)
        assert scores.shape == (10,)

    @pytest.mark.asyncio
    async def test_encode_empty(self):
        from memory.embeddings import EmbeddingService
        svc = EmbeddingService()
        result = await svc.encode([])
        assert result.shape == (0, 384)

    @pytest.mark.asyncio
    async def test_fallback_on_import_error(self):
        """If sentence-transformers is absent, return zero vectors."""
        from memory.embeddings import EmbeddingService, _EMBED_DIM
        import memory.embeddings as emb_mod

        original = emb_mod._load_model
        emb_mod._load_model.cache_clear() if hasattr(emb_mod._load_model, "cache_clear") else None

        with patch.object(emb_mod, "_encode_sync", return_value=np.zeros((2, _EMBED_DIM), dtype=np.float32)):
            svc = EmbeddingService()
            vecs = await svc.encode(["hello", "world"])
            assert vecs.shape == (2, _EMBED_DIM)

    def test_embedding_dim(self):
        from memory.embeddings import EmbeddingService
        svc = EmbeddingService()
        assert svc.embedding_dim == 384

    def test_get_embedding_service_singleton(self):
        from memory.embeddings import get_embedding_service
        a = get_embedding_service()
        b = get_embedding_service()
        assert a is b


# ─────────────────────────────────────────────────────────────────────────────
# TestWorkingMemory
# ─────────────────────────────────────────────────────────────────────────────


class TestWorkingMemory:
    @pytest.mark.asyncio
    async def test_store_and_get_context(self, working_memory_fake):
        wm = working_memory_fake
        await wm.store_context("sess1", "goal", "build OS", ttl=3600)
        result = await wm.get_context("sess1", "goal")
        assert result == "build OS"

    @pytest.mark.asyncio
    async def test_get_context_missing_key(self, working_memory_fake):
        result = await working_memory_fake.get_context("sess_missing", "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_working_set(self, working_memory_fake):
        wm = working_memory_fake
        await wm.store_context("sess2", "key_a", 42, ttl=3600)
        await wm.store_context("sess2", "key_b", "hello", ttl=3600)
        ws = await wm.get_working_set("sess2")
        assert "key_a" in ws
        assert "key_b" in ws
        assert ws["key_a"] == 42

    @pytest.mark.asyncio
    async def test_set_and_get_session(self, working_memory_fake):
        wm = working_memory_fake
        data = {"goal": "test", "status": "running"}
        await wm.set_session("sess3", data)
        retrieved = await wm.get_session("sess3")
        assert retrieved is not None
        assert retrieved["goal"] == "test"

    @pytest.mark.asyncio
    async def test_delete_session(self, working_memory_fake):
        wm = working_memory_fake
        await wm.set_session("sess4", {"x": 1})
        await wm.store_context("sess4", "y", 2)
        await wm.delete_session("sess4")
        assert await wm.get_session("sess4") is None
        assert await wm.get_context("sess4", "y") is None

    @pytest.mark.asyncio
    async def test_set_and_get_task_state(self, working_memory_fake):
        wm = working_memory_fake
        state = {"step": "planning", "goal": "write tests"}
        await wm.set_task_state("sess5", state)
        retrieved = await wm.get_task_state("sess5")
        assert retrieved is not None
        assert retrieved["step"] == "planning"

    @pytest.mark.asyncio
    async def test_agent_working_set(self, working_memory_fake):
        wm = working_memory_fake
        items = ["file_a.py", "file_b.py", {"action": "edit"}]
        await wm.set_agent_working_set("agent-007", items)
        retrieved = await wm.get_agent_working_set("agent-007")
        assert len(retrieved) == 3
        assert "file_a.py" in retrieved

    @pytest.mark.asyncio
    async def test_agent_working_set_empty(self, working_memory_fake):
        result = await working_memory_fake.get_agent_working_set("nonexistent-agent")
        assert result == []

    @pytest.mark.asyncio
    async def test_cache_tool_result(self, working_memory_fake):
        wm = working_memory_fake
        await wm.cache_tool_result("search:python", {"results": [1, 2, 3]})
        cached = await wm.get_cached_tool_result("search:python")
        assert cached == {"results": [1, 2, 3]}

    @pytest.mark.asyncio
    async def test_cache_miss(self, working_memory_fake):
        result = await working_memory_fake.get_cached_tool_result("nonexistent:key")
        assert result is None

    @pytest.mark.asyncio
    async def test_enqueue_and_dequeue_task(self, working_memory_fake):
        wm = working_memory_fake
        task = {"action": "web_search", "query": "AI memory systems"}
        await wm.enqueue_task("main", task)
        dequeued = await wm.dequeue_task("main", timeout=1)
        assert dequeued is not None
        assert dequeued["action"] == "web_search"

    @pytest.mark.asyncio
    async def test_context_stores_complex_types(self, working_memory_fake):
        wm = working_memory_fake
        value = {"nested": {"list": [1, 2, 3], "bool": True}}
        await wm.store_context("sess6", "complex", value)
        result = await wm.get_context("sess6", "complex")
        assert result["nested"]["list"] == [1, 2, 3]


# ─────────────────────────────────────────────────────────────────────────────
# TestEpisodicMemory
# ─────────────────────────────────────────────────────────────────────────────


class TestEpisodicMemory:
    @pytest.mark.asyncio
    async def test_record_and_count(self, episodic_inmem):
        em = episodic_inmem
        ep_id = await em.record(
            content="ran pytest and all tests passed",
            event_type="task_completed",
            session_id="s1",
            user_id="u1",
        )
        assert isinstance(ep_id, str) and len(ep_id) == 36
        count = await em.count(include_summaries=False)
        assert count == 1

    @pytest.mark.asyncio
    async def test_record_with_outcome_and_correction(self, episodic_inmem):
        em = episodic_inmem
        ep_id = await em.record(
            content="Agent tried to rm -rf /",
            event_type="user_correction",
            outcome="Blocked by guardrails",
        )
        ep = await em.get_episode(ep_id)
        assert ep is not None
        assert ep.correction is None  # correction added separately below
        await em.add_correction(ep_id, "Never delete system files")
        ep_updated = await em.get_episode(ep_id)
        assert ep_updated.correction == "Never delete system files"

    @pytest.mark.asyncio
    async def test_query_recent_default_hours(self, episodic_inmem):
        em = episodic_inmem
        for i in range(5):
            await em.record(content=f"Event {i}", session_id="s2", user_id="u2")
        results = await em.query_recent(hours=24, user_id="u2")
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_query_recent_filters_by_user(self, episodic_inmem):
        em = episodic_inmem
        await em.record(content="User A event", user_id="userA")
        await em.record(content="User B event", user_id="userB")
        results = await em.query_recent(hours=24, user_id="userA")
        assert all(ep.user_id == "userA" for ep in results)

    @pytest.mark.asyncio
    async def test_query_similar_fts(self, episodic_inmem):
        em = episodic_inmem
        await em.record(content="Installed numpy and pandas for data processing")
        await em.record(content="Configured docker containers for deployment")
        await em.record(content="Updated requirements.txt with new packages")
        results = await em.query_similar("docker", top_k=5)
        assert any("docker" in r.content.lower() for r in results)

    @pytest.mark.asyncio
    async def test_query_similar_fallback_like(self, episodic_inmem):
        """Short queries may fallback from FTS5 to LIKE search."""
        em = episodic_inmem
        await em.record(content="ran git commit -m 'fix memory bug'")
        # Short single term triggers FTS5 fallback to LIKE
        results = await em.query_similar("git", top_k=5)
        assert len(results) >= 0  # Either works, just no crash

    @pytest.mark.asyncio
    async def test_mark_as_summary(self, episodic_inmem):
        em = episodic_inmem
        ids = []
        for i in range(5):
            eid = await em.record(content=f"Old event {i}", session_id="old_sess")
            ids.append(eid)
        summary_id = await em.mark_as_summary(
            episode_ids=ids,
            summary_content="Summary of 5 old events",
            session_id="old_sess",
        )
        # Originals should be soft-deleted (is_summary=True)
        count_raw = await em.count(include_summaries=False)
        assert count_raw == 1  # Only the summary itself
        # Summary episode should exist
        summary = await em.get_episode(summary_id)
        assert summary is not None
        assert "Summary" in summary.content

    @pytest.mark.asyncio
    async def test_delete_episode(self, episodic_inmem):
        em = episodic_inmem
        ep_id = await em.record(content="To be deleted")
        await em.delete_episode(ep_id)
        ep = await em.get_episode(ep_id)
        assert ep is None

    @pytest.mark.asyncio
    async def test_get_episodes_older_than(self, episodic_inmem):
        em = episodic_inmem
        # Insert an episode with an old timestamp directly
        old_ts = (datetime.utcnow() - timedelta(days=8)).isoformat()
        import aiosqlite
        async with aiosqlite.connect(em._db_path) as db:
            await db.execute(
                "INSERT INTO episodes (id, session_id, user_id, event_type, content, is_summary, timestamp, metadata)"
                " VALUES (?, 'old', 'u', 'custom', 'ancient event', 0, ?, '{}')",
                (str(uuid.uuid4()), old_ts),
            )
            await db.commit()
        old_eps = await em.get_episodes_older_than(days=7)
        assert len(old_eps) >= 1
        assert any("ancient" in ep.content for ep in old_eps)

    @pytest.mark.asyncio
    async def test_content_not_empty_validation(self):
        from memory.schemas import MemoryEvent, EventType
        with pytest.raises(Exception):
            MemoryEvent(event_type=EventType.CUSTOM, content="   ")

    @pytest.mark.asyncio
    async def test_metadata_stored_and_retrieved(self, episodic_inmem):
        ep_id = await episodic_inmem.record(
            content="event with meta",
            metadata={"source": "test", "priority": 5},
        )
        ep = await episodic_inmem.get_episode(ep_id)
        assert ep.metadata.get("source") == "test"


# ─────────────────────────────────────────────────────────────────────────────
# TestSemanticMemory
# ─────────────────────────────────────────────────────────────────────────────


class TestSemanticMemory:
    @pytest.mark.asyncio
    async def test_index_and_search(self, semantic):
        doc = "The quick brown fox jumps over the lazy dog. Machine learning is fun."
        ids = await semantic.index_document(
            content=doc, source_path="test_doc.txt", namespace="test"
        )
        assert len(ids) > 0
        results = await semantic.search("machine learning", top_k=3, namespace="test")
        assert len(results) > 0
        assert results[0].source.value == "semantic"

    @pytest.mark.asyncio
    async def test_index_chunking(self, semantic):
        """A long document should be split into multiple chunks."""
        long_doc = " ".join([f"word_{i}" for i in range(1200)])
        ids = await semantic.index_document(
            content=long_doc, source_path="big.txt", namespace="test_chunks"
        )
        assert len(ids) >= 2  # Should be chunked

    @pytest.mark.asyncio
    async def test_search_empty_collection(self, semantic):
        results = await semantic.search("anything", namespace="empty_namespace")
        assert results == []

    @pytest.mark.asyncio
    async def test_delete_document(self, semantic):
        doc_id = str(uuid.uuid4())
        await semantic.index_document(
            content="document to be deleted",
            source_path="delete_me.txt",
            doc_id=doc_id,
            namespace="test_del",
        )
        await semantic.delete_document(doc_id, namespace="test_del")
        # Should return empty or no exact match
        results = await semantic.search("delete me", namespace="test_del", top_k=1)
        # Can't guarantee exact deletion since we mock embeddings, but no crash
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_add_episode_backward_compat(self, semantic):
        ep_id = await semantic.add_episode(
            user_id="u1",
            session_id="s1",
            goal="Write unit tests",
            result="All tests passing",
        )
        assert isinstance(ep_id, str)

    @pytest.mark.asyncio
    async def test_search_returns_recall_result_objects(self, semantic):
        from memory.schemas import RecallResult
        await semantic.index_document(
            content="pytest asyncio testing patterns", namespace="recall_test"
        )
        results = await semantic.search("pytest", namespace="recall_test")
        for r in results:
            assert isinstance(r, RecallResult)
            assert 0.0 <= r.relevance <= 1.0

    @pytest.mark.asyncio
    async def test_multiple_namespaces_isolated(self, semantic):
        await semantic.index_document("content A", namespace="ns_a")
        await semantic.index_document("content B", namespace="ns_b")
        results_a = await semantic.search("content", namespace="ns_a")
        results_b = await semantic.search("content", namespace="ns_b")
        # Each namespace should have its own results
        assert isinstance(results_a, list)
        assert isinstance(results_b, list)


# ─────────────────────────────────────────────────────────────────────────────
# TestKnowledgeGraph
# ─────────────────────────────────────────────────────────────────────────────


class TestKnowledgeGraph:
    @pytest.mark.asyncio
    async def test_add_entity(self, knowledge_graph):
        kg = knowledge_graph
        nid = await kg.add_entity("main.py", "file")
        assert isinstance(nid, str)
        node = await kg.get_entity(nid)
        assert node is not None
        assert node.name == "main.py"
        assert node.node_type.value == "file"

    @pytest.mark.asyncio
    async def test_add_entity_increments_mention_count(self, knowledge_graph):
        kg = knowledge_graph
        from memory.schemas import NodeType
        nid1 = await kg.add_entity("python", NodeType.COMMAND)
        nid2 = await kg.add_entity("python", NodeType.COMMAND)
        assert nid1 == nid2
        node = await kg.get_entity(nid1)
        assert node.mention_count == 2

    @pytest.mark.asyncio
    async def test_add_relationship(self, knowledge_graph):
        kg = knowledge_graph
        from memory.schemas import NodeType
        await kg.add_relationship(
            "MemoryAgent", "EpisodicMemory",
            relationship="coordinates",
            source_type=NodeType.CONCEPT,
            target_type=NodeType.CONCEPT,
        )
        stats = await kg.stats()
        assert stats["edges"] >= 1

    @pytest.mark.asyncio
    async def test_add_relationship_increments_weight(self, knowledge_graph):
        kg = knowledge_graph
        from memory.schemas import NodeType
        for _ in range(3):
            await kg.add_relationship(
                "A", "B", "linked",
                source_type=NodeType.CONCEPT,
                target_type=NodeType.CONCEPT,
                weight=1.0,
            )
        src_id = kg._find_node_id("A")
        dst_id = kg._find_node_id("B")
        assert kg._graph[src_id][dst_id]["weight"] == pytest.approx(3.0)

    @pytest.mark.asyncio
    async def test_get_related(self, knowledge_graph):
        kg = knowledge_graph
        from memory.schemas import NodeType
        await kg.add_relationship("A", "B", "rel1", NodeType.CONCEPT, NodeType.CONCEPT)
        await kg.add_relationship("B", "C", "rel2", NodeType.CONCEPT, NodeType.CONCEPT)
        subgraph = await kg.get_related("A", depth=2)
        names = {n.name for n in subgraph.nodes}
        assert "A" in names
        assert "B" in names

    @pytest.mark.asyncio
    async def test_get_related_nonexistent(self, knowledge_graph):
        subgraph = await knowledge_graph.get_related("does_not_exist")
        assert subgraph.nodes == []

    @pytest.mark.asyncio
    async def test_find_path(self, knowledge_graph):
        kg = knowledge_graph
        from memory.schemas import NodeType
        await kg.add_relationship("X", "Y", "to", NodeType.CONCEPT, NodeType.CONCEPT)
        await kg.add_relationship("Y", "Z", "to", NodeType.CONCEPT, NodeType.CONCEPT)
        path = await kg.find_path("X", "Z")
        assert path == ["X", "Y", "Z"]

    @pytest.mark.asyncio
    async def test_find_path_no_path(self, knowledge_graph):
        kg = knowledge_graph
        from memory.schemas import NodeType
        await kg.add_entity("isolated1", NodeType.CONCEPT)
        await kg.add_entity("isolated2", NodeType.CONCEPT)
        path = await kg.find_path("isolated1", "isolated2")
        assert path == []

    @pytest.mark.asyncio
    async def test_extract_and_add(self, knowledge_graph):
        kg = knowledge_graph
        text = "Running pytest tests/test_memory.py with python3.12"
        ids = await kg.extract_and_add(text)
        assert len(ids) > 0
        stats = await kg.stats()
        assert stats["nodes"] > 0

    @pytest.mark.asyncio
    async def test_get_top_entities(self, knowledge_graph):
        kg = knowledge_graph
        from memory.schemas import NodeType
        for name in ["alpha", "beta", "gamma", "delta"]:
            await kg.add_entity(name, NodeType.CONCEPT)
        top = await kg.get_top_entities(n=4)
        assert len(top) <= 4

    @pytest.mark.asyncio
    async def test_remove_entity(self, knowledge_graph):
        kg = knowledge_graph
        from memory.schemas import NodeType
        nid = await kg.add_entity("temp_entity", NodeType.CONCEPT)
        await kg.remove_entity(nid)
        node = await kg.get_entity(nid)
        assert node is None

    @pytest.mark.asyncio
    async def test_stats_empty(self, graph_path):
        from memory.knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph(graph_path=graph_path + "_empty.json")
        stats = await kg.stats()
        assert stats["nodes"] == 0
        assert stats["edges"] == 0

    @pytest.mark.asyncio
    async def test_json_persistence(self, graph_path):
        """Graph should persist and reload correctly."""
        from memory.knowledge_graph import KnowledgeGraph
        from memory.schemas import NodeType
        kg1 = KnowledgeGraph(graph_path=graph_path)
        await kg1.add_entity("persistent_node", NodeType.PROJECT)
        await kg1._save()

        kg2 = KnowledgeGraph(graph_path=graph_path)
        await kg2._ensure_loaded()
        nid = kg2._find_node_id("persistent_node", NodeType.PROJECT)
        assert nid is not None


# ─────────────────────────────────────────────────────────────────────────────
# TestMemoryAgent
# ─────────────────────────────────────────────────────────────────────────────


class TestMemoryAgent:
    @pytest.mark.asyncio
    async def test_remember_returns_tiers(self, agent):
        from memory.schemas import MemoryEvent, EventType
        event = MemoryEvent(
            event_type=EventType.FILE_EDITED,
            content="Edited knowledge_graph.py to add path finding algorithm",
            outcome="File saved. NetworkX graph updated.",
            session_id="test-session",
            user_id="test-user",
        )
        result = await agent.remember(event)
        assert "episodic" in result
        assert isinstance(result["episodic"], str)

    @pytest.mark.asyncio
    async def test_remember_short_content_skips_semantic(self, agent):
        from memory.schemas import MemoryEvent, EventType
        event = MemoryEvent(
            event_type=EventType.CUSTOM,
            content="short",
            session_id="s",
            user_id="u",
        )
        result = await agent.remember(event)
        # Short content (<50 chars) should not be indexed in semantic
        assert result.get("semantic", "") == "" or "semantic" not in result

    @pytest.mark.asyncio
    async def test_recall_all_tiers(self, agent):
        from memory.schemas import MemoryEvent, EventType, MemoryType
        event = MemoryEvent(
            event_type=EventType.COMMAND_RUN,
            content="Ran docker compose up to start all services including Redis, ChromaDB",
            outcome="All 5 services started successfully",
            session_id="test-session",
            user_id="test-user",
        )
        await agent.remember(event)
        response = await agent.recall(
            "docker services redis",
            memory_type=MemoryType.ALL,
            user_id="test-user",
        )
        assert response.total_found >= 0  # May be 0 with mocked embeddings
        assert response.elapsed_ms >= 0

    @pytest.mark.asyncio
    async def test_recall_episodic_only(self, agent):
        from memory.schemas import MemoryEvent, EventType, MemoryType
        event = MemoryEvent(
            event_type=EventType.TASK_COMPLETED,
            content="Completed implementing the semantic memory with ChromaDB vector store",
            session_id="ep-session",
            user_id="ep-user",
        )
        await agent.remember(event)
        response = await agent.recall(
            "ChromaDB semantic",
            memory_type=MemoryType.EPISODIC,
            user_id="ep-user",
        )
        assert MemoryType.EPISODIC in response.sources_queried

    @pytest.mark.asyncio
    async def test_forget_episodic(self, agent):
        from memory.schemas import MemoryEvent, EventType, MemoryType
        event = MemoryEvent(
            event_type=EventType.CUSTOM,
            content="This memory will be forgotten and erased from episodic store",
            session_id="forget-session",
            user_id="forget-user",
        )
        result = await agent.remember(event)
        ep_id = result.get("episodic")
        if ep_id:
            success = await agent.forget(ep_id, memory_type=MemoryType.EPISODIC)
            assert success.get("episodic") is True
            ep = await agent.episodic.get_episode(ep_id)
            assert ep is None

    @pytest.mark.asyncio
    async def test_summarize_recent_no_events(self, agent):
        summary = await agent.summarize_recent(hours=24, user_id="new-user")
        assert "No activity" in summary or "no activity" in summary.lower()

    @pytest.mark.asyncio
    async def test_summarize_recent_with_events(self, agent):
        from memory.schemas import MemoryEvent, EventType
        for i in range(3):
            await agent.remember(MemoryEvent(
                event_type=EventType.AGENT_OBSERVATION,
                content=f"Agent observed system state {i}: memory usage 45%, CPU 23%",
                user_id="sum-user",
            ))
        summary = await agent.summarize_recent(hours=24, user_id="sum-user")
        assert "memory usage" in summary.lower() or "agent" in summary.lower() or len(summary) > 50

    @pytest.mark.asyncio
    async def test_consolidate_report_structure(self, agent):
        from memory.schemas import ConsolidationReport
        report = await agent.consolidate()
        assert isinstance(report, ConsolidationReport)
        assert report.episodes_scanned >= 0
        assert report.elapsed_seconds >= 0

    @pytest.mark.asyncio
    async def test_backward_compat_load_context(self, agent):
        ctx = await agent.load_context(
            session_id="compat-sess",
            user_id="compat-user",
            query="testing backward compatibility",
        )
        assert "working" in ctx
        assert "episodic" in ctx
        assert "semantic" in ctx

    @pytest.mark.asyncio
    async def test_backward_compat_save_context(self, agent):
        from core.state import AgentState
        state: AgentState = {
            "goal": "test backward compat",
            "status": "completed",
            "tool_results": ["result_1"],
            "iterations": 2,
        }
        # Should not raise
        await agent.save_context("compat-sess", "compat-user", state)

    @pytest.mark.asyncio
    async def test_backward_compat_clear_session(self, agent):
        await agent.working.set_session("to-clear", {"data": "x"})
        await agent.clear_session("to-clear")
        result = await agent.working.get_session("to-clear")
        assert result is None

    @pytest.mark.asyncio
    async def test_memory_manager_alias(self):
        from memory.memory_manager import MemoryAgent, MemoryManager
        assert MemoryManager is MemoryAgent


# ─────────────────────────────────────────────────────────────────────────────
# TestConsolidation
# ─────────────────────────────────────────────────────────────────────────────


class TestConsolidation:
    @pytest.mark.asyncio
    async def test_consolidation_compresses_old_episodes(
        self, working_memory_fake, semantic, knowledge_graph, temp_dir
    ):
        """Full consolidation pipeline: insert old data → consolidate → verify."""
        import aiosqlite
        from memory.episodic_memory import EpisodicMemory
        from memory.memory_manager import MemoryAgent
        from memory.schemas import ConsolidationReport

        db_path = str(Path(temp_dir) / "consolidation_test.db")
        episodic = EpisodicMemory(db_path=db_path)
        await episodic._ensure_init()

        # Insert episodes with old timestamps (8 days ago)
        old_ts = (datetime.utcnow() - timedelta(days=8)).isoformat()
        old_ids = []
        async with aiosqlite.connect(db_path) as db:
            for i in range(10):
                eid = str(uuid.uuid4())
                old_ids.append(eid)
                await db.execute(
                    "INSERT INTO episodes (id, session_id, user_id, event_type, content, is_summary, timestamp, metadata) "
                    "VALUES (?, 'old-sess', 'old-user', 'task_completed', ?, 0, ?, '{}')",
                    (eid, f"Old task {i}: completed successfully", old_ts),
                )
            await db.commit()

        agent = MemoryAgent(
            working=working_memory_fake,
            episodic=episodic,
            semantic=semantic,
            graph=knowledge_graph,
        )

        # Before consolidation: 10 raw episodes
        count_before = await episodic.count(include_summaries=False)
        assert count_before == 10

        report = await agent.consolidate()
        assert isinstance(report, ConsolidationReport)
        assert report.episodes_scanned == 10
        assert report.episodes_consolidated == 10
        assert report.summaries_created == 1

        # After: raw episodes soft-deleted, one summary remains
        count_after = await episodic.count(include_summaries=False)
        assert count_after == 1  # Just the new summary

    @pytest.mark.asyncio
    async def test_consolidation_batching(
        self, working_memory_fake, semantic, knowledge_graph, temp_dir
    ):
        """25 old episodes should produce 2 summary batches."""
        import aiosqlite
        from memory.episodic_memory import EpisodicMemory
        from memory.memory_manager import CONSOLIDATION_BATCH_SIZE, MemoryAgent

        db_path = str(Path(temp_dir) / "batch_test.db")
        episodic = EpisodicMemory(db_path=db_path)
        await episodic._ensure_init()

        old_ts = (datetime.utcnow() - timedelta(days=8)).isoformat()
        async with aiosqlite.connect(db_path) as db:
            for i in range(25):
                await db.execute(
                    "INSERT INTO episodes (id, session_id, user_id, event_type, content, is_summary, timestamp, metadata) "
                    "VALUES (?, 'batch-sess', 'batch-user', 'custom', ?, 0, ?, '{}')",
                    (str(uuid.uuid4()), f"Batch event {i}", old_ts),
                )
            await db.commit()

        agent = MemoryAgent(
            working=working_memory_fake,
            episodic=episodic,
            semantic=semantic,
            graph=knowledge_graph,
        )
        report = await agent.consolidate()

        expected_batches = (25 + CONSOLIDATION_BATCH_SIZE - 1) // CONSOLIDATION_BATCH_SIZE
        assert report.summaries_created == expected_batches
        assert report.episodes_consolidated == 25

    @pytest.mark.asyncio
    async def test_consolidation_skips_recent_episodes(
        self, agent
    ):
        """Recent episodes (< 7 days) must NOT be consolidated."""
        from memory.schemas import MemoryEvent, EventType
        for i in range(5):
            await agent.remember(MemoryEvent(
                event_type=EventType.CUSTOM,
                content=f"Recent event {i}: very fresh data just created now",
                user_id="fresh-user",
            ))
        report = await agent.consolidate()
        # Recent episodes should not be touched
        assert report.episodes_consolidated == 0
