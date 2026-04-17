"""
memory/knowledge_graph.py — Lightweight NetworkX Knowledge Graph.

Represents entities (people, files, projects, commands, concepts) and their
relationships extracted from documents and agent interactions.

Storage: JSON file at ~/.agentic_os/knowledge_graph.json
Auto-saved on every mutation.

Entity extraction uses regex + heuristics (no LLM required at runtime).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import networkx as nx
from networkx.readwrite import json_graph

from memory.schemas import GraphEdge, GraphNode, GraphSubgraph, NodeType

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
GRAPH_PATH = os.getenv(
    "KNOWLEDGE_GRAPH_PATH",
    str(Path.home() / ".agentic_os" / "knowledge_graph.json"),
)

# ── Entity Extraction Patterns ────────────────────────────────────────────────
_FILE_PATTERN = re.compile(
    r"\b([\w\-./]+\.(?:py|js|ts|json|yaml|yml|md|txt|sh|sql|csv|html|css))\b"
)
_COMMAND_PATTERN = re.compile(
    r"\b(git|python|pip|npm|docker|kubectl|make|bash|curl|wget|ls|cd|grep|awk|sed)\b"
)
_PROJECT_PATTERN = re.compile(
    r"\b([A-Z][a-zA-Z0-9\-]+(?:OS|AI|Agent|App|Service|Bot|API|SDK))\b"
)
_PERSON_PATTERN = re.compile(
    r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b"  # Simple "First Last" heuristic
)


def _extract_entities(text: str) -> list[tuple[str, NodeType]]:
    """
    Extract entities from text using regex heuristics.
    Returns list of (entity_name, node_type) tuples.
    """
    found: list[tuple[str, NodeType]] = []

    for m in _FILE_PATTERN.finditer(text):
        found.append((m.group(1), NodeType.FILE))

    for m in _COMMAND_PATTERN.finditer(text):
        found.append((m.group(1).lower(), NodeType.COMMAND))

    for m in _PROJECT_PATTERN.finditer(text):
        found.append((m.group(1), NodeType.PROJECT))

    for m in _PERSON_PATTERN.finditer(text):
        name = m.group(1)
        # Filter out false positives (common English phrases)
        if name not in {"The First", "The Last", "New York", "Los Angeles"}:
            found.append((name, NodeType.PERSON))

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for item in found:
        if item[0] not in seen:
            seen.add(item[0])
            unique.append(item)
    return unique


class KnowledgeGraph:
    """
    Async-compatible NetworkX knowledge graph for entity/relationship tracking.

    All mutating methods are sync (NetworkX is not async-native) but wrapped
    in asyncio.to_thread where called from async contexts. The public async API
    runs graph operations in a thread pool to avoid blocking.

    Nodes store: id, name, node_type, attributes, mention_count, created_at
    Edges store: relationship, weight, timestamp, metadata
    """

    def __init__(self, graph_path: str = GRAPH_PATH) -> None:
        self._path = Path(graph_path)
        self._graph: nx.DiGraph = nx.DiGraph()
        self._loaded = False
        self._lock = asyncio.Lock()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_sync(self) -> None:
        """Load graph from JSON file (sync, runs in thread pool)."""
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._graph = json_graph.node_link_graph(data)
                logger.info(
                    "Knowledge graph loaded",
                    nodes=self._graph.number_of_nodes(),
                    edges=self._graph.number_of_edges(),
                )
            except Exception as exc:
                logger.warning("Failed to load knowledge graph, starting fresh: %s", exc)
                self._graph = nx.DiGraph()
        self._loaded = True

    def _save_sync(self) -> None:
        """Serialize graph to JSON file (sync, runs in thread pool)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = json_graph.node_link_data(self._graph)
        self._path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    async def _ensure_loaded(self) -> None:
        if not self._loaded:
            async with self._lock:
                if not self._loaded:
                    await asyncio.to_thread(self._load_sync)

    async def _save(self) -> None:
        await asyncio.to_thread(self._save_sync)

    # ── Node Operations ───────────────────────────────────────────────────────

    async def add_entity(
        self,
        name: str,
        node_type: Union[NodeType, str] = NodeType.UNKNOWN,
        attributes: Optional[dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ) -> str:
        """
        Add or update an entity node. If a node with the same name and type
        already exists, its mention_count is incremented. Returns node_id.
        """
        # Coerce string to NodeType
        if isinstance(node_type, str):
            try:
                node_type = NodeType(node_type)
            except ValueError:
                node_type = NodeType.UNKNOWN
        await self._ensure_loaded()

        # Check for existing node by name + type
        existing_id = self._find_node_id(name, node_type)
        if existing_id:
            async with self._lock:
                node = self._graph.nodes[existing_id]
                node["mention_count"] = node.get("mention_count", 1) + 1
                if attributes:
                    node.setdefault("attributes", {}).update(attributes)
            await self._save()
            return existing_id

        nid = node_id or str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{node_type}:{name}"))
        async with self._lock:
            self._graph.add_node(
                nid,
                name=name,
                node_type=node_type.value,
                attributes=attributes or {},
                mention_count=1,
                created_at=datetime.utcnow().isoformat(),
            )
        await self._save()
        logger.debug("Entity added: name=%s, node_type=%s, node_id=%s", name, node_type.value, nid)
        return nid

    async def get_entity(self, node_id: str) -> Optional[GraphNode]:
        """Retrieve a node by its ID."""
        await self._ensure_loaded()
        if node_id not in self._graph.nodes:
            return None
        data = self._graph.nodes[node_id]
        return GraphNode(
            node_id=node_id,
            name=data.get("name", ""),
            node_type=NodeType(data.get("node_type", "unknown")),
            attributes=data.get("attributes", {}),
            created_at=datetime.fromisoformat(
                data.get("created_at", datetime.utcnow().isoformat())
            ),
            mention_count=data.get("mention_count", 1),
        )

    async def remove_entity(self, node_id: str) -> None:
        """Remove a node and all its edges."""
        await self._ensure_loaded()
        async with self._lock:
            if node_id in self._graph:
                self._graph.remove_node(node_id)
        await self._save()

    # ── Edge Operations ───────────────────────────────────────────────────────

    async def add_relationship(
        self,
        source_name: str,
        target_name: str,
        relationship: str,
        source_type: NodeType = NodeType.UNKNOWN,
        target_type: NodeType = NodeType.UNKNOWN,
        weight: float = 1.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Add a directed relationship edge between two entities.
        Entities are created if they don't exist.
        If the edge already exists, its weight is incremented.
        """
        src_id = await self.add_entity(source_name, source_type)
        dst_id = await self.add_entity(target_name, target_type)

        async with self._lock:
            if self._graph.has_edge(src_id, dst_id):
                self._graph[src_id][dst_id]["weight"] = (
                    self._graph[src_id][dst_id].get("weight", 1.0) + weight
                )
            else:
                self._graph.add_edge(
                    src_id,
                    dst_id,
                    relationship=relationship,
                    weight=weight,
                    timestamp=datetime.utcnow().isoformat(),
                    metadata=metadata or {},
                )
        await self._save()

    async def get_related(
        self,
        entity_name: str,
        node_type: Optional[NodeType] = None,
        depth: int = 2,
    ) -> GraphSubgraph:
        """
        Return subgraph of nodes related to the given entity within `depth` hops.
        """
        await self._ensure_loaded()
        nid = self._find_node_id(entity_name, node_type)
        if not nid or nid not in self._graph:
            return GraphSubgraph(root_entity=entity_name)

        # BFS up to `depth` hops
        visited = {nid}
        frontier = {nid}
        for _ in range(depth):
            next_frontier = set()
            for n in frontier:
                next_frontier.update(self._graph.successors(n))
                next_frontier.update(self._graph.predecessors(n))
            frontier = next_frontier - visited
            visited.update(frontier)

        sub = self._graph.subgraph(visited)
        nodes = [await self.get_entity(n) for n in sub.nodes if n in self._graph.nodes]
        edges = [
            GraphEdge(
                source_id=u,
                target_id=v,
                relationship=sub[u][v].get("relationship", "related"),
                weight=sub[u][v].get("weight", 1.0),
                timestamp=datetime.fromisoformat(
                    sub[u][v].get("timestamp", datetime.utcnow().isoformat())
                ),
            )
            for u, v in sub.edges
        ]
        return GraphSubgraph(
            nodes=[n for n in nodes if n is not None],
            edges=edges,
            root_entity=entity_name,
        )

    async def find_path(
        self, source_name: str, target_name: str
    ) -> list[str]:
        """Return the shortest path between two entities (entity names)."""
        await self._ensure_loaded()
        src = self._find_node_id(source_name)
        dst = self._find_node_id(target_name)
        if not src or not dst:
            return []
        try:
            path_ids = nx.shortest_path(self._graph, src, dst)
            return [
                self._graph.nodes[nid].get("name", nid)
                for nid in path_ids
            ]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    async def extract_and_add(
        self,
        text: str,
        source_entity: Optional[str] = None,
        source_type: NodeType = NodeType.CONCEPT,
    ) -> list[str]:
        """
        Extract entities from text and add them to the graph.
        Optionally link all extracted entities to a `source_entity`.
        Returns list of node_ids added.
        """
        entities = _extract_entities(text)
        node_ids = []
        for name, node_type in entities:
            nid = await self.add_entity(name, node_type)
            node_ids.append(nid)
            if source_entity:
                await self.add_relationship(
                    source_name=source_entity,
                    target_name=name,
                    relationship="mentions",
                    source_type=source_type,
                    target_type=node_type,
                )
        return node_ids

    # ── Analytics ─────────────────────────────────────────────────────────────

    async def get_top_entities(self, n: int = 10) -> list[GraphNode]:
        """Return top-N entities by degree centrality (combined in + out degree)."""
        await self._ensure_loaded()
        if self._graph.number_of_nodes() == 0:
            return []
        centrality = nx.degree_centrality(self._graph)
        top_ids = sorted(centrality, key=centrality.get, reverse=True)[:n]  # type: ignore
        result = []
        for nid in top_ids:
            node = await self.get_entity(nid)
            if node:
                result.append(node)
        return result

    async def stats(self) -> dict[str, Any]:
        """Return summary statistics about the knowledge graph."""
        await self._ensure_loaded()
        g = self._graph
        type_counts: dict[str, int] = defaultdict(int)
        for _, data in g.nodes(data=True):
            type_counts[data.get("node_type", "unknown")] += 1
        return {
            "nodes": g.number_of_nodes(),
            "edges": g.number_of_edges(),
            "node_types": dict(type_counts),
            "is_directed": g.is_directed(),
            "density": round(nx.density(g), 6) if g.number_of_nodes() > 1 else 0.0,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _find_node_id(
        self, name: str, node_type: Optional[NodeType] = None
    ) -> Optional[str]:
        """Find a node ID by name (and optionally type). Case-insensitive."""
        name_lower = name.lower()
        for nid, data in self._graph.nodes(data=True):
            if data.get("name", "").lower() == name_lower:
                if node_type is None or data.get("node_type") == node_type.value:
                    return nid
        return None

    async def to_dict(self) -> dict[str, Any]:
        """Export the full graph as a JSON-serializable dict."""
        await self._ensure_loaded()
        return json_graph.node_link_data(self._graph)
