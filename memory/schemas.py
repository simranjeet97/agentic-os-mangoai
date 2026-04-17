"""
memory/schemas.py — Shared Pydantic schemas for the Memory System.

All memory components use these models for type safety and serialization.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class MemoryType(str, Enum):
    """Which memory tier to target for a recall or forget operation."""

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    GRAPH = "graph"
    ALL = "all"


class NodeType(str, Enum):
    """Types of entities stored in the Knowledge Graph."""

    PERSON = "person"
    FILE = "file"
    PROJECT = "project"
    COMMAND = "command"
    CONCEPT = "concept"
    UNKNOWN = "unknown"


class EventType(str, Enum):
    """Types of episodic events that agents can record."""

    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    FILE_EDITED = "file_edited"
    COMMAND_RUN = "command_run"
    WEB_SEARCH = "web_search"
    USER_CORRECTION = "user_correction"
    AGENT_OBSERVATION = "agent_observation"
    MEMORY_CONSOLIDATION = "memory_consolidation"
    CUSTOM = "custom"


# ─────────────────────────────────────────────────────────────────────────────
# Core Event Model
# ─────────────────────────────────────────────────────────────────────────────


class MemoryEvent(BaseModel):
    """An event to be stored in the memory system."""

    event_type: EventType = EventType.CUSTOM
    content: str = Field(..., description="Main content / description of the event")
    outcome: Optional[str] = Field(None, description="Result or outcome of the event")
    correction: Optional[str] = Field(None, description="User correction if applicable")
    session_id: str = Field(default="default", description="Session identifier")
    user_id: str = Field(default="default", description="User identifier")
    agent_id: Optional[str] = Field(None, description="Agent that generated this event")
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tags: list[str] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be empty")
        return v.strip()

    @field_serializer("timestamp")
    def serialize_timestamp(self, dt: datetime, _info: Any) -> str:
        return dt.isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Recall / Search Results
# ─────────────────────────────────────────────────────────────────────────────


class RecallResult(BaseModel):
    """A single result returned from a memory recall operation."""

    item_id: str
    content: str
    source: MemoryType
    relevance: float = Field(default=1.0, ge=0.0, le=1.0)
    timestamp: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecallResponse(BaseModel):
    """Aggregated response from MemoryAgent.recall()."""

    query: str
    results: list[RecallResult] = Field(default_factory=list)
    sources_queried: list[MemoryType] = Field(default_factory=list)
    total_found: int = 0
    elapsed_ms: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Episodic Memory Schema
# ─────────────────────────────────────────────────────────────────────────────


class EpisodeRecord(BaseModel):
    """A record retrieved from EpisodicMemory."""

    episode_id: str
    session_id: str
    user_id: str
    event_type: str
    content: str
    outcome: Optional[str] = None
    correction: Optional[str] = None
    is_summary: bool = False
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Graph Schemas
# ─────────────────────────────────────────────────────────────────────────────


class GraphNode(BaseModel):
    """An entity node in the Knowledge Graph."""

    node_id: str
    name: str
    node_type: NodeType = NodeType.UNKNOWN
    attributes: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    mention_count: int = 1


class GraphEdge(BaseModel):
    """A relationship edge between two Knowledge Graph nodes."""

    source_id: str
    target_id: str
    relationship: str
    weight: float = Field(default=1.0, ge=0.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphSubgraph(BaseModel):
    """A sub-graph returned from traversal queries."""

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    root_entity: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Consolidation Summary
# ─────────────────────────────────────────────────────────────────────────────


class ConsolidationReport(BaseModel):
    """Report produced after a memory consolidation run."""

    episodes_scanned: int = 0
    episodes_consolidated: int = 0
    summaries_created: int = 0
    elapsed_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    errors: list[str] = Field(default_factory=list)
