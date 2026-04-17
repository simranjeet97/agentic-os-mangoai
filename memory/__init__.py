"""
memory/__init__.py — Public API for the Agentic AI OS Memory System.

Four-tier memory architecture:
  WorkingMemory  → Redis, TTL-based active session context
  EpisodicMemory → SQLite, timestamped event log with corrections
  SemanticMemory → ChromaDB, vector-indexed documents + cosine search
  KnowledgeGraph → NetworkX, entity/relationship graph

Coordinated by MemoryAgent (alias: MemoryManager for backward compat).
"""

from memory.schemas import (
    ConsolidationReport,
    EpisodeRecord,
    EventType,
    GraphEdge,
    GraphNode,
    GraphSubgraph,
    MemoryEvent,
    MemoryType,
    NodeType,
    RecallResponse,
    RecallResult,
)
from memory.embeddings import EmbeddingService, get_embedding_service
from memory.working_memory import WorkingMemory, RedisStore
from memory.episodic_memory import EpisodicMemory
from memory.chroma_store import SemanticMemory, ChromaStore
from memory.knowledge_graph import KnowledgeGraph
from memory.memory_manager import MemoryAgent, MemoryManager

__all__ = [
    # Schemas
    "MemoryEvent",
    "MemoryType",
    "EventType",
    "NodeType",
    "RecallResult",
    "RecallResponse",
    "EpisodeRecord",
    "GraphNode",
    "GraphEdge",
    "GraphSubgraph",
    "ConsolidationReport",
    # Embedding
    "EmbeddingService",
    "get_embedding_service",
    # Memory tiers
    "WorkingMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "KnowledgeGraph",
    # Coordinator
    "MemoryAgent",
    # Backward-compat aliases
    "MemoryManager",
    "RedisStore",
    "ChromaStore",
]
