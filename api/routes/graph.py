"""
api/routes/graph.py — REST endpoints for querying the Code Review Graph.
"""

from __future__ import annotations

from typing import Any, List, Optional
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel

from core.logging_config import get_logger
from core.code_graph import get_code_graph

router = APIRouter()
logger = get_logger(__name__)


# ── Models ────────────────────────────────────────────────────────────────────

class SymbolResponse(BaseModel):
    id: str
    name: str
    type: str
    file_path: str
    line_start: int
    line_end: int
    docstring: Optional[str] = None
    metadata: dict[str, Any] = {}

class ImpactResponse(BaseModel):
    symbol: str
    impacted_symbols_count: int
    impacted_files_count: int
    impacted_files: List[str]
    direct_dependencies: List[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/index", summary="Trigger codebase re-indexing")
async def index_codebase(background_tasks: BackgroundTasks, force: bool = False):
    """Scan and index the entire workspace into the structural Knowledge Graph."""
    graph = get_code_graph()
    
    async def _do_index():
        try:
            await graph.index_codebase(force=force)
        except Exception as exc:
            logger.error("Background indexing failed", error=str(exc))

    background_tasks.add_task(_do_index)
    return {"message": "Indexing started in background."}

@router.get("/search", response_model=List[SymbolResponse], summary="Search symbols by name")
async def search_symbols(q: str = Query(..., min_length=2)):
    """Search for classes, functions, or modules by partial name match."""
    graph = get_code_graph()
    await graph.initialize()
    results = graph.find_symbol(q)
    return results

@router.get("/impact/{symbol_id:path}", response_model=ImpactResponse, summary="Analyze symbol impact")
async def get_impact(symbol_id: str):
    """Determine what components depend on this symbol and might break if it changes."""
    graph = get_code_graph()
    await graph.initialize()
    impact = graph.get_impact_analysis(symbol_id)
    if "error" in impact:
        raise HTTPException(status_code=404, detail=impact["error"])
    return impact

@router.get("/context/{symbol_id:path}", summary="Get symbol relationships")
async def get_context(symbol_id: str):
    """Get immediate callers, callees, and other structural neighbors."""
    graph = get_code_graph()
    await graph.initialize()
    neighbors = graph.get_neighbors(symbol_id)
    if not neighbors and symbol_id not in graph.graph:
        raise HTTPException(status_code=404, detail="Symbol not found")
    return {"symbol_id": symbol_id, "neighbors": neighbors}

@router.get("/stats", summary="Get graph statistics")
async def get_stats():
    """Return counts of nodes and edges in the Knowledge Graph, plus extended stats."""
    graph = get_code_graph()
    await graph.initialize()
    extended_stats = graph.get_graph_stats_extended()
    return {
        "nodes": extended_stats["total_nodes"],
        "edges": extended_stats["total_edges"],
        "db_path": graph.db_path,
        "hubs": extended_stats["hub_nodes"]
    }

@router.post("/blast-radius", summary="Analyze blast radius of changed files")
async def get_blast_radius(files: List[str]):
    """Determine file-level impact for a list of modified files."""
    graph = get_code_graph()
    await graph.initialize()
    results = [graph.get_blast_radius(f) for f in files]
    return {"blast_radius": results}

@router.post("/minimal-context", summary="Get minimal context for review")
async def get_minimal_context(files: List[str]):
    """Return the minimal set of files an AI needs to read for these changes."""
    graph = get_code_graph()
    await graph.initialize()
    context = graph.get_minimal_context(files)
    return context

class ExportMarkdownRequest(BaseModel):
    output_dir: str

@router.post("/export/markdown", summary="Export graph as Markdown Wiki")
async def export_markdown_wiki(request: ExportMarkdownRequest):
    """Generates Markdown files for every module mapped in the internal graph."""
    graph = get_code_graph()
    await graph.initialize()
    try:
        files_created = await graph.generate_markdown_wiki(request.output_dir)
        return {"status": "success", "files_created": files_created, "count": len(files_created)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
