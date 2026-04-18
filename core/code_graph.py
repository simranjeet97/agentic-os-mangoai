"""
core/code_graph.py — Codebase Knowledge Graph Engine.

Indexes the codebase into a structural Knowledge Graph (nodes=symbols, edges=calls/imports).
Provides tools for impact analysis, symbol search, and relationship mapping.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiofiles
import aiosqlite
import networkx as nx
from core.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class SymbolNode:
    """A node representing a code symbol (Class, Function, Var)."""
    id: str                  # Unique ID (e.g. "core.code_graph:CodeGraphManager.index")
    name: str                # Symbol name
    type: str                # "class", "function", "module", "async_function"
    file_path: str           # Absolute path to file
    line_start: int
    line_end: int
    docstring: Optional[str] = None
    hash: Optional[str] = None # For incremental indexing
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SymbolEdge:
    """An edge representing a relationship (calls, inherits, imports)."""
    source: str
    target: str
    type: str                # "calls", "inherits", "imports", "contains"
    metadata: Dict[str, Any] = field(default_factory=dict)


class CodeGraphManager:
    """
    Manages the codebase Knowlege Graph.
    Uses networkx for in-memory graph ops and SQLite for persistence.
    """

    def __init__(self, workspace_root: str, db_path: Optional[str] = None):
        self.root = Path(workspace_root).resolve()
        self.db_path = db_path or str(self.root / ".gemini" / "code_graph.db")
        self.graph = nx.DiGraph()
        self._initialized = False

    async def initialize(self):
        """Ensure the .gemini directory and database exist."""
        if self._initialized:
            return
        
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS symbols (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    type TEXT,
                    file_path TEXT,
                    line_start INTEGER,
                    line_end INTEGER,
                    docstring TEXT,
                    hash TEXT,
                    metadata TEXT
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    source TEXT,
                    target TEXT,
                    type TEXT,
                    metadata TEXT,
                    PRIMARY KEY (source, target, type)
                )
            """)
            await db.commit()
            
        await self._load_graph()
        self._initialized = True
        logger.info("CodeGraph initialized", db_path=self.db_path)

    async def _load_graph(self):
        """Load symbols and edges from SQLite into networkx."""
        self.graph.clear()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT * FROM symbols") as cursor:
                async for row in cursor:
                    meta = json.loads(row[8]) if row[8] else {}
                    self.graph.add_node(
                        row[0],
                        name=row[1],
                        type=row[2],
                        file_path=row[3],
                        line_start=row[4],
                        line_end=row[5],
                        docstring=row[6],
                        hash=row[7],
                        **meta
                    )
            
            async with db.execute("SELECT * FROM edges") as cursor:
                async for row in cursor:
                    meta = json.loads(row[3]) if row[3] else {}
                    self.graph.add_edge(row[0], row[1], type=row[2], **meta)
        
        logger.info("Loaded graph into memory", nodes=self.graph.number_of_nodes(), edges=self.graph.number_of_edges())

    async def index_codebase(self, force: bool = False):
        """
        Scan the workspace and update the graph.
        Ignores hidden dirs, virtualenvs, and __pycache__.
        """
        await self.initialize()
        
        ignore_dirs = {".git", ".venv", "__pycache__", "node_modules", ".gemini", "dist", "build", "test_env", "venv", "env"}
        
        current_files = []
        for p in self.root.rglob("*.py"):
            if any(part in ignore_dirs for part in p.parts):
                continue
            current_files.append(p)

        logger.info("Scanning codebase", files_count=len(current_files))

        tasks = [self._index_file(f, force) for f in current_files]
        results = await asyncio.gather(*tasks)
        
        # Merge results and save
        new_nodes = []
        new_edges = []
        for nodes, edges in results:
            new_nodes.extend(nodes)
            new_edges.extend(edges)

        await self._save_to_db(new_nodes, new_edges)
        await self._load_graph()
        logger.info("Indexing complete", new_nodes=len(new_nodes), new_edges=len(new_edges))

    async def _index_file(self, path: Path, force: bool) -> Tuple[List[SymbolNode], List[SymbolEdge]]:
        """Parse a single file and extract symbols and relationships."""
        try:
            async with aiofiles.open(path, "r", encoding="utf-8", errors="replace") as f:
                content = await f.read()
            
            file_hash = hashlib.sha256(content.encode()).hexdigest()
            rel_path = str(path.relative_to(self.root))
            module_name = rel_path.replace("/", ".").replace(".py", "")

            # Check if file has changed
            if not force:
                existing_hash = self.graph.nodes.get(module_name, {}).get("hash")
                if existing_hash == file_hash:
                    return [], []

            tree = ast.parse(content)
            visitor = SymbolVisitor(module_name, rel_path, content)
            visitor.visit(tree)
            
            return visitor.nodes, visitor.edges
        except Exception as exc:
            logger.error("Failed to index file", path=str(path), error=str(exc))
            return [], []

    async def _save_to_db(self, nodes: List[SymbolNode], edges: List[SymbolEdge]):
        """Batch save symbols and edges to SQLite."""
        async with aiosqlite.connect(self.db_path) as db:
            # Update symbols
            for n in nodes:
                await db.execute("""
                    INSERT OR REPLACE INTO symbols (id, name, type, file_path, line_start, line_end, docstring, hash, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (n.id, n.name, n.type, n.file_path, n.line_start, n.line_end, n.docstring, n.hash, json.dumps(n.metadata)))
            
            # Update edges
            for e in edges:
                await db.execute("""
                    INSERT OR REPLACE INTO edges (source, target, type, metadata)
                    VALUES (?, ?, ?, ?)
                """, (e.source, e.target, e.type, json.dumps(e.metadata)))
            
            await db.commit()

    # ── Query API ─────────────────────────────────────────────────────────────

    def find_symbol(self, query: str) -> List[Dict[str, Any]]:
        """Search for symbols by name (case-insensitive partial match)."""
        results = []
        q = query.lower()
        for node, data in self.graph.nodes(data=True):
            if q in node.lower() or q in data.get("name", "").lower():
                results.append({"id": node, **data})
        return results

    def get_neighbors(self, symbol_id: str, direction: str = "both") -> List[Dict[str, Any]]:
        """Find related symbols (calls, imports, etc)."""
        if symbol_id not in self.graph:
            return []
        
        neighbors = []
        if direction in ("out", "both"):
            for target in self.graph.successors(symbol_id):
                edge_data = self.graph.get_edge_data(symbol_id, target)
                neighbors.append({
                    "id": target,
                    "relation": edge_data.get("type"),
                    "direction": "outgoing",
                    **self.graph.nodes[target]
                })
        
        if direction in ("in", "both"):
            for source in self.graph.predecessors(symbol_id):
                edge_data = self.graph.get_edge_data(source, symbol_id)
                neighbors.append({
                    "id": source,
                    "relation": edge_data.get("type"),
                    "direction": "incoming",
                    **self.graph.nodes[source]
                })
        
        return neighbors

    def get_impact_analysis(self, symbol_id: str) -> Dict[str, Any]:
        """Determine what depends on this symbol (upstream impacts)."""
        if symbol_id not in self.graph:
            return {"error": "Symbol not found"}
        
        # All nodes that eventually reach this symbol
        upstream = nx.ancestors(self.graph, symbol_id)
        
        impacted_files = set()
        for node in upstream:
            file_path = self.graph.nodes[node].get("file_path")
            if file_path:
                impacted_files.add(file_path)
        
        return {
            "symbol": symbol_id,
            "impacted_symbols_count": len(upstream),
            "impacted_files_count": len(impacted_files),
            "impacted_files": list(impacted_files),
            "direct_dependencies": [n["id"] for n in self.get_neighbors(symbol_id, "in")]
        }

    def get_blast_radius(self, file_path: str) -> Dict[str, Any]:
        """Determine what might break if this file changes."""
        impacted_files = set()
        impacted_symbols = set()
        
        # Find all symbols in this file
        file_symbols = [n for n, data in self.graph.nodes(data=True) if data.get("file_path") == file_path]
        
        for sym in file_symbols:
            upstream = nx.ancestors(self.graph, sym)
            impacted_symbols.update(upstream)
            for n in upstream:
                fp = self.graph.nodes[n].get("file_path")
                if fp:
                    impacted_files.add(fp)

        return {
            "changed_file": file_path,
            "impacted_files": list(impacted_files),
            "impact_count": len(impacted_files)
        }

    def get_minimal_context(self, changed_files: List[str]) -> Dict[str, Any]:
        """Return minimal review context (the impacted files + metadata) for an AI assistant."""
        all_impacted = set()
        for f in changed_files:
            br = self.get_blast_radius(f)
            all_impacted.update(br.get("impacted_files", []))
            
        return {
            "minimal_files_to_read": list(all_impacted)
        }

    def get_graph_stats_extended(self) -> Dict[str, Any]:
        """Identify hub nodes and architecture communities."""
        degree_dict = dict(self.graph.degree())
        hubs = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "hub_nodes": hubs
        }

    async def generate_markdown_wiki(self, output_dir: str) -> List[str]:
        """Export the code graph as a set of interconnected Markdown files."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Group nodes by file_path
        files_map = {}
        for node, data in self.graph.nodes(data=True):
            fp = data.get("file_path")
            if not fp:
                continue
            if fp not in files_map:
                files_map[fp] = []
            files_map[fp].append((node, data))
            
        generated_files = []
        
        # Generate an index
        index_content = ["# Codebase Knowledge Graph Wiki\n\n## Modules\n"]
        for fp in sorted(files_map.keys()):
            # Rel path if possible
            try:
                rel = str(Path(fp).relative_to(self.root))
            except ValueError:
                rel = str(fp)
                
            md_filename = rel.replace("/", "_").replace(".py", ".md")
            index_content.append(f"- [{rel}](./{md_filename})")
            
            # Module file
            module_content = [f"# Module: `{rel}`\n", f"Path: `{fp}`\n\n## Symbols\n"]
            for node, data in sorted(files_map[fp], key=lambda x: x[1].get("line_start", 0)):
                name = data.get("name", node)
                sym_type = data.get("type", "unknown")
                doc = data.get("docstring") or "No documentation."
                module_content.append(f"### `{name}` ({sym_type})\n")
                module_content.append(f"**Lines**: {data.get('line_start')} - {data.get('line_end')}\n")
                module_content.append(f"{doc}\n")
                
                # Outgoing / Incoming
                out_edges = list(self.graph.successors(node))
                in_edges = list(self.graph.predecessors(node))
                
                if out_edges:
                    module_content.append("\n**Calls/Uses**:\n")
                    for e in out_edges:
                        module_content.append(f"- `{e}`\n")
                if in_edges:
                    module_content.append("\n**Used By**:\n")
                    for e in in_edges:
                        module_content.append(f"- `{e}`\n")
                module_content.append("\n---\n")
                
            out_file = out_path / md_filename
            async with aiofiles.open(out_file, "w") as f:
                await f.write("\n".join(module_content))
            generated_files.append(str(out_file))
            
        # Write Index
        index_file = out_path / "README.md"
        async with aiofiles.open(index_file, "w") as f:
            await f.write("\n".join(index_content))
        generated_files.append(str(index_file))
        
        return generated_files


class SymbolVisitor(ast.NodeVisitor):
    """AST visitor to extract symbols and edges."""

    def __init__(self, module_id: str, file_path: str, content: str):
        self.module_id = module_id
        self.file_path = file_path
        self.content = content
        self.nodes: List[SymbolNode] = []
        self.edges: List[SymbolEdge] = []
        self.scope_stack: List[str] = [module_id]
        
        # Initial module node
        self.nodes.append(SymbolNode(
            id=module_id,
            name=module_id.split(".")[-1],
            type="module",
            file_path=file_path,
            line_start=1,
            line_end=len(content.splitlines()),
            docstring=ast.get_docstring(ast.parse(content))
        ))

    def _get_current_scope(self) -> str:
        return self.scope_stack[-1]

    def visit_ClassDef(self, node: ast.ClassDef):
        class_id = f"{self._get_current_scope()}:{node.name}"
        doc = ast.get_docstring(node)
        
        self.nodes.append(SymbolNode(
            id=class_id,
            name=node.name,
            type="class",
            file_path=self.file_path,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            docstring=doc
        ))
        
        self.edges.append(SymbolEdge(self._get_current_scope(), class_id, "contains"))
        
        # Base classes (inheritance)
        for base in node.bases:
            if isinstance(base, ast.Name):
                self.edges.append(SymbolEdge(class_id, base.id, "inherits"))
        
        self.scope_stack.append(class_id)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._handle_function(node, "function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._handle_function(node, "async_function")

    def _handle_function(self, node: Any, type_str: str):
        func_id = f"{self._get_current_scope()}.{node.name}"
        doc = ast.get_docstring(node)
        
        self.nodes.append(SymbolNode(
            id=func_id,
            name=node.name,
            type=type_str,
            file_path=self.file_path,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            docstring=doc,
            metadata={"args": [arg.arg for arg in node.args.args]}
        ))
        
        self.edges.append(SymbolEdge(self._get_current_scope(), func_id, "contains"))
        
        self.scope_stack.append(func_id)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_Call(self, node: ast.Call):
        # Extract the name of the function being called
        target_name = None
        if isinstance(node.func, ast.Name):
            target_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            target_name = node.func.attr
        
        if target_name:
            # Try to resolve to local function if possible
            local_id = f"{self.module_id}.{target_name}"
            # We don't know for sure if it's local, but we'll add an edge to the Name
            # The query engine can handle resolving these.
            self.edges.append(SymbolEdge(self._get_current_scope(), target_name, "calls"))
        
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.edges.append(SymbolEdge(self.module_id, alias.name, "imports"))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            for alias in node.names:
                full_name = f"{node.module}.{alias.name}"
                self.edges.append(SymbolEdge(self.module_id, full_name, "imports"))
        self.generic_visit(node)


# ── Global Access ─────────────────────────────────────────────────────────────

_manager: Optional[CodeGraphManager] = None

def get_code_graph(workspace_root: Optional[str] = None) -> CodeGraphManager:
    global _manager
    if _manager is None:
        root = workspace_root or os.getcwd()
        _manager = CodeGraphManager(root)
    return _manager
