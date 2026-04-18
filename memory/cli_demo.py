"""
memory/cli_demo.py — Interactive CLI demo for the Memory System.

Commands:
    python -m memory.cli_demo remember
    python -m memory.cli_demo recall "query text"
    python -m memory.cli_demo graph
    python -m memory.cli_demo summarize
    python -m memory.cli_demo consolidate

Uses Rich for beautiful terminal output and Typer for the CLI framework.
"""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

app = typer.Typer(
    name="memory-demo",
    help="🧠 Agentic OS Memory System — Interactive Demo",
    add_completion=False,
)
console = Console()

from typing import Any

# ── Sample Data ───────────────────────────────────────────────────────────────

SAMPLE_EVENTS: list[dict[str, Any]] = [
    {
        "event_type": "file_edited",
        "content": "Edited memory/episodic_memory.py to add FTS5 search support",
        "outcome": "File saved successfully. SQLite schema migrated.",
        "session_id": "demo-session-001",
        "user_id": "demo-user",
        "tags": ["python", "sqlite", "memory"],
    },
    {
        "event_type": "command_run",
        "content": "Ran: pytest tests/test_memory.py -v --tb=short",
        "outcome": "12 passed, 0 failed in 3.2s",
        "session_id": "demo-session-001",
        "user_id": "demo-user",
        "tags": ["testing", "pytest"],
    },
    {
        "event_type": "web_search",
        "content": "Searched: sentence-transformers all-MiniLM-L6-v2 performance benchmarks",
        "outcome": "Found 5 relevant results. Model achieves 58.8 on SBERT benchmark.",
        "session_id": "demo-session-001",
        "user_id": "demo-user",
    },
    {
        "event_type": "user_correction",
        "content": "Agent tried to delete ~/.config/nvim/init.lua",
        "outcome": "Action blocked by guardrail engine",
        "correction": "User clarified: only delete temp files in /tmp, never config files",
        "session_id": "demo-session-001",
        "user_id": "demo-user",
        "tags": ["correction", "guardrails"],
    },
    {
        "event_type": "task_completed",
        "content": "Completed: Build full memory system for Agentic AI OS",
        "outcome": "All 5 components implemented. 12 tests passing. CLI demo working.",
        "session_id": "demo-session-001",
        "user_id": "demo-user",
        "tags": ["milestone", "memory-system"],
    },
]


# ── Async Core ────────────────────────────────────────────────────────────────

async def _build_agent():
    """Build MemoryAgent with ephemeral Chroma client for demo."""
    import chromadb

    from memory.chroma_store import SemanticMemory
    from memory.episodic_memory import EpisodicMemory
    from memory.knowledge_graph import KnowledgeGraph
    from memory.memory_manager import MemoryAgent
    from memory.working_memory import WorkingMemory

    # Use ephemeral ChromaDB for demo (no server needed)
    chroma = chromadb.EphemeralClient()
    semantic = SemanticMemory(chroma_client=chroma)

    # Use in-memory SQLite for demo
    episodic = EpisodicMemory(db_path=":memory:")

    working = WorkingMemory()
    graph = KnowledgeGraph(graph_path="/tmp/agentic_demo_graph.json")

    agent = MemoryAgent(
        working=working,
        episodic=episodic,
        semantic=semantic,
        graph=graph,
    )
    return agent


async def _run_remember():
    """Store 5 sample events across all memory tiers."""
    from memory.schemas import MemoryEvent, EventType

    agent = await _build_agent()

    console.print(Panel.fit(
        "📝 Storing 5 sample events across memory tiers...",
        style="bold cyan",
    ))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Event Type", style="cyan")
    table.add_column("Content Preview")
    table.add_column("Tiers Written", style="green")

    for sample in SAMPLE_EVENTS:
        correction = sample.pop("correction", None)
        tags = sample.pop("tags", [])
        event = MemoryEvent(
            event_type=EventType(sample["event_type"]),
            content=sample["content"],
            outcome=sample.get("outcome"),
            session_id=sample.get("session_id", "default"),
            user_id=sample.get("user_id", "default"),
        )
        result = await agent.remember(event)
        table.add_row(
            event.event_type.value,
            event.content[:60] + "...",
            ", ".join(result.keys()),
        )

    console.print(table)
    console.print("\n[bold green]✓ All events stored successfully![/bold green]")


async def _run_recall(query: str):
    """Fan-out recall across all memory tiers."""
    from memory.schemas import MemoryEvent, EventType, MemoryType

    agent = await _build_agent()

    # First store some sample data
    for sample in SAMPLE_EVENTS[:3]:
        sample.pop("correction", None)
        sample.pop("tags", [])
        event = MemoryEvent(
            event_type=EventType(sample["event_type"]),
            content=sample["content"],
            outcome=sample.get("outcome"),
            session_id=sample.get("session_id", "default"),
            user_id=sample.get("user_id", "default"),
        )
        await agent.remember(event)

    console.print(Panel.fit(
        f"🔍 Recalling: [yellow]'{query}'[/yellow]",
        style="bold cyan",
    ))

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True
    ) as progress:
        progress.add_task("Querying all memory tiers...", total=None)
        response = await agent.recall(
            query=query,
            memory_type=MemoryType.ALL,
            top_k=5,
            user_id="demo-user",
        )

    console.print(f"\n[bold]Sources queried:[/bold] {[s.value for s in response.sources_queried]}")
    console.print(f"[bold]Results found:[/bold] {response.total_found}")
    console.print(f"[bold]Elapsed:[/bold] {response.elapsed_ms:.1f}ms\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Source", style="magenta", width=12)
    table.add_column("Relevance", justify="right", width=10)
    table.add_column("Content Preview")

    for r in response.results[:8]:
        table.add_row(
            r.source.value,
            f"{r.relevance:.3f}",
            r.content[:80].replace("\n", " "),
        )

    console.print(table)


async def _run_graph():
    """Display the knowledge graph entities as a rich tree."""
    from memory.schemas import MemoryEvent, EventType

    agent = await _build_agent()

    # Populate graph with sample events
    for sample in SAMPLE_EVENTS:
        sample.pop("correction", None)
        sample.pop("tags", [])
        event = MemoryEvent(
            event_type=EventType(sample["event_type"]),
            content=sample["content"],
            outcome=sample.get("outcome"),
            session_id=sample.get("session_id", "default"),
            user_id=sample.get("user_id", "default"),
        )
        await agent.remember(event)

    stats = await agent.graph.stats()
    top_entities = await agent.graph.get_top_entities(n=15)

    console.print(Panel.fit(
        f"🕸️  Knowledge Graph  |  "
        f"Nodes: [cyan]{stats['nodes']}[/cyan]  "
        f"Edges: [yellow]{stats['edges']}[/yellow]",
        style="bold",
    ))

    tree = Tree("[bold white]📊 Top Entities by Centrality")
    type_subtrees = {}

    for entity in top_entities:
        etype = entity.node_type.value
        if etype not in type_subtrees:
            icons = {"file": "📄", "command": "⚡", "project": "🏗️", "person": "👤", "concept": "💡", "unknown": "❓"}
            icon = icons.get(etype, "•")
            type_subtrees[etype] = tree.add(f"{icon} [bold magenta]{etype.upper()}[/bold magenta]")
        sub = type_subtrees[etype]
        sub.add(
            f"[cyan]{entity.name}[/cyan]  "
            f"[dim](mentions: {entity.mention_count})[/dim]"
        )

    console.print(tree)
    console.print(f"\n[dim]Node types: {stats['node_types']}[/dim]")


async def _run_summarize(hours: int):
    """Show a 24-hour activity summary."""
    from memory.schemas import MemoryEvent, EventType

    agent = await _build_agent()

    for sample in SAMPLE_EVENTS:
        sample.pop("correction", None)
        sample.pop("tags", [])
        event = MemoryEvent(
            event_type=EventType(sample["event_type"]),
            content=sample["content"],
            outcome=sample.get("outcome"),
            session_id=sample.get("session_id", "default"),
            user_id=sample.get("user_id", "default"),
        )
        await agent.remember(event)

    summary = await agent.summarize_recent(hours=hours, user_id="demo-user")

    console.print(Panel(
        summary,
        title=f"[bold cyan]📊 Memory Summary — Last {hours} Hours[/bold cyan]",
        border_style="cyan",
    ))


async def _run_consolidate():
    """Run memory consolidation and show the report."""
    agent = await _build_agent()

    console.print(Panel.fit(
        "🗜️  Running memory consolidation...",
        style="bold yellow",
    ))

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True
    ) as progress:
        progress.add_task("Compressing old episodes...", total=None)
        report = await agent.consolidate()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold yellow")
    table.add_column(style="white")
    table.add_row("Episodes Scanned", str(report.episodes_scanned))
    table.add_row("Episodes Consolidated", str(report.episodes_consolidated))
    table.add_row("Summaries Created", str(report.summaries_created))
    table.add_row("Time Taken", f"{report.elapsed_seconds:.3f}s")
    table.add_row("Errors", str(len(report.errors)) or "None")

    console.print(table)

    if report.errors:
        for err in report.errors:
            console.print(f"[red]✗ {err}[/red]")
    else:
        console.print("\n[bold green]✓ Consolidation complete![/bold green]")


# ── CLI Commands ──────────────────────────────────────────────────────────────

@app.command()
def remember():
    """Store 5 sample events across all memory tiers."""
    asyncio.run(_run_remember())


@app.command()
def recall(
    query: str = typer.Argument("memory system episodic", help="Search query"),
):
    """Search across all memory tiers for a query."""
    asyncio.run(_run_recall(query))


@app.command()
def graph():
    """Display the knowledge graph entities as a tree."""
    asyncio.run(_run_graph())


@app.command()
def summarize(
    hours: int = typer.Option(24, help="Number of hours to summarize"),
):
    """Show a summary of recent activity."""
    asyncio.run(_run_summarize(hours))


@app.command()
def consolidate():
    """Run memory consolidation (compresses old episodic memories)."""
    asyncio.run(_run_consolidate())


@app.command()
def demo():
    """Run the full demo: remember → recall → graph → summarize."""
    console.print(Panel.fit(
        "🧠 [bold]Agentic OS Memory System — Full Demo[/bold]",
        style="bold blue",
        subtitle="all-MiniLM-L6-v2 · Redis · SQLite · ChromaDB · NetworkX",
    ))
    asyncio.run(_run_remember())
    console.rule()
    asyncio.run(_run_recall("file editing and testing"))
    console.rule()
    asyncio.run(_run_graph())
    console.rule()
    asyncio.run(_run_summarize(24))


if __name__ == "__main__":
    app()
