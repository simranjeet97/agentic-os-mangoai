"""
core/shell.py — NaturalLanguageShell: the main user-facing entry point.

Entry points:
  • CLI:  python -m core.shell
  • API:  from core.shell import NaturalLanguageShell; shell = NaturalLanguageShell()
          await shell.chat("do X then Y")

Features:
  - Streaming output: partial results printed as agents work
  - Stop/resume detection ("stop", "resume")
  - Context switching ("let's do something else instead")
  - Rich terminal UI (colours, spinners, panels)
  - Full async throughout
"""

from __future__ import annotations

import asyncio
import os
import signal
import uuid
from typing import Optional

# ── Optional rich import (graceful fallback) ──────────────────────────────────
try:
    from rich.console import Console
    from rich.markup import escape
    from rich.markdown import Markdown
    from rich.panel import Panel
    _RICH = True
except ImportError:
    _RICH = False
    Console = None  # type: ignore[assignment,misc]

from core.logging_config import get_logger, setup_logging
from core.session_manager import get_session_manager, SessionManager

logger = get_logger(__name__)

BANNER = r"""
 ▄▄▄       ▄████▄  ▓█████  ███▄    █ ▄▄▄█████▓ ██▓ ▄████▄      ▒█████    ██████
▒████▄    ▒██▀ ▀█  ▓█   ▀  ██ ▀█   █ ▓  ██▒ ▓▒▓██▒▒██▀ ▀█     ▒██▒  ██▒▒██    ▒
▒██  ▀█▄  ▒▓█    ▄ ▒███   ▓██  ▀█ ██▒▒ ▓██░ ▒░▒██▒▒▓█    ▄    ▒██░  ██▒░ ▓██▄
░██▄▄▄▄██ ▒▓▓▄ ▄██▒▒▓█  ▄ ▓██▒  ▐▌██▒░ ▓██▓ ░ ░██░▒▓▓▄ ▄██▒   ▒██   ██░  ▒   ██▒
 ▓█   ▓██▒▒ ▓███▀ ░░▒████▒▒██░   ▓██░  ▒██▒ ░ ░██░▒ ▓███▀ ░   ░ ████▓▒░▒██████▒▒
 ▒▒   ▓▒█░░ ░▒ ▒  ░░░ ▒░ ░░ ▒░   ▒ ▒   ▒ ░░   ░▓  ░ ░▒ ▒  ░   ░ ▒░▒░▒░ ▒ ▒▓▒ ▒ ░
  ▒   ▒▒ ░  ░  ▒    ░ ░  ░░ ░░   ░ ▒░    ░     ▒ ░  ░  ▒        ░ ▒ ▒░ ░ ░▒  ░ ░
  ░   ▒   ░           ░      ░   ░ ░   ░       ▒ ░░           ░ ░ ░ ▒  ░  ░  ░
      ░  ░░ ░         ░  ░         ░           ░  ░ ░             ░ ░        ░
          ░                                       ░
             Agentic AI OS  •  NaturalLanguageShell  •  v0.1
"""

HELP_TEXT = """\
Commands:
  <any text>       →  process as an agent task
  stop / cancel    →  stop the current running task
  resume           →  resume an interrupted task
  status           →  show session info
  graph            →  manage codebase graph (index/stats/tree)
  history          →  show turn history
  clear            →  clear the screen
  help             →  show this message
  exit / quit      →  exit the shell
"""


# ── NaturalLanguageShell ──────────────────────────────────────────────────────


class NaturalLanguageShell:
    """
    Main entry point for the Agentic OS.

    User types (or speaks) a command → IntentParser → OrchestratorGraph → response.
    Streaming output: partial results are printed as each agent node completes.
    """

    def __init__(
        self,
        user_id: str = "cli-user",
        stream: bool = True,
    ) -> None:
        setup_logging()
        self.user_id = user_id
        self._stream = stream
        self._session_manager: SessionManager = get_session_manager()
        self._session_id: Optional[str] = None
        self._current_task: Optional[asyncio.Task] = None
        self._console = Console() if _RICH else None
        logger.info("NaturalLanguageShell initialised", user_id=user_id)

    # ── Public API ────────────────────────────────────────────────────────────

    async def start_session(self) -> str:
        """Create (or reuse) a session for this user. Returns session_id."""
        session = await self._session_manager.get_or_create(self.user_id)
        self._session_id = session.session_id
        logger.info("Session ready", session_id=self._session_id)
        return self._session_id

    async def chat(self, user_input: str) -> str:
        """
        Process one user message end-to-end.
        Returns the assistant's final text response.
        """
        if self._session_id is None:
            await self.start_session()

        # ── Built-in commands ─────────────────────────────────────────────────
        cmd = user_input.strip().lower()

        if cmd in ("exit", "quit", "bye"):
            return "__EXIT__"

        if cmd == "help":
            return HELP_TEXT

        if cmd == "clear":
            os.system("clear" if os.name != "nt" else "cls")
            return ""

        if cmd == "status":
            return await self._handle_status()

        if cmd == "history":
            return await self._handle_history()

        if cmd.startswith("graph"):
            return await self._handle_graph(user_input)

        # ── Stop signal ───────────────────────────────────────────────────────
        if SessionManager.detect_stop_signal(user_input):
            return await self._handle_stop()

        # ── Resume signal ─────────────────────────────────────────────────────
        if SessionManager.detect_resume_signal(user_input):
            return await self._handle_resume()

        # ── Normal task / conversation ────────────────────────────────────────
        if self._stream:
            return await self._streaming_chat(user_input)
        return await self._blocking_chat(user_input)

    # ── Streaming chat ────────────────────────────────────────────────────────

    async def _streaming_chat(self, user_input: str) -> str:
        """Stream node-by-node updates as the graph runs."""
        from core.orchestrator_graph import get_orchestrator_graph

        graph = get_orchestrator_graph()
        final_response = ""

        self._print_user(user_input)

        node_labels = {
            "parse_intent":            "🔍 Parsing intent",
            "route_to_agent":          "🗺️  Building plan",
            "execute_with_guardrails": "⚙️  Executing",
            "update_memory":           "💾 Saving memory",
            "respond_to_user":         "💬 Composing response",
        }

        try:
            async for event in graph.stream(
                user_input=user_input,
                user_id=self.user_id,
                session_id=self._session_id,
            ):
                node = event.get("node", "")
                update = event.get("update", {})

                label = node_labels.get(node, f"→ {node}")
                self._print_node_update(label, update)

                # Capture the final assistant message
                msgs = update.get("messages", [])
                for m in msgs:
                    content = m.content if hasattr(m, "content") else str(m)
                    final_response = content   # last one wins

        except asyncio.CancelledError:
            return "⛔ Task cancelled."
        except Exception as exc:
            logger.error("Streaming error", error=str(exc), exc_info=True)
            return f"❌ Error: {exc}"

        if not final_response:
            final_response = "✅ Done."

        self._print_assistant(final_response)
        return final_response

    # ── Blocking chat ─────────────────────────────────────────────────────────

    async def _blocking_chat(self, user_input: str) -> str:
        """Non-streaming: run graph to completion then return answer."""
        from core.orchestrator_graph import get_orchestrator_graph
        graph = get_orchestrator_graph()
        self._print_user(user_input)
        self._print_thinking()

        state = await graph.run(
            user_input=user_input,
            user_id=self.user_id,
            session_id=self._session_id,
        )

        msgs = state.get("messages", [])
        answer = ""
        for m in reversed(msgs):
            if hasattr(m, "content") and m.content:
                answer = m.content
                break
        if not answer:
            answer = "✅ Done."
        self._print_assistant(answer)
        return answer

    # ── Built-in command handlers ─────────────────────────────────────────────

    async def _handle_stop(self) -> str:
        if self._session_id:
            await self._session_manager.interrupt(
                self._session_id,
                reason="User issued stop command",
            )
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
        return "⛔ Task stopped. Say 'resume' to pick up where you left off."

    async def _handle_resume(self) -> str:
        if not self._session_id:
            return "No active session to resume."
        snap = await self._session_manager.resume(self._session_id)
        if snap is None:
            return "Nothing to resume. Start a new task."
        return (
            f"▶️  Resuming task (step {snap.current_step} of the last plan).\n"
            f"Original request: {snap.user_input}"
        )

    async def _handle_status(self) -> str:
        if not self._session_id:
            return "No active session."
        session = await self._session_manager.get(self._session_id)
        if session is None:
            return "Session expired."
        return (
            f"📊 Session status\n"
            f"  ID         : {session.session_id[:16]}…\n"
            f"  User       : {session.user_id}\n"
            f"  Status     : {session.status.value}\n"
            f"  Task       : {session.current_task_id or 'none'}\n"
            f"  Interrupted: {session.interrupt_flag}\n"
            f"  Turns      : {len(session.history)}\n"
            f"  Last active: {session.last_active.strftime('%H:%M:%S')}"
        )

    async def _handle_history(self) -> str:
        if not self._session_id:
            return "No active session."
        session = await self._session_manager.get(self._session_id)
        if not session or not session.history:
            return "No conversation history yet."
        lines = []
        for turn in session.history[-10:]:
            role = "You" if turn["role"] == "user" else "Agent"
            content = turn["content"][:100]
            lines.append(f"  [{role}] {content}")
        return "\n".join(lines)

    async def _handle_graph(self, user_input: str) -> str:
        """Handle graph command: index, stats, tree, search."""
        parts = user_input.strip().split()
        if len(parts) == 1:
            return "Usage: graph [index|stats|tree <name>|search <name>]"
        
        sub = parts[1].lower()
        from core.code_graph import get_code_graph
        graph = get_code_graph()

        if sub == "index":
            self._print_thinking()
            await graph.index_codebase(force="--force" in parts)
            return f"✅ Codebase indexed: {graph.graph.number_of_nodes()} nodes, {graph.graph.number_of_edges()} edges."

        if sub == "stats":
            await graph.initialize()
            return (
                f"📊 Code Review Graph Stats\n"
                f"  Nodes   : {graph.graph.number_of_nodes()}\n"
                f"  Edges   : {graph.graph.number_of_edges()}\n"
                f"  Database: {graph.db_path}"
            )

        if sub == "tree":
            if len(parts) < 3: return "Usage: graph tree <symbol_name>"
            name = parts[2]
            await graph.initialize()
            matches = graph.find_symbol(name)
            if not matches: return f"Symbol '{name}' not found."
            
            target = matches[0]["id"]
            lines = [f"🌲 Call Tree for {target}:"]
            neighbors = graph.get_neighbors(target, direction="out")
            for n in neighbors:
                if n["relation"] == "calls":
                    lines.append(f"  └── {n['id']}")
            return "\n".join(lines)

        if sub == "search":
            if len(parts) < 3: return "Usage: graph search <query>"
            await graph.initialize()
            results = graph.find_symbol(parts[2])
            if not results: return "No matches found."
            lines = [f"🔍 Search results for '{parts[2]}':"]
            for r in results[:10]:
                lines.append(f"  - {r['id']} ({r['type']})")
            return "\n".join(lines)

        return f"Unknown graph subcommand: {sub}"

    # ── Rich printing helpers ─────────────────────────────────────────────────

    def _print_user(self, text: str) -> None:
        if _RICH and self._console:
            self._console.print(f"\n[bold cyan]You[/bold cyan]: {escape(text)}")
        else:
            print(f"\nYou: {text}")

    def _print_assistant(self, text: str) -> None:
        if _RICH and self._console:
            # Use Markdown rendering for the assistant's response
            md = Markdown(text)
            self._console.print(
                Panel(md, title="[bold green]🤖 Agent[/bold green]", border_style="green")
            )
        else:
            print(f"\nAgent: {text}\n")

    def _print_node_update(self, label: str, update: dict) -> None:
        """Print a brief one-liner for each node as it fires."""
        status = update.get("status", "")
        msgs = update.get("messages", [])
        extra = ""
        if msgs:
            last_content = msgs[-1].content if hasattr(msgs[-1], "content") else str(msgs[-1])
            extra = f" — {last_content[:200]}"
        if status:
            extra = f" [{status}]{extra}"
        if _RICH and self._console:
            self._console.print(f"  [dim]{label}{extra}[/dim]")
        else:
            print(f"  {label}{extra}")

    def _print_thinking(self) -> None:
        if _RICH and self._console:
            self._console.print("  [dim]🤔 Thinking…[/dim]")
        else:
            print("  Thinking…")


# ── Interactive CLI loop ─────────────────────────────────────────────────────


async def _cli_loop(user_id: str, stream: bool) -> None:
    """The main interactive REPL loop."""
    shell = NaturalLanguageShell(user_id=user_id, stream=stream)
    await shell.start_session()

    if _RICH:
        console = Console()
        console.print(f"[bold magenta]{BANNER}[/bold magenta]")
        console.print("[dim]Type 'help' for commands, 'exit' to quit.[/dim]\n")
    else:
        print(BANNER)
        print("Type 'help' for commands, 'exit' to quit.\n")

    while True:
        try:
            if _RICH:
                from rich.prompt import Prompt
                user_input = Prompt.ask("[bold yellow]>[/bold yellow]").strip()
            else:
                user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        response = await shell.chat(user_input)

        if response == "__EXIT__":
            if _RICH:
                Console().print("[bold]Goodbye! 👋[/bold]")
            else:
                print("Goodbye!")
            break


def _handle_sigint(signum, frame):
    """Graceful Ctrl-C: let the running task finish cleanly."""
    print("\n[Ctrl-C] Sending stop signal… (press again to force exit)")
    asyncio.get_event_loop().create_task(
        get_session_manager().cleanup_expired()
    )


# ── __main__ entry point ──────────────────────────────────────────────────────


def main() -> None:
    """
    CLI entry point.
    Run: python -m core.shell [--no-stream] [--user USER_ID]
    """
    import argparse
    parser = argparse.ArgumentParser(description="Agentic OS — NaturalLanguageShell")
    parser.add_argument("--user", default=os.getenv("AGENTIC_USER_ID", f"user-{uuid.uuid4().hex[:6]}"))
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_sigint)

    asyncio.run(_cli_loop(user_id=args.user, stream=not args.no_stream))


if __name__ == "__main__":
    main()
