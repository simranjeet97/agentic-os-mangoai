"""
tools/notes_mcp.py — Interact with local markdown notes.
"""
import os
import aiofiles
from pathlib import Path
from typing import Any, Dict, List
from mcp.types import Tool, TextContent

from tools.base import BaseMCPServer
from guardrails.models import ActionType

class NotesMCP(BaseMCPServer):
    name = "notes"

    def __init__(self):
        super().__init__()
        self.notes_dir = Path(os.getcwd()) / "notes"
        self.notes_dir.mkdir(exist_ok=True)

    def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="search_notes",
                description="Full-text search in markdown notes.",
                inputSchema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            ),
            Tool(
                name="create_note",
                description="Create or overwrite a markdown note.",
                inputSchema={
                    "type": "object",
                    "properties": {"title": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["title", "content"]
                }
            ),
            Tool(
                name="append_to_note",
                description="Append text to an existing markdown note.",
                inputSchema={
                    "type": "object",
                    "properties": {"title": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["title", "content"]
                }
            )
        ]

    def get_action_type(self, tool_name: str, arguments: Dict[str, Any]) -> ActionType:
        if tool_name in ("create_note", "append_to_note"):
            return ActionType.FILE_WRITE
        return ActionType.FILE_READ

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        try:
            if name == "search_notes":
                query = arguments["query"].lower()
                matches = []
                for p in self.notes_dir.glob("*.md"):
                    content = p.read_text().lower()
                    if query in content:
                        matches.append(p.name)
                out = "Found in: " + ", ".join(matches) if matches else "No notes found matching query."
                return [TextContent(type="text", text=out)]
                
            elif name == "create_note":
                path = self.notes_dir / f"{arguments['title']}.md"
                async with aiofiles.open(path, "w") as f:
                    await f.write(arguments['content'])
                return [TextContent(type="text", text=f"Note created at {path.name}")]
                
            elif name == "append_to_note":
                path = self.notes_dir / f"{arguments['title']}.md"
                async with aiofiles.open(path, "a") as f:
                    await f.write("\n" + arguments['content'])
                return [TextContent(type="text", text=f"Appended to {path.name}")]
                
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {e}")]
