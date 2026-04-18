"""
tools/filesystem_mcp.py — Filesystem interactions over MCP.
"""

import os
import aiofiles
from pathlib import Path
from typing import Any, Dict, List
from mcp.types import Tool, TextContent

from tools.base import BaseMCPServer
from guardrails.models import ActionType


class FileSystemMCP(BaseMCPServer):
    name = "filesystem"

    def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="read_file",
                description="Read exact file contents. Input path is relative or absolute.",
                inputSchema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }
            ),
            Tool(
                name="write_file",
                description="Write content to file.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["path", "content"]
                }
            ),
            Tool(
                name="list_dir",
                description="List directory contents.",
                inputSchema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }
            ),
            Tool(
                name="search_files",
                description="Search files by name.",
                inputSchema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "path": {"type": "string", "default": "."}},
                    "required": ["query"]
                }
            ),
            Tool(
                name="get_file_info",
                description="Get file metadata.",
                inputSchema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }
            )
        ]

    def get_action_type(self, tool_name: str, arguments: Dict[str, Any]) -> ActionType:
        if tool_name == "write_file":
            return ActionType.FILE_WRITE
        return ActionType.FILE_READ

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        try:
            if name == "read_file":
                async with aiofiles.open(arguments["path"], "r") as f:
                    content = await f.read()
                return [TextContent(type="text", text=content)]
            elif name == "write_file":
                # Ensure directory exists
                Path(arguments["path"]).parent.mkdir(parents=True, exist_ok=True)
                async with aiofiles.open(arguments["path"], "w") as f:
                    await f.write(arguments["content"])
                return [TextContent(type="text", text=f"Written to {arguments['path']}")]
            elif name == "list_dir":
                items = os.listdir(arguments["path"])
                return [TextContent(type="text", text="\n".join(items))]
            elif name == "search_files":
                root = Path(arguments.get("path", "."))
                query = arguments["query"]
                results = [str(p) for p in root.rglob(f"*{query}*")]
                out = "\n".join(results) if results else "No matches"
                return [TextContent(type="text", text=out)]
            elif name == "get_file_info":
                stat = os.stat(arguments["path"])
                info = f"Size: {stat.st_size} bytes\nModified: {stat.st_mtime}"
                return [TextContent(type="text", text=info)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
        return [TextContent(type="text", text=f"Unknown tool: {name}")]
