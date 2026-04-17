"""
tools/mcp_server.py — MCP (Model Context Protocol) server host.
Registers and exposes tools to agent clients.
"""

from __future__ import annotations

import asyncio
from typing import Any

from mcp import Server
from mcp.types import TextContent, Tool

from core.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

server = Server("agentic-os-tools")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Expose available tools to MCP clients."""
    return [
        Tool(
            name="web_search",
            description="Search the web using DuckDuckGo",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="read_file",
            description="Read a file from the sandbox filesystem",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to sandbox root"},
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="write_file",
            description="Write content to a file in the sandbox",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        ),
        Tool(
            name="execute_code",
            description="Execute code in the sandbox",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "language": {"type": "string", "enum": ["python", "javascript", "bash"]},
                },
                "required": ["code", "language"],
            },
        ),
        Tool(
            name="run_command",
            description="Execute a shell command in the sandbox",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Route an MCP tool call to the appropriate agent."""
    logger.info("MCP tool called", tool=name, args=list(arguments.keys()))

    try:
        if name == "web_search":
            from agents.web.agent import WebAgent
            result = await WebAgent()._search(
                arguments["query"],
                max_results=arguments.get("max_results", 5),
            )
            return [TextContent(type="text", text=str(result))]

        elif name == "read_file":
            from agents.file.agent import FileAgent
            result = await FileAgent()._read(arguments["path"])
            return [TextContent(type="text", text=str(result.get("output", result)))]

        elif name == "write_file":
            from agents.file.agent import FileAgent
            result = await FileAgent()._write(arguments["path"], arguments["content"])
            return [TextContent(type="text", text=result.get("output", "Written"))]

        elif name == "execute_code":
            from agents.code.agent import CodeAgent
            result = await CodeAgent()._execute_code(
                arguments["code"], arguments.get("language", "python")
            )
            return [TextContent(type="text", text=str(result.get("output", result)))]

        elif name == "run_command":
            from agents.system.agent import SystemAgent
            result = await SystemAgent()._run_in_sandbox(arguments["command"])
            return [TextContent(type="text", text=str(result.get("output", result)))]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as exc:
        logger.error("MCP tool error", tool=name, error=str(exc))
        return [TextContent(type="text", text=f"Error: {exc}")]


async def main() -> None:
    setup_logging()
    from mcp.server.stdio import stdio_server
    async with stdio_server() as streams:
        await server.run(*streams, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
