"""
tools/mcp_server.py — Main Entrypoint for Agentic OS MCP capabilities.
"""
from __future__ import annotations

import asyncio
from typing import Any

from mcp import Server
from mcp.types import TextContent, Tool

from core.logging_config import get_logger, setup_logging
from core.tool_registry import ToolRegistry

logger = get_logger(__name__)

server = Server("agentic-os-tools")
registry = ToolRegistry.get_instance()

@server.list_tools()
async def list_tools() -> list[Tool]:
    """Expose all discovered tools from the ToolRegistry to MCP clients."""
    return registry.list_all_tools()

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Route an MCP tool call through GuardrailMiddleware via ToolRegistry."""
    logger.info("MCP tool called", tool=name, args=list(arguments.keys()))
    return await registry.tool_call(name, arguments)

async def main() -> None:
    setup_logging()
    
    # Initialize Code Graph on startup for code-mcp to be ready
    from core.code_graph import get_code_graph
    await get_code_graph().initialize()
    
    from mcp.server.stdio import stdio_server
    async with stdio_server() as streams:
        await server.run(*streams, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
