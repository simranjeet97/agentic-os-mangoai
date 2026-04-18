"""
tools/base.py — Base interface for MCP Servers.
"""

from mcp.types import Tool, TextContent
from typing import Any, Dict, List
from guardrails.models import ActionType

class BaseMCPServer:
    """Abstract base class for individual sub-MCP servers."""
    name: str = "base"

    def list_tools(self) -> List[Tool]:
        """List all tools exposed by this server."""
        raise NotImplementedError

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Execute the tool implementation."""
        raise NotImplementedError

    def get_action_type(self, tool_name: str, arguments: Dict[str, Any]) -> ActionType:
        """
        Return the guardrail ActionType for a given tool. By default returns UNKNOWN,
        subclasses should override this.
        """
        return ActionType.UNKNOWN
