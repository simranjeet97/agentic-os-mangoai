"""
tools/calendar_mcp.py — Read local ICS calendar files.
"""

from typing import Any, Dict, List
from mcp.types import Tool, TextContent

from tools.base import BaseMCPServer
from guardrails.models import ActionType

class CalendarMCP(BaseMCPServer):
    name = "calendar"

    def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="get_events",
                description="List upcoming calendar events",
                inputSchema={
                    "type": "object",
                    "properties": {"days": {"type": "integer", "default": 7}}
                }
            ),
            Tool(
                name="add_event",
                description="Add an event to ICS (mock).",
                inputSchema={
                    "type": "object",
                    "properties": {"title": {"type": "string"}, "date": {"type": "string"}}
                }
            )
        ]

    def get_action_type(self, tool_name: str, arguments: Dict[str, Any]) -> ActionType:
        if tool_name == "add_event":
            return ActionType.FILE_WRITE
        return ActionType.FILE_READ

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        if name == "get_events":
            return [TextContent(type="text", text="Calendar support is minimal right now. No events found.")]
        elif name == "add_event":
            return [TextContent(type="text", text=f"Added {arguments.get('title')} to mock calendar.")]
        return [TextContent(type="text", text=f"Unknown tool: {name}")]
