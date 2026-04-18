"""
tools/memory_mcp.py — Expose Memory subsystem over MCP.
"""
from typing import Any, Dict, List
from mcp.types import Tool, TextContent

from tools.base import BaseMCPServer
from guardrails.models import ActionType
from memory.memory_manager import MemoryAgent

class MemoryMCP(BaseMCPServer):
    name = "memory"

    def __init__(self):
        super().__init__()
        self.memory = MemoryAgent(agent_id="mcp-client")

    def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="remember",
                description="Store information into working and semantic memory.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "key": {"type": "string", "description": "Optional key for working memory"}
                    },
                    "required": ["content"]
                }
            ),
            Tool(
                name="recall",
                description="Recall information from semantic memory.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "default": 5}
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="get_recent",
                description="Get recent episodic events.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "default": 10}
                    }
                }
            )
        ]

    def get_action_type(self, tool_name: str, arguments: Dict[str, Any]) -> ActionType:
        if tool_name == "remember":
            return ActionType.MEMORY_WRITE
        return ActionType.MEMORY_READ

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        try:
            step = {"action": name}
            step.update(arguments)
            state = {"user_id": "mcp-client", "session_id": "mcp-session"}
            
            res = await self.memory.execute(step, state)
            return [TextContent(type="text", text=str(res.get("output", res)))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {e}")]
