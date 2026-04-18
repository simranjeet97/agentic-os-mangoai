"""
tools/browser_mcp.py — Playwright Headless Browser over MCP.
"""

from typing import Any, Dict, List
from mcp.types import Tool, TextContent

from tools.base import BaseMCPServer
from guardrails.models import ActionType
from agents.web.agent import WebAgent

class BrowserMCP(BaseMCPServer):
    name = "browser"

    def __init__(self):
        super().__init__()
        self.web = WebAgent()

    def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="navigate",
                description="Browse a URL and return its summary and extracted text.",
                inputSchema={
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"]
                }
            ),
            Tool(
                name="click",
                description="Click an element on a page.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "selector": {"type": "string"},
                        "text": {"type": "string"}
                    },
                    "required": ["url"]
                }
            ),
            Tool(
                name="fill_form",
                description="Fill and submit a form on a page.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "fields": {"type": "object", "description": "Key-value map of selectors/labels to values"},
                        "submit": {"type": "string", "description": "Selector for the submit button"}
                    },
                    "required": ["url", "fields"]
                }
            ),
            Tool(
                name="screenshot",
                description="Take a screenshot of a page and save it.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "selector": {"type": "string", "description": "Optional element to screenshot specifically"}
                    },
                    "required": ["url"]
                }
            )
        ]

    def get_action_type(self, tool_name: str, arguments: Dict[str, Any]) -> ActionType:
        return ActionType.NETWORK_REQUEST

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        try:
            state = {"user_id": "mcp-client"}
            if name == "navigate":
                res = await self.web._browse_url({"url": arguments["url"]}, state)
            elif name == "click":
                res = await self.web._click_element(arguments, state)
            elif name == "fill_form":
                res = await self.web._fill_form(arguments, state)
            elif name == "screenshot":
                res = await self.web._screenshot(arguments, state)
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]
                
            out = str(res.get("output", res))
            return [TextContent(type="text", text=out)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {e}")]
