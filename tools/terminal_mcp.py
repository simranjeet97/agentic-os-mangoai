"""
tools/terminal_mcp.py — Terminal and Code Execution over MCP.
"""

from typing import Any, Dict, List
from mcp.types import Tool, TextContent

from tools.base import BaseMCPServer
from guardrails.models import ActionType
from agents.executor.agent import ExecutorAgent

class TerminalMCP(BaseMCPServer):
    name = "terminal"

    def __init__(self):
        super().__init__()
        self.executor = ExecutorAgent()

    def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="safe_execute",
                description="Execute script safely using ExecutorAgent sandbox. Supports python, bash, javascript, go.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "language": {"type": "string", "default": "bash"}
                    },
                    "required": ["code"]
                }
            ),
            Tool(
                name="run_command",
                description="Run shell command in sandbox.",
                inputSchema={
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"]
                }
            )
        ]

    def get_action_type(self, tool_name: str, arguments: Dict[str, Any]) -> ActionType:
        if tool_name == "safe_execute":
            return ActionType.CODE_EXECUTION
        return ActionType.SHELL_COMMAND

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        try:
            if name == "safe_execute":
                res = await self.executor.safe_execute(
                    code=arguments["code"],
                    language=arguments.get("language", "bash")
                )
            elif name == "run_command":
                res = await self.executor.safe_execute(
                    code=arguments["command"],
                    language="bash"
                )
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]
                
            stdout = res.get('stdout', '')
            stderr = res.get('stderr', '')
            explanation = res.get('explanation', '')
            
            output = f"Exit Code: {res.get('exit_code')}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}\n\nExplanation:\n{explanation}"
            return [TextContent(type="text", text=output.strip())]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
