"""
tools/system_mcp.py — System metrics and control over MCP.
"""
import psutil
import os
from typing import Any, Dict, List
from mcp.types import Tool, TextContent

from tools.base import BaseMCPServer
from guardrails.models import ActionType

class SystemMCP(BaseMCPServer):
    name = "system"

    def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="get_metrics",
                description="Get basic system health metrics (CPU, RAM).",
                inputSchema={"type": "object", "properties": {}}
            ),
            Tool(
                name="list_processes",
                description="List running processes globally.",
                inputSchema={"type": "object", "properties": {}}
            ),
            Tool(
                name="kill_process",
                description="Kill process by PID.",
                inputSchema={
                    "type": "object", 
                    "properties": {"pid": {"type": "integer"}},
                    "required": ["pid"]
                }
            ),
            Tool(
                name="disk_usage",
                description="Get disk usage for a path.",
                inputSchema={"type": "object", "properties": {"path": {"type": "string", "default": "/"}}}
            ),
            Tool(
                name="network_stats",
                description="Get network I/O stats.",
                inputSchema={"type": "object", "properties": {}}
            )
        ]

    def get_action_type(self, tool_name: str, arguments: Dict[str, Any]) -> ActionType:
        if tool_name == "kill_process":
            return ActionType.SHELL_COMMAND
        return ActionType.PROCESS_SPAWN

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        try:
            if name == "get_metrics":
                cpu = psutil.cpu_percent(interval=0.5)
                mem = psutil.virtual_memory()
                return [TextContent(type="text", text=f"CPU: {cpu}%\nRAM: {mem.percent}%")]
                
            elif name == "list_processes":
                procs = []
                for p in psutil.process_iter(['pid', 'name', 'username']):
                    procs.append(f"{p.info['pid']} - {p.info['name']} ({p.info['username']})")
                return [TextContent(type="text", text="\n".join(procs[:50]) + "\n...")]
                
            elif name == "kill_process":
                pid = arguments["pid"]
                os.kill(pid, 9)
                return [TextContent(type="text", text=f"Killed PID {pid}")]
                
            elif name == "disk_usage":
                usage = psutil.disk_usage(arguments.get("path", "/"))
                out = f"Total: {usage.total // 10**9}GB\nUsed: {usage.percent}%"
                return [TextContent(type="text", text=out)]
                
            elif name == "network_stats":
                net = psutil.net_io_counters()
                out = f"Bytes Sent: {net.bytes_sent}\nBytes Recv: {net.bytes_recv}"
                return [TextContent(type="text", text=out)]

            return [TextContent(type="text", text=f"Unknown tool: {name}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error executing {name}: {e}")]
