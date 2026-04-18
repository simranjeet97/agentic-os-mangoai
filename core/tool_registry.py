"""
core/tool_registry.py — Discovers and registers all MCP tool servers.
Pipelines tool calls through GuardrailMiddleware.
"""

from __future__ import annotations

import importlib
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.types import Tool, TextContent

from tools.base import BaseMCPServer
from guardrails.middleware import GuardrailMiddleware
from guardrails.models import AgentAction
from core.logging_config import get_logger

logger = get_logger("tool_registry")

class ToolRegistry:
    """Singleton registry for all MCP sub-servers."""
    _instance: Optional["ToolRegistry"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._servers: Dict[str, BaseMCPServer] = {}
        self._tool_to_server: Dict[str, BaseMCPServer] = {}
        self._guardrails = GuardrailMiddleware()

    @classmethod
    def get_instance(cls) -> "ToolRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    cls._instance._auto_discover()
        return cls._instance

    def register(self, server: BaseMCPServer) -> None:
        self._servers[server.name] = server
        for tool in server.list_tools():
            self._tool_to_server[tool.name] = server
        logger.info("Registered MCP Server", server=server.name)

    def list_all_tools(self) -> List[Tool]:
        tools = []
        for server in self._servers.values():
            tools.extend(server.list_tools())
        return tools

    async def tool_call(self, tool_name: str, arguments: Dict[str, Any], agent_id: str = "mcp-client") -> List[TextContent]:
        server = self._tool_to_server.get(tool_name)
        if not server:
            return [TextContent(type="text", text=f"Unknown tool: {tool_name}")]
            
        action_type = server.get_action_type(tool_name, arguments)
        
        target_paths = []
        if "path" in arguments:
             target_paths.append(arguments["path"])
             
        action = AgentAction(
            agent_id=agent_id,
            action_type=action_type,
            target_paths=target_paths,
            raw_input=f"Tool call: {tool_name} with {arguments}",
            metadata={"tool_name": tool_name, "arguments": arguments} # type: ignore
        )

        try:
            # Check guardrails
            await self._guardrails.evaluate_action(action)
            return await server.call_tool(tool_name, arguments)
        except Exception as e:
            logger.error("Execution blocked by guardrails or execution error", tool=tool_name, error=str(e))
            return [TextContent(type="text", text=f"Error executing {tool_name}: {e}")]

    def _auto_discover(self) -> None:
        tools_dir = Path(__file__).parent.parent / "tools"
        if not tools_dir.exists():
            return
            
        SKIP_FILES = {"base.py", "mcp_server.py", "__init__.py"}
        for file in tools_dir.glob("*_mcp.py"):
            if file.name in SKIP_FILES:
                continue
                
            module_path = f"tools.{file.stem}"
            try:
                mod = importlib.import_module(module_path)
                for attr_name in dir(mod):
                    obj = getattr(mod, attr_name)
                    if isinstance(obj, type) and issubclass(obj, BaseMCPServer) and obj is not BaseMCPServer:
                        self.register(obj())
            except Exception as exc:
                logger.warning("Failed to load MCP server", module_path=module_path, error=str(exc))
