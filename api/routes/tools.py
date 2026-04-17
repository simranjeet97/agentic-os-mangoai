"""
api/routes/tools.py — Direct tool invocation endpoints.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)


class ToolRequest(BaseModel):
    tool: str          # e.g. "web_search", "file_read", "code_execute"
    params: dict[str, Any] = {}
    user_id: str = "anonymous"


@router.post("/execute", summary="Execute a tool directly")
async def execute_tool(request: ToolRequest):
    """
    Directly invoke a registered tool without full agent orchestration.
    Useful for lightweight, targeted operations.
    """
    tool_name = request.tool
    params = request.params
    logger.info("Direct tool invocation", tool=tool_name, user_id=request.user_id)

    try:
        if tool_name == "web_search":
            from agents.web.agent import WebAgent
            agent = WebAgent()
            result = await agent._search(params.get("query", ""))
            return {"tool": tool_name, "result": result}

        elif tool_name == "file_read":
            from agents.file.agent import FileAgent
            agent = FileAgent()
            result = await agent._read(params.get("path", ""))
            return {"tool": tool_name, "result": result}

        elif tool_name == "code_generate":
            from agents.code.agent import CodeAgent
            agent = CodeAgent()
            result = await agent._generate_code(
                params.get("description", ""),
                params.get("language", "python"),
            )
            return {"tool": tool_name, "result": result}

        else:
            raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name}")

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Tool execution failed", tool=tool_name, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/list", summary="List available tools")
async def list_tools():
    """Return all available tools and their descriptions."""
    return {
        "tools": [
            {"name": "web_search", "description": "Search the web via DuckDuckGo", "params": ["query"]},
            {"name": "file_read", "description": "Read a file from the sandbox", "params": ["path"]},
            {"name": "code_generate", "description": "Generate code using LLM", "params": ["description", "language"]},
        ]
    }
