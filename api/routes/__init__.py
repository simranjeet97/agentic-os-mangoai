"""
api/routes/__init__.py — Route aggregation for the Agentic AI OS API.
"""

from api.routes.agent import router as agent_router
from api.routes.memory import router as memory_router
from api.routes.tools import router as tools_router

__all__ = ["agent_router", "memory_router", "tools_router"]
