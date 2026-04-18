"""
api/routes/__init__.py — Route aggregation for the Agentic AI OS API.
"""

from api.routes.agent import router as agent_router
from api.routes.memory import router as memory_router
from api.routes.tools import router as tools_router
from api.routes.session import router as session_router
from api.routes.graph import router as graph_router
from api.routes.tasks import router as tasks_router
from api.routes.agents_status import router as agents_status_router
from api.routes.approval import router as approval_router
from api.routes.audit import router as audit_router
from api.routes.undo import router as undo_router
from api.routes.chat import router as chat_router

__all__ = [
    "agent_router", "memory_router", "tools_router", "session_router", 
    "graph_router", "tasks_router", "agents_status_router", "approval_router", 
    "audit_router", "undo_router", "chat_router"
]
