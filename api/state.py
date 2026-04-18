"""
api/state.py — Centralized in-memory state for the FastAPI backend.
"""

from typing import Any, Dict

# Centralized task store
# Maps task_id -> task_data dict
task_store: Dict[str, Dict[str, Any]] = {}

# Centralized SSE queues
# Maps task_id -> asyncio.Queue
sse_queues: Dict[str, Any] = {}

# Centralized pending approvals
# Maps action_id -> approval_data dict
pending_approvals: Dict[str, Dict[str, Any]] = {}

# Active agents tracking
# Maps agent_name -> status/task dict
active_agents: Dict[str, Dict[str, Any]] = {}
