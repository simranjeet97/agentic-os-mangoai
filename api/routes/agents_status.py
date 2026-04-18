"""
api/routes/agents_status.py — Monitor active agents.
"""

from typing import List
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.state import active_agents
from api.auth.users import get_current_user

router = APIRouter()

class AgentStatus(BaseModel):
    agent_id: str
    agent_type: str
    status: str
    current_task: str

@router.get("/", response_model=List[AgentStatus], summary="List active agents and their status")
async def list_agents(current_user=Depends(get_current_user)):
    """List registered and active agents in the system."""
    result = []
    # For now, return what we have in active_agents or a static list if empty
    if not active_agents:
        return [
            AgentStatus(agent_id="planner-1", agent_type="planner", status="idle", current_task=""),
            AgentStatus(agent_id="executor-1", agent_type="executor", status="idle", current_task=""),
            AgentStatus(agent_id="code-1", agent_type="code", status="idle", current_task=""),
            AgentStatus(agent_id="web-1", agent_type="web", status="idle", current_task="")
        ]
        
    for a_id, info in active_agents.items():
        result.append(AgentStatus(
            agent_id=a_id,
            agent_type=info.get("type", "unknown"),
            status=info.get("status", "unknown"),
            current_task=info.get("current_task", "")
        ))
    return result
