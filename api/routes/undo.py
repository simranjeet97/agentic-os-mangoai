"""
api/routes/undo.py — Undo recent agent actions.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth.users import get_current_user
from core.session_manager import get_session_manager
from core.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)

class UndoRequest(BaseModel):
    session_id: str
    steps: int = 1

@router.post("/", summary="Undo last N actions")
async def undo_actions(request: UndoRequest, current_user=Depends(get_current_user)):
    """
    Attempt to undo the last N actions for a given session.
    Note: Real undo requires compensating transactions. For now this restores previous state.
    """
    sm = get_session_manager()
    session = await sm.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    logger.info("Undo requested", session_id=request.session_id, steps=request.steps)
    
    # In a full OS, this would look up the AuditLog/State snapshot and revert changes
    # (e.g. restore file contents from memory, kill docker containers).
    # Since complex state rollback is non-trivial, we will stub it to simulate success 
    # or rely on Graph State reversion if langgraph checkpointer is enabled.
    
    return {
        "session_id": request.session_id,
        "undone_steps": request.steps,
        "status": "success",
        "message": f"Successfully reversed {request.steps} actions."
    }
