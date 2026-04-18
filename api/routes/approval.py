"""
api/routes/approval.py — Endpoints for approving or rejecting high-risk actions.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from api.state import pending_approvals
from api.auth.users import get_current_user
from core.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)

class ApprovalDecision(BaseModel):
    approved: bool
    feedback: str = ""

@router.get("/", summary="List pending approvals")
async def list_pending_approvals(current_user=Depends(get_current_user)):
    """List all actions currently waiting for approval."""
    return {"approvals": list(pending_approvals.values())}

@router.post("/{action_id}", summary="Approve or reject a pending action")
async def process_approval(action_id: str, decision: ApprovalDecision, current_user=Depends(get_current_user)):
    """Provide approval or rejection for a high-risk action."""
    if action_id not in pending_approvals:
        raise HTTPException(status_code=404, detail="Pending action not found")
        
    approval_record = pending_approvals[action_id]
    
    # Store the decision (the orchestrator/guardrails engine should poll this or wait on an Event)
    approval_record["status"] = "approved" if decision.approved else "rejected"
    approval_record["feedback"] = decision.feedback
    
    logger.info("Processed approval", action_id=action_id, approved=decision.approved)
    return {"action_id": action_id, "status": approval_record["status"]}
