"""
api/routes/audit.py — Audit log query endpoints.
"""

from typing import Optional
from datetime import datetime
from fastapi import APIRouter, Query, Depends, HTTPException

from api.auth.users import get_current_user
from guardrails.audit import get_audit_logger

router = APIRouter()

@router.get("/", summary="Query the audit log")
async def query_audit_log(
    agent: Optional[str] = Query(None, description="Filter by agent id"),
    action_type: Optional[str] = Query(None, description="Filter by action type (e.g., FileWrite, DockerRun)"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level (low, medium, high, critical)"),
    date_from: Optional[datetime] = Query(None, description="Start date (ISO format)"),
    date_to: Optional[datetime] = Query(None, description="End date (ISO format)"),
    limit: int = Query(50, le=1000),
    current_user=Depends(get_current_user)
):
    """Query the audit log with combined filters."""
    audit_logger = get_audit_logger()
    
    try:
        # Fetch matching logs 
        # Using inner method if public api is not comprehensive enough or just filtering in memory for demo
        logs = await audit_logger.get_logs(
            agent_id=agent,
            action_type=action_type,
            level=risk_level,
            limit=limit,
            since=date_from
        )
        # Apply date_to filter if provided as get_logs might only support 'since'
        if date_to:
            logs = [log for log in logs if datetime.fromisoformat(log["timestamp"].replace("Z", "+00:00")) <= date_to]
            
        return {"count": len(logs), "logs": logs}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
