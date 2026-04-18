"""
api/routes/chat.py — Main chat entry point with streaming response.
"""

import json
import uuid
import asyncio
from typing import Optional, List
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.auth.users import get_current_user
from core.orchestrator_graph import get_orchestrator_graph
from core.session_manager import get_session_manager
from api.routes.agent import _serialize

router = APIRouter()

class ChatAttachment(BaseModel):
    filename: str
    content_type: str
    content: str  # Base64 encoded or raw text

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    attachments: List[ChatAttachment] = Field(default_factory=list)

@router.post("/", summary="Submit chat message and stream agent thoughts")
async def chat_stream(request: ChatRequest, http_request: Request, current_user=Depends(get_current_user)):
    """
    Submit a message to the agentic OS and stream its thought process,
    tool calls, trace information, and final output via SSE.
    """
    sm = get_session_manager()
    session = await sm.get_or_create(current_user.username)
    session_id = request.session_id or session.session_id
    
    task_id = str(uuid.uuid4())
    session.current_task_id = task_id
    await sm.update(session)
    
    metadata = {
        "attachments": [a.model_dump() for a in request.attachments],
        "task_id": task_id
    }

    async def _event_generator():
        graph = get_orchestrator_graph()
        
        try:
            # Yield initial task event
            yield f"data: {json.dumps({'event': 'started', 'task_id': task_id, 'session_id': session_id})}\n\n"
            
            async for event in graph.stream(
                user_input=request.message,
                user_id=current_user.username,
                session_id=session_id,
                metadata=metadata
            ):
                if await http_request.is_disconnected():
                    break
                    
                node = event.get("node", "")
                update = event.get("update", {})
                
                # Format to include agent_trace explicitly
                response_event = {
                    "event": "node_update",
                    "node": node,
                    "task_id": task_id,
                    "agent_trace": {
                        "active_agent": update.get("active_agent"),
                        "reason": f"Routing to {update.get('active_agent')} based on orchestration plan."
                    },
                    "data": _serialize(update)
                }
                
                yield f"data: {json.dumps(response_event)}\n\n"
                
            yield f"data: {json.dumps({'event': 'complete', 'task_id': task_id})}\n\n"
            
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            yield f"data: {json.dumps({'event': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(_event_generator(), media_type="text/event-stream")
