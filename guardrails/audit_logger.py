"""
guardrails/audit_logger.py — Tamper-evident audit logging for all agent actions.
Writes structured JSON logs with HMAC signatures.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from core.logging_config import get_logger

logger = get_logger(__name__)

AUDIT_LOG_PATH = Path(os.getenv("LOG_DIR", "logs")) / "audit.jsonl"
AUDIT_SECRET = os.getenv("AUDIT_HMAC_SECRET", "change-me-audit-secret").encode()


class AuditLogger:
    """
    Write signed audit records for all agent actions.
    Each record is a JSON line with an HMAC-SHA256 signature.
    """

    def __init__(self) -> None:
        AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    def _sign(self, payload: str) -> str:
        return hmac.new(AUDIT_SECRET, payload.encode(), hashlib.sha256).hexdigest()

    async def log_action(
        self,
        event_type: str,
        user_id: str,
        task_id: str,
        agent: str,
        details: dict,
        outcome: str = "success",
    ) -> None:
        """Write a signed audit record."""
        record = {
            "audit_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
            "user_id": user_id,
            "task_id": task_id,
            "agent": agent,
            "outcome": outcome,
            "details": details,
        }
        payload = json.dumps(record, sort_keys=True)
        record["signature"] = self._sign(payload)

        try:
            with open(AUDIT_LOG_PATH, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.error("Failed to write audit log", error=str(exc))

    def verify_record(self, record: dict) -> bool:
        """Verify the HMAC signature of an audit record."""
        sig = record.pop("signature", None)
        if not sig:
            return False
        payload = json.dumps(record, sort_keys=True)
        expected = self._sign(payload)
        return hmac.compare_digest(sig, expected)
