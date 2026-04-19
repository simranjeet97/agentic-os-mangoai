"""
guardrails/audit_logger.py — AuditLogger (SQLite, append-only)

Tamper-resistant, append-only audit log using SQLite. Every agent action
is recorded with timestamp, agent_id, action_type, risk_score, approved_by,
outcome, and a HMAC-SHA256 signature. An INSERT trigger prevents UPDATE/DELETE.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_AUDIT_DB_PATH = Path(os.getenv("AUDIT_DB_PATH", "logs/audit.db"))
_AUDIT_HMAC_SECRET = os.getenv("AUDIT_HMAC_SECRET", "change-me-audit-secret").encode()

# DDL — append-only enforced via INSTEAD OF trigger
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    audit_id    TEXT UNIQUE NOT NULL,
    timestamp   TEXT NOT NULL,
    agent_id    TEXT NOT NULL,
    action_type TEXT NOT NULL,
    risk_score  REAL NOT NULL DEFAULT 0.0,
    approved_by TEXT,
    outcome     TEXT NOT NULL,
    details     TEXT,
    hmac_sig    TEXT NOT NULL
);
"""

# Prevent any UPDATE on the table
_CREATE_UPDATE_GUARD_SQL = """
CREATE TRIGGER IF NOT EXISTS prevent_audit_update
BEFORE UPDATE ON audit_log
BEGIN
    SELECT RAISE(ABORT, 'audit_log is append-only: UPDATE is forbidden');
END;
"""

# Prevent any DELETE on the table
_CREATE_DELETE_GUARD_SQL = """
CREATE TRIGGER IF NOT EXISTS prevent_audit_delete
BEFORE DELETE ON audit_log
BEGIN
    SELECT RAISE(ABORT, 'audit_log is append-only: DELETE is forbidden');
END;
"""

_INSERT_SQL = """
INSERT INTO audit_log
    (audit_id, timestamp, agent_id, action_type, risk_score, approved_by, outcome, details, hmac_sig)
VALUES
    (?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


class AuditRecord:
    """Structured representation of a single audit entry."""

    __slots__ = (
        "audit_id", "timestamp", "agent_id", "action_type",
        "risk_score", "approved_by", "outcome", "details", "hmac_sig",
    )

    def __init__(
        self,
        audit_id: str,
        timestamp: str,
        agent_id: str,
        action_type: str,
        risk_score: float,
        outcome: str,
        details: dict[str, Any],
        approved_by: Optional[str] = None,
        hmac_sig: str = "",
    ) -> None:
        self.audit_id = audit_id
        self.timestamp = timestamp
        self.agent_id = agent_id
        self.action_type = action_type
        self.risk_score = risk_score
        self.approved_by = approved_by
        self.outcome = outcome
        self.details = details
        self.hmac_sig = hmac_sig

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "action_type": self.action_type,
            "risk_score": self.risk_score,
            "approved_by": self.approved_by,
            "outcome": self.outcome,
            "details": self.details,
        }


class AuditLogger:
    """
    Append-only SQLite audit log.

    Every agent action is written as an immutable row signed with HMAC-SHA256.
    Triggers prevent UPDATE and DELETE — any tampering raises a sqlite3 error.

    Usage:
        logger = AuditLogger()
        audit_id = await logger.log(
            agent_id="agent-001",
            action_type="shell_command",
            risk_score=4.5,
            outcome="success",
            details={"command": "ls -la"},
        )
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = Path(db_path) if db_path else _AUDIT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Compatibility methods
    # ------------------------------------------------------------------

    async def get_logs(
        self,
        agent_id: Optional[str] = None,
        action_type: Optional[str] = None,
        level: Optional[str] = None,
        limit: int = 100,
        since: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch logs with the interface expected by the API routes.
        Maps 'level' to 'risk_score' and filters by date.
        """
        records = self.query(
            agent_id=agent_id,
            action_type=action_type,
            limit=limit,
        )

        if level:
            # Simple level -> score mapping
            score_map = {"low": 3.0, "medium": 5.0, "high": 7.0, "critical": 9.0}
            target_score = score_map.get(level.lower(), 0.0)
            records = [r for r in records if r.get("risk_score", 0.0) >= target_score]

        if since:
            since_iso = since.isoformat()
            records = [r for r in records if r.get("timestamp", "") >= since_iso]

        return records

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def log(
        self,
        agent_id: str,
        action_type: str,
        risk_score: float,
        outcome: str,
        details: Optional[dict[str, Any]] = None,
        approved_by: Optional[str] = None,
        audit_id: Optional[str] = None,
    ) -> str:
        """
        Write an immutable audit record.

        Args:
            agent_id:    Identifier of the agent performing the action.
            action_type: Type of action (e.g. "shell_command", "file_write").
            risk_score:  Computed risk score 0-10.
            outcome:     One of: success | blocked | pending | error | rolled_back.
            details:     Arbitrary JSON-serialisable metadata.
            approved_by: None = auto-approved; "human:<user_id>" = manual approval.
            audit_id:    Optional pre-generated UUID (generated if omitted).

        Returns:
            The audit_id of the written record.
        """
        aid = audit_id or str(uuid.uuid4())
        ts = datetime.utcnow().isoformat() + "Z"
        details_json = json.dumps(details or {}, sort_keys=True, default=str)

        # Build payload dict for HMAC (deterministic key ordering)
        payload = json.dumps(
            {
                "audit_id": aid,
                "timestamp": ts,
                "agent_id": agent_id,
                "action_type": action_type,
                "risk_score": risk_score,
                "approved_by": approved_by,
                "outcome": outcome,
                "details": details or {},
            },
            sort_keys=True,
            default=str,
        )
        sig = self._sign(payload)

        with self._connect() as conn:
            conn.execute(
                _INSERT_SQL,
                (aid, ts, agent_id, action_type, risk_score, approved_by, outcome, details_json, sig),
            )

        return aid

    def verify_record(self, record: dict[str, Any]) -> bool:
        """
        Verify the HMAC signature of a record dict.

        Args:
            record: Dict with all audit fields including 'hmac_sig'.

        Returns:
            True if the signature is valid.
        """
        sig = record.get("hmac_sig", "")
        # Build the payload using the exact keys used during signing
        expected_keys = [
            "audit_id", "timestamp", "agent_id", "action_type", 
            "risk_score", "approved_by", "outcome", "details"
        ]
        payload_dict = {k: record.get(k) for k in expected_keys}
        payload = json.dumps(payload_dict, sort_keys=True, default=str)
        expected = self._sign(payload)
        return hmac.compare_digest(sig, expected)

    def query(
        self,
        agent_id: Optional[str] = None,
        action_type: Optional[str] = None,
        outcome: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Query audit records with optional filters.

        Args:
            agent_id:    Filter by agent ID.
            action_type: Filter by action type.
            outcome:     Filter by outcome.
            limit:       Max records to return (default 100).

        Returns:
            List of record dicts ordered by timestamp DESC.
        """
        where_clauses: list[str] = []
        params: list[Any] = []

        if agent_id:
            where_clauses.append("agent_id = ?")
            params.append(agent_id)
        if action_type:
            where_clauses.append("action_type = ?")
            params.append(action_type)
        if outcome:
            where_clauses.append("outcome = ?")
            params.append(outcome)

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        sql = f"SELECT * FROM audit_log {where_sql} ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()

        records = []
        for row in rows:
            d = dict(row)
            try:
                d["details"] = json.loads(d.get("details") or "{}")
            except (json.JSONDecodeError, TypeError):
                pass
            records.append(d)
        return records

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate statistics from the audit log."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
            by_outcome = {
                row[0]: row[1]
                for row in conn.execute(
                    "SELECT outcome, COUNT(*) FROM audit_log GROUP BY outcome"
                ).fetchall()
            }
            avg_risk = conn.execute(
                "SELECT AVG(risk_score) FROM audit_log"
            ).fetchone()[0]
        return {
            "total_records": total,
            "by_outcome": by_outcome,
            "average_risk_score": round(avg_risk or 0, 2),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Initialise the database and install append-only triggers."""
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(_CREATE_UPDATE_GUARD_SQL)
            conn.execute(_CREATE_DELETE_GUARD_SQL)

    def _connect(self) -> sqlite3.Connection:
        """Return a SQLite connection with WAL mode for concurrent reads."""
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @staticmethod
    def _sign(payload: str) -> str:
        """Compute HMAC-SHA256 signature of a payload string."""
        return hmac.new(_AUDIT_HMAC_SECRET, payload.encode(), hashlib.sha256).hexdigest()


# ---------------------------------------------------------------------------
# Singleton Factory
# ---------------------------------------------------------------------------

_instance: Optional[AuditLogger] = None


def get_audit_logger(db_path: Optional[str] = None) -> AuditLogger:
    """Return a singleton instance of the AuditLogger."""
    global _instance
    if _instance is None:
        _instance = AuditLogger(db_path)
    return _instance
