"""
guardrails/undo_buffer.py — UndoBuffer

Snapshots filesystem state before destructive agent actions. Supports
rollback of the last 10 operations using a per-snapshot tar.gz archive.

Storage: ~/.agent_undo/ (configurable via UNDO_BUFFER_DIR env var)
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tarfile
import tempfile
import uuid
from collections import deque
from pathlib import Path
from typing import Optional

from guardrails.exceptions import UndoBufferError
from guardrails.models import ActionType, SnapshotMetadata

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_UNDO_DIR = Path(os.getenv("UNDO_BUFFER_DIR", Path.home() / ".agent_undo"))
_MAX_SNAPSHOTS = 10


class UndoBuffer:
    """
    Ring-buffer of up to 10 filesystem snapshots, each taken before a
    destructive agent action. Supports sequential rollback.

    Usage:
        buf = UndoBuffer()
        snapshot_id = await buf.snapshot(
            paths=["/home/agent/workspace/config.yaml"],
            operation_id="op-abc",
            agent_id="agent-001",
            action_type=ActionType.FILE_WRITE,
        )
        await buf.rollback()   # restore to pre-write state
    """

    def __init__(
        self,
        undo_dir: Optional[str] = None,
        max_snapshots: int = _MAX_SNAPSHOTS,
    ) -> None:
        self._undo_dir = Path(undo_dir) if undo_dir else _UNDO_DIR
        self._undo_dir.mkdir(parents=True, exist_ok=True)
        self._max_snapshots = max_snapshots
        # Ring buffer of SnapshotMetadata (most-recent last)
        self._ring: deque[SnapshotMetadata] = deque(maxlen=max_snapshots)
        self._load_existing_snapshots()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def snapshot(
        self,
        paths: list[str],
        operation_id: str,
        agent_id: str,
        action_type: ActionType = ActionType.FILE_WRITE,
    ) -> str:
        """
        Create a tar.gz snapshot of the given file paths.

        Only existing files/directories are included. Non-existent paths
        are recorded in metadata but not archived.

        Args:
            paths:        List of absolute filesystem paths to snapshot.
            operation_id: Unique identifier of the operation about to run.
            agent_id:     Agent performing the action.
            action_type:  Type of action that triggered the snapshot.

        Returns:
            snapshot_id (UUID string)

        Raises:
            UndoBufferError: If archiving fails.
        """
        snapshot_id = str(uuid.uuid4())
        archive_name = f"{snapshot_id}.tar.gz"
        archive_path = self._undo_dir / archive_name

        existing_paths = [p for p in paths if Path(p).exists()]

        # Archive existing paths
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self._create_archive, str(archive_path), existing_paths
        )

        metadata = SnapshotMetadata(
            snapshot_id=snapshot_id,
            operation_id=operation_id,
            agent_id=agent_id,
            action_type=action_type,
            paths=paths,
            archive_path=str(archive_path),
        )

        # Evict oldest snapshot if ring buffer is full
        if len(self._ring) >= self._max_snapshots:
            oldest = self._ring[0]  # deque maxlen handles eviction, but clean file
            self._remove_archive(oldest.archive_path)

        self._ring.append(metadata)
        self._persist_metadata(metadata)
        return snapshot_id

    async def rollback(self, n: int = 1) -> list[str]:
        """
        Restore the last `n` snapshots in reverse chronological order.

        Args:
            n: Number of most-recent snapshots to roll back (default 1).

        Returns:
            List of restored snapshot_ids.

        Raises:
            UndoBufferError: If the ring buffer is empty or n > available.
        """
        if not self._ring:
            raise UndoBufferError("UndoBuffer is empty — no snapshots to roll back.")
        if n > len(self._ring):
            raise UndoBufferError(
                f"Requested rollback of {n} snapshots but only {len(self._ring)} available."
            )

        restored: list[str] = []
        loop = asyncio.get_event_loop()

        for _ in range(n):
            meta = self._ring.pop()  # most recent first
            if not Path(meta.archive_path).exists():
                raise UndoBufferError(
                    f"Archive missing for snapshot {meta.snapshot_id}: {meta.archive_path}"
                )
            await loop.run_in_executor(None, self._extract_archive, meta.archive_path)
            meta.restored = True
            self._persist_metadata(meta)
            # Remove cleaned archive
            self._remove_archive(meta.archive_path)
            restored.append(meta.snapshot_id)

        return restored

    def list_snapshots(self) -> list[dict]:
        """Return metadata for all buffered snapshots (oldest first)."""
        return [
            {
                "snapshot_id": m.snapshot_id,
                "operation_id": m.operation_id,
                "agent_id": m.agent_id,
                "action_type": m.action_type,
                "paths": m.paths,
                "created_at": m.created_at.isoformat(),
                "restored": m.restored,
            }
            for m in self._ring
        ]

    def clear(self) -> None:
        """Remove all snapshots and reset the ring buffer."""
        for meta in self._ring:
            self._remove_archive(meta.archive_path)
        self._ring.clear()
        # Remove all metadata files
        for f in self._undo_dir.glob("*.meta.json"):
            f.unlink(missing_ok=True)

    @property
    def size(self) -> int:
        """Number of snapshots currently in the buffer."""
        return len(self._ring)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_archive(archive_path: str, paths: list[str]) -> None:
        """Create a gzip-compressed tar archive of the given paths."""
        with tarfile.open(archive_path, "w:gz") as tar:
            for path in paths:
                p = Path(path)
                if p.exists():
                    # arcname preserves absolute path structure for restoration
                    tar.add(str(p), arcname=str(p).lstrip("/"))

    @staticmethod
    def _extract_archive(archive_path: str) -> None:
        """Extract archive, restoring files to their original absolute paths."""
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar.getmembers():
                # Reconstruct absolute path
                abs_path = Path("/") / member.name
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                # Extract to a temp location first, then move
                with tempfile.TemporaryDirectory() as tmpdir:
                    tar.extract(member, path=tmpdir)
                    tmp_file = Path(tmpdir) / member.name
                    if tmp_file.exists():
                        shutil.move(str(tmp_file), str(abs_path))

    @staticmethod
    def _remove_archive(archive_path: str) -> None:
        """Silently remove an archive file."""
        try:
            Path(archive_path).unlink(missing_ok=True)
            # Remove metadata too
            meta_path = Path(archive_path).with_suffix("").with_suffix(".meta.json")
            meta_path.unlink(missing_ok=True)
        except Exception:
            pass

    def _persist_metadata(self, meta: SnapshotMetadata) -> None:
        """Write snapshot metadata JSON alongside the archive."""
        meta_path = Path(meta.archive_path).with_suffix("").with_suffix(".meta.json")
        try:
            meta_path.write_text(
                json.dumps(meta.model_dump(mode="json"), default=str, indent=2)
            )
        except Exception:
            pass  # metadata persistence is best-effort

    def _load_existing_snapshots(self) -> None:
        """Reload any metadata files left from a previous session."""
        meta_files = sorted(self._undo_dir.glob("*.meta.json"))
        for mf in meta_files[-self._max_snapshots:]:
            try:
                data = json.loads(mf.read_text())
                meta = SnapshotMetadata(**data)
                if Path(meta.archive_path).exists() and not meta.restored:
                    self._ring.append(meta)
            except Exception:
                pass  # corrupt metadata — skip
