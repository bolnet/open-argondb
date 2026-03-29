"""DC2DC replication engine — replays CDC changes to a target ArangoDB."""

from __future__ import annotations

import logging
import threading
from typing import Any

from open_argondb.models import ReplicationConfig, ReplicationStatus

logger = logging.getLogger("open_argondb")


class ReplicationEngine:
    """CDC-based DC2DC replication — replays changes to a target ArangoDB."""

    def __init__(
        self, source_cdc: Any, target_db: Any, config: ReplicationConfig
    ) -> None:
        self._cdc = source_cdc
        self._target = target_db
        self._config = config
        self._state = "stopped"
        self._last_rev: str | None = None
        self._pending = 0
        self._error: str | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start replication in a background thread."""
        self._state = "running"
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop replication gracefully."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        self._state = "stopped"

    def pause(self) -> None:
        """Pause replication (thread keeps running but skips batches)."""
        self._state = "paused"

    def resume(self) -> None:
        """Resume replication after pause."""
        self._state = "running"

    def status(self) -> ReplicationStatus:
        """Return current replication status."""
        return ReplicationStatus(
            state=self._state,
            last_synced_rev=self._last_rev,
            pending_changes=self._pending,
            error=self._error,
        )

    def replicate_batch(self) -> int:
        """Manual single-batch replication. Returns count replicated."""
        changes = self._cdc.get_changes(
            since_rev=self._last_rev, limit=self._config.batch_size
        )
        count = 0
        for change in changes:
            try:
                self._apply_change(change)
                self._last_rev = change.rev
                count += 1
            except Exception as e:
                logger.error("Replication error: %s", e)
                self._error = str(e)
        return count

    def _poll_loop(self) -> None:
        """Background polling loop."""
        while not self._stop_event.is_set():
            if self._state == "running":
                try:
                    self.replicate_batch()
                except Exception as e:
                    self._state = "error"
                    self._error = str(e)
            self._stop_event.wait(self._config.poll_interval_seconds)

    def _apply_change(self, change: Any) -> None:
        """Replay a change event on the target database."""
        col_name = "memories"  # Default collection
        if not self._target.has_collection(col_name):
            self._target.create_collection(col_name)
        col = self._target.collection(col_name)

        if change.op.value in ("insert", "supersede") and change.after:
            key = change.memory_id.replace("/", "_")
            doc = {"_key": key, **change.after}
            col.insert(doc, overwrite=True)
        elif change.op.value == "update" and change.after:
            key = change.memory_id.replace("/", "_")
            doc = {"_key": key, **change.after}
            try:
                col.update(doc)
            except Exception:
                col.insert(doc, overwrite=True)
        elif change.op.value == "delete":
            key = change.memory_id.replace("/", "_")
            try:
                col.update({"_key": key, "_deleted": True})
            except Exception:
                pass
