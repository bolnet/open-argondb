"""Change Data Capture engine — changelog collection + rev-based tracking."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from open_argondb.models import ChangeEvent, ChangeOp, Memory


class CDCEngine:
    """Tracks all changes to memories via a dedicated changelog collection.

    Unlike Enterprise CDC which tails the WAL, this uses a service-layer
    approach: all writes go through ArgonDB.insert/update/delete which call
    record_change(). Consumers poll get_changes() with a checkpoint.

    This guarantees no missed events as long as all writes go through the
    service layer (which the ArgonDB gateway enforces).
    """

    CHANGELOG = "_argondb_changelog"
    CHECKPOINTS = "_argondb_checkpoints"

    def __init__(self, db: Any, event_bus: Any = None) -> None:
        self._db = db
        self._events = event_bus
        self._ensure_collections()

    def _ensure_collections(self) -> None:
        if not self._db.has_collection(self.CHANGELOG):
            col = self._db.create_collection(self.CHANGELOG, system=True)
            col.add_index({"type": "persistent", "fields": ["timestamp"]})
            col.add_index({"type": "persistent", "fields": ["memory_id"]})
            col.add_index({
                "type": "ttl",
                "fields": ["expires_at"],
                "expireAfter": 0,
            })

        if not self._db.has_collection(self.CHECKPOINTS):
            self._db.create_collection(self.CHECKPOINTS, system=True)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def record_change(
        self,
        op: str,
        memory_id: str,
        before: Memory | None = None,
        after: Memory | None = None,
        ttl_seconds: int = 7 * 86400,  # 7 days default retention
    ) -> str:
        """Record a change in the changelog collection."""
        now = self._now()
        expires = datetime.now(timezone.utc).timestamp() + ttl_seconds
        expires_at = datetime.fromtimestamp(expires, tz=timezone.utc).isoformat()

        doc = {
            "_key": uuid4().hex[:16],
            "op": op,
            "memory_id": memory_id,
            "timestamp": now,
            "expires_at": expires_at,
            "before": self._serialize_memory(before) if before else None,
            "after": self._serialize_memory(after) if after else None,
        }

        result = self._db.collection(self.CHANGELOG).insert(doc)
        return result["_rev"]

    def get_changes(
        self,
        since_rev: str | None = None,
        since_timestamp: str | None = None,
        limit: int = 1000,
    ) -> list[ChangeEvent]:
        """Get changes since a checkpoint. Use either rev or timestamp."""
        bind_vars: dict[str, Any] = {"lim": limit}

        if since_timestamp:
            query = (
                "FOR c IN @@col FILTER c.timestamp > @since "
                "SORT c.timestamp LIMIT @lim RETURN c"
            )
            bind_vars["since"] = since_timestamp
            bind_vars["@col"] = self.CHANGELOG
        elif since_rev:
            query = (
                "FOR c IN @@col FILTER c._rev > @since "
                "SORT c._rev LIMIT @lim RETURN c"
            )
            bind_vars["since"] = since_rev
            bind_vars["@col"] = self.CHANGELOG
        else:
            query = (
                "FOR c IN @@col SORT c.timestamp DESC LIMIT @lim RETURN c"
            )
            bind_vars["@col"] = self.CHANGELOG

        cursor = self._db.aql.execute(query, bind_vars=bind_vars)
        return [self._doc_to_event(doc) for doc in cursor]

    def save_checkpoint(self, consumer_id: str, rev: str) -> None:
        """Save a consumer's checkpoint."""
        self._db.collection(self.CHECKPOINTS).insert({
            "_key": consumer_id,
            "rev": rev,
            "timestamp": self._now(),
        }, overwrite=True)

    def get_checkpoint(self, consumer_id: str) -> str | None:
        """Get a consumer's last checkpoint."""
        try:
            doc = self._db.collection(self.CHECKPOINTS).get(consumer_id)
            return doc["rev"] if doc else None
        except Exception:
            return None

    def _serialize_memory(self, memory: Memory) -> dict[str, Any]:
        return {
            "id": memory.id,
            "content": memory.content,
            "tags": memory.tags,
            "category": memory.category,
            "entity": memory.entity,
            "status": memory.status,
        }

    def _doc_to_event(self, doc: dict[str, Any]) -> ChangeEvent:
        return ChangeEvent(
            op=ChangeOp(doc["op"]),
            memory_id=doc["memory_id"],
            rev=doc.get("_rev", ""),
            timestamp=doc["timestamp"],
            before=doc.get("before"),
            after=doc.get("after"),
        )

    def reset(self) -> None:
        for col_name in [self.CHANGELOG, self.CHECKPOINTS]:
            if self._db.has_collection(col_name):
                self._db.delete_collection(col_name, system=True)
        self._ensure_collections()

    def stop(self) -> None:
        pass
