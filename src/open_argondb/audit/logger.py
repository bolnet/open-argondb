"""Audit logging — every operation recorded to a dedicated collection."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from open_argondb.models import AgentScope


class AuditLogger:
    """Records all database operations for compliance and debugging."""

    COLLECTION = "_argondb_audit"

    def __init__(self, db: Any) -> None:
        self._db = db
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self._db.has_collection(self.COLLECTION):
            col = self._db.create_collection(self.COLLECTION)
            col.add_index({"type": "persistent", "fields": ["timestamp"]})
            col.add_index({"type": "persistent", "fields": ["op"]})
            col.add_index({"type": "persistent", "fields": ["agent_id"]})
            col.add_index({"type": "persistent", "fields": ["document_key"]})
            col.add_index({
                "type": "ttl",
                "fields": ["expires_at"],
                "expireAfter": 0,
            })
        self._col = self._db.collection(self.COLLECTION)

    def log(
        self,
        op: str,
        collection: str,
        document_key: str,
        scope: AgentScope | None = None,
        content_hash: str | None = None,
        retention_days: int = 90,
    ) -> None:
        now = datetime.now(timezone.utc)
        expires = datetime.fromtimestamp(
            now.timestamp() + retention_days * 86400, tz=timezone.utc
        )

        self._col.insert({
            "_key": uuid4().hex[:16],
            "op": op,
            "collection": collection,
            "document_key": document_key,
            "agent_id": scope.agent_id if scope else None,
            "session_id": scope.session_id if scope else None,
            "workflow_id": scope.workflow_id if scope else None,
            "timestamp": now.isoformat(),
            "content_hash": content_hash,
            "expires_at": expires.isoformat(),
        })

    def query(
        self,
        agent_id: str | None = None,
        op: str | None = None,
        since: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        filters = []
        bind_vars: dict[str, Any] = {"lim": limit}

        if agent_id:
            filters.append("doc.agent_id == @aid")
            bind_vars["aid"] = agent_id
        if op:
            filters.append("doc.op == @op")
            bind_vars["op"] = op
        if since:
            filters.append("doc.timestamp > @since")
            bind_vars["since"] = since

        where = f"FILTER {' AND '.join(filters)}" if filters else ""
        query = (
            f"FOR doc IN {self.COLLECTION} {where} "
            "SORT doc.timestamp DESC LIMIT @lim RETURN doc"
        )
        return list(self._db.aql.execute(query, bind_vars=bind_vars))

    def reset(self) -> None:
        if self._db.has_collection(self.COLLECTION):
            self._db.delete_collection(self.COLLECTION)
        self._ensure_collection()
