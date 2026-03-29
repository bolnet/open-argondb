"""ArangoDB document store with soft-delete and scoping support."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from open_arangodb.models import AgentScope, Memory


class DocumentStore:
    """Memory document storage backed by ArangoDB."""

    COLLECTION = "memories"

    def __init__(self, db: Any) -> None:
        self._db = db
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self._db.has_collection(self.COLLECTION):
            col = self._db.create_collection(self.COLLECTION)
            col.add_index({"type": "persistent", "fields": ["category"]})
            col.add_index({"type": "persistent", "fields": ["entity"]})
            col.add_index({"type": "persistent", "fields": ["memory_id"], "unique": True})
            col.add_index({"type": "persistent", "fields": ["status"]})
            col.add_index({"type": "persistent", "fields": ["scope_agent_id"]})
            col.add_index({"type": "persistent", "fields": ["scope_workflow_id"]})
            col.add_index({"type": "persistent", "fields": ["updated_at"]})
            col.add_index({"type": "persistent", "fields": ["_deleted"]})
        self._col = self._db.collection(self.COLLECTION)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _memory_to_doc(self, memory: Memory) -> dict[str, Any]:
        scope = memory.scope
        return {
            "_key": memory.id.replace("/", "_"),
            "memory_id": memory.id,
            "content": memory.content,
            "tags": json.dumps(memory.tags or []),
            "category": memory.category,
            "entity": memory.entity,
            "created_at": memory.created_at or self._now(),
            "event_date": memory.event_date,
            "valid_from": memory.valid_from or self._now(),
            "valid_until": memory.valid_until,
            "superseded_by": memory.superseded_by,
            "confidence": memory.confidence,
            "status": memory.status,
            "embedding": None,
            "metadata": json.dumps(memory.metadata or {}),
            "scope_agent_id": scope.agent_id if scope else None,
            "scope_session_id": scope.session_id if scope else None,
            "scope_workflow_id": scope.workflow_id if scope else None,
            "scope_visibility": scope.visibility.value if scope else "global",
            "updated_at": self._now(),
            "_deleted": False,
        }

    def _doc_to_memory(self, doc: dict[str, Any]) -> Memory:
        tags = doc.get("tags", "[]")
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except (json.JSONDecodeError, TypeError):
                tags = []

        metadata = doc.get("metadata", "{}")
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except (json.JSONDecodeError, TypeError):
                metadata = {}

        scope = None
        if doc.get("scope_agent_id"):
            from open_arangodb.models import Visibility

            scope = AgentScope(
                agent_id=doc["scope_agent_id"],
                session_id=doc.get("scope_session_id"),
                workflow_id=doc.get("scope_workflow_id"),
                visibility=Visibility(doc.get("scope_visibility", "global")),
            )

        return Memory(
            id=doc["memory_id"],
            content=doc["content"],
            tags=tags,
            category=doc.get("category", "general"),
            entity=doc.get("entity"),
            created_at=doc.get("created_at", ""),
            event_date=doc.get("event_date"),
            valid_from=doc.get("valid_from", ""),
            valid_until=doc.get("valid_until"),
            superseded_by=doc.get("superseded_by"),
            confidence=doc.get("confidence", 1.0),
            status=doc.get("status", "active"),
            scope=scope,
            metadata=metadata,
        )

    def insert(self, memory: Memory) -> Memory:
        doc = self._memory_to_doc(memory)
        self._col.insert(doc, overwrite=True)
        return memory

    def update(self, memory: Memory) -> Memory:
        doc = self._memory_to_doc(memory)
        self._col.update(doc)
        return memory

    def get(self, memory_id: str) -> Memory | None:
        cursor = self._db.aql.execute(
            "FOR doc IN memories FILTER doc.memory_id == @mid AND doc._deleted != true "
            "LIMIT 1 RETURN doc",
            bind_vars={"mid": memory_id},
        )
        docs = list(cursor)
        return self._doc_to_memory(docs[0]) if docs else None

    def soft_delete(self, memory_id: str) -> None:
        """Mark as deleted without removing. CDC and audit can still see it."""
        key = memory_id.replace("/", "_")
        self._col.update({"_key": key, "_deleted": True, "updated_at": self._now()})

    def mark_superseded(self, old_id: str, new_id: str) -> None:
        key = old_id.replace("/", "_")
        self._col.update({
            "_key": key,
            "status": "superseded",
            "superseded_by": new_id,
            "valid_until": self._now(),
            "updated_at": self._now(),
        })

    def list_memories(
        self,
        entity: str | None = None,
        scope: AgentScope | None = None,
        limit: int = 50,
    ) -> list[Memory]:
        filters = ["doc._deleted != true", "doc.status == 'active'"]
        bind_vars: dict[str, Any] = {"lim": limit}

        if entity:
            filters.append("LOWER(doc.entity) == LOWER(@ent)")
            bind_vars["ent"] = entity

        if scope:
            filters.append("doc.scope_agent_id == @agent_id")
            bind_vars["agent_id"] = scope.agent_id
            if scope.workflow_id:
                filters.append("doc.scope_workflow_id == @wf_id")
                bind_vars["wf_id"] = scope.workflow_id

        where = " AND ".join(filters)
        query = f"FOR doc IN memories FILTER {where} LIMIT @lim RETURN doc"
        cursor = self._db.aql.execute(query, bind_vars=bind_vars)
        return [self._doc_to_memory(doc) for doc in cursor]

    def reset(self) -> None:
        if self._db.has_collection(self.COLLECTION):
            self._db.delete_collection(self.COLLECTION)
        self._ensure_collection()
