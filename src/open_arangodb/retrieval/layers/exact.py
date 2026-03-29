"""Exact-match retrieval layer — direct memory ID lookup."""

from __future__ import annotations

import json
import re
from typing import Any

from open_arangodb.models import (
    AgentScope,
    Memory,
    RetrievalRequest,
    RetrievalResult,
    Visibility,
)

_ID_PATTERN = re.compile(r"^[\w-]+$")


class ExactMatchLayer:
    """Look up a memory by ID when the query looks like an identifier."""

    def __init__(self, db: Any) -> None:
        self._db = db

    def search(self, request: RetrievalRequest) -> list[RetrievalResult]:
        """Return a single result if query matches a memory ID exactly."""
        query = request.query.strip()
        if not query or not _ID_PATTERN.match(query):
            return []

        cursor = self._db.aql.execute(
            "FOR doc IN memories FILTER doc.memory_id == @mid LIMIT 1 RETURN doc",
            bind_vars={"mid": query},
        )
        docs = list(cursor)
        if not docs:
            return []

        memory = _doc_to_memory(docs[0])
        return [
            RetrievalResult(
                memory=memory,
                score=1.0,
                match_source="exact",
                tier=1,
            )
        ]


def _doc_to_memory(doc: dict[str, Any]) -> Memory:
    """Convert a raw document to a Memory dataclass."""
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

    scope: AgentScope | None = None
    if doc.get("scope_agent_id"):
        scope = AgentScope(
            agent_id=doc["scope_agent_id"],
            session_id=doc.get("scope_session_id"),
            workflow_id=doc.get("scope_workflow_id"),
            visibility=Visibility(doc.get("scope_visibility", "global")),
        )

    return Memory(
        id=doc.get("memory_id", doc.get("_key", "")),
        content=doc.get("content", ""),
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
