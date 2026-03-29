"""Temporal engine — supersession chains, contradiction detection, time-range queries."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import combinations
from typing import Any
from uuid import uuid4

from open_argondb.models import (
    AgentScope,
    Contradiction,
    Memory,
    SupersessionChain,
    TemporalQuery,
)

_MAX_CHAIN_HOPS = 100


class TemporalEngine:
    """Temporal reasoning over the memory store.

    Provides supersession-chain traversal, contradiction detection,
    and time-range querying on top of the raw ArangoDB document layer.
    """

    COLLECTION = "memories"

    def __init__(self, db: Any) -> None:
        self._db = db

    # ── helpers ───────────────────────────────────────────────────────

    def _doc_to_memory(self, doc: dict[str, Any]) -> Memory:
        """Lightweight doc-to-Memory conversion (mirrors DocumentStore)."""
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
            from open_argondb.models import Visibility

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

    def _fetch_by_id(self, memory_id: str) -> dict[str, Any] | None:
        """Fetch a single memory document by memory_id via AQL."""
        cursor = self._db.aql.execute(
            "FOR doc IN memories FILTER doc.memory_id == @mid LIMIT 1 RETURN doc",
            bind_vars={"mid": memory_id},
        )
        docs = list(cursor)
        return docs[0] if docs else None

    # ── public API ───────────────────────────────────────────────────

    def get_supersession_chain(self, memory_id: str) -> SupersessionChain:
        """Walk forward along superseded_by links from *memory_id*.

        Returns a :class:`SupersessionChain` containing every memory_id
        in order, ending at the current (non-superseded) tip.
        Max ``_MAX_CHAIN_HOPS`` iterations to guard against corrupt loops.
        """
        chain_ids: list[str] = []
        current = memory_id
        visited: set[str] = set()

        for _ in range(_MAX_CHAIN_HOPS):
            if current in visited:
                break
            visited.add(current)
            chain_ids.append(current)

            doc = self._fetch_by_id(current)
            if doc is None:
                break

            next_id = doc.get("superseded_by")
            if not next_id:
                break
            current = next_id

        tip = chain_ids[-1] if chain_ids else memory_id
        return SupersessionChain(
            chain_id=uuid4().hex[:16],
            memory_ids=chain_ids,
            current_id=tip,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def get_current_version(self, memory_id: str) -> Memory | None:
        """Follow the supersession chain to its tip and return that Memory."""
        chain = self.get_supersession_chain(memory_id)
        doc = self._fetch_by_id(chain.current_id)
        if doc is None:
            return None
        return self._doc_to_memory(doc)

    def detect_contradictions(
        self,
        entity: str,
        scope: AgentScope | None = None,
    ) -> list[Contradiction]:
        """Find active memories for *entity* whose content differs.

        Only considers memories with ``status == 'active'``.  For each
        unordered pair where content differs, a :class:`Contradiction` is
        emitted.  Uses simple string inequality (no embedding similarity).
        """
        bind_vars: dict[str, Any] = {"ent": entity}
        filters = [
            "doc._deleted != true",
            "doc.status == 'active'",
            "LOWER(doc.entity) == LOWER(@ent)",
        ]

        if scope is not None:
            filters.append("doc.scope_agent_id == @agent_id")
            bind_vars["agent_id"] = scope.agent_id

        where = " AND ".join(filters)
        query = f"FOR doc IN memories FILTER {where} RETURN doc"
        cursor = self._db.aql.execute(query, bind_vars=bind_vars)
        docs = list(cursor)

        contradictions: list[Contradiction] = []
        now = datetime.now(timezone.utc).isoformat()

        for doc_a, doc_b in combinations(docs, 2):
            content_a = doc_a.get("content", "")
            content_b = doc_b.get("content", "")
            if content_a != content_b:
                contradictions.append(
                    Contradiction(
                        memory_a_id=doc_a.get("memory_id", ""),
                        memory_b_id=doc_b.get("memory_id", ""),
                        reason=f"Different content for entity '{entity}'",
                        confidence=1.0,
                        detected_at=now,
                    )
                )

        return contradictions

    def query_temporal_range(self, query: TemporalQuery) -> list[Memory]:
        """Query memories filtered by temporal fields and optional entity.

        Filters on ``valid_from``, ``valid_until``, and ``event_date``
        using the *start* / *end* bounds from *query*.
        """
        filters = ["doc._deleted != true"]
        bind_vars: dict[str, Any] = {}

        if not query.include_superseded:
            filters.append("doc.status != 'superseded'")

        if query.entity is not None:
            filters.append("LOWER(doc.entity) == LOWER(@ent)")
            bind_vars["ent"] = query.entity

        if query.start is not None:
            filters.append("doc.valid_from >= @start")
            bind_vars["start"] = query.start

        if query.end is not None:
            filters.append("doc.valid_from <= @end")
            bind_vars["end"] = query.end

        where = " AND ".join(filters)
        aql = f"FOR doc IN memories FILTER {where} SORT doc.valid_from ASC RETURN doc"
        cursor = self._db.aql.execute(aql, bind_vars=bind_vars)
        return [self._doc_to_memory(doc) for doc in cursor]

    def get_history(self, memory_id: str) -> list[Memory]:
        """Return all versions of a memory in chronological order.

        Walks backward from *memory_id* to find the root (a memory
        with no predecessor), then walks forward via ``superseded_by``
        to collect the full chronological sequence.
        """
        # Step 1: collect all memory docs so we can build a reverse map
        cursor = self._db.aql.execute(
            "FOR doc IN memories RETURN doc",
            bind_vars={},
        )
        all_docs = list(cursor)

        # Build lookup: memory_id -> doc, and reverse map: superseded_by -> predecessor
        by_id: dict[str, dict[str, Any]] = {}
        predecessor_of: dict[str, str] = {}  # target -> source
        for doc in all_docs:
            mid = doc.get("memory_id", "")
            by_id[mid] = doc
            sb = doc.get("superseded_by")
            if sb:
                predecessor_of[sb] = mid

        # Step 2: walk backward to root
        root = memory_id
        visited: set[str] = set()
        for _ in range(_MAX_CHAIN_HOPS):
            if root in visited:
                break
            visited.add(root)
            prev = predecessor_of.get(root)
            if prev is None:
                break
            root = prev

        # Step 3: walk forward from root
        chain: list[Memory] = []
        current: str | None = root
        visited_fwd: set[str] = set()
        for _ in range(_MAX_CHAIN_HOPS):
            if current is None or current in visited_fwd:
                break
            visited_fwd.add(current)
            doc = by_id.get(current)
            if doc is None:
                break
            chain.append(self._doc_to_memory(doc))
            current = doc.get("superseded_by")

        return chain
