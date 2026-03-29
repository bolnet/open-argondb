"""Shared test fixtures — mock ArangoDB layer for unit tests."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any
from unittest.mock import MagicMock

import pytest


# ── Mock ArangoDB Layer ──────────────────────────────────────────────


class MockCursor:
    """Simulates an ArangoDB AQL cursor backed by a result list."""

    def __init__(self, results: list[dict[str, Any]]) -> None:
        self._results = results

    def __iter__(self):
        return iter(self._results)

    def __len__(self):
        return len(self._results)


class MockCollection:
    """In-memory ArangoDB collection."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._docs: dict[str, dict[str, Any]] = {}
        self._indexes: list[dict[str, Any]] = []
        self._rev_counter = 0

    def _next_rev(self) -> str:
        self._rev_counter += 1
        return f"_rev_{self._rev_counter}"

    def insert(self, doc: dict[str, Any], overwrite: bool = False, **kwargs: Any) -> dict[str, Any]:
        key = doc.get("_key", doc.get("_id", str(len(self._docs))))
        if key in self._docs and not overwrite:
            raise Exception(f"Document with _key '{key}' already exists")
        rev = self._next_rev()
        stored = {**doc, "_key": key, "_id": f"{self.name}/{key}", "_rev": rev}
        self._docs[key] = stored
        return {"_key": key, "_id": f"{self.name}/{key}", "_rev": rev}

    def update(self, doc: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        key = doc["_key"]
        if key not in self._docs:
            raise Exception(f"Document with _key '{key}' not found")
        rev = self._next_rev()
        self._docs[key] = {**self._docs[key], **doc, "_rev": rev}
        return {"_key": key, "_id": f"{self.name}/{key}", "_rev": rev}

    def get(self, key: str) -> dict[str, Any] | None:
        return self._docs.get(key)

    def has(self, key: str) -> bool:
        return key in self._docs

    def delete(self, key: str, **kwargs: Any) -> bool:
        if key in self._docs:
            del self._docs[key]
            return True
        return False

    def add_index(self, spec: dict[str, Any]) -> dict[str, Any]:
        self._indexes.append(spec)
        return spec

    def all(self) -> list[dict[str, Any]]:
        return list(self._docs.values())

    def truncate(self) -> None:
        self._docs.clear()


class MockAQL:
    """Minimal AQL executor that evaluates queries against MockCollections."""

    def __init__(self, collections: dict[str, MockCollection]) -> None:
        self._collections = collections

    def execute(
        self, query: str, bind_vars: dict[str, Any] | None = None, **kwargs: Any
    ) -> MockCursor:
        bind_vars = bind_vars or {}
        # Resolve @@col collection binding
        col_name = bind_vars.pop("@col", None)
        if col_name is None:
            # Extract collection name from "FOR x IN collection_name"
            match = re.search(r"FOR\s+\w+\s+IN\s+(\w+)", query)
            col_name = match.group(1) if match else "memories"

        col = self._collections.get(col_name)
        if not col:
            return MockCursor([])

        docs = list(col._docs.values())

        # Apply FILTER conditions (basic subset)
        docs = self._apply_filters(docs, query, bind_vars)

        # Apply SORT
        docs = self._apply_sort(docs, query)

        # Apply LIMIT
        limit_match = re.search(r"LIMIT\s+@?(\w+)", query)
        if limit_match:
            lim_key = limit_match.group(1)
            lim = bind_vars.get(lim_key, int(lim_key) if lim_key.isdigit() else 1000)
            docs = docs[:lim]

        # Apply RETURN projection
        docs = self._apply_return(docs, query)

        return MockCursor(docs)

    def _apply_filters(
        self, docs: list[dict], query: str, bind_vars: dict
    ) -> list[dict]:
        result = docs

        # FILTER doc._deleted != true
        if "_deleted != true" in query or "_deleted == false" in query:
            result = [d for d in result if not d.get("_deleted", False)]

        # FILTER doc.status == 'active'
        if "doc.status == 'active'" in query:
            result = [d for d in result if d.get("status") == "active"]

        # FILTER doc.memory_id == @mid
        if "@mid" in query and "mid" in bind_vars:
            result = [d for d in result if d.get("memory_id") == bind_vars["mid"]]

        # FILTER doc.agent_id == @aid
        if "@aid" in query and "aid" in bind_vars:
            result = [d for d in result if d.get("agent_id") == bind_vars["aid"]]

        # FILTER doc.op == @op
        if "doc.op == @op" in query and "op" in bind_vars:
            result = [d for d in result if d.get("op") == bind_vars["op"]]

        # FILTER doc.timestamp > @since
        if "timestamp > @since" in query and "since" in bind_vars:
            result = [d for d in result if d.get("timestamp", "") > bind_vars["since"]]

        # FILTER c.timestamp > @since (CDC pattern)
        if "c.timestamp > @since" in query and "since" in bind_vars:
            result = [d for d in result if d.get("timestamp", "") > bind_vars["since"]]

        # FILTER c._rev > @since (CDC rev pattern)
        if "c._rev > @since" in query and "since" in bind_vars:
            result = [d for d in result if d.get("_rev", "") > bind_vars["since"]]

        # FILTER doc.scope_agent_id == @agent_id
        if "@agent_id" in query and "agent_id" in bind_vars:
            result = [
                d for d in result if d.get("scope_agent_id") == bind_vars["agent_id"]
            ]

        # FILTER doc.scope_workflow_id == @wf_id
        if "@wf_id" in query and "wf_id" in bind_vars:
            result = [
                d for d in result
                if d.get("scope_workflow_id") == bind_vars["wf_id"]
            ]

        # FILTER LOWER(doc.entity) == LOWER(@ent)
        if "@ent" in query and "ent" in bind_vars:
            ent_lower = bind_vars["ent"].lower()
            result = [
                d for d in result
                if d.get("entity") and d["entity"].lower() == ent_lower
            ]

        # FILTER doc.status != 'superseded'
        if "doc.status != 'superseded'" in query:
            result = [d for d in result if d.get("status") != "superseded"]

        # FILTER doc.valid_from >= @start
        if "@start" in query and "start" in bind_vars:
            result = [
                d for d in result
                if d.get("valid_from", "") >= bind_vars["start"]
            ]

        # FILTER doc.valid_from <= @end
        if "@end" in query and "end" in bind_vars:
            result = [
                d for d in result
                if d.get("valid_from", "") <= bind_vars["end"]
            ]

        # FILTER doc.embedding == null OR NOT HAS(doc, 'embedding')
        if "embedding == null" in query:
            result = [
                d for d in result
                if d.get("embedding") is None
            ]

        # FILTER HAS(doc, 'embedding') AND doc.embedding != null (vector cache)
        if "HAS(doc, 'embedding') AND doc.embedding != null" in query:
            result = [d for d in result if d.get("embedding") is not None]

        return result

    def _apply_sort(self, docs: list[dict], query: str) -> list[dict]:
        sort_match = re.search(r"SORT\s+\w+\.(\w+)\s+(ASC|DESC)?", query)
        if sort_match:
            field = sort_match.group(1)
            desc = sort_match.group(2) == "DESC"
            docs = sorted(docs, key=lambda d: d.get(field, ""), reverse=desc)
        return docs

    def _apply_return(self, docs: list[dict], query: str) -> list[dict]:
        # RETURN doc.content — return just that field
        field_match = re.search(r"RETURN\s+\w+\.(\w+)\s*$", query.strip())
        if field_match:
            field = field_match.group(1)
            return [d.get(field) for d in docs]

        # RETURN {field: doc.field, ...} — return projection
        proj_match = re.search(r"RETURN\s*\{([^}]+)\}", query)
        if proj_match:
            projection_str = proj_match.group(1)
            fields = []
            for part in projection_str.split(","):
                part = part.strip()
                if ":" in part:
                    alias, expr = part.split(":", 1)
                    # Extract field name from doc.field_name
                    fm = re.search(r"\w+\.(\w+)", expr.strip())
                    fields.append((alias.strip(), fm.group(1) if fm else alias.strip()))
                else:
                    fields.append((part, part))
            return [{alias: d.get(src) for alias, src in fields} for d in docs]

        # RETURN doc — return full document
        return docs


class MockDatabase:
    """In-memory ArangoDB database."""

    def __init__(self, name: str = "test_db") -> None:
        self.name = name
        self._collections: dict[str, MockCollection] = {}
        self.aql = MockAQL(self._collections)
        self._graphs: dict[str, dict] = {}

    def has_collection(self, name: str) -> bool:
        return name in self._collections

    def create_collection(self, name: str, **kwargs: Any) -> MockCollection:
        col = MockCollection(name)
        self._collections[name] = col
        return col

    def collection(self, name: str) -> MockCollection:
        if name not in self._collections:
            self._collections[name] = MockCollection(name)
        return self._collections[name]

    def delete_collection(self, name: str) -> None:
        self._collections.pop(name, None)

    def has_graph(self, name: str) -> bool:
        return name in self._graphs

    def create_graph(self, name: str, **kwargs: Any) -> dict:
        self._graphs[name] = {"name": name, **kwargs}
        return self._graphs[name]

    def graph(self, name: str) -> dict:
        return self._graphs.get(name, {})

    def delete_graph(self, name: str, **kwargs: Any) -> None:
        self._graphs.pop(name, None)

    def has_database(self, name: str) -> bool:
        return True

    def create_database(self, name: str) -> None:
        pass


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def mock_db() -> MockDatabase:
    """Fresh in-memory ArangoDB database."""
    return MockDatabase()


@pytest.fixture
def document_store(mock_db: MockDatabase):
    """DocumentStore wired to mock DB."""
    from open_arangodb.store.document_store import DocumentStore
    return DocumentStore(mock_db)


@pytest.fixture
def cdc_engine(mock_db: MockDatabase):
    """CDCEngine wired to mock DB."""
    from open_arangodb.cdc.engine import CDCEngine
    from open_arangodb.events.bus import InProcessBus
    return CDCEngine(mock_db, InProcessBus())


@pytest.fixture
def audit_logger(mock_db: MockDatabase):
    """AuditLogger wired to mock DB."""
    from open_arangodb.audit.logger import AuditLogger
    return AuditLogger(mock_db)


@pytest.fixture
def event_bus():
    """Fresh InProcessBus."""
    from open_arangodb.events.bus import InProcessBus
    return InProcessBus()


@pytest.fixture
def scope_manager(mock_db: MockDatabase):
    """ScopeManager wired to mock DB."""
    from open_arangodb.scoping.manager import ScopeManager
    return ScopeManager(mock_db)
