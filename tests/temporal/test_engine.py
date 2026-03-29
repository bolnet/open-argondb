"""Tests for the temporal engine — chains, contradictions, time-range queries."""

from __future__ import annotations

import json
from typing import Any

import pytest

from open_argondb.models import AgentScope, Memory, TemporalQuery
from open_argondb.temporal.engine import TemporalEngine


# ── Helpers ──────────────────────────────────────────────────────────


def _make_doc(
    memory_id: str,
    content: str = "some content",
    entity: str | None = None,
    status: str = "active",
    superseded_by: str | None = None,
    valid_from: str = "2025-01-01T00:00:00+00:00",
    valid_until: str | None = None,
    event_date: str | None = None,
    scope_agent_id: str | None = None,
) -> dict[str, Any]:
    """Build a raw memory document for insertion into the mock collection."""
    return {
        "_key": memory_id.replace("/", "_"),
        "memory_id": memory_id,
        "content": content,
        "tags": json.dumps([]),
        "category": "general",
        "entity": entity,
        "created_at": valid_from,
        "event_date": event_date,
        "valid_from": valid_from,
        "valid_until": valid_until,
        "superseded_by": superseded_by,
        "confidence": 1.0,
        "status": status,
        "embedding": None,
        "metadata": json.dumps({}),
        "scope_agent_id": scope_agent_id,
        "scope_session_id": None,
        "scope_workflow_id": None,
        "scope_visibility": "global",
        "updated_at": valid_from,
        "_deleted": False,
    }


def _insert(mock_db: Any, doc: dict[str, Any]) -> None:
    """Insert a document into the mock memories collection."""
    col = mock_db.collection("memories")
    col.insert(doc, overwrite=True)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def engine(mock_db: Any) -> TemporalEngine:
    return TemporalEngine(mock_db)


# ── Supersession Chain Tests ─────────────────────────────────────────


def test_get_supersession_chain_single(mock_db: Any, engine: TemporalEngine) -> None:
    """A memory with no superseded_by yields a chain of length 1."""
    _insert(mock_db, _make_doc("m1", content="first"))

    chain = engine.get_supersession_chain("m1")

    assert chain.memory_ids == ["m1"]
    assert chain.current_id == "m1"
    assert chain.chain_id  # non-empty


def test_get_supersession_chain_multi(mock_db: Any, engine: TemporalEngine) -> None:
    """A three-link chain a->b->c is fully traversed."""
    _insert(mock_db, _make_doc("a", content="v1", superseded_by="b", status="superseded"))
    _insert(mock_db, _make_doc("b", content="v2", superseded_by="c", status="superseded"))
    _insert(mock_db, _make_doc("c", content="v3"))

    chain = engine.get_supersession_chain("a")

    assert chain.memory_ids == ["a", "b", "c"]
    assert chain.current_id == "c"


# ── Current Version Tests ────────────────────────────────────────────


def test_get_current_version(mock_db: Any, engine: TemporalEngine) -> None:
    """get_current_version follows the chain to the tip."""
    _insert(mock_db, _make_doc("m1", content="old", superseded_by="m2", status="superseded"))
    _insert(mock_db, _make_doc("m2", content="new"))

    result = engine.get_current_version("m1")

    assert result is not None
    assert result.id == "m2"
    assert result.content == "new"


def test_get_current_version_not_found(mock_db: Any, engine: TemporalEngine) -> None:
    """Returns None when the starting memory_id does not exist."""
    result = engine.get_current_version("nonexistent")

    assert result is None


# ── Contradiction Detection Tests ────────────────────────────────────


def test_detect_contradictions(mock_db: Any, engine: TemporalEngine) -> None:
    """Two active memories with same entity but different content are contradictions."""
    _insert(mock_db, _make_doc("m1", content="Alice lives in NYC", entity="Alice"))
    _insert(mock_db, _make_doc("m2", content="Alice lives in LA", entity="Alice"))

    results = engine.detect_contradictions("Alice")

    assert len(results) == 1
    c = results[0]
    assert {c.memory_a_id, c.memory_b_id} == {"m1", "m2"}
    assert "Alice" in c.reason
    assert c.confidence == 1.0


def test_detect_contradictions_no_conflicts(mock_db: Any, engine: TemporalEngine) -> None:
    """One superseded memory should not cause a contradiction with an active one."""
    _insert(mock_db, _make_doc(
        "m1", content="Alice lives in NYC", entity="Alice", status="superseded",
    ))
    _insert(mock_db, _make_doc("m2", content="Alice lives in LA", entity="Alice"))

    results = engine.detect_contradictions("Alice")

    assert len(results) == 0


# ── Temporal Range Query Tests ───────────────────────────────────────


def test_query_temporal_range(mock_db: Any, engine: TemporalEngine) -> None:
    """Filter memories by start/end date range."""
    _insert(mock_db, _make_doc("m1", valid_from="2025-01-01T00:00:00+00:00"))
    _insert(mock_db, _make_doc("m2", valid_from="2025-06-01T00:00:00+00:00"))
    _insert(mock_db, _make_doc("m3", valid_from="2025-12-01T00:00:00+00:00"))

    query = TemporalQuery(
        start="2025-03-01T00:00:00+00:00",
        end="2025-09-01T00:00:00+00:00",
    )
    results = engine.query_temporal_range(query)

    assert len(results) == 1
    assert results[0].id == "m2"


def test_query_temporal_range_with_entity(mock_db: Any, engine: TemporalEngine) -> None:
    """Entity filter combined with date range."""
    _insert(mock_db, _make_doc(
        "m1", entity="Alice", valid_from="2025-06-01T00:00:00+00:00",
    ))
    _insert(mock_db, _make_doc(
        "m2", entity="Bob", valid_from="2025-06-01T00:00:00+00:00",
    ))

    query = TemporalQuery(
        entity="Alice",
        start="2025-01-01T00:00:00+00:00",
        end="2025-12-01T00:00:00+00:00",
    )
    results = engine.query_temporal_range(query)

    assert len(results) == 1
    assert results[0].id == "m1"
    assert results[0].entity == "Alice"


# ── History Tests ────────────────────────────────────────────────────


def test_get_history(mock_db: Any, engine: TemporalEngine) -> None:
    """get_history returns the full chain in chronological order."""
    _insert(mock_db, _make_doc(
        "m1", content="v1", superseded_by="m2", status="superseded",
        valid_from="2025-01-01T00:00:00+00:00",
    ))
    _insert(mock_db, _make_doc(
        "m2", content="v2", superseded_by="m3", status="superseded",
        valid_from="2025-02-01T00:00:00+00:00",
    ))
    _insert(mock_db, _make_doc(
        "m3", content="v3",
        valid_from="2025-03-01T00:00:00+00:00",
    ))

    history = engine.get_history("m2")

    assert len(history) == 3
    assert [m.id for m in history] == ["m1", "m2", "m3"]
    assert history[0].content == "v1"
    assert history[2].content == "v3"


# ── Circular Protection Test ────────────────────────────────────────


def test_chain_circular_protection(mock_db: Any, engine: TemporalEngine) -> None:
    """Circular superseded_by links do not cause infinite loops."""
    _insert(mock_db, _make_doc("x", superseded_by="y", status="superseded"))
    _insert(mock_db, _make_doc("y", superseded_by="x", status="superseded"))

    chain = engine.get_supersession_chain("x")

    # Should terminate; exact IDs depend on visit order but must be bounded
    assert len(chain.memory_ids) <= 2
    assert "x" in chain.memory_ids
    assert "y" in chain.memory_ids
