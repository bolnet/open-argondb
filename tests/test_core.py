"""Unit tests for ArangoDB core gateway."""

from __future__ import annotations

import sys
from dataclasses import replace
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from open_arangodb.models import AgentScope, Memory, Visibility
from conftest import MockDatabase


# ── Helpers ──────────────────────────────────────────────────────────


def _fake_model(dimension: int = 4) -> MagicMock:
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = dimension

    def _encode(text, normalize_embeddings=True):
        if isinstance(text, str):
            seed = sum(ord(c) for c in text) % 1000
            rng = np.random.RandomState(seed)
            vec = rng.randn(dimension).astype(np.float32)
            if normalize_embeddings:
                vec = vec / np.linalg.norm(vec)
            return vec
        else:
            results = []
            for t in text:
                seed = sum(ord(c) for c in t) % 1000
                rng = np.random.RandomState(seed)
                vec = rng.randn(dimension).astype(np.float32)
                if normalize_embeddings:
                    vec = vec / np.linalg.norm(vec)
                results.append(vec)
            return np.array(results)

    model.encode = MagicMock(side_effect=_encode)
    return model


def _install_fake_sentence_transformers() -> None:
    """Inject a fake sentence_transformers module into sys.modules."""
    fake_st = ModuleType("sentence_transformers")
    fake_st.SentenceTransformer = MagicMock(side_effect=lambda *a, **kw: _fake_model())
    sys.modules["sentence_transformers"] = fake_st


def _make_memory(mid: str = "mem-1", content: str = "test content") -> Memory:
    return Memory(id=mid, content=content, tags=["t1"], category="general")


@pytest.fixture
def mock_db() -> MockDatabase:
    return MockDatabase()


@pytest.fixture
def argon_db(mock_db: MockDatabase):
    """Create ArangoDB instance with mocked ArangoClient and SentenceTransformer."""
    _install_fake_sentence_transformers()
    with patch("open_arangodb.core.ArangoClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.db.return_value = mock_db

        from open_arangodb.core import ArangoDB
        db = ArangoDB(
            host="http://localhost:8529",
            database="test",
            audit_enabled=True,
            cdc_enabled=True,
        )

    return db


@pytest.fixture
def argon_db_no_audit(mock_db: MockDatabase):
    _install_fake_sentence_transformers()
    with patch("open_arangodb.core.ArangoClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.db.return_value = mock_db

        from open_arangodb.core import ArangoDB
        db = ArangoDB(
            host="http://localhost:8529",
            database="test",
            audit_enabled=False,
            cdc_enabled=True,
        )
    return db


@pytest.fixture
def argon_db_no_cdc(mock_db: MockDatabase):
    _install_fake_sentence_transformers()
    with patch("open_arangodb.core.ArangoClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.db.return_value = mock_db

        from open_arangodb.core import ArangoDB
        db = ArangoDB(
            host="http://localhost:8529",
            database="test",
            audit_enabled=True,
            cdc_enabled=False,
        )
    return db


# ── Insert ───────────────────────────────────────────────────────────


class TestInsert:
    def test_insert_returns_memory(self, argon_db) -> None:
        mem = _make_memory()
        result = argon_db.insert(mem)
        assert result.id == "mem-1"
        assert result.content == "test content"

    def test_insert_triggers_audit(self, argon_db) -> None:
        mem = _make_memory()
        argon_db.insert(mem)
        entries = argon_db._audit.query()
        assert len(entries) == 1
        assert entries[0]["op"] == "insert"

    def test_insert_triggers_cdc(self, argon_db) -> None:
        mem = _make_memory()
        argon_db.insert(mem)
        changes = argon_db._cdc.get_changes()
        assert len(changes) == 1
        assert changes[0].op.value == "insert"

    def test_insert_triggers_event(self, argon_db) -> None:
        received: list[dict] = []
        argon_db.events.subscribe("memory.created", lambda t, d: received.append(d))

        argon_db.insert(_make_memory())
        assert len(received) == 1
        assert received[0]["memory_id"] == "mem-1"

    def test_insert_with_scope(self, argon_db) -> None:
        scope = AgentScope(agent_id="a1", visibility=Visibility.PRIVATE)
        mem = _make_memory()
        result = argon_db.insert(mem, scope=scope)
        assert result.scope is not None
        assert result.scope.agent_id == "a1"


# ── Update ───────────────────────────────────────────────────────────


class TestUpdate:
    def test_update_persists(self, argon_db) -> None:
        mem = _make_memory()
        argon_db.insert(mem)
        updated = replace(mem, content="new content")
        argon_db.update(updated)

        result = argon_db.get("mem-1")
        assert result is not None
        assert result.content == "new content"

    def test_update_triggers_audit(self, argon_db) -> None:
        argon_db.insert(_make_memory())
        argon_db.update(replace(_make_memory(), content="v2"))
        entries = argon_db._audit.query(op="update")
        assert len(entries) == 1

    def test_update_triggers_cdc(self, argon_db) -> None:
        argon_db.insert(_make_memory())
        argon_db.update(replace(_make_memory(), content="v2"))
        changes = argon_db._cdc.get_changes()
        update_changes = [c for c in changes if c.op.value == "update"]
        assert len(update_changes) == 1

    def test_update_triggers_event(self, argon_db) -> None:
        argon_db.insert(_make_memory())
        received: list[dict] = []
        argon_db.events.subscribe("memory.updated", lambda t, d: received.append(d))
        argon_db.update(replace(_make_memory(), content="v2"))
        assert len(received) == 1


# ── Delete ───────────────────────────────────────────────────────────


class TestDelete:
    def test_delete_soft_deletes(self, argon_db) -> None:
        argon_db.insert(_make_memory())
        argon_db.delete("mem-1")
        assert argon_db.get("mem-1") is None

    def test_delete_triggers_audit(self, argon_db) -> None:
        argon_db.insert(_make_memory())
        argon_db.delete("mem-1")
        entries = argon_db._audit.query(op="delete")
        assert len(entries) == 1

    def test_delete_triggers_cdc(self, argon_db) -> None:
        argon_db.insert(_make_memory())
        argon_db.delete("mem-1")
        changes = argon_db._cdc.get_changes()
        delete_changes = [c for c in changes if c.op.value == "delete"]
        assert len(delete_changes) == 1

    def test_delete_triggers_event(self, argon_db) -> None:
        argon_db.insert(_make_memory())
        received: list[dict] = []
        argon_db.events.subscribe("memory.deleted", lambda t, d: received.append(d))
        argon_db.delete("mem-1")
        assert len(received) == 1


# ── Supersede ────────────────────────────────────────────────────────


class TestSupersede:
    def test_supersede_marks_old_and_inserts_new(self, argon_db) -> None:
        old = _make_memory(mid="old-1", content="old")
        argon_db.insert(old)

        new = _make_memory(mid="new-1", content="new")
        result = argon_db.supersede("old-1", new)
        assert result.id == "new-1"

        # Old is superseded
        raw = argon_db._store._col.get("old-1")
        assert raw["status"] == "superseded"
        assert raw["superseded_by"] == "new-1"

    def test_supersede_triggers_event(self, argon_db) -> None:
        argon_db.insert(_make_memory(mid="old-1"))
        received: list[dict] = []
        argon_db.events.subscribe("memory.superseded", lambda t, d: received.append(d))

        argon_db.supersede("old-1", _make_memory(mid="new-1"))
        assert len(received) == 1
        assert received[0]["old_id"] == "old-1"
        assert received[0]["new_id"] == "new-1"


# ── Get / Search / List ─────────────────────────────────────────────


class TestRead:
    def test_get_returns_memory(self, argon_db) -> None:
        argon_db.insert(_make_memory())
        result = argon_db.get("mem-1")
        assert result is not None
        assert result.id == "mem-1"

    def test_get_missing(self, argon_db) -> None:
        assert argon_db.get("nope") is None

    def test_search_returns_results(self, argon_db) -> None:
        argon_db.insert(_make_memory(mid="m1", content="cats are great"))
        argon_db.embed("m1", "cats are great")

        results = argon_db.search("cats")
        assert len(results) > 0

    def test_list_memories(self, argon_db) -> None:
        argon_db.insert(_make_memory(mid="m1"))
        argon_db.insert(_make_memory(mid="m2"))
        results = argon_db.list_memories()
        assert len(results) == 2


# ── Optional Features Disabled ───────────────────────────────────────


class TestOptionalFeatures:
    def test_audit_disabled(self, argon_db_no_audit) -> None:
        argon_db_no_audit.insert(_make_memory())
        assert argon_db_no_audit._audit is None  # no audit logger

    def test_cdc_disabled(self, argon_db_no_cdc) -> None:
        argon_db_no_cdc.insert(_make_memory())
        assert argon_db_no_cdc._cdc is None

    def test_get_changes_raises_without_cdc(self, argon_db_no_cdc) -> None:
        with pytest.raises(RuntimeError, match="CDC is not enabled"):
            argon_db_no_cdc.get_changes()


# ── Reset / Close ────────────────────────────────────────────────────


class TestLifecycle:
    def test_reset_clears_everything(self, argon_db) -> None:
        argon_db.insert(_make_memory())
        argon_db.reset()
        assert argon_db.get("mem-1") is None

    def test_close_does_not_raise(self, argon_db) -> None:
        argon_db.close()
