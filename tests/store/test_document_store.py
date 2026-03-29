"""Unit tests for DocumentStore."""

from __future__ import annotations

import json

import pytest

from open_argondb.models import AgentScope, Memory, Visibility
from open_argondb.store.document_store import DocumentStore


# ── Helpers ──────────────────────────────────────────────────────────


def _make_memory(
    mid: str = "mem-1",
    content: str = "test content",
    **kwargs,
) -> Memory:
    defaults = {
        "id": mid,
        "content": content,
        "tags": ["tag1", "tag2"],
        "category": "general",
        "entity": "alice",
        "status": "active",
    }
    defaults.update(kwargs)
    return Memory(**defaults)


# ── Insert ───────────────────────────────────────────────────────────


class TestInsert:
    def test_insert_returns_memory(self, document_store: DocumentStore) -> None:
        mem = _make_memory()
        result = document_store.insert(mem)
        assert result.id == "mem-1"
        assert result.content == "test content"

    def test_insert_persists_to_collection(self, document_store: DocumentStore) -> None:
        mem = _make_memory()
        document_store.insert(mem)
        retrieved = document_store.get("mem-1")
        assert retrieved is not None
        assert retrieved.content == "test content"

    def test_insert_with_scope(self, document_store: DocumentStore) -> None:
        scope = AgentScope(agent_id="agent-1", session_id="sess-1", visibility=Visibility.PRIVATE)
        mem = _make_memory(scope=scope)
        document_store.insert(mem)
        retrieved = document_store.get("mem-1")
        assert retrieved is not None
        assert retrieved.scope is not None
        assert retrieved.scope.agent_id == "agent-1"
        assert retrieved.scope.session_id == "sess-1"
        assert retrieved.scope.visibility == Visibility.PRIVATE


# ── Update ───────────────────────────────────────────────────────────


class TestUpdate:
    def test_update_changes_content(self, document_store: DocumentStore) -> None:
        from dataclasses import replace

        mem = _make_memory()
        document_store.insert(mem)

        updated = replace(mem, content="updated content")
        document_store.update(updated)

        retrieved = document_store.get("mem-1")
        assert retrieved is not None
        assert retrieved.content == "updated content"

    def test_update_nonexistent_raises(self, document_store: DocumentStore) -> None:
        mem = _make_memory(mid="no-such-id")
        with pytest.raises(Exception):
            document_store.update(mem)


# ── Get ──────────────────────────────────────────────────────────────


class TestGet:
    def test_get_existing(self, document_store: DocumentStore) -> None:
        document_store.insert(_make_memory())
        result = document_store.get("mem-1")
        assert result is not None
        assert result.id == "mem-1"

    def test_get_missing_returns_none(self, document_store: DocumentStore) -> None:
        result = document_store.get("nonexistent")
        assert result is None

    def test_get_deleted_returns_none(self, document_store: DocumentStore) -> None:
        document_store.insert(_make_memory())
        document_store.soft_delete("mem-1")
        result = document_store.get("mem-1")
        assert result is None


# ── Soft Delete ──────────────────────────────────────────────────────


class TestSoftDelete:
    def test_soft_delete_marks_deleted(self, document_store: DocumentStore) -> None:
        document_store.insert(_make_memory())
        document_store.soft_delete("mem-1")
        # The raw document should still exist in the collection
        raw = document_store._col.get("mem-1")
        assert raw is not None
        assert raw["_deleted"] is True

    def test_soft_delete_hides_from_get(self, document_store: DocumentStore) -> None:
        document_store.insert(_make_memory())
        document_store.soft_delete("mem-1")
        assert document_store.get("mem-1") is None


# ── Mark Superseded ──────────────────────────────────────────────────


class TestMarkSuperseded:
    def test_mark_superseded_sets_fields(self, document_store: DocumentStore) -> None:
        document_store.insert(_make_memory())
        document_store.mark_superseded("mem-1", "mem-2")

        raw = document_store._col.get("mem-1")
        assert raw is not None
        assert raw["status"] == "superseded"
        assert raw["superseded_by"] == "mem-2"
        assert raw["valid_until"] is not None


# ── List Memories ────────────────────────────────────────────────────


class TestListMemories:
    def test_list_returns_active_only(self, document_store: DocumentStore) -> None:
        document_store.insert(_make_memory(mid="m1"))
        document_store.insert(_make_memory(mid="m2"))
        document_store.soft_delete("m1")

        results = document_store.list_memories()
        ids = [m.id for m in results]
        assert "m2" in ids
        assert "m1" not in ids

    def test_list_filters_by_entity(self, document_store: DocumentStore) -> None:
        document_store.insert(_make_memory(mid="m1", entity="alice"))
        document_store.insert(_make_memory(mid="m2", entity="bob"))

        results = document_store.list_memories(entity="alice")
        assert len(results) == 1
        assert results[0].entity == "alice"

    def test_list_filters_by_scope(self, document_store: DocumentStore) -> None:
        scope_a = AgentScope(agent_id="agent-a")
        scope_b = AgentScope(agent_id="agent-b")
        document_store.insert(_make_memory(mid="m1", scope=scope_a))
        document_store.insert(_make_memory(mid="m2", scope=scope_b))

        results = document_store.list_memories(scope=scope_a)
        assert len(results) == 1
        assert results[0].scope.agent_id == "agent-a"

    def test_list_respects_limit(self, document_store: DocumentStore) -> None:
        for i in range(5):
            document_store.insert(_make_memory(mid=f"m{i}"))
        results = document_store.list_memories(limit=2)
        assert len(results) == 2


# ── Roundtrip ────────────────────────────────────────────────────────


class TestRoundtrip:
    def test_memory_to_doc_and_back(self, document_store: DocumentStore) -> None:
        scope = AgentScope(
            agent_id="agent-1",
            session_id="sess-1",
            workflow_id="wf-1",
            visibility=Visibility.WORKFLOW,
        )
        mem = _make_memory(
            tags=["a", "b"],
            category="people",
            entity="alice",
            confidence=0.9,
            metadata={"source": "chat"},
            scope=scope,
        )

        doc = document_store._memory_to_doc(mem)
        assert doc["memory_id"] == "mem-1"
        assert doc["tags"] == json.dumps(["a", "b"])
        assert doc["scope_agent_id"] == "agent-1"
        assert doc["scope_visibility"] == "workflow"
        assert doc["_deleted"] is False

        restored = document_store._doc_to_memory(doc)
        assert restored.id == mem.id
        assert restored.content == mem.content
        assert restored.tags == ["a", "b"]
        assert restored.category == "people"
        assert restored.confidence == 0.9
        assert restored.metadata == {"source": "chat"}
        assert restored.scope.agent_id == "agent-1"
        assert restored.scope.visibility == Visibility.WORKFLOW

    def test_doc_to_memory_handles_string_tags(self, document_store: DocumentStore) -> None:
        doc = {
            "memory_id": "m1",
            "content": "hello",
            "tags": '["x", "y"]',
            "metadata": "{}",
        }
        mem = document_store._doc_to_memory(doc)
        assert mem.tags == ["x", "y"]

    def test_doc_to_memory_handles_broken_tags(self, document_store: DocumentStore) -> None:
        doc = {
            "memory_id": "m1",
            "content": "hello",
            "tags": "not-json",
            "metadata": "not-json",
        }
        mem = document_store._doc_to_memory(doc)
        assert mem.tags == []
        assert mem.metadata == {}


# ── Reset ────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_all_documents(self, document_store: DocumentStore) -> None:
        document_store.insert(_make_memory())
        document_store.reset()
        assert document_store.get("mem-1") is None
