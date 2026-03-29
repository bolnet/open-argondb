"""Unit tests for ScopeManager."""

from __future__ import annotations

import pytest

from open_argondb.models import AgentScope, Memory, Visibility
from open_argondb.scoping.manager import ScopeManager
from open_argondb.store.document_store import DocumentStore


# ── Helpers ──────────────────────────────────────────────────────────


def _make_memory(mid: str = "mem-1", content: str = "test") -> Memory:
    return Memory(id=mid, content=content)


def _make_scope(
    agent_id: str = "agent-1",
    workflow_id: str | None = None,
    visibility: Visibility = Visibility.GLOBAL,
) -> AgentScope:
    return AgentScope(
        agent_id=agent_id,
        workflow_id=workflow_id,
        visibility=visibility,
    )


# ── apply() ──────────────────────────────────────────────────────────


class TestApply:
    def test_apply_returns_new_memory_with_scope(self, scope_manager: ScopeManager) -> None:
        mem = _make_memory()
        scope = _make_scope()
        result = scope_manager.apply(mem, scope)

        assert result is not mem  # new object (frozen dataclass)
        assert result.scope == scope
        assert result.id == mem.id
        assert result.content == mem.content

    def test_apply_preserves_all_fields(self, scope_manager: ScopeManager) -> None:
        mem = Memory(
            id="m1",
            content="hello",
            tags=["a"],
            category="people",
            entity="alice",
            confidence=0.8,
        )
        scope = _make_scope(agent_id="a1", visibility=Visibility.PRIVATE)
        result = scope_manager.apply(mem, scope)

        assert result.tags == ["a"]
        assert result.category == "people"
        assert result.entity == "alice"
        assert result.confidence == 0.8
        assert result.scope.agent_id == "a1"
        assert result.scope.visibility == Visibility.PRIVATE

    def test_apply_replaces_existing_scope(self, scope_manager: ScopeManager) -> None:
        old_scope = _make_scope(agent_id="old")
        mem = Memory(id="mem-1", content="test", scope=old_scope)
        new_scope = _make_scope(agent_id="new")
        result = scope_manager.apply(mem, new_scope)
        assert result.scope.agent_id == "new"


# ── filter_results() ────────────────────────────────────────────────


class TestFilterResults:
    @pytest.fixture(autouse=True)
    def _seed_memories(self, mock_db) -> None:
        """Insert documents into mock DB for scope lookups."""
        store = DocumentStore(mock_db)
        scope_global = AgentScope(agent_id="agent-1", visibility=Visibility.GLOBAL)
        scope_private = AgentScope(agent_id="agent-1", visibility=Visibility.PRIVATE)
        scope_workflow = AgentScope(
            agent_id="agent-2",
            workflow_id="wf-1",
            visibility=Visibility.WORKFLOW,
        )
        scope_private_other = AgentScope(agent_id="agent-2", visibility=Visibility.PRIVATE)

        store.insert(Memory(id="global-1", content="g", scope=scope_global))
        store.insert(Memory(id="private-1", content="p", scope=scope_private))
        store.insert(Memory(id="wf-1", content="w", scope=scope_workflow))
        store.insert(Memory(id="private-other", content="po", scope=scope_private_other))

    def test_global_visible_to_all(self, scope_manager: ScopeManager) -> None:
        results = [{"memory_id": "global-1"}]
        scope = _make_scope(agent_id="anyone")
        filtered = scope_manager.filter_results(results, scope)
        assert len(filtered) == 1

    def test_private_visible_to_owner(self, scope_manager: ScopeManager) -> None:
        results = [{"memory_id": "private-1"}]
        scope = _make_scope(agent_id="agent-1")
        filtered = scope_manager.filter_results(results, scope)
        assert len(filtered) == 1

    def test_private_hidden_from_others(self, scope_manager: ScopeManager) -> None:
        results = [{"memory_id": "private-1"}]
        scope = _make_scope(agent_id="agent-999")
        filtered = scope_manager.filter_results(results, scope)
        assert len(filtered) == 0

    def test_workflow_visible_to_same_workflow(self, scope_manager: ScopeManager) -> None:
        results = [{"memory_id": "wf-1"}]
        scope = _make_scope(agent_id="agent-3", workflow_id="wf-1")
        filtered = scope_manager.filter_results(results, scope)
        assert len(filtered) == 1

    def test_workflow_hidden_from_different_workflow(self, scope_manager: ScopeManager) -> None:
        results = [{"memory_id": "wf-1"}]
        scope = _make_scope(agent_id="agent-3", workflow_id="wf-other")
        filtered = scope_manager.filter_results(results, scope)
        assert len(filtered) == 0

    def test_mixed_results_filtering(self, scope_manager: ScopeManager) -> None:
        results = [
            {"memory_id": "global-1"},
            {"memory_id": "private-1"},
            {"memory_id": "private-other"},
        ]
        scope = _make_scope(agent_id="agent-1")
        filtered = scope_manager.filter_results(results, scope)
        ids = [r["memory_id"] for r in filtered]
        assert "global-1" in ids
        assert "private-1" in ids
        assert "private-other" not in ids

    def test_unknown_memory_passes_through(self, scope_manager: ScopeManager) -> None:
        results = [{"memory_id": "nonexistent"}]
        scope = _make_scope(agent_id="agent-1")
        filtered = scope_manager.filter_results(results, scope)
        assert len(filtered) == 1  # no doc found -> passes through
