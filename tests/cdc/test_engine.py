"""Unit tests for CDCEngine."""

from __future__ import annotations

import pytest

from open_arangodb.cdc.engine import CDCEngine
from open_arangodb.models import ChangeOp, Memory


# ── Helpers ──────────────────────────────────────────────────────────


def _make_memory(mid: str = "mem-1", content: str = "test") -> Memory:
    return Memory(id=mid, content=content, tags=["t1"], category="general")


# ── record_change ────────────────────────────────────────────────────


class TestRecordChange:
    def test_insert_op(self, cdc_engine: CDCEngine) -> None:
        mem = _make_memory()
        rev = cdc_engine.record_change("insert", mem.id, after=mem)
        assert rev  # non-empty revision string

        changes = cdc_engine.get_changes()
        assert len(changes) == 1
        assert changes[0].op == ChangeOp.INSERT
        assert changes[0].memory_id == "mem-1"
        assert changes[0].after is not None
        assert changes[0].before is None

    def test_update_op(self, cdc_engine: CDCEngine) -> None:
        old = _make_memory(content="old")
        new = _make_memory(content="new")
        cdc_engine.record_change("update", "mem-1", before=old, after=new)

        changes = cdc_engine.get_changes()
        assert changes[0].op == ChangeOp.UPDATE
        assert changes[0].before is not None
        assert changes[0].after is not None

    def test_delete_op(self, cdc_engine: CDCEngine) -> None:
        mem = _make_memory()
        cdc_engine.record_change("delete", "mem-1", before=mem)

        changes = cdc_engine.get_changes()
        assert changes[0].op == ChangeOp.DELETE
        assert changes[0].before is not None
        assert changes[0].after is None

    def test_supersede_op(self, cdc_engine: CDCEngine) -> None:
        old = _make_memory(mid="mem-1", content="old")
        new = _make_memory(mid="mem-2", content="new")
        cdc_engine.record_change("supersede", "mem-1", before=old, after=new)

        changes = cdc_engine.get_changes()
        assert changes[0].op == ChangeOp.SUPERSEDE


# ── get_changes with filters ────────────────────────────────────────


class TestGetChanges:
    def test_get_all_changes(self, cdc_engine: CDCEngine) -> None:
        for i in range(3):
            cdc_engine.record_change("insert", f"mem-{i}", after=_make_memory(f"mem-{i}"))
        changes = cdc_engine.get_changes()
        assert len(changes) == 3

    def test_since_timestamp(self, cdc_engine: CDCEngine) -> None:
        # Record a change, capture its timestamp, then record another
        cdc_engine.record_change("insert", "mem-0", after=_make_memory("mem-0"))
        changes_before = cdc_engine.get_changes()
        ts = changes_before[0].timestamp

        cdc_engine.record_change("insert", "mem-1", after=_make_memory("mem-1"))

        changes_after = cdc_engine.get_changes(since_timestamp=ts)
        # Only the second change should be returned (timestamp strictly greater)
        assert len(changes_after) >= 1
        memory_ids = [c.memory_id for c in changes_after]
        assert "mem-1" in memory_ids

    def test_since_rev(self, cdc_engine: CDCEngine) -> None:
        rev1 = cdc_engine.record_change("insert", "mem-0", after=_make_memory("mem-0"))
        cdc_engine.record_change("insert", "mem-1", after=_make_memory("mem-1"))

        changes = cdc_engine.get_changes(since_rev=rev1)
        assert len(changes) >= 1
        memory_ids = [c.memory_id for c in changes]
        assert "mem-1" in memory_ids

    def test_limit(self, cdc_engine: CDCEngine) -> None:
        for i in range(5):
            cdc_engine.record_change("insert", f"mem-{i}", after=_make_memory(f"mem-{i}"))
        changes = cdc_engine.get_changes(limit=2)
        assert len(changes) == 2


# ── Checkpoints ──────────────────────────────────────────────────────


class TestCheckpoints:
    def test_save_and_get_checkpoint(self, cdc_engine: CDCEngine) -> None:
        cdc_engine.save_checkpoint("consumer-1", "rev-abc")
        result = cdc_engine.get_checkpoint("consumer-1")
        assert result == "rev-abc"

    def test_get_checkpoint_missing(self, cdc_engine: CDCEngine) -> None:
        result = cdc_engine.get_checkpoint("unknown-consumer")
        assert result is None

    def test_save_checkpoint_overwrites(self, cdc_engine: CDCEngine) -> None:
        cdc_engine.save_checkpoint("consumer-1", "rev-1")
        cdc_engine.save_checkpoint("consumer-1", "rev-2")
        result = cdc_engine.get_checkpoint("consumer-1")
        assert result == "rev-2"


# ── TTL field ────────────────────────────────────────────────────────


class TestTTL:
    def test_expires_at_is_set(self, cdc_engine: CDCEngine) -> None:
        cdc_engine.record_change("insert", "mem-1", after=_make_memory())
        col = cdc_engine._db.collection(CDCEngine.CHANGELOG)
        docs = col.all()
        assert len(docs) == 1
        assert "expires_at" in docs[0]
        assert docs[0]["expires_at"]  # non-empty

    def test_custom_ttl(self, cdc_engine: CDCEngine) -> None:
        cdc_engine.record_change("insert", "mem-1", after=_make_memory(), ttl_seconds=3600)
        col = cdc_engine._db.collection(CDCEngine.CHANGELOG)
        docs = col.all()
        assert docs[0]["expires_at"]  # present and non-empty


# ── Reset ────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_changelog(self, cdc_engine: CDCEngine) -> None:
        cdc_engine.record_change("insert", "mem-1", after=_make_memory())
        cdc_engine.save_checkpoint("c1", "r1")
        cdc_engine.reset()

        assert cdc_engine.get_changes() == []
        assert cdc_engine.get_checkpoint("c1") is None
