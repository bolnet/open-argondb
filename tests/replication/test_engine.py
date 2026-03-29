"""Tests for ReplicationEngine."""

from __future__ import annotations

import pytest

from open_argondb.cdc.engine import CDCEngine
from open_argondb.events.bus import InProcessBus
from open_argondb.models import Memory, ReplicationConfig, ReplicationStatus
from open_argondb.replication.engine import ReplicationEngine

try:
    from tests.conftest import MockDatabase
except ImportError:
    pass


@pytest.fixture
def source_db(mock_db):
    """Source database with CDC engine."""
    return mock_db


@pytest.fixture
def source_cdc(source_db) -> CDCEngine:
    return CDCEngine(source_db, InProcessBus())


@pytest.fixture
def target_db() -> MockDatabase:
    from tests.conftest import MockDatabase

    return MockDatabase("target_db")


@pytest.fixture
def repl_config() -> ReplicationConfig:
    return ReplicationConfig(
        source_host="localhost:8529",
        target_host="localhost:8530",
        batch_size=10,
        poll_interval_seconds=0.1,
    )


@pytest.fixture
def engine(source_cdc, target_db, repl_config) -> ReplicationEngine:
    return ReplicationEngine(source_cdc, target_db, repl_config)


def _make_memory(mid: str, content: str) -> Memory:
    return Memory(id=mid, content=content, tags=["test"], category="general")


class TestReplicationEngine:
    def test_replicate_batch(self, engine, source_cdc, target_db) -> None:
        """Changes from CDC should be applied to target."""
        mem = _make_memory("mem/1", "hello world")
        source_cdc.record_change("insert", "mem/1", after=mem)

        count = engine.replicate_batch()
        assert count == 1

        col = target_db.collection("memories")
        doc = col.get("mem_1")
        assert doc is not None
        assert doc["content"] == "hello world"

    def test_replicate_insert(self, engine, source_cdc, target_db) -> None:
        """Insert operation should create doc in target."""
        mem = _make_memory("mem/insert", "inserted")
        source_cdc.record_change("insert", "mem/insert", after=mem)

        engine.replicate_batch()

        doc = target_db.collection("memories").get("mem_insert")
        assert doc is not None
        assert doc["content"] == "inserted"

    def test_replicate_update(self, engine, source_cdc, target_db) -> None:
        """Update operation should update existing doc or upsert."""
        mem_v1 = _make_memory("mem/upd", "version 1")
        source_cdc.record_change("insert", "mem/upd", after=mem_v1)
        engine.replicate_batch()

        mem_v2 = _make_memory("mem/upd", "version 2")
        source_cdc.record_change("update", "mem/upd", before=mem_v1, after=mem_v2)
        engine.replicate_batch()

        doc = target_db.collection("memories").get("mem_upd")
        assert doc is not None
        assert doc["content"] == "version 2"

    def test_replicate_delete(self, engine, source_cdc, target_db) -> None:
        """Delete operation should soft-delete in target."""
        mem = _make_memory("mem/del", "to delete")
        source_cdc.record_change("insert", "mem/del", after=mem)
        engine.replicate_batch()

        source_cdc.record_change("delete", "mem/del", before=mem)
        engine.replicate_batch()

        doc = target_db.collection("memories").get("mem_del")
        assert doc is not None
        assert doc.get("_deleted") is True

    def test_status(self, engine) -> None:
        """status() should return correct state."""
        status = engine.status()
        assert isinstance(status, ReplicationStatus)
        assert status.state == "stopped"
        assert status.last_synced_rev is None

    def test_start_stop(self, engine) -> None:
        """start/stop lifecycle should update state."""
        engine.start()
        assert engine.status().state == "running"

        engine.stop()
        assert engine.status().state == "stopped"

    def test_pause_resume(self, engine) -> None:
        """pause/resume should toggle state."""
        engine.start()
        engine.pause()
        assert engine.status().state == "paused"

        engine.resume()
        assert engine.status().state == "running"

        engine.stop()

    def test_checkpoint_tracking(self, engine, source_cdc) -> None:
        """last_rev should update after each replicated change."""
        mem1 = _make_memory("mem/a", "first")
        mem2 = _make_memory("mem/b", "second")
        source_cdc.record_change("insert", "mem/a", after=mem1)
        source_cdc.record_change("insert", "mem/b", after=mem2)

        engine.replicate_batch()
        status = engine.status()
        assert status.last_synced_rev is not None
        assert status.last_synced_rev != ""
