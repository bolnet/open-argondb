"""Integration tests — exercises the full ArangoDB gateway with all modules enabled."""

from __future__ import annotations

import sys
from dataclasses import replace
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from conftest import MockDatabase
from open_arangodb.models import (
    AgentScope,
    EdgeDefinition,
    GraphConfig,
    Memory,
    RetrievalConfig,
    RetrievalRequest,
    SatelliteConfig,
    Visibility,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _fake_model(dimension: int = 4) -> MagicMock:
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = dimension

    def _encode(text: Any, normalize_embeddings: bool = True) -> Any:
        if isinstance(text, str):
            seed = sum(ord(c) for c in text) % 1000
            rng = np.random.RandomState(seed)
            vec = rng.randn(dimension).astype(np.float32)
            if normalize_embeddings:
                vec = vec / np.linalg.norm(vec)
            return vec
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
    fake_st = ModuleType("sentence_transformers")
    fake_st.SentenceTransformer = MagicMock(side_effect=lambda *a, **kw: _fake_model())
    sys.modules["sentence_transformers"] = fake_st


def _make_memory(
    mid: str = "mem-1",
    content: str = "test content",
    tags: list[str] | None = None,
    entity: str | None = None,
) -> Memory:
    return Memory(
        id=mid,
        content=content,
        tags=tags or ["t1"],
        category="general",
        entity=entity,
        status="active",
    )


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def mock_db() -> MockDatabase:
    return MockDatabase()


@pytest.fixture
def argon_full(mock_db: MockDatabase):
    """ArangoDB with all optional modules enabled."""
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
            graph_enabled=True,
            retrieval_enabled=True,
            temporal_enabled=True,
            backup_enabled=True,
            encryption_check=True,
        )
    return db


@pytest.fixture
def argon_minimal(mock_db: MockDatabase):
    """ArangoDB with no optional modules (backward-compat check)."""
    _install_fake_sentence_transformers()
    with patch("open_arangodb.core.ArangoClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.db.return_value = mock_db

        from open_arangodb.core import ArangoDB

        db = ArangoDB(
            host="http://localhost:8529",
            database="test",
        )
    return db


@pytest.fixture
def argon_with_satellite(mock_db: MockDatabase):
    """ArangoDB with satellite cache configured."""
    _install_fake_sentence_transformers()

    # Pre-populate a collection for satellite cache to sync from
    col = mock_db.create_collection("ref_data")
    col.insert({"_key": "item-1", "name": "Widget"})
    col.insert({"_key": "item-2", "name": "Gadget"})

    with patch("open_arangodb.core.ArangoClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.db.return_value = mock_db

        from open_arangodb.core import ArangoDB

        db = ArangoDB(
            host="http://localhost:8529",
            database="test",
            satellite_configs=[
                SatelliteConfig(collection="ref_data", ttl_seconds=60),
            ],
        )
    return db


# ── Test: Full Lifecycle ─────────────────────────────────────────────


class TestFullLifecycle:
    """Insert -> embed -> search -> supersede -> verify temporal chain."""

    def test_insert_embed_search_supersede(self, argon_full: Any) -> None:
        db = argon_full

        # Insert
        mem1 = _make_memory(mid="lc-1", content="Alice works at Acme", entity="Alice")
        result = db.insert(mem1)
        assert result.id == "lc-1"

        # Embed and search
        db.embed("lc-1", "Alice works at Acme")
        search_results = db.search("Alice Acme")
        assert len(search_results) > 0

        # Supersede
        mem2 = _make_memory(mid="lc-2", content="Alice works at Globex", entity="Alice")
        superseded = db.supersede("lc-1", mem2)
        assert superseded.id == "lc-2"

        # Verify temporal chain
        chain = db.get_supersession_chain("lc-1")
        assert "lc-1" in chain.memory_ids
        assert "lc-2" in chain.memory_ids
        assert chain.current_id == "lc-2"

        # Get current version
        current = db.get_current_version("lc-1")
        assert current is not None
        assert current.id == "lc-2"


# ── Test: Retrieval Pipeline ─────────────────────────────────────────


class TestRetrievalPipeline:
    """Insert memories with tags -> retrieve with multiple layers."""

    def test_tag_and_exact_retrieval(self, argon_full: Any) -> None:
        db = argon_full

        # Insert several memories with tags and entity
        db.insert(_make_memory(
            mid="ret-1", content="Bob likes cats", tags=["pets", "cats"], entity="Bob",
        ))
        db.insert(_make_memory(
            mid="ret-2", content="Bob likes dogs", tags=["pets", "dogs"], entity="Bob",
        ))
        db.insert(_make_memory(
            mid="ret-3", content="Carol likes birds", tags=["pets", "birds"], entity="Carol",
        ))

        # Retrieve by exact ID
        req_exact = RetrievalRequest(
            query="ret-1",
            config=RetrievalConfig(layers=["exact"], max_results=5),
        )
        exact_results = db.retrieve(req_exact)
        assert len(exact_results) == 1
        assert exact_results[0].memory.id == "ret-1"

        # Retrieve by tag overlap for entity Bob
        req_tags = RetrievalRequest(
            query="pets",
            entity="Bob",
            tags=["pets"],
            config=RetrievalConfig(layers=["tag"], max_results=10),
        )
        tag_results = db.retrieve(req_tags)
        assert len(tag_results) >= 2
        result_ids = {r.memory.id for r in tag_results}
        assert "ret-1" in result_ids
        assert "ret-2" in result_ids


# ── Test: Graph Operations ───────────────────────────────────────────


class TestGraphOperations:
    """Create graph -> insert vertices/edges -> traverse."""

    def test_create_and_traverse(self, argon_full: Any) -> None:
        db = argon_full

        config = GraphConfig(
            name="social",
            edge_definitions=[
                EdgeDefinition(
                    collection="knows",
                    from_vertex_collections=["people"],
                    to_vertex_collections=["people"],
                ),
            ],
        )
        db.create_graph(config)

        # Insert vertices via graph manager
        db._graph.insert_vertex("people", {"_key": "alice", "name": "Alice"})
        db._graph.insert_vertex("people", {"_key": "bob", "name": "Bob"})
        db._graph.insert_vertex("people", {"_key": "carol", "name": "Carol"})

        # Insert edges
        db._graph.insert_edge("knows", "people/alice", "people/bob")
        db._graph.insert_edge("knows", "people/bob", "people/carol")

        # Traverse from alice
        result = db.traverse(
            "people/alice",
            edge_collection="knows",
            direction="outbound",
            min_depth=1,
            max_depth=2,
        )
        vertex_keys = {v["_key"] for v in result.vertices}
        assert "bob" in vertex_keys
        assert "carol" in vertex_keys

    def test_parallel_traverse(self, argon_full: Any) -> None:
        db = argon_full

        config = GraphConfig(
            name="net",
            edge_definitions=[
                EdgeDefinition(
                    collection="links",
                    from_vertex_collections=["nodes"],
                    to_vertex_collections=["nodes"],
                ),
            ],
        )
        db.create_graph(config)

        db._graph.insert_vertex("nodes", {"_key": "n1", "name": "N1"})
        db._graph.insert_vertex("nodes", {"_key": "n2", "name": "N2"})
        db._graph.insert_vertex("nodes", {"_key": "n3", "name": "N3"})
        db._graph.insert_edge("links", "nodes/n1", "nodes/n2")
        db._graph.insert_edge("links", "nodes/n2", "nodes/n3")

        results = db.traverse_parallel(
            ["nodes/n1", "nodes/n2"],
            edge_collection="links",
            direction="outbound",
            min_depth=1,
            max_depth=1,
        )
        assert len(results) == 2
        # n1 -> n2
        assert len(results[0].vertices) >= 1
        # n2 -> n3
        assert len(results[1].vertices) >= 1


# ── Test: Feature Flags ──────────────────────────────────────────────


class TestFeatureFlags:
    """Disabled modules raise RuntimeError with descriptive message."""

    def test_retrieval_disabled(self, argon_minimal: Any) -> None:
        with pytest.raises(RuntimeError, match="Retrieval is not enabled"):
            argon_minimal.retrieve(RetrievalRequest(query="test"))

    def test_temporal_disabled(self, argon_minimal: Any) -> None:
        with pytest.raises(RuntimeError, match="Temporal is not enabled"):
            argon_minimal.get_supersession_chain("mem-1")

        with pytest.raises(RuntimeError, match="Temporal is not enabled"):
            argon_minimal.get_current_version("mem-1")

        with pytest.raises(RuntimeError, match="Temporal is not enabled"):
            argon_minimal.detect_contradictions("entity")

    def test_graph_disabled(self, argon_minimal: Any) -> None:
        config = GraphConfig(name="test")
        with pytest.raises(RuntimeError, match="Graph is not enabled"):
            argon_minimal.create_graph(config)

        with pytest.raises(RuntimeError, match="Graph is not enabled"):
            argon_minimal.traverse("v/1")

        with pytest.raises(RuntimeError, match="Graph is not enabled"):
            argon_minimal.traverse_parallel(["v/1"])

    def test_backup_disabled(self, argon_minimal: Any) -> None:
        from open_arangodb.models import BackupConfig

        with pytest.raises(RuntimeError, match="Backup is not enabled"):
            argon_minimal.create_backup(BackupConfig(output_dir="/tmp"))

    def test_encryption_disabled(self, argon_minimal: Any) -> None:
        with pytest.raises(RuntimeError, match="Encryption check is not enabled"):
            argon_minimal.check_encryption()

    def test_satellite_disabled(self, argon_minimal: Any) -> None:
        with pytest.raises(RuntimeError, match="No satellite cache for"):
            argon_minimal.get_satellite("nonexistent")

    def test_ldap_disabled(self, argon_minimal: Any) -> None:
        with pytest.raises(RuntimeError, match="LDAP auth is not enabled"):
            argon_minimal.authenticate("user", "pass")

    def test_existing_features_still_work(self, argon_minimal: Any) -> None:
        """Ensure backward compatibility: basic insert/get still works."""
        mem = _make_memory()
        result = argon_minimal.insert(mem)
        assert result.id == "mem-1"

        fetched = argon_minimal.get("mem-1")
        assert fetched is not None
        assert fetched.content == "test content"


# ── Test: Satellite Cache ────────────────────────────────────────────


class TestSatelliteIntegration:
    """Satellite cache wired through the gateway."""

    def test_get_satellite(self, argon_with_satellite: Any) -> None:
        sat = argon_with_satellite.get_satellite("ref_data")
        assert sat is not None

        # Should have synced on init
        doc = sat.get("item-1")
        assert doc is not None
        assert doc["name"] == "Widget"

    def test_satellite_stats(self, argon_with_satellite: Any) -> None:
        sat = argon_with_satellite.get_satellite("ref_data")
        sat.get("item-1")  # cache hit
        sat.get("item-1")  # cache hit
        stats = sat.stats()
        assert stats.cached_count == 2
        assert stats.hit_count >= 2


# ── Test: Close Cleanup ──────────────────────────────────────────────


class TestCloseCleanup:
    """close() shuts down all components without errors."""

    def test_close_full(self, argon_full: Any) -> None:
        # Insert something to exercise components
        argon_full.insert(_make_memory())
        argon_full.close()
        # Should not raise

    def test_close_minimal(self, argon_minimal: Any) -> None:
        argon_minimal.close()
        # Should not raise

    def test_close_with_satellites(self, argon_with_satellite: Any) -> None:
        argon_with_satellite.close()
        # Should not raise — satellite.stop() called

    def test_reset_with_graph(self, argon_full: Any) -> None:
        """reset() clears graph manager state."""
        config = GraphConfig(
            name="test_graph",
            edge_definitions=[
                EdgeDefinition(
                    collection="edges",
                    from_vertex_collections=["verts"],
                    to_vertex_collections=["verts"],
                ),
            ],
        )
        argon_full.create_graph(config)
        assert "test_graph" in argon_full._graph._graph_configs

        argon_full.reset()
        assert len(argon_full._graph._graph_configs) == 0


# ── Test: Contradiction Detection ────────────────────────────────────


class TestContradictions:
    """Temporal contradiction detection through the gateway."""

    def test_detect_contradictions(self, argon_full: Any) -> None:
        db = argon_full

        db.insert(_make_memory(
            mid="c-1", content="Alice is 30", entity="Alice",
        ))
        db.insert(_make_memory(
            mid="c-2", content="Alice is 31", entity="Alice",
        ))

        contradictions = db.detect_contradictions("Alice")
        assert len(contradictions) == 1
        assert contradictions[0].memory_a_id in ("c-1", "c-2")
        assert contradictions[0].memory_b_id in ("c-1", "c-2")
