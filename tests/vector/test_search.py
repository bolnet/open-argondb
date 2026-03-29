"""Unit tests for VectorSearch."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from open_argondb.models import Memory
from conftest import MockDatabase


# ── Helpers ──────────────────────────────────────────────────────────


def _fake_model(dimension: int = 4) -> MagicMock:
    """Create a mock SentenceTransformer with deterministic embeddings."""
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = dimension

    def _encode(text, normalize_embeddings=True):
        if isinstance(text, str):
            # Deterministic embedding from text hash
            seed = sum(ord(c) for c in text) % 1000
            rng = np.random.RandomState(seed)
            vec = rng.randn(dimension).astype(np.float32)
            if normalize_embeddings:
                vec = vec / np.linalg.norm(vec)
            return vec
        else:
            # Batch mode
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


@pytest.fixture
def mock_db() -> MockDatabase:
    return MockDatabase()


@pytest.fixture
def vector_search(mock_db: MockDatabase):
    fake_st = ModuleType("sentence_transformers")
    fake_st.SentenceTransformer = MagicMock(side_effect=lambda *a, **kw: _fake_model())
    sys.modules["sentence_transformers"] = fake_st

    from open_argondb.vector.search import VectorSearch
    # Force re-creation of embedding fn by clearing any cached state
    vs = VectorSearch(mock_db, model_name="test-model")
    return vs


def _seed_memory(mock_db: MockDatabase, mid: str, content: str) -> None:
    """Insert a bare memory doc into the mock DB for vector search to update."""
    from open_argondb.store.document_store import DocumentStore
    store = DocumentStore(mock_db)
    store.insert(Memory(id=mid, content=content))


# ── add() ────────────────────────────────────────────────────────────


class TestAdd:
    def test_add_stores_embedding(self, vector_search, mock_db: MockDatabase) -> None:
        _seed_memory(mock_db, "m1", "hello world")
        vector_search.add("m1", "hello world")

        doc = mock_db.collection("memories").get("m1")
        assert doc is not None
        assert doc["embedding"] is not None
        assert isinstance(doc["embedding"], list)
        assert len(doc["embedding"]) == 4

    def test_add_invalidates_cache(self, vector_search, mock_db: MockDatabase) -> None:
        _seed_memory(mock_db, "m1", "hello")
        vector_search._cache_matrix = np.array([[1.0]])  # fake cache
        vector_search.add("m1", "hello")
        assert vector_search._cache_matrix is None

    def test_add_without_existing_doc_creates_one(self, vector_search, mock_db: MockDatabase) -> None:
        # No pre-seeded memory doc with this key
        # The update will fail, so add() falls back to insert
        vector_search.add("new-mem", "brand new")
        doc = mock_db.collection("memories").get("new-mem")
        assert doc is not None
        assert doc["embedding"] is not None


# ── search() numpy path ─────────────────────────────────────────────


class TestSearchNumpy:
    def test_search_returns_results(self, vector_search, mock_db: MockDatabase) -> None:
        _seed_memory(mock_db, "m1", "cats are great")
        _seed_memory(mock_db, "m2", "dogs are friendly")
        vector_search.add("m1", "cats are great")
        vector_search.add("m2", "dogs are friendly")

        results = vector_search.search("cats", limit=5)
        assert len(results) > 0
        assert "memory_id" in results[0]
        assert "distance" in results[0]

    def test_search_empty_db(self, vector_search) -> None:
        results = vector_search.search("anything")
        assert results == []

    def test_search_respects_limit(self, vector_search, mock_db: MockDatabase) -> None:
        for i in range(5):
            _seed_memory(mock_db, f"m{i}", f"content {i}")
            vector_search.add(f"m{i}", f"content {i}")

        results = vector_search.search("query", limit=2)
        assert len(results) == 2

    def test_search_results_contain_memory_id_and_content(self, vector_search, mock_db: MockDatabase) -> None:
        _seed_memory(mock_db, "m1", "hello world")
        vector_search.add("m1", "hello world")

        results = vector_search.search("hello world", limit=1)
        assert len(results) == 1
        assert results[0]["memory_id"] == "m1"
        assert "distance" in results[0]


# ── batch_embed ──────────────────────────────────────────────────────


class TestBatchEmbed:
    def test_batch_embed_fills_missing(self, vector_search, mock_db: MockDatabase) -> None:
        _seed_memory(mock_db, "m1", "first")
        _seed_memory(mock_db, "m2", "second")

        count = vector_search.batch_embed()
        assert count == 2

        doc1 = mock_db.collection("memories").get("m1")
        doc2 = mock_db.collection("memories").get("m2")
        assert doc1["embedding"] is not None
        assert doc2["embedding"] is not None

    def test_batch_embed_skips_already_embedded(self, vector_search, mock_db: MockDatabase) -> None:
        _seed_memory(mock_db, "m1", "first")
        vector_search.add("m1", "first")  # already embedded

        _seed_memory(mock_db, "m2", "second")
        count = vector_search.batch_embed()
        assert count == 1  # only m2

    def test_batch_embed_empty_db(self, vector_search) -> None:
        count = vector_search.batch_embed()
        assert count == 0

    def test_batch_embed_invalidates_cache(self, vector_search, mock_db: MockDatabase) -> None:
        _seed_memory(mock_db, "m1", "test")
        vector_search._cache_matrix = np.array([[1.0]])
        vector_search.batch_embed()
        assert vector_search._cache_matrix is None
