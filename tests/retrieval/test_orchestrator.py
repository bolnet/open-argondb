"""Tests for RetrievalOrchestrator."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from open_arangodb.models import (
    Memory,
    RetrievalConfig,
    RetrievalRequest,
)
from open_arangodb.retrieval.orchestrator import RetrievalOrchestrator


class TestRetrievalOrchestrator:
    """RetrievalOrchestrator unit tests."""

    def _insert_memory(
        self, mock_db, key: str, content: str, tags: list[str] | None = None,
        entity: str | None = None, valid_from: str = "2024-01-01T00:00:00Z",
    ) -> None:
        """Helper to insert a test memory document."""
        mock_db.collection("memories").insert({
            "_key": key,
            "memory_id": key,
            "content": content,
            "tags": json.dumps(tags or []),
            "entity": entity,
            "category": "general",
            "status": "active",
            "created_at": "2024-01-01T00:00:00Z",
            "valid_from": valid_from,
            "_deleted": False,
        })

    def test_full_pipeline(self, mock_db) -> None:
        """All 4 layers contribute to fused results."""
        self._insert_memory(mock_db, "mem-full-1", "Memory one", tags=["alpha"])

        mock_vector = MagicMock()
        mock_vector.search.return_value = [
            {"memory_id": "mem-full-1", "content": "Memory one", "distance": 0.3},
        ]

        mock_temporal = MagicMock()
        mock_temporal.query_temporal_range.return_value = [
            Memory(
                id="mem-full-1",
                content="Memory one",
                valid_from="2024-01-01T00:00:00Z",
            ),
        ]

        orchestrator = RetrievalOrchestrator(
            db=mock_db,
            vector_search=mock_vector,
            temporal_engine=mock_temporal,
        )

        config = RetrievalConfig(
            layers=["exact", "tag", "semantic", "temporal"],
        )
        request = RetrievalRequest(
            query="mem-full-1",
            tags=["alpha"],
            config=config,
        )
        results = orchestrator.retrieve(request)

        # mem-full-1 should appear (fused from multiple layers)
        assert len(results) >= 1
        assert any(r.memory.id == "mem-full-1" for r in results)

    def test_layer_config(self, mock_db) -> None:
        """Only enabled layers run."""
        self._insert_memory(mock_db, "mem-cfg-1", "Configured memory", tags=["beta"])

        mock_vector = MagicMock()

        orchestrator = RetrievalOrchestrator(
            db=mock_db,
            vector_search=mock_vector,
        )

        # Only run tag layer
        config = RetrievalConfig(layers=["tag"])
        request = RetrievalRequest(
            query="search",
            tags=["beta"],
            config=config,
        )
        results = orchestrator.retrieve(request)

        # Vector search should not have been called
        mock_vector.search.assert_not_called()
        # Should still find by tag
        assert len(results) >= 1

    def test_min_score_filter(self, mock_db) -> None:
        """Results below min_score are filtered out."""
        self._insert_memory(mock_db, "mem-ms-1", "Low score memory", tags=["gamma"])

        orchestrator = RetrievalOrchestrator(db=mock_db)

        config = RetrievalConfig(
            layers=["tag"],
            min_score=1.0,  # Very high threshold
        )
        request = RetrievalRequest(
            query="search",
            tags=["gamma"],
            config=config,
        )
        results = orchestrator.retrieve(request)

        # RRF score of a single result from one layer: 1/(60+1) ~ 0.016 < 1.0
        assert results == []

    def test_max_results_limit(self, mock_db) -> None:
        """Results are capped at max_results."""
        for i in range(10):
            self._insert_memory(
                mock_db,
                f"mem-lim-{i}",
                f"Memory {i}",
                tags=["shared"],
            )

        orchestrator = RetrievalOrchestrator(db=mock_db)

        config = RetrievalConfig(
            layers=["tag"],
            max_results=3,
        )
        request = RetrievalRequest(
            query="search",
            tags=["shared"],
            config=config,
        )
        results = orchestrator.retrieve(request)

        assert len(results) <= 3

    def test_missing_dependencies_skipped(self, mock_db) -> None:
        """Semantic layer is skipped when vector_search not provided."""
        self._insert_memory(mock_db, "mem-skip-1", "No vector", tags=["delta"])

        # No vector_search provided
        orchestrator = RetrievalOrchestrator(db=mock_db)

        config = RetrievalConfig(
            layers=["semantic", "tag"],
        )
        request = RetrievalRequest(
            query="search",
            tags=["delta"],
            config=config,
        )
        results = orchestrator.retrieve(request)

        # Should still return tag results even though semantic was skipped
        assert len(results) >= 1
        assert all(r.memory.id == "mem-skip-1" for r in results)
