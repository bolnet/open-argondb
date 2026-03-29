"""Tests for ExactMatchLayer."""

from __future__ import annotations

import json

import pytest

from open_argondb.models import RetrievalRequest
from open_argondb.retrieval.layers.exact import ExactMatchLayer


class TestExactMatchLayer:
    """ExactMatchLayer unit tests."""

    def test_exact_match_by_id(self, mock_db) -> None:
        """Memory ID lookup returns score 1.0."""
        mock_db.collection("memories").insert({
            "_key": "mem-001",
            "memory_id": "mem-001",
            "content": "Test memory content",
            "tags": json.dumps(["test"]),
            "category": "general",
            "status": "active",
            "created_at": "2024-01-01T00:00:00Z",
            "valid_from": "2024-01-01T00:00:00Z",
            "_deleted": False,
        })

        layer = ExactMatchLayer(mock_db)
        request = RetrievalRequest(query="mem-001")
        results = layer.search(request)

        assert len(results) == 1
        assert results[0].score == 1.0
        assert results[0].match_source == "exact"
        assert results[0].tier == 1
        assert results[0].memory.id == "mem-001"
        assert results[0].memory.content == "Test memory content"

    def test_exact_match_not_found(self, mock_db) -> None:
        """Returns empty list when memory ID does not exist."""
        mock_db.collection("memories")  # ensure collection exists

        layer = ExactMatchLayer(mock_db)
        request = RetrievalRequest(query="nonexistent-id")
        results = layer.search(request)

        assert results == []

    def test_exact_skip_query_with_spaces(self, mock_db) -> None:
        """Queries with spaces are not treated as IDs."""
        mock_db.collection("memories").insert({
            "_key": "mem-002",
            "memory_id": "mem-002",
            "content": "Another memory",
            "tags": json.dumps([]),
            "category": "general",
            "status": "active",
            "created_at": "2024-01-01T00:00:00Z",
            "valid_from": "2024-01-01T00:00:00Z",
            "_deleted": False,
        })

        layer = ExactMatchLayer(mock_db)
        request = RetrievalRequest(query="find something about cats")
        results = layer.search(request)

        assert results == []
