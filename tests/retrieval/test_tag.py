"""Tests for TagMatchLayer."""

from __future__ import annotations

import json

import pytest

from open_arangodb.models import RetrievalRequest
from open_arangodb.retrieval.layers.tag import TagMatchLayer


class TestTagMatchLayer:
    """TagMatchLayer unit tests."""

    def test_tag_match_jaccard(self, mock_db) -> None:
        """Correct Jaccard similarity scoring."""
        mock_db.collection("memories").insert({
            "_key": "mem-t1",
            "memory_id": "mem-t1",
            "content": "Python programming tips",
            "tags": json.dumps(["python", "programming", "tips"]),
            "entity": "coding",
            "category": "general",
            "status": "active",
            "created_at": "2024-01-01T00:00:00Z",
            "valid_from": "2024-01-01T00:00:00Z",
            "_deleted": False,
        })

        layer = TagMatchLayer(mock_db)
        # Request tags: ["python", "tips"] -> intersection=2, union=3 -> Jaccard=2/3
        request = RetrievalRequest(
            query="python tips",
            tags=["python", "tips"],
        )
        results = layer.search(request)

        assert len(results) == 1
        assert abs(results[0].score - 2.0 / 3.0) < 1e-9
        assert results[0].match_source == "tag"
        assert results[0].tier == 2

    def test_tag_match_entity(self, mock_db) -> None:
        """Entity-only match returns results with base score."""
        mock_db.collection("memories").insert({
            "_key": "mem-t2",
            "memory_id": "mem-t2",
            "content": "Alice's preferences",
            "tags": json.dumps([]),
            "entity": "Alice",
            "category": "general",
            "status": "active",
            "created_at": "2024-01-01T00:00:00Z",
            "valid_from": "2024-01-01T00:00:00Z",
            "_deleted": False,
        })

        layer = TagMatchLayer(mock_db)
        request = RetrievalRequest(
            query="tell me about Alice",
            entity="Alice",
        )
        results = layer.search(request)

        assert len(results) == 1
        assert results[0].score == 0.5
        assert results[0].memory.entity == "Alice"

    def test_tag_no_criteria(self, mock_db) -> None:
        """Returns empty when no tags or entity provided."""
        mock_db.collection("memories").insert({
            "_key": "mem-t3",
            "memory_id": "mem-t3",
            "content": "Orphan memory",
            "tags": json.dumps(["orphan"]),
            "category": "general",
            "status": "active",
            "created_at": "2024-01-01T00:00:00Z",
            "valid_from": "2024-01-01T00:00:00Z",
            "_deleted": False,
        })

        layer = TagMatchLayer(mock_db)
        request = RetrievalRequest(query="something")
        results = layer.search(request)

        assert results == []
