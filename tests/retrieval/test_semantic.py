"""Tests for SemanticLayer."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from open_argondb.models import RetrievalRequest
from open_argondb.retrieval.layers.semantic import SemanticLayer


class TestSemanticLayer:
    """SemanticLayer unit tests."""

    def test_semantic_delegates_to_vector(self) -> None:
        """Verify delegation to VectorSearch.search()."""
        mock_vector = MagicMock()
        mock_vector.search.return_value = [
            {"memory_id": "mem-s1", "content": "Semantic result", "distance": 0.2},
        ]

        layer = SemanticLayer(mock_vector)
        request = RetrievalRequest(query="find similar things")
        results = layer.search(request)

        mock_vector.search.assert_called_once_with("find similar things")
        assert len(results) == 1
        assert results[0].memory.id == "mem-s1"
        assert results[0].match_source == "semantic"
        assert results[0].tier == 3

    def test_semantic_score_conversion(self) -> None:
        """Distance to score: score = 1.0 - distance."""
        mock_vector = MagicMock()
        mock_vector.search.return_value = [
            {"memory_id": "mem-s2", "content": "Close match", "distance": 0.1},
            {"memory_id": "mem-s3", "content": "Far match", "distance": 0.8},
        ]

        layer = SemanticLayer(mock_vector)
        request = RetrievalRequest(query="test query")
        results = layer.search(request)

        assert len(results) == 2
        assert abs(results[0].score - 0.9) < 1e-9
        assert abs(results[1].score - 0.2) < 1e-9
