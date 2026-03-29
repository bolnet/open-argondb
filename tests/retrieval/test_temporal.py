"""Tests for TemporalLayer."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from open_argondb.models import Memory, RetrievalRequest, TemporalQuery
from open_argondb.retrieval.layers.temporal import TemporalLayer


class TestTemporalLayer:
    """TemporalLayer unit tests."""

    def test_temporal_date_extraction(self, mock_db) -> None:
        """ISO date found in query triggers temporal search."""
        mock_engine = MagicMock()
        mock_engine.query_temporal_range.return_value = [
            Memory(
                id="mem-temp1",
                content="Meeting on Jan 15",
                valid_from="2024-01-15T00:00:00Z",
            ),
        ]

        layer = TemporalLayer(mock_db, mock_engine)
        request = RetrievalRequest(query="what happened on 2024-01-15")
        results = layer.search(request)

        assert len(results) == 1
        assert results[0].match_source == "temporal"
        assert results[0].tier == 3
        mock_engine.query_temporal_range.assert_called_once()

        # Verify the TemporalQuery passed to the engine
        call_args = mock_engine.query_temporal_range.call_args[0][0]
        assert call_args.start == "2024-01-15"
        assert call_args.end == "2024-01-15"

    def test_temporal_no_dates(self, mock_db) -> None:
        """Returns empty when no date references found in query."""
        mock_engine = MagicMock()

        layer = TemporalLayer(mock_db, mock_engine)
        request = RetrievalRequest(query="tell me about cats")
        results = layer.search(request)

        assert results == []
        mock_engine.query_temporal_range.assert_not_called()

    def test_temporal_recency_scoring(self, mock_db) -> None:
        """Newer memories score higher than older ones."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        mock_engine = MagicMock()
        mock_engine.query_temporal_range.return_value = [
            Memory(
                id="mem-new",
                content="Recent event",
                valid_from=f"{today}T00:00:00Z",
            ),
            Memory(
                id="mem-old",
                content="Old event",
                valid_from="2020-01-01T00:00:00Z",
            ),
        ]

        layer = TemporalLayer(mock_db, mock_engine)
        request = RetrievalRequest(query=f"events on {today}")
        results = layer.search(request)

        assert len(results) == 2
        # Results sorted by score descending — newer should be first
        assert results[0].memory.id == "mem-new"
        assert results[1].memory.id == "mem-old"
        assert results[0].score > results[1].score
