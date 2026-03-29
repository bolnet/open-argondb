"""Tests for RRFFusion."""

from __future__ import annotations

import pytest

from open_arangodb.models import Memory, RetrievalResult
from open_arangodb.retrieval.fusion import RRFFusion


def _make_result(mid: str, score: float, source: str = "test") -> RetrievalResult:
    """Helper to create a RetrievalResult with a minimal Memory."""
    return RetrievalResult(
        memory=Memory(id=mid, content=f"Content for {mid}"),
        score=score,
        match_source=source,
    )


class TestRRFFusion:
    """RRFFusion unit tests."""

    def test_rrf_single_layer(self) -> None:
        """Single layer results pass through with RRF scoring."""
        fusion = RRFFusion()
        layer_results = {
            "exact": [
                _make_result("m1", 1.0, "exact"),
                _make_result("m2", 0.8, "exact"),
            ],
        }
        fused = fusion.fuse(layer_results, k=60)

        assert len(fused) == 2
        # rank 1: 1/(60+1), rank 2: 1/(60+2)
        assert abs(fused[0].score - 1.0 / 61) < 1e-9
        assert abs(fused[1].score - 1.0 / 62) < 1e-9

    def test_rrf_multi_layer(self) -> None:
        """Correct RRF math across multiple layers."""
        fusion = RRFFusion()
        layer_results = {
            "exact": [_make_result("m1", 1.0, "exact")],
            "tag": [_make_result("m1", 0.5, "tag"), _make_result("m2", 0.3, "tag")],
        }
        fused = fusion.fuse(layer_results, k=60)

        # m1 appears in both: 1/(60+1) + 1/(60+1) = 2/61
        # m2 appears in tag only at rank 2: 1/(60+2) = 1/62
        m1_result = next(r for r in fused if r.memory.id == "m1")
        m2_result = next(r for r in fused if r.memory.id == "m2")

        assert abs(m1_result.score - 2.0 / 61) < 1e-9
        assert abs(m2_result.score - 1.0 / 62) < 1e-9
        # m1 should rank higher
        assert fused[0].memory.id == "m1"

    def test_rrf_deduplication(self) -> None:
        """Same memory from multiple layers keeps highest individual score source."""
        fusion = RRFFusion()
        layer_results = {
            "tag": [_make_result("m1", 0.3, "tag")],
            "exact": [_make_result("m1", 1.0, "exact")],
        }
        fused = fusion.fuse(layer_results, k=60)

        assert len(fused) == 1
        # Should keep the "exact" source since it had the higher individual score
        assert fused[0].match_source == "exact"

    def test_rrf_empty(self) -> None:
        """No results returns empty list."""
        fusion = RRFFusion()
        fused = fusion.fuse({}, k=60)

        assert fused == []
