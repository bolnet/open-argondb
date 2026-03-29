"""Reciprocal Rank Fusion across multiple retrieval layers."""

from __future__ import annotations

from open_argondb.models import RetrievalResult


class RRFFusion:
    """Reciprocal Rank Fusion across multiple retrieval layers."""

    def fuse(
        self,
        layer_results: dict[str, list[RetrievalResult]],
        k: int = 60,
    ) -> list[RetrievalResult]:
        """Fuse results from multiple layers using RRF.

        For each memory_id appearing in any layer's results:
        rrf_score = sum(1 / (k + rank)) across all layers where it appears

        Deduplicate by memory_id, keep the RetrievalResult from the layer
        that gave it the highest individual score.
        """
        # Track RRF scores and best result per memory_id
        rrf_scores: dict[str, float] = {}
        best_result: dict[str, RetrievalResult] = {}
        best_score: dict[str, float] = {}

        for _layer_name, results in layer_results.items():
            for rank, result in enumerate(results, start=1):
                mid = result.memory.id
                rrf_scores[mid] = rrf_scores.get(mid, 0.0) + 1.0 / (k + rank)

                # Keep the result with the highest individual score
                if mid not in best_result or result.score > best_score[mid]:
                    best_result[mid] = result
                    best_score[mid] = result.score

        # Build final list with RRF scores
        fused: list[RetrievalResult] = []
        for mid, rrf_score in rrf_scores.items():
            original = best_result[mid]
            fused.append(
                RetrievalResult(
                    memory=original.memory,
                    score=rrf_score,
                    match_source=original.match_source,
                    tier=original.tier,
                )
            )

        # Sort by RRF score descending
        return sorted(fused, key=lambda r: r.score, reverse=True)
