"""Semantic retrieval layer — delegates to VectorSearch."""

from __future__ import annotations

from typing import Any

from open_arangodb.models import RetrievalRequest, RetrievalResult


class SemanticLayer:
    """Wraps VectorSearch to produce RetrievalResult objects."""

    def __init__(self, vector_search: Any) -> None:
        self._vector = vector_search

    def search(self, request: RetrievalRequest) -> list[RetrievalResult]:
        """Delegate to vector_search.search() and convert results."""
        raw_results = self._vector.search(request.query)
        results: list[RetrievalResult] = []

        for item in raw_results:
            distance = item.get("distance", 0.0)
            score = max(0.0, 1.0 - distance)

            # Build a minimal Memory from vector search result
            from open_arangodb.models import Memory

            memory = Memory(
                id=item.get("memory_id", ""),
                content=item.get("content", ""),
            )
            results.append(
                RetrievalResult(
                    memory=memory,
                    score=score,
                    match_source="semantic",
                    tier=3,
                )
            )

        return results
