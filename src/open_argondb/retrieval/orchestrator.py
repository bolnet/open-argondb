"""Retrieval orchestrator — runs enabled layers, fuses with RRF."""

from __future__ import annotations

from typing import Any

from open_argondb.models import RetrievalConfig, RetrievalRequest, RetrievalResult
from open_argondb.retrieval.fusion import RRFFusion


class RetrievalOrchestrator:
    """Runs enabled retrieval layers, fuses results with RRF."""

    def __init__(
        self,
        db: Any,
        vector_search: Any | None = None,
        temporal_engine: Any | None = None,
        scope_manager: Any | None = None,
    ) -> None:
        self._db = db
        self._vector = vector_search
        self._temporal = temporal_engine
        self._scope = scope_manager
        self._fusion = RRFFusion()

    def retrieve(self, request: RetrievalRequest) -> list[RetrievalResult]:
        """Run configured layers, fuse, filter by scope, return top-N."""
        config = request.config or RetrievalConfig()

        layer_results: dict[str, list[RetrievalResult]] = {}
        for layer_name in config.layers:
            layer = self._get_layer(layer_name)
            if layer is not None:
                results = layer.search(request)
                if results:
                    layer_results[layer_name] = results

        fused = self._fusion.fuse(layer_results, k=config.rrf_k)

        # Filter by min_score
        fused = [r for r in fused if r.score >= config.min_score]

        return fused[: config.max_results]

    def _get_layer(self, name: str) -> Any | None:
        """Lazily instantiate layers based on name and available dependencies."""
        if name == "exact":
            from open_argondb.retrieval.layers.exact import ExactMatchLayer

            return ExactMatchLayer(self._db)

        if name == "tag":
            from open_argondb.retrieval.layers.tag import TagMatchLayer

            return TagMatchLayer(self._db)

        if name == "semantic":
            if self._vector is None:
                return None
            from open_argondb.retrieval.layers.semantic import SemanticLayer

            return SemanticLayer(self._vector)

        if name == "temporal":
            if self._temporal is None:
                return None
            from open_argondb.retrieval.layers.temporal import TemporalLayer

            return TemporalLayer(self._db, self._temporal)

        return None
