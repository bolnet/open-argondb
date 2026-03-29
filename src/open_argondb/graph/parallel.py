"""Parallel graph traversal using concurrent.futures."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from open_argondb.graph.manager import GraphManager
from open_argondb.models import TraversalResult

logger = logging.getLogger("open_argondb.graph")


class ParallelTraverser:
    """Parallel graph traversal using concurrent.futures."""

    def __init__(self, graph_manager: GraphManager) -> None:
        self._graph = graph_manager

    def traverse_parallel(
        self,
        start_vertices: list[str],
        max_workers: int = 4,
        **traverse_kwargs: Any,
    ) -> list[TraversalResult]:
        """Run traversals from multiple start vertices in parallel.

        Returns one TraversalResult per start vertex, in the same order.
        Individual failures yield an empty TraversalResult.
        """
        results: dict[int, TraversalResult] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(
                    self._graph.traverse,
                    start_vertex=sv,
                    **traverse_kwargs,
                ): idx
                for idx, sv in enumerate(start_vertices)
            }

            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    logger.exception(
                        "Traversal failed for vertex at index %d", idx
                    )
                    results[idx] = TraversalResult()

        return [results[i] for i in range(len(start_vertices))]
