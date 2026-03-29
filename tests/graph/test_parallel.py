"""Tests for ParallelTraverser — concurrent graph traversal."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from open_argondb.graph.manager import GraphManager
from open_argondb.graph.parallel import ParallelTraverser
from open_argondb.models import EdgeDefinition, GraphConfig, TraversalResult


@pytest.fixture
def graph_manager(mock_db):
    """GraphManager wired to mock DB."""
    return GraphManager(mock_db)


@pytest.fixture
def parallel_traverser(graph_manager):
    """ParallelTraverser wired to graph manager."""
    return ParallelTraverser(graph_manager)


@pytest.fixture
def populated_graph(mock_db, graph_manager):
    """Set up a graph with 3 separate start vertices, each with one neighbor."""
    config = GraphConfig(
        name="test_graph",
        edge_definitions=[
            EdgeDefinition(
                collection="edges",
                from_vertex_collections=["nodes"],
                to_vertex_collections=["nodes"],
            ),
        ],
    )
    graph_manager.create_graph(config)

    nodes = mock_db.collection("nodes")
    edges = mock_db.collection("edges")

    # Three separate subgraphs: a->b, c->d, e->f
    nodes.insert({"_key": "a", "name": "A"})
    nodes.insert({"_key": "b", "name": "B"})
    nodes.insert({"_key": "c", "name": "C"})
    nodes.insert({"_key": "d", "name": "D"})
    nodes.insert({"_key": "e", "name": "E"})
    nodes.insert({"_key": "f", "name": "F"})

    edges.insert({"_key": "e1", "_from": "nodes/a", "_to": "nodes/b"})
    edges.insert({"_key": "e2", "_from": "nodes/c", "_to": "nodes/d"})
    edges.insert({"_key": "e3", "_from": "nodes/e", "_to": "nodes/f"})

    return config


class TestParallelTraversal:
    def test_parallel_traversal(self, parallel_traverser, populated_graph):
        results = parallel_traverser.traverse_parallel(
            start_vertices=["nodes/a", "nodes/c", "nodes/e"],
            edge_collection="edges",
            min_depth=1,
            max_depth=2,
        )

        assert len(results) == 3
        # Each traversal should find exactly one neighbor
        for result in results:
            assert len(result.vertices) == 1

        # Check specific results by vertex name
        names = [r.vertices[0]["name"] for r in results]
        assert names == ["B", "D", "F"]

    def test_parallel_max_workers(self, parallel_traverser, populated_graph):
        """Respects max_workers parameter (runs with 1 worker)."""
        results = parallel_traverser.traverse_parallel(
            start_vertices=["nodes/a", "nodes/c"],
            max_workers=1,
            edge_collection="edges",
            min_depth=1,
            max_depth=2,
        )

        assert len(results) == 2
        assert len(results[0].vertices) == 1
        assert len(results[1].vertices) == 1

    def test_parallel_error_handling(self, graph_manager, populated_graph):
        """One failure does not kill others — returns empty TraversalResult."""
        original_traverse = graph_manager.traverse

        call_count = 0

        def failing_traverse(start_vertex, **kwargs):
            nonlocal call_count
            call_count += 1
            if start_vertex == "nodes/c":
                raise RuntimeError("Simulated failure")
            return original_traverse(start_vertex=start_vertex, **kwargs)

        graph_manager.traverse = failing_traverse
        traverser = ParallelTraverser(graph_manager)

        results = traverser.traverse_parallel(
            start_vertices=["nodes/a", "nodes/c", "nodes/e"],
            edge_collection="edges",
            min_depth=1,
            max_depth=2,
        )

        assert len(results) == 3
        # nodes/a and nodes/e should succeed
        successful = [r for r in results if len(r.vertices) > 0]
        assert len(successful) == 2

        # nodes/c (index 1) should be an empty result
        assert results[1].vertices == []
        assert results[1].edges == []

    def test_parallel_empty_input(self, parallel_traverser):
        """Empty start_vertices returns empty list."""
        results = parallel_traverser.traverse_parallel(
            start_vertices=[],
            edge_collection="edges",
        )
        assert results == []
