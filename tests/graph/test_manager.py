"""Tests for GraphManager — SmartGraph-like partitioning and traversal."""

from __future__ import annotations

import pytest

from open_arangodb.graph.manager import GraphManager
from open_arangodb.models import EdgeDefinition, GraphConfig, PartitionKey


@pytest.fixture
def graph_manager(mock_db):
    """GraphManager wired to mock DB."""
    return GraphManager(mock_db)


@pytest.fixture
def social_graph_config() -> GraphConfig:
    """A simple social graph config for testing."""
    return GraphConfig(
        name="social",
        edge_definitions=[
            EdgeDefinition(
                collection="follows",
                from_vertex_collections=["users"],
                to_vertex_collections=["users"],
            ),
        ],
    )


@pytest.fixture
def smart_graph_config() -> GraphConfig:
    """A SmartGraph config with partition attribute."""
    return GraphConfig(
        name="tenant_graph",
        edge_definitions=[
            EdgeDefinition(
                collection="connections",
                from_vertex_collections=["people"],
                to_vertex_collections=["people"],
            ),
        ],
        smart_attribute="tenant_id",
    )


def _insert_users_and_edges(mock_db):
    """Helper to set up a small graph: alice -> bob -> carol."""
    users = mock_db.collection("users")
    follows = mock_db.collection("follows")

    users.insert({"_key": "alice", "name": "Alice"})
    users.insert({"_key": "bob", "name": "Bob"})
    users.insert({"_key": "carol", "name": "Carol"})

    follows.insert({"_key": "e1", "_from": "users/alice", "_to": "users/bob"})
    follows.insert({"_key": "e2", "_from": "users/bob", "_to": "users/carol"})


class TestCreateGraph:
    def test_create_graph(self, mock_db, graph_manager, social_graph_config):
        graph_manager.create_graph(social_graph_config)

        assert mock_db.has_graph("social")
        assert mock_db.has_collection("follows")
        assert mock_db.has_collection("users")

    def test_create_graph_with_smart_attribute(
        self, mock_db, graph_manager, smart_graph_config
    ):
        graph_manager.create_graph(smart_graph_config)

        assert mock_db.has_graph("tenant_graph")
        people_col = mock_db.collection("people")
        # Should have a persistent index on tenant_id
        assert any(
            idx.get("fields") == ["tenant_id"] for idx in people_col._indexes
        )


class TestInsertVertex:
    def test_insert_vertex(self, mock_db, graph_manager, social_graph_config):
        graph_manager.create_graph(social_graph_config)
        result = graph_manager.insert_vertex("users", {"_key": "alice", "name": "Alice"})

        assert result["_key"] == "alice"
        doc = mock_db.collection("users").get("alice")
        assert doc["name"] == "Alice"

    def test_insert_vertex_auto_key(self, mock_db, graph_manager, social_graph_config):
        graph_manager.create_graph(social_graph_config)
        result = graph_manager.insert_vertex("users", {"name": "Bob"})

        assert result["_key"]  # auto-generated key exists
        assert len(result["_key"]) == 12

    def test_insert_vertex_with_partition(
        self, mock_db, graph_manager, smart_graph_config
    ):
        graph_manager.create_graph(smart_graph_config)
        pk = PartitionKey(attribute="tenant_id", value="acme")

        result = graph_manager.insert_vertex(
            "people", {"_key": "alice", "name": "Alice"}, partition_key=pk
        )

        assert result["_key"] == "acme:alice"
        doc = mock_db.collection("people").get("acme:alice")
        assert doc["tenant_id"] == "acme"
        assert doc["name"] == "Alice"


class TestInsertEdge:
    def test_insert_edge(self, mock_db, graph_manager, social_graph_config):
        graph_manager.create_graph(social_graph_config)
        graph_manager.insert_vertex("users", {"_key": "alice", "name": "Alice"})
        graph_manager.insert_vertex("users", {"_key": "bob", "name": "Bob"})

        result = graph_manager.insert_edge(
            "follows", "users/alice", "users/bob", {"weight": 1.0}
        )

        assert result["_key"]
        edge = mock_db.collection("follows").get(result["_key"])
        assert edge["_from"] == "users/alice"
        assert edge["_to"] == "users/bob"
        assert edge["weight"] == 1.0


class TestGetNeighbors:
    def test_get_neighbors_outbound(self, mock_db, graph_manager, social_graph_config):
        graph_manager.create_graph(social_graph_config)
        _insert_users_and_edges(mock_db)

        neighbors = graph_manager.get_neighbors(
            "users/alice", "follows", direction="outbound"
        )

        assert len(neighbors) == 1
        assert neighbors[0]["name"] == "Bob"

    def test_get_neighbors_any(self, mock_db, graph_manager, social_graph_config):
        graph_manager.create_graph(social_graph_config)
        _insert_users_and_edges(mock_db)

        neighbors = graph_manager.get_neighbors(
            "users/bob", "follows", direction="any"
        )

        assert len(neighbors) == 2
        names = {n["name"] for n in neighbors}
        assert names == {"Alice", "Carol"}


class TestTraverse:
    def test_traverse(self, mock_db, graph_manager, social_graph_config):
        graph_manager.create_graph(social_graph_config)
        _insert_users_and_edges(mock_db)

        result = graph_manager.traverse(
            start_vertex="users/alice",
            graph_name="social",
            min_depth=1,
            max_depth=3,
        )

        assert len(result.vertices) == 2  # bob and carol
        names = {v["name"] for v in result.vertices}
        assert names == {"Bob", "Carol"}
        assert len(result.edges) >= 2
        assert len(result.paths) >= 2

    def test_traverse_with_edge_collection(
        self, mock_db, graph_manager, social_graph_config
    ):
        graph_manager.create_graph(social_graph_config)
        _insert_users_and_edges(mock_db)

        result = graph_manager.traverse(
            start_vertex="users/alice",
            edge_collection="follows",
            min_depth=1,
            max_depth=2,
        )

        assert len(result.vertices) >= 1
        assert any(v["name"] == "Bob" for v in result.vertices)

    def test_traverse_with_partition(self, mock_db, graph_manager, smart_graph_config):
        graph_manager.create_graph(smart_graph_config)

        people = mock_db.collection("people")
        conns = mock_db.collection("connections")

        people.insert({"_key": "acme:alice", "name": "Alice", "tenant_id": "acme"})
        people.insert({"_key": "acme:bob", "name": "Bob", "tenant_id": "acme"})
        people.insert({"_key": "other:carol", "name": "Carol", "tenant_id": "other"})

        conns.insert({
            "_key": "e1",
            "_from": "people/acme:alice",
            "_to": "people/acme:bob",
        })
        conns.insert({
            "_key": "e2",
            "_from": "people/acme:alice",
            "_to": "people/other:carol",
        })

        pk = PartitionKey(attribute="tenant_id", value="acme")
        result = graph_manager.traverse(
            start_vertex="people/acme:alice",
            graph_name="tenant_graph",
            partition_key=pk,
            min_depth=1,
            max_depth=2,
        )

        # Should only include acme tenant vertices, not carol
        assert len(result.vertices) == 1
        assert result.vertices[0]["name"] == "Bob"

    def test_traverse_no_edge_collection(self, graph_manager):
        """Traverse with no graph_name or edge_collection returns empty."""
        result = graph_manager.traverse(start_vertex="users/alice")
        assert result.vertices == []
        assert result.edges == []


class TestDropGraph:
    def test_drop_graph(self, mock_db, graph_manager, social_graph_config):
        graph_manager.create_graph(social_graph_config)
        assert mock_db.has_graph("social")

        graph_manager.drop_graph("social")
        assert not mock_db.has_graph("social")


class TestReset:
    def test_reset(self, mock_db, graph_manager, social_graph_config):
        graph_manager.create_graph(social_graph_config)
        assert mock_db.has_graph("social")

        graph_manager.reset()
        assert not mock_db.has_graph("social")
        assert len(graph_manager._graph_configs) == 0
