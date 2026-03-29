"""SmartGraph-like partitioning and native ArangoDB graph operations."""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from open_arangodb.models import GraphConfig, PartitionKey, TraversalResult

logger = logging.getLogger("open_arangodb.graph")


class GraphManager:
    """SmartGraph-like partitioning and native ArangoDB graph operations."""

    def __init__(self, db: Any) -> None:
        self._db = db
        self._graph_configs: dict[str, GraphConfig] = {}

    def create_graph(self, config: GraphConfig) -> None:
        """Create a named graph. If smart_attribute is set, add partition index."""
        edge_defs = []
        for ed in config.edge_definitions:
            # Ensure edge collection exists
            if not self._db.has_collection(ed.collection):
                self._db.create_collection(ed.collection)

            # Ensure vertex collections exist
            for vc in ed.from_vertex_collections:
                if not self._db.has_collection(vc):
                    self._db.create_collection(vc)
            for vc in ed.to_vertex_collections:
                if not self._db.has_collection(vc):
                    self._db.create_collection(vc)

            edge_defs.append({
                "edge_collection": ed.collection,
                "from_vertex_collections": ed.from_vertex_collections,
                "to_vertex_collections": ed.to_vertex_collections,
            })

        self._db.create_graph(
            config.name,
            edge_definitions=edge_defs,
        )

        # If smart_attribute is set, add persistent index on vertex collections
        if config.smart_attribute:
            vertex_collections: set[str] = set()
            for ed in config.edge_definitions:
                vertex_collections.update(ed.from_vertex_collections)
                vertex_collections.update(ed.to_vertex_collections)
            for vc_name in vertex_collections:
                col = self._db.collection(vc_name)
                col.add_index({
                    "type": "persistent",
                    "fields": [config.smart_attribute],
                })

        self._graph_configs[config.name] = config

    def drop_graph(self, name: str) -> None:
        """Drop a named graph."""
        self._db.delete_graph(name)
        self._graph_configs.pop(name, None)

    def insert_vertex(
        self,
        collection: str,
        data: dict[str, Any],
        partition_key: PartitionKey | None = None,
    ) -> dict[str, Any]:
        """Insert a vertex. If partition_key, prefix _key with partition value."""
        doc = {**data}
        if partition_key:
            raw_key = doc.get("_key", uuid4().hex[:12])
            doc["_key"] = f"{partition_key.value}:{raw_key}"
            doc[partition_key.attribute] = partition_key.value
        elif "_key" not in doc:
            doc["_key"] = uuid4().hex[:12]

        col = self._db.collection(collection)
        return col.insert(doc)

    def insert_edge(
        self,
        collection: str,
        from_id: str,
        to_id: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Insert an edge between two vertices."""
        edge = {"_from": from_id, "_to": to_id, **(data or {})}
        col = self._db.collection(collection)
        return col.insert(edge)

    def traverse(
        self,
        start_vertex: str,
        graph_name: str | None = None,
        edge_collection: str | None = None,
        direction: str = "outbound",
        min_depth: int = 1,
        max_depth: int = 3,
        partition_key: PartitionKey | None = None,
    ) -> TraversalResult:
        """Graph traversal. Uses edge-walking approach compatible with MockAQL."""
        resolved_edge_collection = edge_collection
        if resolved_edge_collection is None and graph_name:
            config = self._graph_configs.get(graph_name)
            if config and config.edge_definitions:
                resolved_edge_collection = config.edge_definitions[0].collection

        if resolved_edge_collection is None:
            return TraversalResult()

        return self._traverse_via_edges(
            start_vertex=start_vertex,
            edge_collection=resolved_edge_collection,
            direction=direction,
            min_depth=min_depth,
            max_depth=max_depth,
            partition_key=partition_key,
        )

    def _traverse_via_edges(
        self,
        start_vertex: str,
        edge_collection: str,
        direction: str,
        min_depth: int,
        max_depth: int,
        partition_key: PartitionKey | None = None,
    ) -> TraversalResult:
        """Walk edges iteratively from the start vertex."""
        edge_col = self._db.collection(edge_collection)
        all_edges = edge_col.all() if hasattr(edge_col, "all") else []

        visited_vertices: dict[str, dict[str, Any]] = {}
        collected_edges: list[dict[str, Any]] = []
        paths: list[list[str]] = []

        # BFS traversal
        # Each item: (vertex_id, current_depth, path_so_far)
        queue: list[tuple[str, int, list[str]]] = [(start_vertex, 0, [start_vertex])]
        seen: set[str] = {start_vertex}

        while queue:
            current_id, depth, path = queue.pop(0)

            if depth >= max_depth:
                continue

            # Find edges from/to current vertex based on direction
            for edge in all_edges:
                next_id: str | None = None
                if direction in ("outbound", "any") and edge.get("_from") == current_id:
                    next_id = edge["_to"]
                if direction in ("inbound", "any") and edge.get("_to") == current_id:
                    next_id = edge["_from"]

                if next_id is None:
                    continue

                new_depth = depth + 1
                new_path = [*path, next_id]

                # Apply partition filter if specified
                if partition_key:
                    # Look up the target vertex to check partition attribute
                    target_doc = self._resolve_vertex(next_id)
                    pk_val = target_doc.get(partition_key.attribute)
                    if target_doc and pk_val != partition_key.value:
                        continue

                if new_depth >= min_depth:
                    collected_edges.append(edge)
                    target = self._resolve_vertex(next_id)
                    if target:
                        visited_vertices[next_id] = target
                    paths.append(new_path)

                if next_id not in seen:
                    seen.add(next_id)
                    queue.append((next_id, new_depth, new_path))

        return TraversalResult(
            vertices=list(visited_vertices.values()),
            edges=collected_edges,
            paths=paths,
        )

    def _resolve_vertex(self, vertex_id: str) -> dict[str, Any] | None:
        """Resolve a vertex ID (collection/key) to its document."""
        if "/" not in vertex_id:
            return None
        col_name, key = vertex_id.split("/", 1)
        col = self._db.collection(col_name)
        return col.get(key)

    def get_neighbors(
        self,
        start_vertex: str,
        edge_collection: str,
        direction: str = "any",
    ) -> list[dict[str, Any]]:
        """Get immediate neighbors of a vertex."""
        edge_col = self._db.collection(edge_collection)
        all_edges = edge_col.all() if hasattr(edge_col, "all") else []

        neighbor_ids: list[str] = []
        for edge in all_edges:
            if direction in ("outbound", "any") and edge.get("_from") == start_vertex:
                neighbor_ids.append(edge["_to"])
            if direction in ("inbound", "any") and edge.get("_to") == start_vertex:
                neighbor_ids.append(edge["_from"])

        neighbors: list[dict[str, Any]] = []
        for nid in neighbor_ids:
            doc = self._resolve_vertex(nid)
            if doc:
                neighbors.append(doc)
        return neighbors

    def reset(self) -> None:
        """Clean up all tracked graphs."""
        for name in list(self._graph_configs):
            self._db.delete_graph(name)
        self._graph_configs.clear()
