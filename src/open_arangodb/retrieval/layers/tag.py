"""Tag-match retrieval layer — Jaccard similarity on tags and entity."""

from __future__ import annotations

import json
from typing import Any

from open_arangodb.models import RetrievalRequest, RetrievalResult
from open_arangodb.retrieval.layers.exact import _doc_to_memory


class TagMatchLayer:
    """Find memories by tag overlap or entity match, scored by Jaccard similarity."""

    def __init__(self, db: Any) -> None:
        self._db = db

    def search(self, request: RetrievalRequest) -> list[RetrievalResult]:
        """Return memories whose tags overlap with request.tags or entity matches."""
        if request.tags is None and request.entity is None:
            return []

        # Build query to find candidate documents
        filters = ["doc._deleted != true", "doc.status == 'active'"]
        bind_vars: dict[str, Any] = {}

        if request.entity is not None:
            filters.append("LOWER(doc.entity) == LOWER(@ent)")
            bind_vars["ent"] = request.entity

        where = " AND ".join(filters)
        aql = f"FOR doc IN memories FILTER {where} RETURN doc"
        cursor = self._db.aql.execute(aql, bind_vars=bind_vars)
        docs = list(cursor)

        request_tags = set(request.tags) if request.tags else set()
        results: list[RetrievalResult] = []

        for doc in docs:
            doc_tags_raw = doc.get("tags", "[]")
            if isinstance(doc_tags_raw, str):
                try:
                    doc_tags = set(json.loads(doc_tags_raw))
                except (json.JSONDecodeError, TypeError):
                    doc_tags = set()
            else:
                doc_tags = set(doc_tags_raw)

            # Calculate Jaccard similarity between request tags and doc tags
            if request_tags and doc_tags:
                intersection = request_tags & doc_tags
                union = request_tags | doc_tags
                score = len(intersection) / len(union) if union else 0.0
            elif request.entity is not None:
                # Entity-only match: give a base score
                score = 0.5
            else:
                score = 0.0

            if score > 0.0:
                memory = _doc_to_memory(doc)
                results.append(
                    RetrievalResult(
                        memory=memory,
                        score=score,
                        match_source="tag",
                        tier=2,
                    )
                )

        # Sort by score descending
        return sorted(results, key=lambda r: r.score, reverse=True)
