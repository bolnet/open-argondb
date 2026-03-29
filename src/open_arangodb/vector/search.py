"""Vector search — numpy cosine with auto-upgrade to native ArangoDB vector index."""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger("open_arangodb")


class VectorSearch:
    """Dual-mode vector search: numpy fallback + native ArangoDB vector index."""

    COLLECTION = "memories"

    def __init__(self, db: Any, model_name: str = "BAAI/bge-m3") -> None:
        self._db = db
        self._model_name = model_name
        self._embedding_fn: Any = None
        self._ensure_embedding_fn()

        # In-memory cache for numpy cosine search
        self._cache_ids: list[str] = []
        self._cache_matrix: np.ndarray | None = None

        # Check if native vector index is available
        self._native_available = self._check_native_vector()
        if self._native_available:
            logger.info("Native ArangoDB vector index available — using it")
        else:
            logger.info("Using numpy cosine search (native vector index not available)")

    def _ensure_embedding_fn(self) -> None:
        if self._embedding_fn is not None:
            return

        model_name = os.environ.get("ARANGODB_EMBEDDING_MODEL", self._model_name)

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)

        class _Embedder:
            def __init__(self, m: Any) -> None:
                self._m = m
                self.dimension: int = m.get_sentence_embedding_dimension()

            def embed(self, text: str) -> list[float]:
                return self._m.encode(text, normalize_embeddings=True).tolist()

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return self._m.encode(texts, normalize_embeddings=True).tolist()

        self._embedding_fn = _Embedder(model)

    def _check_native_vector(self) -> bool:
        """Check if ArangoDB has experimental vector index enabled."""
        try:
            # Try creating a dummy vector index — if it works, native is available
            if not self._db.has_collection("_arangodb_vector_check"):
                col = self._db.create_collection("_arangodb_vector_check")
                col.add_index({
                    "type": "vector",
                    "fields": ["_test_vec"],
                    "params": {"metric": "cosine", "dimension": 3, "nLists": 1},
                })
                self._db.delete_collection("_arangodb_vector_check")
                return True
        except Exception:
            # Clean up on failure
            if self._db.has_collection("_arangodb_vector_check"):
                self._db.delete_collection("_arangodb_vector_check")
            return False
        return False

    def _invalidate_cache(self) -> None:
        self._cache_matrix = None
        self._cache_ids = []

    def _build_cache(self) -> None:
        """Load all embeddings into numpy matrix for cosine search."""
        cursor = self._db.aql.execute(
            "FOR doc IN memories "
            "FILTER HAS(doc, 'embedding') AND doc.embedding != null "
            "AND doc._deleted != true "
            "RETURN {memory_id: doc.memory_id, embedding: doc.embedding}"
        )
        rows = list(cursor)
        if not rows:
            self._cache_ids = []
            self._cache_matrix = None
            return

        self._cache_ids = [r["memory_id"] for r in rows]
        self._cache_matrix = np.array(
            [r["embedding"] for r in rows], dtype=np.float32
        )
        norms = np.linalg.norm(self._cache_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._cache_matrix = self._cache_matrix / norms

    def add(self, memory_id: str, content: str) -> None:
        """Generate embedding and store it."""
        embedding = self._embedding_fn.embed(content)
        key = memory_id.replace("/", "_")
        try:
            self._db.collection(self.COLLECTION).update({
                "_key": key,
                "embedding": embedding,
            })
        except Exception:
            self._db.collection(self.COLLECTION).insert({
                "_key": key,
                "memory_id": memory_id,
                "content": content,
                "embedding": embedding,
                "tags": "[]",
                "category": "general",
                "status": "active",
                "created_at": "",
                "valid_from": "",
                "_deleted": False,
            }, overwrite=True)
        self._invalidate_cache()

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Search by vector similarity."""
        if self._native_available:
            return self._search_native(query, limit)
        return self._search_numpy(query, limit)

    def _search_numpy(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Numpy-based cosine search (fallback)."""
        if self._cache_matrix is None:
            self._build_cache()

        if self._cache_matrix is None or len(self._cache_ids) == 0:
            return []

        q_emb = np.array(self._embedding_fn.embed(query), dtype=np.float32)
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 0:
            q_emb = q_emb / q_norm

        similarities = self._cache_matrix @ q_emb
        top_k = min(limit, len(similarities))

        if top_k >= len(similarities):
            top_indices = np.argsort(-similarities)[:top_k]
        else:
            top_indices = np.argpartition(-similarities, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-similarities[top_indices])]

        results = []
        for idx in top_indices:
            mid = self._cache_ids[idx]
            distance = float(1.0 - similarities[idx])
            cursor = self._db.aql.execute(
                "FOR doc IN memories FILTER doc.memory_id == @mid LIMIT 1 "
                "RETURN doc.content",
                bind_vars={"mid": mid},
            )
            content_list = list(cursor)
            results.append({
                "memory_id": mid,
                "content": content_list[0] if content_list else "",
                "distance": distance,
            })

        return results

    def _search_native(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Native ArangoDB vector index search (when available)."""
        q_emb = self._embedding_fn.embed(query)
        # ArangoDB vector search via AQL
        cursor = self._db.aql.execute(
            """
            FOR doc IN memories
            FILTER doc._deleted != true AND doc.embedding != null
            LET dist = COSINE_SIMILARITY(doc.embedding, @query_vec)
            SORT dist DESC
            LIMIT @lim
            RETURN {memory_id: doc.memory_id, content: doc.content, distance: 1 - dist}
            """,
            bind_vars={"query_vec": q_emb, "lim": limit},
        )
        return list(cursor)

    def batch_embed(self) -> int:
        """Embed all memories missing embeddings."""
        cursor = self._db.aql.execute(
            "FOR doc IN memories "
            "FILTER doc.embedding == null OR NOT HAS(doc, 'embedding') "
            "RETURN {_key: doc._key, content: doc.content}"
        )
        rows = list(cursor)
        if not rows:
            return 0

        contents = [r["content"] for r in rows]
        embeddings = self._embedding_fn.embed_batch(contents)
        col = self._db.collection(self.COLLECTION)

        for row, emb in zip(rows, embeddings):
            col.update({"_key": row["_key"], "embedding": emb})

        self._invalidate_cache()
        return len(rows)

    def reset(self) -> None:
        self._invalidate_cache()
