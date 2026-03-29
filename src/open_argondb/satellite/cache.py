"""Satellite cache — in-memory replication of small reference collections."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

from open_argondb.models import SatelliteConfig, SatelliteStats


class SatelliteCache:
    """In-memory cache simulating ArangoDB Enterprise SatelliteCollections.

    Satellite collections in Enterprise are replicated to every DB server.
    This caches small reference collections in-memory for fast local access.
    """

    def __init__(self, db: Any, config: SatelliteConfig) -> None:
        self._db = db
        self._config = config
        self._cache: dict[str, dict] = {}
        self._hits = 0
        self._misses = 0
        self._last_sync = ""
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self.sync()  # Initial load

    def get(self, key: str) -> dict | None:
        """Get document by _key. Cache hit or miss-through to DB."""
        with self._lock:
            if key in self._cache:
                self._hits += 1
                return self._cache[key]
        self._misses += 1
        # Miss-through: fetch from DB
        doc = self._db.collection(self._config.collection).get(key)
        if doc:
            with self._lock:
                self._cache[key] = doc
        return doc

    def get_all(self) -> list[dict]:
        """Return all cached documents."""
        with self._lock:
            return list(self._cache.values())

    def sync(self) -> int:
        """Refresh cache from ArangoDB. Returns count synced."""
        col = self._db.collection(self._config.collection)
        docs = col.all()  # MockCollection.all() returns list
        with self._lock:
            self._cache = {d["_key"]: d for d in docs[: self._config.max_size]}
            self._last_sync = datetime.now(timezone.utc).isoformat()
        return len(self._cache)

    def invalidate(self, key: str | None = None) -> None:
        """Invalidate one key or entire cache."""
        with self._lock:
            if key:
                self._cache.pop(key, None)
            else:
                self._cache.clear()

    def stats(self) -> SatelliteStats:
        """Return cache statistics."""
        with self._lock:
            return SatelliteStats(
                collection=self._config.collection,
                cached_count=len(self._cache),
                hit_count=self._hits,
                miss_count=self._misses,
                last_sync=self._last_sync,
            )

    def start_auto_sync(self) -> None:
        """Start periodic sync timer."""

        def _sync_and_reschedule() -> None:
            self.sync()
            if not self._stop_requested:
                self._timer = threading.Timer(
                    self._config.ttl_seconds, _sync_and_reschedule
                )
                self._timer.daemon = True
                self._timer.start()

        self._stop_requested = False
        self._timer = threading.Timer(self._config.ttl_seconds, _sync_and_reschedule)
        self._timer.daemon = True
        self._timer.start()

    def stop(self) -> None:
        """Stop auto-sync timer."""
        self._stop_requested = True
        if self._timer:
            self._timer.cancel()
            self._timer = None
