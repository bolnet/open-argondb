"""Tests for SatelliteCache."""

from __future__ import annotations

import pytest

from open_arangodb.models import SatelliteConfig, SatelliteStats
from open_arangodb.satellite.cache import SatelliteCache


@pytest.fixture
def satellite_config() -> SatelliteConfig:
    return SatelliteConfig(collection="countries", ttl_seconds=60, max_size=5)


@pytest.fixture
def populated_db(mock_db):
    """MockDatabase with a 'countries' collection containing sample docs."""
    col = mock_db.create_collection("countries")
    col.insert({"_key": "us", "name": "United States", "code": "US"})
    col.insert({"_key": "uk", "name": "United Kingdom", "code": "UK"})
    col.insert({"_key": "de", "name": "Germany", "code": "DE"})
    return mock_db


@pytest.fixture
def cache(populated_db, satellite_config) -> SatelliteCache:
    return SatelliteCache(populated_db, satellite_config)


class TestSatelliteCache:
    def test_initial_sync(self, cache: SatelliteCache) -> None:
        """Cache should be populated on init via sync()."""
        all_docs = cache.get_all()
        assert len(all_docs) == 3
        keys = {d["_key"] for d in all_docs}
        assert keys == {"us", "uk", "de"}

    def test_get_hit(self, cache: SatelliteCache) -> None:
        """Getting a cached key should return the doc and increment hits."""
        doc = cache.get("us")
        assert doc is not None
        assert doc["name"] == "United States"

        stats = cache.stats()
        assert stats.hit_count == 1
        assert stats.miss_count == 0

    def test_get_miss_through(self, populated_db, satellite_config) -> None:
        """Miss should fetch from DB and cache the result."""
        # Insert a doc after cache init so it's not in cache
        populated_db.collection("countries").insert(
            {"_key": "fr", "name": "France", "code": "FR"}
        )
        cache = SatelliteCache(populated_db, satellite_config)
        # fr is now in cache from sync — invalidate it to force miss
        cache.invalidate("fr")

        doc = cache.get("fr")
        assert doc is not None
        assert doc["name"] == "France"

        stats = cache.stats()
        assert stats.miss_count == 1

    def test_get_all(self, cache: SatelliteCache) -> None:
        """get_all should return all cached documents."""
        docs = cache.get_all()
        assert len(docs) == 3
        names = {d["name"] for d in docs}
        assert "United States" in names
        assert "United Kingdom" in names
        assert "Germany" in names

    def test_sync_refreshes(self, populated_db, cache: SatelliteCache) -> None:
        """sync() should pick up new docs added to the DB."""
        populated_db.collection("countries").insert(
            {"_key": "jp", "name": "Japan", "code": "JP"}
        )
        count = cache.sync()
        assert count == 4
        assert cache.get("jp") is not None

    def test_invalidate_single(self, cache: SatelliteCache) -> None:
        """invalidate(key) should remove one key from cache."""
        cache.invalidate("us")
        docs = cache.get_all()
        keys = {d["_key"] for d in docs}
        assert "us" not in keys
        assert len(docs) == 2

    def test_invalidate_all(self, cache: SatelliteCache) -> None:
        """invalidate() with no key should clear entire cache."""
        cache.invalidate()
        assert cache.get_all() == []
        assert cache.stats().cached_count == 0

    def test_stats(self, cache: SatelliteCache) -> None:
        """stats() should return correct counts."""
        cache.get("us")  # hit
        cache.get("us")  # hit
        cache.invalidate("nonexistent_key")  # no effect
        cache.get("nonexistent")  # miss (not in DB either)

        stats = cache.stats()
        assert isinstance(stats, SatelliteStats)
        assert stats.collection == "countries"
        assert stats.cached_count == 3
        assert stats.hit_count == 2
        assert stats.miss_count == 1
        assert stats.last_sync != ""

    def test_max_size(self, mock_db) -> None:
        """Cache should respect max_size limit."""
        col = mock_db.create_collection("big")
        for i in range(10):
            col.insert({"_key": f"doc_{i}", "value": i})

        config = SatelliteConfig(collection="big", max_size=3)
        cache = SatelliteCache(mock_db, config)
        assert len(cache.get_all()) == 3
