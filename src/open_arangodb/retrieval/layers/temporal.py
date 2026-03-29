"""Temporal retrieval layer — date-based memory search."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from open_arangodb.models import (
    RetrievalRequest,
    RetrievalResult,
    TemporalQuery,
)

_ISO_DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")
_RELATIVE_PATTERNS = {
    "today": 0,
    "yesterday": -1,
}


class TemporalLayer:
    """Find memories by temporal references in the query text."""

    def __init__(self, db: Any, temporal_engine: Any) -> None:
        self._db = db
        self._temporal = temporal_engine

    def search(self, request: RetrievalRequest) -> list[RetrievalResult]:
        """Extract date references from query and search temporal range."""
        dates = self._extract_dates(request.query)
        if not dates:
            return []

        # Use the range of extracted dates as start/end
        start = min(dates)
        end = max(dates)

        tq = TemporalQuery(
            entity=request.entity,
            start=start,
            end=end,
        )
        memories = self._temporal.query_temporal_range(tq)

        if not memories:
            return []

        # Score by recency: newer memories get higher scores
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        results: list[RetrievalResult] = []

        for memory in memories:
            score = self._recency_score(memory.valid_from, now_str)
            results.append(
                RetrievalResult(
                    memory=memory,
                    score=score,
                    match_source="temporal",
                    tier=3,
                )
            )

        return sorted(results, key=lambda r: r.score, reverse=True)

    def _extract_dates(self, query: str) -> list[str]:
        """Extract ISO dates and resolve relative date references."""
        dates: list[str] = []

        # Find ISO dates (YYYY-MM-DD)
        for match in _ISO_DATE_PATTERN.finditer(query):
            dates.append(match.group())

        # Resolve relative references
        today = datetime.now(timezone.utc).date()
        query_lower = query.lower()
        for word, offset in _RELATIVE_PATTERNS.items():
            if word in query_lower:
                target = today + timedelta(days=offset)
                dates.append(target.isoformat())

        return dates

    def _recency_score(self, valid_from: str, now: str) -> float:
        """Score based on how recent the memory is. Newer = higher score."""
        if not valid_from:
            return 0.1

        try:
            mem_date = datetime.fromisoformat(valid_from.replace("Z", "+00:00"))
            now_date = datetime.fromisoformat(now)
        except (ValueError, TypeError):
            return 0.1

        # Days difference, capped at 365
        delta = abs((now_date.date() - mem_date.date()).days)
        capped = min(delta, 365)

        # Linear decay: 1.0 for today, 0.1 for 365+ days ago
        return max(0.1, 1.0 - (capped / 365.0) * 0.9)
