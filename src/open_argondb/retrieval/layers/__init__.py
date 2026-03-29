"""Retrieval layer protocol and implementations."""

from __future__ import annotations

from typing import Protocol

from open_argondb.models import RetrievalRequest, RetrievalResult


class RetrievalLayer(Protocol):
    """Protocol for a single retrieval layer in the cascade."""

    def search(self, request: RetrievalRequest) -> list[RetrievalResult]: ...
