"""Unit tests for InProcessBus."""

from __future__ import annotations

from typing import Any

import pytest

from open_arangodb.events.bus import InProcessBus


# ── Publish / Subscribe ──────────────────────────────────────────────


class TestPubSub:
    def test_subscribe_and_publish(self, event_bus: InProcessBus) -> None:
        received: list[tuple[str, dict]] = []

        def handler(topic: str, data: dict[str, Any]) -> None:
            received.append((topic, data))

        event_bus.subscribe("test.topic", handler)
        event_bus.publish("test.topic", {"key": "value"})

        assert len(received) == 1
        assert received[0] == ("test.topic", {"key": "value"})

    def test_multiple_subscribers(self, event_bus: InProcessBus) -> None:
        calls: list[str] = []

        event_bus.subscribe("t", lambda topic, data: calls.append("a"))
        event_bus.subscribe("t", lambda topic, data: calls.append("b"))
        event_bus.publish("t", {})

        assert calls == ["a", "b"]

    def test_publish_no_subscribers(self, event_bus: InProcessBus) -> None:
        # Should not raise
        event_bus.publish("no.listeners", {"x": 1})

    def test_publish_different_topics(self, event_bus: InProcessBus) -> None:
        received: list[str] = []

        event_bus.subscribe("topic-a", lambda t, d: received.append("a"))
        event_bus.subscribe("topic-b", lambda t, d: received.append("b"))

        event_bus.publish("topic-a", {})
        assert received == ["a"]

    def test_publish_sends_correct_data(self, event_bus: InProcessBus) -> None:
        captured: list[dict] = []

        event_bus.subscribe("data.test", lambda t, d: captured.append(d))
        event_bus.publish("data.test", {"memory_id": "m1", "op": "insert"})

        assert captured[0] == {"memory_id": "m1", "op": "insert"}


# ── Exception Handling ───────────────────────────────────────────────


class TestExceptionHandling:
    def test_callback_exception_does_not_propagate(self, event_bus: InProcessBus) -> None:
        def bad_handler(topic: str, data: dict) -> None:
            raise ValueError("boom")

        event_bus.subscribe("err.topic", bad_handler)
        # Should not raise — exception is caught and logged
        event_bus.publish("err.topic", {"x": 1})

    def test_exception_in_one_does_not_block_others(self, event_bus: InProcessBus) -> None:
        results: list[str] = []

        def bad(topic: str, data: dict) -> None:
            raise RuntimeError("fail")

        def good(topic: str, data: dict) -> None:
            results.append("ok")

        event_bus.subscribe("multi", bad)
        event_bus.subscribe("multi", good)
        event_bus.publish("multi", {})

        assert results == ["ok"]


# ── Close ────────────────────────────────────────────────────────────


class TestClose:
    def test_close_clears_subscriptions(self, event_bus: InProcessBus) -> None:
        received: list[str] = []
        event_bus.subscribe("t", lambda t, d: received.append("x"))
        event_bus.close()
        event_bus.publish("t", {})

        assert received == []

    def test_close_idempotent(self, event_bus: InProcessBus) -> None:
        event_bus.close()
        event_bus.close()  # Should not raise
