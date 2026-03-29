"""Event bus — in-process pub/sub with pluggable backends (Redis, NATS)."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("open_argondb")

Callback = Callable[[str, dict[str, Any]], None]


class EventBus(ABC):
    @abstractmethod
    def publish(self, topic: str, data: dict[str, Any]) -> None: ...

    @abstractmethod
    def subscribe(self, topic: str, callback: Callback) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class InProcessBus(EventBus):
    """Simple in-process pub/sub for single-process multi-agent setups."""

    def __init__(self) -> None:
        self._subs: dict[str, list[Callback]] = defaultdict(list)

    def publish(self, topic: str, data: dict[str, Any]) -> None:
        for cb in self._subs.get(topic, []):
            try:
                cb(topic, data)
            except Exception:
                logger.exception("Event callback error on topic=%s", topic)

    def subscribe(self, topic: str, callback: Callback) -> None:
        self._subs[topic].append(callback)

    def close(self) -> None:
        self._subs.clear()


class RedisBus(EventBus):
    """Redis pub/sub for distributed multi-agent setups."""

    def __init__(self, url: str = "redis://localhost:6379") -> None:
        import redis
        self._redis = redis.from_url(url)
        self._pubsub = self._redis.pubsub()
        self._subs: dict[str, list[Callback]] = defaultdict(list)

    def publish(self, topic: str, data: dict[str, Any]) -> None:
        import json
        self._redis.publish(f"argondb:{topic}", json.dumps(data))

    def subscribe(self, topic: str, callback: Callback) -> None:
        self._subs[topic].append(callback)
        self._pubsub.subscribe(**{f"argondb:{topic}": lambda msg: self._dispatch(topic, msg)})

    def _dispatch(self, topic: str, msg: Any) -> None:
        import json
        if msg["type"] == "message":
            data = json.loads(msg["data"])
            for cb in self._subs.get(topic, []):
                cb(topic, data)

    def close(self) -> None:
        self._pubsub.close()
        self._redis.close()
