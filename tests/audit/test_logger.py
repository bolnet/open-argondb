"""Unit tests for AuditLogger."""

from __future__ import annotations

import pytest

from open_arangodb.audit.logger import AuditLogger
from open_arangodb.models import AgentScope, Visibility


# ── log() ────────────────────────────────────────────────────────────


class TestLog:
    def test_log_creates_document(self, audit_logger: AuditLogger) -> None:
        audit_logger.log("insert", "memories", "mem-1")
        results = audit_logger.query()
        assert len(results) == 1

    def test_log_stores_op_and_key(self, audit_logger: AuditLogger) -> None:
        audit_logger.log("update", "memories", "mem-42")
        results = audit_logger.query()
        assert results[0]["op"] == "update"
        assert results[0]["collection"] == "memories"
        assert results[0]["document_key"] == "mem-42"

    def test_log_stores_scope_fields(self, audit_logger: AuditLogger) -> None:
        scope = AgentScope(
            agent_id="agent-1",
            session_id="sess-1",
            workflow_id="wf-1",
            visibility=Visibility.PRIVATE,
        )
        audit_logger.log("insert", "memories", "mem-1", scope=scope)
        results = audit_logger.query()
        assert results[0]["agent_id"] == "agent-1"
        assert results[0]["session_id"] == "sess-1"
        assert results[0]["workflow_id"] == "wf-1"

    def test_log_without_scope_sets_none(self, audit_logger: AuditLogger) -> None:
        audit_logger.log("insert", "memories", "mem-1")
        results = audit_logger.query()
        assert results[0]["agent_id"] is None
        assert results[0]["session_id"] is None

    def test_log_stores_content_hash(self, audit_logger: AuditLogger) -> None:
        audit_logger.log("insert", "memories", "mem-1", content_hash="abc123")
        results = audit_logger.query()
        assert results[0]["content_hash"] == "abc123"

    def test_log_sets_timestamp(self, audit_logger: AuditLogger) -> None:
        audit_logger.log("insert", "memories", "mem-1")
        results = audit_logger.query()
        assert results[0]["timestamp"]  # non-empty

    def test_log_sets_ttl_expires_at(self, audit_logger: AuditLogger) -> None:
        audit_logger.log("insert", "memories", "mem-1", retention_days=30)
        results = audit_logger.query()
        assert "expires_at" in results[0]
        assert results[0]["expires_at"]  # non-empty


# ── query() with filters ────────────────────────────────────────────


class TestQuery:
    def test_query_all(self, audit_logger: AuditLogger) -> None:
        audit_logger.log("insert", "memories", "m1")
        audit_logger.log("update", "memories", "m2")
        results = audit_logger.query()
        assert len(results) == 2

    def test_query_by_agent_id(self, audit_logger: AuditLogger) -> None:
        scope_a = AgentScope(agent_id="agent-a")
        scope_b = AgentScope(agent_id="agent-b")
        audit_logger.log("insert", "memories", "m1", scope=scope_a)
        audit_logger.log("insert", "memories", "m2", scope=scope_b)

        results = audit_logger.query(agent_id="agent-a")
        assert len(results) == 1
        assert results[0]["agent_id"] == "agent-a"

    def test_query_by_op(self, audit_logger: AuditLogger) -> None:
        audit_logger.log("insert", "memories", "m1")
        audit_logger.log("delete", "memories", "m2")

        results = audit_logger.query(op="delete")
        assert len(results) == 1
        assert results[0]["op"] == "delete"

    def test_query_by_since(self, audit_logger: AuditLogger) -> None:
        audit_logger.log("insert", "memories", "m1")
        entries = audit_logger.query()
        ts = entries[0]["timestamp"]

        audit_logger.log("insert", "memories", "m2")
        results = audit_logger.query(since=ts)
        assert len(results) >= 1
        keys = [r["document_key"] for r in results]
        assert "m2" in keys

    def test_query_respects_limit(self, audit_logger: AuditLogger) -> None:
        for i in range(5):
            audit_logger.log("insert", "memories", f"m{i}")
        results = audit_logger.query(limit=2)
        assert len(results) == 2


# ── reset() ──────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_audit_log(self, audit_logger: AuditLogger) -> None:
        audit_logger.log("insert", "memories", "m1")
        audit_logger.reset()
        results = audit_logger.query()
        assert len(results) == 0
