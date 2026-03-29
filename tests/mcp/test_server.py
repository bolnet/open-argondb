"""Unit tests for ArangoDB MCP server."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from open_arangodb.mcp.server import ArangoDBMCPServer
from open_arangodb.models import (
    ChangeEvent,
    ChangeOp,
    EncryptionStatus,
    Memory,
    RetrievalResult,
)


# ── Fixtures ─────────────────────────────────────────────────────────


def _make_memory(
    mid: str = "mem-1",
    content: str = "test content",
    tags: list[str] | None = None,
    entity: str | None = None,
    category: str = "general",
    status: str = "active",
) -> Memory:
    return Memory(
        id=mid,
        content=content,
        tags=tags or ["t1"],
        entity=entity,
        category=category,
        status=status,
    )


@pytest.fixture
def mock_db() -> MagicMock:
    """Create a MagicMock that mimics ArangoDB gateway methods."""
    db = MagicMock()
    db._audit = MagicMock()
    return db


@pytest.fixture
def server(mock_db: MagicMock) -> ArangoDBMCPServer:
    return ArangoDBMCPServer(mock_db)


# ── Tool Definitions ─────────────────────────────────────────────────


def test_get_tools(server: ArangoDBMCPServer) -> None:
    """get_tools returns a list of tool definitions with name keys."""
    tools = server.get_tools()
    assert isinstance(tools, list)
    assert len(tools) > 0

    names = {t["name"] for t in tools}
    expected = {
        "memory_insert",
        "memory_get",
        "memory_search",
        "memory_update",
        "memory_delete",
        "memory_supersede",
        "retrieval_search",
        "audit_query",
        "changes_since",
        "encryption_check",
    }
    assert expected == names

    # Each tool has description and parameters
    for tool in tools:
        assert "description" in tool
        assert "parameters" in tool


# ── Memory CRUD ──────────────────────────────────────────────────────


def test_memory_insert(
    server: ArangoDBMCPServer, mock_db: MagicMock
) -> None:
    """memory_insert creates a memory via the gateway."""
    returned = _make_memory(mid="new-1")
    mock_db.insert.return_value = returned

    result = server.call_tool(
        "memory_insert",
        {"content": "hello world", "tags": ["greeting"], "category": "chat"},
    )

    assert result["status"] == "created"
    assert result["memory_id"] == "new-1"
    mock_db.insert.assert_called_once()

    # Verify the Memory passed to insert has correct fields
    call_args = mock_db.insert.call_args
    memory_arg = call_args[0][0]
    assert memory_arg.content == "hello world"
    assert memory_arg.tags == ["greeting"]
    assert memory_arg.category == "chat"
    assert call_args[1]["scope"] is None


def test_memory_insert_with_agent_scope(
    server: ArangoDBMCPServer, mock_db: MagicMock
) -> None:
    """memory_insert passes agent scope when agent_id is provided."""
    returned = _make_memory(mid="scoped-1")
    mock_db.insert.return_value = returned

    result = server.call_tool(
        "memory_insert",
        {"content": "scoped content", "agent_id": "agent-42"},
    )

    assert result["status"] == "created"
    call_args = mock_db.insert.call_args
    scope = call_args[1]["scope"]
    assert scope is not None
    assert scope.agent_id == "agent-42"


def test_memory_get(
    server: ArangoDBMCPServer, mock_db: MagicMock
) -> None:
    """memory_get retrieves a memory by ID."""
    mem = _make_memory(mid="mem-1", content="found it", entity="alice")
    mock_db.get.return_value = mem

    result = server.call_tool("memory_get", {"memory_id": "mem-1"})

    assert result["memory_id"] == "mem-1"
    assert result["content"] == "found it"
    assert result["entity"] == "alice"
    assert result["category"] == "general"
    assert result["status"] == "active"
    mock_db.get.assert_called_once_with("mem-1")


def test_memory_get_not_found(
    server: ArangoDBMCPServer, mock_db: MagicMock
) -> None:
    """memory_get returns error when memory does not exist."""
    mock_db.get.return_value = None

    result = server.call_tool("memory_get", {"memory_id": "no-such"})

    assert "error" in result
    assert "not found" in result["error"].lower()


def test_memory_search(
    server: ArangoDBMCPServer, mock_db: MagicMock
) -> None:
    """memory_search delegates to db.search with limit."""
    mock_db.search.return_value = [
        {"id": "r1", "score": 0.9},
        {"id": "r2", "score": 0.8},
    ]

    result = server.call_tool(
        "memory_search", {"query": "what happened?", "limit": 5}
    )

    assert result["count"] == 2
    assert len(result["results"]) == 2
    mock_db.search.assert_called_once_with("what happened?", limit=5)


def test_memory_update(
    server: ArangoDBMCPServer, mock_db: MagicMock
) -> None:
    """memory_update fetches then updates the memory."""
    old = _make_memory(mid="mem-1", content="old content", tags=["old"])
    updated = _make_memory(mid="mem-1", content="new content", tags=["new"])
    mock_db.get.return_value = old
    mock_db.update.return_value = updated

    result = server.call_tool(
        "memory_update",
        {"memory_id": "mem-1", "content": "new content", "tags": ["new"]},
    )

    assert result["status"] == "updated"
    assert result["memory_id"] == "mem-1"
    mock_db.get.assert_called_once_with("mem-1")
    mock_db.update.assert_called_once()

    # Verify the updated Memory
    updated_arg = mock_db.update.call_args[0][0]
    assert updated_arg.content == "new content"
    assert updated_arg.tags == ["new"]


def test_memory_update_not_found(
    server: ArangoDBMCPServer, mock_db: MagicMock
) -> None:
    """memory_update returns error when memory does not exist."""
    mock_db.get.return_value = None

    result = server.call_tool(
        "memory_update", {"memory_id": "gone", "content": "x"}
    )

    assert "error" in result
    assert "not found" in result["error"].lower()


def test_memory_delete(
    server: ArangoDBMCPServer, mock_db: MagicMock
) -> None:
    """memory_delete soft-deletes via the gateway."""
    result = server.call_tool("memory_delete", {"memory_id": "mem-1"})

    assert result["status"] == "deleted"
    assert result["memory_id"] == "mem-1"
    mock_db.delete.assert_called_once_with("mem-1")


def test_memory_supersede(
    server: ArangoDBMCPServer, mock_db: MagicMock
) -> None:
    """memory_supersede replaces old memory with new one."""
    new_mem = _make_memory(mid="new-2", content="replacement")
    mock_db.supersede.return_value = new_mem

    result = server.call_tool(
        "memory_supersede",
        {"old_id": "old-1", "content": "replacement", "tags": ["v2"]},
    )

    assert result["status"] == "superseded"
    assert result["old_id"] == "old-1"
    assert result["new_id"] == "new-2"
    mock_db.supersede.assert_called_once()

    call_args = mock_db.supersede.call_args
    assert call_args[0][0] == "old-1"
    new_arg = call_args[0][1]
    assert new_arg.content == "replacement"
    assert new_arg.tags == ["v2"]


# ── Retrieval Search ─────────────────────────────────────────────────


def test_retrieval_search_with_orchestrator(
    server: ArangoDBMCPServer, mock_db: MagicMock
) -> None:
    """retrieval_search uses retrieve() when available."""
    mock_db.retrieve.return_value = [
        RetrievalResult(
            memory=_make_memory(mid="r1", content="match 1"),
            score=0.95,
            match_source="semantic",
        ),
    ]

    result = server.call_tool(
        "retrieval_search",
        {"query": "test query", "layers": ["semantic"], "max_results": 10},
    )

    assert result["count"] == 1
    assert result["results"][0]["memory_id"] == "r1"
    assert result["results"][0]["score"] == 0.95
    assert result["results"][0]["source"] == "semantic"


def test_retrieval_search_fallback(
    server: ArangoDBMCPServer, mock_db: MagicMock
) -> None:
    """retrieval_search falls back to basic search when retrieve() missing."""
    # Remove the retrieve attribute so hasattr returns False
    del mock_db.retrieve
    mock_db.search.return_value = [{"id": "r1", "score": 0.7}]

    result = server.call_tool(
        "retrieval_search", {"query": "fallback query"}
    )

    assert result["count"] == 1
    mock_db.search.assert_called_once()


# ── Audit & CDC ──────────────────────────────────────────────────────


def test_audit_query(
    server: ArangoDBMCPServer, mock_db: MagicMock
) -> None:
    """audit_query delegates to _audit.query."""
    mock_db._audit.query.return_value = [
        {"op": "insert", "document_key": "mem-1"},
    ]

    result = server.call_tool(
        "audit_query", {"agent_id": "agent-1", "limit": 50}
    )

    assert result["count"] == 1
    assert len(result["entries"]) == 1
    mock_db._audit.query.assert_called_once_with(
        agent_id="agent-1", op=None, since=None, limit=50
    )


def test_audit_query_not_enabled(
    server: ArangoDBMCPServer, mock_db: MagicMock
) -> None:
    """audit_query returns error when audit logging is disabled."""
    mock_db._audit = None

    result = server.call_tool("audit_query", {})

    assert "error" in result
    assert "not enabled" in result["error"].lower()


def test_changes_since(
    server: ArangoDBMCPServer, mock_db: MagicMock
) -> None:
    """changes_since returns CDC change events."""
    mock_db.get_changes.return_value = [
        ChangeEvent(
            op=ChangeOp.INSERT,
            memory_id="mem-1",
            rev="rev-100",
            timestamp="2026-01-01T00:00:00Z",
        ),
        ChangeEvent(
            op=ChangeOp.UPDATE,
            memory_id="mem-2",
            rev="rev-101",
            timestamp="2026-01-01T00:01:00Z",
        ),
    ]

    result = server.call_tool("changes_since", {"rev": "rev-99"})

    assert result["count"] == 2
    assert result["changes"][0]["op"] == "insert"
    assert result["changes"][0]["memory_id"] == "mem-1"
    assert result["changes"][1]["op"] == "update"
    mock_db.get_changes.assert_called_once_with(since_rev="rev-99")


# ── Encryption ───────────────────────────────────────────────────────


def test_encryption_check(server: ArangoDBMCPServer) -> None:
    """encryption_check returns encryption status."""
    fake_status = EncryptionStatus(
        encrypted=True,
        method="filevault",
        details={"path": "/"},
        checked_at="2026-01-01T00:00:00Z",
    )

    with patch.object(
        __import__(
            "open_arangodb.encryption.validator", fromlist=["EncryptionValidator"]
        ).EncryptionValidator,
        "check",
        return_value=fake_status,
    ):
        result = server.call_tool("encryption_check", {})

    assert result["encrypted"] is True
    assert result["method"] == "filevault"
    assert result["checked_at"] == "2026-01-01T00:00:00Z"


# ── Error Handling ───────────────────────────────────────────────────


def test_unknown_tool(server: ArangoDBMCPServer) -> None:
    """Calling an unknown tool returns an error dict."""
    result = server.call_tool("nonexistent_tool", {})

    assert "error" in result
    assert "Unknown tool" in result["error"]
    assert "nonexistent_tool" in result["error"]


def test_tool_exception(
    server: ArangoDBMCPServer, mock_db: MagicMock
) -> None:
    """Exceptions in tool handlers are caught and returned as errors."""
    mock_db.get.side_effect = RuntimeError("database connection lost")

    result = server.call_tool("memory_get", {"memory_id": "mem-1"})

    assert "error" in result
    assert "database connection lost" in result["error"]
