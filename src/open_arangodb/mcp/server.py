"""OpenArangoDB MCP server -- exposes memory operations to Claude Code."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any
from uuid import uuid4

from open_arangodb.encryption.validator import EncryptionValidator
from open_arangodb.models import (
    AgentScope,
    Memory,
    RetrievalConfig,
    RetrievalRequest,
)

logger = logging.getLogger("open_arangodb")


class ArangoDBMCPServer:
    """MCP server exposing OpenArangoDB tools.

    Wraps an ArangoDB instance and exposes tools for:
    - Memory CRUD (insert, get, update, delete, supersede)
    - Search (vector similarity, retrieval pipeline)
    - Graph traversal
    - Audit log queries
    - CDC changes
    - Backup and encryption status
    """

    def __init__(self, argondb: Any) -> None:
        self._db = argondb
        self._tools = self._register_tools()

    def _register_tools(self) -> dict[str, dict[str, Any]]:
        """Register all MCP tool definitions."""
        return {
            "memory_insert": {
                "description": "Insert a new memory",
                "parameters": {
                    "content": {
                        "type": "string",
                        "description": "Memory content",
                        "required": True,
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags",
                    },
                    "entity": {"type": "string", "description": "Entity name"},
                    "category": {
                        "type": "string",
                        "description": "Category",
                        "default": "general",
                    },
                    "agent_id": {
                        "type": "string",
                        "description": "Agent ID for scoping",
                    },
                },
            },
            "memory_get": {
                "description": "Retrieve a memory by ID",
                "parameters": {
                    "memory_id": {"type": "string", "required": True},
                },
            },
            "memory_search": {
                "description": "Search memories by vector similarity",
                "parameters": {
                    "query": {"type": "string", "required": True},
                    "limit": {"type": "integer", "default": 20},
                },
            },
            "memory_update": {
                "description": "Update an existing memory",
                "parameters": {
                    "memory_id": {"type": "string", "required": True},
                    "content": {"type": "string", "required": True},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
            },
            "memory_delete": {
                "description": "Soft-delete a memory",
                "parameters": {
                    "memory_id": {"type": "string", "required": True},
                },
            },
            "memory_supersede": {
                "description": "Replace an old memory with a new one",
                "parameters": {
                    "old_id": {"type": "string", "required": True},
                    "content": {"type": "string", "required": True},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
            },
            "retrieval_search": {
                "description": "Multi-layer retrieval search with RRF fusion",
                "parameters": {
                    "query": {"type": "string", "required": True},
                    "layers": {"type": "array", "items": {"type": "string"}},
                    "entity": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "max_results": {"type": "integer", "default": 20},
                },
            },
            "audit_query": {
                "description": "Query audit logs",
                "parameters": {
                    "agent_id": {"type": "string"},
                    "op": {"type": "string"},
                    "since": {"type": "string"},
                    "limit": {"type": "integer", "default": 100},
                },
            },
            "changes_since": {
                "description": "Get CDC changes since a revision",
                "parameters": {
                    "rev": {"type": "string"},
                },
            },
            "encryption_check": {
                "description": "Check encryption at rest status",
                "parameters": {},
            },
        }

    def get_tools(self) -> list[dict[str, Any]]:
        """Return MCP tool definitions."""
        return [
            {"name": name, **defn}
            for name, defn in self._tools.items()
        ]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute an MCP tool call. Returns result dict."""
        handler = getattr(self, f"_handle_{name}", None)
        if not handler:
            return {"error": f"Unknown tool: {name}"}
        try:
            return handler(arguments)
        except Exception as exc:
            logger.exception("MCP tool error: %s", name)
            return {"error": str(exc)}

    # ── Tool Handlers ────────────────────────────────────────────────

    def _handle_memory_insert(self, args: dict[str, Any]) -> dict[str, Any]:
        memory = Memory(
            id=uuid4().hex[:16],
            content=args["content"],
            tags=args.get("tags", []),
            entity=args.get("entity"),
            category=args.get("category", "general"),
        )
        scope = (
            AgentScope(agent_id=args["agent_id"])
            if args.get("agent_id")
            else None
        )
        result = self._db.insert(memory, scope=scope)
        return {"memory_id": result.id, "status": "created"}

    def _handle_memory_get(self, args: dict[str, Any]) -> dict[str, Any]:
        result = self._db.get(args["memory_id"])
        if not result:
            return {"error": "Memory not found"}
        return {
            "memory_id": result.id,
            "content": result.content,
            "tags": result.tags,
            "entity": result.entity,
            "category": result.category,
            "status": result.status,
        }

    def _handle_memory_search(self, args: dict[str, Any]) -> dict[str, Any]:
        results = self._db.search(args["query"], limit=args.get("limit", 20))
        return {"results": results, "count": len(results)}

    def _handle_memory_update(self, args: dict[str, Any]) -> dict[str, Any]:
        old = self._db.get(args["memory_id"])
        if not old:
            return {"error": "Memory not found"}
        updated = replace(
            old,
            content=args["content"],
            tags=args.get("tags", old.tags),
        )
        result = self._db.update(updated)
        return {"memory_id": result.id, "status": "updated"}

    def _handle_memory_delete(self, args: dict[str, Any]) -> dict[str, Any]:
        self._db.delete(args["memory_id"])
        return {"memory_id": args["memory_id"], "status": "deleted"}

    def _handle_memory_supersede(self, args: dict[str, Any]) -> dict[str, Any]:
        new_memory = Memory(
            id=uuid4().hex[:16],
            content=args["content"],
            tags=args.get("tags", []),
        )
        result = self._db.supersede(args["old_id"], new_memory)
        return {
            "old_id": args["old_id"],
            "new_id": result.id,
            "status": "superseded",
        }

    def _handle_retrieval_search(self, args: dict[str, Any]) -> dict[str, Any]:
        # Use retrieval orchestrator if available on the ArangoDB instance
        if hasattr(self._db, "retrieve"):
            config = RetrievalConfig(
                layers=args.get(
                    "layers", ["exact", "tag", "semantic", "temporal"]
                ),
                max_results=args.get("max_results", 20),
            )
            request = RetrievalRequest(
                query=args["query"],
                entity=args.get("entity"),
                tags=args.get("tags"),
                config=config,
            )
            results = self._db.retrieve(request)
            return {
                "results": [
                    {
                        "memory_id": r.memory.id,
                        "content": r.memory.content,
                        "score": r.score,
                        "source": r.match_source,
                    }
                    for r in results
                ],
                "count": len(results),
            }
        # Fallback to basic vector search
        return self._handle_memory_search(
            {"query": args["query"], "limit": args.get("max_results", 20)}
        )

    def _handle_audit_query(self, args: dict[str, Any]) -> dict[str, Any]:
        if not hasattr(self._db, "_audit") or not self._db._audit:
            return {"error": "Audit logging not enabled"}
        results = self._db._audit.query(
            agent_id=args.get("agent_id"),
            op=args.get("op"),
            since=args.get("since"),
            limit=args.get("limit", 100),
        )
        return {"entries": results, "count": len(results)}

    def _handle_changes_since(self, args: dict[str, Any]) -> dict[str, Any]:
        changes = self._db.get_changes(since_rev=args.get("rev"))
        return {
            "changes": [
                {
                    "op": c.op.value,
                    "memory_id": c.memory_id,
                    "rev": c.rev,
                    "timestamp": c.timestamp,
                }
                for c in changes
            ],
            "count": len(changes),
        }

    def _handle_encryption_check(self, args: dict[str, Any]) -> dict[str, Any]:
        validator = EncryptionValidator()
        status = validator.check()
        return {
            "encrypted": status.encrypted,
            "method": status.method,
            "details": status.details,
            "checked_at": status.checked_at,
        }
