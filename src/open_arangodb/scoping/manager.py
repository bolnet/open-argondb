"""Agent scoping — controls memory visibility across agents and workflows."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from open_arangodb.models import AgentScope, Memory, Visibility


class ScopeManager:
    """Manages agent/session/workflow scoping on memories."""

    def __init__(self, db: Any) -> None:
        self._db = db

    def apply(self, memory: Memory, scope: AgentScope) -> Memory:
        """Attach scope to a memory (returns new frozen copy)."""
        return replace(memory, scope=scope)

    def filter_results(
        self, results: list[dict[str, Any]], scope: AgentScope
    ) -> list[dict[str, Any]]:
        """Filter search results by visibility rules."""
        filtered = []
        for r in results:
            mid = r["memory_id"]
            doc = self._get_scope_fields(mid)
            if not doc:
                filtered.append(r)
                continue

            vis = doc.get("scope_visibility", "global")
            doc_agent = doc.get("scope_agent_id")
            doc_workflow = doc.get("scope_workflow_id")

            if vis == Visibility.GLOBAL.value:
                filtered.append(r)
            elif vis == Visibility.WORKFLOW.value and doc_workflow == scope.workflow_id:
                filtered.append(r)
            elif vis == Visibility.PRIVATE.value and doc_agent == scope.agent_id:
                filtered.append(r)

        return filtered

    def _get_scope_fields(self, memory_id: str) -> dict[str, Any] | None:
        cursor = self._db.aql.execute(
            "FOR doc IN memories FILTER doc.memory_id == @mid LIMIT 1 "
            "RETURN {scope_visibility: doc.scope_visibility, "
            "scope_agent_id: doc.scope_agent_id, "
            "scope_workflow_id: doc.scope_workflow_id}",
            bind_vars={"mid": memory_id},
        )
        docs = list(cursor)
        return docs[0] if docs else None
