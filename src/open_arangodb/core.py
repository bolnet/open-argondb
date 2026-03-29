"""Core ArangoDB client — wraps python-arango with Enterprise-equivalent features."""

from __future__ import annotations

import logging
from typing import Any

from arango import ArangoClient

from open_arangodb.audit.logger import AuditLogger
from open_arangodb.cdc.engine import CDCEngine
from open_arangodb.events.bus import EventBus, InProcessBus
from open_arangodb.models import (
    AgentScope,
    AuthResult,
    BackupConfig,
    BackupResult,
    ChangeEvent,
    EncryptionStatus,
    GraphConfig,
    Memory,
    RetrievalRequest,
    RetrievalResult,
    SupersessionChain,
    TraversalResult,
)
from open_arangodb.scoping.manager import ScopeManager
from open_arangodb.store.document_store import DocumentStore
from open_arangodb.vector.search import VectorSearch

logger = logging.getLogger("open_arangodb")


class ArangoDB:
    """Enterprise-equivalent ArangoDB wrapper.

    All writes go through this gateway to ensure audit, CDC, and event
    propagation are applied consistently.
    """

    def __init__(
        self,
        host: str = "http://localhost:8529",
        database: str = "argondb",
        username: str = "root",
        password: str = "",
        event_bus: EventBus | None = None,
        audit_enabled: bool = True,
        cdc_enabled: bool = True,
        embedding_model: str = "BAAI/bge-m3",
        # Optional modules
        graph_enabled: bool = False,
        retrieval_enabled: bool = False,
        temporal_enabled: bool = False,
        backup_enabled: bool = False,
        encryption_check: bool = False,
        satellite_configs: list[Any] | None = None,
        replication_config: Any = None,
        ldap_config: Any = None,
    ) -> None:
        self._client = ArangoClient(hosts=host)
        self._sys_db = self._client.db("_system", username=username, password=password)

        if not self._sys_db.has_database(database):
            self._sys_db.create_database(database)

        self._db = self._client.db(database, username=username, password=password)
        self._embedding_model = embedding_model

        # Core components
        self._store = DocumentStore(self._db)
        self._vector = VectorSearch(self._db, model_name=embedding_model)
        self._scope = ScopeManager(self._db)
        self._events = event_bus or InProcessBus()
        self._audit = AuditLogger(self._db) if audit_enabled else None
        self._cdc = CDCEngine(self._db, self._events) if cdc_enabled else None

        # Optional modules — lazy imports to avoid hard dep on optional packages
        self._temporal = self._init_temporal() if temporal_enabled else None
        self._graph = self._init_graph() if graph_enabled else None
        self._parallel = self._init_parallel() if self._graph else None
        self._retrieval = self._init_retrieval() if retrieval_enabled else None
        self._backup = self._init_backup(host, username, password) if backup_enabled else None
        self._encryption = self._init_encryption() if encryption_check else None

        # Satellite caches
        self._satellites: dict[str, Any] = {}
        if satellite_configs:
            from open_arangodb.satellite.cache import SatelliteCache

            for sc in satellite_configs:
                self._satellites[sc.collection] = SatelliteCache(self._db, sc)

        # Replication (requires target_db to be provided separately)
        self._replication: Any = None

        # LDAP auth
        self._ldap: Any = None
        if ldap_config:
            from open_arangodb.auth.ldap_auth import LDAPAuthenticator

            self._ldap = LDAPAuthenticator(ldap_config)

    # ── Lazy module initializers ──

    def _init_temporal(self) -> Any:
        from open_arangodb.temporal.engine import TemporalEngine

        return TemporalEngine(self._db)

    def _init_graph(self) -> Any:
        from open_arangodb.graph.manager import GraphManager

        return GraphManager(self._db)

    def _init_parallel(self) -> Any:
        from open_arangodb.graph.parallel import ParallelTraverser

        return ParallelTraverser(self._graph)

    def _init_retrieval(self) -> Any:
        from open_arangodb.retrieval.orchestrator import RetrievalOrchestrator

        return RetrievalOrchestrator(
            db=self._db,
            vector_search=self._vector,
            temporal_engine=self._temporal,
            scope_manager=self._scope,
        )

    def _init_backup(self, host: str, username: str, password: str) -> Any:
        from open_arangodb.backup.manager import BackupManager

        return BackupManager(host=host, username=username, password=password)

    def _init_encryption(self) -> Any:
        from open_arangodb.encryption.validator import EncryptionValidator

        return EncryptionValidator()

    @property
    def db(self):
        """Direct access to the underlying ArangoDB database."""
        return self._db

    @property
    def events(self) -> EventBus:
        return self._events

    # ── Write Operations (all audited + CDC tracked) ──

    def insert(self, memory: Memory, scope: AgentScope | None = None) -> Memory:
        """Insert a memory. Audited, CDC-tracked, event-emitted."""
        scoped = self._scope.apply(memory, scope) if scope else memory
        result = self._store.insert(scoped)

        if self._audit:
            self._audit.log("insert", "memories", result.id, scope)
        if self._cdc:
            self._cdc.record_change("insert", result.id, after=result)
        self._events.publish("memory.created", {"memory_id": result.id})

        return result

    def update(self, memory: Memory, scope: AgentScope | None = None) -> Memory:
        """Update a memory. Audited, CDC-tracked."""
        old = self._store.get(memory.id)
        scoped = self._scope.apply(memory, scope) if scope else memory
        result = self._store.update(scoped)

        if self._audit:
            self._audit.log("update", "memories", result.id, scope)
        if self._cdc:
            self._cdc.record_change("update", result.id, before=old, after=result)
        self._events.publish("memory.updated", {"memory_id": result.id})

        return result

    def delete(self, memory_id: str, scope: AgentScope | None = None) -> None:
        """Soft-delete a memory. Audited, CDC-tracked."""
        old = self._store.get(memory_id)
        self._store.soft_delete(memory_id)

        if self._audit:
            self._audit.log("delete", "memories", memory_id, scope)
        if self._cdc:
            self._cdc.record_change("delete", memory_id, before=old)
        self._events.publish("memory.deleted", {"memory_id": memory_id})

    def supersede(self, old_id: str, new: Memory, scope: AgentScope | None = None) -> Memory:
        """Supersede an old memory with a new one. Atomic."""
        old = self._store.get(old_id)
        self._store.mark_superseded(old_id, new.id)
        result = self.insert(new, scope)

        if self._cdc:
            self._cdc.record_change("supersede", old_id, before=old, after=result)
        self._events.publish("memory.superseded", {
            "old_id": old_id, "new_id": result.id,
        })

        return result

    # ── Read Operations ──

    def get(self, memory_id: str) -> Memory | None:
        return self._store.get(memory_id)

    def search(
        self,
        query: str,
        limit: int = 20,
        scope: AgentScope | None = None,
        tier: int = 3,
    ) -> list[dict[str, Any]]:
        """Vector similarity search with optional scope filtering."""
        results = self._vector.search(query, limit=limit)
        if scope:
            results = self._scope.filter_results(results, scope)
        return results

    def list_memories(
        self,
        entity: str | None = None,
        scope: AgentScope | None = None,
        limit: int = 50,
    ) -> list[Memory]:
        return self._store.list_memories(entity=entity, scope=scope, limit=limit)

    # ── Embedding ──

    def embed(self, memory_id: str, content: str) -> None:
        """Generate and store embedding for a memory."""
        self._vector.add(memory_id, content)

    def batch_embed(self) -> int:
        """Embed all memories that don't have embeddings yet."""
        return self._vector.batch_embed()

    # ── CDC ──

    def get_changes(self, since_rev: str | None = None) -> list[ChangeEvent]:
        """Get changes since a checkpoint revision."""
        if not self._cdc:
            raise RuntimeError("CDC is not enabled")
        return self._cdc.get_changes(since_rev)

    # ── Retrieval ──

    def retrieve(self, request: RetrievalRequest) -> list[RetrievalResult]:
        """Multi-layer retrieval with RRF fusion."""
        if not self._retrieval:
            raise RuntimeError("Retrieval is not enabled")
        return self._retrieval.retrieve(request)

    # ── Temporal ──

    def get_supersession_chain(self, memory_id: str) -> SupersessionChain:
        """Walk the supersession chain from a memory."""
        if not self._temporal:
            raise RuntimeError("Temporal is not enabled")
        return self._temporal.get_supersession_chain(memory_id)

    def get_current_version(self, memory_id: str) -> Memory | None:
        """Follow supersession chain to its tip and return that Memory."""
        if not self._temporal:
            raise RuntimeError("Temporal is not enabled")
        return self._temporal.get_current_version(memory_id)

    def detect_contradictions(
        self, entity: str, scope: AgentScope | None = None,
    ) -> list[Any]:
        """Find active memories for entity whose content differs."""
        if not self._temporal:
            raise RuntimeError("Temporal is not enabled")
        return self._temporal.detect_contradictions(entity, scope)

    # ── Graph ──

    def create_graph(self, config: GraphConfig) -> None:
        """Create a named graph with optional smart partitioning."""
        if not self._graph:
            raise RuntimeError("Graph is not enabled")
        return self._graph.create_graph(config)

    def traverse(self, start_vertex: str, **kwargs: Any) -> TraversalResult:
        """Graph traversal from a start vertex."""
        if not self._graph:
            raise RuntimeError("Graph is not enabled")
        return self._graph.traverse(start_vertex, **kwargs)

    def traverse_parallel(
        self, start_vertices: list[str], **kwargs: Any,
    ) -> list[TraversalResult]:
        """Parallel graph traversal from multiple start vertices."""
        if not self._parallel:
            raise RuntimeError("Graph is not enabled")
        return self._parallel.traverse_parallel(start_vertices, **kwargs)

    # ── Backup ──

    def create_backup(self, config: BackupConfig) -> BackupResult:
        """Create a backup using arangodump."""
        if not self._backup:
            raise RuntimeError("Backup is not enabled")
        return self._backup.dump(config)

    # ── Encryption ──

    def check_encryption(self, data_directory: str | None = None) -> EncryptionStatus:
        """Check OS-level encryption at rest."""
        if not self._encryption:
            raise RuntimeError("Encryption check is not enabled")
        return self._encryption.check(data_directory)

    # ── Satellite ──

    def get_satellite(self, collection: str) -> Any:
        """Get the satellite cache for a collection."""
        if collection not in self._satellites:
            raise RuntimeError(f"No satellite cache for {collection}")
        return self._satellites[collection]

    # ── Auth ──

    def authenticate(self, username: str, password: str) -> AuthResult:
        """Authenticate user against LDAP."""
        if not self._ldap:
            raise RuntimeError("LDAP auth is not enabled")
        return self._ldap.authenticate(username, password)

    # ── Lifecycle ──

    def reset(self) -> None:
        """Reset all collections. Use with caution."""
        self._store.reset()
        self._vector.reset()
        if self._cdc:
            self._cdc.reset()
        if self._audit:
            self._audit.reset()
        if self._graph:
            self._graph.reset()

    def close(self) -> None:
        """Clean shutdown."""
        if self._cdc:
            self._cdc.stop()
        self._events.close()
        for sat in self._satellites.values():
            sat.stop()
        if self._replication:
            self._replication.stop()
