"""Core data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Visibility(str, Enum):
    PRIVATE = "private"
    WORKFLOW = "workflow"
    GLOBAL = "global"


class ChangeOp(str, Enum):
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    SUPERSEDE = "supersede"


@dataclass(frozen=True)
class AgentScope:
    """Identifies who owns and can access a memory."""

    agent_id: str
    session_id: str | None = None
    workflow_id: str | None = None
    visibility: Visibility = Visibility.GLOBAL


@dataclass(frozen=True)
class Memory:
    """A single memory record."""

    id: str
    content: str
    tags: list[str] = field(default_factory=list)
    category: str = "general"
    entity: str | None = None
    created_at: str = ""
    event_date: str | None = None
    valid_from: str = ""
    valid_until: str | None = None
    superseded_by: str | None = None
    confidence: float = 1.0
    status: str = "active"
    scope: AgentScope | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    """A scored memory from retrieval."""

    memory: Memory
    score: float
    match_source: str = "unknown"
    tier: int = 3  # 1=ids, 2=summary, 3=full


@dataclass(frozen=True)
class ChangeEvent:
    """A change event from the CDC engine."""

    op: ChangeOp
    memory_id: str
    rev: str
    timestamp: str
    agent_scope: AgentScope | None = None
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None


@dataclass(frozen=True)
class AuditEntry:
    """An audit log entry."""

    id: str
    op: str
    collection: str
    document_key: str
    agent_id: str | None = None
    session_id: str | None = None
    timestamp: str = ""
    content_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SupersessionChain:
    """A chain of superseded memories."""

    chain_id: str
    memory_ids: list[str] = field(default_factory=list)
    current_id: str = ""
    created_at: str = ""


@dataclass(frozen=True)
class Contradiction:
    """Two memories that may contradict each other."""

    memory_a_id: str
    memory_b_id: str
    reason: str
    confidence: float = 0.0
    detected_at: str = ""


@dataclass(frozen=True)
class TemporalQuery:
    """Parameters for temporal range queries."""

    entity: str | None = None
    start: str | None = None
    end: str | None = None
    include_superseded: bool = False


@dataclass(frozen=True)
class EdgeDefinition:
    """Defines an edge collection and its vertex endpoints."""

    collection: str
    from_vertex_collections: list[str] = field(default_factory=list)
    to_vertex_collections: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GraphConfig:
    """Configuration for creating a named graph."""

    name: str
    edge_definitions: list[EdgeDefinition] = field(default_factory=list)
    smart_attribute: str | None = None


@dataclass(frozen=True)
class TraversalResult:
    """Result of a graph traversal."""

    vertices: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    paths: list[list[str]] = field(default_factory=list)


@dataclass(frozen=True)
class PartitionKey:
    """Identifies a SmartGraph-like partition."""

    attribute: str
    value: str


@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration for the retrieval pipeline."""

    layers: list[str] = field(default_factory=lambda: ["exact", "tag", "semantic", "temporal"])
    rrf_k: int = 60
    max_results: int = 20
    min_score: float = 0.0


@dataclass(frozen=True)
class RetrievalRequest:
    """A retrieval query."""

    query: str
    entity: str | None = None
    tags: list[str] | None = None
    scope: AgentScope | None = None
    config: RetrievalConfig | None = None


@dataclass(frozen=True)
class BackupConfig:
    """Configuration for arangodump backup."""

    output_dir: str
    collections: list[str] | None = None
    include_system: bool = False
    compress: bool = True


@dataclass(frozen=True)
class BackupResult:
    """Result of a backup or restore operation."""

    path: str
    timestamp: str = ""
    size_bytes: int = 0
    collections: list[str] = field(default_factory=list)
    success: bool = True
    error: str | None = None


@dataclass(frozen=True)
class RestoreConfig:
    """Configuration for arangorestore."""

    input_dir: str
    collections: list[str] | None = None
    create_database: bool = True


@dataclass(frozen=True)
class SnapshotConfig:
    """Configuration for filesystem snapshots."""

    method: str  # "lvm", "zfs", "aws_ebs", "gcp_disk", "azure_disk"
    volume: str | None = None
    label: str | None = None


@dataclass(frozen=True)
class EncryptionStatus:
    """Result of encryption detection."""

    encrypted: bool
    method: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    checked_at: str = ""


@dataclass(frozen=True)
class EncryptionRequirement:
    """Required encryption configuration."""

    require_encryption: bool = True
    allowed_methods: list[str] = field(default_factory=lambda: [
        "luks", "filevault", "bitlocker", "aws_ebs", "gcp_cmek", "azure_sse",
    ])


@dataclass(frozen=True)
class SatelliteConfig:
    """Configuration for a satellite (cached) collection."""

    collection: str
    ttl_seconds: int = 300
    max_size: int = 10000
    backend: str = "memory"  # "memory" or "redis"


@dataclass(frozen=True)
class SatelliteStats:
    """Statistics for a satellite cache."""

    collection: str
    cached_count: int = 0
    hit_count: int = 0
    miss_count: int = 0
    last_sync: str = ""


@dataclass(frozen=True)
class ReplicationConfig:
    """Configuration for DC2DC replication."""

    source_host: str = ""
    target_host: str = ""
    database: str = "argondb"
    collections: list[str] | None = None
    batch_size: int = 100
    poll_interval_seconds: float = 5.0


@dataclass(frozen=True)
class ReplicationStatus:
    """Current status of replication."""

    state: str = "stopped"  # running, paused, error, stopped
    last_synced_rev: str | None = None
    pending_changes: int = 0
    lag_seconds: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class LDAPConfig:
    """LDAP server configuration."""

    server_url: str
    base_dn: str
    bind_dn: str = ""
    bind_password: str = ""
    user_filter: str = "(uid={username})"
    group_filter: str = "(member={user_dn})"
    tls: bool = True
    timeout: int = 10


@dataclass(frozen=True)
class AuthResult:
    """Result of authentication attempt."""

    authenticated: bool
    username: str = ""
    groups: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class RoleMapping:
    """Maps LDAP group to application role."""

    ldap_group: str
    argondb_role: str
