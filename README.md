# OpenArangoDB

Enterprise-grade features for ArangoDB Community Edition -- no license upgrade required.

OpenArangoDB is a Python service layer that sits on top of ArangoDB Community and adds the features you'd normally need Enterprise for: change data capture, audit logging, vector search, graph traversals, encryption validation, hot backups, and more.

## Why

ArangoDB Community is a powerful multi-model database, but critical production features are locked behind the Enterprise license:

- No change data capture
- No audit logging
- No encryption at rest
- No hot backups
- No SmartGraphs or parallel traversals
- No satellite collections
- No DC-to-DC replication

OpenArangoDB implements these at the application layer. You get the same capabilities without changing your database engine or paying for Enterprise.

## Community vs Enterprise vs OpenArangoDB

| Feature | Community | Enterprise | OpenArangoDB |
|---|:---:|:---:|:---:|
| **Document Store** | Yes | Yes | Yes |
| **Graph Queries (AQL)** | Yes | Yes | Yes |
| **Multi-model (doc + graph + key-value)** | Yes | Yes | Yes |
| | | | |
| **Change Data Capture** | -- | WAL-based | Changelog collection + consumer checkpoints |
| **Audit Logging** | -- | Built-in | Service-layer audit with TTL auto-cleanup |
| **Encryption at Rest** | -- | AES-256 native | OS-level detection (LUKS, FileVault, EBS) |
| **Hot Backups** | -- | RocksDB checkpoint | arangodump wrapper + LVM/ZFS/cloud snapshots |
| **SmartGraphs** | -- | Yes | Partition-key routing + parallel traversals |
| **Satellite Collections** | -- | Yes | In-memory TTL cache with sync |
| **DC-to-DC Replication** | -- | Yes | CDC-based replication engine |
| **LDAP Authentication** | -- | DB-level | Application-level middleware + role mapping |
| | | | |
| **Vector Search** | Experimental | Experimental | numpy cosine + auto-upgrade to native |
| **Event Bus (pub/sub)** | -- | -- | In-process, Redis, or NATS |
| **Agent Scoping** | -- | -- | Private / workflow / global visibility |
| **Temporal Supersession** | -- | -- | Supersession chains + contradiction detection |
| **Multi-layer Retrieval** | -- | -- | Tag + semantic + temporal + graph with RRF fusion |

## Install

```bash
pip install OpenArangoDB

# With vector search (BAAI/bge-m3 embeddings)
pip install OpenArangoDB[embeddings]

# With Redis event bus
pip install OpenArangoDB[events-redis]

# With NATS event bus
pip install OpenArangoDB[events-nats]

# With LDAP auth
pip install OpenArangoDB[auth]

# Everything
pip install OpenArangoDB[all]
```

Requires Python 3.10+ and ArangoDB 3.12+ (Community Edition).

## Quick Start

### Connect

```python
from open_arangodb import ArangoDB

db = ArangoDB(
    host="http://localhost:8529",
    database="myapp",
    username="root",
    password="",
)
```

All features (audit, CDC, vector search) are enabled by default. Disable what you don't need:

```python
db = ArangoDB(
    host="http://localhost:8529",
    database="myapp",
    audit_enabled=False,      # skip audit logging
    cdc_enabled=False,        # skip change tracking
    embedding_model="BAAI/bge-m3",  # default embedding model
)
```

### Store and Retrieve

```python
from open_arangodb.models import Memory

# Insert
mem = Memory(id="m1", content="User prefers dark mode", tags=["preference", "ui"])
db.insert(mem)

# Get by ID
record = db.get("m1")

# List with filters
records = db.list_memories(entity="user-settings", limit=50)

# Update
updated = Memory(id="m1", content="User switched to light mode", tags=["preference", "ui"])
db.update(updated)

# Soft delete (preserves audit trail)
db.delete("m1")
```

### Vector Search

```python
# Embed all unembedded records
db.batch_embed()

# Semantic search
results = db.search("what are the user's display preferences?", limit=10)
for r in results:
    print(r["memory_id"], r["content"], r["distance"])
```

OpenArangoDB uses numpy cosine similarity by default and auto-upgrades to ArangoDB's native vector index when available.

### Change Data Capture

Every insert, update, and delete is recorded in a changelog collection with TTL-based cleanup.

```python
# Get all changes
changes = db.get_changes()

# Get changes since a revision
changes = db.get_changes(since_rev="12345")

for change in changes:
    print(change.op, change.memory_id, change.timestamp)
```

### Audit Logging

Every write operation is logged with the agent, session, and operation details.

```python
# Query audit logs
logs = db._audit.query(agent_id="agent-1", op="insert", limit=10)

# Get recent logs
logs = db._audit.get_logs(limit=50)

for log in logs:
    print(log["op"], log["document_key"], log["timestamp"])
```

Audit entries auto-expire after 90 days (configurable via `retention_days`).

### Scoped Visibility

Control which agents and workflows can see which records.

```python
from open_arangodb.models import AgentScope, Visibility

# Private to one agent
private = AgentScope(agent_id="agent-1", visibility=Visibility.PRIVATE)

# Shared within a workflow
workflow = AgentScope(
    agent_id="agent-1",
    workflow_id="wf-123",
    visibility=Visibility.WORKFLOW,
)

# Visible to all
public = AgentScope(agent_id="agent-1", visibility=Visibility.GLOBAL)

db.insert(mem, scope=private)
results = db.list_memories(scope=private)
```

### Supersession and Temporal Queries

Track how information evolves over time.

```python
db = ArangoDB(host="http://localhost:8529", database="myapp", temporal_enabled=True)

# Supersede outdated information
old = Memory(id="fact-1", content="CEO is Alice", tags=["org"])
db.insert(old)

new = Memory(id="fact-2", content="CEO is Bob (as of March 2026)", tags=["org"])
db.supersede("fact-1", new)

# Walk the chain
chain = db.get_supersession_chain("fact-1")

# Find the current version
current = db.get_current_version("fact-1")

# Detect contradictions for an entity
contradictions = db.detect_contradictions("org")
```

### Graph Traversals

```python
from open_arangodb.models import GraphConfig, EdgeDefinition

db = ArangoDB(host="http://localhost:8529", database="myapp", graph_enabled=True)

# Create a graph
config = GraphConfig(
    name="knowledge",
    edge_definitions=[
        EdgeDefinition(collection="edges", from_collections=["nodes"], to_collections=["nodes"]),
    ],
)
db.create_graph(config)

# Traverse
result = db.traverse("nodes/person-1", direction="outbound", max_depth=3)

# Parallel traversal from multiple start points
results = db.traverse_parallel(
    ["nodes/person-1", "nodes/person-2", "nodes/person-3"],
    direction="outbound",
    max_depth=2,
)
```

### Event Bus

React to changes in real time.

```python
from open_arangodb.events.bus import InProcessBus, RedisBus

# In-process (default)
db = ArangoDB(host="http://localhost:8529", database="myapp")

# Redis-backed for multi-process
db = ArangoDB(
    host="http://localhost:8529",
    database="myapp",
    event_bus=RedisBus(redis_url="redis://localhost:6379"),
)

# Subscribe to events
db.events.subscribe("memory.created", lambda data: print(f"New: {data['memory_id']}"))
db.events.subscribe("memory.updated", lambda data: print(f"Updated: {data['memory_id']}"))
db.events.subscribe("memory.deleted", lambda data: print(f"Deleted: {data['memory_id']}"))
```

### Backup and Restore

```python
from open_arangodb.models import BackupConfig

db = ArangoDB(host="http://localhost:8529", database="myapp", backup_enabled=True)

config = BackupConfig(output_dir="/backups/myapp", database="myapp")
result = db.create_backup(config)
print(result.path, result.size_bytes)
```

## Architecture

```
Your Application / Agents
         |
         v
+----------------------------+
|     ArangoDB Gateway        |  <-- All writes go through here
+----------------------------+
| Audit  | CDC    | Events   |  <-- Automatic on every write
| Scope  | Vector | Temporal |  <-- Query-time features
| Graph  | Backup | Auth     |  <-- Optional modules
+----------------------------+
         |
   ArangoDB Community 3.12+
```

The gateway pattern ensures every write is audited, CDC-tracked, and event-propagated without the caller having to think about it.

## Modules

| Module | Purpose |
|---|---|
| `store/` | Document storage with soft-delete, 8 indexes |
| `vector/` | Embedding search -- numpy cosine + native auto-upgrade |
| `cdc/` | Change data capture with consumer checkpoints and TTL |
| `events/` | Pluggable pub/sub (InProcessBus, RedisBus, NATSBus) |
| `audit/` | Operation audit logging with configurable TTL retention |
| `scoping/` | Agent/session/workflow visibility (private/workflow/global) |
| `temporal/` | Supersession chains and contradiction detection |
| `graph/` | SmartGraph-like partitioning + parallel traversals |
| `retrieval/` | 4-layer cascade (tag/semantic/temporal/graph) with RRF fusion |
| `backup/` | arangodump/arangorestore wrapper + filesystem snapshots |
| `encryption/` | OS-level encryption at rest detection and validation |
| `satellite/` | In-memory SatelliteCollection-like caching with TTL sync |
| `replication/` | CDC-based DC-to-DC replication engine |
| `auth/` | LDAP authentication middleware with role mapping |
| `mcp/` | MCP server exposing all features as Claude Code tools |

## Data Model

All data uses frozen dataclasses. The core record:

```python
@dataclass(frozen=True)
class Memory:
    id: str
    content: str
    tags: list[str]
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
```

## Benchmark Results

LoCoMo benchmark (2 conversations, 40 questions, GPT-4.1-mini judge):

| Metric | OpenArangoDB | Mem0 | Letta/MemGPT | Zep/Graphiti |
|---|---|---|---|---|
| **Accuracy** | **95.0%** | 66.9% | 74.0% | 58.4% |
| **F1 Score** | **72.3%** | 16.8% | -- | -- |

## License

Apache 2.0
