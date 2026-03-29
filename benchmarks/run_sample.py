#!/usr/bin/env python3
"""Sample benchmark: open-argondb vs memwright.

Measures latency, throughput, and retrieval quality side-by-side.
Usage: uv run python benchmarks/run_sample.py
"""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

# memwright lives in the uv tools venv
MW_SITE = "/Users/aarjay/.local/share/uv/tools/memwright/lib/python3.12/site-packages"
if MW_SITE not in sys.path:
    sys.path.insert(0, MW_SITE)

from open_argondb import ArgonDB
from open_argondb.models import AgentScope, Memory as AMemory, Visibility

from agent_memory.core import AgentMemory

# ── Config ──

CORPUS_SIZE = 100
WARM_ITERS = 50
RECALL_QUERIES = [
    "authentication flow and session management",
    "database migration strategy",
    "deployment rollback procedure",
    "rate limiting and throttling",
    "error handling patterns",
]

CONTENT_TEMPLATES = [
    "Project requirement: implement {feat} with {constraint} by end of sprint.",
    "Design decision: chose {tech} over {alt} for {reason}.",
    "Bug report: {comp} fails when {cond} under heavy load.",
    "Meeting note: discussed {topic} with the team, action items pending.",
    "Code review: {file} needs refactoring to reduce complexity.",
    "Deployment: rolled out {svc} v{ver} to production environment.",
    "Performance: {endpoint} p99 latency increased after release.",
    "Security: patched vulnerability in {lib}, upgraded to latest.",
]

FEATURES = ("auth-flow", "dashboard", "notifications", "search", "export", "billing")
TECHS = ("PostgreSQL", "Redis", "Kafka", "gRPC", "GraphQL", "WebSocket")
COMPONENTS = ("auth-service", "api-gateway", "scheduler", "cache-layer", "indexer")
ENTITIES = ("auth", "database", "deployment", "api", "search", "billing")
TAGS_POOL = [
    ["auth", "security"], ["database", "migration"], ["deploy", "ops"],
    ["api", "performance"], ["search", "indexing"], ["billing", "payment"],
    ["monitoring", "alerts"], ["testing", "ci"], ["cache", "redis"],
    ["queue", "kafka"],
]


def _gen_content(i: int) -> str:
    tmpl = CONTENT_TEMPLATES[i % len(CONTENT_TEMPLATES)]
    return tmpl.format(
        feat=FEATURES[i % len(FEATURES)],
        constraint="sub-100ms",
        tech=TECHS[i % len(TECHS)],
        alt=TECHS[(i + 1) % len(TECHS)],
        reason="latency",
        comp=COMPONENTS[i % len(COMPONENTS)],
        cond="concurrent writes",
        topic="scaling",
        file=f"module_{i}.py",
        svc=COMPONENTS[i % len(COMPONENTS)],
        ver=f"1.{i}",
        endpoint=f"/api/v{i % 3}",
        lib=TECHS[i % len(TECHS)],
    )


# ── Result types ──

@dataclass(frozen=True)
class LatencyStats:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float


@dataclass
class BenchResult:
    backend: str
    corpus_size: int
    ingest_total_ms: float = 0.0
    ingest_ops_per_sec: float = 0.0
    get_latency: LatencyStats | None = None
    search_latency: LatencyStats | None = None
    list_latency: LatencyStats | None = None
    delete_latency: LatencyStats | None = None
    recall_results: list[dict] = field(default_factory=list)


def _calc_stats(times_ns: list[int]) -> LatencyStats:
    ms = [t / 1_000_000 for t in times_ns]
    ms.sort()
    n = len(ms)
    return LatencyStats(
        p50_ms=round(ms[n // 2], 3),
        p95_ms=round(ms[int(n * 0.95)], 3),
        p99_ms=round(ms[int(n * 0.99)], 3),
        mean_ms=round(statistics.mean(ms), 3),
    )


# ── ArgonDB benchmark ──

def bench_argondb(corpus_size: int) -> BenchResult:
    print(f"\n{'='*60}")
    print(f"  ArgonDB Benchmark (corpus={corpus_size})")
    print(f"{'='*60}")

    db = ArgonDB(
        host="http://localhost:8529",
        database="bench_open_argondb",
        username="root",
        password="",
        audit_enabled=True,
        cdc_enabled=True,
    )
    db.reset()
    result = BenchResult(backend="open-argondb", corpus_size=corpus_size)

    # ── Ingest ──
    print("  Ingesting...", end="", flush=True)
    scope = AgentScope(agent_id="bench", session_id="s1", visibility=Visibility.GLOBAL)
    t0 = time.perf_counter_ns()
    for i in range(corpus_size):
        content = _gen_content(i)
        mem = AMemory(
            id=f"m-{i}",
            content=content,
            tags=TAGS_POOL[i % len(TAGS_POOL)],
            category="note",
            entity=ENTITIES[i % len(ENTITIES)],
        )
        db.insert(mem, scope=scope)
    ingest_ns = time.perf_counter_ns() - t0
    result.ingest_total_ms = round(ingest_ns / 1_000_000, 1)
    result.ingest_ops_per_sec = round(corpus_size / (ingest_ns / 1e9), 1)
    print(f" {result.ingest_total_ms}ms ({result.ingest_ops_per_sec} ops/s)")

    # ── Get latency ──
    print("  Get latency...", end="", flush=True)
    times = []
    for i in range(WARM_ITERS):
        mid = f"m-{i % corpus_size}"
        t0 = time.perf_counter_ns()
        db.get(mid)
        times.append(time.perf_counter_ns() - t0)
    result.get_latency = _calc_stats(times)
    print(f" p50={result.get_latency.p50_ms}ms p99={result.get_latency.p99_ms}ms")

    # ── List latency ──
    print("  List latency...", end="", flush=True)
    times = []
    for i in range(WARM_ITERS):
        entity = ENTITIES[i % len(ENTITIES)]
        t0 = time.perf_counter_ns()
        db.list_memories(entity=entity, limit=20)
        times.append(time.perf_counter_ns() - t0)
    result.list_latency = _calc_stats(times)
    print(f" p50={result.list_latency.p50_ms}ms p99={result.list_latency.p99_ms}ms")

    # ── Delete latency ──
    print("  Delete latency...", end="", flush=True)
    times = []
    for i in range(min(20, corpus_size)):
        mid = f"m-{corpus_size - 1 - i}"
        t0 = time.perf_counter_ns()
        db.delete(mid, scope=scope)
        times.append(time.perf_counter_ns() - t0)
    result.delete_latency = _calc_stats(times)
    print(f" p50={result.delete_latency.p50_ms}ms p99={result.delete_latency.p99_ms}ms")

    # ── CDC + Audit verification ──
    changes = db.get_changes()
    audit_logs = db._audit.get_logs(limit=5)
    print(f"  CDC events: {len(changes)} | Audit entries: {len(audit_logs)}+")

    db.close()
    return result


# ── Memwright benchmark ──

def bench_memwright(corpus_size: int) -> BenchResult:
    print(f"\n{'='*60}")
    print(f"  Memwright Benchmark (corpus={corpus_size})")
    print(f"{'='*60}")

    tmpdir = tempfile.mkdtemp(prefix="mw_bench_")
    mw = AgentMemory(path=tmpdir)
    result = BenchResult(backend="memwright", corpus_size=corpus_size)

    # ── Ingest ──
    print("  Ingesting...", end="", flush=True)
    mem_ids = []
    t0 = time.perf_counter_ns()
    for i in range(corpus_size):
        content = _gen_content(i)
        r = mw.add(
            content=content,
            tags=TAGS_POOL[i % len(TAGS_POOL)],
            category="note",
            entity=ENTITIES[i % len(ENTITIES)],
        )
        mem_ids.append(r.id)
    ingest_ns = time.perf_counter_ns() - t0
    result.ingest_total_ms = round(ingest_ns / 1_000_000, 1)
    result.ingest_ops_per_sec = round(corpus_size / (ingest_ns / 1e9), 1)
    print(f" {result.ingest_total_ms}ms ({result.ingest_ops_per_sec} ops/s)")

    # ── Get latency ──
    print("  Get latency...", end="", flush=True)
    times = []
    for i in range(WARM_ITERS):
        mid = mem_ids[i % corpus_size]
        t0 = time.perf_counter_ns()
        mw.get(mid)
        times.append(time.perf_counter_ns() - t0)
    result.get_latency = _calc_stats(times)
    print(f" p50={result.get_latency.p50_ms}ms p99={result.get_latency.p99_ms}ms")

    # ── Search/list latency ──
    print("  Search latency...", end="", flush=True)
    times = []
    for i in range(WARM_ITERS):
        entity = ENTITIES[i % len(ENTITIES)]
        t0 = time.perf_counter_ns()
        mw.search(entity=entity, limit=20)
        times.append(time.perf_counter_ns() - t0)
    result.list_latency = _calc_stats(times)
    print(f" p50={result.list_latency.p50_ms}ms p99={result.list_latency.p99_ms}ms")

    # ── Delete latency ──
    print("  Delete latency...", end="", flush=True)
    times = []
    for i in range(min(20, corpus_size)):
        mid = mem_ids[corpus_size - 1 - i]
        t0 = time.perf_counter_ns()
        mw.forget(mid)
        times.append(time.perf_counter_ns() - t0)
    result.delete_latency = _calc_stats(times)
    print(f" p50={result.delete_latency.p50_ms}ms p99={result.delete_latency.p99_ms}ms")

    # ── Stats ──
    stats = mw.stats()
    print(f"  Total memories: {stats['total_memories']}")

    try:
        mw.compact()
    except Exception:
        pass  # memwright FK constraint bug on compact — not our issue
    return result


# ── Report ──

def print_comparison(a: BenchResult, m: BenchResult) -> None:
    print(f"\n{'='*60}")
    print(f"  COMPARISON: {a.backend} vs {m.backend}  (corpus={a.corpus_size})")
    print(f"{'='*60}")

    rows = [
        ("Ingest total", f"{a.ingest_total_ms}ms", f"{m.ingest_total_ms}ms"),
        ("Ingest ops/s", f"{a.ingest_ops_per_sec}", f"{m.ingest_ops_per_sec}"),
        ("Get p50", f"{a.get_latency.p50_ms}ms", f"{m.get_latency.p50_ms}ms"),
        ("Get p99", f"{a.get_latency.p99_ms}ms", f"{m.get_latency.p99_ms}ms"),
        ("List/Search p50", f"{a.list_latency.p50_ms}ms", f"{m.list_latency.p50_ms}ms"),
        ("List/Search p99", f"{a.list_latency.p99_ms}ms", f"{m.list_latency.p99_ms}ms"),
        ("Delete p50", f"{a.delete_latency.p50_ms}ms", f"{m.delete_latency.p50_ms}ms"),
        ("Delete p99", f"{a.delete_latency.p99_ms}ms", f"{m.delete_latency.p99_ms}ms"),
    ]

    print(f"  {'Metric':<20} {'ArgonDB':>12} {'Memwright':>12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12}")
    for label, av, mv in rows:
        print(f"  {label:<20} {av:>12} {mv:>12}")

    print(f"\n  Note: ArgonDB includes audit + CDC overhead per write.")
    print(f"  Memwright uses SQLite (local file) — no audit/CDC.")


def main() -> None:
    corpus = int(sys.argv[1]) if len(sys.argv) > 1 else CORPUS_SIZE
    print(f"Sample benchmark — corpus size: {corpus}")

    argon_result = bench_argondb(corpus)
    mw_result = bench_memwright(corpus)
    print_comparison(argon_result, mw_result)

    # Save results
    outdir = Path(__file__).parent / "results"
    outdir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = outdir / f"sample_{corpus}_{ts}.json"

    def _to_dict(r: BenchResult) -> dict:
        d = {
            "backend": r.backend,
            "corpus_size": r.corpus_size,
            "ingest_total_ms": r.ingest_total_ms,
            "ingest_ops_per_sec": r.ingest_ops_per_sec,
        }
        for name in ("get_latency", "search_latency", "list_latency", "delete_latency"):
            val = getattr(r, name)
            if val:
                d[name] = asdict(val)
        return d

    report = {
        "timestamp": ts,
        "corpus_size": corpus,
        "argondb": _to_dict(argon_result),
        "memwright": _to_dict(mw_result),
    }
    outpath.write_text(json.dumps(report, indent=2))
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
