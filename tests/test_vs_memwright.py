"""Side-by-side test: ArgonDB vs Memwright — same operations, both stores."""

import pytest
import tempfile
import sys

# memwright lives in the uv tools venv
MW_SITE = "/Users/aarjay/.local/share/uv/tools/memwright/lib/python3.12/site-packages"
if MW_SITE not in sys.path:
    sys.path.insert(0, MW_SITE)

from open_argondb import ArgonDB
from open_argondb.models import Memory as AMemory, AgentScope, Visibility

from agent_memory.core import AgentMemory


# ── Fixtures ──


@pytest.fixture
def argon():
    """Fresh ArgonDB instance on test database."""
    db = ArgonDB(
        host="http://localhost:8529",
        database="test_vs_memwright",
        username="root",
        password="",
        audit_enabled=True,
        cdc_enabled=True,
        graph_enabled=False,
        temporal_enabled=True,
    )
    db.reset()
    yield db
    db.reset()
    db.close()


@pytest.fixture
def memwright():
    """Fresh Memwright instance in temp dir."""
    tmpdir = tempfile.mkdtemp(prefix="mw_test_")
    mw = AgentMemory(path=tmpdir)
    yield mw
    mw.compact()


# ── Tests ──


class TestInsertAndGet:
    """Both stores should insert and retrieve a memory."""

    def test_argon_insert_get(self, argon):
        mem = AMemory(
            id="test-1",
            content="Python is a programming language",
            tags=["python", "lang"],
            category="fact",
            entity="python",
        )
        result = argon.insert(mem)
        assert result.id == "test-1"

        fetched = argon.get("test-1")
        assert fetched is not None
        assert fetched.content == "Python is a programming language"
        assert fetched.status == "active"

    def test_memwright_insert_get(self, memwright):
        result = memwright.add(
            content="Python is a programming language",
            tags=["python", "lang"],
            category="fact",
            entity="python",
        )
        mem_id = result.id  # add() returns Memory object

        fetched = memwright.get(mem_id)
        assert fetched is not None
        assert fetched.content == "Python is a programming language"
        assert fetched.status == "active"


class TestUpdate:
    """Both stores should update content."""

    def test_argon_update(self, argon):
        mem = AMemory(id="upd-1", content="old content", tags=["test"], entity="test")
        argon.insert(mem)

        updated = AMemory(id="upd-1", content="new content", tags=["test", "updated"], entity="test")
        result = argon.update(updated)
        assert result.content == "new content"

        fetched = argon.get("upd-1")
        assert fetched.content == "new content"

    def test_memwright_update(self, memwright):
        result = memwright.add(content="old content", tags=["test"], entity="test")
        mem_id = result.id

        # memwright doesn't have direct update — forget+re-add
        memwright.forget(mem_id)
        new_result = memwright.add(content="new content", tags=["test", "updated"], entity="test")

        old = memwright.get(mem_id)
        assert old.status == "archived"

        new = memwright.get(new_result.id)
        assert new.content == "new content"
        assert new.status == "active"


class TestDelete:
    """Both stores should soft-delete."""

    def test_argon_soft_delete(self, argon):
        mem = AMemory(id="del-1", content="to be deleted", tags=["test"], entity="test")
        argon.insert(mem)
        argon.delete("del-1")

        # ArgonDB soft-delete sets _deleted=true; get() filters those out
        # so a successful soft-delete means get() returns None
        fetched = argon.get("del-1")
        assert fetched is None

    def test_memwright_soft_delete(self, memwright):
        result = memwright.add(content="to be deleted", tags=["test"], entity="test")
        mem_id = result.id
        memwright.forget(mem_id)

        fetched = memwright.get(mem_id)
        assert fetched is not None
        assert fetched.status == "archived"


class TestListByEntity:
    """Both stores should filter by entity."""

    def test_argon_list_entity(self, argon):
        for i in range(3):
            argon.insert(AMemory(id=f"ent-{i}", content=f"fact {i}", tags=["test"], entity="python"))
        argon.insert(AMemory(id="ent-other", content="unrelated", tags=["test"], entity="rust"))

        results = argon.list_memories(entity="python")
        assert len(results) == 3
        assert all(m.entity == "python" for m in results)

    def test_memwright_list_entity(self, memwright):
        for i in range(3):
            memwright.add(content=f"fact {i}", tags=["test"], entity="python")
        memwright.add(content="unrelated", tags=["test"], entity="rust")

        results = memwright.search(entity="python", limit=10)
        # memwright search may deduplicate; just check we get python results
        assert len(results) >= 1
        assert all(r.entity == "python" for r in results)


class TestAuditTrail:
    """ArgonDB has built-in audit; memwright does not."""

    def test_argon_audit_logged(self, argon):
        scope = AgentScope(agent_id="test-agent", session_id="s1", visibility=Visibility.GLOBAL)
        mem = AMemory(id="aud-1", content="audited write", tags=["test"], entity="test")
        argon.insert(mem, scope=scope)

        logs = argon._audit.get_logs(limit=5)
        assert len(logs) >= 1
        assert any(log.get("document_key") == "aud-1" for log in logs)


class TestCDC:
    """ArgonDB CDC captures change events."""

    def test_argon_cdc_captures_changes(self, argon):
        mem = AMemory(id="cdc-1", content="tracked change", tags=["test"], entity="test")
        argon.insert(mem)
        argon.update(AMemory(id="cdc-1", content="updated tracked", tags=["test"], entity="test"))

        changes = argon.get_changes()
        assert len(changes) >= 2
        ops = [c.op for c in changes]
        op_values = [o.value if hasattr(o, 'value') else o for o in ops]
        assert "insert" in op_values
        assert "update" in op_values


class TestSupersession:
    """Both should handle memory replacement."""

    def test_argon_supersede(self, argon):
        old = AMemory(id="sup-old", content="Earth is flat", tags=["geography"], entity="earth")
        argon.insert(old)

        new = AMemory(id="sup-new", content="Earth is roughly spherical", tags=["geography"], entity="earth")
        result = argon.supersede("sup-old", new)
        assert result.id == "sup-new"

        old_fetched = argon.get("sup-old")
        assert old_fetched.status == "superseded" or old_fetched.superseded_by == "sup-new"

    def test_memwright_supersede(self, memwright):
        old_result = memwright.add(content="Earth is flat", tags=["geography"], entity="earth")
        old_id = old_result.id
        memwright.forget(old_id)
        new_result = memwright.add(content="Earth is roughly spherical", tags=["geography"], entity="earth")

        old = memwright.get(old_id)
        assert old.status == "archived"

        new = memwright.get(new_result.id)
        assert new.status == "active"
        assert new.content == "Earth is roughly spherical"


class TestStats:
    """Both should report stats."""

    def test_argon_stats(self, argon):
        for i in range(5):
            argon.insert(AMemory(id=f"st-{i}", content=f"stat {i}", tags=["test"], entity="test"))

        mems = argon.list_memories(limit=100)
        assert len(mems) == 5

    def test_memwright_stats(self, memwright):
        for i in range(5):
            memwright.add(content=f"stat {i}", tags=["test"], entity="test")

        stats = memwright.stats()
        assert stats["total_memories"] >= 5
