"""Tests for plan revision loop (#1).

Verifies that:
- TaskNode has failure_reason, actual_tool_used fields
- (§4PT: request_revision was removed with the ReplanBridge, its only caller;
  §4PV: revision_count, which nothing wrote after that, was removed too)
- generate_retrospective produces structured output
- to_json includes new fields
"""

import pytest
from ghost_agent.core.planning import TaskTree, TaskNode, TaskStatus


@pytest.fixture
def tree():
    t = TaskTree()
    root = t.add_task("Root task")
    child1 = t.add_task("Step 1", parent_id=root)
    child2 = t.add_task("Step 2", parent_id=root)
    return t, root, child1, child2


class TestTaskNodeExtensions:
    def test_failure_reason_default(self):
        node = TaskNode(id="1", description="test")
        assert node.failure_reason == ""
        assert not hasattr(node, "revision_count")
        assert node.actual_tool_used is None

    def test_update_status_with_failure_reason(self, tree):
        t, root, child1, child2 = tree
        t.update_status(child1, TaskStatus.FAILED, failure_reason="timeout error")
        assert t.nodes[child1].failure_reason == "timeout error"

    def test_update_status_with_actual_tool(self, tree):
        t, root, child1, child2 = tree
        t.update_status(child1, TaskStatus.DONE, actual_tool="execute")
        assert t.nodes[child1].actual_tool_used == "execute"


class TestGetFailedTasks:
    def test_returns_failed_tasks(self, tree):
        t, root, child1, child2 = tree
        t.update_status(child1, TaskStatus.FAILED, failure_reason="error A")
        t.update_status(child2, TaskStatus.DONE)

        failed = t.get_failed_tasks()
        assert len(failed) == 1
        assert failed[0].id == child1

    def test_returns_empty_when_all_done(self, tree):
        t, root, child1, child2 = tree
        t.update_status(child1, TaskStatus.DONE)
        t.update_status(child2, TaskStatus.DONE)

        assert len(t.get_failed_tasks()) == 0


class TestRetrospective:
    def test_successful_retrospective(self, tree):
        t, root, child1, child2 = tree
        t.update_status(child1, TaskStatus.DONE, actual_tool="execute")
        t.update_status(child2, TaskStatus.DONE, actual_tool="file_system")

        retro = t.generate_retrospective()
        assert retro["completed"] == 3  # root auto-completes
        assert retro["failed"] == 0
        assert retro["success_rate"] == 1.0
        assert "successfully" in retro["summary"]

    def test_partial_failure_retrospective(self, tree):
        t, root, child1, child2 = tree
        t.update_status(child1, TaskStatus.DONE)
        t.update_status(child2, TaskStatus.FAILED, failure_reason="tool not found")

        retro = t.generate_retrospective()
        assert retro["failed"] >= 1
        assert len(retro["what_failed"]) >= 1
        assert retro["what_failed"][0]["reason"] == "tool not found"

    def test_no_revision_keys_in_retrospective(self, tree):
        # §4PV: the revision fields were always 0/empty once their only
        # writer went (§4PT), so the retrospective no longer carries them.
        t, root, child1, child2 = tree
        t.update_status(child1, TaskStatus.DONE)
        t.update_status(child2, TaskStatus.FAILED, failure_reason="boom")

        retro = t.generate_retrospective()
        assert "revised" not in retro and "revision_history" not in retro
        assert retro["what_failed"] and all("revisions" not in f for f in retro["what_failed"])
        assert retro["what_failed"][0]["reason"] == "boom"

    def test_empty_tree_retrospective(self):
        t = TaskTree()
        retro = t.generate_retrospective()
        assert retro["summary"] == "No plan executed."


class TestToJsonExtensions:
    def test_includes_failure_reason(self, tree):
        t, root, child1, child2 = tree
        t.update_status(child1, TaskStatus.FAILED, failure_reason="connection error")

        data = t.to_json()
        child_data = data["children"][0]
        assert child_data.get("failure_reason") == "connection error"

    def test_never_includes_revision_count(self, tree):
        t, root, child1, child2 = tree
        t.update_status(child1, TaskStatus.FAILED, failure_reason="x")

        data = t.to_json()
        assert all("revision_count" not in c for c in data["children"])

    def test_omits_empty_fields(self, tree):
        t, root, child1, child2 = tree
        data = t.to_json()
        # Default values should not appear
        assert "failure_reason" not in data
        assert "revision_count" not in data


class TestStoreRevisionColumnLegacy:
    """§4PV: the tasks.revision_count column stays (old databases have it)
    but nothing reads or writes it."""

    def test_update_task_refuses_revision_count(self, tmp_path):
        from ghost_agent.memory.projects import ProjectStore
        store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
        pid = store.create_project("Legacy column")
        tid = store.add_task(pid, "step")
        with pytest.raises(ValueError):
            store.update_task(tid, revision_count=3)

    def test_old_nonzero_row_hydrates_and_roundtrips(self, tmp_path):
        import sqlite3
        from ghost_agent.core.planning import ProjectPlan
        from ghost_agent.memory.projects import ProjectStore
        store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
        pid = store.create_project("Legacy column")
        tid = store.add_task(pid, "step")
        db = next((tmp_path / "m").rglob("projects.db"))
        with sqlite3.connect(db) as c:
            c.execute("UPDATE tasks SET revision_count = 4 WHERE id = ?", (tid,))
        plan = ProjectPlan(store, pid)
        node = plan.tree.nodes[tid]
        assert not hasattr(node, "revision_count")
        plan.update_status(tid, TaskStatus.FAILED, failure_reason="nope")
        row = next(r for r in store.list_tasks(pid) if r["id"] == tid)
        assert row["status"] == "FAILED" and row["failure_reason"] == "nope"
        assert "revision_count" not in str(plan.tree.to_json())
