"""§4E Tier 3 — the closing-turn stamp and the `task_reopened` event.

The store stamps the closing turn's req_id at close time
(`tasks.closed_req_id`) and logs a `task_reopened` event carrying it on the
task-level DONE -> open transition.

§4PJ (2026-10-16): the `on_task_reopened` hook and the calibration
retro-negative it fed were removed with the calibration corpus. The stamp and
the event stay as project history, so these pins read the EVENT LOG.
"""
import json

import pytest

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.utils.logging import request_id_context


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path)


def _project_with_task(store, *, extra_open_task=False):
    pid = store.create_project("Test project", goal="test")
    tid = store.add_task(pid, "the work item")
    other = store.add_task(pid, "still open") if extra_open_task else None
    return pid, tid, other


def _close_as(store, tid, req_id):
    token = request_id_context.set(req_id)
    try:
        assert store.update_task(tid, status="DONE")
    finally:
        request_id_context.reset(token)


def _reopen_events(store, pid):
    out = []
    for e in store.list_events(pid, event_type="task_reopened"):
        payload = e["payload"]
        out.append(json.loads(payload) if isinstance(payload, str) else payload)
    return out


class TestClosingStamp:
    def test_close_stamps_the_closing_turns_req_id(self, store):
        _, tid, _ = _project_with_task(store)
        _close_as(store, tid, "abcd1234")
        assert store.get_task(tid)["closed_req_id"] == "abcd1234"

    def test_system_context_is_not_stamped(self, store):
        # Boot reaper / maintenance closes carry no turn to blame.
        _, tid, _ = _project_with_task(store)
        assert store.update_task(tid, status="DONE")
        assert store.get_task(tid)["closed_req_id"] == ""

    def test_reopen_consumes_the_stamp(self, store):
        # The event carries the closing req_id, and the column is cleared in
        # the same transition, so a later re-close re-stamps fresh (a stale
        # stamp would attribute a SYSTEM re-close to a turn that didn't
        # perform it).
        pid, tid, _ = _project_with_task(store)
        _close_as(store, tid, "abcd1234")
        store.update_task(tid, status="PENDING")
        assert [e["closed_req_id"] for e in _reopen_events(store, pid)] == ["abcd1234"]
        assert store.get_task(tid)["closed_req_id"] == ""

    def test_system_reclose_blanks_rather_than_inherits(self, store):
        # close(A) -> reopen -> re-close under SYSTEM: the stamp must be
        # blank, and the second reopen must carry NO closing turn.
        pid, tid, _ = _project_with_task(store)
        _close_as(store, tid, "abcd1234")
        store.update_task(tid, status="PENDING")
        assert store.update_task(tid, status="DONE")  # SYSTEM context
        assert store.get_task(tid)["closed_req_id"] == ""
        store.update_task(tid, status="PENDING")
        # the event log reads newest first
        assert [e["closed_req_id"] for e in _reopen_events(store, pid)] == ["", "abcd1234"]

    def test_stamp_survives_unrelated_updates(self, store):
        _, tid, _ = _project_with_task(store)
        _close_as(store, tid, "abcd1234")
        token = request_id_context.set("ffff9999")
        try:
            store.update_task(tid, description="edited later")
            # Re-asserting DONE from another turn must not steal the close.
            store.update_task(tid, status="DONE")
        finally:
            request_id_context.reset(token)
        assert store.get_task(tid)["closed_req_id"] == "abcd1234"


class TestReopenEvent:
    def test_reopen_logs_task_reopened_event(self, store):
        pid, tid, _ = _project_with_task(store)
        _close_as(store, tid, "abcd1234")
        store.update_task(tid, status="PENDING")
        events = _reopen_events(store, pid)
        assert len(events) == 1
        assert events[0]["from_status"] == "DONE"
        assert events[0]["closed_req_id"] == "abcd1234"

    def test_reopen_on_active_project_still_logs(self, store):
        # A task revived on a project that never left ACTIVE (another task
        # still open): the TASK-level transition is the trigger, not
        # project_reopened.
        pid, tid, _ = _project_with_task(store, extra_open_task=True)
        _close_as(store, tid, "abcd1234")
        assert store.get_project(pid)["status"] == "ACTIVE"
        assert store.update_task(tid, status="IN_PROGRESS")
        assert len(_reopen_events(store, pid)) == 1

    def test_non_done_transitions_log_nothing(self, store):
        pid, tid, _ = _project_with_task(store)
        store.update_task(tid, status="IN_PROGRESS")
        store.update_task(tid, status="PENDING")
        assert _reopen_events(store, pid) == []

    def test_the_hook_is_gone(self, store):
        """§4PJ: nothing may be wired to fire on a reopen any more — a
        restored hook would be a write path with no reader."""
        assert not hasattr(store, "on_task_reopened")
        assert not hasattr(ProjectStore, "_fire_task_reopened")
