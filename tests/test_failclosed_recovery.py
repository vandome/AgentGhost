"""A fail-closed store must be able to come BACK (§4FP, 2026-09-09).

Item 4 of the 2026-09-08 list: audit every "arm after N failures" guard for
a DISARM path. The sticky no-think policy was the second instance; this is
the third and largest.

Four memory stores refuse to write after a read of their own file fails
with an OSError — the file is PRESENT but unreadable (EIO, EACCES,
ENFILE/EMFILE), so overwriting it with whatever is in memory would destroy
it. The guard is right and was measured: without it `adaptive_threshold`'s
next `record()` atomically overwrote the whole learned window.

The way back existed in two of them and not in the other two.
`contradiction_log` and `profile` re-read on every operation, so a
successful read clears the flag by itself. `adaptive_threshold` and
`competence` read only in `__init__` — so one transient
file-descriptor exhaustion stopped that store learning for the LIFE OF THE
PROCESS, which for this agent is days, announced by one log line. Their own
comments cite each other as sharing the discipline; two of the four had
quietly drifted out of it.

The last test enumerates the class from the AST, because fixing this shape
site-by-site is what let them drift.
"""
import ast
import json
import logging
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from ghost_agent.memory import failclosed
from ghost_agent.memory.adaptive_threshold import AdaptiveThreshold
from ghost_agent.memory.failclosed import FailClosedStore


@pytest.fixture(autouse=True)
def _instant_retries(monkeypatch):
    """The cadence is pinned by its own test; everywhere else it only
    slows the suite down."""
    monkeypatch.setattr(failclosed, "RETRY_EVERY_S", 0.0)


def _sick(name):
    """`Path.read_text` that fails for ONE file, as a real EIO does."""
    real = Path.read_text

    def reader(self, *a, **k):
        if self.name == name:
            raise OSError(5, "Input/output error")
        return real(self, *a, **k)
    return reader


# --- adaptive_threshold --------------------------------------------------

def test_the_threshold_store_recovers_and_loses_no_observation(tmp_path):
    """THE REGRESSION. World where it fails: `_degraded` is cleared only in
    `__init__`, so this store never records again until a restart."""
    f = tmp_path / "adaptive_threshold.json"
    f.write_text(json.dumps({"threshold": 0.55,
                             "window": [[0.9, True, True], [0.8, True, True]]}))
    on_disk = f.read_text()

    with patch.object(Path, "read_text", _sick(f.name)):
        at = AdaptiveThreshold(tmp_path, initial=0.7)
        assert at._degraded is True
        assert list(at.window) == []          # the history could not be read
        at.record(0.95, True)
        at.record(0.4, False)
        assert len(at.window) == 2            # …but the new ones are held
    assert f.read_text() == on_disk, "the intact file was overwritten"

    at.record(0.85, True)                     # the save that retries the read
    assert at._degraded is False
    saved = json.loads(f.read_text())
    assert len(saved["window"]) == 5, saved   # 2 from disk + 3 recorded blind
    assert saved["window"][0] == [0.9, True, True], "history came back first"
    assert saved["window"][-1] == [0.85, True, True], "blind records replayed"


def test_a_still_unreadable_file_leaves_the_store_exactly_as_it_was(tmp_path):
    """A failed retry must cost nothing: still closed, and the observations
    held in memory are still held."""
    f = tmp_path / "adaptive_threshold.json"
    f.write_text(json.dumps({"threshold": 0.55, "window": []}))
    with patch.object(Path, "read_text", _sick(f.name)):
        at = AdaptiveThreshold(tmp_path, initial=0.7)
        at.record(0.95, True)
        at.record(0.91, True)
        before = list(at.window)
        at.record(0.92, True)                 # retries, fails, keeps going
        assert at._degraded is True
        assert list(at.window)[:len(before)] == before
        assert len(at.window) == 3


# --- competence ----------------------------------------------------------

# --- the policy ----------------------------------------------------------

def test_the_first_save_after_arming_retries_immediately(tmp_path, monkeypatch):
    """A blip that lasts one write should cost one write: the retry clock
    starts EXPIRED, so the very next save re-reads. World where it fails:
    `_fc_last_retry` is set to `now` when arming, and every store waits a
    full period before its first attempt."""
    monkeypatch.setattr(failclosed, "RETRY_EVERY_S", 30.0)
    f = tmp_path / "adaptive_threshold.json"
    f.write_text(json.dumps({"threshold": 0.5, "window": []}))
    with patch.object(Path, "read_text", _sick(f.name)):
        at = AdaptiveThreshold(tmp_path, initial=0.7)
    assert at._fc_last_retry == 0.0
    at.record(0.9, True)
    assert at._degraded is False, "the first save did not retry the read"


def test_retries_are_rate_limited_while_the_disk_stays_sick(tmp_path, monkeypatch):
    """A store that saves on every observation must not stat a sick disk
    thousands of times a minute."""
    monkeypatch.setattr(failclosed, "RETRY_EVERY_S", 30.0)
    f = tmp_path / "adaptive_threshold.json"
    f.write_text(json.dumps({"threshold": 0.5, "window": []}))
    reads = {"n": 0}
    real = Path.read_text

    def counting(self, *a, **k):
        if self.name == f.name:
            reads["n"] += 1
            raise OSError(5, "Input/output error")
        return real(self, *a, **k)

    with patch.object(Path, "read_text", counting):
        at = AdaptiveThreshold(tmp_path, initial=0.7)
        after_init = reads["n"]
        for _ in range(20):
            at.record(0.9, True)
        # the first save retries (clock starts expired); the other 19 are
        # inside the window and must not touch the disk
        assert reads["n"] - after_init == 1, reads


def test_time_alone_never_clears_the_flag(tmp_path, monkeypatch):
    """Recovery is evidence, not patience: the flag clears because a read
    SUCCEEDED. World where it fails: the retry marks itself recovered when
    the period elapses."""
    monkeypatch.setattr(failclosed, "RETRY_EVERY_S", 0.0)
    f = tmp_path / "adaptive_threshold.json"
    f.write_text(json.dumps({"threshold": 0.5, "window": []}))
    with patch.object(Path, "read_text", _sick(f.name)):
        at = AdaptiveThreshold(tmp_path, initial=0.7)
        for _ in range(5):
            at.record(0.9, True)
            assert at._degraded is True


def test_one_error_per_arming_however_many_retries(tmp_path, caplog, monkeypatch):
    """One sick disk must not become thousands of identical ERROR lines —
    the retry clears the flag so the reload can re-arm it, and a naive
    re-arm logs every time."""
    monkeypatch.setattr(failclosed, "RETRY_EVERY_S", 0.0)
    f = tmp_path / "adaptive_threshold.json"
    f.write_text(json.dumps({"threshold": 0.5, "window": []}))
    with caplog.at_level(logging.ERROR, logger="GhostAgent"):
        with patch.object(Path, "read_text", _sick(f.name)):
            at = AdaptiveThreshold(tmp_path, initial=0.7)
            for _ in range(10):
                at.record(0.9, True)
    armed = [r for r in caplog.records if "REFUSING to overwrite" in r.getMessage()]
    assert len(armed) == 1, [r.getMessage()[:60] for r in armed]


def test_a_recovery_hook_that_raises_leaves_the_store_closed(tmp_path):
    """Recovery runs on the answer path; it must never take a turn down,
    and a hook that explodes must not be read as success."""
    class Boom(FailClosedStore):
        def _fc_reload_from_disk(self):
            raise RuntimeError("disk on fire")

    b = Boom()
    b._degraded = True
    b._fc_last_retry = 0.0
    assert b._fc_ready_to_write() is False
    assert b._degraded is True


def test_a_healthy_store_never_pays_for_the_guard():
    class Never(FailClosedStore):
        def _fc_reload_from_disk(self):
            raise AssertionError("must not be called when healthy")

    assert Never()._fc_ready_to_write() is True


# --- the CLASS, not the two sites ---------------------------------------

def _fail_closed_classes():
    """Every class in `memory/` that arms a fail-closed flag, with the
    methods that CLEAR it, read from the AST."""
    root = Path(__file__).resolve().parents[1] / "src" / "ghost_agent" / "memory"
    found = {}
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            arms, clears = [], []
            for fn in [n for n in ast.walk(cls)
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
                for node in ast.walk(fn):
                    if isinstance(node, ast.Assign):
                        for t in node.targets:
                            if (isinstance(t, ast.Attribute) and t.attr == "_degraded"
                                    and isinstance(node.value, ast.Constant)):
                                (arms if node.value.value is True else clears).append(fn.name)
                    if (isinstance(node, ast.Call)
                            and getattr(node.func, "attr", "") == "_fc_arm"):
                        arms.append(fn.name)
                    if (isinstance(node, ast.Call)
                            and getattr(node.func, "attr", "") == "_fc_ready_to_write"):
                        clears.append(fn.name)
            if arms:
                found[f"{path.name}::{cls.name}"] = {
                    "arms": sorted(set(arms)),
                    "clears_outside_init": sorted(
                        {c for c in clears if c != "__init__"}),
                    "bases": [getattr(b, "id", getattr(b, "attr", "")) for b in cls.bases],
                }
    return found


def test_every_fail_closed_store_can_be_disarmed_without_a_restart():
    """⚠ THE CLASS, ENUMERATED. A store that blocks its own writes after a
    read error and can only be unblocked by `__init__` is blocked until the
    process restarts — days, here.

    World where it fails: a new store copies the fail-closed guard from
    one of these and stops at the guard, which is exactly how two of the
    original four drifted out of the discipline the other two document.
    """
    stores = _fail_closed_classes()
    # the three known ones, at least (§4PJ: the competence profile, the
    # fourth, was removed)
    assert len(stores) >= 3, stores
    for name, info in stores.items():
        # a class that INHERITS the mixin but never calls `_fc_ready_to_write`
        # is blocked until restart all the same — inheritance is not a
        # disarm (review, 2026-09-09); the call counts as a clear above
        recovers = bool(info["clears_outside_init"])
        assert recovers, (
            f"{name} arms a fail-closed flag in {info['arms']} and can only "
            f"clear it in __init__ — it is blocked until the process restarts")


def test_the_store_that_already_recovered_still_does():
    """`profile` clears the flag on any successful read. It is the sibling
    implementation (§4NN: `contradiction_log` was removed), and the audit
    must not have quietly broken it."""
    stores = _fail_closed_classes()
    prof = next(v for k, v in stores.items() if k.startswith("profile"))
    assert "load_raw" in prof["clears_outside_init"], prof


# --- the other counter the audit found -----------------------------------

@pytest.mark.asyncio
async def test_a_duplicate_create_loop_counter_measures_a_burst_not_a_lifetime(tmp_path):
    """A "you are in a LOOP" counter kept on the project record with nothing
    that ever cleared it: three duplicate creates once, and every duplicate
    create that project ever saw again — months later — got the alarmed
    "STOP. This is retry #N" instruction and fired an operator WARNING.

    Driven through the REAL tool, not a re-implementation of its arithmetic.
    World where it fails: the count is read straight from metadata with no
    recency test, which is what shipped until §4FP.
    """
    import json as _json
    import time
    from types import SimpleNamespace

    from ghost_agent.memory.projects import ProjectStore
    from ghost_agent.memory.scratchpad import Scratchpad
    from ghost_agent.tools import projects as P
    from ghost_agent.tools.projects import tool_manage_projects

    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    ctx = SimpleNamespace(project_store=store,
                          scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
                          graph_memory=None, workspace_model=None,
                          current_project_id=None, last_user_content="")

    res = _json.loads(await tool_manage_projects(ctx, action="create", title="Widget"))
    pid = res["created"]

    # a burst: three duplicates in a row escalate
    for _ in range(3):
        await tool_manage_projects(ctx, action="create", title="Widget")
    meta = store.get_project(pid)["metadata"]
    assert meta["duplicate_create_retries"] == 3, meta
    # the burst stamped its own clock — nothing in this test wrote it
    assert meta.get("duplicate_create_last_at", 0) > 0, meta
    meta_before_age = float(meta["duplicate_create_last_at"])

    # …now age the last attempt past the loop window and try again: this is
    # a fresh mistake, not turn 4 of a loop.
    meta = dict(meta)
    meta["duplicate_create_last_at"] = (
        time.time() - P._DUPLICATE_CREATE_LOOP_WINDOW_SECONDS - 60)
    store.update_project(pid, metadata=meta)
    await tool_manage_projects(ctx, action="create", title="Widget")
    after = store.get_project(pid)["metadata"]
    assert after["duplicate_create_retries"] == 1, after
    # …and the attempt STAMPED itself, or the window has nothing to measure
    # from and the reset above can never fire in production (it only fired
    # here because the test wrote the field by hand).
    assert after["duplicate_create_last_at"] >= meta_before_age, after


def test_the_loop_window_is_read_at_the_call_site():
    """The reset must live where the counter is INCREMENTED, not in a helper
    nothing calls."""
    import inspect

    from ghost_agent.tools import projects as P
    src = inspect.getsource(P)
    i = src.index("duplicate_create_retries")
    around = src[max(0, i - 900):i + 900]
    assert "_DUPLICATE_CREATE_LOOP_WINDOW_SECONDS" in around
    assert "duplicate_create_last_at" in around
