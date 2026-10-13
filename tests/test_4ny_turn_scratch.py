"""§4NY (2026-10-12, operator): the agent cleans up the screenshots and
throwaway test scripts it made only to check its own work. Live: request
1c343374 left seven screenshots in sandbox/webos/ that its reply never named.
Every test names the world it fails in."""
import ast
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ghost_agent.tools import turn_scratch as TS


@pytest.fixture(autouse=True)
def _clean_notes():
    TS._NOTED.clear()
    yield
    TS._NOTED.clear()


@pytest.fixture
def dirs(tmp_path):
    sb, sysd = tmp_path / "sandbox", tmp_path / "system"
    sb.mkdir()
    sysd.mkdir()
    return sb, sysd


def _png(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x89PNG\r\n\x1a\nshot")
    return p


def _trash(sysd: Path):
    t = sysd / TS.TRASH_DIRNAME
    return sorted(str(p.relative_to(t)) for p in t.rglob("*") if p.is_file()) if t.exists() else []


def _end(rid, reply, user, sb, sysd):
    """A turn's end, then the idle sweep once the delay has passed."""
    out = TS.finish_turn(rid, reply, user, sb, sysd, now=1000.0)
    out["moved"] = TS.sweep_due(sb, sysd, now=1000.0 + TS.DELAY_S + 1)
    return out


def _shot(sb, rel, rid):
    p = sb / rel
    TS.note_created(p, "screenshot", req_id=rid)
    return _png(p)


# ── screenshots ──────────────────────────────────────────────────────────── #

def test_an_unnamed_check_screenshot_is_moved_to_the_host_trash(dirs):
    """Fails where the seven webos/*.png check screenshots stayed next to the
    deliverable (request 1c343374)."""
    sb, sysd = dirs
    shot = _shot(sb, "webos/calculator_result.png", "1c343374abcd")
    first = TS.finish_turn("1c343374abcd", "GhostOS is written — open webos/index.html", "write a webos",
                           sb, sysd, now=1000.0)
    # still there for the late verifier's visual check (review r2 MAJOR)
    assert first["queued"] == ["webos/calculator_result.png"] and shot.exists()
    assert TS.sweep_due(sb, sysd, now=1000.0 + 60) == [] and shot.exists()
    assert TS.sweep_due(sb, sysd, now=1000.0 + TS.DELAY_S + 1) == ["webos/calculator_result.png"]
    assert not shot.exists()
    assert _trash(sysd) == [f"{time.strftime('%Y%m%d')}/1c343374abcd/webos/calculator_result.png"]
    assert not (sb / ".trash").exists()                    # nothing of it in the sandbox


@pytest.mark.parametrize("reply,user", [
    ("Here it is: ![desktop](/api/download/webos/desktop.png)", "build it"),
    ("Done.", "save the result as webos/desktop.png"),
    ("See desktop.png for the result.", "build it"),
    ("Here: ![d](/api/download/webos/desk%74op.png)", "build it"),            # URL-encoded
    ("Here's what the page looks like.", "take a screenshot of the desktop"),  # asked for one
])
def test_a_screenshot_the_reply_or_the_user_wants_is_kept(dirs, reply, user):
    sb, sysd = dirs
    shot = _shot(sb, "webos/desktop.png", "r1")
    out = _end("r1", reply, user, sb, sysd)
    assert shot.exists() and out["kept"] == ["webos/desktop.png"] and _trash(sysd) == []


def test_a_file_that_existed_before_the_turn_is_never_noted(dirs):
    sb, sysd = dirs
    mine = _png(sb / "photo.png")
    TS.note_created(mine, "screenshot", req_id="r2")
    _end("r2", "", "", sb, sysd)
    assert mine.exists() and _trash(sysd) == []


def test_a_path_another_turn_writes_again_is_no_longer_this_turns(dirs):
    """Fails where turn A, still streaming, moved the screenshot turn B had
    just re-taken at the same path (review r1)."""
    sb, sysd = dirs
    shot = _shot(sb, "screenshot.png", "turn-a")
    TS.note_created(shot, "screenshot", req_id="turn-b")       # B overwrites it
    _end("turn-a", "", "", sb, sysd)
    assert shot.exists()


def test_another_requests_files_are_left_alone(dirs):
    sb, sysd = dirs
    a, b = _shot(sb, "a.png", "ra"), _shot(sb, "b.png", "rb")
    _end("ra", "", "", sb, sysd)
    assert not a.exists() and b.exists()


def test_a_symlink_or_a_path_outside_the_sandbox_is_never_moved(dirs, tmp_path):
    sb, sysd = dirs
    outside = _png(tmp_path / "home" / "keep.png")
    link = sb / "link.png"
    TS.note_created(link, "screenshot", req_id="r3")
    link.symlink_to(outside)
    TS.note_created(tmp_path / "home" / "new.png", "screenshot", req_id="r3")
    _png(tmp_path / "home" / "new.png")
    _end("r3", "", "", sb, sysd)
    assert link.is_symlink() and outside.exists() and (tmp_path / "home" / "new.png").exists()


def test_a_symlink_to_another_sandbox_file_does_not_move_that_file(dirs):
    """Fails where a noted path that became a link to the owner's file moved
    the owner's file (the move resolves the path)."""
    sb, sysd = dirs
    owner = _png(sb / "site" / "logo.png")
    link = sb / "shot.png"
    TS.note_created(link, "screenshot", req_id="r9")
    link.symlink_to(owner)
    _end("r9", "", "", sb, sysd)
    assert owner.exists() and link.is_symlink() and _trash(sysd) == []


def test_a_client_request_id_cannot_steer_the_trash_path(dirs, tmp_path):
    """Fails where X-Request-ID '../../../outside' moved a file out of the
    trash (review r1 MAJOR)."""
    sb, sysd = dirs
    _shot(sb, "a.png", "../../../outside")
    _end("../../../outside", "", "", sb, sysd)
    assert _trash(sysd) == [f"{time.strftime('%Y%m%d')}/outside/a.png"]
    assert not (tmp_path / "outside").exists()


def test_a_trash_symlink_planted_in_the_sandbox_is_irrelevant(dirs, tmp_path):
    """Fails where a `.trash` symlink in the sandbox turned the purge into an
    rmtree of a host folder (review r1 CRITICAL): the trash is host-side."""
    sb, sysd = dirs
    host = tmp_path / "photos" / "20250101"
    host.mkdir(parents=True)
    (host / "a.jpg").write_text("x")
    (sb / ".trash").symlink_to(tmp_path / "photos")
    TS.purge_old_trash(sysd / TS.TRASH_DIRNAME)
    TS.sweep_due(sb, sysd)
    assert (host / "a.jpg").exists()


def test_trash_days_past_a_week_are_purged_and_a_symlinked_day_is_not_followed(dirs, tmp_path):
    sb, sysd = dirs
    trash = sysd / TS.TRASH_DIRNAME
    old = _png(trash / "20260901" / "r" / "a.png")
    new = _png(trash / time.strftime("%Y%m%d") / "r" / "b.png")
    elsewhere = tmp_path / "keepme"
    elsewhere.mkdir()
    (elsewhere / "x").write_text("x")
    (trash / "20250101").symlink_to(elsewhere)
    TS.purge_old_trash(trash)
    assert not old.exists() and new.exists() and (elsewhere / "x").exists()


# ── throwaway test scripts ───────────────────────────────────────────────── #

@pytest.mark.parametrize("name,scratch", [
    ("test_calc.py", True), ("debug_layout.js", True), ("calc_test.py", True), ("test_page.html", True),
    ("calc.py", False), ("index.html", False), ("testimonials.html", False), ("contest.py", False),
    ("check_site.py", False), ("temp_converter.py", False), ("verify_backup.sh", False),
    ("test_data.json", False), ("scratch_notes.txt", False), ("probe_ports.py", False),
])
def test_only_throwaway_script_names_count(name, scratch):
    """Fails where a deliverable like check_site.py or test_data.json was
    taken for a throwaway check (review r1 MAJOR)."""
    assert TS.is_scratch_test_name(name) is scratch


def _script(sb, rel, rid, body="assert 1"):
    t = sb / rel
    TS.note_created(t, "test", req_id=rid)
    t.parent.mkdir(parents=True, exist_ok=True)
    t.write_text(body)
    return t


def test_a_test_script_waits_for_the_late_verifier_then_moves(dirs):
    """Fails where a test file the late verifier still reads (FILE-ARTIFACT,
    the audit's source read) was moved at turn end — or never moved."""
    sb, sysd = dirs
    t = _script(sb, "webos/test_calc.py", "r4")
    now = 1_000_000.0
    out = TS.finish_turn("r4", "done", "make the calculator work", sb, sysd, now=now)
    assert out["queued"] == ["webos/test_calc.py"] and t.exists()
    assert TS.sweep_due(sb, sysd, now=now + 60) == [] and t.exists()          # verifier window
    assert TS.sweep_due(sb, sysd, now=now + TS.TEST_DELAY_S + 1) == ["webos/test_calc.py"]
    assert not t.exists() and json.loads((sysd / TS.LEDGER_NAME).read_text()) == []


def test_a_later_turn_that_names_a_queued_script_keeps_it(dirs):
    """Fails where "keep test_api.py and run it hourly" ten minutes later did
    not stop the deferred move (review r1 MAJOR)."""
    sb, sysd = dirs
    t = _script(sb, "test_api.py", "r5")
    TS.finish_turn("r5", "", "", sb, sysd, now=100.0)
    TS.finish_turn("r6", "Scheduled.", "keep test_api.py and run it every hour", sb, sysd, now=700.0)
    assert TS.sweep_due(sb, sysd, now=100.0 + TS.TEST_DELAY_S + 1) == [] and t.exists()


def test_a_test_script_used_again_after_the_turn_is_kept(dirs):
    sb, sysd = dirs
    t = _script(sb, "test_x.py", "r7")
    TS.finish_turn("r7", "", "", sb, sysd, now=100.0)
    os.utime(t, (time.time() + 50, time.time() + 50))           # a later turn edited it
    assert TS.sweep_due(sb, sysd, now=100.0 + TS.TEST_DELAY_S + 1) == [] and t.exists()


@pytest.mark.parametrize("rel,user", [("test_api.py", "write unit tests for the api"),
                                      ("projects/abc123/test_app.py", "fix the bug")])
def test_asked_for_tests_and_project_tests_are_deliverables(dirs, rel, user):
    sb, sysd = dirs
    _script(sb, rel, "r8")
    out = TS.finish_turn("r8", "Done.", user, sb, sysd)
    assert out["kept"] == [rel] and out["queued"] == []


def test_a_forged_ledger_row_cannot_move_a_file_outside_the_sandbox(dirs, tmp_path):
    sb, sysd = dirs
    victim = tmp_path / "home" / "notes.txt"
    victim.parent.mkdir()
    victim.write_text("x")
    (sysd / TS.LEDGER_NAME).write_text(json.dumps([{"path": str(victim), "req": "r", "due": 0,
                                                    "mtime": victim.stat().st_mtime}]))
    assert TS.sweep_due(sb, sysd, now=10.0) == [] and victim.exists()


def test_notes_of_turns_that_never_finished_are_bounded(tmp_path):
    for i in range(TS._MAX_OPEN_TURNS + 10):
        TS.note_created(tmp_path / f"{i}.png", "screenshot", req_id=f"r{i}")
    assert len(TS._NOTED) == TS._MAX_OPEN_TURNS and "r0" not in TS._NOTED


# ── the wiring ───────────────────────────────────────────────────────────── #

def _stub(payload):
    stub = MagicMock()

    def _execute(cmd, timeout=300, **kw):
        return (f"[BROWSER_OK] {json.dumps(payload)}\n", 0)
    stub.execute = _execute
    return stub


async def test_the_browser_notes_a_new_screenshot_and_says_it_is_a_working_file(tmp_path):
    from ghost_agent.tools import browser as B
    from ghost_agent.utils.logging import request_id_context
    tok = request_id_context.set("req-shot-1")
    try:
        res = await B.tool_browser(sandbox_dir=tmp_path, sandbox_manager=_stub(
            {"url": "file:///workspace/a.html", "title": "t", "status": 200, "path": "/workspace/s.png"}),
            operation="screenshot", url="file:///workspace/a.html", out_path="webos/check.png")
    finally:
        request_id_context.reset(tok)
    assert str(tmp_path / "webos" / "check.png") in TS._NOTED.get("req-shot-1", {})
    assert "removed from the workspace" in str(res)


async def test_a_new_test_script_written_with_file_system_is_noted(tmp_path):
    from ghost_agent.tools.file_system import tool_write_file
    from ghost_agent.utils.logging import request_id_context
    tok = request_id_context.set("req-test-1")
    try:
        await tool_write_file("test_calc.py", "assert 1\n", tmp_path)
        await tool_write_file("calc.py", "x = 1\n", tmp_path)
    finally:
        request_id_context.reset(tok)
    assert TS._NOTED.get("req-test-1") == {str(tmp_path / "test_calc.py"): "test"}


def _agent(A, sb, sysd):
    agent = A.GhostAgent.__new__(A.GhostAgent)
    agent.context = SimpleNamespace(sandbox_dir=sb, memory_dir=sysd / "memory")
    return agent


def test_the_agent_finishes_the_turn_scratch_and_logs_it(dirs, monkeypatch):
    import ghost_agent.core.agent as A
    sb, sysd = dirs
    shot = _shot(sb, "webos/screenshot2.png", "req-a")
    calls = []
    monkeypatch.setattr(A, "pretty_log", lambda *a, **k: calls.append((a, k)))
    _agent(A, sb, sysd)._finish_turn_scratch("req-a", "GhostOS is ready.", "write a webos")
    assert shot.exists()                                   # the late verifier may still look at it
    assert calls and calls[0][0][0] == "Workspace Cleanup" and "webos/screenshot2.png" in calls[0][0][1]
    assert "in 30 min" in calls[0][0][1]
    assert TS.sweep_due(sb, sysd, now=time.time() + TS.DELAY_S + 1) == ["webos/screenshot2.png"]


@pytest.mark.parametrize("rid,moved", [("a1b2c3d4e5f60708", True), ("probe-4ny-x", True),
                                       ("sub-leaf-1", False), ("sched-daily", False), ("job-abc", False)])
def test_only_interactive_turns_clean_up(dirs, monkeypatch, rid, moved):
    """Fails where a coding leaf's (sub-) or a job's check files were moved
    while later steps still needed them."""
    import ghost_agent.core.agent as A
    sb, sysd = dirs
    shot = _shot(sb, "s.png", rid)
    monkeypatch.setattr(A, "pretty_log", lambda *a, **k: None)
    _agent(A, sb, sysd)._finish_turn_scratch(rid, "", "")
    TS.sweep_due(sb, sysd, now=time.time() + TS.DELAY_S + 1)
    assert shot.exists() is (not moved)
    assert rid not in TS._NOTED                       # the notes are dropped either way


def test_both_turn_endings_and_the_idle_tick_run_the_cleanup():
    """Enumerated over the parsed source: the streamed and non-streamed
    finalize each call _finish_turn_scratch right after their Turn Outcome
    line (the streamed one with what the user saw live, full_content), and
    the idle tick runs the deferred sweep."""
    import ghost_agent.core.agent as A
    tree = ast.parse(Path(A.__file__).read_text(encoding="utf-8"))
    finish, sweep, streamed_full = 0, 0, False
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for i, st in enumerate(body[:-1]):
            if (isinstance(st, ast.Expr) and isinstance(st.value, ast.Call)
                    and getattr(st.value.func, "attr", "") == "_emit_turn_outcome_line"):
                nxt = body[i + 1]
                if (isinstance(nxt, ast.Expr) and isinstance(nxt.value, ast.Call)
                        and getattr(nxt.value.func, "attr", "") == "_finish_turn_scratch"):
                    finish += 1
                    names = {n.id for n in ast.walk(nxt.value.args[1]) if isinstance(n, ast.Name)}
                    streamed_full = streamed_full or "full_content" in names
    for c in ast.walk(tree):
        if (isinstance(c, ast.Call) and getattr(c.func, "id", "") == "_sweep" and len(c.args) == 2
                and isinstance(c.args[1], ast.Call)
                and getattr(c.args[1].func, "attr", "") == "_turn_scratch_system_dir"):
            sweep += 1
    assert finish == 2 and sweep == 1 and streamed_full


@pytest.mark.parametrize("rid,origin", [("sub-leaf-9", "probe"), ("SYSTEM", "")])
def test_a_sub_agent_under_a_probe_and_internal_runs_never_clean_up(dirs, monkeypatch, rid, origin):
    """Fails where a sub-agent spawned in a probe turn (request_kind reads
    the inherited probe origin) moved the screenshots it made for its
    parent (review r2), or an internal run cleaned up."""
    import ghost_agent.core.agent as A
    from ghost_agent.utils.logging import request_origin_context
    sb, sysd = dirs
    shot = _shot(sb, "s.png", rid)
    monkeypatch.setattr(A, "pretty_log", lambda *a, **k: None)
    tok = request_origin_context.set(origin)
    try:
        _agent(A, sb, sysd)._finish_turn_scratch(rid, "", "")
    finally:
        request_origin_context.reset(tok)
    TS.sweep_due(sb, sysd, now=time.time() + TS.DELAY_S + 1)
    assert shot.exists()


def test_when_the_reply_shows_the_asked_for_screenshot_the_failed_tries_go(dirs):
    """Fails where "take a screenshot" kept the failed first try next to the
    one the reply linked (probe-4ny)."""
    sb, sysd = dirs
    first, second = _shot(sb, "p/screenshot.png", "r10"), _shot(sb, "p/screenshot2.png", "r10")
    out = _end("r10", "Here: ![page](/api/download/p/screenshot2.png)", "take a screenshot of the page", sb, sysd)
    assert second.exists() and not first.exists() and out["moved"] == ["p/screenshot.png"]


def test_a_turn_that_re_takes_another_turns_check_file_owns_it(dirs):
    """Fails where turn B re-took turn A's screenshot.png within the window:
    A dropped it, B never noted it (it existed), and it stayed forever."""
    sb, sysd = dirs
    shot = _shot(sb, "screenshot.png", "turn-a")
    TS.note_created(shot, "screenshot", req_id="turn-b")      # B writes it again
    TS.finish_turn("turn-a", "", "", sb, sysd, now=1000.0)
    out = _end("turn-b", "", "", sb, sysd)
    assert out["moved"] == ["screenshot.png"] and not shot.exists()
