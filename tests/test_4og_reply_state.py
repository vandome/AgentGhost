"""§4OG-4 — the reply and the state across turns: narration beside every tool
kind is dropped, the non-stream landed check sees only the report turn's
rows, STATE OF WORK is cleared only once shown, the streamed drain reads a
snapshot of the forced-final flags, the uniquified request id reaches every
contextvar reader, and the task-state events leave a durable log line."""
import ast

from ghost_agent.core import agent as ag
from tests.pin_ast import tree, U, calls, call_kw, assigns, assigned_values, ifs, getattr_consts, first_line
from ghost_agent.core.agent import _narration_batch, _all_lookup_calls, StreamState
from ghost_agent.core.reply_smoothing import drop_pre_tool_segments


def _tc(name):
    return {"id": "x", "function": {"name": name, "arguments": "{}"}}


def test_narration_beside_any_tool_kind_is_recorded_except_a_memory_write():
    """Fails where only lookup batches were recorded: the coding reply opened
    with five superseded status notes written beside file_system / execute /
    manage_projects calls (§4OF defect 5)."""
    T = tree(ag)
    assert _narration_batch([_tc("file_system"), _tc("execute")])
    assert _narration_batch([_tc("manage_projects")])
    assert _narration_batch([_tc("web_search")]) and _all_lookup_calls([_tc("web_search")])
    assert not _narration_batch([_tc("remember")])                      # "Saved that." IS the answer
    assert not _narration_batch([_tc("update_profile"), _tc("remember")])
    assert _narration_batch([_tc("remember"), _tc("file_system")])      # a write beside a memory call: narration
    assert not _narration_batch([])
    # beside a WORK batch the segment is a WorkSegment: dropped only when the rest outweighs it twice over (reader 3 M2)
    from ghost_agent.core.reply_smoothing import WorkSegment
    answer = "The capital of the dataset is 42 rows per day; the median latency is 118 ms and the p99 is 410 ms across " \
             "the three regions, with eu-west the slowest because of the Tor hop on every call to the sandbox."
    tail = "I closed the task and updated the ledger with the measurement; let me know if you want the per-region table " \
           "exported as CSV or a chart added to the project page as well, or the raw samples attached to the work log."
    reply = answer + "\n\n" + tail
    assert drop_pre_tool_segments(reply, [WorkSegment(answer)]) == reply              # the answer stays
    assert drop_pre_tool_segments(reply, [answer]) != reply                            # a LOOKUP segment of the same text would go (§4LH rule)
    notes = ["4 remaining failures. Two issues: CLI persistence.", "Two remaining fixes."]
    long_final = "All 76 tests pass. Here is the summary:\n\n" + "\n".join(f"| module{i} | does thing {i} in the ledger package |" for i in range(8))
    out = drop_pre_tool_segments("\n\n".join(notes) + "\n\n" + long_final, [WorkSegment(n) for n in notes])
    assert out == long_final                                                           # stale notes before a long answer go
    # the §4OG coding probe's shape: a 467-char fix list (over the 400 lookup cap) before a 2,300-char answer
    fixes = "14 failures to fix. Let me address them all in parallel:\n\n" + "\n".join(
        f"{i}. **item {i}** - something is broken in module {i} and needs the parser rewritten" for i in range(1, 7))
    assert 400 < len(fixes) < 1000
    big = "All done. Here is the summary:\n\n" + "\n".join(f"| module{i} | does thing {i} in the ledger package, with a longer description |" for i in range(30))
    assert drop_pre_tool_segments(fixes + "\n\n" + big, [WorkSegment(fixes)]) == big
    # the same list before a SHORT answer stays (the rest must be 1.5× the segment)
    small = "All done; 96 tests pass. The package is in probe/ledger with the CLI at ledger.py and the SQLite file beside it, " \
            "and the trial balance printed below balances at 6000.00 on both sides as the request asked for."
    assert drop_pre_tool_segments(fixes + "\n\n" + small, [WorkSegment(fixes)]) == fixes + "\n\n" + small
    app = [c for c in calls(T, "append") if U(c.func) == "_pts.append"]
    assert app and U(app[0].args[0]) == "ui_content if _all_lookup_calls(tool_calls) else _WorkSeg(ui_content)"
    assert ifs(T, "tool_calls and _narration_batch(tool_calls)") and not ifs(T, "tool_calls and _all_lookup_calls(tool_calls)")
    # the drop itself (unchanged): whole stale paragraphs go, the answer stays
    reply = ("4 remaining failures. Two issues: CLI persistence.\n\nTwo remaining fixes.\n\nAll 76 tests pass. Here is the summary:\n\n"
             + "\n".join(f"| module{i} | does the thing number {i} in the ledger package |" for i in range(8)) + "\n\nDone.")
    segs = ["4 remaining failures. Two issues: CLI persistence.", "Two remaining fixes."]
    out = drop_pre_tool_segments(reply, segs)
    assert out.startswith("All 76 tests pass") and "remaining" not in out


def test_the_non_stream_landed_check_sees_only_the_report_turns_rows():
    """Reader C MAJOR 1: the whole request ledger made a FAILED rewrite of an
    earlier draft look landed."""
    t = tree(ag)
    vals = assigned_values(t, "self.context._report_write_rows_at")
    assert "len(tools_run_this_turn or [])" in vals and "0" in vals            # recorded at the allowance; reset per request
    landed = calls(t, "_report_writes_landed")
    assert any(U(c.args[1]) == "(tools_run_this_turn or [])[_rw_at:]" for c in landed if len(c.args) > 1)
    assert "int(getattr(self.context, '_report_write_rows_at', 0) or 0)" in assigned_values(t, "_rw_at")


def test_state_of_work_is_cleared_only_after_it_was_shown():
    """Reader C MAJOR 3: a queued Stop or an unrelated quick turn destroyed
    the note unseen."""
    t = tree(ag)
    shown = ifs(t, "_sow_sid and getattr(self.context, '_sow_shown_sid', '') == _sow_sid")
    assert shown and any(calls(i, "_sow_clear") for i in shown)
    assert not any(calls(i, "_sow_clear") for i in ifs(t, "_sow_sid")), "cleared regardless of being shown"
    vals = assigned_values(t, "self.context._sow_shown_sid")
    assert "_sow_sid" in vals and vals.count("''") == 2                      # set when shown; reset per request AND at request end (reader 3 m1)


def test_the_streamed_drain_reads_a_snapshot_of_the_forced_final_flags():
    """Reader C MAJOR 4: the drain read `_breaker_forced_final` and
    `_report_write_used` off the shared context after the lock was released."""
    fields = {f.name for f in StreamState.__dataclass_fields__.values()}
    assert {"ff_breaker", "ff_report_write_used", "ff_breaker_reason"} <= fields
    d = tree(ag.GhostAgent._stream_final_generation)
    assert "ff_breaker is True" in assigned_values(d, "_ff_breaker")
    assert not ({"_breaker_forced_final", "_report_write_used", "_breaker_reason"} & getattr_consts(d))
    rw = [c for c in calls(d, "_run_streamed_report_writes")]
    assert rw and U(call_kw(rw[0], "why")) == "str(ff_breaker_reason or '')"   # reader 3 M4: the helper takes the snapshot's reason
    h = tree(ag.GhostAgent._run_streamed_report_writes)
    assert not assigns(h, "self.context._report_write_used")                   # no shared-flag write from the drain
    assert "str(why or '') or 'it ran out of its budget'" in assigned_values(h, "_why")
    ctor = tree(ag)
    ss = [c for c in calls(ctor, "StreamState")]
    assert ss and U(call_kw(ss[0], "ff_breaker")) == "getattr(self.context, '_breaker_forced_final', False) is True"


def test_the_uniquified_request_id_is_set_on_the_context():
    """Reader C MAJOR 2: `register()` uniquified the id after the contextvar
    was set, so the turn-start deadline check read a key nobody timed."""
    t = tree(ag)
    adopt = [a for a in assigns(t, "req_id") if U(a.value) == "_active_turn.req_id"]
    assert adopt
    sets = [c for c in calls(t, "set") if U(c.func) == "request_id_context.set" and c.args and U(c.args[0]) == "req_id"]
    assert any(first_line(c) > first_line(adopt[0]) for c in sets), "the uniquified id never reaches the contextvar"


def test_task_state_events_leave_a_durable_log_line():
    t = tree(ag)
    logs = [c for c in calls(t, "pretty_log") if c.args and isinstance(c.args[0], ast.Constant)]
    sow = [c for c in logs if c.args[0].value == "State Of Work"]
    assert len(sow) >= 3 and all(U(call_kw(c, "durable_only")) == "True" for c in sow)
    rr = [c for c in logs if c.args[0].value == "Report Reserve"]
    assert rr and U(call_kw(rr[0], "durable_only")) == "True"


