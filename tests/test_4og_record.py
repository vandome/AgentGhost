"""§4OG-2 — the RECORD of a delivered long task: the same-error shape rule
yields to a delivered turn, and a document deliverable is read back for the
claim lens (the §4OF research report was REFUTED "lacks the Sources section"
while the file had one; the §4OE build was recorded FAILED with 23/23 green)."""
from pathlib import Path

from ghost_agent.core import agent as ag
from ghost_agent.core.agent import delivered_ok_from_calls, _written_documents_for_audit
from ghost_agent.distill.outcome_heuristics import classify_chat_outcome
from ghost_agent.distill.schema import Trajectory, ToolCall, Outcome
from tests.pin_ast import tree, U, calls, assigns, assigned_values, strings, first_line


def _traj(calls, final, extra=None):
    t = Trajectory(id="t1", user_request="build it", final_response=final, tool_calls=calls,
                   outcome=Outcome.UNKNOWN.value)
    if extra is not None:
        t.extra = extra
    return t


ERR = ToolCall(name="execute", arguments={"command": "python -m pytest -q"},
               result="FAILED tests/test_a.py::test_x - AssertionError\n1 failed in 0.2s\nEXIT CODE: 1", error="exit 1")


def test_a_green_last_run_is_delivered_and_the_shape_rule_yields():
    """Fails where three identical failing runs before the green one recorded
    the delivered build FAILED (§4OE coding probe, 998 s, 23/23)."""
    green = ToolCall(name="execute", arguments={"command": "python -m pytest -q"},
                     result="===== 23 passed in 0.4s =====\nEXIT CODE: 0")
    calls = [ERR, ERR, ERR, green]
    assert delivered_ok_from_calls(calls, "All 23 tests pass.")
    assert classify_chat_outcome(_traj(calls, "All 23 tests pass.", {"delivered_ok": True})).outcome == Outcome.UNKNOWN.value
    # without the extra the rule still fires (the recorder sets it; the heuristic trusts the row)
    assert classify_chat_outcome(_traj(calls, "All 23 tests pass.")).outcome == Outcome.FAILED.value
    # a last run that FAILED is not delivered — and it VETOES the file arm (reader 2 M1)
    assert not delivered_ok_from_calls([green, ERR], "done")
    wrote = ToolCall(name="file_system", arguments={"operation": "write", "path": "app.py"},
                     result="SUCCESS: Wrote 10 chars to 'app.py'. Script-side path (from sandbox cwd): 'app.py'.")
    assert not delivered_ok_from_calls([wrote, ERR], "I updated `app.py` but 5 tests still fail: [app.py](/api/download/app.py)")
    assert classify_chat_outcome(_traj([ERR, ERR, ERR], "could not fix it", {"delivered_ok": False})).outcome == Outcome.FAILED.value


def test_a_written_file_the_reply_hands_over_is_delivered():
    """Fails where four browser errors (two of them the host memo's own
    refusal text) recorded the research turn FAILED with the report on disk."""
    blocked = ToolCall(name="browser", arguments={"url": "https://x/a"},
                       result="--- BROWSER RESULT ---\nSTATUS: ERROR x failed 2 times in the last few hours (blocked: HTTP 403)", error="err")
    wrote = ToolCall(name="file_system", arguments={"operation": "write", "path": "probe/vectordb.md"},
                     result="SUCCESS: Wrote 12552 chars to 'probe/vectordb.md'. Script-side path (from sandbox cwd): 'probe/vectordb.md'.")
    calls = [blocked, blocked, blocked, blocked, wrote]
    reply = "Done. The report is at `probe/vectordb.md`: [Download](/api/download/probe/vectordb.md)"
    assert delivered_ok_from_calls(calls, reply)
    assert not delivered_ok_from_calls(calls, "I could not finish the report.")          # the file is not handed over
    assert not delivered_ok_from_calls(calls, "I could not finish vectordb.md; it has only the outline.")   # a MENTION is not a hand-over (reader 2 M1)
    assert delivered_ok_from_calls(calls, "The report is saved to probe/vectordb.md.")
    assert delivered_ok_from_calls(calls, "Written to `vectordb.md`.")
    refused = ToolCall(name="file_system", arguments={"operation": "write", "path": "probe/vectordb.md"},
                       result="REJECTED: 'probe/vectordb.md' already exists and this request has not read it")
    assert not delivered_ok_from_calls([blocked, blocked, blocked, refused], reply)      # a refused write is no deliverable
    assert classify_chat_outcome(_traj(calls, reply, {"delivered_ok": True})).outcome == Outcome.UNKNOWN.value


def test_the_recorder_stamps_delivered_ok_from_the_calls():
    t = tree(ag.GhostAgent._record_turn_trajectory)
    c = calls(t, "delivered_ok_from_calls")
    assert c and [U(a) for a in c[0].args] == ["tool_calls", "final_content"]
    assert "True" in assigned_values(t, "_extra['delivered_ok']")


def test_document_deliverables_are_read_back_for_the_claim_lens(tmp_path):
    """Fails where the claim lens packed tool RESULT text only ("SUCCESS:
    Wrote 12552 chars") and judged the report on the reply's summary."""
    (tmp_path / "report.md").write_text("# Report\n\nbody\n\n## Sources\n- https://a\n- https://b\n")
    (tmp_path / "app.py").write_text("print(1)\n")
    rows = [
        {"name": "file_system", "args": {"operation": "write", "path": "report.md"},
         "content": "SUCCESS: Wrote 60 chars to 'report.md'. Script-side path (from sandbox cwd): 'report.md'."},
        {"name": "file_system", "args": {"operation": "write", "path": "app.py"},
         "content": "SUCCESS: Wrote 9 chars to 'app.py'. Script-side path (from sandbox cwd): 'app.py'."},
    ]
    pack = _written_documents_for_audit(rows, tmp_path)
    assert "file this turn wrote: report.md" in pack and "## Sources" in pack and "https://b" in pack
    assert "app.py" not in pack                                     # code is the code lens's business
    assert _written_documents_for_audit([], tmp_path) == ""
    # the claim lens packs it INSIDE the evidence budget and prepends it (reader 2 M2), logs the
    # digest after (m4), and carries no instruction inside the evidence (m5)
    t = tree(ag.GhostAgent._compute_verifier_verdict)
    assert calls(t, "_written_documents_for_audit")
    assert "len(_docs_prefix)" in assigned_values(t, "_docs_len")
    pack = [a for a in assigns(t, "claim_evidence") if U(a.value) == "_docs_prefix + claim_evidence"]
    assert pack, "the pack is not prepended"
    logs = calls(t, "_log_evidence_digest")
    assert logs and min(first_line(c) for c in logs) > first_line(pack[0])
    assert not any("judge the claim against the FILE" in x for x in strings(t))
