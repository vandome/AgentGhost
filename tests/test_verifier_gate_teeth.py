"""Tests for the verifier gate gaining real authority over the outcome.

Regression target: the gate was advisory-only. A REFUTED verdict merely
stapled a note onto the reply; the finalize confidence was computed from
historical domain competence + a hardcoded neutral entropy, so it equalled
competence (0.92/0.96) regardless of whether the work was actually broken.
Result: req_44 finished "REFUTED (95%)" yet reported C=0.92 below=no, and
req_C0 finalised an untested file write ("gate skipped") at C=0.96. Two
fixes locked in here:
  1. (confidence.score(outcome_penalty=...) — removed with the confidence
     score, §4PJ.)
  2. _is_unverified_mutation(...) — finalising on an untested write/replace
     is recorded as a failed outcome, not a silent success.
"""
from ghost_agent.core.agent import _is_unverified_mutation


# ── outcome_penalty on the confidence reading ────────────────────────
# ── _is_unverified_mutation classification ───────────────────────────
def test_write_success_is_unverified_mutation():
    tool = {"name": "file_system",
            "content": "SUCCESS: Wrote 21044 chars to 'index.html'. Script-side path: 'index.html'."}
    assert _is_unverified_mutation(tool) is True


def test_replace_success_is_a_targeted_edit_not_the_gates_shape():
    # Operator decision 2026-09-23 (§4KC): the gate is for WHOLE-FILE writes;
    # a targeted replace/edit of an existing file does not force a run.
    tool = {"name": "file_system",
            "content": "SUCCESS: Exact match found and replaced in 'index.html'."}
    assert _is_unverified_mutation(tool) is False


def test_file_read_is_not_a_mutation():
    # a READ returns the file body, not a write/replace SUCCESS line
    tool = {"name": "file_system",
            "content": "<html><body><script>const player = {...}</script></body></html>"}
    assert _is_unverified_mutation(tool) is False


def test_non_filesystem_tool_is_not_a_mutation():
    # an execute that happens to print "wrote" is not a file mutation
    tool = {"name": "execute", "content": "SUCCESS: the program wrote output"}
    assert _is_unverified_mutation(tool) is False


def test_none_and_garbage_are_safe():
    assert _is_unverified_mutation(None) is False
    assert _is_unverified_mutation({}) is False
    assert _is_unverified_mutation({"name": "file_system"}) is False
