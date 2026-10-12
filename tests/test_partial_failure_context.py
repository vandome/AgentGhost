"""Regression tests for the partial-failure / stale-context bug.

Background
----------
A user asked the agent to "delete `ecef207c0d4b` and `516217d294cc`". One id
resolved and was deleted; the other did not exist and returned an ERROR. The
agent then produced a *non-sensical* response: it re-answered the PREVIOUS
turn's "what is the difference between these projects" question instead of
reporting the delete outcome.

Two independent defects combined to cause it:

1. Contextual query expansion prepended the previous assistant reply as
   ``Context: ...`` onto the memory-search query for ANY short message —
   including self-contained imperatives like "delete `<id>` and `<id>`".
   The 200-char prior-reply snippet then dominated vector retrieval, so the
   injected memory described the previous topic, not the delete.

2. A multi-id command is N separate tool calls. A partial failure (one ok,
   one error) was collapsed into a single generic strike whose diagnostic
   named only the last error — the model never saw a clean
   "A deleted, B not found" picture.

These tests lock in both fixes:
  * ``summarize_multi_op_outcomes`` — produces the explicit partial summary.
"""

import pytest

from ghost_agent.core.agent import GhostAgent
from ghost_agent.tools.tool_failure import summarize_multi_op_outcomes




class TestSummarizeMultiOpOutcomes:
    """Fix #2/#3: a partial failure across >=2 calls yields an explicit
    'N ok / M failed' summary that is authoritative over stale context."""

    def _delete_partial(self):
        return [
            {"tool": "manage_projects", "ok": True, "preview": None},
            {"tool": "manage_projects", "ok": False,
             "preview": "project not found: '516217d294cc' — NOTHING was deleted."},
        ]

    def test_partial_failure_summarized(self):
        out = summarize_multi_op_outcomes(self._delete_partial())
        assert "1 of 2" in out
        assert "SUCCEEDED" in out and "FAILED" in out
        assert "516217d294cc" in out

    def test_summary_marks_successes_authoritative(self):
        out = summarize_multi_op_outcomes(self._delete_partial())
        # The model must be told the success took effect and not to retry it.
        assert "AUTHORITATIVE" in out
        assert "do NOT retry" in out or "do not retry" in out.lower()

    def test_single_op_failure_has_no_summary(self):
        # A lone failure keeps its existing terse diagnostic — no noise.
        assert summarize_multi_op_outcomes(
            [{"tool": "manage_projects", "ok": False, "preview": "boom"}]) == ""

    def test_all_success_has_no_summary(self):
        assert summarize_multi_op_outcomes([
            {"tool": "manage_projects", "ok": True, "preview": None},
            {"tool": "manage_projects", "ok": True, "preview": None},
        ]) == ""

    def test_all_fail_has_no_summary(self):
        # Uniform all-fail is served fine by the normal diagnostic.
        assert summarize_multi_op_outcomes([
            {"tool": "x", "ok": False, "preview": "a"},
            {"tool": "y", "ok": False, "preview": "b"},
        ]) == ""

    @pytest.mark.parametrize("empty", [[], None])
    def test_empty_is_safe(self, empty):
        assert summarize_multi_op_outcomes(empty) == ""

    def test_missing_preview_falls_back(self):
        out = summarize_multi_op_outcomes([
            {"tool": "a", "ok": True},
            {"tool": "b", "ok": False},  # no preview key
        ])
        assert "b: failed" in out
