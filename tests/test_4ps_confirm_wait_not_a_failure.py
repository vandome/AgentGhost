"""§4PS: live 15f54e1b — "forget yt-skVFTeB4gEk.captions.el": the model
previewed, then confirmed in the same turn; the gate refused ("the user has
not answered yet") and the system answered with the preview's question —
the designed behaviour — yet the stream printed "Execution Fail" and
"turn outcome failed". The forget / reset_all confirm refusals are a
DESIGNED stop, like the project's `confirm_dead_end`: no strike, no failed
turn, no corpus failure flag."""
from __future__ import annotations

from ghost_agent.distill.outcome_heuristics import (DESIGNED_STOP_REASONS as H, is_designed_stop,
                                                    tool_failure_flags)
from ghost_agent.tools.outcome import DESIGNED_STOP_REASONS as T, ToolOutcome as TO
from tests.test_4pn_confirm_wait import _batches, real_kb  # noqa: F401 — fixture

WAIT = "NOT deleted: the user has not answered yet — show them the list and wait for their reply."


def test_only_an_answered_wait_is_a_designed_stop():
    assert T == H
    for code in ("forget_wait_shown", "reset_wait_shown"):
        assert is_designed_stop(TO.rejected(WAIT, reason_code=code))
    # r1 MAJOR: the gate's code itself covers refusals the MODEL must explain
    # (a narrowed yes, outside content read) — those stay failures
    for code in ("forget_not_confirmed", "reset_not_confirmed", "forget_bad_selection"):
        assert not is_designed_stop(TO.rejected("NOT deleted: x", reason_code=code))


def test_the_corpus_reads_no_failure_for_an_answered_wait_only():
    shown = TO.rejected(WAIT, reason_code="forget_wait_shown")
    assert tool_failure_flags([{"name": "knowledge_base", "content": shown}]) == []
    assert is_designed_stop(WAIT + "\n[designed stop: forget_wait_shown]")
    narrowed = TO.rejected("NOT deleted: the user's yes carries a condition.", reason_code="forget_not_confirmed")
    assert tool_failure_flags([{"name": "knowledge_base", "content": narrowed}]) == [True]


def test_the_census_class_is_asked_user():
    from ghost_agent.core.failure_census import tool_class
    out = TO.rejected(WAIT, reason_code="forget_wait_shown")
    assert tool_class({"name": "knowledge_base", "content": out}) == "asked_user"
    assert tool_class({"name": "knowledge_base",
                       "result": WAIT + "\n[designed stop: forget_wait_shown]"}) == "asked_user"


async def test_preview_then_same_turn_confirm_draws_no_strike(real_kb):
    """The 15f54e1b shape through the REAL dispatch and the REAL gate."""
    ts = await _batches([[("knowledge_base", {"action": "forget", "target": "yt-skVFTeB4gEk.captions.el"})],
                         lambda: [("knowledge_base", {"action": "forget", "confirm": real_kb.seen["tok"],
                                                      "items": "all"})]],
                        {"knowledge_base": real_kb})
    assert ts.force_stop is True and ts.final_ai_content.startswith("Nothing has been deleted yet.")
    assert ts.execution_failure_count == 0 and ts.last_was_failure is False
    kb_rows = [r for r in ts.tools_run_this_turn if r.get("name") == "knowledge_base"]
    assert kb_rows[-1]["content"].reason_code == "forget_wait_shown"    # what the recorder marks


async def test_a_reset_wait_draws_no_strike():
    from unittest.mock import MagicMock
    import ghost_agent.tools.memory as M
    ms = MagicMock()
    ms.collection.count.return_value = 12
    reset = M._reset_preview(ms, MagicMock(count_edges=MagicMock(return_value=3)))

    async def kb(**kw):
        if kw.get("confirm"):
            return TO.rejected("NOT executed: the user has not answered yet.", reason_code="reset_not_confirmed")
        return reset
    ts = await _batches([[("knowledge_base", {"action": "reset_all"})],
                         [("knowledge_base", {"action": "reset_all", "confirm": "yes"})]],
                        {"knowledge_base": kb})
    assert ts.force_stop is True and ts.execution_failure_count == 0


async def test_a_wait_the_takeover_declines_stays_a_strike():
    """r1 MAJOR: two previews → the model writes the reply (it said "deleted"
    3/3 in §4PN), so the refusal keeps its failure flag and its strike."""
    from tests.test_4pn_confirm_wait import _pv
    a = _pv("A", main=("fact 'a'",)).replace("tok1", "aaaa1111")
    b = _pv("B", main=("fact 'b'",)).replace("tok1", "bbbb2222")

    async def kb(**kw):
        if kw.get("confirm"):
            return TO.rejected(WAIT, reason_code="forget_not_confirmed")
        return a if kw.get("target") == "A" else b
    ts = await _batches([[("knowledge_base", {"action": "forget", "target": "A"}),
                          ("knowledge_base", {"action": "forget", "target": "B"})],
                         [("knowledge_base", {"action": "forget", "confirm": "aaaa1111", "items": "all"})]],
                        {"knowledge_base": kb})
    assert not ts.force_stop and ts.execution_failure_count == 1
    assert ts.tools_run_this_turn[-1]["content"].reason_code == "forget_not_confirmed"


async def test_an_earlier_strike_does_not_make_the_answered_wait_terminal(real_kb):
    """r1 m2: an invented token (a strike), then preview + confirm — the
    takeover answers, and the last call is not a failure (exec_terminal off)."""
    ts = await _batches([[("knowledge_base", {"action": "forget", "confirm": "deadbeef", "items": "all"})],
                         [("knowledge_base", {"action": "forget", "target": "yt-x.captions.en"})],
                         lambda: [("knowledge_base", {"action": "forget", "confirm": real_kb.seen["tok"],
                                                      "items": "all"})]],
                        {"knowledge_base": real_kb})
    assert ts.force_stop is True and ts.final_ai_content.startswith("Nothing has been deleted yet.")
    assert ts.last_was_failure is False


async def test_any_designed_stop_is_not_a_failed_last_call():
    """r1 m2, the project side: a `confirm_dead_end` (the project's own wait)
    as the last call does not make an earlier strike terminal."""
    async def mp(**kw):
        return TO.rejected("Error: already previewed in this turn — STOP calling delete in this turn.",
                           world_changed=False, reason_code="confirm_dead_end")
    ts = await _batches([[("manage_projects", {"action": "delete", "project_id": "p1"})]],
                        {"manage_projects": mp})
    assert ts.execution_failure_count == 0 and ts.last_was_failure is False


async def test_a_second_confirm_in_the_answered_batch_is_the_same_stop(real_kb):
    """r2 m1: two confirms in one batch — the takeover answers the first;
    the second must not draw a strike and make the turn read failed."""
    ts = await _batches([[("knowledge_base", {"action": "forget", "target": "yt-x.captions.en"})],
                         lambda: [("knowledge_base", {"action": "forget", "confirm": real_kb.seen["tok"],
                                                      "items": "all"}),
                                  ("knowledge_base", {"action": "forget", "confirm": real_kb.seen["tok"],
                                                      "items": "1"})]],
                        {"knowledge_base": real_kb})
    assert ts.force_stop is True and ts.final_ai_content.startswith("Nothing has been deleted yet.")
    assert ts.execution_failure_count == 0 and ts.last_was_failure is False
    assert [r["content"].reason_code for r in ts.tools_run_this_turn[-2:]] == ["forget_wait_shown"] * 2

