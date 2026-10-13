"""§4OG-5 — the clock: the report reserve sized from this request's measured
generation rate and its longest tool batch, re-checked before each dispatch,
and page previews capped under a client deadline."""
import inspect

from ghost_agent.core import task_state as TS
from ghost_agent.core import agent as ag
from ghost_agent.core.agent import wait_crosses_deadline, _BATCH_WALL_TOOLS
from ghost_agent.tools import browser as B
from ghost_agent.tools import browser_support as BS
from tests.pin_ast import tree, U, calls, call_kw, assigns, assigned_values, ifs


def test_the_reserve_is_sized_from_the_rate_and_the_longest_batch():
    """Fails where the reserve was 2× the longest LLM reply: at 25 tok/s an
    8k-token report is 328 s, and a 180 s batch started just outside a 300 s
    reserve ran into it (§4OF defect 6)."""
    rid = "req-4og-f"
    TS.forget_reply_seconds(rid)
    assert TS.sized_report_floor(300.0, 1800.0, rid) == 300.0                      # nothing measured yet
    TS.note_reply_tokens(rid, 100, 10.0)                                           # too short to measure
    assert TS.reply_rate_tok_s(rid) == 0.0
    TS.note_reply_tokens(rid, 2500, 50.0)                                          # 50 tok/s
    TS.note_reply_tokens(rid, 1000, 40.0)                                          # 25 tok/s — the slowest wins
    assert TS.reply_rate_tok_s(rid) == 25.0
    est = TS.report_turn_estimate_s(rid)
    assert abs(est - 1.5 * 8192 / 25.0) < 0.01                                     # ≈ 491 s
    assert TS.sized_report_floor(300.0, 1800.0, rid) == 450.0                      # the 25 % cap binds
    assert abs(TS.sized_report_floor(300.0, 3600.0, rid) - (est + 60.0)) < 0.01    # estimate + slack
    # the batch wall is kept PER TOOL KIND and is the pre-dispatch check's, not the floor's (reader 3 M5)
    TS.note_batch_wall(rid, 90.0, "browser"); TS.note_batch_wall(rid, 600.0, "execute"); TS.note_batch_wall(rid, 20.0, "browser")
    assert TS.max_batch_wall(rid, "browser") == 90.0 and TS.max_batch_wall(rid, "execute") == 600.0
    assert TS.max_batch_wall(rid, "web_search") == 0.0 and TS.max_batch_wall(rid) == 600.0
    assert abs(TS.sized_report_floor(300.0, 3600.0, rid) - (est + 60.0)) < 0.01       # unchanged by the wall
    # a fast request keeps the constant floor
    fast = "req-4og-fast"
    TS.note_reply_tokens(fast, 3000, 20.0)                                         # 150 tok/s → 82 s
    assert TS.sized_report_floor(300.0, 1800.0, fast) == 300.0
    # before any rate: the §4OE sizing from the longest reply still applies
    old = "req-4og-old"
    TS.note_reply_seconds(old, 200.0)
    assert TS.sized_report_floor(300.0, 3600.0, old) == 460.0
    TS.forget_reply_seconds(rid)
    assert TS.reply_rate_tok_s(rid) == 0.0 and TS.max_batch_wall(rid) == 0.0


def test_the_loop_records_the_rate_and_the_batch_wall_and_rechecks_before_dispatch():
    t = tree(ag)
    # the rate is GENERATION only: tokens from the first delta, not prefill/TTFT (reader 3 M3)
    assert "time.monotonic() - _first_delta_at if _first_delta_at else thinking_duration" in assigned_values(t, "_gen_s")
    rt = calls(t, "_note_reply_t")
    assert rt and U(rt[0].args[1]) == "int(reasoning_token_count or 0) + int(content_token_count or 0) + int(tool_arg_token_count or 0)" and U(rt[0].args[2]) == "_gen_s"
    # a native tool-call delta is a generated token and starts the generation clock (the live probe's 44 KB write batch
    # read as 5 tok/s with the arguments uncounted)
    assert assigned_values(t, "_first_delta_at").count("time.monotonic()") == 3
    assert any(isinstance(a, __import__("ast").AugAssign) for a in assigns(t, "tool_arg_token_count"))
    # this call's wall, per tool kind, only for the long tools (reader 3 M5)
    bw = [c for c in calls(t, "_note_bw")]
    assert bw and [U(a) for a in bw[0].args] == ["request_id_context.get()", "_call_dur", "fname"]
    assert any(calls(i, "_note_bw") for i in ifs(t, "fname in _BATCH_WALL_TOOLS"))
    assert "max(float(_dl_wait or 0.0), _mbw(request_id_context.get(), fname))" in assigned_values(t, "_dl_wait")
    assert "browser" in _BATCH_WALL_TOOLS and "execute" in _BATCH_WALL_TOOLS and "file_system" not in _BATCH_WALL_TOOLS
    # the predicate: a call as long as this kind's longest, started with less than that beyond the reserve, is refused
    assert wait_crosses_deadline(180.0, 301.0, 300.0) and not wait_crosses_deadline(180.0, 500.0, 300.0)


def test_page_previews_are_capped_under_a_client_deadline():
    """Fails where a 65,536-char preview per navigate was prefilled under a
    240 s deadline (97 of 185 s)."""
    from ghost_agent.utils.logging import client_deadline_context
    tok = client_deadline_context.set(1800.0)
    try:
        assert BS.preview_cap_chars(B._MAX_TEXT_CHARS) == BS.DEADLINE_PREVIEW_CAP == 16 * 1024
    finally:
        client_deadline_context.reset(tok)
    tok = client_deadline_context.set(0.0)
    try:
        assert BS.preview_cap_chars(B._MAX_TEXT_CHARS) == B._MAX_TEXT_CHARS
    finally:
        client_deadline_context.reset(tok)
    t = tree(B)
    caps = [c for c in calls(t, "preview_cap_chars")]
    assert caps and U(caps[0].args[0]) == "_MAX_TEXT_CHARS"
    assert any("preview_cap_chars" in U(a.value) for a in assigns(t, "payload['nav_text_chars']"))


