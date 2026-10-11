"""§4KW — two defects from the overnight log (2026-10-02).

(1) A tool-free reply that only ANNOUNCED work shipped as the answer
    (slack-3120ad1e, slack-541256be). The §4IW opener/verb list
    (`_announced_work_without_acting`) gives one continuation; the worker-
    model check that used to back it up was switched off in §4NF and has
    been removed. Fails in the world where the list's YES does not trigger
    the continuation, or where it stacks on the pending-promise steer.

(2) A request a loop breaker closed printed "ok · recovered N strike(s)"
    when no trajectory row was written (sim 18f014a7). Fails in the world
    where the outcome line ignores the breaker stamp.
"""
import pytest

from ghost_agent.core import agent as A
from tests import test_4kv_slack12_fixes as K


# ══ (1) through the turn loop ════════════════════════════════════════════════

async def _drive(monkeypatch, tmp_path, script, *, request="φτιάξε μου το poster",
                 collector=True, tools=("web_search", "image_generation"), role="member"):
    """A member's turn, as both live cases were — and a member's turn runs no
    verifier, so nothing but the guard can send a second generation."""
    orig = K._recording_agent

    def wired(mp, tp):
        agent, ctx, x = orig(mp, tp)
        if not collector:
            ctx.trajectory_collector = None
        return agent, ctx, x
    monkeypatch.setattr(K, "_recording_agent", wired)
    final, model, logged, rows, agent = await K._drive(monkeypatch, tmp_path, script, request=request, tools=tools,
                                                       role=role)
    return final, model, logged, rows, agent


def _directive_sent(payload):
    return any(A._ANNOUNCED_WORK_DIRECTIVE in u for u in K._user_texts(payload))


# (the request is in the reply's language: the reply-language gate would otherwise regenerate it)
async def test_the_reply_shape_rule_gives_the_continuation(monkeypatch, tmp_path):
    listed = "Let me search for the opening hours of the museum."
    assert A._announced_work_without_acting(listed, request="when does it open")
    _, model, logged, _, _ = await _drive(monkeypatch, tmp_path, [
        ("say", listed), ("say", "It opens at 9."), ("say", "x")], request="when does it open")
    assert _directive_sent(model.payloads[1])
    assert any("(reply shape, no tool ran)" in c for t, c, _ in logged if t == "Turn Budget")


async def test_one_promise_nudge_per_request(monkeypatch, tmp_path):
    """Review §4KW: after a tool ran, the announced-work continuation and the
    older pending-promise steer both fired on one request (4 generations,
    two directives that disagree). One nudge in all."""
    _, model, logged, _, _ = await _drive(monkeypatch, tmp_path, [
        ("call", "web_search", {"query": "x"}), ("say", "Let me search more specifically for the sender domain."),
        ("say", "Let me look it up now."), ("say", "The sender domain is not in any result."), ("say", "x")],
        request="find the sender domain")
    assert len(model.payloads) == 3, [t for t, c, _ in logged if t in ("Turn Budget", "Pending-Promise Guard")]
    assert not any(t == "Pending-Promise Guard" for t, _, _ in logged)


async def test_the_pending_promise_guard_still_works_alone(monkeypatch, tmp_path):
    """When the announced-work guard did not fire, the older guard is untouched."""
    promise = "I found the page. I'll update the config file next."
    assert not A._announced_work_without_acting(promise, request="fix the config")
    _, model, logged, _, _ = await _drive(monkeypatch, tmp_path, [
        ("call", "web_search", {"query": "x"}), ("say", promise), ("say", "Done: it was NOT updated."), ("say", "x")],
        request="fix the config")
    assert any(t == "Pending-Promise Guard" for t, _, _ in logged) and len(model.payloads) == 3


# ══ (2) the outcome line of a turn with no row ═══════════════════════════════

async def test_a_breaker_closed_turn_with_no_row_prints_failed(monkeypatch, tmp_path):
    """Sim 18f014a7's shape: two thinking-loop kills, the breaker's report,
    and no trajectory collector."""
    _, _, logged, rows, _ = await _drive(monkeypatch, tmp_path, [
        ("loop", K.FRAME), ("call", "web_search", {"query": "x"}), ("loop", K.STOP_FRAME),
        ("say", "Δεν ξέρω — η αναζήτηση βρήκε μόνο επαγγελματικά προφίλ.")], collector=False)
    outcome = [c for t, c, _ in logged if t == "Turn Outcome"]
    assert outcome and outcome[-1].startswith("failed") and "recovered" not in outcome[-1], outcome


async def test_an_ordinary_turn_with_no_row_still_prints_ok(monkeypatch, tmp_path):
    _, _, logged, _, _ = await _drive(monkeypatch, tmp_path, [("say", "Εντάξει, έγινε."), ("say", "x")],
                                      collector=False)
    assert [c for t, c, _ in logged if t == "Turn Outcome"][-1].startswith("ok")


def test_the_stamp_alone_turns_the_label(monkeypatch):
    """Unit: the emitter, no row, with and without the request's stamp."""
    from tests.helpers import make_agent
    agent = make_agent()
    lines = []
    monkeypatch.setattr(A, "pretty_log", lambda title, content=None, **kw: lines.append((title, str(content))))
    kw = dict(trajectory_id=None, final_content="x", tools=[], execution_failure_count=1,
              exec_terminal=False, unacked_total_failure=False, budget_exhausted=False, shape_failed=False)
    agent._emit_turn_outcome_line(req_id="sim-a", **kw)
    A.stamp_loop_breaker(agent.context, "sim-b", "thinking_loop")
    agent._emit_turn_outcome_line(req_id="sim-b", **kw)
    out = [c for t, c in lines if t == "Turn Outcome"]
    assert out[0].startswith("ok") and "recovered 1 strike(s)" in out[0]
    assert out[1].startswith("failed") and "recovered" not in out[1]


# ══ (3) the suite's own order-dependence (found by this change's full run) ═══
# Two tests in this order, in one file: the first reloads the verifier, the
# second checks the class this file imported at collection time is still the
# module's. Fails in the world without conftest's `_restore_reloaded_modules`.
from ghost_agent.core import verifier as _V_AT_IMPORT
from ghost_agent.core.verifier import VerifyVerdict as _VERDICT_AT_IMPORT


def test_zz_reload_1_a_test_reloads_the_verifier():
    import importlib
    importlib.reload(_V_AT_IMPORT)
    assert _V_AT_IMPORT.VerifyVerdict is not _VERDICT_AT_IMPORT     # a reload DOES rebuild the class


def test_zz_reload_2_the_next_test_sees_the_original_class():
    assert _V_AT_IMPORT.VerifyVerdict is _VERDICT_AT_IMPORT


def test_an_abort_marker_alone_turns_the_label(monkeypatch):
    """The corpus's first rule, mirrored for a turn with no row: 18 sims in the
    log printed "ok · recovered 2 strike(s)" over "[ATTEMPT_ABORTED_THINKING_LOOP] …"."""
    from tests.helpers import make_agent
    agent = make_agent()
    lines = []
    monkeypatch.setattr(A, "pretty_log", lambda title, content=None, **kw: lines.append((title, str(content))))
    kw = dict(trajectory_id=None, tools=[], execution_failure_count=2,
              exec_terminal=False, unacked_total_failure=False, budget_exhausted=False, shape_failed=False)
    agent._emit_turn_outcome_line(req_id="sim-c", final_content="[ATTEMPT_ABORTED_THINKING_LOOP] The solver hit the cap.", **kw)
    agent._emit_turn_outcome_line(req_id="sim-d", final_content="Done: the file is written.", **kw)
    agent._emit_turn_outcome_line(req_id="sim-e", final_content="evidence…\n\n[ATTEMPT_ABORTED_STRIKE_CAP] I hit a hard limit.", **kw)
    out = [c for t, c in lines if t == "Turn Outcome"]
    assert out[0].startswith("failed") and out[2].startswith("failed")
    assert out[1].startswith("ok") and "recovered 2 strike(s)" in out[1]


# Leaks between tests (conftest `_StateIsolation`): a test's own changes to
# os.environ and to ghost_agent module globals are undone after it; what
# session/module fixtures set is kept. Fails in the world without it (the
# leak detector found GHOST_VERIFY_TWO_STAGE, a stale GHOST_HOME and
# `core.agent.request_id_context = MagicMock()` crossing into later files).
import os as _os
from ghost_agent.core import agent as _AGENT_AT_IMPORT
_RIC_AT_IMPORT = _AGENT_AT_IMPORT.request_id_context


_os.environ.setdefault("GHOST_4KW_PRESET", "as-collected")


def test_zz_leak_1_a_test_rebinds_globals_and_env():
    _os.environ["GHOST_4KW_LEAK_PROBE"] = "1"
    _os.environ["GHOST_4KW_PRESET"] = "changed-by-a-test"
    _AGENT_AT_IMPORT.request_id_context = object()


def test_zz_leak_2_the_next_test_does_not_see_them():
    assert "GHOST_4KW_LEAK_PROBE" not in _os.environ                      # added → removed
    assert _os.environ.get("GHOST_4KW_PRESET") == "as-collected"          # changed → restored
    assert _AGENT_AT_IMPORT.request_id_context is _RIC_AT_IMPORT


def test_zz_leak_3_session_fixture_env_survives():
    """The detached-job registry is set by a SESSION fixture: kept."""
    from tests.conftest import JOB_REGISTRY
    assert _os.environ.get("GHOST_TEST_JOB_REGISTRY") == str(JOB_REGISTRY)


@pytest.mark.parametrize("marker", ["[ATTEMPT_ABORTED_NO_PROGRESS]", "[ATTEMPT_ABORTED_CROSS_TURN_LOOP]",
                                    "[ATTEMPT_ABORTED_TURN]", "[ATTEMPT_ABORTED_STRIKE_CAP]"])
def test_every_abort_marker_in_the_store_turns_the_label(monkeypatch, marker):
    """The markers present in the trajectory store, each alone, no row."""
    from tests.helpers import make_agent
    agent = make_agent()
    lines = []
    monkeypatch.setattr(A, "pretty_log", lambda title, content=None, **kw: lines.append((title, str(content))))
    agent._emit_turn_outcome_line(req_id="sim-m", trajectory_id=None, final_content=f"x\n\n{marker} stopped.", tools=[],
                                  execution_failure_count=0, exec_terminal=False, unacked_total_failure=False,
                                  budget_exhausted=False, shape_failed=False)
    assert [c for t, c in lines if t == "Turn Outcome"][0].startswith("failed")


def test_a_marker_quoted_by_the_correction_banner_does_not_fail_the_turn(monkeypatch):
    """Open item (§4KW): on finalize the previous turn's correction banner is
    prepended AFTER the record; the line reads the reply as recorded."""
    from tests.helpers import make_agent
    agent = make_agent()
    lines = []
    monkeypatch.setattr(A, "pretty_log", lambda title, content=None, **kw: lines.append((title, str(content))))
    banner = "*(Correction to my previous reply: it said '[ATTEMPT_ABORTED_TURN]' in error.)*\n\n"
    kw = dict(trajectory_id=None, tools=[], execution_failure_count=0, exec_terminal=False,
              unacked_total_failure=False, budget_exhausted=False, shape_failed=False)
    agent._emit_turn_outcome_line(req_id="t-1", final_content=banner + "The answer is 4.",
                                  marker_text="The answer is 4.", **kw)
    agent._emit_turn_outcome_line(req_id="t-2", final_content=banner + "x\n\n[ATTEMPT_ABORTED_TURN] stopped.",
                                  marker_text="x\n\n[ATTEMPT_ABORTED_TURN] stopped.", **kw)
    out = [c for t, c in lines if t == "Turn Outcome"]
    assert out[0].startswith("ok") and out[1].startswith("failed")


def test_finalize_passes_the_recorded_reply_to_the_line():
    """AST: the finalize call passes `marker_text=_recorded_reply`, captured
    BEFORE the banner is prepended."""
    import ast as _ast, inspect as _inspect
    tree = _ast.parse(_inspect.getsource(A))
    calls = [c for c in _ast.walk(tree) if isinstance(c, _ast.Call)
             and getattr(c.func, "attr", "") == "_emit_turn_outcome_line"
             and any(k.arg == "marker_text" for k in c.keywords)]
    assert len(calls) == 1
    # capped as the ROW stores it (fresh review: a marker in the elided middle)
    assert _ast.unparse(next(k.value for k in calls[0].keywords if k.arg == "marker_text")) == "_cap_recorded_reply(_recorded_reply)"
    fn = next(n for n in _ast.walk(tree) if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))
              and any(c is calls[0] for c in _ast.walk(n)))
    src = _ast.unparse(fn)
    assert src.index("_recorded_reply = final_ai_content") < src.index("_take_active_correction()")



# ══ route(): the deadline by cancellation; keepalive and our own queueing ══════

async def test_route_gives_up_at_its_deadline_without_blaming_the_node(monkeypatch):
    """Review (11 live ReadTimeouts at +12.0 s on decompose/expand, 6 next to
    our own critic dispatch; one opened the breaker). route() now returns its
    free fallback at the deadline by CANCELLATION, and the node is not charged."""
    import asyncio, time, httpx
    from ghost_agent.core import llm as L
    from ghost_agent.utils.logging import request_id_context
    monkeypatch.setattr(L, "_ROUTE_TIMEOUT_S", 0.4)
    c = L.LLMClient("http://127.0.0.1:1", worker_nodes=[{"url": "http://worker.invalid:8088", "model": "Nova"}])
    node = c.worker_clients[0]

    async def slow_post(*a, **kw):
        await asyncio.sleep(min(float(kw.get("timeout") or 30.0), 30.0))
        raise httpx.ReadTimeout("slow node")
    node["client"].post = slow_post
    monkeypatch.setattr(c, "_known_slots", lambda *_a, **_k: 4, raising=False)
    tok = request_id_context.set("route-deadline")
    try:
        for _ in range(4):
            t = time.monotonic()
            assert await c.route("DECOMPOSE_QUERY", {"model": "m", "messages": []}, fallback="fb") == "fb"
            assert time.monotonic() - t < 1.5
    finally:
        request_id_context.reset(tok)
    st = c.circuit_breaker._get_state(node["url"])
    assert st["failures"] == 0 and st.get("state") != "open", st


def test_a_keepalive_stuck_behind_our_own_jobs_is_not_a_node_fault():
    import httpx
    from ghost_agent.core import llm as L
    c = L.LLMClient("http://127.0.0.1:1", worker_nodes=[{"url": "http://n:8088", "model": "Nova"}])
    node = c.worker_clients[0]
    c._node_slot_caps = {"http://n:8088": 2}
    c._node_run_tasks = {"http://n:8088": {1: None, 2: None}}          # our gated jobs fill both slots
    e = httpx.ReadTimeout("x")
    assert L._charge_node_fault(c, node, e, "keepalive") is False
    c._node_run_tasks = {"http://n:8088": {1: None}}                   # a free slot: a stuck ping is real
    assert L._charge_node_fault(c, node, e, "keepalive") is True
    c._node_run_tasks = {}
    assert L._charge_node_fault(c, node, e, "keepalive") is True
    c._node_run_tasks = {"http://n:8088": {1: None, 2: None}}
    assert L._charge_node_fault(c, node, e, "decompose query") is True   # only the ping is excused
    assert L._charge_node_fault(c, node, e, "warmup") is False
    resp = httpx.Response(400, request=httpx.Request("POST", "http://n"))
    assert L._charge_node_fault(c, node, httpx.HTTPStatusError("bad", request=resp.request, response=resp), "x") is False
