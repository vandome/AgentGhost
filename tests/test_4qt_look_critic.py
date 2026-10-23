"""§4QT — the render look asked of the critic pool (operator, 2026-10-22:
"change look to nova, test it!"). The vision pool is the main server itself
(-np 1): the §4QP look costs 3 s of vision and ~32 s of the owner's context
re-prefilling. GHOST_LOOK_POOL=critic (or the probe arm `look_critic`) sends
the look to the critic pool and falls back to the vision pool when that
fails. Every pin names the world it fails in."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

import ghost_agent.tools.image_gen as IG
from ghost_agent.core.receipt_marks import UNSEEN_IMAGE_NOTE
from ghost_agent.tools.image_gen import tool_generate_image
from tests.test_4oz_subjects_collage import _llm, _vision
from tests.test_4qp_render_look import SCENE, SEEN, on  # noqa: F401 — fixture

NOVA = [{"model": "Nova", "url": "http://nova:8088"}]


@pytest.fixture
def critic(monkeypatch):
    monkeypatch.setenv(IG.LOOK_POOL_ENV, "critic")


def _pool_llm(critic_clients=NOVA, answers=None):
    """A client whose chat_completion answers in order; `critic_clients` is a
    real list (an AsyncMock attribute is a MagicMock — not a pool)."""
    llm = _llm()
    llm.critic_clients = critic_clients
    if answers is None:
        llm.chat_completion = AsyncMock(return_value=_vision({"scene": SCENE}))
    else:
        llm.chat_completion = AsyncMock(side_effect=answers)
    return llm


def _pools(llm):
    return [("critic" if c.kwargs.get("use_critic") else "vision" if c.kwargs.get("use_vision") else "?")
            for c in llm.chat_completion.await_args_list]


# ── the default is the vision pool, exactly as §4QP shipped ──────────────────

async def test_without_the_switch_the_look_stays_on_the_vision_pool(tmp_path, on, monkeypatch):
    monkeypatch.delenv(IG.LOOK_POOL_ENV, raising=False)
    llm = _pool_llm()
    out = await tool_generate_image(prompt="a cat in space", llm_client=llm, sandbox_dir=tmp_path)
    assert _pools(llm) == ["vision"] and SEEN + SCENE in out


@pytest.mark.parametrize("value", ["", "vision", "main", "0", "1", "nova"])
async def test_only_the_word_critic_moves_the_look(tmp_path, on, monkeypatch, value):
    monkeypatch.setenv(IG.LOOK_POOL_ENV, value)
    llm = _pool_llm()
    await tool_generate_image(prompt="a cat", llm_client=llm, sandbox_dir=tmp_path)
    assert _pools(llm) == ["vision"]


# ── the switch ───────────────────────────────────────────────────────────────

async def test_with_the_switch_the_look_is_asked_of_the_critic_pool_off_main(tmp_path, on, critic):
    llm = _pool_llm()
    out = await tool_generate_image(prompt="a cat in space", llm_client=llm, sandbox_dir=tmp_path)
    assert _pools(llm) == ["critic"]
    kw = llm.chat_completion.await_args.kwargs
    assert kw.get("off_main_only") is True, "a dead critic pool must RAISE, not land the look on the 35B unnamed"
    assert kw.get("timeout") == IG.LOOK_CRITIC_TIMEOUT_S and IG.LOOK_CRITIC_TIMEOUT_S < IG.LOOK_TIMEOUT_S
    assert kw.get("task_label") == "render-look"
    assert SEEN + SCENE in out and UNSEEN_IMAGE_NOTE not in out
    assert len(IG.RECENT_LOOK_SECONDS) == 1


async def test_the_critic_leg_sends_the_same_picture_and_question(tmp_path, on, critic):
    llm = _pool_llm()
    await tool_generate_image(prompt="a cat", llm_client=llm, sandbox_dir=tmp_path)
    payload = llm.chat_completion.await_args.args[0]
    parts = payload["messages"][0]["content"]
    assert parts[0]["text"].startswith(IG._LOOK_QUESTION) and "never guess who anyone is" in parts[0]["text"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert payload.get("max_tokens") == 200


@pytest.mark.parametrize("clients", [[], None, (), "not-a-pool"])
async def test_without_a_critic_pool_the_switch_keeps_the_vision_pool(tmp_path, on, critic, clients):
    llm = _pool_llm(critic_clients=clients)
    out = await tool_generate_image(prompt="a cat", llm_client=llm, sandbox_dir=tmp_path)
    assert _pools(llm) == ["vision"] and SEEN + SCENE in out


async def test_a_test_double_without_a_real_pool_list_is_not_a_pool(tmp_path, on, critic):
    llm = _llm()                                   # an AsyncMock: `critic_clients` is a MagicMock
    llm.chat_completion = AsyncMock(return_value=_vision({"scene": SCENE}))
    await tool_generate_image(prompt="a cat", llm_client=llm, sandbox_dir=tmp_path)
    assert _pools(llm) == ["vision"]


# ── the fallback ─────────────────────────────────────────────────────────────

class _Down(Exception):
    pass


async def test_a_failed_critic_look_falls_back_to_the_vision_pool(tmp_path, on, critic):
    llm = _pool_llm(answers=[_Down("all critic nodes failed"), _vision({"scene": SCENE})])
    out = await tool_generate_image(prompt="a cat in space", llm_client=llm, sandbox_dir=tmp_path)
    assert _pools(llm) == ["critic", "vision"]
    assert SEEN + SCENE in out and UNSEEN_IMAGE_NOTE not in out
    assert len(IG.RECENT_LOOK_SECONDS) == 1, "one look, one booking — the fallback is part of the same look"


async def test_both_pools_failing_keeps_the_picture_and_the_not_seen_sentence(tmp_path, on, critic):
    llm = _pool_llm(answers=[_Down("critic"), _Down("vision")])
    out = await tool_generate_image(prompt="a cat", llm_client=llm, sandbox_dir=tmp_path)
    assert _pools(llm) == ["critic", "vision"]
    assert "SUCCESS" in out and UNSEEN_IMAGE_NOTE in out and SEEN not in out
    assert list(tmp_path.glob("gen_*.png")) and len(IG.RECENT_LOOK_SECONDS) == 1


async def test_a_cancel_during_the_critic_look_unwinds_and_is_not_a_fallback(tmp_path, on, critic):
    import asyncio
    llm = _pool_llm(answers=[asyncio.CancelledError(), _vision({"scene": SCENE})])
    with pytest.raises(asyncio.CancelledError):
        await tool_generate_image(prompt="a cat", llm_client=llm, sandbox_dir=tmp_path)
    assert _pools(llm) == ["critic"], "a cancel is not a node failure: the vision pool is never asked"
    assert list(tmp_path.glob("gen_*.png")), "the picture was saved before the look (§4QP r1)"


async def test_an_unusable_critic_answer_is_not_retried_on_the_vision_pool(tmp_path, on, critic):
    llm = _pool_llm(answers=[_vision("no json here"), _vision({"scene": SCENE})])
    out = await tool_generate_image(prompt="a cat", llm_client=llm, sandbox_dir=tmp_path)
    assert _pools(llm) == ["critic"], "the fallback is for a pool that FAILED, not for a node that answered badly"
    assert UNSEEN_IMAGE_NOTE in out


# ── the arm ──────────────────────────────────────────────────────────────────

def test_the_critic_arm_reaches_a_probe_only_and_implies_the_look(monkeypatch):
    monkeypatch.delenv(IG.LOOK_ENV, raising=False)
    monkeypatch.delenv(IG.LOOK_POOL_ENV, raising=False)
    from ghost_agent.utils.logging import prompt_arm_context, request_id_context, parse_prompt_arm, PROMPT_ARMS
    assert IG.LOOK_CRITIC_ARM in PROMPT_ARMS and parse_prompt_arm(IG.LOOK_CRITIC_ARM) == IG.LOOK_CRITIC_ARM
    llm = _pool_llm()
    tok = prompt_arm_context.set(IG.LOOK_CRITIC_ARM)
    rid = request_id_context.set("probe-4qt-1")
    try:
        assert IG.render_look_enabled() is True and IG.look_on_critic(llm) is True
        request_id_context.reset(rid); rid = request_id_context.set("web-4qt-1")
        assert IG.render_look_enabled() is False and IG.look_on_critic(llm) is False
    finally:
        request_id_context.reset(rid); prompt_arm_context.reset(tok)


def test_the_render_look_arm_alone_keeps_the_vision_pool(monkeypatch):
    monkeypatch.delenv(IG.LOOK_POOL_ENV, raising=False)
    from ghost_agent.utils.logging import prompt_arm_context, request_id_context
    tok = prompt_arm_context.set(IG.LOOK_ARM)
    rid = request_id_context.set("probe-4qt-2")
    try:
        assert IG.render_look_enabled() is True and IG.look_on_critic(_pool_llm()) is False
    finally:
        request_id_context.reset(rid); prompt_arm_context.reset(tok)


# ── scope ────────────────────────────────────────────────────────────────────

async def test_the_collage_verdict_stays_on_the_vision_pool(critic):
    llm = _pool_llm(answers=[_vision({"collage": "NO", "confidence": 0.9, "evidence": "one", "scene": SCENE})])
    v = await IG.collage_verdict(llm, b"\x89PNG\r\n\x1a\nxx")
    assert v and v.get("collage") is False
    assert _pools(llm) == ["vision"], "the §4OZ verdict was judged 30/30 on the main model; the switch moves the LOOK only"


def test_the_estimate_reads_the_booked_seconds_whichever_pool_answered(on):
    IG.RECENT_LOOK_SECONDS.clear()
    assert IG._look_seconds() == IG.LOOK_DEFAULT_S
    IG.RECENT_LOOK_SECONDS.append(2.5)
    assert IG._look_seconds() == 2.5


# ── the verifier's pixel check (operator "yes", 2026-10-22): same switch, same pool ──

def _png(tmp_path):
    import io
    from PIL import Image
    p = tmp_path / "vision_0123456789ab.png"
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (200, 50, 50)).save(buf, format="PNG")
    p.write_bytes(buf.getvalue())
    return str(p)


class _VStub:
    """A client whose chat_completion records the pool it was asked for."""

    def __init__(self, critic_clients=NOVA, fail_first=False):
        self.critic_clients = critic_clients
        self.calls = []
        self.fail_first = fail_first

    async def chat_completion(self, payload, **kw):
        self.calls.append(kw)
        if self.fail_first and len(self.calls) == 1:
            raise _Down("all critic nodes failed")
        return {"choices": [{"message": {"content": '{"verdict":"CONFIRMED","confidence":0.9,"reasoning":"matches"}'}}]}


def _vpools(stub):
    return [("critic" if c.get("use_critic") else "vision" if c.get("use_vision") else "?") for c in stub.calls]


async def test_the_pixel_check_goes_to_the_critic_pool_with_the_switch(tmp_path, critic):
    from ghost_agent.core.verifier import Verifier, VerifyVerdict
    stub = _VStub()
    r = await Verifier(llm_client=stub).verify_visual(symptom="x", claim="y", after_image=_png(tmp_path))
    assert _vpools(stub) == ["critic"]
    kw = stub.calls[0]
    from ghost_agent.core import verifier as V
    assert kw.get("off_main_only") is True and kw.get("timeout") == V._VISUAL_CRITIC_TIMEOUT_S
    assert kw.get("task_label") == "verify-visual"
    assert r is not None and r.verdict == VerifyVerdict.CONFIRMED


async def test_without_the_switch_the_pixel_check_stays_on_the_vision_pool(tmp_path, monkeypatch):
    monkeypatch.delenv(IG.LOOK_POOL_ENV, raising=False)
    from ghost_agent.core.verifier import Verifier
    stub = _VStub()
    await Verifier(llm_client=stub).verify_visual(symptom="x", claim="y", after_image=_png(tmp_path))
    assert _vpools(stub) == ["vision"]


async def test_without_a_critic_pool_the_pixel_check_stays_on_the_vision_pool(tmp_path, critic):
    from ghost_agent.core.verifier import Verifier
    stub = _VStub(critic_clients=[])
    await Verifier(llm_client=stub).verify_visual(symptom="x", claim="y", after_image=_png(tmp_path))
    assert _vpools(stub) == ["vision"]


async def test_a_failed_critic_pixel_check_asks_the_vision_pool(tmp_path, critic):
    from ghost_agent.core.verifier import Verifier, VerifyVerdict
    stub = _VStub(fail_first=True)
    r = await Verifier(llm_client=stub).verify_visual(symptom="x", claim="y", after_image=_png(tmp_path))
    assert _vpools(stub) == ["critic", "vision"] and r is not None and r.verdict == VerifyVerdict.CONFIRMED


async def test_the_critic_pixel_check_runs_while_the_main_slot_is_marked_busy(tmp_path, critic):
    """The §4OT main-busy mark skips a verdict that would need the main slot;
    the critic pool is not that slot."""
    from ghost_agent.core import verifier as V
    stub = _VStub()
    tok = V.deferred_verdict_context.set({"main_busy": True})
    try:
        out = await V.Verifier(llm_client=stub)._call_llm_vision("q", [_png(tmp_path)])
        assert _vpools(stub) == ["critic"] and out.get("verdict") == "CONFIRMED"
        stub2 = _VStub(fail_first=True)
        out2 = await V.Verifier(llm_client=stub2)._call_llm_vision("q", [_png(tmp_path)])
        assert _vpools(stub2) == ["critic"] and out2 == {}, "after the critic leg fails, the main-busy mark still skips the vision leg"
    finally:
        V.deferred_verdict_context.reset(tok)


# ── §4QT r1 (fresh-eye review): the critic legs are bounded and laned like the text leg ──

async def test_the_critic_look_bounds_its_queue_and_its_request_together(tmp_path, on, critic):
    """r1 MAJOR 2: `timeout` bounded only the POST; the permit queue kept the
    client's 90 s default, so a saturated nova could stall the owner's image
    turn ~90 s before the vision fallback even started. The leg now states a
    short permit wait and a total budget equal to its timeout."""
    llm = _pool_llm()
    await tool_generate_image(prompt="a cat in space", llm_client=llm, sandbox_dir=tmp_path)
    kw = llm.chat_completion.await_args.kwargs
    assert kw.get("slot_wait") == IG.LOOK_CRITIC_SLOT_WAIT_S and 0 < IG.LOOK_CRITIC_SLOT_WAIT_S < IG.LOOK_CRITIC_TIMEOUT_S
    assert kw.get("total_budget") == IG.LOOK_CRITIC_TIMEOUT_S, "queue AND request share the 30 s; the vision fallback waits for neither"


class _Saturated(Exception):
    """What the client raises when nova only had no permit (OffMainNodeUnavailable(saturated=True) / LaneBusy)."""
    saturated = True


async def test_an_in_turn_pixel_check_states_the_in_turn_slot_wait_and_stays_foreground(tmp_path, critic):
    from ghost_agent.core import verifier as V
    stub = _VStub()
    stub.foreground_requests = 1
    await V.Verifier(llm_client=stub)._call_llm_vision("q", [_png(tmp_path)])
    kw = stub.calls[0]
    assert kw.get("slot_wait") == V._VERIFY_SLOT_WAIT_S
    assert not kw.get("is_background"), "inside the owner's turn a background call would park on its own request"


async def test_a_deferred_pixel_check_is_background_laned_and_waits_the_deferred_slot_wait(tmp_path, critic):
    """r1 MAJOR 1: the leg was foreground, un-laned and queued the default
    wait — every post-reply pixel check counted as a live user (idle tick,
    early re-warm held) and a burst could take all four nova permits from the
    in-turn render look."""
    from ghost_agent.core import verifier as V
    stub = _VStub()
    stub.foreground_requests = 0
    tok = V.deferred_verdict_context.set({"deferred": True})
    try:
        out = await V.Verifier(llm_client=stub)._call_llm_vision("q", [_png(tmp_path)])
    finally:
        V.deferred_verdict_context.reset(tok)
    kw = stub.calls[0]
    assert _vpools(stub) == ["critic"] and out.get("verdict") == "CONFIRMED"
    assert kw.get("slot_wait") == V._DEFERRED_SLOT_WAIT_S
    assert kw.get("is_background") is True
    assert kw.get("task_label") == "deferred-verdict", "the client's background lane keys on this label (llm.py)"
    assert "timeout" in kw and kw["timeout"] == V._VISUAL_CRITIC_TIMEOUT_S, "the check's own timeout, not the fallback's"


async def test_a_deferred_pixel_check_takes_the_deferred_lane(tmp_path, critic, monkeypatch):
    from ghost_agent.core import verifier as V
    entered = []
    class _Lane:
        async def __aenter__(self): entered.append(1); return self
        async def __aexit__(self, *a): return False
    monkeypatch.setattr(V, "_critic_lane", _Lane)
    stub = _VStub()
    tok = V.deferred_verdict_context.set({"deferred": True})
    try:
        await V.Verifier(llm_client=stub)._call_llm_vision("q", [_png(tmp_path)])
        assert entered == [1], "a deferred pixel check must hold a lane permit (DEFERRED_CRITIC_CALLS of nova's four)"
    finally:
        V.deferred_verdict_context.reset(tok)
    entered.clear()
    await V.Verifier(llm_client=_VStub())._call_llm_vision("q", [_png(tmp_path)])
    assert entered == [], "an in-turn pixel check does not queue in the deferred lane"


async def test_a_deferred_pixel_check_that_finds_nova_busy_is_skipped_not_rerun_on_the_main_slot(tmp_path, critic):
    from ghost_agent.core import verifier as V
    class _Busy(_VStub):
        async def chat_completion(self, payload, **kw):
            self.calls.append(kw); raise _Saturated("no Nova permit")
    stub = _Busy()
    mark = {"deferred": True}
    tok = V.deferred_verdict_context.set(mark)
    try:
        out = await V.Verifier(llm_client=stub)._call_llm_vision("q", [_png(tmp_path)])
    finally:
        V.deferred_verdict_context.reset(tok)
    assert out == {} and _vpools(stub) == ["critic"], "nova only busy: no verdict, and NOT the vision pool (the main slot)"
    assert mark.get("nova_busy") is True, "the verdict remembers it so a parallel call stops too"


async def test_a_verdict_that_already_found_nova_busy_makes_no_pixel_call(tmp_path, critic, monkeypatch):
    """Stops BEFORE the lane: a verdict that already found Nova busy must not
    queue for a lane permit (up to LANE_WAIT_S) only to raise inside it."""
    from ghost_agent.core import verifier as V
    entered = []
    class _Lane:
        async def __aenter__(self): entered.append(1); return self
        async def __aexit__(self, *a): return False
    monkeypatch.setattr(V, "_critic_lane", _Lane)
    stub = _VStub()
    tok = V.deferred_verdict_context.set({"nova_busy": True})
    try:
        out = await V.Verifier(llm_client=stub)._call_llm_vision("q", [_png(tmp_path)])
    finally:
        V.deferred_verdict_context.reset(tok)
    assert out == {} and stub.calls == []
    assert entered == [], "the busy mark is read before the lane, not inside it"


async def test_an_in_turn_pixel_check_still_falls_to_the_vision_pool_when_nova_is_busy(tmp_path, critic):
    """Inside the turn the owner is waiting and the main slot is theirs anyway:
    the §4QT fallback stays (bounded by `_bounded_fallback_kwargs`)."""
    from ghost_agent.core import verifier as V
    class _Busy(_VStub):
        async def chat_completion(self, payload, **kw):
            if not self.calls:
                self.calls.append(kw)
                raise _Saturated("no Nova permit")
            return await _VStub.chat_completion(self, payload, **kw)
    stub = _Busy()
    out = await V.Verifier(llm_client=stub)._call_llm_vision("q", [_png(tmp_path)])
    assert _vpools(stub) == ["critic", "vision"] and out.get("verdict") == "CONFIRMED"
