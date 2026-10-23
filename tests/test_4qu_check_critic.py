"""§4QU — the verifier's pixel check on its own switch (2026-10-23).

§4QT moved the render look AND the pixel check to the critic pool under one
switch; the replay priced them apart: the look on nova is a worse look, the
check on nova is as good and spares the owner's NEXT message the ~30 s
re-prefill. `GHOST_VISUAL_CHECK_POOL=critic` (probe arm `check_critic`) moves
the check alone; the look switch still implies both.
"""
from __future__ import annotations

import pytest

import ghost_agent.tools.image_gen as IG
from ghost_agent.tools.image_gen import tool_generate_image
from tests.test_4qp_render_look import on  # noqa: F401 — fixture: the look is ON
from tests.test_4qt_look_critic import NOVA, _VStub, _png, _pool_llm, _pools, _vpools

pytestmark = pytest.mark.asyncio


@pytest.fixture
def check(monkeypatch):
    monkeypatch.delenv(IG.LOOK_POOL_ENV, raising=False)
    monkeypatch.setenv(IG.VISUAL_CHECK_POOL_ENV, "critic")


@pytest.fixture(autouse=True)
def _no_look_switch(monkeypatch):
    monkeypatch.delenv(IG.LOOK_POOL_ENV, raising=False)
    monkeypatch.delenv(IG.VISUAL_CHECK_POOL_ENV, raising=False)


async def test_without_either_switch_the_check_stays_on_the_vision_pool(tmp_path):
    from ghost_agent.core.verifier import Verifier
    stub = _VStub()
    await Verifier(llm_client=stub).verify_visual(symptom="x", claim="y", after_image=_png(tmp_path))
    assert _vpools(stub) == ["vision"]


async def test_the_check_switch_moves_the_pixel_check_to_the_critic_pool(tmp_path, check):
    from ghost_agent.core import verifier as V
    stub = _VStub()
    r = await V.Verifier(llm_client=stub).verify_visual(symptom="x", claim="y", after_image=_png(tmp_path))
    assert _vpools(stub) == ["critic"]
    kw = stub.calls[0]
    assert kw.get("off_main_only") is True and kw.get("task_label") == "verify-visual"
    assert kw.get("timeout") == V._VISUAL_CRITIC_TIMEOUT_S and V._VISUAL_CRITIC_TIMEOUT_S > IG.LOOK_CRITIC_TIMEOUT_S, \
        "the check answers with more tokens and up to two pictures: its own bound, not the look's"
    assert r is not None and r.verdict == V.VerifyVerdict.CONFIRMED


async def test_the_check_switch_leaves_the_look_on_the_vision_pool(tmp_path, on, check):
    """The whole point: the look stays where it is accurate."""
    llm = _pool_llm()
    await tool_generate_image(prompt="a cat in space", llm_client=llm, sandbox_dir=tmp_path)
    assert _pools(llm) == ["vision"]
    assert IG.look_on_critic(llm) is False and IG.visual_check_on_critic(llm) is True


@pytest.mark.parametrize("value", ["1", "true", "nova", "Critic ", "CRITIC"])
async def test_only_the_word_critic_moves_the_check(tmp_path, monkeypatch, value):
    monkeypatch.setenv(IG.VISUAL_CHECK_POOL_ENV, value)
    from ghost_agent.core.verifier import Verifier
    stub = _VStub()
    await Verifier(llm_client=stub).verify_visual(symptom="x", claim="y", after_image=_png(tmp_path))
    assert _vpools(stub) == (["critic"] if value.strip().lower() == "critic" else ["vision"])


async def test_the_look_switch_still_implies_the_check(tmp_path, monkeypatch):
    monkeypatch.setenv(IG.LOOK_POOL_ENV, "critic")
    from ghost_agent.core.verifier import Verifier
    stub = _VStub()
    await Verifier(llm_client=stub).verify_visual(symptom="x", claim="y", after_image=_png(tmp_path))
    assert _vpools(stub) == ["critic"], "§4QT's switch keeps its meaning: look AND check"


async def test_without_a_critic_pool_the_check_switch_keeps_the_vision_pool(tmp_path, check):
    from ghost_agent.core.verifier import Verifier
    for clients in ([], None, "nova"):
        stub = _VStub(critic_clients=clients)
        await Verifier(llm_client=stub).verify_visual(symptom="x", claim="y", after_image=_png(tmp_path))
        assert _vpools(stub) == ["vision"], clients


def test_the_check_arm_reaches_a_probe_only_and_does_not_imply_the_look(monkeypatch):
    monkeypatch.delenv(IG.LOOK_ENV, raising=False)
    from ghost_agent.utils.logging import prompt_arm_context, request_id_context, parse_prompt_arm, PROMPT_ARMS
    assert IG.CHECK_CRITIC_ARM in PROMPT_ARMS and parse_prompt_arm(IG.CHECK_CRITIC_ARM) == IG.CHECK_CRITIC_ARM
    llm = _pool_llm()
    tok = prompt_arm_context.set(IG.CHECK_CRITIC_ARM)
    rid = request_id_context.set("probe-4qu-1")
    try:
        assert IG.visual_check_on_critic(llm) is True
        assert IG.look_on_critic(llm) is False
        assert IG.render_look_enabled() is False, "the check arm says nothing about the look (production has its own switch)"
        request_id_context.reset(rid); rid = request_id_context.set("web-4qu-1")
        assert IG.visual_check_on_critic(llm) is False, "an arm reaches a probe only"
    finally:
        request_id_context.reset(rid); prompt_arm_context.reset(tok)


async def test_a_failed_critic_check_still_asks_the_vision_pool_in_turn(tmp_path, check):
    from ghost_agent.core.verifier import Verifier
    stub = _VStub(fail_first=True)
    await Verifier(llm_client=stub).verify_visual(symptom="x", claim="y", after_image=_png(tmp_path))
    assert _vpools(stub) == ["critic", "vision"]


# ── §4QW: a borderline judge — the claim is the description only, a REFUTED is asked twice ──

def test_the_pixel_judge_is_handed_the_description_without_the_markdown_line_or_the_footer():
    from ghost_agent.core.verifier import _claim_for_pixels
    claim = ("![generated image](/api/download/gen_7d530550.png)\n\n"
             "A cartoon illustration of a grey kitten next to a ringed planet.\n\n"
             "---\n**Verifier note:** The cat is not in space.")
    assert _claim_for_pixels(claim) == "A cartoon illustration of a grey kitten next to a ringed planet."
    assert _claim_for_pixels("") == "" and _claim_for_pixels(None) == ""


async def test_verify_visual_sends_the_stripped_claim(tmp_path, check):
    from ghost_agent.core import verifier as V
    import json as _json
    class _Rec(_VStub):
        async def chat_completion(self, payload, **kw):
            self.text = _json.dumps(payload, ensure_ascii=False)
            return await _VStub.chat_completion(self, payload, **kw)
    stub = _Rec()
    await V.Verifier(llm_client=stub).verify_visual(symptom="a cat in space",
                                                     claim="![generated image](/api/download/x.png)\n\nA grey kitten on purple.",
                                                     after_image=_png(tmp_path))
    assert "A grey kitten on purple." in stub.text and "![generated image]" not in stub.text


def _answers(*verdicts):
    return [{"choices": [{"message": {"content": '{"verdict":"%s","confidence":0.9,"reasoning":"r"}' % v}}]} for v in verdicts]


class _Seq(_VStub):
    def __init__(self, answers, **kw):
        super().__init__(**kw); self.answers = list(answers)
    async def chat_completion(self, payload, **kw):
        self.calls.append(kw); return self.answers.pop(0)


async def test_a_critic_refuted_that_does_not_repeat_is_not_a_verdict(tmp_path, check):
    from ghost_agent.core import verifier as V
    stub = _Seq(_answers("REFUTED", "CONFIRMED"))
    out = await V.Verifier(llm_client=stub)._call_llm_vision("q", [_png(tmp_path)])
    assert _vpools(stub) == ["critic", "critic"] and out.get("verdict") == "CONFIRMED"


async def test_a_critic_refuted_that_repeats_stands(tmp_path, check):
    from ghost_agent.core import verifier as V
    first = {"choices": [{"message": {"content": '{"verdict":"REFUTED","confidence":0.9,"reasoning":"first"}'}}]}
    second = {"choices": [{"message": {"content": '{"verdict":"REFUTED.","confidence":0.6,"reasoning":"second"}'}}]}
    stub = _Seq([first, second])
    out = await V.Verifier(llm_client=stub)._call_llm_vision("q", [_png(tmp_path)])
    assert _vpools(stub) == ["critic", "critic"]
    assert out.get("verdict") == "REFUTED" and out.get("confidence") == 0.9 and out.get("reasoning") == "first", \
        "a repeated REFUTED stands as the FIRST answer; 'REFUTED.' counts as a repeat (the normaliser's reading)"


async def test_the_verdict_word_is_read_as_the_result_builder_reads_it(tmp_path, check):
    """r1 MAJOR 1: 'REFUTED.', 'REFUTE', ' REFUTED (partially)' are REFUTED to
    _build_verify_result; an exact-string compare let them skip the re-ask."""
    from ghost_agent.core import verifier as V
    assert V._verdict_word({"verdict": "REFUTED."}) == "REFUTED"
    assert V._verdict_word({"verdict": "refute"}) == "REFUTED"
    assert V._verdict_word({"verdict": " REFUTED (partially)"}) == "REFUTED"
    assert V._verdict_word({"verdict": "CONFIRMED"}) == "CONFIRMED" and V._verdict_word({}) == "" and V._verdict_word(None) == ""
    first = {"choices": [{"message": {"content": '{"verdict":"REFUTED.","confidence":0.9,"reasoning":"first"}'}}]}
    stub = _Seq([first, _answers("CONFIRMED")[0]])
    out = await V.Verifier(llm_client=stub)._call_llm_vision("q", [_png(tmp_path)])
    assert _vpools(stub) == ["critic", "critic"] and out.get("verdict") == "CONFIRMED"


async def test_a_failed_second_ask_skips_the_check_and_never_asks_the_main_slot(tmp_path, check):
    from ghost_agent.core import verifier as V
    class _Fail(_VStub):
        async def chat_completion(self, payload, **kw):
            self.calls.append(kw)
            if len(self.calls) == 1:
                return _answers("REFUTED")[0]
            raise _Down("nova 502")
    from tests.test_4qt_look_critic import _Down
    stub = _Fail()
    out = await V.Verifier(llm_client=stub)._call_llm_vision("q", [_png(tmp_path)])
    assert _vpools(stub) == ["critic", "critic"] and out == {}


async def test_an_unusable_second_answer_skips_the_check(tmp_path, check):
    from ghost_agent.core import verifier as V
    stub = _Seq([_answers("REFUTED")[0], {"choices": [{"message": {"content": "I cannot tell."}}]}])
    out = await V.Verifier(llm_client=stub)._call_llm_vision("q", [_png(tmp_path)])
    assert _vpools(stub) == ["critic", "critic"] and out == {}


async def test_a_deferred_re_ask_takes_the_lane_again_and_honours_the_busy_mark(tmp_path, check, monkeypatch):
    """r1 MINOR 3/9: the second ask re-enters the real deferred lane; a busy
    mark set between the asks stops it (nova only busy → skipped, not main)."""
    from ghost_agent.core import verifier as V
    entered = []
    class _Lane:
        async def __aenter__(self): entered.append(1); return self
        async def __aexit__(self, *a): return False
    monkeypatch.setattr(V, "_critic_lane", _Lane)
    mark = {"deferred": True}
    stub = _Seq([_answers("REFUTED")[0], _answers("CONFIRMED")[0]])
    tok = V.deferred_verdict_context.set(mark)
    try:
        out = await V.Verifier(llm_client=stub)._call_llm_vision("q", [_png(tmp_path)])
        assert _vpools(stub) == ["critic", "critic"] and out.get("verdict") == "CONFIRMED"
        assert entered == [1, 1], "the second ask holds a lane permit too (r2 MINOR 1)"
        class _Marks(_VStub):
            async def chat_completion(self, payload, **kw):
                self.calls.append(kw); mark["nova_busy"] = True
                return _answers("REFUTED")[0]
        stub2 = _Marks()
        out2 = await V.Verifier(llm_client=stub2)._call_llm_vision("q", [_png(tmp_path)])
        assert _vpools(stub2) == ["critic"] and out2 == {}, "the busy mark between the asks stops the re-ask; no main-slot call"
    finally:
        V.deferred_verdict_context.reset(tok)


async def test_a_reply_that_is_only_the_image_line_is_not_judged(tmp_path, check):
    from ghost_agent.core.verifier import Verifier
    stub = _VStub()
    r = await Verifier(llm_client=stub).verify_visual(symptom="a cat", claim="![generated image](/api/download/x.png)",
                                                       after_image=_png(tmp_path))
    assert r is None and stub.calls == []


async def test_the_likeness_prompt_is_handed_the_stripped_claim(tmp_path, check):
    import json as _json
    from ghost_agent.core import verifier as V
    class _Rec(_VStub):
        async def chat_completion(self, payload, **kw):
            self.text = _json.dumps(payload, ensure_ascii=False); return await _VStub.chat_completion(self, payload, **kw)
    stub = _Rec()
    await V.Verifier(llm_client=stub).verify_visual(symptom="a photo of them", claim="![generated image](/x.png)\n\nTwo people on a bench.",
                                                     after_image=_png(tmp_path), before_image=_png(tmp_path), likeness_subjects="A, B")
    assert "Two people on a bench." in stub.text and "![generated image]" not in stub.text


async def test_a_critic_confirmed_is_asked_once(tmp_path, check):
    from ghost_agent.core import verifier as V
    stub = _Seq(_answers("CONFIRMED"))
    out = await V.Verifier(llm_client=stub)._call_llm_vision("q", [_png(tmp_path)])
    assert _vpools(stub) == ["critic"] and out.get("verdict") == "CONFIRMED"


async def test_the_main_slot_leg_is_never_re_asked(tmp_path, monkeypatch):
    """Without the switch the check runs on the main slot; a second call there
    would evict the owner's conversation — the re-ask is the critic pool's only."""
    monkeypatch.delenv(IG.VISUAL_CHECK_POOL_ENV, raising=False)
    from ghost_agent.core import verifier as V
    stub = _Seq(_answers("REFUTED", "CONFIRMED"))
    out = await V.Verifier(llm_client=stub)._call_llm_vision("q", [_png(tmp_path)])
    assert _vpools(stub) == ["vision"] and out.get("verdict") == "REFUTED"
