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
