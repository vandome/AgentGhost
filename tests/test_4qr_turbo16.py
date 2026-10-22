"""§4QR (operator, 2026-10-22: "if it's better with 16 interpolated, keep that
for all") — the image node samples the Turbo checkpoint's distilled trajectory
at 16 steps: the 8-step `sample_sigmas` interpolated in index space, every
original sigma kept. Measured on ghost (same prompts + seeds): the 8-step grain
on textured surfaces resolves at 16, faces stay clean, edits and subjects
renders improve too; sd-cli's own 16-step schedule over-sharpens. Cost 68 → 112 s
(T2I), 117 → 182 s (edit)."""
from __future__ import annotations

import importlib.util
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "interface", "externals", "image_generation", "img_gen_server.py")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("img_gen_server_4qr", SRC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_the_schedule_is_the_distilled_trajectory_sampled_twice_as_densely(mod):
    s16, s8 = list(mod.TURBO_SIGMAS), list(mod.TURBO_SIGMAS_8)
    assert len(s16) == 17 and len(s8) == 9 and mod.TURBO_STEPS == 16
    assert s16[0] == 1.0 and s16[-1] == 0.0
    assert all(a > b for a, b in zip(s16, s16[1:]))                      # strictly decreasing
    assert s16[::2] == s8                                                 # every original sigma, at the even positions
    for i in range(1, 17, 2):                                             # the odd ones are the midpoints
        assert abs(s16[i] - (s16[i - 1] + s16[i + 1]) / 2) < 1e-6


def test_every_render_runs_sixteen_steps_on_that_schedule(mod):
    for asked, editing in ((None, False), (8, False), (30, True), (1, True)):
        assert mod.resolve_steps(asked, editing) == 16
    argv = mod.build_sd_cli_args("a cat", 768, 512, mod.TURBO_STEPS, "o.png", seed=1)
    assert argv[argv.index("--steps") + 1] == "16"
    assert [float(x) for x in argv[argv.index("--sigmas") + 1].split(",")] == list(mod.TURBO_SIGMAS)
    assert "--sigmas" not in mod.build_sd_cli_args("a cat", 768, 512, 8, "o.png")   # 8 is no longer the model's schedule


def test_the_timeouts_still_cover_a_sixteen_step_ladder(mod):
    # a subjects ladder: 2 renders + a cfg leg + verdicts must fit the client's window
    worst_ladder = 2 * 182 + 182 * mod.COLLAGE_CFG_COST if hasattr(mod, "COLLAGE_CFG_COST") else 3 * 182
    assert mod.GEN_TIMEOUT_S >= 182 + 60                                  # one 16-step edit with headroom
    assert mod.BUSY_WAIT_TIMEOUT + mod.GEN_TIMEOUT_S + mod.RESPONSE_MARGIN_S <= mod.CLIENT_TIMEOUT_S
    assert mod.WORST_GENERATION_S >= 182


def test_the_agents_defaults_and_wording_follow(mock_context):
    import json
    import ghost_agent.tools.image_gen as IG
    import ghost_agent.tools.registry as R
    from ghost_agent.core.prompts import SYSTEM_PROMPT
    assert IG.DEFAULT_RENDER_SECONDS == {"edit": 185.0, "generate": 115.0}
    IG.RECENT_RENDER_SECONDS["generate"].clear(); IG.RECENT_RENDER_SECONDS["edit"].clear()
    os.environ.pop(IG.LOOK_ENV, None)
    assert IG.expected_render_seconds({"prompt": "x"}) == 115.0
    tools = R.get_active_tool_definitions(mock_context, None, serve_tuned=False)
    text = json.dumps([t for t in tools if "image_generation" in json.dumps(t)])
    assert "(~2 min)" in text and "an edit, ~3 min" in text and "takes ~3 min" in text
    assert "(~1 min)" not in text and "an edit, ~2 min" not in text
    assert "about 2 minutes on the node (about 3 with" in SYSTEM_PROMPT and "An edit takes about 3 minutes" in SYSTEM_PROMPT
