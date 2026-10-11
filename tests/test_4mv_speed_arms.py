"""§4MV (2026-10-09): the two operator speed calls, measured with probe-only
arms and decided by the operator — the tool-description trim is baked into
the tool sources; memory placement is unchanged; the per-request block's
churning counters are rendered as bands. The probe-arm plumbing remains (no
arms defined)."""
from __future__ import annotations

import pytest

from ghost_agent.utils import logging as L


# ── the probe-only arm switch (no arms defined now) ───────────────────

@pytest.mark.parametrize("raw", ["hyd_tail", "tools_trim", "anything", None])
def test_no_arm_is_defined_after_the_decision(raw):
    assert L.parse_prompt_arm(raw) == ""


def test_an_arm_never_reaches_a_production_request(monkeypatch):
    monkeypatch.setattr(L, "PROMPT_ARMS", frozenset({"x"}))
    for rid, want in (("probe-abc", "x"), ("abc123", "")):
        t1 = L.request_id_context.set(rid)
        t2 = L.prompt_arm_context.set("x")
        try:
            assert L.prompt_arm() == want
        finally:
            L.prompt_arm_context.reset(t2)
            L.request_id_context.reset(t1)


# ── the per-request block stops churning ──────────────────────────────

# ── the trim is in the sources ────────────────────────────────────────

def test_the_trimmed_tool_block_is_what_production_serves():
    """The shipped trim (−1,589 tokens of the measured T1's −1,759): tool names, parameter sets and
    required lists unchanged; the description prose shorter."""
    import json
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    browser = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "browser")
    text = json.dumps(browser)
    assert ".last_url" not in text and "WebOS" not in text


def test_the_seed_parameter_keeps_its_caveat():
    """r2 review: the trim dropped "an edit's or a `subjects` render's result
    does not [record a seed]" — the model would quote a seed that is not there."""
    import json
    from unittest.mock import MagicMock
    from ghost_agent.tools.registry import get_active_tool_definitions
    ctx = MagicMock()
    ctx.llm_client.image_gen_clients = ["http://gpu"]
    tools = get_active_tool_definitions(ctx)
    img = next(t for t in tools if t["function"]["name"] == "image_generation")
    assert "render's result does not" in json.dumps(img["function"]["parameters"]["properties"]["seed"])


# ── §4MW: a candidate rule under test rides only an operator probe ─────

def test_a_probe_rule_reaches_only_a_probe_request():
    for rid, want in (("probe-x", "say only what the evidence states"), ("abc123", "")):
        t1 = L.request_id_context.set(rid)
        t2 = L.probe_rule_context.set("say only what the evidence states")
        try:
            assert L.probe_rule() == want
        finally:
            L.probe_rule_context.reset(t2)
            L.request_id_context.reset(t1)


def test_the_route_sets_a_probe_rule_only_inside_the_probe_branch():
    import ast, inspect, textwrap
    from ghost_agent.api import routes
    src = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(routes))))
    i_reset = src.index("probe_rule_context.set('')")
    i_probe = src.index("== ORIGIN_PROBE:", i_reset)
    i_set = src.index("probe_rule_context.set(str(request.headers.get('X-Ghost-Probe-Rule') or '')[:800])")
    assert i_reset < i_probe < i_set
