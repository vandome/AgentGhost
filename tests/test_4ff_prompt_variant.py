"""§4FF / §4JG pins — the compiled system prompt is RETIRED.

§4FF shipped SYSTEM_PROMPT_COMPILED (the same policy compiled to 12 rules)
behind a probe-only variant switch so `scripts/if_bench.py` could pair it
against SYSTEM_PROMPT. The full banded bank (2026-09-20, 95 pairs) measured
NO DIFFERENCE — 87/95 vs 87/95, paired McNemar p = 1.0, easy/tool/deep at
ceiling for both — so on 2026-09-21 (§4JG) the variant, its selector branch
and the route header were removed. These pins hold the deletion (pin-the-
deletion): the constant is gone, no body or header can select anything but
SYSTEM_PROMPT, and the bench refuses a variant that does not exist.
"""
import pytest

from ghost_agent.core import agent as agent_mod
from ghost_agent.core import prompts as prompts_mod
from ghost_agent.core.prompts import SYSTEM_PROMPT
from ghost_agent.utils.logging import request_id_context


def test_the_compiled_prompt_no_longer_exists():
    assert not hasattr(prompts_mod, "SYSTEM_PROMPT_COMPILED")


@pytest.mark.parametrize("body", [None, {}, {"_prompt_variant": "compiled"},
                                  {"_prompt_variant": "control"}, {"_prompt_variant": "anything"}])
def test_nothing_in_the_body_selects_another_prompt(body):
    """Executed through the real selector, under a PROBE request id (the
    only context the old seam honoured): the answer is SYSTEM_PROMPT."""
    class Stub:
        pass
    tok = request_id_context.set("probe-ifb-1")
    try:
        assert agent_mod.GhostAgent._select_system_prompt(Stub(), body) is SYSTEM_PROMPT
    finally:
        request_id_context.reset(tok)


def test_the_live_turn_still_selects_through_the_one_seam():
    """The reattachment point for a future A/B is the method, and the live
    turn site must keep going through it (a direct SYSTEM_PROMPT read at
    the turn site would silently bypass any future variant)."""
    import ast, inspect
    src = inspect.getsource(agent_mod)
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", None) == "_select_system_prompt"]
    assert calls, "the live turn no longer selects its prompt through _select_system_prompt"




