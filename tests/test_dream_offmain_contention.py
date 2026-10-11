"""Contention guard: dream background LLM calls must never fall
back onto the single MAIN inference slot when the off-main (worker/critic/
coding) pool fails.

`LLMClient.chat_completion(..., is_background=True)` only parks on the
foreground slot while an off-main pool is CONFIGURED; when that pool FAILS
mid-call the request historically fell back to the main 35B model — so a
worker (Nova) hiccup during idle self-play landed unbounded generations
(max_tokens=16384, no timeout) on the main model while the operator was
chatting. The fix (2026-07-22) passes `off_main_only=True` on every
background dream call, turning that fallback into an
`OffMainNodeUnavailable` raise that each call site's existing graceful
catch absorbs (skip / retry next cycle).

These tests pin the contract two ways:
  1. AST-level: every `chat_completion(...)` call in core/dream.py that
     passes `is_background=` also passes `off_main_only=` (True for
     hardcoded-background sites, mirroring `is_background` for the
     conditional user-triggered/background sites) plus a `timeout=`.
  2. Behavioral: a recording mock client sees `off_main_only=True`, and an
     `OffMainNodeUnavailable` raise degrades gracefully instead of
     propagating.
§4NM: the lesson-extraction site left with synthetic self-play; its two
behavioural tests were removed.
"""

import ast
import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest

import ghost_agent.core.dream as dream_mod
from ghost_agent.core.dream import Dreamer
from ghost_agent.core.llm import OffMainNodeUnavailable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _background_chat_calls():
    """Yield (lineno, keywords_by_name) for every chat_completion /
    stream_chat_completion call in dream.py that passes is_background=."""
    tree = ast.parse(inspect.getsource(dream_mod))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "attr", None) or getattr(func, "id", None)
        if name not in ("chat_completion", "stream_chat_completion"):
            continue
        kws = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        if "is_background" in kws:
            yield node.lineno, kws


# ---------------------------------------------------------------------------
# 1. Source-level contract over ALL background call sites
# ---------------------------------------------------------------------------

def test_all_background_calls_found():
    """Sanity guard for the AST scan itself: dream.py currently has 2
    background chat_completion call sites (graph merge and manifest
    backfill; §4NM removed synthetic self-play's lesson extraction,
    challenge generation, reference repair and validator repair sites).
    If this drops the scanner is broken; if it grows, the new site is
    covered by the assertions below."""
    calls = list(_background_chat_calls())
    assert len(calls) >= 2, (
        f"expected >=2 background chat_completion sites in dream.py, "
        f"found {len(calls)} at lines {[ln for ln, _ in calls]}"
    )


def test_every_background_call_passes_off_main_only():
    missing = [
        lineno for lineno, kws in _background_chat_calls()
        if "off_main_only" not in kws
    ]
    assert not missing, (
        f"chat_completion(is_background=...) without off_main_only= at "
        f"dream.py lines {missing} — a worker-pool failure there falls "
        f"back onto the MAIN model and dogpiles the foreground slot"
    )


def test_off_main_only_value_matches_background_mode():
    """Hardcoded is_background=True sites must pass off_main_only=True.
    Conditional sites (is_background=is_background) must mirror the flag
    (off_main_only=is_background) — or pass literal True — so a
    USER-TRIGGERED foreground run keeps its main-model fallback while
    background cycles degrade. (§4NM: dream.py has no conditional site
    left; the rule stays for any new one.)"""
    bad = []
    for lineno, kws in _background_chat_calls():
        bg = kws["is_background"]
        omo = kws["off_main_only"]
        if isinstance(bg, ast.Constant) and bg.value is True:
            ok = isinstance(omo, ast.Constant) and omo.value is True
        elif isinstance(bg, ast.Name):
            ok = (isinstance(omo, ast.Name) and omo.id == bg.id) or (
                isinstance(omo, ast.Constant) and omo.value is True
            )
        else:
            ok = False
        if not ok:
            bad.append(lineno)
    assert not bad, (
        f"off_main_only value does not match is_background mode at "
        f"dream.py lines {bad}"
    )


def test_every_background_call_has_timeout():
    """No background dream call may run unbounded: a wedged-but-connected
    node must not pin a generation (some sites use max_tokens=16384)."""
    missing = [
        lineno for lineno, kws in _background_chat_calls()
        if "timeout" not in kws
    ]
    assert not missing, (
        f"chat_completion(is_background=...) without timeout= at "
        f"dream.py lines {missing}"
    )


# ---------------------------------------------------------------------------
# 2. Behavioral: recording mock sees the kwargs; OffMainNodeUnavailable
#    degrades instead of propagating.
# ---------------------------------------------------------------------------

def _make_context(llm_client):
    ctx = MagicMock()
    ctx.llm_client = llm_client
    ctx.memory_system = MagicMock()
    return ctx


def _llm_reply(content: str):
    return {"choices": [{"message": {"content": content}}]}


@pytest.mark.asyncio
async def test_graph_merge_call_is_off_main_only():
    llm = MagicMock()
    llm.chat_completion = AsyncMock(
        return_value=_llm_reply('{"same_entity": [1]}'))
    ctx = _make_context(llm)
    graph = MagicMock()
    graph.propose_merge_candidates = MagicMock(return_value=[
        {"old_node": "new", "new_node": "news", "kind": "fuzzy"},
    ])
    graph.execute_graph_compression = MagicMock(return_value=1)
    ctx.graph_memory = graph

    dreamer = Dreamer(ctx)
    applied = await dreamer._compress_graph_nodes("test-model")

    assert applied == 1
    kwargs = llm.chat_completion.call_args.kwargs
    assert kwargs.get("is_background") is True
    assert kwargs.get("off_main_only") is True, (
        "graph-merge worker call must not fall back to the main model")
    assert kwargs.get("timeout") is not None


@pytest.mark.asyncio
async def test_graph_merge_degrades_on_off_main_unavailable():
    """Worker pool down: the fuzzy-merge confirmation is SKIPPED (no merge
    applied) and the exception never propagates — instead of the old
    behavior of re-running the prompt on the main model."""
    llm = MagicMock()
    llm.chat_completion = AsyncMock(
        side_effect=OffMainNodeUnavailable("all off-main nodes failed"))
    ctx = _make_context(llm)
    graph = MagicMock()
    graph.propose_merge_candidates = MagicMock(return_value=[
        {"old_node": "new", "new_node": "news", "kind": "fuzzy"},
    ])
    graph.execute_graph_compression = MagicMock(return_value=0)
    ctx.graph_memory = graph

    dreamer = Dreamer(ctx)
    applied = await dreamer._compress_graph_nodes("test-model")

    assert applied == 0
    graph.execute_graph_compression.assert_not_called()
