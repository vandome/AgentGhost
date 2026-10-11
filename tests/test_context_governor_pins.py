"""§ context slice R1 (2026-08-19, lens C sweep) — Driven behavioral pins for the 8 SURVIVED mutants of the CX1C sweep.
Each pin: exact input -> observable. Must PASS on pristine, FAIL on its mutant.
Lives OUTSIDE tests/ on purpose (auditor sandbox only, never shipped)."""
import asyncio
import os
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from ghost_agent.core.agent import GhostAgent, GhostContext, TurnState
from ghost_agent.core.strikes import StrikeLedger


def _tok_patch(monkeypatch):
    import ghost_agent.core.agent as agent_mod
    monkeypatch.setattr(agent_mod, "estimate_tokens", lambda t: len(str(t)) // 4)


def _prune_agent():
    ag = GhostAgent.__new__(GhostAgent)
    ag.context = MagicMock()
    ag.context.memory_system = None
    ag.context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "SUMMARY"}}]})
    return ag


# ── P6: recent-window must not START on an orphaned tool result ──────
@pytest.mark.asyncio
async def test_pin_P6_recent_window_pulls_in_tool_caller(monkeypatch):
    _tok_patch(monkeypatch)
    ag = _prune_agent()
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "goal " + "g" * 1200},
        {"role": "assistant", "content": "A1CALLER plain text " + "a" * 1200},
        {"role": "tool", "name": "fs", "content": "ok short"},
        {"role": "tool", "name": "fs", "content": "T2 " + "t" * 1200},
        {"role": "assistant", "content": "a2 " + "b" * 1200},
        {"role": "tool", "name": "fs", "content": "T3 " + "c" * 1200},
        {"role": "assistant", "content": "a3 " + "d" * 1200},
        {"role": "tool", "name": "fs", "content": "T4 " + "e" * 1200},
        {"role": "assistant", "content": "a4 " + "f" * 1200},
    ]
    out = await ag._prune_context(msgs, max_tokens=500, model="m")
    # pristine: pull-in walks recent_start 3->2->1, middle empty, caller kept verbatim
    assert any("A1CALLER" in str(m.get("content", "")) for m in out), \
        "orphaned-tool pull-in dead: caller was summarized away"


# ── P8: anchored findings must survive the prune ─────────────────────
@pytest.mark.asyncio
async def test_pin_P8_anchored_finding_survives(monkeypatch):
    _tok_patch(monkeypatch)
    ag = _prune_agent()
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "goal " + "g" * 1200},
        {"role": "assistant", "content": "root cause FINDMARKER frobnicator broken " + "a" * 400},
        {"role": "tool", "name": "fs", "content": "tx " + "t" * 1200},
        {"role": "assistant", "content": "r1 " + "b" * 1200},
        {"role": "assistant", "content": "r2 " + "c" * 1200},
        {"role": "assistant", "content": "r3 " + "d" * 1200},
        {"role": "assistant", "content": "r4 " + "e" * 1200},
        {"role": "assistant", "content": "r5 " + "f" * 1200},
        {"role": "assistant", "content": "r6 " + "h" * 1200},
    ]
    out = await ag._prune_context(msgs, max_tokens=500, model="m")
    marked = [m for m in out if "FINDMARKER" in str(m.get("content", ""))]
    assert marked and any("[ANCHORED]" in str(m.get("content", "")) for m in marked), \
        "anchoring dead: middle finding vanished from pruned context"


# ── C1: pinned stable block must be the PREFIX of the first user msg ─
def test_pin_C1_stable_block_is_prefix():
    msgs = [{"role": "user", "content": "the goal"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "latest"}]
    out = GhostAgent._compose_injection(msgs, "STABLEBLOCK", "DYN", pin=True)
    assert out[0]["content"].startswith("<session_context>"), \
        "stable block is not the byte-prefix -> KV cache can never reuse it"
    assert "STABLEBLOCK" in out[0]["content"].split("[USER INSTRUCTION]")[0]


# ── handle_chat harness (adapted from tests/test_bench_handle_chat_1c.py) ──
class _FakeBgTasks:
    def add_task(self, *a, **k):
        pass


# ── S1: first overflow must emit the context-pressure steer ──────────


# ── S3: request start must CLEAR a stale lockdown (disarm path) ──────


# ── S4: per-batch read budget must be the window-derived constant ────


# ── S5: lockdown must zero the batch read budget ─────────────────────


# ── X1: the compression ladder must engage below 95% occupancy ───────
def test_pin_X1_ladder_engages_at_80pct():
    from ghost_agent.core.context_manager import ContextManager
    cm = ContextManager(max_tokens=1000)   # default len//4 estimator
    big_tool = "\n".join(f"line {i} " + "w" * 70 for i in range(40))  # ~3.1k chars
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "goal"},
        {"role": "tool", "name": "fs", "content": big_tool},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
    ]
    out = cm.compress_if_needed([dict(m) for m in msgs], max_level=3)
    assert cm.compression_level >= 1, \
        f"ladder never engaged at ~80% occupancy (level={cm.compression_level})"
    tool_out = next(m["content"] for m in out if m.get("role") == "tool")
    assert "compressed]" in tool_out and len(tool_out) < len(big_tool), \
        "old tool output was not compressed at ~80% occupancy"


# ── S2b: SECOND overflow must actually ARM the lockdown (behavioral;
#         the only existing kill is a getsource token pin) ─────────────


# ── S6b: budget-construction failure under lockdown must fail CLOSED
#         (behavioral; the only existing kill is a file-text token pin) ─
