"""The self-play slot cooldown in _biological_tick.

§4NM: the FrontierTracker-driven adaptive cooldown left with synthetic
self-play (its halve/double/gate tests were removed); the slot cooldown is
now the static `_SELFPLAY_COOLDOWN`, lazily initialised on the first tick.
"""

import datetime
from unittest.mock import MagicMock, patch

import pytest

from ghost_agent.core.agent import GhostAgent, GhostContext
from ghost_agent.memory.frontier import FrontierTracker


# ⚠ This file used to define its own autouse `_isolate_ghost_home` that
# DELETED the var — which SHADOWED conftest's same-named fixture and
# re-armed the very failure conftest documents: with GHOST_HOME unset,
# modules fall back to a real path in the user's $HOME (~/ghost_llamacpp),
# so "hermetic" was strictly worse than the tmp home it displaced. The
# conftest fixture already covers this file's stated need (a throwaway
# home per test); the local override is deleted rather than renamed
# (found by the 2026-08-17 feedback-channel review).


def _make_agent_with_tracker(tmp_path, idle_seconds=4000):
    ctx = MagicMock(spec=GhostContext)
    ctx.args = MagicMock()
    ctx.args.model = "test-model"
    ctx.args.temperature = 0.5
    ctx.args.max_context = 4000
    ctx.args.smart_memory = 0.0

    ctx.llm_client = MagicMock()
    ctx.llm_client.foreground_tasks = 0
    ctx.llm_client.foreground_requests = 0  # HARD LOCK also gates on this now

    ctx.memory_system = MagicMock()
    ctx.memory_system.collection.get.return_value = {"ids": []}

    ctx.profile_memory = MagicMock()
    ctx.scratchpad = MagicMock()
    ctx.skill_memory = None
    ctx.graph_memory = None
    ctx.sandbox_dir = str(tmp_path)
    ctx.journal = None
    ctx.last_activity_time = datetime.datetime.now() - datetime.timedelta(seconds=idle_seconds)

    ctx.frontier_tracker = FrontierTracker(tmp_path)
    return GhostAgent(ctx)


@pytest.mark.asyncio
async def test_default_cooldown_before_any_run(tmp_path):
    agent = _make_agent_with_tracker(tmp_path)
    # Trigger the lazy init path via one tick (force random gate to skip so
    # the actual dreamer is never invoked).
    with patch("ghost_agent.core.dream.Dreamer"), \
         patch("ghost_agent.core.agent.random.random", return_value=0.9):
        await agent._biological_tick()
    assert agent._current_selfplay_cooldown == GhostAgent._SELFPLAY_COOLDOWN
