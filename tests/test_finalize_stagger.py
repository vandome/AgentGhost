"""Deferred-verdict handle: `_attach_late_verdict_handler` publishes its
task as `agent._deferred_verdict_task` and clears it when the verdict
lands (2026-07-16).

The handle let the end-of-turn worker calls run back-to-back instead of
colliding on the single worker node. Its original consumer, the
hydration-usefulness judge, was removed with the lesson producers (§4NM);
the handle itself stays and is what these tests pin.
"""

import asyncio
import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.core.agent import GhostAgent


async def _drain_bg():
    # Let spawned background tasks run to completion.
    for _ in range(20):
        await asyncio.sleep(0)


class TestLateVerdictHandlerPublishesTask:
    async def test_task_published_then_cleared_on_completion(self):
        agent = GhostAgent.__new__(GhostAgent)
        agent._record_late_verdict = MagicMock()

        gate = asyncio.Event()

        async def verdict():
            await gate.wait()
            return "vr", "lt"

        task = asyncio.create_task(verdict())
        agent._attach_late_verdict_handler(task, "traj-1", "conv-1")
        assert agent._deferred_verdict_task is task

        gate.set()
        await task
        await _drain_bg()
        # done-callback clears the stagger handle and applies side effects.
        assert agent._deferred_verdict_task is None
        agent._record_late_verdict.assert_called_once_with(
            "vr", "traj-1", "conv-1", last_tool="lt",
            # §4EP threads the turn's tool COUNT to the skip message so it can
            # name the cause it observed; the handler forwards whatever the
            # spawn site captured (None = not captured).
            force_correction=False, project_id=None, n_tools=None)

    async def test_cancelled_verdict_still_clears_handle(self):
        agent = GhostAgent.__new__(GhostAgent)
        agent._record_late_verdict = MagicMock()

        async def verdict():
            await asyncio.sleep(30)

        task = asyncio.create_task(verdict())
        agent._attach_late_verdict_handler(task, "traj-1", "conv-1")
        task.cancel()
        await _drain_bg()
        assert agent._deferred_verdict_task is None
        agent._record_late_verdict.assert_not_called()
