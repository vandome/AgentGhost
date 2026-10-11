"""EpisodicMemory must be passed into MemoryBus so `_fetch_episodic`
actually sees an instance during context hydration. Before this wiring,
the bus's episodic slot stayed None even though main.py constructed
an EpisodicMemory instance on the context.
"""

import pytest
from unittest.mock import MagicMock


class TestAgentLazyBusIncludesEpisodic:
    """When the agent lazily builds a MemoryBus from context (tests path),
    the episodic_memory slot on context must be threaded through."""

    def test_agent_lazy_bus_propagates_episodic(self):
        from ghost_agent.core.agent import GhostAgent, GhostContext
        import argparse, tempfile
        from pathlib import Path

        args = argparse.Namespace(
            max_context=4096, smart_memory=0.0, anonymous=True,
            model="test", verbose=False, no_memory=True,
        )
        with tempfile.TemporaryDirectory() as td:
            ctx = GhostContext(args, Path(td), Path(td), "socks5://none")
            ctx.memory_bus = None  # force the lazy path
            sentinel = MagicMock()
            ctx.episodic_memory = sentinel
            agent = GhostAgent(ctx)
            bus = agent._get_memory_bus()
            assert bus.episodic is sentinel

