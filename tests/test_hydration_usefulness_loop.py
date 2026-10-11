"""The hydration stash and the vector helpful-credit (2026-07-14).

`hydrate_context` stashes this turn's surviving items (+ intent, ts) on
`bus.last_hydration` (and per turn in `_hydration_by_turn`), and
`VectorMemory.bump_helpful` is the double-half-life credit for an item a
reply actually drew on. (§4NM: the post-turn usefulness judge, its
observations ledger and the dream's RRF refit were removed — the stash
and the credit primitive are what remain.)
"""

from unittest.mock import MagicMock

from ghost_agent.core.bus import MemoryBus
from ghost_agent.memory.vector import VectorMemory


class TestStash:
    async def test_hydrate_context_stashes_survivors(self):
        skill = MagicMock()
        skill.get_playbook_items.return_value = [
            {"text": "Ask the supervisor for a port.", "trigger": "deploy the web app"},
        ]
        bus = MemoryBus(skill_memory=skill)

        out = await bus.hydrate_context("how do I deploy the web app")

        assert out
        assert bus.last_hydration is not None
        assert bus.last_hydration["survivors"][0]["trigger"] == "deploy the web app"
        assert "intent" in bus.last_hydration and "ts" in bus.last_hydration


class TestVectorBumpHelpful:
    def test_bump_helpful_increments_and_touches(self):
        vm = VectorMemory.__new__(VectorMemory)  # bypass heavy __init__
        vm.collection = MagicMock()
        vm.collection.get.return_value = {
            "ids": ["v1"], "metadatas": [{"helpful_count": 2}],
        }
        vm.bump_helpful(["v1", "v1", None])
        vm.collection.get.assert_called_once_with(ids=["v1"], include=["metadatas"])
        meta = vm.collection.update.call_args.kwargs["metadatas"][0]
        assert meta["helpful_count"] == 3
        assert "last_accessed" in meta

    def test_bump_helpful_empty_is_noop(self):
        vm = VectorMemory.__new__(VectorMemory)
        vm.collection = MagicMock()
        vm.bump_helpful([])
        vm.collection.get.assert_not_called()
