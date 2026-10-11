"""Lesson quarantine on SkillMemory (the surface the counterfactual-replay
loop used).

The counterfactual replay phase itself (core/counterfactual.py: persist,
classify, batch, candidate/recheck, learning-state gate) was removed in
batch 3 (§4NM); its tests went with it, and the dream.py self-play seam
pins went with synthetic self-play (§4NM batch 4). What remains pins the
live quarantine / unquarantine surface on `SkillMemory`.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.memory.skills import SkillMemory


def _mem_with_lessons(tmp_path, *triggers):
    (tmp_path / "mem").mkdir(exist_ok=True)
    sm = SkillMemory(tmp_path / "mem")
    for trig in triggers:
        sm.learn_lesson(trig, "mistake", "correction", trigger=trig,
                        source="learn_skill")   # §4NM: the writer gate is a hard rule
    return sm


class TestSkillsQuarantine:
    def test_filter_covers_both_retrieval_surfaces(self, tmp_path):
        sm = SkillMemory(tmp_path)
        sm.learn_lesson("parse json safely", "used eval",
                        "use json.loads", trigger="parse json safely",
                        source="learn_skill")
        assert sm.quarantine_lesson("parse json safely", "test") == 1
        assert sm.get_playbook_items("parse json safely") == []
        out = sm.get_playbook_context("parse json safely")
        assert "parse json" not in (out or "")

    def test_hydration_side_channel_set(self, tmp_path):
        sm = SkillMemory(tmp_path)
        sm.learn_lesson("regex over split", "split on comma",
                        "use re for csv edge cases",
                        trigger="regex over split", source="learn_skill")
        sm.get_playbook_context("regex over split")
        assert sm.last_playbook_triggers == ["regex over split"]


class TestQuarantineHasAWayBack:
    def test_unquarantine_is_scoped_to_the_reason_that_imposed_it(self, tmp_path):
        sm = _mem_with_lessons(tmp_path, "a", "b")
        sm.quarantine_lesson("a", reason="counterfactual regression on challenge X: …")
        sm.quarantine_lesson("b", reason="mirror audit: contradicts the profile")
        assert sm.unquarantine_lesson("b", reason_contains="challenge X") == 0   # not this loop's
        assert sm.unquarantine_lesson("a", reason_contains="challenge X") == 1
        rows = {r["trigger"]: r for r in sm._load_playbook()}
        assert rows["a"]["quarantined"] is False and rows["b"]["quarantined"] is True
        assert sm.unquarantine_lesson("a", reason_contains="challenge X") == 0   # idempotent
