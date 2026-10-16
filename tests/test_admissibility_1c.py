"""§4BF Track 1c — the origin-aware seams that outlived the bench.

Pins the seams that still key on a trajectory/calibration origin:
experiment scopes (SCOPE_BENCH specs never reach live/sim turns), the
calibration origin + oracle re-label + equal-mass cap, the learning-health
denominator surfaces, and the agent's hydration-stamp / digest gates.
(The PRM rows left with ghost_agent/prm; §4NM also removed
core/admissibility.py, the bench corpus readers, optim/trainset.py,
optim/tool_fixtures.py and dream's lesson gate — their pins went too.)
"""
import json
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

from ghost_agent.distill.collector import TrajectoryCollector
from ghost_agent.distill.schema import Trajectory


# ──────────────────────────────────────────────────────────────────────
# Experiment scopes
# ──────────────────────────────────────────────────────────────────────

def _traj(kind, stamp=None, outcome="passed"):
    extra = {"experiments": stamp} if stamp else {}
    return types.SimpleNamespace(task_kind=kind, extra=extra,
                                 outcome=outcome, n_steps=2, duration_s=1.0)


# ──────────────────────────────────────────────────────────────────────
# R1 review fixes — behavioral pins
# ──────────────────────────────────────────────────────────────────────

class TestHydratedLessonsStamp:
    def _record(self, ctx_extra):
        from ghost_agent.core.agent import GhostAgent

        class _Col:
            def __init__(self):
                self.appended = []

            def append(self, traj):
                self.appended.append(traj)
                return Path("/dev/null")

        col = _Col()
        ctx = types.SimpleNamespace(
            trajectory_collector=col, self_model=None, profile_memory=None,
            _recent_trajectories_for_correction=None, **ctx_extra)
        agent = GhostAgent.__new__(GhostAgent)
        agent.context = ctx
        agent._record_turn_trajectory(
            messages=[{"role": "user", "content": "solve it"},
                      {"role": "assistant", "content": "done"}],
            final_content="done", req_id="rH", model="m",
            user_request="solve it")
        return col.appended[0]

    def test_bench_records_never_carry_the_real_turns_lessons(self):
        # R5 review MAJOR: the read-only wrapper passed
        # last_playbook_triggers through, so bench trajectories were
        # stamped with the LAST REAL TURN's lesson set as their own
        # hydration — counterfactual-attribution poison.
        sm = types.SimpleNamespace(
            is_read_only=True,
            last_playbook_triggers=["real lesson trigger"])
        traj = self._record({
            "trajectory_task_kind": "bench",
            "turn_origin_label": "bench",
            "skill_memory": sm,
        })
        assert traj.task_kind == "bench"
        assert "hydrated_lessons" not in (traj.extra or {})

    def test_real_turns_still_stamp_their_hydration(self):
        sm = types.SimpleNamespace(
            is_read_only=False,
            last_playbook_triggers=["real lesson trigger"])
        traj = self._record({"skill_memory": sm})
        assert (traj.extra or {}).get("hydrated_lessons") == \
            ["real lesson trigger"]


class TestSourceWindowPins:
    # Honest window pins for one-line guards whose behavioral harnesses
    # would cost more than they return (R3 MAJOR-2 c/d).
    def test_digest_block_is_user_origin_gated(self):
        import inspect
        import ghost_agent.core.agent as agent_mod
        src = inspect.getsource(agent_mod)
        i = src.index("Autoadvance Digest")
        window = src[max(0, i - 4000):i]
        assert 'turn_origin(self.context) == "user"' in window

    def test_activity_digest_block_is_user_origin_gated_too(self):
        # R4 review CRIT: the SIBLING activity-digest block 110 lines
        # below the project digest was left with only the internal-request
        # gate — bench finalizes consumed the watermark and ate the very
        # notify events (incl. bench experiment verdicts) 1c produces.
        import inspect
        import ghost_agent.core.agent as agent_mod
        src = inspect.getsource(agent_mod)
        i = src.index("render_activity_digest as _render_adg")
        window = src[i:i + 1200]
        assert 'turn_origin(self.context) == "user"' in window

    def test_playbook_triggers_stay_off_the_readonly_passthrough(self):
        # R6: this R5 fix was revertible with 297 tests green (the belt
        # caught the observable effect, but the fix itself was unpinned).
        # The passthrough serving the LAST REAL TURN's lesson set to sim
        # reads is the root cause; it must stay denied.
        #
        # §4CL S1: the façade moved to `core/isolation.py`, and the pin
        # moved from a source-window grep to the executed contract — a
        # sim read of that attribute must RAISE, not serve.
        import pytest as _pytest
        from unittest.mock import MagicMock
        from ghost_agent.core.isolation import ReadOnlySkillMemory

        assert "last_playbook_triggers" not in (
            ReadOnlySkillMemory._SAFE_PASSTHROUGH)
        real = MagicMock()
        real.last_playbook_triggers = ["a-real-users-lesson"]
        with _pytest.raises(AttributeError, match="passthrough whitelist"):
            ReadOnlySkillMemory(real).last_playbook_triggers
