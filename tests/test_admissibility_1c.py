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
# Calibration: origin, oracle re-label, equal-mass cap
# ──────────────────────────────────────────────────────────────────────

class TestCalibrationOrigin:
    def _tracker(self, tmp_path):
        from ghost_agent.core.calibration import CalibrationTracker
        return CalibrationTracker(tmp_path / "calibration")

    def _record_turn(self, ct, req_id, origin, outcome=1.0):
        assert ct.record(composite=0.7, entropy_component=0.5,
                         competence_component=0.6, outcome=outcome,
                         source="turn", req_id=req_id, origin=origin)

    def test_origin_round_trips_and_legacy_defaults_to_user(self, tmp_path):
        ct = self._tracker(tmp_path)
        self._record_turn(ct, "rb", "bench")
        # Hand-write a legacy row with no origin field at all.
        with ct.history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"composite": 0.5, "entropy_component": 0.5,
                                 "competence_component": 0.5, "outcome": 1.0,
                                 "ts": "2026-08-13T00:00:00Z",
                                 "source": "turn", "req_id": "legacy"}) + "\n")
        samples = ct._load_samples()
        by_req = {s.req_id: s for s in samples}
        assert by_req["rb"].origin == "bench"
        assert by_req["legacy"].origin == "user"



    def test_rank_order_oracle_beats_verifier_human_beats_oracle(self):
        from ghost_agent.core.calibration import (
            CalibrationSample, _SOURCE_RANK, _resolve_superseded)
        assert _SOURCE_RANK["bench_validator"] > _SOURCE_RANK["verifier_late"]
        assert _SOURCE_RANK["user_correction"] > _SOURCE_RANK["bench_validator"]

        def s(source, outcome):
            return CalibrationSample(
                composite=0.5, entropy_component=0.5,
                competence_component=0.5, uncertainty_pressure=0.0,
                outcome=outcome, req_id="r1", source=source)
        resolved = _resolve_superseded(
            [s("turn", 1.0), s("bench_validator", 0.0),
             s("verifier_late", 1.0)])
        assert len(resolved) == 1
        assert resolved[0].source == "bench_validator"

# ──────────────────────────────────────────────────────────────────────
# Learning-health denominators
# ──────────────────────────────────────────────────────────────────────

class TestLearningHealthSurfaces:
    def test_label_health_counts_origins(self):
        from ghost_agent.core.learning_health import _label_health
        out = _label_health([
            {"outcome": 1.0, "source": "turn"},
            {"outcome": 0.0, "source": "turn", "origin": "bench"},
            {"outcome": 1.0, "source": "bench_validator", "origin": "bench"},
        ])
        assert out["label_origins"] == {"user": 1, "bench": 2}


# ──────────────────────────────────────────────────────────────────────
# R1 review fixes — behavioral pins
# ──────────────────────────────────────────────────────────────────────

class TestCalibrationCapOrdering:

    def test_user_correction_row_supersedes_via_req_id(self, tmp_path):
        # R1 MAJOR (pre-existing): the correction stash dropped req_id, so
        # the ground-truth negative sat NEXT TO the proxy positive in the
        # fit instead of replacing it.
        from ghost_agent.core.calibration import CalibrationTracker
        ct = CalibrationTracker(tmp_path / "calibration")
        ct.record(composite=0.8, entropy_component=0.5,
                  competence_component=0.7, outcome=1.0,
                  source="turn", req_id="R9")
        ct.record(composite=0.8, entropy_component=0.5,
                  competence_component=0.7, outcome=0.0,
                  source="user_correction", req_id="R9")
        rows = ct._load_epoch()
        assert len(rows) == 1
        assert rows[0].source == "user_correction"
        assert rows[0].outcome == 0.0

class TestFloodDetectorPopulation:
    def test_raw_origin_mix_is_current_epoch_precap(self):
        # R3: counting ALL epochs let ~retired real rows mask a current
        # epoch flooded with bench. The detector must see the epoch the
        # fit reads, pre-cap.
        from ghost_agent.core.calibration import CURRENT_EPOCH
        from ghost_agent.core.learning_health import _live_epoch_filter
        rows = (
            # A retired epoch full of real rows.
            [{"origin": "user", "req_id": f"old{i}", "epoch": "old",
              "ts": "2026-01-01T00:00:00Z"} for i in range(50)]
            # A current epoch flooded 4:1 bench.
            + [{"origin": "bench", "req_id": f"b{i}",
                "epoch": CURRENT_EPOCH} for i in range(8)]
            + [{"origin": "user", "req_id": f"r{i}",
                "epoch": CURRENT_EPOCH} for i in range(2)]
        )
        capped, _n_other, _n_sup, precap = _live_epoch_filter(rows)
        pre_mix = {}
        for s in precap:
            pre_mix[s.get("origin") or "user"] = \
                pre_mix.get(s.get("origin") or "user", 0) + 1
        # Pre-cap = the flood is visible (8 bench vs 2 real), and the
        # retired-epoch real mass is NOT diluting it.
        assert pre_mix == {"bench": 8, "user": 2}
        # §4NM: the equal-mass cap is gone — the fit's population is the
        # epoch-scoped one (no stored verdict excludes bench here).
        assert len(capped) == 10


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
            _recent_calib_for_correction=None,
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


class TestBenchOnlyEpochRender:
    def test_render_survives_a_bench_only_epoch(self, tmp_path, monkeypatch):
        # R4 review, REPRODUCED: a bench-only current epoch empties the
        # capped population, _label_health returns {}, and the first cut
        # of the flood-detector render line took the WHOLE learning-health
        # report down with a KeyError — the operator instrument dying at
        # exactly maximal flood.
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        from ghost_agent.core.calibration import CalibrationTracker
        md = tmp_path / "system" / "memory"
        md.mkdir(parents=True, exist_ok=True)
        ct = CalibrationTracker(tmp_path / "system" / "calibration")
        for i in range(5):
            ct.record(composite=0.6, entropy_component=0.5,
                      competence_component=0.5, outcome=0.0,
                      source="turn", req_id=f"b{i}", origin="bench")
        from ghost_agent.core.learning_health import render_learning_health
        out = render_learning_health(md)
        assert "origin mix (current epoch, pre-cap)" in out
        assert "BENCH-ONLY epoch" in out

    def test_recent_samples_filters_before_the_window(self, tmp_path):
        # R4 review: filtering AFTER the tail was a no-op against
        # dilution — the real rows bench pushed out stayed out.
        from ghost_agent.core.calibration import CalibrationTracker
        ct = CalibrationTracker(tmp_path / "calibration")
        ct.record(composite=0.5, entropy_component=0.5,
                  competence_component=0.5, outcome=1.0,
                  source="turn", req_id="real-1", origin="user")
        for i in range(10):
            ct.record(composite=0.5, entropy_component=0.5,
                      competence_component=0.5, outcome=0.0,
                      source="turn", req_id=f"b{i}", origin="bench")
        # A window of 5 taken tail-first holds zero real rows…
        assert all(s.origin == "bench" for s in ct.recent_samples(5))
        # …but the origin-aware window still finds the real one.
        got = ct.recent_samples(5, exclude_origin="bench")
        assert [s.req_id for s in got] == ["real-1"]


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
