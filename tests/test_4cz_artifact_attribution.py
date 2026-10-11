"""§4CZ — which tuned artifact served this turn, recorded on the turn.

⚠ THE PROVENANCE WAS COMPUTED AND THROWN AWAY. `optim/loader.py` has always
derived an sha8 for each artifact it serves (`_ARTIFACT_SHAS`) and **nothing
outside that module ever read it**. So a promoted prompt could only ever be
judged by its pre-ship holdout — a few dozen examples — and never by what it
did in production. The artifact retired in §4CW served every planner turn for
weeks on a win nobody could reproduce, and that was invisible for exactly this
reason.

Attribution alone gives a BEFORE/AFTER comparison, which is confounded by time:
the artifact is deployed to everything at once. So the loader also honours an
optional randomized arm — the experiment named by `loader.experiment_name()` — where the
control arm is served the hand-written baseline. That is the difference between
evidence that can support a revert and evidence that cannot.

Nothing changes until an operator registers that experiment: with no context,
or no registered experiment, the artifact serves everything exactly as before.
"""

import json
import os
from pathlib import Path

import pytest

from ghost_agent.optim import loader


@pytest.fixture(autouse=True)
def _fresh(monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    d = tmp_path / "system" / "optim"
    d.mkdir(parents=True)
    (d / "planning.decompose.json").write_text(json.dumps({
        "signature_name": "planning.decompose",
        "optimized_instruction": "TUNED TEXT",
        "gate_arm": "token-F1 A/B, private holdout"}))
    def _reset():
        loader.clear_cache()
        loader._SERVED_RING.clear()
        # ⚠ The activation counters are process-global and `clear_cache`
        # does not touch them, so without this the telemetry tests below
        # read whatever earlier tests in the file spent — a leak that made
        # two of them fail for a reason unrelated to what they assert.
        loader._APPLIED_COUNTS.clear()
        loader._FALLBACK_COUNTS.clear()
        loader._REJECTED_COUNTS.clear()
    _reset()
    yield
    _reset()


def _served(req_id):
    return loader.served_for_request(req_id).get("planning.decompose")


class TestTheProvenanceIsNowReadable:
    def test_the_sha_reaches_a_caller(self):
        loader.tuned_instruction("planning.decompose", "BASE", req_id="r")
        assert _served("r")["sha"], (
            "the artifact's sha is still computed and discarded")

    def test_it_is_the_sha_OF_THE_SERVED_TEXT(self):
        """⚠ Identity, not merely presence: an sha that does not hash the
        text it served cannot attribute anything
        (`pin-identity-not-property`)."""
        import hashlib
        out = loader.tuned_instruction("planning.decompose", "BASE",
                                       req_id="r")
        assert out == "TUNED TEXT"
        want = hashlib.sha256(out.encode("utf-8")).hexdigest()[:8]
        assert _served("r")["sha"] == want

    def test_a_DIFFERENT_artifact_gets_a_different_sha(self, tmp_path):
        """Two copies of the same value cannot tell a real hash from a
        constant — the twin trap, thirteen instances of which this session
        already produced."""
        first = loader.tuned_instruction("planning.decompose", "BASE",
                                         req_id="a")
        sha_a = _served("a")["sha"]
        (tmp_path / "system" / "optim" / "planning.decompose.json").write_text(
            json.dumps({"signature_name": "planning.decompose",
                        "optimized_instruction": "A DIFFERENT INSTRUCTION",
                        "gate_arm": "x"}))
        loader.clear_cache()
        second = loader.tuned_instruction("planning.decompose", "BASE",
                                          req_id="b")
        assert first != second
        assert _served("b")["sha"] != sha_a

    def test_the_CACHED_path_attributes_too(self):
        """⚠ THE BUG I SHIPPED FIRST. Placed after the `_CACHE`
        short-circuit, the stamp ran on a process's FIRST call and silently
        not on every later one — a corpus that looks populated and is
        almost entirely unattributed, which is worse than none."""
        loader.tuned_instruction("planning.decompose", "BASE", req_id="cold")
        for rid in ("warm1", "warm2", "warm3"):
            loader.tuned_instruction("planning.decompose", "BASE", req_id=rid)
            assert _served(rid) is not None, f"{rid} was not attributed"

    def test_no_req_id_records_nothing(self):
        loader.tuned_instruction("planning.decompose", "BASE")
        assert loader.served_for_request("") == {}

    def test_forget_request_clears_it(self):
        loader.tuned_instruction("planning.decompose", "BASE", req_id="r")
        assert _served("r")
        loader.forget_request("r")
        assert loader.served_for_request("r") == {}

    def test_the_ring_is_BOUNDED(self):
        """A per-request map that only grows is a leak in a process that
        runs for weeks."""
        # ⚠ NOT `<= loader._SERVED_RING_MAX` ALONE: comparing the bound
        # against its own constant re-points when the constant moves, so
        # `_SERVED_RING_MAX = 1` passed (`self-calibrating-index-adapts`)
        # — and at 1 a second concurrent request unattributes the first,
        # which is the concurrency case the ring exists to prevent.
        assert loader._SERVED_RING_MAX >= 32, (
            "the ring is too small to hold the turns in flight")
        for i in range(loader._SERVED_RING_MAX + 25):
            loader.tuned_instruction("planning.decompose", "BASE",
                                     req_id=f"r{i}")
        assert len(loader._SERVED_RING) <= loader._SERVED_RING_MAX
        assert _served(f"r{loader._SERVED_RING_MAX + 24}") is not None, (
            "the ring evicted the NEWEST entry — it is FIFO, so the oldest "
            "goes first")


class TestTheArmIsHonoured:
    def test_UNENROLLED_is_a_third_state_not_a_control(self):
        """⚠ Turns served outside any experiment are NOT a control group —
        pooling them would reintroduce exactly the confounded before/after
        comparison the arm exists to avoid."""
        loader.tuned_instruction("planning.decompose", "BASE", req_id="u")
        assert _served("u")["arm"] == "unenrolled"
        assert _served("u")["arm"] != "control"

    def test_a_BROKEN_registry_never_breaks_the_turn(self):
        class Exploding:
            @property
            def _experiment_arms(self):
                raise RuntimeError("registry on fire")
        out = loader.tuned_instruction("planning.decompose", "BASE",
                                       context=Exploding(), req_id="e")
        assert out == "TUNED TEXT"

    def test_the_experiment_name_is_derived_from_the_signature(self):
        assert loader.experiment_name("planning.decompose") == \
            "gepa_planning_decompose"


class TestTheWARM_path_behaves_like_the_cold_one:
    """⚠ EVERY TEST ABOVE RUNS ON A COLD CACHE, and `tuned_instruction` has
    TWO control branches — one before the `_CACHE` short-circuit and one
    after. Mutating only the cached branch survived the whole file: the
    arm logic that runs on every turn after a process's first was
    untested. Warm the cache, then assert."""

    def _warm(self):
        loader.tuned_instruction("planning.decompose", "BASE")

    def test_unenrolled_on_a_WARM_cache_is_still_NOT_control(self):
        self._warm()
        loader.tuned_instruction("planning.decompose", "BASE", req_id="u")
        assert _served("u")["arm"] == "unenrolled"


class TestTheStampReachesTheTrajectory:
    """⚠ THE WIRING WAS UNPINNED. The loader can attribute perfectly and it
    buys nothing if `_record_turn_trajectory` does not carry it onto the
    record — `built-but-unwired-loops`, which is the exact defect §4CZ
    exists to fix one layer down."""

    def _agent(self, collector):
        from unittest.mock import MagicMock
        from ghost_agent.core.agent import GhostAgent
        ctx = MagicMock()
        ctx.trajectory_collector = collector
        a = GhostAgent.__new__(GhostAgent)
        a.context = ctx
        return a

    def _record(self, tmp_path, req_id):
        from ghost_agent.distill.collector import TrajectoryCollector
        coll = TrajectoryCollector(root=tmp_path / "traj", session_id="t")
        self._agent(coll)._record_turn_trajectory(
            messages=[{"role": "user", "content": "q"},
                      {"role": "assistant", "content": "a"}],
            final_content="a", req_id=req_id, model="m")
        return list(coll.iter_trajectories())

    def test_an_unattributed_turn_stamps_nothing(self, tmp_path):
        assert "optim_artifacts" not in (
            self._record(tmp_path, "never-served")[0].extra or {})

class TestTheNegativeCacheHolds:
    def test_a_MISSING_artifact_is_read_from_disk_only_once(self, tmp_path,
                                                             monkeypatch):
        """⚠ LOAD-BEARING RIGHT NOW: there is no artifact on disk, so every
        planner turn takes this path. Without the negative cache each turn
        re-does `Path.exists()` on the request hot path — measured 10 vs 1
        over ten turns."""
        (tmp_path / "system" / "optim" /
         "planning.decompose.json").unlink()
        loader.clear_cache()
        calls = {"n": 0}
        real = Path.exists

        def _counting(self, *a, **kw):
            if self.name == "planning.decompose.json":
                calls["n"] += 1
            return real(self, *a, **kw)

        monkeypatch.setattr(Path, "exists", _counting)
        for i in range(10):
            loader.tuned_instruction("planning.decompose", "BASE",
                                     req_id=f"m{i}")
        assert calls["n"] <= 1, (
            f"the missing artifact was probed {calls['n']} times — the "
            f"negative cache is not holding")


class TestTheWarmPathCarriesTheSHA:
    def test_the_warm_sha_is_the_hash_of_the_served_text(self):
        """⚠ Every sha pin ran on a COLD cache; the warm path — which is
        every turn after a process's first — was checked only for `arm`.
        A warm stamp carrying sha="" survived the whole file."""
        import hashlib
        loader.tuned_instruction("planning.decompose", "BASE")   # warm it
        out = loader.tuned_instruction("planning.decompose", "BASE",
                                       req_id="w")
        want = hashlib.sha256(out.encode("utf-8")).hexdigest()[:8]
        assert _served("w")["sha"] == want


class TestOneRequestCanServeSeveralSignatures:
    def test_a_second_signature_does_not_erase_the_first(self, tmp_path):
        """The production planner serves BOTH `planning.decompose` and
        `tool_selection.pick` under one req_id. A ring that replaced the
        slot instead of updating it would silently drop half the
        attribution, and no test put two signatures in one request."""
        (tmp_path / "system" / "optim" /
         "tool_selection.pick.json").write_text(json.dumps({
             "optimized_instruction": "PICK TEXT", "gate_arm": "g"}))
        loader.tuned_instruction("planning.decompose", "BASE", req_id="R")
        loader.tuned_instruction("tool_selection.pick", "BASE", req_id="R")
        got = loader.served_for_request("R")
        assert set(got) == {"planning.decompose", "tool_selection.pick"}
        assert got["planning.decompose"]["sha"] != \
            got["tool_selection.pick"]["sha"]


class TestTheContextMutationContract:
    def test_an_UNENROLLED_turn_is_not_marked_either(self, tmp_path):
        """⚠ Stamped on `any(sha)` this flagged unenrolled turns too, and
        every other member of CONTEXT_MUTATING_KEYS means "one arm saw
        this". `optim/tool_fixtures.py` DROPS flagged turns because
        "replaying it would optimize against a prompt only one arm ever
        sees" — false when the artifact served everyone, and it would have
        starved that miner of every planner turn the moment an artifact
        was promoted. Only `control` was pinned; unenrolled is the state
        every turn is in today."""
        from unittest.mock import MagicMock
        from ghost_agent.core.agent import GhostAgent
        from ghost_agent.distill.collector import TrajectoryCollector
        coll = TrajectoryCollector(root=tmp_path / "tj", session_id="s")
        ctx = MagicMock()
        ctx.trajectory_collector = coll
        a = GhostAgent.__new__(GhostAgent)
        a.context = ctx
        loader.tuned_instruction("planning.decompose", "BASE", req_id="U")
        assert _served("U")["arm"] == "unenrolled"
        a._record_turn_trajectory(
            messages=[{"role": "user", "content": "q"},
                      {"role": "assistant", "content": "a"}],
            final_content="a", req_id="U", model="m")
        t = list(coll.iter_trajectories())[0]
        assert "gepa_artifact_applied" not in (t.extra or {}), (
            "an unenrolled turn was marked context-mutating; every other "
            "key in that tuple means an ENROLLED treatment turn")
