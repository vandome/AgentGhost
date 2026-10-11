"""Tests for §4F Phase 0 — optimizer eval hygiene.

What remains after §4NM (the optimizer, `optim/trainset` and `optim/run_gepa`
were removed; the holdout-split, iteration-clamp, valset and dspy-conversion
tests went with them):
  * loader activation telemetry (tuned-vs-baseline application counts)
    surfaced through learning-health;
  * the trajectory collector's default root.
"""

import json
import sys


# ----------------------------------------------------- default-root alignment

class TestTrajectoryDefaultRoot:
    def test_default_root_matches_prod_write_path(self, monkeypatch, tmp_path):
        """Prod writes via memory_dir.parent → $GHOST_HOME/system/trajectories
        (main.py). The collector's bare default MUST point at the same place —
        it once lacked the system/ segment, so the GEPA ignition run read an
        empty directory while 24 days of trajectories sat one level down."""
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        from ghost_agent.distill.collector import _default_root
        assert _default_root() == tmp_path / "system" / "trajectories"


# ------------------------------------------------------- activation telemetry

class TestActivationTelemetry:
    def _setup(self, monkeypatch, tmp):
        monkeypatch.setenv("GHOST_HOME", str(tmp))
        import ghost_agent.optim.loader as L
        L.clear_cache()
        L._APPLIED_COUNTS.clear()
        L._FALLBACK_COUNTS.clear()
        return L

    def _write_artifact(self, tmp, name, instruction="Tuned. Be terse."):
        d = tmp / "system" / "optim"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.json").write_text(
            json.dumps({"optimized_instruction": instruction}))

    def test_applied_counts_on_tuned(self, monkeypatch, tmp_path):
        L = self._setup(monkeypatch, tmp_path)
        self._write_artifact(tmp_path, "planning.decompose")
        L.tuned_instruction("planning.decompose", "baseline")   # uncached path
        L.tuned_instruction("planning.decompose", "baseline")   # cached path
        stats = L.activation_stats()
        assert stats["planning.decompose"]["applied"] == 2
        assert stats["planning.decompose"]["fallback"] == 0

    def test_fallback_counts_when_absent(self, monkeypatch, tmp_path):
        L = self._setup(monkeypatch, tmp_path)
        L.tuned_instruction("tool_selection.pick", "baseline")
        L.tuned_instruction("tool_selection.pick", "baseline")
        stats = L.activation_stats()
        assert stats["tool_selection.pick"]["applied"] == 0
        assert stats["tool_selection.pick"]["fallback"] == 2

    def test_clear_cache_preserves_counters(self, monkeypatch, tmp_path):
        L = self._setup(monkeypatch, tmp_path)
        self._write_artifact(tmp_path, "planning.decompose")
        L.tuned_instruction("planning.decompose", "baseline")
        L.clear_cache()
        assert L.activation_stats()["planning.decompose"]["applied"] == 1

    def test_learning_health_pairs_artifacts_with_activation(
            self, monkeypatch, tmp_path):
        L = self._setup(monkeypatch, tmp_path)
        self._write_artifact(tmp_path, "planning.decompose")
        # Staging candidates must not be reported as live artifacts.
        (tmp_path / "system" / "optim" / "x.json.candidate").write_text("{}")

        from ghost_agent.core.learning_health import _optim_activation
        report = _optim_activation(tmp_path / "system" / "optim")
        assert report["artifacts"] == {
            "planning.decompose": {"chars": len("Tuned. Be terse."), "valid": True}}
        # Tuned artifact on disk, zero applies this process — the render
        # must carry the warning flag (the write-only defect detector).
        assert report["activation"].get("planning.decompose") is None

        L.tuned_instruction("planning.decompose", "baseline")
        report = _optim_activation(tmp_path / "system" / "optim")
        assert report["activation"]["planning.decompose"]["applied"] == 1

    def test_render_flags_tuned_but_never_applied(self, monkeypatch, tmp_path):
        self._setup(monkeypatch, tmp_path)
        self._write_artifact(tmp_path, "planning.decompose")
        memory_dir = tmp_path / "system" / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)

        from ghost_agent.core.learning_health import render_learning_health
        # §4L Lens-C MINOR-2: the applies counter is per-process, so a
        # HEADLESS render (argv is pytest here) must say "wrong process"
        # instead of crying wolf; the in-process render (argv = main.py)
        # keeps the real alarm.
        text = render_learning_health(memory_dir)
        assert "PROMPT OPTIMIZATION" in text
        assert "per-process counter" in text
        monkeypatch.setattr(sys, "argv",
                            ["/repo/src/ghost_agent/main.py"])
        text2 = render_learning_health(memory_dir)
        assert "0 applies since boot" in text2
