"""Tests for the GEPA prompt loader (M7) — the offline-optimization stage
that was previously write-only."""

import json
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------- GEPA loader

class TestGepaLoader:
    def _setup(self, monkeypatch, tmp):
        monkeypatch.setenv("GHOST_HOME", str(tmp))
        import ghost_agent.optim.loader as L
        L.clear_cache()
        return L

    def test_absent_returns_baseline(self, monkeypatch, tmp_path):
        L = self._setup(monkeypatch, tmp_path)
        assert L.tuned_instruction("planning.decompose", "BASELINE") == "BASELINE"

    def test_present_returns_optimized(self, monkeypatch, tmp_path):
        L = self._setup(monkeypatch, tmp_path)
        d = tmp_path / "system" / "optim"
        d.mkdir(parents=True)
        (d / "planning.decompose.json").write_text(json.dumps({
            "signature_name": "planning.decompose",
            "optimized_instruction": "TUNED: plan tersely.",
        }))
        L.clear_cache()
        assert L.tuned_instruction("planning.decompose", "BASELINE") == "TUNED: plan tersely."

    def test_corrupt_falls_back(self, monkeypatch, tmp_path):
        L = self._setup(monkeypatch, tmp_path)
        d = tmp_path / "system" / "optim"
        d.mkdir(parents=True)
        (d / "reflection.critique.json").write_text("{ not json ]")
        L.clear_cache()
        assert L.tuned_instruction("reflection.critique", "BASE") == "BASE"

    def test_empty_optimized_falls_back(self, monkeypatch, tmp_path):
        L = self._setup(monkeypatch, tmp_path)
        d = tmp_path / "system" / "optim"
        d.mkdir(parents=True)
        (d / "x.json").write_text(json.dumps({"optimized_instruction": "   "}))
        L.clear_cache()
        assert L.tuned_instruction("x", "BASE") == "BASE"


