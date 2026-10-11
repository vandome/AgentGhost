"""Re-audit of the 2026-08-04 GEPA/optim fixes (§4J).

§4NM: the optimizer side (`scripts/run_gepa.py`,
`scripts/optimize_tool_descriptions.py`, `scripts/recheck_gepa_incumbent.py`,
`optim/ab_eval`, `optim/trainset`, `optim/tool_fixtures`) was removed and
every test that drove it went with it. What remains is the one pin on the
LOADER's activation telemetry that needed nothing from the optimizer: an
artifact set the read site drops as a whole must count as REJECTED, never as
APPLIED.
"""

import json

import pytest


@pytest.fixture
def baselines():
    """The hand-written tool descriptions, keyed by tool name — what the
    removed gate's `_baseline_descriptions()` returned."""
    from ghost_agent.tools import registry as reg
    return {t["function"]["name"]: t["function"].get("description", "")
            for t in reg.TOOL_DEFINITIONS}


class TestActivationCountsApplicationsNotLoads:
    def test_an_aggregate_rejected_artifact_reads_applied_zero(
            self, baselines, tmp_path, monkeypatch):
        """The ONE instrument built to catch silent inoperativeness counted
        LOADS, so over-inflated artifacts read `applied: 1, fallback: 0`
        while nothing whatsoever reached the model."""
        from ghost_agent.optim import loader
        from ghost_agent.tools import registry as reg

        (tmp_path / "system" / "optim").mkdir(parents=True)
        slack = reg._TOOL_DESC_AGGREGATE_SLACK
        headroom = sorted(((max(6000, 3 * len(baselines[n])) - len(baselines[n]), n)
                           for n in baselines), reverse=True)
        names = [n for _, n in headroom[:5]]
        for n in names:
            (tmp_path / "system" / "optim" /
             f"tool_description.{n}.json").write_text(json.dumps({
                 "optimized_instruction":
                     baselines[n] + "X" * ((slack // 5) + 50)}))

        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        reg._reset_tool_desc_cache()
        loader.clear_cache()
        loader._APPLIED_COUNTS.clear()
        loader._FALLBACK_COUNTS.clear()
        loader._REJECTED_COUNTS.clear()
        try:
            reg._apply_tuned_descriptions(
                [{"type": "function",
                  "function": {"name": n, "description": baselines[n]}}
                 for n in names])
            stats = loader.activation_stats()
            for n in names:
                s = stats[f"tool_description.{n}"]
                assert s["applied"] == 0, f"{n} reads APPLIED while inert"
                assert s["rejected"] >= 1, f"{n} rejection uncounted"
        finally:
            reg._reset_tool_desc_cache()
            loader.clear_cache()
            loader._APPLIED_COUNTS.clear()
            loader._FALLBACK_COUNTS.clear()
            loader._REJECTED_COUNTS.clear()
