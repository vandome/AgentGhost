"""§4DA round 2 — what the round-1 fixes left open, and what nobody read.

Round 1 closed the significance gap and pinned its call site. Round 2's two
lenses found that the round-1 fix itself was half of a fix, and that nothing
DOWNSTREAM of the gate had been looked at at all:

  * the margin was never de-contaminated, only the statistic — so the exact
    one-arm transport outage round 1 recorded as closed still promoted
    through `--allow-insignificant-ship`, which ships on the margin ALONE;
  * the artifact recording that override was written in a shape its only
    reader cannot open, with no `gate_arm`, so production logged a fresh
    promotion as "predates the gate schema";
  * the tool-description read site never passed a `req_id`, so §4CZ's live
    judge could never see a single attributed turn for the ONE optimizer
    §4DA gates — `--revert` was structurally unreachable;
  * the refusal message's remedy was bench-inflated by 3.4x, and the supply
    gate it routes the operator through counts bench too.

§4NM: the optimizer, its ship gate (`scripts/optimize_tool_descriptions.py`),
`optim/ab_eval` and the fixture miner were removed; their tests went with
them. What remains pins the READ SITE (`tools/registry.py` + `optim/loader.py`)
attribution that round 2 wired.
"""

import json
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════
# MAJOR-4b — the ONE optimizer §4DA gates had no live judge
# ══════════════════════════════════════════════════════════════════════
class TestTheToolDescriptionReadSiteIsAttributed:
    def test_the_REAL_registry_stamps_the_served_artifact(self, tmp_path,
                                                          monkeypatch):
        """⚠ `tuned_instruction(sig, "")` WITH NO req_id RETURNS AT
        `_note_served`'s EMPTY-req_id GUARD. Measured before the fix:
        `activation_stats` counted the artifact applied while
        `served_for_request()` was empty and `_SERVED_RING` had nothing in
        it — so `gepa_live_check --signature tool_description.*` could only
        ever say CONFOUNDED and `--revert` was structurally unreachable."""
        from ghost_agent.optim import loader as L
        from ghost_agent.tools import registry as R
        from ghost_agent.utils.logging import request_id_context

        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        name = "web_search"
        base = R.TOOL_DEFINITIONS
        tool = next(t for t in base
                    if (t.get("function") or {}).get("name") == name)
        baseline = tool["function"]["description"]
        tuned = baseline + " Prefer it for current events."
        (home / "system" / "optim"
         / f"tool_description.{name}.json").write_text(json.dumps({
             "signature_name": f"tool_description.{name}",
             "optimized_instruction": tuned,
             "gate_arm": "tool-choice fidelity A/B, private holdout"}))
        monkeypatch.setenv("GHOST_HOME", str(home))
        L.clear_cache()
        # ⚠ `_TUNED_DESC_NAMES` IS A PROCESS-WIDE LAZY FROZENSET, scanned
        # once. Any earlier test in this worker that touched the read site
        # leaves it populated for a DIFFERENT GHOST_HOME, so this
        # artifact's tool is absent, nothing swaps, and the assertion
        # below fails for a reason that has nothing to do with the stamp.
        # Found under `-n 8 --dist loadfile`, passing alone — a suite bug,
        # not a scheduler artifact (§4CX).
        monkeypatch.setattr(R, "_TUNED_DESC_NAMES", None, raising=False)

        token = request_id_context.set("req-attrib-1")
        try:
            out = R._apply_tuned_descriptions(
                [{"type": "function",
                  "function": {"name": name, "description": baseline,
                               "parameters": {}}}],
                context=object())
        finally:
            request_id_context.reset(token)
        assert out[0]["function"]["description"] == tuned, \
            "the artifact did not reach the read site at all"
        served = L.served_for_request("req-attrib-1")
        assert f"tool_description.{name}" in served, (
            "the read site served the artifact and stamped nothing — "
            "gepa_live_check can never judge it")
        L.clear_cache()
        R._TUNED_DESC_NAMES = None

    def test_the_CALL_SITE_passes_the_context(self):
        """⚠ The plumbing exists and the one caller passes nothing —
        which is exactly the state §4CZ shipped in."""
        import ast
        src = Path("src/ghost_agent/tools/registry.py").read_text()
        tree = ast.parse(src)
        calls = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call)
                 and getattr(n.func, "id", "") == "_apply_tuned_descriptions"]
        assert calls, "the read site is gone"
        assert any(len(c.args) > 1 or c.keywords for c in calls), \
            "get_active_tool_definitions still drops the context"
