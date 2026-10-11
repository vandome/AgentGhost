"""§4DA round 13 — a set-level win, stamped on every member as its own.

`ships` is ONE decision from ONE A/B over the whole candidate set, and the
promotion loop wrote N per-component artifacts each carrying that set-level
gate block. Driven with the optimizer mutating exactly one component — which is
what a real proposal does — two descriptions **byte-identical to the incumbent**
were promoted, each stamped `p_value: 0.003906, candidate_wins: 8, gate_arm:
"…[paired-v2]"`: a claim of a measured, significant win belonging entirely to a
third component. §4DA is the entry that ADDED those fields so a promotion could
be re-examined.

And the gate had no re-draw guard. `run_gepa` refuses when the live artifact is
younger than `--min-promotion-age-days` — "each run is a fresh draw at the gate,
so re-promoting before the last one can be judged turns one decision into many".
This gate WROTE `promoted_utc` and never read it. It bites harder here: α=0.05
promotes under the null 5% per run, `recheck_gepa_incumbent` exits 3 for every
`tool_description.*` so it cannot re-score them, and every re-promotion changes
the sha and resets the live check's era — discarding every accrued turn.

§4NM: the ship gate and the recheck reader were removed; their tests went with
them. The loader provenance and read-site pins remain.
"""

import json


class TestUNGATED_isNotProvenance:
    def test_the_loader_warns_about_an_ungated_artifact(self, tmp_path,
                                                        monkeypatch,
                                                        caplog):
        """⚠ `--no-ab-gate` stamps `gate_arm: "UNGATED (--no-ab-gate)"`,
        which satisfied a bare truthiness test — so an artifact whose own
        record says `metric: "none — adopted unverified"` loaded at the
        same level as a gated one, silencing the only apply-time warning
        that an unverified prompt is serving production."""
        import logging
        from ghost_agent.optim import loader as L
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        sig = "planning.decompose"
        (home / "system" / "optim" / f"{sig}.json").write_text(json.dumps({
            "signature_name": sig, "optimized_instruction": "TUNED",
            "gate_arm": "UNGATED (--no-ab-gate)"}))
        monkeypatch.setenv("GHOST_HOME", str(home))
        L.clear_cache()
        with caplog.at_level(logging.WARNING, logger=L.logger.name):
            out = L.tuned_instruction(sig, "BASE")
        L.clear_cache()
        assert out == "TUNED", "the artifact must still be served"
        msgs = [r.getMessage() for r in caplog.records]
        assert any("UNGATED" in m for m in msgs), msgs
        assert any("no A/B measured it" in m for m in msgs), msgs

    def test_a_GATED_artifact_is_not_warned_about(self, tmp_path,
                                                  monkeypatch, caplog):
        import logging
        from ghost_agent.optim import loader as L
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        sig = "planning.decompose"
        (home / "system" / "optim" / f"{sig}.json").write_text(json.dumps({
            "signature_name": sig, "optimized_instruction": "TUNED",
            "gate_arm": "token-F1 A/B, private holdout [paired-v2]"}))
        monkeypatch.setenv("GHOST_HOME", str(home))
        L.clear_cache()
        with caplog.at_level(logging.WARNING, logger=L.logger.name):
            L.tuned_instruction(sig, "BASE")
        L.clear_cache()
        assert not [r for r in caplog.records
                    if r.levelno >= logging.WARNING], \
            [r.getMessage() for r in caplog.records]

    def test_the_experiment_arm_is_NOT_the_gate_arm(self, tmp_path,
                                                    monkeypatch):
        """⚠ MY OWN FIX SHADOWED `_arm`. Naming the gate-identity string
        `_arm` inside `tuned_instruction` overwrote this request's
        EXPERIMENT arm, so every served turn was stamped with the gate
        identity instead of control/treatment — every trajectory's arm
        unreadable. Caught by the neighbouring tests within a minute."""
        from ghost_agent.optim import loader as L
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        sig = "planning.decompose"
        (home / "system" / "optim" / f"{sig}.json").write_text(json.dumps({
            "signature_name": sig, "optimized_instruction": "TUNED",
            "gate_arm": "token-F1 A/B, private holdout [paired-v2]"}))
        monkeypatch.setenv("GHOST_HOME", str(home))
        L.clear_cache()
        L.tuned_instruction(sig, "BASE", context=object(), req_id="r1")
        got = (L.served_for_request("r1") or {}).get(sig) or {}
        L.clear_cache()
        assert got.get("arm") in ("unenrolled", "control", "treatment"), got


class TestANonStringDescriptionDoesNotRaise:
    def test_both_arms_survive_it(self, tmp_path, monkeypatch):
        """⚠ Round 11 widened `len(baseline)` from one arm to both. Not
        reachable from TOOL_DEFINITIONS today; a crash in prompt assembly
        is not worth the bet."""
        from ghost_agent.optim import loader as L
        from ghost_agent.tools import registry as R
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        # ⚠ THE ARTIFACT MUST PASS THE VALIDATOR, or it is refused and
        # the inflation arithmetic — the line that raises — is never
        # reached. My first fixture used "T"*200, which the per-tool cap
        # rejects, so the mutant survived.
        _base = next(t for t in R.TOOL_DEFINITIONS
                     if t["function"]["name"] == "web_search"
                     )["function"]["description"]
        (home / "system" / "optim"
         / "tool_description.web_search.json").write_text(json.dumps({
             "signature_name": "tool_description.web_search",
             "optimized_instruction": _base + " Prefer it for news.",
             "gate_arm": "g"}))
        monkeypatch.setenv("GHOST_HOME", str(home))
        monkeypatch.setattr(R, "_TUNED_DESC_NAMES", None, raising=False)
        L.clear_cache()
        from ghost_agent.utils.logging import request_id_context
        tok = request_id_context.set("r-none")
        try:
            out = R._apply_tuned_descriptions(
                [{"type": "function",
                  "function": {"name": "web_search",
                               "description": None, "parameters": {}}}],
                context=object())
        finally:
            request_id_context.reset(tok)
            L.clear_cache()
            R._TUNED_DESC_NAMES = None
        assert len(out) == 1
        # The artifact must actually have been APPLIED, or the
        # arithmetic under test never ran.
        assert out[0]["function"]["description"] != None  # noqa: E711
        assert "Prefer it for news." in out[0]["function"]["description"]
