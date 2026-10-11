"""§4DA round 3 — the stamp that lied, the key that collided, the torn write.

Round 2 wired the tool-description read site into §4CZ's attribution so a
promoted artifact could be judged by production turns and reverted. Round 3's
production-safety lens found that the artefact round 2 shipped — the stamp —
can assert "this artifact served this turn" for turns where nothing reached
the model, that the randomization key collides on 17 of the 70 live tool
names, and that the promotion write ports half of the `run_gepa` discipline
its own comment cites.

All three corrupt the causal claim `--revert` acts on, which is the one thing
the round was for.
"""

import json
import os
import re
import sys
from pathlib import Path

import pytest

from ghost_agent.optim import loader as L
from ghost_agent.tools import registry as R


# ══════════════════════════════════════════════════════════════════════
# MAJOR-1 — the stamp fired at LOAD time; the refusal happens two layers up
# ══════════════════════════════════════════════════════════════════════
class TestTheStampMatchesWhatTheModelACTUALLY_saw:
    @staticmethod
    def _artifact(home, tool, text, *, gate=True):
        d = home / "system" / "optim"
        d.mkdir(parents=True, exist_ok=True)
        payload = {"signature_name": f"tool_description.{tool}",
                   "optimized_instruction": text}
        if gate:
            payload["gate_arm"] = "tool-choice fidelity A/B, private holdout"
        (d / f"tool_description.{tool}.json").write_text(json.dumps(payload))

    @staticmethod
    def _build(tools, req_id):
        from ghost_agent.utils.logging import request_id_context
        token = request_id_context.set(req_id)
        try:
            return R._apply_tuned_descriptions(tools, context=object())
        finally:
            request_id_context.reset(token)

    def _tools(self, names):
        out = []
        for n in names:
            t = next(t for t in R.TOOL_DEFINITIONS
                     if (t.get("function") or {}).get("name") == n)
            out.append({"type": "function",
                        "function": {"name": n,
                                     "description":
                                         t["function"]["description"],
                                     "parameters": {}}})
        return out

    @pytest.fixture(autouse=True)
    def _clean(self):
        L.clear_cache()
        R._TUNED_DESC_NAMES = None
        yield
        L.clear_cache()
        R._TUNED_DESC_NAMES = None

    def test_the_AGGREGATE_reject_leaves_no_served_stamp(self, tmp_path,
                                                         monkeypatch):
        """⚠ MEASURED: 8 individually-valid artifacts summing past the
        20,000 ceiling → 40 of 40 requests rendered baselines ONLY, while
        21 treatment turns carried the served-stamp and `gepa_live_check`
        returned a KEEP verdict comparing two arms whose prompts were
        BYTE-IDENTICAL. `activation_stats` already knew the truth
        (`applied: 0, rejected: 21`); only the stamp disagreed."""
        home = tmp_path / "home"
        monkeypatch.setenv("GHOST_HOME", str(home))
        names = [t["function"]["name"] for t in R.TOOL_DEFINITIONS[:4]]
        for n in names:
            base = next(t for t in R.TOOL_DEFINITIONS
                        if t["function"]["name"] == n)["function"]["description"]
            self._artifact(home, n, base + " " + ("y" * 400))
        monkeypatch.setattr(R, "_TOOL_DESC_AGGREGATE_SLACK", 100)
        R._TUNED_DESC_NAMES = None
        L.clear_cache()

        tools = self._tools(names)
        out = self._build(tools, "req-agg-1")
        for t_in, t_out in zip(tools, out):
            assert (t_out["function"]["description"]
                    == t_in["function"]["description"]), \
                "the ceiling did not fire — this test proves nothing"
        assert L.served_for_request("req-agg-1") == {}, (
            "a turn that saw only baselines was stamped as SERVED the "
            "artifact — live_check would compare identical prompts")

    def test_the_PER_TOOL_reject_leaves_no_served_stamp(self, tmp_path,
                                                        monkeypatch):
        """The other refusal point, which fires per tool rather than for
        the set."""
        home = tmp_path / "home"
        monkeypatch.setenv("GHOST_HOME", str(home))
        n = R.TOOL_DEFINITIONS[0]["function"]["name"]
        # Far over any per-tool cap, so `_validate_tool_description` refuses.
        self._artifact(home, n, "z" * 50_000)
        R._TUNED_DESC_NAMES = None
        L.clear_cache()
        tools = self._tools([n])
        out = self._build(tools, "req-cap-1")
        assert (out[0]["function"]["description"]
                == tools[0]["function"]["description"]), \
            "the per-tool validator did not fire"
        assert L.served_for_request("req-cap-1") == {}

    def test_an_ACCEPTED_artifact_IS_stamped(self, tmp_path, monkeypatch):
        """⚠ THE ADMIT SIDE. Un-stamping everything closes the false
        positives and destroys the mechanism — `--revert` needs the stamp
        on the turns the artifact really served."""
        home = tmp_path / "home"
        monkeypatch.setenv("GHOST_HOME", str(home))
        n = R.TOOL_DEFINITIONS[0]["function"]["name"]
        base = R.TOOL_DEFINITIONS[0]["function"]["description"]
        self._artifact(home, n, base + " Prefer it for current events.")
        monkeypatch.setattr(R, "_TOOL_DESC_AGGREGATE_SLACK", 20_000)
        R._TUNED_DESC_NAMES = None
        L.clear_cache()
        tools = self._tools([n])
        out = self._build(tools, "req-ok-1")
        assert out[0]["function"]["description"] != \
            tools[0]["function"]["description"], "nothing was swapped"
        served = L.served_for_request("req-ok-1")
        assert f"tool_description.{n}" in served, (
            "a genuinely-served artifact lost its stamp — --revert has "
            "nothing to judge")

    def test_unnote_leaves_OTHER_signatures_alone(self):
        """One refused tool must not clear a sibling's stamp."""
        L._SERVED_RING.clear()
        L._note_served("r1", "tool_description.a", "sha", "treatment")
        L._note_served("r1", "tool_description.b", "sha", "treatment")
        L.unnote_served("r1", "tool_description.a")
        assert L.served_for_request("r1") == {
            "tool_description.b": {"sha": "sha", "arm": "treatment"}}
        L.unnote_served("r1", "tool_description.b")
        assert L.served_for_request("r1") == {}
        L._SERVED_RING.clear()

    def test_unnote_never_raises(self):
        for args in (("", "x"), ("nope", "x"), ("r", ""), (None, None)):
            L.unnote_served(*args)


# ══════════════════════════════════════════════════════════════════════
# MAJOR-2 — the randomization key collided on live names
# ══════════════════════════════════════════════════════════════════════
class TestTheExperimentNameIsUNIQUE:
    """⚠ `_NAME_RE` caps at 40 chars and `gepa_` + `tool_description_`
    spends 22, leaving 18 characters of tool name. The 39 static tools are
    clean; the read site also covers COMPOSED SKILLS, and over the 70 live
    names a bare truncation produced 7 collision groups covering 17 names.
    Colliding signatures are never independently randomized — control
    withholds BOTH and treatment serves BOTH — so `--revert` on one can
    retire an artifact whose measured loss belongs to the other."""

    COLLIDING = [
        "tool_description.auto_file_system_file_system_execute",
        "tool_description.auto_file_system_file_system_report_pdf",
        "tool_description.auto_file_system_manage_projects",
        "tool_description.auto_file_system_manage_services",
        "tool_description.auto_file_system_file_system_manage_services",
    ]

    def test_long_names_do_not_collide(self):
        seen = {}
        for n in self.COLLIDING:
            e = L.experiment_name(n)
            assert e not in seen, (
                f"{n} and {seen[e]} share the experiment name {e} — one "
                f"--revert would retire the other's artifact")
            seen[e] = n

    def test_short_names_are_UNCHANGED(self):
        """The three GEPA signatures already registered must keep the
        names an operator may already have in experiments.json."""
        assert L.experiment_name("planning.decompose") == \
            "gepa_planning_decompose"
        assert L.experiment_name("tool_selection.pick") == \
            "gepa_tool_selection_pick"
        assert L.experiment_name("tool_description.web_search") == \
            "gepa_tool_description_web_search"

    def test_the_suffix_is_derived_from_the_WHOLE_name(self):
        """Two names differing only past the truncation point must differ."""
        a = L.experiment_name("tool_description." + "x" * 40 + "aaa")
        b = L.experiment_name("tool_description." + "x" * 40 + "bbb")
        assert a != b


# ══════════════════════════════════════════════════════════════════════
# MAJOR-4 — "" from a REGISTERED experiment is control, not "serve it"
# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
# MAJOR-3 — the promotion write was not atomic
# ══════════════════════════════════════════════════════════════════════
