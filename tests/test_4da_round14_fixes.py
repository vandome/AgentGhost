"""§4DA round 14 — the prune was arm-dependent again, and it deleted a stamp
it still needed.

Round 11's fix stated its premise in the code: *"whether the set is dropped does
not depend on which arm THIS turn drew."* That is contradicted by the comment
180 lines above it in the same file, which cites a measurement. The union of the
two ceiling branches is arm-independent **only while every artifact is at least
as long as its baseline** — and the reflector is explicitly told to "propose a
SHORTER one" when the per-tool cap bites.

Driven with one artifact 5,000 chars shorter than its baseline and five others
already serving: `treatment 156/313 vs control 96/132, REVERT p=5.16e-06`, on an
artifact **neutral by construction**. A control draw of the short one pushed the
sum over the ceiling; a treatment draw did not.

And the hypothetical branch deleted stamps for descriptions it still **rendered**:
194 of 200 turns rendered a tuned description and 0 of 200 kept a stamp — so the
live check saw nothing forever, and every one of those turns was mined into the
pool that trains and ship-gates the next run as if no arm had mutated it.
"""

import json
import sys

import pytest

from ghost_agent.optim import loader as L
from ghost_agent.tools import registry as R
from ghost_agent.utils.logging import request_id_context



def _mk(home, name, text):
    (home / "system" / "optim"
     / f"tool_description.{name}.json").write_text(json.dumps({
         "signature_name": f"tool_description.{name}",
         "optimized_instruction": text, "gate_arm": "g"}))


def _base(name):
    return next(t for t in R.TOOL_DEFINITIONS
                if t["function"]["name"] == name)["function"]["description"]


def _render(names, req, ctx):
    tools = [{"type": "function",
              "function": {"name": n, "description": _base(n),
                           "parameters": {}}} for n in names]
    tok = request_id_context.set(req)
    try:
        return R._apply_tuned_descriptions(tools, context=ctx)
    finally:
        request_id_context.reset(tok)


class TestThePruneDecisionIsArmINVARIANT:
    def test_the_SAME_turn_prunes_the_same_way_whichever_arm_it_drew(
            self, tmp_path, monkeypatch):
        """The invariant, stated directly rather than through counts: the
        prune reads `_worst_inflation`, which is the positive deltas of
        every artifact the turn could render — a property of the TURN."""
        import ast
        from pathlib import Path
        src = Path("src/ghost_agent/tools/registry.py").read_text()
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_apply_tuned_descriptions")
        body = ast.unparse(fn)
        # ⚠ THE PRUNE'S CONDITION, PARSED. It must read `_worst_inflation`
        # and NOTHING arm-shaped: gating it on `_withheld_names` let a
        # turn that drew treatment everywhere escape entirely, which is
        # the same de-randomization one conjunct out.
        import ast as _ast
        prune = None
        for n in _ast.walk(fn):
            if (isinstance(n, _ast.If)
                    and "_exclude_optim_served" in _ast.unparse(n)):
                prune = _ast.unparse(n.test)
                break
        assert prune is not None, "no prune branch found"
        assert "_worst_inflation" in prune, prune
        assert "_withheld_names" not in prune, (
            "the prune is gated on an ARM-SHAPED quantity: a turn that "
            "drew treatment on every artifact escapes it — " + prune)
        assert "_withheld_inflation" not in prune, prune
        assert "max(0," in body, "the worst case is not a positive part"


class TestAnAllTreatmentTurnDoesNotEscapeThePrune:
    def test_the_prune_condition_reads_NOTHING_arm_shaped(self):
        import ast
        from pathlib import Path
        src = Path("src/ghost_agent/tools/registry.py").read_text()
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_apply_tuned_descriptions")
        prune = next(ast.unparse(n.test) for n in ast.walk(fn)
                     if isinstance(n, ast.If)
                     and "_exclude_optim_served" in ast.unparse(n))
        assert prune.strip() == \
            "_worst_inflation > _TOOL_DESC_AGGREGATE_SLACK", prune


class TestTheLivenessProbesSeeEveryLoadLine:
    def test_all_three_load_shapes_match(self):
        """⚠ Round 13 added a THIRD load line ("was promoted UNGATED")
        and neither probe matched it — so the instrument whose job is to
        catch a silently-inoperative subsystem went blind precisely for
        the artifacts adopted with NO A/B."""
        import re
        from ghost_agent.core import liveness as LV
        pat = re.compile(LV._GEPA_LOAD_PATTERN)
        for line in (
                "GEPA: loaded tuned instruction for x (10 chars, sha "
                "abc12345, gate g)",
                "GEPA: artifact 'x' (sha abc12345) predates the gate "
                "schema — no gate identity",
                "GEPA: artifact 'x' (sha abc12345) was promoted UNGATED "
                "(--no-ab-gate): no A/B measured it"):
            assert pat.search(line), line

    def test_the_applies_probe_uses_the_SAME_pattern(self):
        """A second, narrower copy is how the two views came to disagree
        in the first place."""
        import ast
        from pathlib import Path
        src = Path("src/ghost_agent/core/liveness.py").read_text()
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_gepa_applies_probe")
        body = ast.unparse(fn)
        assert "_GEPA_LOAD_PATTERN" in body, body[:400]
        assert "'GEPA: loaded tuned instruction'" not in body


class TestTheSetLevelCaveatReachesTheReader:

    def test_a_SOLO_promotion_says_nothing_of_the_kind(self, tmp_path):
        import subprocess
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        sig = "tool_description.web_search"
        (home / "system" / "optim" / f"{sig}.json").write_text(json.dumps({
            "signature_name": sig, "optimized_instruction": "X",
            "gate_arm": "g",
            "gate": {"n_private": 60, "delta": 0.1, "min_delta": 0.02,
                     "co_promoted": [sig]}}))
        r = subprocess.run(
            [sys.executable, "scripts/recheck_gepa_incumbent.py",
             "--signature", sig, "--home", str(home)],
            capture_output=True, text=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "src",
                 "HOME": str(__import__("pathlib").Path.home()),
                 "GHOST_HOME": str(home)})
        assert "SET-LEVEL EVIDENCE" not in r.stdout, r.stdout
