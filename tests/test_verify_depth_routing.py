"""§4BQ: verification DEPTH plumbing (self-consistency n=3 on a deep turn).

⚠ The complexity router that decided the `verify_depth` arm was REMOVED;
the decision-rule and handle_chat stamp tests went with it. What remains
pins the depth knob, the wire into the cheap leg, and the registered spec.

On a turn the complexity router called confidently HARD (label=hard, not
escalated), the cheap leg's adjudication runs self-consistency n=3
(majority vote) instead of a single sample.

WHY THIS CONSUMER AND NOT THE OBVIOUS ONE. The MCTS depth gate is the
router's only historical consumer and it is switched off pending an
execution-grounded value function (no trained PRM exists), disabled by a
paired ablation measuring 78% vs 80% at ~1.8x latency. Re-enabling it
would re-introduce a measured-harmful mechanism. Self-consistency is a
different trade entirely:

  * it targets the MISS side — the judge's own confidence carries NO
    signal about false-CONFIRM (AUC 0.5087, chance), while cross-sample
    disagreement does;
  * majority-of-3 was measured to fix `artifact_leak` (36% of the miss
    mass), and `clean` never varied, so the false-positive gate survives;
  * the samples run on the CRITIC node, so none of it lands on the main
    model — and a sample that degrades onto the main route stops the vote.

Routing by difficulty is what makes it affordable: n=3 on the ~58% of
turns the router calls confidently hard, rather than on all of them.

⚠ THE COST CLAIM HERE WAS ONCE WRONG, AND WRONG BECAUSE IT WAS REASONED
RATHER THAN MEASURED. This file previously said the samples ran under
`asyncio.gather` so the added wall-clock was "~one sample", and that the
trigger fired on ~41% of turns. Timed against the live critic (§4BR R17):
gather cost 24.9s vs 10.8s for one sample — and 17.8s for the SEQUENTIAL
loop its comment dismissed, because llama.cpp's per-slot prompt cache
makes a repeat of the same prompt nearly free while concurrent samples
land on different slots and each re-prefills 2,285 tokens. The live
corpus put the trigger rate at 58.1% (122/210), not 41%. Both numbers
now come from measurements; the sampler is sequential and early-stops
once the majority is decided (~13.0s in the common case).

⚠ THE PREMISE IS UNTESTED. "Harder turns are likelier wrong" has never
been shown here — the corpus carries no independent correctness signal,
which is why the verdict sidecar exists. So this ships as a RANDOMISED
ARM, not a default: both arms compute and stamp the trigger, and the
deploy is the measurement. §4AN's lesson was that this router was
accurate overall and worthless where it acted.

These tests pin the SAFETY properties, not the benefit (the benefit is
the arm's job):
  1. depth reaches ONLY the cheap leg — never the main-model escalations;
  2. the code-output route, which has no two-stage leg, is untouched and
     is NAMED so the analysis can condition on it;
  3. routing raises a floor and never lowers an operator's setting;
  4. it is killable;
  5. a failure in the routing can never fail a verdict.
"""

import os
import re
from pathlib import Path

import pytest

from ghost_agent.core import verifier as V

_AGENT = Path(__file__).resolve().parents[1] / "src" / "ghost_agent" / "core" / "agent.py"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("GHOST_VERIFY_DEPTH_ROUTING", raising=False)
    monkeypatch.delenv("GHOST_VERIFY_SELF_CONSISTENCY", raising=False)


class TestTheDepthKnob:
    def test_a_normal_turn_is_unchanged(self):
        assert V._self_consistency_n() == 1

    def test_a_confident_hard_turn_gets_majority_of_three(self):
        assert V._self_consistency_n(deep=True) == 3

    def test_routing_raises_a_floor_and_never_lowers(self, monkeypatch):
        """An operator who set 5 globally keeps 5 on hard turns."""
        monkeypatch.setenv("GHOST_VERIFY_SELF_CONSISTENCY", "5")
        assert V._self_consistency_n(deep=True) == 5
        assert V._self_consistency_n() == 5

    def test_it_is_killable(self, monkeypatch):
        monkeypatch.setenv("GHOST_VERIFY_DEPTH_ROUTING", "0")
        assert V._self_consistency_n(deep=True) == 1

    def test_the_kill_switch_is_read_per_call(self, monkeypatch):
        """Same idiom as _two_stage_enabled: flippable without a restart."""
        assert V._self_consistency_n(deep=True) == 3
        monkeypatch.setenv("GHOST_VERIFY_DEPTH_ROUTING", "false")
        assert V._self_consistency_n(deep=True) == 1


class TestTheWireIsConnectedEndToEnd:
    """Execution, not grep: does `deep` actually change how many
    adjudications happen?"""

    @staticmethod
    def _verifier(calls):
        from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
        v = V.Verifier(llm_client=None)

        async def _call(prompt, *a, **k):
            # Stage 1 is the FIRST call by construction; everything after it
            # is an adjudication sample. Keyword-sniffing the prompt made
            # this test pass vacuously when stage 1 failed.
            calls.append("enumerate" if not calls else "adjudicate")
            if k.get("route_out") is not None:
                k["route_out"]["route"] = "critic"
            return {"suspects": [{"quote": "q", "check": "support",
                                  "reason": "r"}],
                    "verdict": "CONFIRMED", "confidence": 0.8,
                    "reasoning": "r", "issues": []}

        object.__setattr__(v, "_call_llm", _call)
        return v

    def _run(self, deep):
        import asyncio
        calls = []
        v = self._verifier(calls)
        asyncio.run(v._verify_claim_two_stage("c", "e", "ctx", deep=deep))
        return calls

    def test_a_normal_turn_adjudicates_once(self):
        assert self._run(deep=False).count("adjudicate") == 1

    def test_a_deep_turn_adjudicates_more_than_once(self):
        """The mutation that survived 19 green tests: if `deep` stops
        reaching `_self_consistency_n`, this drops back to 1."""
        assert self._run(deep=True).count("adjudicate") > 1

    def test_stage_one_still_runs_exactly_once(self):
        """Depth samples the ADJUDICATION only. Re-running the enumeration
        would multiply the expensive half for no vote."""
        assert self._run(deep=True).count("enumerate") == 1


class TestDepthReachesTheCheapLegOnly:
    """The two `force_main=True` call sites are main-model escalations.
    Multiplying those would put 3x on the 35B for the most expensive turns
    — the exact cost profile this consumer exists to avoid."""

    @staticmethod
    def _src():
        return Path(V.__file__).read_text()

    def test_the_cheap_leg_receives_deep(self):
        """Structural: the textual version pinned the whole argument list
        and broke when an unrelated kwarg was added, which is a test
        guarding formatting rather than behaviour."""
        import ast
        import inspect
        import textwrap
        # §4IN: the incumbent pipeline body lives in `_verify_claim_incumbent`
        src = textwrap.dedent(inspect.getsource(V.Verifier._verify_claim_incumbent))
        calls = [c for c in ast.walk(ast.parse(src)) if isinstance(c, ast.Call)
                 and getattr(c.func, "attr", "") == "_verify_claim_two_stage"]
        kws = [{k.arg: k.value for k in c.keywords} for c in calls]
        cheap = [k for k in kws if "force_main" not in k]
        main = [k for k in kws if "force_main" in k]
        assert len(cheap) == 1, f"expected one cheap-leg call, got {len(cheap)}"
        assert isinstance(cheap[0].get("deep"), ast.Name)
        assert cheap[0]["deep"].id == "deep"
        # `verify_claim` also contains the `_reverify_on_main` closure. That
        # leg is the main-model escalation and must never be given depth.
        for k in main:
            assert "deep" not in k, "an escalation leg was given depth"

    def test_no_force_main_call_site_receives_deep(self):
        src = self._src()
        for m in re.finditer(r"_verify_claim_two_stage\((.*?)\)", src, re.S):
            args = m.group(1)
            if "force_main=True" in args:
                assert "deep" not in args, args

    def test_the_scope_is_explained_where_it_is_enforced(self):
        """A future reader must not 'fix' the asymmetry by threading deep
        into the escalations."""
        src = self._src()
        assert "CHEAP leg ONLY" in src


class TestTheUntreatableRouteIsExcludedAndNamed:
    """`verify_code_output` has no two-stage leg (one _call_llm, by
    design), so there is no adjudication to sample. Passing `deep` there
    would be an accepted-and-ignored parameter."""

    @staticmethod
    def _agent_src():
        return _AGENT.read_text()

    def test_code_output_is_not_given_deep(self):
        src = self._agent_src()
        call = src[src.index("verifier.verify_code_output("):]
        call = call[:call.index(")")]
        assert "deep=" not in call, call

    def test_verify_code_output_does_not_accept_deep(self):
        """If it ever grows the parameter without a two-stage leg, that is
        the dead-mechanism shape."""
        import inspect
        params = inspect.signature(V.Verifier.verify_code_output).parameters
        assert "deep" not in params

    def test_the_route_is_recorded_for_the_analysis(self):
        """A confident-hard turn that took the code route gets NO
        treatment. Diluting the arm with those silently would understate
        the effect."""
        src = self._agent_src()
        # §4FN round 4: the recording block is one recorder
        # (`_record_verdict_instruments`); the choke point hands it the
        # route and it forwards the route to turn-facts and the sidecar.
        assert 'verify_route=_verify_route)' in src
        assert 'verify_route=str(verify_route)' in src
        assert '_verify_route = "code_output"' in src


class TestTheDecisionPoint:
    @staticmethod
    def _agent_src():
        return _AGENT.read_text()

    def test_it_is_computed_beside_high_stakes(self):
        """`_high_stakes` is the RETROSPECTIVE signal and carries a comment
        explaining that this is the ONE place every verdict is produced —
        so depth can never be live on one delivery path and dark on
        another. The prospective signal belongs in the same place."""
        src = self._agent_src()
        hs = src.index("_high_stakes = _turn_had_tool_failure")
        deep = src.index("_deep = False")
        assert 0 < deep - hs < 2000, deep - hs

    def test_EVERY_verify_claim_call_forwards_the_ROUTED_value(self):
        """§4BR R17 MAJOR-7: mutating `deep=_deep` to `deep=False` left the
        whole suite green — the feature was inert and nothing noticed.

        Structural rather than textual on purpose: this fails on a literal
        (`deep=False`), on a dropped kwarg, and on a THIRD call site added
        later that forgets it — which is the realistic way this wire gets
        cut, since there are already two.
        """
        import ast
        tree = ast.parse(self._agent_src())
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.AsyncFunctionDef)
                  and n.name == "_compute_verifier_verdict")
        calls = [c for c in ast.walk(fn)
                 if isinstance(c, ast.Call)
                 and getattr(c.func, "attr", "") == "verify_claim"]
        assert len(calls) == 2, f"expected 2 verify_claim call sites, got {len(calls)}"
        for c in calls:
            kw = {k.arg: k.value for k in c.keywords}
            assert "deep" in kw, "a verify_claim call forwards no depth at all"
            assert isinstance(kw["deep"], ast.Name), (
                f"deep must forward the routed variable, not "
                f"{ast.dump(kw['deep'])}")
            assert kw["deep"].id == "_deep"

    @staticmethod
    def _depth_decision_fn(tree):
        """The function that OWNS the depth read, located by WHAT IT DOES.

        ⚠ These pins named `_compute_verifier_verdict` and broke when the
        decision was extracted into a callable predicate (§4EC) — while every
        property they assert still held. A pin bound to a location tests the
        location. Found instead by "reads trigger_flags for
        verify_depth_fired", and asserted unique so a second decision site
        fails here rather than silently splitting the rule.
        """
        import ast
        owners = [n for n in ast.walk(tree)
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and any(isinstance(c, ast.Call)
                          and getattr(c.func, "attr", "") == "trigger_flags"
                          for c in ast.walk(n))
                  and "verify_depth_fired" in ast.dump(n)]
        assert len(owners) == 1, (
            f"the depth decision must live in exactly one function, found "
            f"{[o.name for o in owners]}")
        return owners[0]

    def test_routing_can_never_fail_a_verdict(self):
        """A routing fault must degrade to control depth, never fail a turn.

        ⚠ REPINNED 2026-08-31 (§4EC) FROM A CHARACTER WINDOW. This read the
        2200 chars after `_deep = False` and grepped them for
        `except Exception` — a selector that says nothing about structure
        and breaks on any insertion, which is exactly what adding a second
        trigger did. Structural now: every assignment that DECIDES `_deep`
        must sit inside a `try` with an `except Exception` handler, located
        by the AST. The one exception is the safe default `_deep = False`,
        which is what a swallowed fault degrades TO.
        """
        import ast
        tree = ast.parse(self._agent_src())
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and n.name == "_compute_verifier_verdict")

        guarded_lines = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Try) and any(
                    isinstance(h.type, ast.Name) and h.type.id == "Exception"
                    for h in node.handlers):
                for inner in ast.walk(node):
                    guarded_lines.add(getattr(inner, "lineno", -1))

        decisions, safe_default = [], 0
        for node in ast.walk(fn):
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(t, ast.Name) and t.id == "_deep"
                       for t in node.targets):
                continue
            if isinstance(node.value, ast.Constant) and node.value.value is False:
                safe_default += 1
                continue                      # the degrade-to target itself
            if node.lineno not in guarded_lines:
                decisions.append(node.lineno)

        assert safe_default == 1, (
            "the `_deep = False` safe default is missing or duplicated; a "
            "swallowed routing fault has nothing well-defined to degrade to")
        assert not decisions, (
            f"`_deep` is decided outside a try/except at lines {decisions} — "
            "a routing fault would fail the verdict instead of degrading "
            "to control depth")
        # ...and the guarded region must actually decide it, or the pin is
        # satisfied by a function that never routes at all.
        assert any(isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "_deep"
                           for t in n.targets)
                   and n.lineno in guarded_lines
                   for n in ast.walk(fn)), "no guarded `_deep` decision at all"
