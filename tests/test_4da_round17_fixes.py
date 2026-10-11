"""§4DA post-redesign round 1 — the bindings, not the names.

Three lenses reviewed the redesign. The name half of shape 1 held (every
defect found uses only registered keys and declared codes); what they found
is the BINDING half — which arm a named field receives, which state a
declared code maps to, whether a flag means "the condition held" or "the
check was refused" — plus two axes the enumerated world-space held constant:
call topology and the input-set = prompt-set identity.

The worst finding sat inside the redesign's own schema: the tool-description
gate's seed arm recorded the two pass rates SWAPPED (the decision helper's
"incumbent" slot holds the candidate on that call), so the artifact promoted
under --allow-seed-loss — promoted BECAUSE it lost to the hand-written text —
carried rates saying it won. `validate_seed_arm` now enforces
`delta == seed_rate - candidate_rate`, which is the only check a swap cannot
pass; names cannot see it.

§4NM: the ship gates, `scripts/run_gepa.py` and the recheck reader were
removed; their tests went with them. The read-site / request-state /
gate_contract reader pins remain.
"""

from types import SimpleNamespace

from ghost_agent.optim import gate_contract as GC, loader as L
from ghost_agent.tools import registry as R
from ghost_agent.utils.logging import request_id_context

from tests.test_4da_round14_fixes import _base


class TestAnOverCapOverrideIsNotServed:
    """⚠ Lens B, R3: deleting the override-path validation survived. An
    override is the offline knob (`_TOOL_DESC_OVERRIDES`), but the cap
    guards the KV-pinned tools block wherever the text comes from."""

    def test_the_baseline_survives_a_runaway_override(self, monkeypatch):
        monkeypatch.setitem(R._TOOL_DESC_OVERRIDES, "web_search",
                            "x" * 60_000)
        got = R._tuned_tool_description("web_search", _base("web_search"))
        assert got == _base("web_search"), (
            "a 60,000-char override reached the prompt")

    def test_a_sane_override_IS_served(self, monkeypatch):
        monkeypatch.setitem(R._TOOL_DESC_OVERRIDES, "web_search",
                            _base("web_search") + " Override.")
        got = R._tuned_tool_description("web_search", _base("web_search"))
        assert got.endswith("Override."), got


class TestTheReaderBindingsForLegacyAndDefaults:
    """⚠ Final pass, finding 4: the absent-`undecidable` default (flip to
    True survived, printing "AN OUTAGE ATE THE PAIRS" for every legacy
    record). §4NM: the recheck reader that printed it is gone; the
    contract reader's default is what remains to pin."""

    def test_a_legacy_record_is_not_called_an_outage(self):
        """`undecidable` absent means False."""
        assert GC.read_seed_arm(
            {"seed_arm": {"seed_pass_rate": 0.5,
                          "candidate_pass_rate": 0.4}})["undecidable"] \
            is False
