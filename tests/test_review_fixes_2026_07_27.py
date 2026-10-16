"""Defects found reviewing the same day's own changes — 2026-07-27 (later 6).

Four bugs, three of them introduced earlier the same session. They are the
kind that a green suite does not catch, because each only fires on a path
the tests never construct: a penalised turn that is then corrected, a
correction on an already-failed turn, and an LLM reply whose labels contain
more than one number.

D4 is the important one: a "tolerant" coercion added hours earlier turned a
recoverable parse failure into a SILENT INVERSION of the model's verdict.
"""

import re

import pytest


SRC = (__import__("pathlib").Path(__file__).resolve().parents[1]
       / "src" / "ghost_agent" / "core" / "agent.py").read_text()


# ──────────────────────────────────────────────────────────────────────
# D4 — survivor-index coercion must not fabricate an index
# ──────────────────────────────────────────────────────────────────────

def _coerce(label, n_hypotheses):
    """Mirrors the parsing in agent.py's hypothesis adjudication."""
    out = set()
    m = re.match(r"\D*(\d+)", str(label))
    if m:
        ix = int(m.group(1))
        if 0 <= ix < n_hypotheses:
            out.add(ix)
    return out


class TestSurvivorIndexCoercion:
    @pytest.mark.parametrize("label,expected", [
        ("H0", {0}),
        ("1", {1}),
        ("hypothesis 2", {2}),
        # The regression: stripping ALL non-digits concatenated separate
        # numbers. "H2: retry after 30s" became 230.
        ("H2: retry after 30s", {2}),
        ("H0 — the config at line 42 is wrong", {0}),
        # Out of range must be DISCARDED, not kept as a phantom survivor.
        ("H97", set()),
        ("no numbers here", set()),
        ("", set()),
    ])
    def test_first_digit_run_only_and_range_checked(self, label, expected):
        assert _coerce(label, 3) == expected

    def test_prose_label_no_longer_inverts_the_verdict(self):
        """With the old `re.sub(r"\\D","")`, a prose label produced a
        non-empty out-of-range set — truthy — so the promote/demote loop ran
        and marked EVERY deferred hypothesis refuted. The model said H2
        survived; the code concluded all were refuted."""
        old = {int(re.sub(r"\D", "", "H2: retry after 30s"))}
        assert old == {230}, "documents the old behaviour"
        assert 2 not in old, "the real survivor was lost"
        new = _coerce("H2: retry after 30s", 3)
        assert new == {2}


# ──────────────────────────────────────────────────────────────────────
# D1/D2/D3 — the user-correction stash
# ──────────────────────────────────────────────────────────────────────
