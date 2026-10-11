"""The GEPA gate/judge contract, enforced against the real files.

⚠ THIS FILE EXISTS BECAUSE §4DA DID NOT CONVERGE IN SIXTEEN REVIEW ROUNDS,
and the reason was structural. The ship *rule* settled by round 3. Almost
every defect found in rounds 4-16 was one shape: **a concept with no single
definition, restated in each of the files that touch it, with two of the
restatements disagreeing.**

  round 4   a corpus gap given the semantics of an outage
  round 5   the lesson never went upstream
  round 7   the gate stamped seven fields the only reader could not open
  round 9   the veto arm nobody gave the outage handling to
  round 11  the un-stamping was one-armed
  round 12  the fail-open was one-armed too
  round 13  a set-level win stamped on each member as its own
  round 14  the `excluded` bucket left outside the era filter
  round 15  four instruments, four exit vocabularies
  round 16  two gates, two seed-arm schemas — and the judge read one

Each review round samples ONE pair of restatements. There are O(n²) pairs
and the graph regrows whenever anything is added, so review cannot close
it: round 16's own fix reintroduced round 7's defect verbatim, in the seed
arm it had just ported.

The AST scans over the gate/judge scripts went with those scripts (§4NM);
what remains pins the contract module's own schema and exit vocabulary.
"""


import pytest

from ghost_agent.optim import gate_contract as GC


class TestTheSeedArmHasOneSchema:
    """⚠ THE ROUND-16 DEFECT, MEASURED. `run_gepa.py` wrote
    `overridden`/`seed_pass_rate`/`seed_wins`; the tool-description gate
    wrote `seed_loss_overridden`/`hand_written_pass_rate`/`vetoed`; and
    `recheck_gepa_incumbent.py` read `overridden` — so its
    "THAT PROMOTION USED --allow-seed-loss" warning was structurally
    unreachable for every artifact the second gate writes."""


    def test_the_builder_and_the_reader_agree_on_every_key(self):
        built = GC.build_seed_arm(
            seed_pass_rate=0.5, candidate_pass_rate=0.4,
            seed_minus_candidate_delta=0.1,
            seed_minus_candidate_raw_delta=0.08, n_usable_pairs=40,
            transport_excluded=2, seed_wins=6, candidate_wins=1,
            p_value=0.03, vetoed=True, overridden=True)
        back = GC.read_seed_arm({"seed_arm": built})
        assert back == built, (built, back)

    def test_an_override_of_nothing_is_refused(self):
        with pytest.raises(ValueError):
            GC.build_seed_arm(
                seed_pass_rate=0.5, candidate_pass_rate=0.6,
                seed_minus_candidate_delta=-0.1,
                seed_minus_candidate_raw_delta=-0.1, n_usable_pairs=40,
                vetoed=False, overridden=True)

    def test_the_reader_understands_BOTH_pre_contract_shapes(self):
        """Files written before the contract existed must not become
        unreadable — that would be the same defect one migration later."""
        old_gate1 = {"seed_arm": {"seed_pass_rate": 0.5,
                                  "candidate_pass_rate": 0.4,
                                  "overridden": True}}
        old_gate2 = {"seed_arm": {"hand_written_pass_rate": 0.5,
                                  "candidate_pass_rate": 0.4,
                                  "seed_loss_overridden": True}}
        for blk in (old_gate1, old_gate2):
            got = GC.read_seed_arm(blk)
            assert got["seed_pass_rate"] == 0.5, (blk, got)
            assert got["overridden"] is True, (blk, got)
            assert got["vetoed"] is True, (blk, got)

    def test_it_does_not_GUESS_the_veto_from_a_sign(self):
        """⚠ An earlier draft inferred `vetoed` from `delta > 0` — wrong
        for exactly the legacy files it was meant to help, because the
        two gates recorded that quantity with OPPOSITE signs. Unknown
        stays unknown."""
        got = GC.read_seed_arm({"seed_arm": {
            "seed_pass_rate": 0.5, "candidate_pass_rate": 0.4,
            "seed_minus_candidate_delta": 0.1}})
        assert got["vetoed"] is None, got


class TestTheTwoExitContractsAreNamedAndDistinct:
    """⚠ Round 15's journal claimed all four instruments share ONE exit
    contract. They cannot: a GATE's 0 means the incumbent was REPLACED, a
    JUDGE's 0 means it STANDS. What all four share is 2."""

    def test_the_shared_code_is_could_not_measure(self):
        assert GC.GateExit.COULD_NOT_MEASURE == 2
        assert GC.JudgeExit.COULD_NOT_MEASURE == 2
        assert GC.COULD_NOT_MEASURE == 2

    def test_the_two_zeros_mean_opposite_things(self):
        assert GC.GateExit.PROMOTED == 0 and GC.JudgeExit.STILL_WINS == 0
        assert GC.GateExit.REJECTED == 1
        assert GC.JudgeExit.NO_LONGER_WINS == 1


class TestTheValidatorsActuallyEXECUTE:
    """⚠ Lens A, F2: `validate_gate_record` — the check the contract
    module's own docstring calls "THE POINT" — had ZERO calls anywhere in
    the test tree; deleting its body survived 821 tests, and so did
    disabling either half of `validate_seed_arm`'s schema check. A
    validator nobody executes is documentation (`pin-must-fail-somewhere`).
    """

    def test_an_invented_key_is_refused(self):
        with pytest.raises(ValueError, match="no reader opens"):
            GC.validate_gate_record({"metric": "m", "my_private_field": 1},
                                    writer="test")

    def test_a_registered_key_set_passes(self):
        GC.validate_gate_record(
            {k: None for k in GC.GATE_RECORD_SHARED_KEYS}, writer="test")

    def test_a_missing_key_is_tolerated_but_an_extra_is_not(self):
        GC.validate_gate_record({"metric": "m"}, writer="test")
        with pytest.raises(ValueError):
            GC.validate_gate_record({"metric": "m", "extra_key_x": 1},
                                    writer="test")

    def test_the_nested_seed_arm_is_validated_too(self):
        with pytest.raises(ValueError):
            GC.validate_gate_record(
                {"metric": "m",
                 "seed_arm": {"overridden": True, "vetoed": False}},
                writer="test")

    def test_seed_arm_schema_refuses_BOTH_halves(self):
        """M12/M13/M14: `or`→`and`, missing-half off, extra-half off —
        each survived. Both halves, separately."""
        ok = {k: None for k in GC.SEED_ARM_KEYS}
        with pytest.raises(ValueError, match="unknown"):
            GC.validate_seed_arm({**ok, "invented": 1})
        missing = dict(ok)
        missing.pop("seed_pass_rate")
        with pytest.raises(ValueError, match="missing"):
            GC.validate_seed_arm(missing)

    def test_vetoed_and_undecidable_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="undecidable"):
            GC.build_seed_arm(
                seed_pass_rate=0.5, candidate_pass_rate=0.4,
                seed_minus_candidate_delta=0.1,
                seed_minus_candidate_raw_delta=0.1, n_usable_pairs=40,
                vetoed=True, undecidable=True)

    def test_the_delta_rate_identity_is_ENFORCED(self):
        """⚠ Lens C, B1: the first artifact written through this schema
        carried swapped rate fields — `seed_pass_rate: 0.9,
        candidate_pass_rate: 1.0, seed_minus_candidate_delta: +0.1` — so
        an artifact promoted BECAUSE it lost recorded a win. The
        identity `delta == seed_rate - candidate_rate` is the only check
        that can see a swap; names cannot."""
        with pytest.raises(ValueError, match="mislabelled"):
            GC.validate_seed_arm({
                **{k: None for k in GC.SEED_ARM_KEYS},
                "seed_pass_rate": 0.9, "candidate_pass_rate": 1.0,
                "seed_minus_candidate_delta": 0.1})
        GC.validate_seed_arm({
            **{k: None for k in GC.SEED_ARM_KEYS},
            "seed_pass_rate": 1.0, "candidate_pass_rate": 0.9,
            "seed_minus_candidate_delta": 0.1})

    def test_p_value_precision_survives_the_builder(self):
        """M29: p rounded to ONE decimal survived — p=0.0156 recorded as
        0.0, 'significant at any bar' in the audit trail."""
        b = GC.build_seed_arm(
            seed_pass_rate=0.5, candidate_pass_rate=0.4,
            seed_minus_candidate_delta=0.1,
            seed_minus_candidate_raw_delta=0.1, n_usable_pairs=40,
            p_value=0.015625)
        assert b["p_value"] == 0.015625, b["p_value"]
