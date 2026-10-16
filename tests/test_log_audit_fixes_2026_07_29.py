"""Live-log audit fixes — 2026-07-29 (daytime log, boots 09:47/09:49).

Eleven defects/improvements from one morning's ghost-agent.log:

1.  Self-play short-circuit skipped the confirmation turn on exit 0 alone,
    shipping a wrong-FORMAT solution straight to the validator (request 9D:
    per-resource debug lines vs `TotalAllocatedQuantity: <sum>`).
2.  Validator-selftest instrumenter exit-43'd on common validator shapes
    (expected built inside `def main()`, inline f-string compares, loosely
    named holders) — the echo gate was dark for exactly the challenge that
    then failed on output mismatch.
3.  `metacog calib refit=ok` logged in the same cycle whose probability map
    was REJECTED as anti-correlated.
4.  "no JSON twin to bump" was mostly a WRONG LOOKUP (bump searched the
    playbook under the incoming trigger, not the stored duplicate's), and
    true orphan vectors permanently vetoed re-learns.
5.  Generator negative-examples block quoted 12 full same-skeleton openers
    — in-context reinforcement of the banned shape.
6.  REM announced "Entering REM cycle / dreaming over 40 digests" before the
    freshness gate skipped (4× churn in 2.5h); the hook re-woke every 30min.
7.  Warmup and live prefill logs measured different segments in different
    units — warmup effectiveness unverifiable.
8.  Thinking display muted mid-turn when reasoning QUOTED the rule text
    ("Emit EXACTLY ONE `<tool_call>` block") — the quoted marker latched
    stop_printing.
9.  REM minted "When asked about the weather, use the system_utility tool."
    as a lesson — bare topic→tool routing the registry already encodes.
10. "Targeting cluster 'None'" rendered a literal None.
11. Cheap judge refuted a weather reply for the subjective gloss
    "warm and clear" (overturned on escalation, 24s of a 43s turn).

§4NM: items 2, 5 and 10 lived in synthetic self-play (validator self-test
instrumenter, challenge-generator fingerprints, seed cluster rendering);
their tests were removed with it.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = (ROOT / "src" / "ghost_agent" / "core" / "agent.py").read_text()


# ──────────────────────────────────────────────────────────────────────
# 1. Short-circuit format gate — _challenge_output_prefixes
# ──────────────────────────────────────────────────────────────────────

from ghost_agent.core.agent import _challenge_output_prefixes


class TestChallengeOutputPrefixes:
    def test_extracts_the_9d_shape(self):
        """The live failure: challenge pinned `TotalAllocatedQuantity: <sum>`."""
        text = ("Compute the total. Output exactly one line: "
                "`TotalAllocatedQuantity: <sum>` where <sum> is an integer.")
        assert _challenge_output_prefixes(text) == ["TotalAllocatedQuantity:"]

    def test_brace_placeholder_and_equals_label(self):
        assert _challenge_output_prefixes("print `Result: {n}`") == ["Result:"]
        assert _challenge_output_prefixes("emit `count = <n>`") == ["count ="]

    def test_column_sketch_without_placeholder_is_ignored(self):
        """Request 84's shape — `user_id net_balance` is a sketch, not a
        literal, and must NOT become a veto token."""
        assert _challenge_output_prefixes(
            "each line: `user_id net_balance` (2 decimal places)") == []

    def test_code_comparison_is_not_a_template(self):
        assert _challenge_output_prefixes("check `if x < 5` in your code") == []

    def test_bare_placeholder_and_lone_punctuation_rejected(self):
        assert _challenge_output_prefixes("print `<sum>`") == []
        assert _challenge_output_prefixes("write `: <x>`") == []

    def test_dedup_and_empty(self):
        text = "line `Total: <a>` then again `Total: <b>`"
        assert _challenge_output_prefixes(text) == ["Total:"]
        assert _challenge_output_prefixes("") == []
        assert _challenge_output_prefixes(None) == []

    def test_gate_semantics_veto_and_pass(self):
        """Mirrors the short-circuit condition: veto only when a pinned
        prefix exists and stdout carries none of them."""
        want = _challenge_output_prefixes("Output: `TotalAllocatedQuantity: <sum>`")
        wrong_stdout = "res_1: max_priority=10, allocated=1\nres_2: ..."
        right_stdout = "TotalAllocatedQuantity: 15"
        assert want and not any(p in wrong_stdout for p in want)   # veto
        assert any(p in right_stdout for p in want)                # pass

    def test_short_circuit_wired_to_the_gate(self):
        """The sim short-circuit must consult the format gate before
        force-stopping."""
        assert "_challenge_output_prefixes(_user_txt)" in AGENT_SRC
        assert "_format_ok" in AGENT_SRC


# (3. Calibration map_status honesty — removed with the calibration refit, §4PJ)



# ──────────────────────────────────────────────────────────────────────
# 4. Skill store — reworded-twin bump + orphan self-heal + reconcile
#    (§4NM: written as the live writer, learn_skill)
# ──────────────────────────────────────────────────────────────────────

from ghost_agent.memory.skills import (
    SkillMemory, _dup_trigger_from_vector_text,
)


class _FakeCollection:
    """Minimal chroma-shaped stub for the dedup/reconcile paths."""

    def __init__(self, query_hit=None, get_result=None):
        self._hit = query_hit
        self._get = get_result or {"ids": [], "metadatas": [], "documents": []}
        self.deleted_ids = []

    def query(self, query_texts, n_results, where):
        if self._hit is None:
            return {"distances": [[]], "ids": [[]], "documents": [[]],
                    "metadatas": [[]]}
        return {
            "distances": [[self._hit["distance"]]],
            "ids": [[self._hit["id"]]],
            "documents": [[self._hit["text"]]],
            "metadatas": [[{"trigger": self._hit.get("trigger", "")}]],
        }

    def get(self, where=None, limit=None, include=None):
        return self._get

    def delete(self, ids=None, where=None):
        self.deleted_ids.extend(ids or [])


class _FakeMemorySystem:
    def __init__(self, collection):
        self.collection = collection
        self.added = []

    def add(self, text, meta):
        self.added.append((text, meta))


_REAL_MISTAKE = "Used a relative path inside the Docker sandbox."
_FIX = "Always use absolute paths inside the Docker sandbox."


class TestVectorDedupTwinLookup:
    def test_reworded_twin_bumps_under_the_stored_trigger(self, tmp_path):
        """The 'no JSON twin' log was mostly this: the bump searched the
        playbook under the INCOMING trigger while the twin sat under its
        own (reworded) trigger, recoverable from the vector metadata."""
        sm = SkillMemory(tmp_path)
        stored_trigger = "Relative path errors in Docker sandbox execution"
        assert sm.learn_lesson(stored_trigger, _REAL_MISTAKE, _FIX, source="learn_skill") == "written"

        hit = {"id": "vec-1", "distance": 0.05, "trigger": stored_trigger,
               "text": f"SITUATION: {stored_trigger}\nMISTAKE: x\nSOLUTION: y"}
        ms = _FakeMemorySystem(_FakeCollection(query_hit=hit))
        out = sm.learn_lesson(
            "Docker sandbox failing on relative file paths",  # reworded
            _REAL_MISTAKE, _FIX, memory_system=ms, source="learn_skill")
        assert out == "reinforced"
        lesson = sm.list_lessons(scope="all", limit=10)[0]
        assert int(lesson.get("frequency", 1)) == 2
        assert ms.collection.deleted_ids == []

    def test_true_orphan_is_dropped_and_lesson_written_fresh(self, tmp_path):
        """Vector hit with NO playbook entry under either trigger: the old
        path skipped the write forever; now it deletes the orphan and
        writes the lesson."""
        sm = SkillMemory(tmp_path)
        hit = {"id": "orphan-7", "distance": 0.05,
               "trigger": "Completely unrelated stored trigger",
               "text": "SITUATION: Completely unrelated stored trigger\n"
                       "MISTAKE: m\nSOLUTION: s"}
        ms = _FakeMemorySystem(_FakeCollection(query_hit=hit))
        out = sm.learn_lesson(
            "Relative path errors in Docker sandbox execution",
            _REAL_MISTAKE, _FIX, memory_system=ms, source="learn_skill")
        assert out == "written"
        assert ms.collection.deleted_ids == ["orphan-7"]
        assert len(sm.list_lessons(scope="all", limit=10)) == 1

    def test_dup_trigger_recovered_from_document_text(self):
        txt = "SITUATION: Parse JSON safely\nMISTAKE: none\nSOLUTION: use json"
        assert _dup_trigger_from_vector_text(txt) == "Parse JSON safely"
        assert _dup_trigger_from_vector_text("") == ""
        assert _dup_trigger_from_vector_text("garbage") == ""


class TestReconcileVectorOrphans:
    def test_deletes_only_clear_orphans(self, tmp_path):
        sm = SkillMemory(tmp_path)
        live_trigger = "Relative path errors in Docker sandbox execution"
        sm.learn_lesson(live_trigger, _REAL_MISTAKE, _FIX, source="learn_skill")
        coll = _FakeCollection(get_result={
            "ids": ["keep-1", "drop-1", "skip-1"],
            "metadatas": [
                {"trigger": live_trigger},              # has twin → keep
                {"trigger": "Ancient scrubbed lesson about tor circuits"},
                {},                                     # unidentifiable → keep
            ],
            "documents": ["", "", ""],
        })
        ms = _FakeMemorySystem(coll)
        n = sm.reconcile_vector_orphans(ms)
        assert n == 1
        assert coll.deleted_ids == ["drop-1"]

    def test_truncated_metadata_trigger_still_matches_twin(self, tmp_path):
        """A 200-char-truncated metadata trigger must match its full JSON
        twin (token-subset), not be deleted as an orphan."""
        sm = SkillMemory(tmp_path)
        long_trigger = ("Repeated timeout failures when scraping paginated "
                        "vendor catalogs through the anonymous Tor circuit "
                        "pool with per-engine racing and adaptive circuit "
                        "selection enabled for resilient onion searches "
                        "across multiple engines simultaneously")
        assert len(long_trigger) > 200
        sm.learn_lesson(long_trigger, _REAL_MISTAKE, _FIX, source="learn_skill")
        coll = _FakeCollection(get_result={
            "ids": ["v-long"],
            "metadatas": [{"trigger": long_trigger[:200]}],
            "documents": [""],
        })
        n = sm.reconcile_vector_orphans(_FakeMemorySystem(coll))
        assert n == 0
        assert coll.deleted_ids == []

    def test_no_collection_is_a_noop(self, tmp_path):
        sm = SkillMemory(tmp_path)
        assert sm.reconcile_vector_orphans(None) == 0
        assert sm.reconcile_vector_orphans(SimpleNamespace(collection=None)) == 0

    def test_wired_into_the_store_hygiene_idle_phase(self):
        assert "reconcile_vector_orphans" in AGENT_SRC


# ──────────────────────────────────────────────────────────────────────
# 6. REM churn — hook backoff (§4NM: the REM cycle itself is gone; the
#    announce-after-gate and pool-thin pins went with it)
# ──────────────────────────────────────────────────────────────────────

class TestRemChurn:
    def test_hook_backoff_scales_the_cooldown(self):
        assert "_dream_skip_streak" in AGENT_SRC
        assert "_dream_cooldown_eff" in AGENT_SRC
        # Streak is capped so a long quiet stretch can't push REM out
        # indefinitely.
        assert 'min(int(getattr(self, "_dream_skip_streak", 0)), 3)' in AGENT_SRC

    def test_a_pass_that_wrote_nothing_mints_no_ledger_row(self):
        # §4NM: the "REM cycle ran" ledger is gone; only a pass that wrote
        # something ledgers (under "maintenance"), the rest heartbeat.
        assert "if not _wrote:" in AGENT_SRC     # behaviour: tests/test_4nm_batch5.py


# ──────────────────────────────────────────────────────────────────────
# 7. Warmup vs live prefix — comparable hashes
# ──────────────────────────────────────────────────────────────────────

class TestPrefixHashComparability:
    def test_both_log_lines_carry_the_sys_slot_hash(self):
        # Warmup line and the live Prefill Cache line share the marker.
        assert AGENT_SRC.count("sys h=") >= 2

    def test_live_line_hashes_the_actual_system_message(self):
        assert 'if m.get("role") == "system"' in AGENT_SRC


# ──────────────────────────────────────────────────────────────────────
# 8. Stop-marker quoted-mention guard
# ──────────────────────────────────────────────────────────────────────

from ghost_agent.core.stream_guards import _tail_has_stop_marker


def _stream(chunks):
    """Feed chunks like the display loop does; True if the mute ever latches."""
    buf = ""
    for c in chunks:
        buf += c
        if _tail_has_stop_marker(buf, c):
            return True
    return False


class TestQuotedMarkerMention:
    def test_backtick_quoted_tool_call_does_not_mute(self):
        """The live truncation: thinking QUOTED the constraint text."""
        chunks = ['Wait - the rule says "Emit EXACTLY ONE ',
                  '`<tool_call', '>` block per turn". So I should...']
        assert _stream(chunks) is False

    def test_real_tool_call_transition_still_latches(self):
        assert _stream(["I will call the tool now.\n", "<tool_call", ">"]) is True

    def test_marker_straddling_chunks_still_latches(self):
        assert _stream(["answer\n<tool", "_call>"]) is True

    def test_quoted_think_close_does_not_mute(self):
        assert _stream(['the tag `</think', '`` is special']) is False

    def test_real_think_close_latches(self):
        assert _stream(["done reasoning</think", ">"]) is True

    def test_double_quoted_mention_does_not_mute(self):
        assert _stream(['it says "<tool_call', '>" verbatim']) is False


# ──────────────────────────────────────────────────────────────────────
# 9. Trivial tool-routing floor
# ──────────────────────────────────────────────────────────────────────

from ghost_agent.memory.lesson_quality import (
    is_actionable_lesson, _is_actionable_heuristic,
)


class TestTrivialToolRoutingFloor:
    def test_the_live_minted_skill_is_now_rejected(self):
        assert is_actionable_lesson(
            "None",
            "When asked about the weather, use the system_utility tool.",
            "When asked about the weather") is False

    @pytest.mark.parametrize("rule", [
        "When asked about the weather, use the system_utility tool.",
        "If asked for the current time, call system_utility",
        "Whenever the user asks about disk space, use the `system` tool.",
        "When asked to search the web, always use the search tool.",
    ])
    def test_bare_routing_shapes_rejected(self, rule):
        assert _is_actionable_heuristic(rule) is False

    @pytest.mark.parametrize("rule", [
        # Qualifiers past the tool name carry real content — keep.
        "When asked about the weather, use the system_utility tool and "
        "include humidity and wind speed.",
        "When asked about the weather, use system_utility with "
        "location defaulting to Athens.",
        # Non-routing heuristics unaffected.
        "Always use absolute paths in Docker.",
        "When a tool fails twice, escalate to the operator instead of retrying.",
    ])
    def test_substantive_rules_still_pass(self, rule):
        assert _is_actionable_heuristic(rule) is True

    def test_real_mistake_lessons_unaffected(self):
        assert is_actionable_lesson(
            "Called the tool with a relative path and it failed.",
            "Use absolute paths.",
            "Docker sandbox path errors") is True


# ──────────────────────────────────────────────────────────────────────
# 11. Cheap-judge subjective-gloss rule
# ──────────────────────────────────────────────────────────────────────

class TestSubjectiveGlossRule:
    def test_present_in_single_prompt_rubric(self):
        from ghost_agent.core.verifier import _VERIFY_CLAIM_PROMPT
        assert "SUBJECTIVE characterizations" in _VERIFY_CLAIM_PROMPT
        assert "warm and clear" in _VERIFY_CLAIM_PROMPT

    def test_present_in_adjudication_rubric(self):
        from ghost_agent.core.verifier import _VERIFY_ADJUDICATE_PROMPT
        assert "SUBJECTIVE characterizations" in _VERIFY_ADJUDICATE_PROMPT
        assert "FALSE ALARMS" in _VERIFY_ADJUDICATE_PROMPT
