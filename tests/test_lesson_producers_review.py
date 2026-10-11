"""Fresh review of the lesson PRODUCERS (dream, self-play, bench, episodes,
distilled, learn_skill) — 2026-10-03. Each test names the world it fails in.
§4NM: only ``learn_skill`` / ``operator_repair`` write lessons now, so the
tests that exercise a learn_lesson mechanic write with ``source="learn_skill"``;
the dream fragment cache is gone with REM."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ghost_agent.memory.lesson_quality import is_actionable_lesson
from ghost_agent.memory.skills import SkillMemory


@pytest.mark.parametrize("mistake", ["None observed; the solution was direct and efficient.", "None.",
                                     "N/A - no mistake", "No mistakes were made", "There was no error"])
def test_a_no_mistake_phrasing_needs_an_actionable_fix(mistake):
    """Fails in the world where only the exact words none/n/a counted, so an
    observation ("The solution correctly…") passed as a correction."""
    assert is_actionable_lesson(mistake, "The solution correctly parsed the log in two passes.", "x") is False
    assert is_actionable_lesson(mistake, "When parsing logs, always strip trailing whitespace first.", "x") is True


def test_a_fix_identical_to_the_mistake_is_no_lesson():
    assert is_actionable_lesson("x = a / b", "x = a / b", "Division") is False


@pytest.mark.parametrize("fix", ["Make stdout reference a token from input.txt so the validator passes.",
                                 "Satisfy the checker by printing the expected header first.",
                                 "Read the hidden test's expectations before coding."])
def test_a_rule_about_the_test_harness_is_refused(tmp_path, fix):
    """Fails in the world where self-play rules taught the agent to game
    its own validator."""
    sm = SkillMemory(tmp_path)
    assert sm.learn_lesson("When the output format is unclear", "Guessed the format", fix, source="self_play") is None


def test_a_rule_restating_its_word_problem_is_scoped_to_that_request(tmp_path):
    """The bench producer's restating rule was refused; §4NM removed bench. The
    live writer (learn_skill) keeps a rule that restates its request, but
    SCOPED to that request — a general rule stays general."""
    from ghost_agent.memory.lesson_scope import lesson_scope, SCOPE_GENERAL, SCOPE_REQUEST
    sm = SkillMemory(tmp_path)
    challenge = "Jan is 30 years old now and his son is 16. How old was Jan two years ago?"
    out = sm.learn_lesson("Age word problems involving relative time shifts", "Forgot to shift every age",
                          "If Jan is 30 now, two years ago Jan was 28 and his son was 14.",
                          source="learn_skill", generality_context=challenge)
    assert out == "written"
    assert lesson_scope(sm._load_playbook()[0]) == SCOPE_REQUEST
    (tmp_path / "general").mkdir()
    sm = SkillMemory(tmp_path / "general")      # a scoped trigger is never reused by a general row
    out = sm.learn_lesson("Age word problems involving relative time shifts", "Forgot to shift every age",
                          "Shift every person's age by the same interval before comparing them.",
                          source="learn_skill", generality_context=challenge)
    assert out == "written"
    assert lesson_scope(sm._load_playbook()[0]) == SCOPE_GENERAL


def _archive(tmp_path, trigger, reason):
    with open(tmp_path / "skills_pruned_archive.jsonl", "a") as fh:
        fh.write(json.dumps({"reason": reason, "lesson": {"trigger": trigger}}) + "\n")


def test_a_retracted_lesson_is_not_relearned(tmp_path):
    """Fails in the world where the next dream cycle re-minted a lesson the
    operator had just retracted."""
    sm = SkillMemory(tmp_path)
    _archive(tmp_path, "When a final turn is reached, always answer directly", "removed_by_trigger")
    assert sm.learn_lesson("When a final turn is reached, always answer directly", "none",
                           "Always provide a direct answer on the final turn.", source="dream") is None


def test_a_cap_trimmed_lesson_may_come_back(tmp_path):
    sm = SkillMemory(tmp_path)
    _archive(tmp_path, "When parsing dates from logs", "playbook_cap_trim")
    assert sm.learn_lesson("When parsing dates from logs", "guessed", "Use ISO 8601.", source="learn_skill") == "written"


def test_a_replaced_fix_brings_its_mistake(tmp_path):
    """Fails in the world where a re-learned row's new pattern sat next to
    the previous pattern's mistake."""
    sm = SkillMemory(tmp_path)
    t = "When handling tool/files paths"
    sm.learn_lesson(t, "Assuming paths exist", "Check the path exists.", trigger=t, source="learn_skill")
    sm.learn_lesson(t, "Trusting one unverified source", "Always validate critical information against a second source first.",
                    trigger=t, source="learn_skill")
    row = sm._load_playbook()[0]
    assert row["mistake"] == "Trusting one unverified source" and row["solution"].startswith("Always validate")


@pytest.mark.parametrize("distance,rows", [(0.03, 1), (0.2, 2)])
def test_a_near_identical_twin_from_another_producer_merges(tmp_path, distance, rows):
    """Fails in the world where any cross-producer twin became its own row,
    however close. (An incoming learn_skill lesson is never absorbed into
    another producer's row, so the merge is exercised with the operator's
    repair source.)"""
    sm = SkillMemory(tmp_path)
    a = "When a web page returns 403, try an archived copy"
    sm.save_playbook([{"trigger": a, "task": a, "mistake": "m", "solution": "Use archive.org", "source": "reflection"}])
    sm._find_duplicate_lesson = lambda *x, **k: {"source": "vector", "trigger": a, "text": "", "distance": distance}
    sm.learn_lesson("If a site answers 403, use a cached copy", "m", "Use a cached copy.", MagicMock(), source="operator_repair")
    assert len(sm._load_playbook()) == rows


def test_learn_skill_checks_the_request_in_hand():
    import asyncio
    from ghost_agent.core.bus import MemoryBus
    from ghost_agent.memory.lesson_scope import current_request
    skill = MagicMock()
    skill.learn_lesson.return_value = "written"
    bus = MemoryBus(skill_memory=skill)
    tok = current_request.set("count the lines in notes.txt")
    try:
        asyncio.run(bus.publish_fact("learn_skill", {"skill": {"task": "t", "mistake": "m", "solution": "s"}}))
    finally:
        current_request.reset(tok)
    kw = skill.learn_lesson.call_args.kwargs
    assert kw["source"] == "learn_skill" and kw["generality_context"] == "count the lines in notes.txt"






