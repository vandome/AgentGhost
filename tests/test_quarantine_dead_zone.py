"""A quarantined lesson must never absorb a later correct re-learn.

Before the fix, `_find_duplicate_lesson` had no quarantine awareness: a
re-learn matched the quarantined row, returned "reinforced", bumped ITS
frequency — and `_filter_quarantined` then kept that row out of every prompt.
The new correct lesson was written nowhere and retrievable nowhere, so the
topic became permanently un-learnable, while each bump raised the quarantined
row's utility and protected it from `prune_low_utility`.
"""
from __future__ import annotations

import pytest

from ghost_agent.memory.skills import SkillMemory


@pytest.fixture
def mem(tmp_path):
    return SkillMemory(tmp_path)


def test_relearning_a_quarantined_lesson_creates_a_live_entry(mem):
    mem.learn_lesson(task="deploy the web app", mistake="used port 80",
                     solution="ask the supervisor for a port", source="learn_skill")
    assert mem.quarantine_lesson("deploy the web app") >= 1

    # The corrected lesson, learned again from a real failure.
    mem.learn_lesson(task="deploy the web app", mistake="hardcoded the port",
                     solution="call manage_services and use $PORT", source="learn_skill")

    ctx = mem.get_playbook_context("deploy the web app")
    assert "manage_services" in ctx, (
        "the re-learned lesson must be retrievable; it was swallowed by the "
        "quarantined row")


def test_the_quarantined_row_is_still_suppressed(mem):
    mem.learn_lesson(task="deploy the web app", mistake="used port 80",
                     solution="POISONED ADVICE", source="learn_skill")
    mem.quarantine_lesson("deploy the web app")
    mem.learn_lesson(task="deploy the web app", mistake="hardcoded the port",
                     solution="call manage_services and use $PORT", source="learn_skill")
    ctx = mem.get_playbook_context("deploy the web app")
    assert "POISONED ADVICE" not in ctx


def test_quarantined_row_does_not_absorb_the_frequency_bump(mem):
    mem.learn_lesson(task="deploy the web app", mistake="a", solution="b", source="learn_skill")
    mem.quarantine_lesson("deploy the web app")
    mem.learn_lesson(task="deploy the web app", mistake="c", solution="d", source="learn_skill")
    playbook = mem._load_playbook()
    quarantined = [p for p in playbook if p.get("quarantined")]
    assert quarantined, "the quarantined row should still exist for review"
    assert all(int(p.get("frequency") or 1) == 1 for p in quarantined), (
        "a quarantined row must not accumulate frequency from re-learns")

