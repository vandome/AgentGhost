"""§4PX (request 3252f3fb, "forget all proposed rules"): a PROPOSED rule can be
dismissed — one ("dismiss rule N") or all ("dismiss all proposed rules") —
with a preview and an explicit confirm; a message about the rules that is no
command tells the model it cannot change them (it listed the LESSONS and
named 26 of them as the proposed rules)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ghost_agent.core import failure_replay as FR


def _ledger(tmp_path):
    """Rules 1 adopted, 2 and 3 proposed, 4 forgotten, 5 still under test."""
    FR.save(tmp_path, [
        {"source_id": "a", "stage": "done", "rule": "Adopted rule text.", "adopted_at": 2.0},
        {"source_id": "b", "stage": "done", "rule": "Waiting rule B."},
        {"source_id": "c", "stage": "done", "rule": "Waiting rule C."},
        {"source_id": "d", "stage": "done", "rule": "Gone rule.", "adopted_at": 1.0, "forgotten_at": 3.0},
        {"source_id": "e", "stage": "test", "rule": "Still testing."},
    ])


def _states(tmp_path):
    return [FR.rule_state(e) for e in FR.load(tmp_path)]


def _cmd(text, tmp_path, sm=None):
    return FR.owner_rule_command(text, tmp_path, sm)


# ── the commands parse ────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "forget all proposed rules", "dismiss all proposed rules", "ok, dismiss the candidate rules",
    "reject all pending rules.", "delete all of the proposed rules", "clear my proposed rules",
    "decline all proposed rules", "dismiss all the suggested rules",
    "“dismiss all proposed rules”"])
def test_the_bulk_dismiss_parses(text):
    assert FR._DISMISS_ALL_RE.match(text) and FR.is_rule_command(text)


@pytest.mark.parametrize("text", [
    "forget all rules", "forget all the rules about parking", "dismiss all proposed rules of the club",
    # §4PZ r3: no "all"/"my" — the model's own suggestions as often
    "dismiss the suggested rules", "remove the suggested rules", "clear pending rules", "drop the pending rules",
    "can you forget all proposed rules?"])
def test_a_wider_or_longer_message_is_not_the_bulk_command(text):
    assert not FR._DISMISS_ALL_RE.match(text)


@pytest.mark.parametrize("text", ["yes", "confirm", "yes, dismiss all proposed rules",
                                  "confirm dismiss all rules", "confirm dismiss rule 2 and 3"])
def test_only_the_explicit_phrase_confirms(text):
    assert not FR._CONFIRM_ALL_RE.match(text) and not FR._CONFIRM_RE.match(text)


def test_the_confirm_phrases_parse():
    for t in ("confirm dismiss all proposed rules", "ok, confirm forget all the proposed rules!",
              "confirm decline all proposed rules",                    # r1: every preview verb confirms
              "confirm dismiss rule 2", "confirm reject rule #3."):
        assert FR.is_rule_command(t), t


# ── one rule ──────────────────────────────────────────────────────────

def test_dismiss_previews_then_dismisses_on_confirm(tmp_path):
    _ledger(tmp_path)
    p = _cmd("dismiss rule 2", tmp_path)
    assert "confirm dismiss rule 2" in p.banner and "Waiting rule B." in p.banner
    assert _states(tmp_path)[1] == "proposed"                         # nothing changed yet
    c = _cmd("confirm dismiss rule 2", tmp_path)
    assert c.banner.startswith("✓ Rule 2 dismissed") and "do not call any tool" in c
    assert _states(tmp_path)[:3] == ["adopted", "dismissed", "proposed"]
    assert "DISMISSED" in _cmd("show rule 2", tmp_path).banner


def test_a_confirm_without_a_fresh_preview_dismisses_nothing(tmp_path, monkeypatch):
    _ledger(tmp_path)
    assert "first" in _cmd("confirm dismiss rule 2", tmp_path).banner
    _cmd("dismiss rule 2", tmp_path)
    later = FR.time.time() + FR.FORGET_CONFIRM_S + 5
    monkeypatch.setattr(FR.time, "time", lambda: later)
    assert "first" in _cmd("confirm dismiss rule 2", tmp_path).banner
    assert _states(tmp_path)[1] == "proposed"


def test_a_refusal_uses_the_owners_verb(tmp_path):
    _ledger(tmp_path)
    assert "nothing to forget" in _cmd("forget rule 4", tmp_path).banner
    assert "nothing to forget" in _cmd("confirm forget rule 4", tmp_path).banner
    assert "nothing to dismiss" in _cmd("dismiss rule 4", tmp_path).banner


def test_show_offers_both_decisions_on_a_proposed_rule(tmp_path):
    _ledger(tmp_path)
    b = _cmd("show rule 2", tmp_path).banner
    assert "learn rule 2" in b and "dismiss rule 2" in b
    assert "dismiss rule 1" not in _cmd("show rule 1", tmp_path).banner
    assert "dismiss rule 5" not in _cmd("show rule 5", tmp_path).banner     # r2: still under test


@pytest.mark.parametrize("n,want", [(1, "forget rule 1"), (4, "forgotten earlier"), (5, "still being tested")])
def test_only_a_proposed_rule_is_dismissed(tmp_path, n, want):
    _ledger(tmp_path)
    before = _states(tmp_path)
    assert want in _cmd(f"dismiss rule {n}", tmp_path).banner
    assert want in _cmd(f"confirm dismiss rule {n}", tmp_path).banner
    assert _states(tmp_path) == before


def test_forget_on_an_adopted_rule_is_still_the_forget(tmp_path):
    _ledger(tmp_path)
    assert "confirm forget rule 1" in _cmd("forget rule 1", tmp_path).banner


def test_a_rule_adopted_after_the_preview_is_not_dismissed(tmp_path):
    _ledger(tmp_path)
    _cmd("dismiss rule 2", tmp_path)
    entries = FR.load(tmp_path)
    entries[1]["adopted_at"] = FR.time.time()                          # "learn rule 2" in between
    FR.save(tmp_path, entries)
    assert "adopted" in _cmd("confirm dismiss rule 2", tmp_path).banner
    assert _states(tmp_path)[1] == "adopted"


def test_learn_rule_brings_a_dismissed_rule_back(tmp_path):
    (tmp_path / "mem").mkdir()
    _ledger(tmp_path)
    from ghost_agent.memory.skills import SkillMemory
    sm = SkillMemory(tmp_path / "mem")
    _cmd("dismiss rule 2", tmp_path)
    _cmd("confirm dismiss rule 2", tmp_path)
    assert "✓ Rule 2 adopted" in _cmd("learn rule 2", tmp_path, sm).banner
    assert _states(tmp_path)[1] == "adopted" and FR.is_adopted(FR.load(tmp_path)[1])
    assert any(r.get("solution") == "Waiting rule B." for r in json.loads(sm.file_path.read_text()))


@pytest.mark.parametrize("bulk", [True, False])
def test_a_stale_writer_between_preview_and_confirm_keeps_the_preview(tmp_path, bulk):
    """r1: the replay stage holds its copy across minutes of replays and saves
    after the owner's preview — the confirm must still find the preview."""
    _ledger(tmp_path)
    stale = FR.load(tmp_path)
    _cmd("dismiss all proposed rules" if bulk else "dismiss rule 2", tmp_path)
    FR.save(tmp_path, stale)
    b = _cmd("confirm dismiss all proposed rules" if bulk else "confirm dismiss rule 2", tmp_path).banner
    assert b.startswith("✓") and _states(tmp_path)[1] == "dismissed"


def test_a_stale_writer_never_undoes_a_dismiss(tmp_path):
    _ledger(tmp_path)
    stale = FR.load(tmp_path)
    _cmd("dismiss all proposed rules", tmp_path)
    _cmd("confirm dismiss all proposed rules", tmp_path)
    FR.save(tmp_path, stale)                                           # the idle stage's older copy
    assert _states(tmp_path)[1:3] == ["dismissed", "dismissed"]


def test_the_trim_keeps_a_dismissed_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(FR, "_MAX_ENTRIES", 3)
    FR.save(tmp_path, [{"source_id": "x", "stage": "done", "rule": "R.", "dismissed_at": 1.0}]
            + [{"source_id": f"s{i}", "stage": "abandoned"} for i in range(5)])
    assert FR.load(tmp_path)[0]["source_id"] == "x"


def test_only_the_newest_dismissed_rules_outlive_the_trim(tmp_path, monkeypatch):
    """r1: kept forever, dismissed rows would fill the ledger and every save would
    drop the case the replay stage just opened."""
    monkeypatch.setattr(FR, "_MAX_ENTRIES", 4)          # r2: small enough that the cap BINDS
    monkeypatch.setattr(FR, "DISMISSED_KEPT", 2)
    # dismissed long ago (1970), the NEWEST last in time but listed first
    rows = [{"source_id": f"d{i}", "stage": "done", "rule": f"R{i}.", "dismissed_at": float(10 - i)} for i in range(5)]
    rows.append({"source_id": "open", "stage": "base"})
    FR.save(tmp_path, rows)
    ids = [e["source_id"] for e in FR.load(tmp_path)]
    assert "open" in ids and {"d0", "d1"} <= set(ids) and "d2" not in ids    # by time, not position


def test_open_cases_keep_room_when_the_kept_rows_fill_the_ledger(tmp_path, monkeypatch):
    """r3: the dismissed hold has no cap — the open case being saved must still survive."""
    monkeypatch.setattr(FR, "_MAX_ENTRIES", 30)                     # floor = 30 // 15 = 2
    now = FR.time.time()
    rows = [{"source_id": f"d{i}", "stage": "done", "rule": f"R{i}.", "dismissed_at": now - i} for i in range(40)]
    FR.save(tmp_path, rows + [{"source_id": "old", "stage": "abandoned"}, {"source_id": "open", "stage": "base"}])
    ids = [e["source_id"] for e in FR.load(tmp_path)]
    assert ids[-2:] == ["old", "open"] and len(ids) == 42


def test_a_rule_dismissed_inside_the_replay_window_is_never_trimmed(tmp_path, monkeypatch):
    """r2: trimmed, its failure could be picked again and re-proposed."""
    monkeypatch.setattr(FR, "_MAX_ENTRIES", 3)
    monkeypatch.setattr(FR, "DISMISSED_KEPT", 1)
    now = FR.time.time()
    rows = [{"source_id": f"d{i}", "stage": "done", "rule": f"R{i}.", "dismissed_at": now - i} for i in range(4)]
    FR.save(tmp_path, rows + [{"source_id": f"s{i}", "stage": "abandoned"} for i in range(4)])
    assert {f"d{i}" for i in range(4)} <= {e["source_id"] for e in FR.load(tmp_path)}


# ── every proposed rule ───────────────────────────────────────────────

def test_dismiss_all_previews_exactly_the_proposed_ones(tmp_path):
    _ledger(tmp_path)
    b = _cmd("forget all proposed rules", tmp_path).banner
    assert "these 2 proposed rules" in b and "**Rule 2**" in b and "**Rule 3**" in b
    assert "Adopted rule" not in b and "Gone rule" not in b and "Still testing" not in b
    assert "confirm dismiss all proposed rules" in b and "adopted rules are not touched" in b
    assert _states(tmp_path)[1:3] == ["proposed", "proposed"]


def test_confirm_all_dismisses_the_shown_ones_and_nothing_else(tmp_path):
    _ledger(tmp_path)
    _cmd("dismiss all proposed rules", tmp_path)
    c = _cmd("confirm dismiss all proposed rules", tmp_path)
    assert c.banner.startswith("✓ Dismissed rules 2, 3") and "Still waiting" not in c.banner
    assert _states(tmp_path) == ["adopted", "dismissed", "dismissed", "forgotten", "proposed"]
    assert "Dismissed" in _cmd("show all rules", tmp_path).banner


def test_a_rule_proposed_after_the_list_is_left_waiting(tmp_path):
    _ledger(tmp_path)
    _cmd("dismiss all proposed rules", tmp_path)
    entries = FR.load(tmp_path)
    entries.append({"source_id": "f", "stage": "done", "rule": "Brand new."})
    entries[2]["adopted_at"] = FR.time.time()                          # rule 3 adopted in between
    FR.save(tmp_path, entries)
    b = _cmd("confirm dismiss all proposed rules", tmp_path).banner
    assert b.startswith("✓ Dismissed rule 2 ") and "Still waiting" in b and "rule 6" in b
    assert "rule 3 is adopted now" in b                                # r1: shown, then changed — said
    assert _states(tmp_path)[1:3] == ["dismissed", "adopted"] and _states(tmp_path)[5] == "proposed"


def test_a_repeated_confirm_says_already_done(tmp_path):
    _ledger(tmp_path)
    _cmd("dismiss all proposed rules", tmp_path)
    _cmd("confirm dismiss all proposed rules", tmp_path)
    b = _cmd("confirm dismiss all proposed rules", tmp_path).banner
    assert b.startswith("Already done — rules 2, 3 were dismissed")
    assert "do not call any tool" in _cmd("confirm dismiss all proposed rules", tmp_path)


@pytest.mark.parametrize("between", ["learn", "single"])
def test_already_done_only_when_this_batch_did_it_and_it_still_holds(tmp_path, between):
    """r3: a rule adopted since, or dismissed one by one, is not 'already done' by the batch."""
    (tmp_path / "mem").mkdir()
    _ledger(tmp_path)
    _cmd("dismiss all proposed rules", tmp_path)
    if between == "learn":
        _cmd("confirm dismiss all proposed rules", tmp_path)
        from ghost_agent.memory.skills import SkillMemory
        _cmd("learn rule 2", tmp_path, SkillMemory(tmp_path / "mem"))
    else:
        _cmd("confirm dismiss rule 2", tmp_path)
        _cmd("confirm dismiss rule 3", tmp_path)
    b = _cmd("confirm dismiss all proposed rules", tmp_path).banner
    if between == "learn":       # r4: rule 3 WAS done by this batch; rule 2 changed since — both said
        assert b == "Already done — rule 3 was dismissed. Left as it is: rule 2 is adopted now."
    else:                        # dismissed one by one — not by the batch
        assert b.startswith("Nothing was dismissed") and "rule 2 is dismissed now" in b


def test_a_stale_writer_after_the_confirm_keeps_already_done(tmp_path):
    _ledger(tmp_path)
    _cmd("dismiss all proposed rules", tmp_path)
    stale = FR.load(tmp_path)                                          # the replay stage's copy
    _cmd("confirm dismiss all proposed rules", tmp_path)
    FR.save(tmp_path, stale)
    assert _cmd("confirm dismiss all proposed rules", tmp_path).banner.startswith("Already done — rules 2, 3")


def test_a_long_batch_line_is_capped(tmp_path):
    FR.save(tmp_path, [{"source_id": f"p{i}", "stage": "done", "rule": f"R{i}."} for i in range(30)])
    _cmd("dismiss all proposed rules", tmp_path)
    b = _cmd("confirm dismiss all proposed rules", tmp_path).banner
    assert "and 10 more" in b and ", 21," not in b


def test_confirm_all_needs_a_fresh_bulk_preview(tmp_path, monkeypatch):
    _ledger(tmp_path)
    assert "first" in _cmd("confirm dismiss all proposed rules", tmp_path).banner
    _cmd("dismiss rule 2", tmp_path)                                   # a single preview is not the list
    assert "first" in _cmd("confirm dismiss all proposed rules", tmp_path).banner
    _cmd("dismiss all proposed rules", tmp_path)
    later = FR.time.time() + FR.FORGET_CONFIRM_S + 5
    monkeypatch.setattr(FR.time, "time", lambda: later)
    assert "first" in _cmd("confirm dismiss all proposed rules", tmp_path).banner
    assert _states(tmp_path)[1:3] == ["proposed", "proposed"]


def test_no_proposed_rules_says_so(tmp_path):
    FR.save(tmp_path, [{"source_id": "a", "stage": "done", "rule": "A.", "adopted_at": 2.0}])
    assert "no proposed rules" in _cmd("dismiss all proposed rules", tmp_path).banner
    assert FR.is_adopted(FR.load(tmp_path)[0])


def test_the_list_shows_the_dismiss_commands_and_group(tmp_path):
    _ledger(tmp_path)
    b = _cmd("show all rules", tmp_path).banner
    assert "dismiss all proposed rules" in b and "**Dismissed" not in b
    _cmd("dismiss rule 3", tmp_path)
    _cmd("confirm dismiss rule 3", tmp_path)
    waiting, dismissed = _cmd("show all rules", tmp_path).banner.split("**Dismissed", 1)
    assert "Waiting rule C." in dismissed and "Waiting rule C." not in waiting


# ── no command: the model is told it cannot change a rule ────────────

@pytest.mark.parametrize("text", [
    "forget all rules", "remove rule 5 please", "can you delete the proposed rules?",
    "drop rules 5 and 6", "approve the candidate rule about news",
    "forget the rules you proposed", "dismiss the rule proposals",   # r1: missed wordings
    "should I learn rule 5?"])
def test_a_message_about_the_rules_gets_the_hint(text):
    assert not FR.is_rule_command(text)
    h = FR.rule_command_hint(text)
    assert "list_lessons" in h and "dismiss all proposed rules" in h and "NOTHING was changed" in h
    assert "show rule N" in h and "QUESTION, answer it" in h        # r1: a question is answered, not redirected


@pytest.mark.parametrize("text", [
    "dismiss all proposed rules", "forget rule 2",                        # commands: their own note
    "what are the rules of chess", "forget it", "learn python with me",
    "Did the EU adopt new rules on AI chips?", "explain your rules for replace operations",   # r2: no bare new/your
    "explain " + "very " * 25 + "long rule 5 of the style guide and remove it"])
def test_no_hint_for_a_command_or_an_unrelated_message(text):
    assert FR.rule_command_hint(text) == ""


@pytest.mark.parametrize("kind,want", [("owner", True), ("member", False)])
def test_the_hint_reaches_only_the_owner_note(tmp_path, monkeypatch, kind, want):
    import asyncio
    from ghost_agent.core import agent as A
    from ghost_agent.utils import logging as L
    monkeypatch.setattr(L, "request_kind", lambda: kind)
    monkeypatch.setattr(A, "reply_is_public", lambda: False)
    (tmp_path / "memory").mkdir()
    fake = SimpleNamespace(context=SimpleNamespace(memory_dir=str(tmp_path / "memory")))
    body = {"messages": [{"role": "user", "content": "forget all rules please"}]}
    monkeypatch.setattr(A.logger, "warning", lambda *a, **k: pytest.fail(f"note raised: {a}"))   # r1
    note = asyncio.run(A.GhostAgent._owner_rule_note(fake, body))
    assert bool(note) is want and (not want or "list_lessons" in note)
    assert not getattr(note, "banner", "")                             # a note, never a reply line
