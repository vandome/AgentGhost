"""§4OR — owner rules change only by the owner's own commands (operator:
"i'm afraid that rules will override themselves over time")."""
import json

from ghost_agent.core import failure_replay as FR
from ghost_agent.memory import skills as sk
from tests.pin_ast import tree, calls, U

H_T = "when asked how you are, how things are going, or about the system's status, health or performance"
H_R = ("Do not assert system health status (e.g., 'healthy', 'green', 'fine') or specific performance metrics "
       "unless the tool output explicitly confirms them via a dedicated health check action.")
S_T = "when providing specific numerical data or statistics"
S_R = ("When citing specific statistics or data points, ensure they are explicitly supported by the content of the "
       "retrieved tool outputs, rather than relying on general knowledge or unverified claims.")


def _case(sid, rule, when, **kw):
    c = {"source_id": sid, "stage": "done", "request": "r", "rule": rule,
         "diagnosis": {"cause": "unsupported_claim", "explanation": "x", "when": when},
         "base": [{"verdict": "CONFIRMED"}], "test": [{"verdict": "CONFIRMED"}]}
    c.update(kw)
    return c


def _rows(tmp):
    return json.loads((tmp / "memory" / "skills_playbook.json").read_text())


def _setup(tmp_path, cases):
    (tmp_path / "memory").mkdir()
    store = sk.SkillMemory(tmp_path / "memory")
    FR.save(tmp_path, cases)
    return store


def test_only_the_owners_commands_change_an_owner_rule(tmp_path):
    store = _setup(tmp_path, [_case("c1", H_R, H_T)])
    assert "adopted" in FR.owner_rule_command("learn rule 1", tmp_path, store).banner
    pb = store._load_playbook()
    rule = next(r for r in pb if r.get("origin") == "owner_rule")
    # another writer rewrites the text, quarantines it, and adds a fake owner rule
    rule["solution"] = "something else"
    rule["quarantined"] = True
    rule["retrievals"] = 7                                       # usage counters may change
    pb.append({"trigger": "x", "task": "x", "solution": "y", "origin": "owner_rule", "verified": True})
    store.save_playbook(pb)
    after = _rows(tmp_path)
    kept = next(r for r in after if sk._rule_key(r) == H_T.lower())
    assert kept["solution"] == H_R and not kept.get("quarantined") and kept["retrievals"] == 7
    assert next(r for r in after if r.get("trigger") == "x")["origin"] == ""
    # a removal outside the owner's command is put back
    store.save_playbook([r for r in after if r.get("origin") != "owner_rule"])
    assert any(r.get("origin") == "owner_rule" and r["solution"] == H_R for r in _rows(tmp_path))
    # inside the owner's command the change stands
    with sk.owner_rule_writes():
        store.save_playbook([r for r in _rows(tmp_path) if r.get("origin") != "owner_rule"])
    assert not any(r.get("origin") == "owner_rule" for r in _rows(tmp_path))


def test_the_10_12_incident_keeps_both_rules(tmp_path, monkeypatch):
    store = _setup(tmp_path, [_case("c1", H_R, H_T), _case("c4", S_R, S_T)])
    FR.owner_rule_command("learn rule 1", tmp_path, store)
    monkeypatch.setattr(store, "_find_duplicate_lesson", lambda *a, **k: {
        "source": "vector", "distance": 0.15, "trigger": H_T, "text": f"SITUATION: {H_T}\nSOLUTION: {H_R}"})
    note = FR.owner_rule_command("learn rule 2", tmp_path, store)
    assert "adopted" in note.banner, note.banner
    sol = {sk._rule_key(r): r["solution"] for r in _rows(tmp_path) if r.get("origin") == "owner_rule"}
    assert sol == {H_T.lower(): H_R, S_T.lower(): S_R}


def test_a_rule_for_the_same_situation_asks_and_replace_swaps(tmp_path):
    other = "Answer how you are in one short sentence without any numbers."
    store = _setup(tmp_path, [_case("c1", H_R, H_T), _case("c2", other, H_T)])
    FR.owner_rule_command("learn rule 1", tmp_path, store)
    note = FR.owner_rule_command("learn rule 2", tmp_path, store)
    assert "same situation" in note.banner and "replace rule 1 with rule 2" in note.banner
    assert [r["solution"] for r in _rows(tmp_path) if r.get("origin") == "owner_rule"] == [H_R]
    assert FR.is_rule_command("replace rule 1 with rule 2")
    note = FR.owner_rule_command("replace rule 1 with rule 2", tmp_path, store)
    assert "replaced by rule 2" in note.banner, note.banner
    assert [r["solution"] for r in _rows(tmp_path) if r.get("origin") == "owner_rule"] == [other]
    led = FR.load(tmp_path)
    assert led[0].get("forgotten_at") and FR.is_adopted(led[1])


def test_a_full_block_says_which_rule_is_no_longer_shown(tmp_path, monkeypatch):
    monkeypatch.setattr(sk, "OWNER_RULES_MAX", 1)
    store = _setup(tmp_path, [_case("c1", H_R, H_T), _case("c2", S_R, S_T)])
    FR.owner_rule_command("learn rule 1", tmp_path, store)
    note = FR.owner_rule_command("learn rule 2", tmp_path, store)
    assert "rule 1 is no longer shown" in note.banner
    assert sk.OWNER_RULES_MAX == 1 and len(store.owner_rules()) == 1


def test_the_integrity_check_restores_the_adopted_text(tmp_path):
    store = _setup(tmp_path, [_case("c1", H_R, H_T), _case("c2", S_R, S_T)])
    FR.owner_rule_command("learn rule 1", tmp_path, store)
    FR.owner_rule_command("learn rule 2", tmp_path, store)
    assert FR.check_rule_integrity(tmp_path, store) == []                     # intact: nothing to do
    with sk.owner_rule_writes():                                               # simulate the 10-12 corruption
        pb = store._load_playbook()
        next(r for r in pb if sk._rule_key(r) == H_T.lower())["solution"] = S_R
        store._save_playbook_unlocked(pb)
    fixes = FR.check_rule_integrity(tmp_path, store)
    assert fixes == ["rule 1 (text restored)"]
    with sk.owner_rule_writes():                                               # a vanished rule is re-added
        store._save_playbook_unlocked([r for r in store._load_playbook() if sk._rule_key(r) != S_T.lower()])
    assert FR.check_rule_integrity(tmp_path, store) == ["rule 2 (re-added)"]
    sol = sorted(r["solution"] for r in _rows(tmp_path) if r.get("origin") == "owner_rule")
    assert sol == sorted([H_R, S_R])


def test_the_guard_sits_at_the_save_point_and_maintenance_runs_the_check():
    from ghost_agent.core import dream
    t = tree(sk)
    assert [c for c in calls(t, "_protect_owner_rules")]
    assert [c for c in calls(tree(dream), "to_thread") if c.args and U(c.args[0]) == "check_rule_integrity"]
    assert sk.OWNER_RULES_MAX == 10


# ── reader round 1 ───────────────────────────────────────────────────────── #

def test_r1_a_reordered_trigger_for_the_same_situation_asks_and_overwrites_nothing(tmp_path):
    reordered = "When asked about the system's status, health or performance, or how things are going or how you are."
    store = _setup(tmp_path, [_case("c1", H_R, H_T), _case("c4", S_R, reordered)])
    FR.owner_rule_command("learn rule 1", tmp_path, store)
    note = FR.owner_rule_command("learn rule 2", tmp_path, store)
    assert "same situation" in note.banner
    assert [r["solution"] for r in _rows(tmp_path) if r.get("origin") == "owner_rule"] == [H_R]


def test_r1_the_memory_forget_never_offers_or_removes_an_adopted_rule(tmp_path):
    store = _setup(tmp_path, [_case("c1", H_R, H_T)])
    FR.owner_rule_command("learn rule 1", tmp_path, store)
    assert store.lessons_mentioning("health") == []
    assert store.remove_by_trigger(H_T) is False
    assert store.remove_rows(lambda r: True) == 0 or all(r.get("origin") == "owner_rule" for r in _rows(tmp_path))
    assert [r["solution"] for r in _rows(tmp_path) if r.get("origin") == "owner_rule"] == [H_R]


def test_r1_a_chat_instruction_for_the_same_situation_is_its_own_lesson(tmp_path):
    store = _setup(tmp_path, [_case("c1", H_R, H_T)])
    FR.owner_rule_command("learn rule 1", tmp_path, store)
    before = next(r for r in _rows(tmp_path) if r.get("origin") == "owner_rule")
    store.learn_lesson(H_T, "gave no date", "From now on also say when the check ran.", trigger=H_T,
                       source="learn_skill", source_trajectory_id="t-owner-1",
                       generality_context="from now on also say when the check ran")
    after = next(r for r in _rows(tmp_path) if r.get("origin") == "owner_rule")
    assert {k: v for k, v in after.items() if k not in sk._OWNER_RULE_COUNTERS} == \
           {k: v for k, v in before.items() if k not in sk._OWNER_RULE_COUNTERS}
    assert any(r.get("solution") == "From now on also say when the check ran." and r.get("origin") != "owner_rule"
               for r in _rows(tmp_path))


def test_r1_adoption_beside_an_ordinary_lesson_with_the_same_trigger_is_a_real_owner_rule(tmp_path):
    store = _setup(tmp_path, [_case("c1", H_R, H_T)])
    store.learn_lesson(H_T, "x", "Some older automatic advice.", trigger=H_T, source="operator_repair")
    assert any(r.get("solution") == "Some older automatic advice." for r in _rows(tmp_path))     # the setup really wrote it
    note = FR.owner_rule_command("learn rule 1", tmp_path, store)
    assert "adopted" in note.banner
    assert [r["solution"] for r in store.owner_rules()] == [H_R]
    assert any(r.get("solution") == "Some older automatic advice." for r in _rows(tmp_path))


def test_r1_replace_checks_the_new_rule_before_touching_the_old(tmp_path):
    v_t = "when asked for the latest version of a piece of software"
    store = _setup(tmp_path, [_case("c1", H_R, H_T), _case("c2", "Use the vendor table.", v_t),
                              _case("c4", S_R, "When asked for the latest version of a piece of software!")])
    FR.owner_rule_command("learn rule 1", tmp_path, store)
    FR.owner_rule_command("learn rule 2", tmp_path, store)
    note = FR.owner_rule_command("replace rule 1 with rule 3", tmp_path, store)
    assert "Nothing changed" in note.banner
    assert sorted(r["solution"] for r in _rows(tmp_path) if r.get("origin") == "owner_rule") == \
           sorted([H_R, "Use the vendor table."])
    assert FR.is_adopted(FR.load(tmp_path)[0])


def test_r1_integrity_demotes_a_rule_nobody_adopted_and_keeps_the_adopted_trigger(tmp_path):
    store = _setup(tmp_path, [_case("c1", H_R, H_T)])
    FR.owner_rule_command("learn rule 1", tmp_path, store)
    assert FR.load(tmp_path)[0]["trigger"] == H_T                          # stored at adoption
    with sk.owner_rule_writes():
        pb = store._load_playbook()
        pb.append({"trigger": "z", "task": "z", "solution": "stray", "origin": "owner_rule", "verified": True})
        store._save_playbook_unlocked(pb)
    fixes = FR.check_rule_integrity(tmp_path, store)
    assert any("never adopted" in f for f in fixes)
    assert [r["solution"] for r in store.owner_rules()] == [H_R]


def test_r1_a_command_may_change_only_the_rule_it_names_and_every_field_is_guarded(tmp_path):
    store = _setup(tmp_path, [_case("c1", H_R, H_T), _case("c2", S_R, S_T)])
    FR.owner_rule_command("learn rule 1", tmp_path, store)
    FR.owner_rule_command("learn rule 2", tmp_path, store)
    with sk.owner_rule_writes({"c1"}):
        pb = store._load_playbook()
        for r in pb:
            if r.get("rule_id") == "c2":
                r["solution"] = "changed under rule 1's permission"
        store.save_playbook(pb)
    assert next(r for r in _rows(tmp_path) if r.get("rule_id") == "c2")["solution"] == S_R
    pb = store._load_playbook()                                        # any field, not only the text
    for r in pb:
        if r.get("rule_id") == "c1":
            r.update(anti_pattern="gave no date", source_trajectory_id="t-x", timestamp="2099-01-01")
    store.save_playbook(pb)
    c1 = next(r for r in _rows(tmp_path) if r.get("rule_id") == "c1")
    assert c1["anti_pattern"] != "gave no date" and c1["source_trajectory_id"] == "c1" and c1["timestamp"] != "2099-01-01"


def test_r1_remove_rows_reports_no_owner_rule_removed(tmp_path):
    store = _setup(tmp_path, [_case("c1", H_R, H_T)])
    FR.owner_rule_command("learn rule 1", tmp_path, store)
    assert store.remove_rows(lambda r: r.get("origin") == "owner_rule") == 0


def test_r1_a_refused_replacement_never_forgets_the_old_rule_even_for_a_moment(tmp_path):
    v_t = "when asked for the latest version of a piece of software"
    store = _setup(tmp_path, [_case("c1", H_R, H_T), _case("c2", "Use the vendor table.", v_t),
                              _case("c4", S_R, "When asked for the latest version of a piece of software!")])
    FR.owner_rule_command("learn rule 1", tmp_path, store)
    FR.owner_rule_command("learn rule 2", tmp_path, store)
    FR.owner_rule_command("replace rule 1 with rule 3", tmp_path, store)
    assert not FR.load(tmp_path)[0].get("forgotten_at")


# ── reader round 2 ───────────────────────────────────────────────────────── #

def test_r2_the_ledger_trim_never_drops_an_adopted_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(FR, "_MAX_ENTRIES", 5)
    store = _setup(tmp_path, [_case("c1", H_R, H_T, n=1)])
    FR.owner_rule_command("learn rule 1", tmp_path, store)
    entries = FR.load(tmp_path) + [{"source_id": f"x{i}", "stage": "abandoned", "n": i + 2} for i in range(20)]
    FR.save(tmp_path, entries)
    led = FR.load(tmp_path)
    assert any(e["source_id"] == "c1" for e in led) and len(led) <= 6
    assert FR.check_rule_integrity(tmp_path, store) == [] and [r["solution"] for r in store.owner_rules()] == [H_R]


def test_r2_forget_with_two_adopted_cases_of_one_text_is_checked_by_id(tmp_path):
    store = _setup(tmp_path, [_case("c1", H_R, H_T), _case("c2", H_R, S_T)])
    FR.owner_rule_command("learn rule 1", tmp_path, store)
    FR.owner_rule_command("learn rule 2", tmp_path, store)
    FR.owner_rule_command("forget rule 1", tmp_path, store)
    note = FR.owner_rule_command("confirm forget rule 1", tmp_path, store)
    assert "forgotten" in note.banner and "NOT" not in note.banner, note.banner
    assert [r["rule_id"] for r in _rows(tmp_path) if r.get("origin") == "owner_rule"] == ["c2"]
    assert FR.check_rule_integrity(tmp_path, store) == []


def test_r2_the_privacy_forget_names_an_adopted_rule_that_mentions_the_target(tmp_path):
    rule = "When Spiros Denaxas asks for a paper, send the PDF link."
    store = _setup(tmp_path, [_case("c1", rule, "when someone asks for a paper")])
    FR.owner_rule_command("learn rule 1", tmp_path, store)
    hits = store.owner_rules_mentioning("Spiros Denaxas")
    assert [h["solution"] for h in hits] == [rule]
    from ghost_agent.tools import memory as mem
    src = open(mem.__file__).read()
    import ast
    t = ast.parse(src)
    assert any(isinstance(n, ast.Attribute) and n.attr == "owner_rules_mentioning" for n in ast.walk(t))


def test_r2_quarantine_and_retraction_leave_an_adopted_rule_alone_and_say_so(tmp_path):
    store = _setup(tmp_path, [_case("c1", H_R, H_T)])
    FR.owner_rule_command("learn rule 1", tmp_path, store)
    assert store.quarantine_lesson(H_T, "test") == 0
    assert store.retract_lessons_from_trajectory("c1") == 0
    assert [r["solution"] for r in store.owner_rules()] == [H_R]


def test_r2_the_slack_backstop_blocks_the_owner_rule_writes(tmp_path, monkeypatch):
    store = _setup(tmp_path, [_case("c1", H_R, H_T)])
    FR.owner_rule_command("learn rule 1", tmp_path, store)
    monkeypatch.setattr(sk, "playbook_writes_blocked", lambda: True)
    assert store.write_owner_rule("c2", S_T, S_R) == ""
    assert store.remove_owner_rule("c1") == 0
    monkeypatch.setattr(sk, "playbook_writes_blocked", lambda: False)
    assert [r["solution"] for r in store.owner_rules()] == [H_R]


async def test_r2_the_forget_preview_tells_the_owner_about_a_rule_that_mentions_the_target(tmp_path):
    from ghost_agent.tools import memory as mem
    rule = "When Spiros Denaxas asks for a paper, send the PDF link."
    store = _setup(tmp_path, [_case("c1", rule, "when someone asks for a paper")])
    FR.owner_rule_command("learn rule 1", tmp_path, store)
    out = await mem.forget_preview("Spiros Denaxas", skill_memory=store)
    assert "adopted rules mentions" in out and "forget rule N" in out, out
