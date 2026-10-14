"""§4ON — an owner rule is its own row: adopting rule 4 (statistics) must not
overwrite the adopted rule 1 (health) because their texts embed close."""
import json

from ghost_agent.memory import skills as sk

HEALTH_T = "when asked how you are, how things are going, or about the system's status, health or performance"
HEALTH_R = ("Do not assert system health status (e.g., 'healthy', 'green', 'fine') or specific performance metrics "
            "unless the tool output explicitly confirms them via a dedicated health check action.")
STATS_T = "when providing specific numerical data or statistics"
STATS_R = ("When citing specific statistics or data points, ensure they are explicitly supported by the content of "
           "the retrieved tool outputs, rather than relying on general knowledge or unverified claims.")


def _adopt(store, trigger, rule, sid):
    with sk.owner_rule_writes():                 # §4OR: the adoption runs as the owner's own write
        return _learn(store, trigger, rule, sid)


def _learn(store, trigger, rule, sid):
    return store.learn_lesson(trigger, "unsupported_claim (found by replaying a real failed request)", rule,
                              trigger=trigger, verified=True, source="learn_skill", origin="owner_rule",
                              source_trajectory_id=sid, generality_context=f"learn this rule: {rule}")


def test_a_second_owner_rule_twin_is_written_separately_and_rule_one_keeps_its_text(tmp_path, monkeypatch):
    store = sk.SkillMemory(tmp_path)
    assert _adopt(store, HEALTH_T, HEALTH_R, "case-1") is not None
    # the 2026-10-12 shape: the vector store calls rule 4 a twin of rule 1
    monkeypatch.setattr(store, "_find_duplicate_lesson",
                        lambda *a, **k: {"source": "vector", "distance": 0.15, "trigger": HEALTH_T,
                                         "text": f"SITUATION: {HEALTH_T}\nSOLUTION: {HEALTH_R}"})
    _adopt(store, STATS_T, STATS_R, "case-4")
    rows = json.loads((tmp_path / "skills_playbook.json").read_text())
    by_trigger = {str(r.get("trigger") or r.get("task")).lower(): r for r in rows}
    assert by_trigger[HEALTH_T.lower()]["solution"] == HEALTH_R                 # rule 1 untouched
    assert by_trigger[STATS_T.lower()]["solution"] == STATS_R                   # rule 4 its own row
    assert by_trigger[STATS_T.lower()]["origin"] == "owner_rule"



def test_auto_lessons_are_retired_from_prompts_and_owner_rules_still_surface(tmp_path, monkeypatch):
    monkeypatch.delenv("GHOST_AUTO_LESSONS", raising=False)
    assert sk.auto_lessons_enabled() is False
    store = sk.SkillMemory(tmp_path)
    store.learn_lesson("restart the nginx service", "none", "Run nginx -t before reloading nginx.",
                       trigger="restart the nginx service", source="operator_repair")
    with sk.owner_rule_writes({"c1"}):
        store.write_owner_rule("c1", HEALTH_T, HEALTH_R)
    assert store.get_playbook_context("restart the nginx service now") == ""
    items = store.get_playbook_items("how are you, how are things going, system status health performance")
    assert all(sk._rule_key({"trigger": i.get("trigger")}) == HEALTH_T.lower() for i in items)
    assert not any("nginx" in str(i.get("text")) for i in store.get_playbook_items("restart the nginx service now"))
    monkeypatch.setenv("GHOST_AUTO_LESSONS", "1")
    assert any("nginx" in str(i.get("text")) for i in store.get_playbook_items("restart the nginx service now"))
