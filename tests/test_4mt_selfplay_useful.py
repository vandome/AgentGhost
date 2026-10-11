"""§4MT (2026-10-09, operator: "i want self-play to be useful"): self-play
practises the owner's real failures in the evidence skills, keeps a lesson
only when it beats a no-lesson control, surfaces it near the request it came
from, and withdraws it when it keeps failing there. Each test names the
measured defect or the operator decision it holds.
§4NM: the lesson-proof, practice-calibration and synthetic self-play run
tests were removed with those subsystems; the practice templates, the brief
and the owner-seed picker went in the post-§4NN tidy-up. The leak-hash gate,
the reaction judge and the supervised slot remain."""
from __future__ import annotations

import ast
import asyncio
import datetime
import inspect
import json
import re
import textwrap
from types import SimpleNamespace

import pytest

from ghost_agent.distill.schema import ToolCall, Trajectory


# ── the leak gate: rare tokens hashed, never the owner's words ────────

def _traj(**kw):
    kw.setdefault("task_kind", "user_request")
    return Trajectory(**kw)


def test_a_generated_challenge_copying_a_rare_source_token_is_caught_by_hash():
    from ghost_agent.core.practice_brief import leaked_tokens, source_token_hashes
    t = _traj(user_request="parse sales_2026.csv from Moser and sum order 77123")
    hashes = source_token_hashes(t)
    assert all(len(h) == 16 and "moser" not in h for h in hashes)
    assert leaked_tokens("Read sales_2026.csv and total order 77123.", hashes) == ["77123", "sales_2026.csv"]
    assert leaked_tokens("Read data.csv and total order 5.", hashes) == []
    assert leaked_tokens("A file from Moser, then 77123", hashes) == ["77123", "Moser"]


# ── supply: the suspect-failure rule + the reaction judge ──────────────

class _Collector:
    def __init__(self, rows):
        self.rows = rows

    def iter_trajectories(self, since_days=None, **_):
        return iter(self.rows)


def test_the_reaction_cache_makes_a_turn_a_seed_and_nothing_else_does():
    """Only 1 of 23 graded wrong/partial turns was recorded FAILED; the
    owner re-asks instead of saying "wrong"."""
    from ghost_agent.core.owner_seeds import failure_signal
    t = _traj(outcome="passed", tool_calls=[ToolCall(name="web_search")])
    assert failure_signal(t, [], {}) == ""
    assert failure_signal(t, [], {t.id: False}) == ""
    assert failure_signal(t, [], {t.id: True}) == "reaction"
    t.final_response = "[ATTEMPT_ABORTED_TURN] x"
    assert failure_signal(t, [], {t.id: True}) == ""                 # an abort is never a seed


def _ts(sec):
    return datetime.datetime.fromtimestamp(sec, datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def test_reaction_pairs_use_the_calibrations_adjacency():
    """Same channel, next turn STARTING within 20 min of this one's end (the
    timestamp is written at turn end)."""
    from ghost_agent.core.owner_seeds import reaction_pairs
    base = 1_790_000_000
    a = _traj(timestamp=_ts(base), extra={"req_id": "web-1"})
    slack = _traj(timestamp=_ts(base + 60), extra={"req_id": "slack-9"})
    b = _traj(timestamp=_ts(base + 1300), duration_s=200, extra={"req_id": "web-2"})  # starts at +1100 s
    c = _traj(timestamp=_ts(base + 1300 + 1500), duration_s=10, extra={"req_id": "web-3"})
    pairs = [(x.id, y.id) for x, y in reaction_pairs([c, slack, b, a])]
    assert pairs == [(a.id, b.id)]


class _LLM:
    def __init__(self, answers, fail_at=None, exc=None):
        self.answers = list(answers)
        self.calls = []
        self.fail_at = fail_at
        self.exc = exc

    async def chat_completion(self, payload, **kw):
        self.calls.append((payload, kw))
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            from ghost_agent.core.llm import BackgroundDeferred
            raise self.exc or BackgroundDeferred("slot busy")
        return {"choices": [{"message": {"content": self.answers.pop(0)}}]}


def _pairs(n):
    base = 1_790_000_000
    out = []
    for i in range(n):
        out.append(_traj(timestamp=_ts(base + i * 100), user_request=f"r{i}", final_response=f"a{i}",
                         tool_calls=[ToolCall(name="web_search")], extra={"req_id": f"web-{i}"}))
    return out


async def test_the_judge_pass_caches_verdicts_is_bounded_and_runs_on_the_main_model(tmp_path, monkeypatch):
    from ghost_agent.core import owner_seeds as O
    monkeypatch.delenv("GHOST_REACTION_JUDGE", raising=False)
    rows = _pairs(5)
    llm = _LLM(["FAILED", "OK", "ok.", "OK", "OK"])
    made = await O.judge_reactions(_Collector(rows), tmp_path, llm, "m", cap=3)
    assert made == 3 and len(llm.calls) == 3
    cache = O.load_reactions(tmp_path)
    # newest pair first; the last turn has no next message
    assert cache == {rows[3].id: True, rows[2].id: False, rows[1].id: False}
    payload, kw = llm.calls[0]
    assert payload["temperature"] == 0.0 and payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "use_worker" not in kw and kw["is_background"] is True     # calibrated on the main model
    # a second pass judges only what is left, then stops at a busy slot
    llm2 = _LLM(["FAILED"], fail_at=1)
    assert await O.judge_reactions(_Collector(rows), tmp_path, llm2, "m") == 0
    assert O.load_reactions(tmp_path) == cache
    # a busy slot ends the pass at once — it does not try every turn
    fresh = tmp_path / "fresh"
    llm3 = _LLM(["OK"] * 5, fail_at=1)
    assert await O.judge_reactions(_Collector(rows), fresh, llm3, "m") == 0 and len(llm3.calls) == 1


async def test_the_judge_is_off_with_its_kill_switch(tmp_path, monkeypatch):
    from ghost_agent.core import owner_seeds as O
    monkeypatch.setenv("GHOST_REACTION_JUDGE", "0")
    llm = _LLM(["FAILED"])
    assert await O.judge_reactions(_Collector(_pairs(3)), tmp_path, llm, "m") == 0 and not llm.calls


@pytest.mark.parametrize("text,want", [("FAILED", True), ("OK", False), ("<think>x OK</think>FAILED", True),
                                       ("ok.", False), ("maybe", None), ("", None)])
def test_the_judge_reply_is_read_strictly(text, want):
    from ghost_agent.distill.reaction_judge import parse_verdict
    assert parse_verdict(text) is want


# ── playbook rows: dedup and seed-side retrieval ──────────────────────

def _sm(tmp_path, *rows):
    from ghost_agent.memory.skills import SkillMemory
    d = tmp_path / "memory"
    d.mkdir(parents=True, exist_ok=True)
    sm = SkillMemory(d)
    sm.save_playbook([dict(r) for r in rows])
    return sm


def _row(trig="when answering from fetched pages", **kw):
    r = {"trigger": trig, "correct_pattern": "cite the newest dated source", "anti_pattern": "trusting a snippet",
         "quarantined": True, "quarantine_reason": "proof_pending", "proof": "pending"}
    r.update(kw)
    return r


# ── surfaced near the seed request ─────────────────────────────────────

class _EmbedVM:
    """A vector memory whose skill query admits nothing (the trigger is about
    a drill) and whose query embedding is a fixed direction per text."""
    def __init__(self, vecs):
        self.vecs = vecs
        self.collection = SimpleNamespace(query=lambda **kw: {"documents": [[]], "distances": [[]],
                                                              "metadatas": [[]]})

    def embed_query(self, text):
        return [self.vecs[text]]


def test_a_proven_lesson_surfaces_near_its_seed_request_only(tmp_path):
    """Self-play lessons reached 1 of 311 owner turns: trigger distance 0.45
    against a 0.30 gate. Operator: surface it when a request resembles the
    original failure."""
    from ghost_agent.memory.skills import SEED_SIM_MIN
    proven = _row("practise reading release pages", quarantined=False, proof="kept", seed_embedding=[1.0, 0.0])
    proven.pop("quarantine_reason")
    sm = _sm(tmp_path, proven)
    near, far = "what's the latest postgres release", "book me a table"
    c = SEED_SIM_MIN + 0.01
    vm = _EmbedVM({near: [c, (1 - c * c) ** 0.5], far: [0.2, 0.98]})
    items, branch = sm._playbook_items_and_branch(near, vm)
    assert [i["trigger"] for i in items] == ["practise reading release pages"] and branch == "vector"
    assert sm._playbook_items_and_branch(far, vm)[0] == []


@pytest.mark.parametrize("change", ["quarantined", "pending", "non_latin"])
def test_seed_retrieval_abstains(tmp_path, change):
    proven = _row("practise reading release pages", quarantined=False, proof="kept", seed_embedding=[1.0, 0.0])
    if change == "quarantined":
        proven["quarantined"] = True
    if change == "pending":
        proven["proof"] = "pending"
    sm = _sm(tmp_path, proven)
    q = "ποια είναι η τελευταία έκδοση" if change == "non_latin" else "latest release"
    vm = _EmbedVM({q: [1.0, 0.0]})
    assert sm._playbook_items_and_branch(q, vm)[0] == []


# ── the self-play run ─────────────────────────────────────────────────

# ── fresh reader (r1) findings ─────────────────────────────────────────


def test_an_owner_practice_relearn_never_rewrites_a_live_rows_text(tmp_path):
    """MAJOR: a "reinforced" dedup hit replaced a live (or proven) row's fix
    with unproven text — no proof, no quarantine."""
    # §4NM: the writer gate refuses source="self_play" outright, so the dedup
    # rule is driven through an admitted source (it is the same code path).
    sm = _sm(tmp_path)
    trig = "when reporting a tool's output to the user"
    assert sm.learn_lesson(trig, "dropped rows", "copy every row", trigger=trig, correct_pattern="copy every row",
                           anti_pattern="dropped rows", source="learn_skill", confidence=0.6) == "written"
    longer = "copy every row the tool returned, in its order, and say unknown for a missing value"
    sm.learn_lesson(trig, "dropped rows", longer, trigger=trig, correct_pattern=longer, anti_pattern="dropped rows",
                    source="learn_skill", confidence=0.6, source_trajectory_id="tX", replace_text=False)
    assert sm._load_playbook()[0]["solution"] == "copy every row"
    sm.learn_lesson(trig, "dropped rows", longer, trigger=trig, correct_pattern=longer, anti_pattern="dropped rows",
                    source="learn_skill", confidence=0.6, source_trajectory_id="tY")
    assert sm._load_playbook()[0]["solution"] == longer        # the default path still replaces


async def test_an_unreadable_verdict_is_cached_and_each_verdict_is_saved_at_once(tmp_path, monkeypatch):
    from ghost_agent.core import owner_seeds as O
    monkeypatch.delenv("GHOST_REACTION_JUDGE", raising=False)
    rows = _pairs(4)
    llm = _LLM(["hmm", "FAILED"], fail_at=3)
    assert await O.judge_reactions(_Collector(rows), tmp_path, llm, "m") == 2
    assert O.load_reactions(tmp_path) == {rows[2].id: None, rows[1].id: True}
    llm2 = _LLM(["OK"])
    await O.judge_reactions(_Collector(rows), tmp_path, llm2, "m")
    assert len(llm2.calls) == 1                      # only rows[0] was left


def test_neither_dedup_branch_lets_an_owner_practice_relearn_rewrite_text(tmp_path):
    """Both learn_lesson dedup sites (JSON match, vector twin) honour
    replace_text=False (battery r2: the vector-twin site was unpinned)."""
    from unittest.mock import MagicMock
    trig = "when reporting a tool's output to the user"
    longer = "copy every row the tool returned, in its order, and say unknown for a missing value"
    for replace, want in ((False, "copy every row"), (True, longer)):
        # §4NM: source="self_play" is refused at the writer gate; the dedup
        # branches are the same for an admitted source (the row and the
        # relearn share one source — a foreign source never replaces text)
        sm = _sm(tmp_path / str(replace), {"trigger": trig, "task": trig, "mistake": "dropped rows",
                                           "solution": "copy every row", "correct_pattern": "copy every row",
                                           "source": "learn_skill", "frequency": 1})
        mem = MagicMock()
        mem.collection.query.return_value = {
            "ids": [["id1"]], "documents": [[f"SITUATION: {trig}\nMISTAKE: dropped rows\nSOLUTION: copy every row"]],
            "distances": [[0.02]], "metadatas": [[{"trigger": trig, "type": "skill", "source": "learn_skill"}]]}
        r = sm.learn_lesson(trig, "dropped rows", longer, memory_system=mem, trigger=trig, correct_pattern=longer,
                            anti_pattern="dropped rows", source="learn_skill", confidence=0.6,
                            source_trajectory_id="tZ", replace_text=replace)
        assert r == "reinforced"
        assert sm._load_playbook()[0]["solution"] == want, (replace, sm._load_playbook()[0]["solution"])


async def test_a_turn_the_judge_call_fails_on_is_skipped_not_retried_first_forever(tmp_path, monkeypatch):
    """r2: any error ended the pass, so a turn that always errors (too long)
    blocked every older turn on every pass."""
    from ghost_agent.core import owner_seeds as O
    monkeypatch.delenv("GHOST_REACTION_JUDGE", raising=False)
    rows = _pairs(4)
    llm = _LLM(["FAILED", "OK"], fail_at=1, exc=ValueError("context too long"))
    assert await O.judge_reactions(_Collector(rows), tmp_path, llm, "m") == 3
    assert O.load_reactions(tmp_path) == {rows[2].id: None, rows[1].id: True, rows[0].id: False}
    llm2 = _LLM(["OK"], fail_at=1, exc=ConnectionError("down"))
    fresh = tmp_path / "f"
    assert await O.judge_reactions(_Collector(rows), fresh, llm2, "m") == 0 and len(llm2.calls) == 1


# ── fresh reader (r2) findings ─────────────────────────────────────────

async def test_the_idle_runner_says_why_it_stopped():
    from tests.test_4ms_idle_cycle import _agent_with_llm
    a = _agent_with_llm()

    async def job():
        await asyncio.sleep(30)
    assert await a._run_idle_job(job(), "x", cap_s=0.3) is None and a._last_idle_stop == "cap"
    runner = asyncio.ensure_future(a._run_idle_job(job(), "x"))
    await asyncio.sleep(0.2)
    a.context.llm_client.foreground_requests = 1
    assert await runner is None and a._last_idle_stop == "owner"
    a.context.llm_client.foreground_requests = 0

    async def quick():
        return 1
    assert await a._run_idle_job(quick(), "x") == 1 and a._last_idle_stop == ""


@pytest.mark.parametrize("fix", ["End the reply with an ANSWER: line giving the version.",
                                 "Always finish with the STATUS line listing each step.",
                                 "Read web/results.json before answering.",
                                 "Match the line format the task gives."])
def test_a_lesson_about_the_practice_format_is_refused(tmp_path, fix):
    """MAJOR (r2): a failed-on-format practice yields a format lesson, whose
    arm passes first try against a control that needed the retry — it would
    win every pair and reach owner prompts."""
    sm = _sm(tmp_path)
    assert sm.learn_lesson("when answering a research question", "no final line", fix,
                           trigger="when answering a research question", correct_pattern=fix,
                           anti_pattern="no final line", source="self_play") is None


def test_an_unproven_relearn_never_marks_a_live_row_verified(tmp_path):
    trig = "when reporting a tool's output to the user"
    sm = _sm(tmp_path, {"trigger": trig, "task": trig, "mistake": "dropped rows", "solution": "copy every row",
                        "correct_pattern": "copy every row", "source": "self_play", "verified": False})
    sm.learn_lesson(trig, "dropped rows", "copy every row", trigger=trig, correct_pattern="copy every row",
                    anti_pattern="dropped rows", source="self_play", verified=True, source_trajectory_id="t1",
                    replace_text=False)
    assert not sm._load_playbook()[0].get("verified")


# ── the operator's supervised run ──────────────────────────────────────

async def test_the_supervised_run_executes_the_same_slot_and_reports_its_outcome(monkeypatch):
    from ghost_agent.core.agent import GhostAgent, _NoOwnerSeed
    import ghost_agent.core.agent as A
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = SimpleNamespace(memory_dir=None)
    calls = []

    async def body(ctx):
        calls.append(ctx)
        raise _NoOwnerSeed()
    agent._self_play_slot_body = body
    monkeypatch.setattr(A, "_sync_idle_anchors", lambda *a, **k: None)
    assert agent.start_supervised_self_play_slot() is True
    assert agent.start_supervised_self_play_slot() is False          # one at a time
    for _ in range(5):
        await asyncio.sleep(0)
    assert calls == [agent.context]
    assert agent._supervised_slot["state"] == "done"
    assert "no unpractised owner failure" in agent._supervised_slot["outcome"]
    assert agent.start_supervised_self_play_slot() is True           # done → may run again


def test_the_tick_and_the_operator_run_share_one_body():
    src = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(__import__(
        "ghost_agent.core.agent", fromlist=["x"]).GhostAgent))))
    assert src.count("await self._self_play_slot_body(ctx)") == 2


def test_the_operator_route_refuses_while_a_user_request_is_live():
    import ghost_agent.api.routes as R
    src = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(R.operator_self_play_slot))))
    assert "if int(getattr(llm, 'foreground_requests', 0) or 0) > 0:" in src
    assert "status_code=409" in src
