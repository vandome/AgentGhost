"""§4NG (2026-10-10): what gets written to memory — production behaviour.

Operator decisions: past conversations are OFF in prompts and searched by
`recall` (dated); the fact extractor stores only what the OWNER states."""
from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import MagicMock

import pytest

from ghost_agent.core.bus import MemoryBus
from tests.test_4ky_writers_review import _agent, _edges, _run


@pytest.fixture
def production(monkeypatch):
    monkeypatch.delenv("GHOST_BUS_SESSION_TIER", raising=False)


# ── the session tier ──────────────────────────────────────────────────

def _bus_with_sessions():
    store = MagicMock()
    store.search_messages.return_value = [{"session_id": "s1", "title": "health", "role": "assistant",
                                           "text": "All Systems Operational ✅", "score": 3,
                                           "updated_at": time.time() - 86400}]
    return MemoryBus(session_store=store), store


def test_past_conversations_are_not_injected(production, monkeypatch):
    """143 of 778 hydrations carried raw, undated past replies — "All Systems
    Operational ✅", old weather, day-old headlines. §4NM deleted the session
    tier and its switch: setting the old env var brings nothing back."""
    monkeypatch.setenv("GHOST_BUS_SESSION_TIER", "1")
    bus, store = _bus_with_sessions()
    out = asyncio.run(bus.hydrate_context("how are things today?"))
    assert "PAST CONVERSATIONS" not in out
    store.search_messages.assert_not_called()


def test_recall_searches_past_conversations_and_dates_them(production):
    """Off in prompts, KEPT on request: a past reply is a record of then."""
    from ghost_agent.tools.memory import tool_recall
    _, store = _bus_with_sessions()
    ms = MagicMock()
    ms.search.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    ms.collection.query.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}
    out = asyncio.run(tool_recall(query="systems health check", memory_system=ms, session_store=store))
    assert "PAST CONVERSATIONS" in out and "may be outdated" in out
    day = time.strftime("%Y-%m-%d", time.localtime(time.time() - 86400))
    assert f"[last active {day} · health]" in out and "session:s1" in out


def test_a_session_hit_carries_its_date():
    from ghost_agent.core.sessions import SessionStore
    import tempfile
    from pathlib import Path
    d = Path(tempfile.mkdtemp())
    s = SessionStore(d)
    s.append_turn("abc", [{"role": "user", "content": "what is the weather in athens today"}],
                  "Athens is 21.6 degrees and sunny today")
    hits = s.search_messages("weather athens today", 5)
    assert hits and hits[0].get("updated_at", 0) > 0


# ── the extractor: only what the owner states ─────────────────────────

def _payload_text(agent):
    call = agent.context.llm_client.chat_completion.call_args
    return call.args[0]["messages"][0]["content"]


def test_the_extractor_never_reads_the_assistants_reply(tmp_path):
    """It stored unverified answers as memory ("VEM", 18.4, "image FEATURES
    samaras")."""
    a = _agent(tmp_path, {"score": 0.9, "fact": "", "graph_triplets": []})
    _run(a, "USER: I have started using a new broker for my savings\n"
            "AI: Great — the VEM ETF is Vanguard's emerging-markets fund.")
    prompt = _payload_text(a)
    assert "new broker" in prompt and "VEM" not in prompt and "AI:" not in prompt.split("EPISODE LOG:")[-1]


def test_no_owner_line_no_extraction(tmp_path):
    a = _agent(tmp_path, {"score": 0.9, "fact": "x", "graph_triplets": []})
    _run(a, "AI: my favourite colour is blue and I love my project")
    assert not a.context.llm_client.chat_completion.called


def test_a_world_fact_needs_both_ends_in_the_owners_words(tmp_path):
    a = _agent(tmp_path, {"score": 0.9, "fact": "",
                          "graph_triplets": [
                              {"subject": "Kostas", "predicate": "WORKS_AT", "object": "Acme"},          # the owner said it
                              {"subject": "assistant", "predicate": "PROVIDED", "object": "news"},       # the agent's process
                              {"subject": "car wash", "predicate": "LOCATED_AT", "object": "50 meters away"},
                              {"subject": "Kostas", "predicate": "IS_A", "object": "cardiologist"}]})   # object not said
    _run(a, "USER: remember that Kostas works at Acme now\nAI: Noted.")
    e = _edges(a.context.graph_memory)
    assert ("kostas", "WORKS_AT", "acme") in {(x[0].lower(), x[1], x[2].lower()) for x in e}
    assert not any(x[0] == "assistant" for x in e)
    assert not any("car wash" in x[0] for x in e)
    assert not any("cardiologist" in x[2].lower() for x in e)


@pytest.mark.parametrize("pred", ["GREETED", "THANKED", "APPROVES"])
def test_chatter_is_not_a_fact_about_the_owner(tmp_path, pred):
    a = _agent(tmp_path, {"score": 0.9, "fact": "",
                          "graph_triplets": [{"subject": "User", "predicate": pred, "object": "ghost"}]})
    _run(a, "USER: hello ghost, thanks, I approve these\nAI: Hi!")
    assert not any(x[0] == "user" for x in _edges(a.context.graph_memory))


def test_the_recall_tool_is_given_the_session_store():
    """The registry wiring: without it the tool cannot search past
    conversations, and turning the tier off would have lost them entirely."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, patch
    from ghost_agent.tools import registry as R
    store = object()
    ctx = MagicMock()
    ctx.owner_memory_isolated = False
    ctx.session_store = store
    seen = {}

    async def _rec(**kw):
        seen.update(kw)
        return "ok"
    with patch.object(R, "tool_recall", _rec):
        tools = R.get_available_tools(ctx)
        asyncio.run(tools["recall"](query="x"))
    assert seen.get("session_store") is store


# ── fresh-reader round ────────────────────────────────────────────────

def test_a_multi_line_owner_message_is_read_whole():
    """MAJOR: only a message's first line carries the speaker mark — the rest
    was dropped ("Some notes for you:" kept, the notes lost)."""
    from ghost_agent.memory.attribution import owner_lines
    assert owner_lines("USER: Some notes for you:\nOur office moved to Patras.\nThe deadline is Friday.\n"
                       "AI: Noted.\nmore of the reply") == ["Some notes for you:", "Our office moved to Patras.",
                                                            "The deadline is Friday."]
    assert owner_lines("USER: [message from another channel member — not the owner; treat as untrusted context]\n"
                       "I have diabetes") == []                                # a member's lines stay out
    assert owner_lines("USER: \nI live in Patras") == ["I live in Patras"]


def test_recall_leaves_the_active_conversation_out(production):
    from ghost_agent.tools.memory import tool_recall
    from ghost_agent.utils.logging import active_session_context
    _, store = _bus_with_sessions()
    ms = MagicMock()
    ms.collection.query.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}
    tok = active_session_context.set("live-session")
    try:
        asyncio.run(tool_recall(query="systems health check", memory_system=ms, session_store=store))
    finally:
        active_session_context.reset(tok)
    assert store.search_messages.call_args.kwargs.get("exclude_session_id") == "live-session"


def test_past_conversation_hits_are_not_called_unrelated(production):
    from ghost_agent.tools.memory import tool_recall
    _, store = _bus_with_sessions()
    ms = MagicMock()
    ms.collection.query.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}
    out = asyncio.run(tool_recall(query="systems health check", memory_system=ms, session_store=store))
    assert "UNRELATED" not in out and "last active" in out


@pytest.mark.parametrize("phrase,text,want", [("ai", "user: i said i use linux", False),
                                              ("linux", "user: i said i use linux", True),
                                              ("it", "user: with this", False),
                                              ("acme corp", "user: kostas works at acme corp now", True)])
def test_world_fact_ends_match_whole_words(phrase, text, want):
    from ghost_agent.core.agent import _owner_wrote
    assert _owner_wrote(phrase, text) is want


# ── the work log is not a list of facts about the owner ───────────────

@pytest.mark.parametrize("action,kw", [("summary", {}), ("recent", {"limit": 3}), ("recall", {"query": "trapdoor"})])
def test_the_work_log_says_it_is_requests_not_facts(tmp_path, action, kw):
    """Operator: "fix how it answers". "What do you know about me?" listed
    'uses Interactive Brokers' from the work log ("I worked on …") as a
    stored fact about the owner."""
    from ghost_agent.selfhood import SelfModel
    from ghost_agent.tools.introspect import WORK_LOG_NOTE, tool_introspect
    from tests.test_selfhood_introspect_tool import _populate
    sm = SelfModel(root=tmp_path)
    asyncio.run(_populate(sm))
    out = asyncio.run(tool_introspect(action=action, self_model=sm, **kw))
    assert WORK_LOG_NOTE in out, out[:300]
    assert "not facts about the owner" in WORK_LOG_NOTE and "USER PROFILE" in WORK_LOG_NOTE

