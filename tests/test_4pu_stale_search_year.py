"""§4PU: live 3c527b55 (2026-10-17) — the prompt carried "CURRENT TIME:
2026-10-17" and the model searched "VUAA Vanguard S&P 500 ETF price today
2025" (its training ends in 2025). A past year the model adds to a query
about NOW is dropped before the search — web_search, deep_research (and
fact_check through it), darkweb_search, darkweb_research — and the result
tells the model why. Replay over all 3,828 logged queries changed 4, all
right (this one and two probes' "onion address 2024 2025 current")."""
from __future__ import annotations

import datetime as dt

import pytest

from ghost_agent.tools import search as S
from ghost_agent.tools import darkweb_search as D
from ghost_agent.utils import provenance as P
from ghost_agent.utils.logging import request_id_context

T = dt.date(2026, 10, 17)


@pytest.mark.parametrize("q,want,dropped", [
    ("VUAA Vanguard S&P 500 ETF price today 2025", "VUAA Vanguard S&P 500 ETF price today", ["2025"]),   # live 3c527b55
    ("bitcoin price now 2025", "bitcoin price now", ["2025"]),
    ("τιμή χρυσού σήμερα 2025", "τιμή χρυσού σήμερα", ["2025"]),
    ("weather Athens tonight 2024 2025", "weather Athens tonight", ["2024", "2025"]),
    ("current CEO of Twitter now 2025 ?", "current CEO of Twitter now?", ["2025"]),
    ("gold price now 2024 2025", "gold price now", ["2024", "2025"]),      # a run after the word
    ("Athens concerts this week 2025", "Athens concerts this week", ["2025"]),   # battery B4
])
def test_a_past_year_on_a_query_about_now_is_dropped(q, want, dropped):
    assert S.drop_stale_year(q, "", T) == (want, dropped)


@pytest.mark.parametrize("q,user", [
    ("Tempi train crash 2023", ""),                          # no now-word: history
    ("olive oil price 2025 record", ""),                     # price alone is not "now"
    ("best laptops 2024", ""),
    ("opencode best option current 2025 2026", ""),          # the model already names this year
    ("VUAA price today 2025", "what was the VUAA price in 2025?"),   # the USER said 2025
    ("python latest release 2024-10", ""),                   # part of a date, not a bare year
    ("Greece current account deficit 2023", ""),             # economics, not now
    # r2 MAJOR 1-3: a year beside a noun, not beside a point-in-time word
    ("SQL Server 2025 latest cumulative update", ""),
    ("latest Visual Studio 2022 update now", ""),
    ("Office 2024 current price", ""),
    ("Euro 2024 final today replay", ""),
    ("house prices now higher than 2024", ""),
    ("Greece inflation current 2025", ""),
    ("current F1 season standings 2025", ""),
    ("VUAA price today October 2025", ""),                  # a date: dropping only the year would break it
    ("BBC News official onion address 2024 2025 current", ""),   # "current" is not point-in-time
    ("σύγκριση τιμών σήμερα έναντι 2024", ""),
    ("Office 2024 current price today", ""),                # battery A4: next to "current", not to "today"
    # r3: a year BEFORE the word is a product / model / edition
    ("Hyundai Tucson 2025 now recall", ""),
    ("Windows Server 2025 now available ARM", ""),
    ("ISO 27001 2022 currently", ""),
    ("2025 today news Greece", ""),
    ("latest news 2027", ""),                                # a future year is left alone
    ("Switch 2 price today", ""),                            # no year at all
    ("postgres version released 2023", "what version of postgres was released in 2023 ?"),   # operator's case
    ("latest postgres release 2023", "what version of postgres was released in 2023 ?"),     # now-word, but the USER said 2023
    # review r1 MAJOR A: a year the user wrote at a sentence end / in a code
    ("latest Fed decisions 2025", "Show me the latest Fed decisions of 2025."),
    ("current FY results 2025 Apple", "how did Apple do in FY2025?"),
    # review r1 MAJOR B: the past year is the SUBJECT
    ("VUAA performance since 2025 now", ""),
    ("latest 2025 annual report Apple 10-K", ""),
    ("current 2025 tax brackets", ""),
    ("latest iphone in 2024", ""),
    ("current Q3 2025 GDP Greece", ""),
    ("VUAA price today vs 31 Dec 2025", ""),
    # MINOR: an amount, a possessive, "live"
    ("price today $2025", ""),
    ("latest 2025's best albums", ""),
    ("live 2024 recordings", ""),
])


def test_everything_else_is_left_alone(q, user):
    assert S.drop_stale_year(q, user, T) == (q, [])


def test_a_year_from_an_earlier_user_message_is_the_users():
    """r1 MAJOR B: turn 1 named the date, turn 2 asks about today."""
    said = ["VUAA close on 31 Dec 2025?", "and today, how much has it gained since then?"]
    assert S.drop_stale_year("VUAA price today and 2025 close", said, T) == ("VUAA price today and 2025 close", [])


def test_only_the_copy_after_the_word_goes():
    """r3 MINOR: "Camry 2025 price today 2025" keeps the model year."""
    assert S.drop_stale_year("Camry 2025 price today 2025", "", T) == ("Camry 2025 price today", ["2025"])


def test_a_non_string_query_passes_through():
    assert S._drop_stale_year_logged(["a list"], "web_search") == (["a list"], [])
    assert S._drop_stale_year_logged(None, "web_search") == (None, [])


def test_brackets_left_by_a_dropped_year_are_cleaned():
    assert S.drop_stale_year("news today (2025)", "", T) == ("news today", ["2025"])


def test_recent_user_messages_are_kept_per_request():
    P.note_recent_user_messages(["what was VUAA in 2025?", "and now?"], "web-4pu-a")
    P.note_user_message("and now?", "web-4pu-a")
    P.note_recent_user_messages(["hello"], "web-4pu-b")
    assert P.recent_user_messages("web-4pu-a") == ["what was VUAA in 2025?", "and now?"]
    assert P.recent_user_messages("web-4pu-b") == ["hello"]


async def test_web_search_keeps_a_year_from_an_earlier_turn(monkeypatch, rid):
    sent = []

    async def fake(query, tor_proxy):
        sent.append(query)
        return "RESULTS"
    monkeypatch.setattr(S, "tool_search_ddgs", fake)
    P.note_recent_user_messages(["what did VUAA close at in 2025?", "and today?"], rid)
    P.note_user_message("and today?", rid)
    await S.tool_search(query="VUAA price today 2025 close")
    assert sent == ["VUAA price today 2025 close"]


async def test_darkweb_search_drops_it_from_a_string_extra_query(monkeypatch, rid):
    got = []

    async def core(query=None, extra_queries=None, *a, **k):
        got.append(extra_queries)
        return "ONIONS"
    monkeypatch.setattr(D, "_darkweb_search_core", core)
    P.note_user_message("find it", rid)
    await D.tool_darkweb_search(query="dread forum", extra_queries="dread mirror now 2025")
    assert got == ["dread mirror now"]


def test_the_note_names_the_year_and_today():
    n = S.stale_year_note(["2025"], T)
    assert "'2025'" in n and "2026-10-17" in n and "'since 2025'" in n   # r1: how to keep it on purpose


@pytest.fixture
def rid():
    tok = request_id_context.set("web-4pu-1")
    try:
        yield "web-4pu-1"
    finally:
        request_id_context.reset(tok)


async def test_web_search_sends_the_query_without_the_year(monkeypatch, rid):
    sent = []

    async def fake(query, tor_proxy):
        sent.append(query)
        return "RESULTS: VUAA 149.38"
    monkeypatch.setattr(S, "tool_search_ddgs", fake)
    P.note_user_message("what's the price of VUAA?", rid)
    out = await S.tool_search(query="VUAA Vanguard S&P 500 ETF price today 2025")
    assert sent == ["VUAA Vanguard S&P 500 ETF price today"]
    assert "Searched without '2025'" in out


async def test_web_search_keeps_a_year_the_user_wrote(monkeypatch, rid):
    sent = []

    async def fake(query, tor_proxy):
        sent.append(query)
        return "RESULTS"
    monkeypatch.setattr(S, "tool_search_ddgs", fake)
    P.note_user_message("VUAA price today vs its 2025 high", rid)
    out = await S.tool_search(query="VUAA price today 2025")
    assert sent == ["VUAA price today 2025"] and "Searched without" not in out


async def test_deep_research_drops_it_too(monkeypatch, rid):
    got = []

    async def core(query=None, *a, **k):
        got.append(query)
        return "REPORT"
    monkeypatch.setattr(S, "_deep_research_core", core)
    P.note_user_message("current VUAA price", rid)
    out = await S.tool_deep_research(query="VUAA price now 2025", anonymous=False)
    assert got == ["VUAA price now"] and "Searched without '2025'" in out


async def test_darkweb_search_drops_it_from_every_query(monkeypatch, rid):
    got = []

    async def core(query=None, extra_queries=None, *a, **k):
        got.append((query, extra_queries))
        return "ONIONS"
    monkeypatch.setattr(D, "_darkweb_search_core", core)
    P.note_user_message("find the BBC onion address", rid)
    out = await D.tool_darkweb_search(query="BBC onion address now 2025",
                                      extra_queries=["BBC tor mirror today 2024", "BBC news onion"])
    assert got == [("BBC onion address now", ["BBC tor mirror today", "BBC news onion"])]
    assert "Searched without '2024', '2025'" in out


async def test_darkweb_research_drops_it_too(monkeypatch, rid):
    got = []

    async def core(query=None, *a, **k):
        got.append(query)
        return "REPORT"
    monkeypatch.setattr(D, "_darkweb_research_core", core)
    P.note_user_message("latest dread forum news", rid)
    out = await D.tool_darkweb_research(query="dread forum news today 2025")
    assert got == ["dread forum news today"] and "Searched without '2025'" in out


async def test_handle_chat_records_the_recent_user_messages():
    """The agent side of r1 MAJOR B: a real turn stores this request's recent
    user messages where the search tools read them."""
    from unittest.mock import AsyncMock, MagicMock
    from ghost_agent.core.agent import GhostAgent, GhostContext
    args = MagicMock()
    args.model, args.temperature, args.max_context, args.smart_memory = "m", 0.0, 8000, 0.0
    ctx = GhostContext(args=args, sandbox_dir=None, memory_dir=None, tor_proxy=None)
    ctx.llm_client = MagicMock()
    ctx.scratchpad = MagicMock()
    ctx.scratchpad.list_all.return_value = "Empty"
    ctx.llm_client.chat_completion = AsyncMock(return_value={"choices": [{"message": {
        "role": "assistant", "content": "It closed at 120.", "tool_calls": []}}]})
    agent = GhostAgent(ctx)
    await agent.handle_chat({"messages": [{"role": "user", "content": "VUAA close on 31 Dec 2025?"},
                                          {"role": "assistant", "content": "120."},
                                          {"role": "user", "content": "and today?"}]},
                            MagicMock(), request_id="web-4pu-hc")
    said = P.recent_user_messages("web-4pu-hc")
    assert "VUAA close on 31 Dec 2025?" in said and "and today?" in said


def test_an_anchor_before_a_run_of_years_still_counts():
    """battery R4: "since 2024 2025" — the anchor sits before the FIRST year."""
    assert S.drop_stale_year("VUAA gains since 2024 2025 today", "", T) == ("VUAA gains since 2024 2025 today", [])
    # …also when the first year is skipped because the user wrote it
    # ("in" is only a word-before anchor; "since" is also a whole-query one)
    assert S.drop_stale_year("VUAA prices in 2024 2025 today", "and in 2024?", T) == (
        "VUAA prices in 2024 2025 today", [])


async def test_deep_research_rechecks_after_the_anonymising_rewrite(monkeypatch, rid):
    """r1 MINOR: the LLM re-author can put the dropped year back."""
    import ghost_agent.utils.stylometry as ST

    async def neutral(q, **k):
        return q + " 2025"
    seen = []

    class Stop(Exception):
        pass

    def sanit(q):
        seen.append(q)
        raise Stop
    monkeypatch.setattr(ST, "neutralize_query", neutral)
    monkeypatch.setattr(S, "_sanitize_query", sanit)
    P.note_user_message("current VUAA price", rid)
    with pytest.raises(Stop):
        await S.tool_deep_research(query="VUAA price now 2025", anonymous=True)
    assert seen == ["VUAA price now"]


async def test_darkweb_research_rechecks_after_the_anonymising_rewrite(monkeypatch, rid):
    """r2 MAJOR 4: the same LLM re-author runs in darkweb_research."""
    import ghost_agent.utils.stylometry as ST

    async def neutral(q, **k):
        return q + " 2025"

    class Stop(Exception):
        pass
    seen = []

    def scrub(q, *a, **k):
        seen.append(q)
        raise Stop
    monkeypatch.setattr(ST, "neutralize_query", neutral)
    monkeypatch.setattr(D, "_apply_anonymous_scrub", scrub)
    P.note_user_message("dread news", rid)
    with pytest.raises(Stop):
        await D.tool_darkweb_research(query="dread forum news today 2025", anonymous=True)
    assert seen == ["dread forum news today"]
