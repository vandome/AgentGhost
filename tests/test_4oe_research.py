"""§4OE-4 — a research request remembers what it fetched, caps its searches,
cites only what it read, and continues a cut report with its sources in view."""
from unittest.mock import AsyncMock

from ghost_agent.core import fetch_memo as FM
from ghost_agent.core import link_grounding as LG
from ghost_agent.core import strikes as ST
from ghost_agent.core import write_continuation as W
from ghost_agent.core.agent import GhostAgent
from ghost_agent.tools.outcome import ToolOutcome
from tests.helpers import FakeBgTasks, make_context
from tests.test_4ji_request_echo_narration import _resp, _tc

PAGE = "--- BROWSER RESULT ---\nSTATUS: OK\nURL: https://ex.org/a\nTITLE: A\n\nThe page says the thing."


def test_fetch_memo_keys_remembers_and_answers_repeats():
    k = FM.fetch_key("browser", {"operation": "navigate", "url": "https://Ex.org/a/"})
    assert k == ("browser", "https://ex.org/a", "navigate", "")
    assert FM.fetch_key("browser", {"operation": "screenshot", "url": "https://ex.org/a"}) is None
    assert FM.fetch_key("browser", {"operation": "navigate", "url": "https://ex.org/a", "actions": [{"action": "click"}]}) is None
    assert FM.fetch_key("web_search", {"query": "  Vector   DBs "}) == ("web_search", "vector dbs", "", "")
    assert FM.fetch_key("file_system", {"path": "x"}) is None
    memo = {}
    FM.remember(memo, k, PAGE)
    assert FM.repeat_result(memo, k).startswith("ALREADY READ this request") and "The page says the thing" in FM.repeat_result(memo, k)
    assert FM.repeat_result(memo, ("browser", "https://ex.org/b", "navigate", "")) is None
    FM.remember(memo, k, "x" * 5000)
    assert len(memo[k]) == FM.MEMO_HEAD_CHARS


def test_search_counters_and_fetched_urls_and_snippet_free_haystack():
    led = ST.StrikeLedger()
    for _ in range(3):
        led.note_search_yield("web_search")
    led.note_search_yield("browser")
    assert led.search_run == 0 and led.search_total == 3 and led.SEARCH_TOTAL_CAP == ST.SEARCH_TOTAL_CAP
    runs = [{"name": "browser", "content": ToolOutcome.ok(PAGE)},
            {"name": "browser", "content": ToolOutcome.failed("--- BROWSER RESULT ---\nSTATUS: ERROR\nURL: https://ex.org/bad\nTimeout")},
            {"name": "web_search", "content": "1. https://ex.org/snippet — a snippet"}]
    assert LG.fetched_urls(runs) == ["https://ex.org/a"]
    dr = [{"name": "deep_research", "content": ToolOutcome.ok("## Findings\n### SOURCE: https://ex.org/deep\nstuff\n### SOURCE: https://ex.org/deep2\n")}]
    assert LG.fetched_urls(dr) == ["https://ex.org/deep", "https://ex.org/deep2"]      # review: deep_research pages count
    dead = [{"name": "deep_research", "content": ToolOutcome.failed("--- DEEP RESEARCH RESULT ---\n### SOURCE: https://ex.org/dead\nError: per-URL timeout\n")}]
    assert LG.fetched_urls(dead) == []                                                    # a declared failure read nothing
    mixed = [{"name": "deep_research", "content": ToolOutcome.ok("### SOURCE: https://ex.org/ok\nreal text\n### SOURCE: https://ex.org/timeout\nError: per-URL timeout (80 s)\n")}]
    assert LG.fetched_urls(mixed) == ["https://ex.org/ok"]                               # a failed source block is not a page read
    assert FM.fetch_key("browser", {"op": "navigate", "link": "https://ex.org/a"}) == ("browser", "https://ex.org/a", "navigate", "")
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "s1", "function": {"name": "web_search", "arguments": "{\"query\": \"q\"}"}}]},
            {"role": "tool", "tool_call_id": "s1", "content": "1. https://ex.org/snippet-only — seen in a snippet"},
            {"role": "tool", "tool_call_id": "b1", "content": PAGE}]
    hay = LG.haystack_from(msgs, [], snippet_ids={"s1"})               # the helper can leave snippets out …
    assert "snippet-only" not in hay and "ex.org/a" in hay
    assert "snippet-only" in LG.haystack_from(msgs, [])                 # … the reply path keeps them (§4LH design)


def test_the_continuation_prompt_carries_the_sources_for_text_deliverables():
    note = W.sources_note(["https://ex.org/a", "https://ex.org/b"], "probe/report.md")
    assert "cite ONLY these" in note and "- https://ex.org/a" in note
    assert W.sources_note(["https://ex.org/a"], "gen.py") == "" and W.sources_note([], "r.md") == ""
    msgs = W.continuation_messages("r.md", "write it", "head", "tail", 1, 0, sources_note=note)
    assert "cite ONLY these" in msgs[1]["content"]


def _scripted(monkeypatch, scripted, tools):
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    monkeypatch.setenv("GHOST_CRITIC_REPAIR_BUDGET", "0")
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    ctx = make_context()
    agent = GhostAgent(ctx)
    agent.available_tools = tools
    ctx.llm_client.chat_completion = AsyncMock(side_effect=scripted)
    return agent, ctx


async def test_a_repeat_page_fetch_is_answered_from_the_memo_without_a_tool_call(monkeypatch):
    """Fails where 43 of 285 fetches re-read a page already read in the same
    request (§4OD): the second identical navigate never reaches the browser."""
    browser = AsyncMock(return_value=ToolOutcome.ok(PAGE))
    agent, ctx = _scripted(monkeypatch, [
        _resp("reading", [_tc("b1", "browser", {"operation": "navigate", "url": "https://ex.org/a"})]),
        _resp("reading again", [_tc("b2", "browser", {"operation": "navigate", "url": "https://ex.org/a"})]),
        _resp("Done: the page says the thing."),
    ], {"browser": browser, "web_search": AsyncMock(return_value="r")})
    await agent.handle_chat({"messages": [{"role": "user", "content": "what does ex.org/a say"}]}, FakeBgTasks())
    assert browser.await_count == 1
    second = [m for m in agent.context.__dict__.get("_fetch_memo", {}) ]
    assert ("browser", "https://ex.org/a", "navigate", "") in agent.context._fetch_memo
    calls = ctx.llm_client.chat_completion.call_args_list
    third = "\n".join(str(m.get("content")) for m in (calls[2].kwargs.get("messages") or calls[2].args[0].get("messages")))
    assert "ALREADY READ this request" in third


async def test_the_search_cap_refuses_the_thirty_first_search(monkeypatch):
    n = ST.SEARCH_TOTAL_CAP + 1
    scripted = [_resp(f"s{i}", [_tc(f"q{i}", "web_search", {"query": f"query number {i}"})]) for i in range(n)]
    scripted.append(_resp("Done."))
    search = AsyncMock(return_value="Results:\n1. https://ex.org/x — x")
    agent, ctx = _scripted(monkeypatch, scripted, {"web_search": search, "browser": AsyncMock(return_value=ToolOutcome.ok(PAGE))})
    agent.max_turns_override = n + 5
    await agent.handle_chat({"messages": [{"role": "user", "content": "find everything"}]}, FakeBgTasks())
    assert search.await_count == ST.SEARCH_TOTAL_CAP            # the 31st never ran
    calls = ctx.llm_client.chat_completion.call_args_list
    last = "\n".join(str(m.get("content")) for m in (calls[-1].kwargs.get("messages") or calls[-1].args[0].get("messages")))
    assert "SYSTEM SEARCH CAP" in last and "SYSTEM ALERT (search budget)" in last
