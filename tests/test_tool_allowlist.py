"""GHOST_TOOL_ALLOWLIST — a deployment's fixed tool surface (personal mode).

World where each pin fails:
  * unset / empty / separators-only changes the advertised or dispatchable
    tool set;
  * a registered tool that is not on the list is still advertised to the
    model, or is still in the dispatch dict;
  * the dispatch dict is replaced instead of filtered in place, so the
    composed-skill step executor (which captures it) still reaches every
    tool;
  * the list re-enables something another gate removed (`disabled`,
    missing configuration, a sub-agent allowlist);
  * an unknown name is not warned about, warned about twice, or the warning
    carries more than names;
  * a call to a removed tool runs — including the fuzzy-canonicalisation
    route where `darkweb_search` quietly dispatched as `web_search`;
  * the MCTS planner's tool list ignores the allowlist;
  * one of the deployment's personal-mode names no longer exists.
"""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core import agent as A
from ghost_agent.core import experiments as E
from ghost_agent.core.strikes import StrikeLedger
from ghost_agent.tools import registry as R
from tests.test_4jj_search_yield_steer import _agent as _sy_agent, _ts as _sy_ts
from tests.test_4kl_member_capability import _no_arm_calls

# deploy/config.env PERSONAL_TOOL_ALLOWLIST (ghostai), verbatim.
PERSONAL = ("file_system,execute,web_search,deep_research,recall,knowledge_base,"
            "update_profile,scratchpad,manage_tasks,abort_attempt,replan,system_utility")
PERSONAL_SET = frozenset(PERSONAL.split(","))


@pytest.fixture(autouse=True)
def _fresh_allowlist_state(monkeypatch):
    monkeypatch.setattr(R, "_ALLOWLIST_DROPPED", set())
    monkeypatch.setattr(R, "_ALLOWLIST_WARNED", set())
    monkeypatch.setattr(R, "_ALLOWLIST_ANNOUNCED", set())
    monkeypatch.setattr(R, "pretty_log", lambda *a, **k: None)


def _ctx(**over):
    ctx = SimpleNamespace(
        llm_client=SimpleNamespace(swarm_clients=None, image_gen_clients=None),
        args=SimpleNamespace(default_db=None, max_context=8000, anonymous=True, model="m"),
        sandbox_dir=None, memory_dir=None, memory_system=None, tor_proxy=None,
        sandbox_manager=None, profile_memory=None, scratchpad=None, scheduler=None,
        skill_memory=None, llm=None,
    )
    for k, v in over.items():
        setattr(ctx, k, v)
    return ctx


def _adv(ctx=None, **kw):
    return [t["function"]["name"]
            for t in R.get_active_tool_definitions(ctx or _ctx(), serve_tuned=False, **kw)]


# ── parsing ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expect", [
    (None, None),
    ("", None),
    ("  ", None),
    (" , ,, ", None),
    (" web_search , recall,,", frozenset({"web_search", "recall"})),
    ("Web_Search", frozenset({"Web_Search"})),          # exact, no case folding
])
def test_the_variable_parses_to_trimmed_exact_names(monkeypatch, raw, expect):
    if raw is None:
        monkeypatch.delenv(R.TOOL_ALLOWLIST_ENV, raising=False)
    else:
        monkeypatch.setenv(R.TOOL_ALLOWLIST_ENV, raw)
    assert R.tool_allowlist() == expect


# ── unset / empty is a no-op ──────────────────────────────────────────

@pytest.mark.parametrize("raw", [None, "", " , "])
def test_unset_or_empty_changes_nothing(monkeypatch, raw):
    monkeypatch.delenv(R.TOOL_ALLOWLIST_ENV, raising=False)
    base_adv, base_disp = _adv(), set(R.get_available_tools(_ctx()))
    if raw is not None:
        monkeypatch.setenv(R.TOOL_ALLOWLIST_ENV, raw)
    assert _adv() == base_adv
    assert set(R.get_available_tools(_ctx())) == base_disp
    assert {"browser", "delegate", "manage_projects"} <= base_disp
    assert R.tool_refused_by_allowlist("browser") is False
    assert R.tool_allowed("browser") is True


# ── the personal-mode list ────────────────────────────────────────────

def test_every_personal_mode_name_is_still_a_registered_tool(monkeypatch):
    """A rename upstream would silently shrink the personal surface."""
    monkeypatch.delenv(R.TOOL_ALLOWLIST_ENV, raising=False)
    assert PERSONAL_SET <= set(R.get_available_tools(_ctx()))
    assert PERSONAL_SET <= set(_adv())


def test_personal_mode_advertises_and_dispatches_only_the_allowlist(monkeypatch):
    monkeypatch.setenv(R.TOOL_ALLOWLIST_ENV, PERSONAL)
    assert set(_adv()) == PERSONAL_SET
    assert set(R.get_available_tools(_ctx())) == PERSONAL_SET


def test_the_served_prompt_path_is_filtered_too(monkeypatch):
    """serve_tuned=True is the prompt build; the filter runs before it."""
    monkeypatch.setenv(R.TOOL_ALLOWLIST_ENV, PERSONAL)
    names = {t["function"]["name"] for t in R.get_active_tool_definitions(_ctx(), query="hi")}
    assert names == PERSONAL_SET


def test_the_dispatch_dict_is_filtered_in_place(monkeypatch):
    """The macro step executor holds the dict get_available_tools builds;
    a filtered COPY would leave it every tool (and the macro itself)."""
    captured = {}

    def _wire(tools, context):
        captured["ref"] = tools
        tools["my_macro"] = AsyncMock()
        return 1
    monkeypatch.setattr(R, "register_composed_skill_runners", _wire)
    monkeypatch.setenv(R.TOOL_ALLOWLIST_ENV, "web_search,recall")
    out = R.get_available_tools(_ctx())
    assert out is captured["ref"]
    assert set(captured["ref"]) == {"web_search", "recall"}
    assert R.tool_refused_by_allowlist("my_macro") is True


def test_a_listed_macro_or_skill_stays(monkeypatch):
    monkeypatch.setattr(R, "register_composed_skill_runners",
                        lambda tools, context: tools.update(my_macro=AsyncMock()) or 1)
    monkeypatch.setenv(R.TOOL_ALLOWLIST_ENV, "web_search,my_macro")
    assert set(R.get_available_tools(_ctx())) == {"web_search", "my_macro"}


# ── it only ever removes ──────────────────────────────────────────────

def test_the_allowlist_cannot_re_enable_a_disabled_tool(monkeypatch):
    monkeypatch.setenv(R.TOOL_ALLOWLIST_ENV, PERSONAL)
    assert set(_adv(disabled={"execute"})) == PERSONAL_SET - {"execute"}


def test_the_allowlist_cannot_add_an_unconfigured_tool(monkeypatch):
    """postgres_admin without a DB, delegate_to_swarm without a swarm and
    image_generation without an image node stay out even when listed."""
    monkeypatch.setenv(R.TOOL_ALLOWLIST_ENV,
                       "web_search,postgres_admin,delegate_to_swarm,image_generation")
    assert set(_adv()) == {"web_search"}
    assert "image_generation" not in R.get_available_tools(_ctx())


def test_a_dispatch_miss_rebuild_cannot_heal_a_removed_tool(monkeypatch):
    """A sub-agent allowlist that names `browser` cannot bring it back."""
    monkeypatch.setenv(R.TOOL_ALLOWLIST_ENV, "web_search,recall")
    ctx = _ctx(_subagent_allowed_tools=frozenset({"browser", "web_search"}))
    agent = A.GhostAgent.__new__(A.GhostAgent)
    agent.context = ctx
    agent.disabled_tools = set()
    assert set(agent._rebuild_available_tools()) == {"web_search"}


# ── unknown names ─────────────────────────────────────────────────────

def test_unknown_names_are_warned_once_names_only(monkeypatch):
    log = MagicMock()
    monkeypatch.setattr(R, "logger", log)
    monkeypatch.setenv(R.TOOL_ALLOWLIST_ENV, "web_search, no_such_tool ,zz_gone")
    out1 = R.get_available_tools(_ctx())
    R.get_available_tools(_ctx())
    assert set(out1) == {"web_search"}
    assert log.warning.call_count == 1
    fmt, env, names = log.warning.call_args.args
    assert env == "GHOST_TOOL_ALLOWLIST" and names == "no_such_tool, zz_gone"
    assert R.tool_refused_by_allowlist("no_such_tool") is False   # not a tool: unknown-tool path


# ── dispatch refusal ──────────────────────────────────────────────────

@pytest.mark.parametrize("name,refused", [
    ("browser", True),
    ("darkweb_search", True),
    ("Darkweb-Search", True),           # a separator/case variant of a removed tool
    ("web_search", False),
    ("Web-Search", False),              # a variant of an allowed tool
    ("websearch", False),
    ("totally_made_up", False),         # unregistered: the unknown-tool path
    ("", False),
    (None, False),
])
def test_refusal_is_keyed_on_removed_registered_names(monkeypatch, name, refused):
    monkeypatch.setenv(R.TOOL_ALLOWLIST_ENV, PERSONAL)
    R.get_available_tools(_ctx())
    assert R.tool_refused_by_allowlist(name) is refused


def _dispatch_agent(monkeypatch, allow):
    monkeypatch.setenv(R.TOOL_ALLOWLIST_ENV, allow)
    _no_arm_calls(monkeypatch, E.CONTROL)
    agent = _sy_agent()               # GhostAgent.__init__ ran get_available_tools under the env
    search = AsyncMock(return_value="### 1. hit\nsnippet\n[Source: https://example.org/1]\n")
    agent.available_tools = {"web_search": search}       # what the filtered registry yields
    monkeypatch.setattr(agent, "_rebuild_available_tools", lambda: agent.available_tools)
    return agent, search


@pytest.mark.parametrize("called", ["darkweb_search", "browser", "darkwebsearch"])
async def test_a_call_to_a_removed_tool_is_refused_not_remapped(monkeypatch, called):
    """`darkweb_search` is within difflib's 0.7 cutoff of `web_search`, so
    canonicalisation alone would have RUN web_search for it."""
    agent, search = _dispatch_agent(monkeypatch, "web_search,recall")
    ts = _sy_ts([(called, {"query": "q"})], StrikeLedger(), set())
    await agent._dispatch_and_process_tool_batch(ts)
    search.assert_not_called()
    refusals = [m for m in ts.messages if m.get("role") == "tool"]
    assert len(refusals) == 1
    assert f"Tool '{called}' is not enabled in this deployment" in str(refusals[0]["content"])
    assert ts.execution_failure_count == 1 and ts.last_was_failure is True


async def test_an_allowed_tool_still_dispatches(monkeypatch):
    agent, search = _dispatch_agent(monkeypatch, "web_search,recall")
    ts = _sy_ts([("web_search", {"query": "q"})], StrikeLedger(), set())
    await agent._dispatch_and_process_tool_batch(ts)
    search.assert_awaited_once()
    assert ts.execution_failure_count == 0


async def test_unset_keeps_the_old_disabled_message(monkeypatch):
    monkeypatch.delenv(R.TOOL_ALLOWLIST_ENV, raising=False)
    _no_arm_calls(monkeypatch, E.CONTROL)
    agent = _sy_agent()
    agent.available_tools = {"web_search": AsyncMock()}
    agent.disabled_tools = {"web_search"}
    ts = _sy_ts([("web_search", {"query": "q"})], StrikeLedger(), set())
    await agent._dispatch_and_process_tool_batch(ts)
    msg = [m for m in ts.messages if m.get("role") == "tool"][0]
    assert "is explicitly disabled in this context" in str(msg["content"])


# ── planners ──────────────────────────────────────────────────────────

def test_the_mcts_planner_tool_list_honours_the_allowlist():
    """The MCTS turn-start lookahead reads TOOL_DEFINITIONS directly; its
    name list must go through tool_allowed."""
    import inspect
    src = inspect.getsource(A.GhostAgent)
    i = src.index("TOOL_DEFINITIONS as _MCTS_TD")
    assert "tool_allowed(" in src[i:i + 600]


def test_the_planner_name_line_comes_from_the_filtered_definitions(monkeypatch):
    """The task-tree planner builds its 'available tools' line from
    get_active_tool_definitions(serve_tuned=False) — filtered there."""
    monkeypatch.setenv(R.TOOL_ALLOWLIST_ENV, "web_search,recall")
    assert set(_adv()) == {"web_search", "recall"}


def test_the_personal_head_is_much_smaller(monkeypatch):
    """Sanity bound for PERSONAL_MAX_CONTEXT=20480: the advertised schema
    under the personal list is a fraction of the full one."""
    monkeypatch.delenv(R.TOOL_ALLOWLIST_ENV, raising=False)
    full = len(json.dumps(R.get_active_tool_definitions(_ctx(), serve_tuned=False)))
    monkeypatch.setenv(R.TOOL_ALLOWLIST_ENV, PERSONAL)
    small = len(json.dumps(R.get_active_tool_definitions(_ctx(), serve_tuned=False)))
    assert small < full / 2
