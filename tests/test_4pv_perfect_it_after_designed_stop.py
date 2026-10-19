"""§4PV: the Perfect-It follow-up must not run on a turn that ended at a
designed stop (the turn asked the user something). §4PS made a confirm wait
no longer count as a failure, which opened its gate: an `execute` earlier in
the turn plus a short closing question was enough to append an
"optimisation suggestion" under a question the user still has to answer."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from ghost_agent.core.agent import GhostAgent
from ghost_agent.tools.outcome import DESIGNED_STOP_REASONS, ToolOutcome


def _agent():
    context = MagicMock()
    context.llm_client.chat_completion = AsyncMock()
    context.args.smart_memory = 0.0
    context.args.temperature = 0.1
    context.args.max_context = 8000
    context.args.perfect_it = True
    context.sandbox_dir = "/tmp"
    context.memory_system = MagicMock()
    context.profile_memory = MagicMock()
    context.profile_memory.get_context_string.return_value = ""
    agent = GhostAgent(context)
    agent.available_tools = {
        "execute": AsyncMock(return_value="EXIT CODE: 0"),
        "test_tool": AsyncMock(return_value="Result"),
    }
    return agent


def _call(i, name, args):
    return {"choices": [{"message": {"content": None, "tool_calls": [{
        "id": f"call_{i}", "function": {"name": name, "arguments": args}}]}}]}


def _perfect_it_calls(agent):
    return [c for c in agent.context.llm_client.chat_completion.call_args_list
            if "Perfection Protocol" in str(c[0][0].get("messages", []))
            and "<system_directive>" in str(c[0][0].get("messages", []))]


async def _run(agent, second_result):
    agent.available_tools["test_tool"].return_value = second_result
    agent.context.llm_client.chat_completion.side_effect = [
        _call(1, "execute", '{"code": "print(1)"}'),
        _call(2, "test_tool", '{}'),
        {"choices": [{"message": {"content": "Which one?"}}]},
        {"choices": [{"message": {"content": "Optimization: Do X"}}]},
        {"choices": [{"message": {"content": "spare"}}]},
        {"choices": [{"message": {"content": "spare"}}]},
    ]
    return await agent.handle_chat(
        {"messages": [{"role": "user", "content": "Run code"}]}, [])


@pytest.mark.asyncio
async def test_control_plain_result_still_gets_perfect_it():
    # The control: same flow, the second tool just succeeds → Perfect-It runs.
    agent = _agent()
    await _run(agent, "Result")
    assert _perfect_it_calls(agent), "control: Perfect-It should run on a clean heavy turn"


@pytest.mark.asyncio
@pytest.mark.parametrize("code", sorted(DESIGNED_STOP_REASONS))
async def test_designed_stop_skips_perfect_it(code):
    agent = _agent()
    await _run(agent, ToolOutcome.rejected("Which file do you mean?", reason_code=code))
    assert not _perfect_it_calls(agent), f"Perfect-It ran after a designed stop ({code})"


@pytest.mark.asyncio
async def test_plain_rejection_is_not_a_designed_stop_and_stays_blocked_as_before():
    # A non-designed refusal is a failure → Perfect-It was already skipped;
    # the new guard must not change that either way.
    agent = _agent()
    await _run(agent, ToolOutcome.rejected("NOT done: refused", reason_code="something_else"))
    assert not _perfect_it_calls(agent)
