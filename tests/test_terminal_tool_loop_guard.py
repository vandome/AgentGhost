"""Terminal-tool loop guard.

Prior bug: on the second `run self play` user message in a conversation,
the agent would call the `self_play` tool 2-3 times in a row instead of
once. The main LLM, seeing the system prompt's "Call this tool EVERY
TIME the user requests it" alongside the prior self_play run still in
history, kept re-issuing `<tool_call>self_play</tool_call>` even though
the tool's own result text said "DO NOT call the `self_play` tool again
automatically."

Fix: after a terminal tool (`dream_mode`; `self_play` until §4NM) returns
successfully, set `force_final_response=True`, and drop any tool_calls the
LLM still tries to emit on the next turn. These tests assert both halves of
that contract. §4NM: the self_play test went with the tool.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ghost_agent.core.agent import GhostAgent, GhostContext


@pytest.fixture
def agent():
    context = MagicMock(spec=GhostContext)
    context.llm_client = MagicMock()
    context.llm_client.vision_clients = None
    context.sandbox_dir = "/tmp/sandbox"
    context.args = MagicMock()
    context.args.shell = "bash"
    context.args.max_context = 8000
    context.args.temperature = 0.5
    context.args.smart_memory = 0.0
    context.args.model = "qwen3.6"
    context.args.perfect_it = False
    context.profile_memory = MagicMock()
    context.profile_memory.get_context_string.return_value = ""
    context.memory_system = None
    context.skill_memory = None
    context.scratchpad = MagicMock()
    context.scratchpad.list_all.return_value = ""
    return GhostAgent(context)


def _tool_call_xml(name: str) -> str:
    return f'<tool_call>\n<function name="{name}">\n</function>\n</tool_call>'


@pytest.mark.asyncio
async def test_dream_mode_runs_exactly_once_per_user_request(agent):
    """dream_mode is a terminal tool: it runs once per user request."""
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"content": _tool_call_xml("dream_mode"), "tool_calls": []}}]},
        {"choices": [{"message": {"content": _tool_call_xml("dream_mode"), "tool_calls": []}}]},
        {"choices": [{"message": {"content": "Dream cycle summarized.", "tool_calls": []}}]},
    ])
    dream_mock = AsyncMock(return_value=(
        "Dream cycle completed.\n\nSYSTEM: SESSION FINISHED. STAND BY."
    ))
    agent.available_tools = {"dream_mode": dream_mock}

    body = {"messages": [{"role": "user", "content": "go to sleep"}]}

    class FakeBgTasks:
        def add_task(self, *a, **k): pass

    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[{"function": {"name": "dream_mode"}}]):
        await agent.handle_chat(body, FakeBgTasks())

    assert dream_mock.call_count == 1, (
        f"Expected dream_mode to run exactly once; ran {dream_mock.call_count} times."
    )


@pytest.mark.asyncio
async def test_non_terminal_tool_can_be_called_twice_if_model_asks(agent):
    """Negative control: the loop guard must NOT fire on ordinary tools.
    `file_system` / `execute` are routinely called multiple times per
    user request; suppressing those would break normal task flow."""
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"content": _tool_call_xml("echo"), "tool_calls": []}}]},
        {"choices": [{"message": {"content": _tool_call_xml("echo"), "tool_calls": []}}]},
        {"choices": [{"message": {"content": "All done.", "tool_calls": []}}]},
    ])
    echo_mock = AsyncMock(return_value="ok")
    agent.available_tools = {"echo": echo_mock}

    body = {"messages": [{"role": "user", "content": "echo twice"}]}

    class FakeBgTasks:
        def add_task(self, *a, **k): pass

    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[{"function": {"name": "echo"}}]):
        await agent.handle_chat(body, FakeBgTasks())

    assert echo_mock.call_count == 2, (
        f"Non-terminal tool should be callable multiple times; ran {echo_mock.call_count}."
    )
