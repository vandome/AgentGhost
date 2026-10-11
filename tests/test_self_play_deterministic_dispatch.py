"""Deterministic terminal-tool dispatch for explicit terminal commands.

Prior bug: a bare "self play" ran the cycle only on the FIRST ask. On
every repeat ("self play again") the agent replayed the previous turn's
text summary instead of re-firing the tool. Root cause: the
direct-from-tool-summary bypass persists the cycle result as a PLAIN-TEXT
assistant turn with NO tool_call, and the memory bus re-hydrates it as
"Context: Self-play complete…", so the model's only in-context example
says the right response to "self play" is to reprint that text — which it
does (identical ~126-char reply, tool=- in the metacog trace), never
re-running.

Fix: when the user's message is unambiguously a terminal command, the
agent dispatches the tool deterministically BEFORE the LLM turn and
records it in history as a real tool_call + result. §4NM: the self_play
tool and `_is_single_self_play_command` are gone; dream_mode is the only
terminal tool left, and these tests pin its intent gate and the
end-to-end "runs every time, even with poisoned history" behaviour.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ghost_agent.core.agent import (
    GhostAgent,
    GhostContext,
    _is_dream_command,
    _explicit_terminal_command,
)


# --------------------------------------------------------------------------
# Pure intent gate
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "dream mode",
    "go to sleep",
    "time to sleep",
    "consolidate memories",
    "consolidate your memories",
    "memory consolidation",
    "go to sleep now please",
])
def test_dream_command_matches(text):
    assert _is_dream_command(text) is True


@pytest.mark.parametrize("text", [
    "",
    "i had an interesting dream about a flask app last night",  # prose, >6 words + "rest"/"dream" traps
    "tell me how memory consolidation works in the brain in detail",  # prose mention
    "build me a flask app",                                          # unrelated
])
def test_non_dream_command_rejected(text):
    assert _is_dream_command(text) is False


def test_explicit_terminal_command_routing():
    assert _explicit_terminal_command("dream mode") == "dream_mode"
    assert _explicit_terminal_command("go to sleep") == "dream_mode"
    assert _explicit_terminal_command("build me a flask app") is None
    # §4NM: self-play commands no longer route anywhere (the tool is gone).
    assert _explicit_terminal_command("self play again") is None
    assert _explicit_terminal_command("keep practicing in a loop") is None


# --------------------------------------------------------------------------
# End-to-end dispatch
# --------------------------------------------------------------------------

@pytest.fixture
def agent():
    context = MagicMock(spec=GhostContext)
    context.llm_client = MagicMock()
    context.llm_client.vision_clients = None
    # No worker pool → query-expansion / routing helpers take their legacy
    # synchronous fallback instead of awaiting a MagicMock `route`.
    context.llm_client.worker_clients = None
    context.memory_bus = None
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


class FakeBgTasks:
    def add_task(self, *a, **k):
        pass


_DREAM_RESULT = (
    "Active Memory Consolidation complete. Merged 3 episodic memories.\n\n"
    "SYSTEM: SESSION FINISHED. STAND BY."
)


@pytest.mark.asyncio
async def test_explicit_dream_dispatches_without_consulting_llm(agent):
    """dream_mode is the other terminal tool subject to the same replay bug;
    an explicit 'go to sleep' must dispatch deterministically and bypass the
    LLM, even with a poisoned prior-summary turn in history."""
    agent.context.llm_client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {
            "content": "Dream cycle complete.\n\nActive Memory Consolidation complete.",
            "tool_calls": [],
        }}]}
    )
    dream_mock = AsyncMock(return_value=_DREAM_RESULT)
    agent.available_tools = {"dream_mode": dream_mock}

    body = {"messages": [
        {"role": "user", "content": "go to sleep"},
        {"role": "assistant", "content": "Dream cycle complete.\n\nActive Memory Consolidation complete."},
        {"role": "user", "content": "go to sleep"},
    ]}

    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[{"function": {"name": "dream_mode"}}]):
        result = await agent.handle_chat(body, FakeBgTasks())

    assert dream_mock.call_count == 1
    assert agent.context.llm_client.chat_completion.call_count == 0
    final_content = result[0] if isinstance(result, tuple) else result
    assert isinstance(final_content, str)
    assert "Dream cycle complete." in final_content
    # The STAND BY trailer is for other consumers — must not leak.
    assert "STAND BY" not in final_content
