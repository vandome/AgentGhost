"""`GhostContext.last_user_content` — the current turn's user text.

Incident context (2026-04-24, webOS build session): a long-context
hallucination made the model call the `self_play` tool the user never
asked for. The fix stashed the turn's user text on the context
(`handle_chat` sets it right after parsing the request body) so tools
can check what the user actually asked. §4NM: the `self_play` /
`self_play_loop` tools and their intent guard are gone; the context
field remains (read widely across agent.py), and these tests pin it.
"""

from types import SimpleNamespace


# ---------------------------------------------------------------------------
# GhostContext wiring — last_user_content must be stashed by handle_chat
# ---------------------------------------------------------------------------


def test_ghost_context_has_last_user_content_field():
    """The attribute must exist on a fresh context so tools never hit
    AttributeError before the first handle_chat runs."""
    from ghost_agent.core.agent import GhostContext

    args = SimpleNamespace(model="m")
    ctx = GhostContext(args=args, sandbox_dir="/tmp", memory_dir="/tmp", tor_proxy=None)
    assert hasattr(ctx, "last_user_content")
    assert ctx.last_user_content == ""


def test_handle_chat_source_stashes_last_user_content():
    """Grep-assert that handle_chat actually assigns
    `self.context.last_user_content = last_user_content`. Without the
    assignment, the guard would refuse every legitimate call because
    the field stays at its empty-string default.
    """
    from pathlib import Path

    agent_path = (
        Path(__file__).resolve().parent.parent
        / "src" / "ghost_agent" / "core" / "agent.py"
    )
    src = agent_path.read_text(encoding="utf-8")
    assert "self.context.last_user_content = last_user_content" in src, (
        "handle_chat must stash last_user_content on the context so "
        "tools can see the current turn's text"
    )
