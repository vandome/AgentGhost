"""Regression: journal writes must respect ``--smart-memory``.

CLAUDE.md contract: "Memory writes are gated on ``--smart-memory`` /
``--no-memory``. New write paths must respect them or memories leak
unconditionally into the vector store."

History: the per-turn ``post_mortem`` journal producer once queued items
unconditionally and ``--smart-memory 0.0`` leaked auto-extracted lessons
into the playbook. §4NM removed that producer (and its phase-1 consumer
``_execute_post_mortem``) outright — lessons come only from ``learn_skill``
and the operator's repairs — so this file now pins two things: no
``post_mortem`` journal append exists any more, and the surviving
``smart_memory`` journal appends (streaming + non-streaming) and the
episode record keep the canonical gate.
"""
from pathlib import Path


SRC = Path("src/ghost_agent/core/agent.py")


def test_no_post_mortem_journal_append_exists():
    """§4NM: the producer is gone — neither path appends a ``post_mortem``
    item, and the old journal helper that served it is gone with it."""
    src = SRC.read_text()
    assert "'post_mortem', {'user': last_user_content" not in src
    assert "_tools_for_journal(" not in src
    assert "_execute_post_mortem" not in src


def test_smart_memory_appends_keep_the_canonical_gate():
    """The surviving auto-memory writers (the ``smart_memory`` journal item
    on both reply paths and the episode record) use the SAME gate
    expression, so the user's mental model stays "one knob controls all
    auto-memory writes"."""
    src = SRC.read_text()
    occurrences = src.count("self.context.args.smart_memory > 0.0")
    assert occurrences >= 3, (
        f"Expected at least 3 `smart_memory > 0.0` gate sites "
        f"(2 smart_memory journal appends + the episode record); found {occurrences}"
    )
