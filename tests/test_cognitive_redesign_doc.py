"""Guard: the cognitive-layer redesign rationale must not silently vanish
again (IMPROVEMENTS.md #25).

The original COGNITIVE_LAYER_REDESIGN.md was lost with no VCS; core/agent.py
comments and docs still cite it. This pins that the doc exists and that the
toggle constants it documents match their default-OFF state, so the doc and
code can't drift apart.
"""
from pathlib import Path

import ghost_agent.core.agent as agent_mod


_REPO = Path(__file__).resolve().parents[1]


def test_redesign_doc_exists_and_covers_toggles():
    # The redesign rationale now lives in PROJECT_JOURNAL.md §3 (the unified
    # journal that replaced the separate COGNITIVE_LAYER_REDESIGN.md +
    # BUGHUNT*.md + IMPROVEMENTS.md, 2026-07-07).
    doc = _REPO / "PROJECT_JOURNAL.md"
    assert doc.exists(), "PROJECT_JOURNAL.md was lost — the redesign rationale lives in its §3"
    text = doc.read_text()
    # (_MCTS_TURNSTART_ENABLED is no longer a toggle: the MCTS lookahead was
    # deleted outright in §4NM.)
    # (_SELFHOOD_PREFIX_ENABLED went the same way with the selfhood wake-up
    # prefix, §4NM batch 5.)
    # (_METACOG_ARBITER_ENABLED went with the arbiter in §4PJ; no module
    # toggle is left, but the rationale and its criteria stay documented.)
    assert "_METACOG_ARBITER_ENABLED" in text
    # Re-enable criteria must be present (not just a status dump).
    assert "Re-enable criteri" in text


def test_off_toggles_match_documented_default():
    # The doc describes these as OFF on the request path.
    assert not hasattr(agent_mod, "_SELFHOOD_PREFIX_ENABLED")   # §4NM: deleted
    assert not hasattr(agent_mod, "_METACOG_ARBITER_ENABLED")   # §4PJ: deleted
    # §4NN: the grounded hypothesis loop (--deep-reason) was removed too.
    assert not hasattr(agent_mod, "_HYPOTHESIS_GROUNDING_ENABLED")
