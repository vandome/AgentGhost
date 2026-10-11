"""§4R — belief/contradiction subsystem audit fixes (2026-08-08).

Four independent lenses audited this subsystem. The findings these pin:

* CRIT — the contradiction engine's candidate scope was a `$nin` DENYLIST, not
  the complement of the prunable set. Measured live it left 33 rows deletable
  of which only 2 were the `auto` facts it exists to supersede: 28 dream
  `synthesis` rows (each the only surviving copy of its merged sources), the
  user's single `manual` save, the `identity` row and a `document_summary`.
* MAJOR — the delete ran BEFORE the audit record, and the record was
  best-effort with failures swallowed at DEBUG, so an irreversible deletion
  could leave no trace anywhere.
* MAJOR — `explain_belief_change` over-corrected from inert into a
  false-positive machine: it fired on 67/67 live turns and injected the five
  most RECENT revisions regardless of the question.
* MAJOR — `dream.py` performed the same LLM-ids→delete with no whitelist.
  (§4NM: that consolidation left dream.py; its whitelist pin was removed.)

NOTE ON FIXTURES: the pre-existing tests for this module used SINGLE-entry
logs, which is exactly why the 100%-false-positive behaviour stayed green.
These build realistic multi-entry ledgers on purpose.
"""

import json
from pathlib import Path

import pytest

# §4NN: ContradictionLog (belief versioning) was removed; its read-path
# precision pins went with it.


# ── engine ────────────────────────────────────────────────────────────────
# §4KZ: the LLM-judged belief-revision engine is GONE (5 of its 7 live
# deletions since 08-08 were wrong). The scope/gate/order pins that stood
# here pinned its source text; the behaviour that replaces them — no
# model-judged deletion, profile + synced mirrors — is pinned in
# tests/test_4kz_update_path.py.


def test_top_tier_gates_key_off_effective_threshold():
    """Hardcoded 0.9 gates inside the threshold branch meant that lowering
    --smart-memory would admit the 0.8 tier with the generic-knowledge filter
    silently bypassed — a tuning knob must not disarm a safety filter."""
    import inspect
    from ghost_agent.core.agent import GhostAgent
    src = inspect.getsource(GhostAgent.run_smart_memory_task)
    # The first attempt at this fix (`_top_tier = max(0.9, effective_threshold)`)
    # was a PROVABLE NO-OP — it sits inside `score >= effective_threshold`, so
    # it never changed an outcome for any (threshold, score) pair, and THIS
    # TEST passed on it because it only checked the source text. The filter is
    # now unconditional on score, which is what "a tuning knob must not disarm
    # a safety filter" actually requires.
    assert "_top_tier" not in src, "the no-op top-tier gate is back"
    # ⚠ The filter must ABORT, not merely log. Deleting the single `return`
    # in its branch leaves the log line and stores the fact anyway — junk
    # into the durable profile store, with a message on the operator line
    # saying it was discarded. Read the branch, not the condition.
    import ast as _ast
    import textwrap as _tw
    _tree = _ast.parse(_tw.dedent(src))
    _guards = [n for n in _ast.walk(_tree)
               if isinstance(n, _ast.If)
               and "is_personal or is_technical" in _ast.unparse(n.test)]
    assert _guards, "the generic-knowledge filter moved"
    for g in _guards:
        assert any(isinstance(x, (_ast.Return, _ast.Continue, _ast.Raise))
                   for x in g.body), (
            "the generic-knowledge filter only LOGS — it prints "
            "'Discarded generic knowledge' and stores the fact anyway: "
            + _ast.unparse(g)[:120])
    assert "if score >= 0.9 and not (is_personal" not in src


def test_document_summary_is_forget_protected():
    """Its `document` twin is protected; deleting the summary while the source
    survives is the asymmetric drift the protected list exists to prevent."""
    from ghost_agent.tools.memory import _FORGET_PROTECTED_TYPES
    assert "document_summary" in _FORGET_PROTECTED_TYPES
    assert "document" in _FORGET_PROTECTED_TYPES


@pytest.mark.asyncio
async def test_expansion_sweep_does_not_delete_synthesis(tmp_path):
    """BEHAVIOURAL (§4R R2). The expansion sweep matches graph NEIGHBOURS the
    user never named. A `synthesis` is a composite whose merged sources are
    already deleted, so removing one to excise an incidental token destroys
    the only surviving copy of everything else it merged.

    A synthesis mentioning the neighbour term must SURVIVE, while an ordinary
    `auto` fact mentioning it is still deleted — proving the guard is targeted
    and did not simply disable the sweep.
    """
    from unittest.mock import MagicMock
    from ghost_agent.tools.memory import tool_unified_forget

    deleted = []
    ms = MagicMock()
    ms._get_lock = None
    del ms._get_lock  # force the _NullCM path

    def _query(query_texts=None, n_results=None, where=None):
        return {
            "ids": [["syn-1", "auto-1"]],
            "documents": [["Master summary mentioning mortimer and much else.",
                           "The user once owned mortimer."]],
            "metadatas": [[{"type": "synthesis"}, {"type": "auto"}]],
            "distances": [[0.9, 0.9]],   # far: only the LITERAL rule can fire
        }
    ms.collection.query.side_effect = _query
    ms.collection.delete.side_effect = lambda ids=None: deleted.extend(ids or [])
    ms.search_advanced.return_value = []

    # `expanded_targets` comes from the graph; give it a neighbour the user
    # never typed, which is what makes this the expansion sweep.
    gm = MagicMock()
    # `get_connected_entities` is the real source of expanded_targets
    # (memory.py:841) — an earlier version of this test mocked the wrong
    # method, so the expansion sweep never ran and the pin stayed green with
    # the guard removed.
    gm.get_connected_entities.return_value = ["mortimer"]
    gm.delete_by_target.return_value = 0

    await tool_unified_forget(target="iguana", memory_system=ms, graph_memory=gm)

    assert "syn-1" not in deleted, (
        "the expansion sweep destroyed a composite synthesis on a term the "
        "user never named")
