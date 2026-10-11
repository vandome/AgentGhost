"""Regression tests for the High-tier audit fixes:

* dream._VALID_LESSON_DOMAINS / _generalization_guard: §4NM removed them with
  synthetic self-play; those tests are gone.
* graph.add_triplets: re-asserting a previously-expired functional fact
  did not reset valid_until → the fact stayed invisible after restart.

(The mcts._simulate_parallel and prm.labels._build_state_for_step fixes
this file also pinned were removed with those modules in batch 3, §4NM.)
"""

import sqlite3


# -----------------------------------------------------------------
# graph: re-asserting an expired functional fact revives it
# -----------------------------------------------------------------

def test_reasserted_functional_fact_is_revived(tmp_path):
    from ghost_agent.memory.graph import GraphMemory
    gm = GraphMemory(tmp_path)

    gm.add_triplets([{"subject": "bob", "predicate": "WORKS_AT", "object": "acme"}])
    # A different object for the functional predicate expires acme.
    gm.add_triplets([{"subject": "bob", "predicate": "WORKS_AT", "object": "google"}])
    # Re-assert acme — must REVIVE it (valid_until back to NULL).
    gm.add_triplets([{"subject": "bob", "predicate": "WORKS_AT", "object": "acme"}])

    with sqlite3.connect(gm.db_path) as conn:
        acme_vu = conn.execute(
            "SELECT valid_until FROM triplets WHERE subject=? AND predicate=? AND object=?",
            ("bob", "WORKS_AT", "acme"),
        ).fetchone()[0]
        google_vu = conn.execute(
            "SELECT valid_until FROM triplets WHERE subject=? AND predicate=? AND object=?",
            ("bob", "WORKS_AT", "google"),
        ).fetchone()[0]

    assert acme_vu is None, "re-asserted fact stayed expired (valid_until not reset)"
    assert google_vu is not None, "the superseded object should be expired"

    # And it must be visible to a fresh load (initialize_graph filters valid_until IS NULL).
    gm2 = GraphMemory(tmp_path)
    assert gm2.nx_graph.has_edge("bob", "acme")
