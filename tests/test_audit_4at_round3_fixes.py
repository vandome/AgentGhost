"""§4AT round-3 fixes across five subsystems (2026-08-11).

Each pin names the live defect it closes. All were found by the audit's
pre-registered questions and verified against the real code before fixing.
(The section-H pins on the post-mortem engine went with the engine, §4NM.)
"""
from __future__ import annotations

import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../src")))

from ghost_agent.core import dream as DREAM              # noqa: E402
from ghost_agent.core import project_advancer as PA      # noqa: E402
from ghost_agent.core.coding_executor import (           # noqa: E402
    _replace_failure_kind,
)
from ghost_agent.tools import registry as REG            # noqa: E402


# ── F: the retry-steering mislabel (req 6e9efd6a's 20-minute grind) ──────

def test_a_syntax_rollback_is_not_reported_as_a_missing_anchor():
    """THE LIVE DEFECT. Every non-SUCCESS replace was labelled "anchor not
    found" and fed to the next attempt behind "YOUR PREVIOUS ATTEMPT FAILED:".
    A syntax-regression rollback therefore sent the model hunting for a better
    ANCHOR while the defect was its replacement's INDENTATION — 14 attempts,
    20 minutes, 5 of them rejected by the same guard."""
    out = ("REJECTED: that replace would introduce a syntax error and was NOT "
           "applied — 'x.py' is unchanged on disk: unexpected indent (line 238).")
    kind = _replace_failure_kind(out)
    assert "syntax" in kind.lower()
    assert "anchor not found" not in kind.lower()
    assert "replacement" in kind.lower(), "must point at the REPLACE block"


def test_each_distinct_rejection_gets_its_own_label():
    assert "ambiguous" in _replace_failure_kind(
        "SYSTEM INSTRUCTION: Multiple instances of this text block found").lower()
    # ⚠ THE REAL STRING, copied from file_system.py:1671. The first version
    # of this test invented "None of the blocks matched the file contents",
    # which appears nowhere in the tree — so it validated the classifier
    # against a string production never emits, and the genuine multi-block
    # failure fell through to the neutral branch unnoticed.
    assert _replace_failure_kind(
        "SYSTEM INSTRUCTION: None of the SEARCH/REPLACE blocks matched in "
        "'x.py'.\nCould not find block:\ndef foo()..."
    ) == "anchor not found"
    # …and a MISSING FILE is not a missing anchor (file_system.py:1362)
    assert "file" in _replace_failure_kind("Error: 'utils.py' not found.").lower()
    assert "anchor" not in _replace_failure_kind("Error: 'utils.py' not found.").lower()
    assert "marker" in _replace_failure_kind(
        "REJECTED: replacement contains SEARCH/REPLACE markers (====)").lower()
    assert "block" in _replace_failure_kind(
        "SYSTEM BLOCK: released project is immutable; this write was NOT applied.").lower()


def test_an_UNRECOGNISED_rejection_stays_neutral():
    """⚠ The whole defect was a confident wrong guess. An unknown shape must
    NOT fall back to the most common label — the caller appends the tool's raw
    text anyway, so saying less is strictly better than saying something
    false."""
    kind = _replace_failure_kind("some novel failure nobody has seen")
    assert "anchor not found" not in kind.lower()
    assert "syntax" not in kind.lower()


def test_the_call_site_actually_uses_the_classifier():
    """⚠ SEAM. The helper can be perfect while the caller still hardcodes the
    old string."""
    src = inspect.getsource(
        sys.modules["ghost_agent.core.coding_executor"])
    assert "_replace_failure_kind(out)" in src
    assert "did not apply (anchor not found)" not in src


# ── G: simulation must not author real selfhood ─────────────────────────


# ── C: a string constraint must not shred into characters ───────────────

def test_project_advancer_uses_the_SHARED_constraint_normaliser():
    """Metadata is model-written JSON, so `{"constraints": "no external
    APIs"}` is legal — and iterating it raw yields 17 single-character
    "constraints" fed to the constraint gate. `_constraint_list` exists
    because this shredding already destroyed a record once (2026-08-01); this
    call site kept its own copy and so kept the bug."""
    src = inspect.getsource(PA)
    assert "_constraint_list(" in src, "shared normaliser not used"
    assert "[str(c) for c in\n                        ((proj.get" not in src

    from ghost_agent.memory.projects import _constraint_list
    assert _constraint_list("no external APIs") == ["no external APIs"]
    assert len(_constraint_list("no external APIs")) == 1, "shredded again"
    assert _constraint_list(["a", "b"]) == ["a", "b"]
    assert _constraint_list(None) == []


# ── E: the live A/B must differ by exactly the treatment ────────────────

# ── B: self-play attribution, re-landed after the data repair ───────────









# ── F: a dropped edit must not read as "nothing to do" ──────────────────

@pytest.mark.asyncio
async def test_edits_for_an_UNSNAPSHOTTED_file_return_a_REASON_not_silence():
    """⚠ THE CRITICAL. `edits` are applied only when `path in snap`, and the
    prompt snapshot is hard-capped at 12 files. On a larger project an `edits`
    entry for an unsnapshotted file fell through to the `content` branch,
    `content` is None, and the function returned (None, None) — "skipped, not
    a failure": zero tool calls, no reason, nothing in `written`. A sibling
    entry that DID apply then made the whole spec look successful, so the leaf
    closed **DONE claiming work that never happened**.

    §4QA-d: the file is now read from disk first (see
    tests/test_4qa_unsnapshotted_edits.py); a file that cannot be read, or
    edits that cannot apply, still return a REASON — never silence.
    """
    from ghost_agent.core import coding_executor as CE

    calls = []

    async def _runner(tool, args):
        calls.append((tool, args))
        return "SUCCESS"            # a read without CONTENTS: unreadable

    written, reason = await CE._apply_file(
        _runner,
        {"path": "utils.js", "edits": [{"search": "a", "replace": "b"}]},
        {"index.html": "<html>"},               # utils.js NOT snapshotted
        touched=set(), fresh=set())
    assert written is None
    assert reason, "a dropped edit returned no reason — silently 'nothing to do'"
    assert "utils.js" in reason
    # §4QA-d r21: an unreadable reply is probed (ranged read) and the edits
    # tried without a copy — none is usable here, so nothing may be written
    assert {a.get("operation") for _t, a in calls} == {"read"}, "only reads may run"


@pytest.mark.asyncio
async def test_a_rewrite_is_never_judged_against_a_TRUNCATED_snapshot():
    """⚠ THE SECOND CRITICAL. `_gather_project_files` keeps only a prefix once
    its 400 KB budget is nearly spent, and that truncation was SILENT — so a
    300 KB index.html snapshotted as 4 KB made a 20 KB full rewrite look like
    GROWTH (20000 > 4000×0.85). The non-regression guard waved through an
    overwrite that destroyed 280 KB of working code. For `.py` the prefix
    usually fails `ast.parse`, disabling the overwrite guard outright.

    A marked entry now means "unknown baseline" → re-read from disk. If the
    re-read fails, REFUSE: a refusal costs one retry, a wrong pass costs the
    file."""
    from ghost_agent.core import coding_executor as CE
    from ghost_agent.core.project_advancer import SNAPSHOT_TRUNCATED_MARK

    big = "x" * 300_000
    truncated_snap = big[:4000] + SNAPSHOT_TRUNCATED_MARK
    reads = []

    async def _runner(tool, args):
        if args.get("operation") in ("read", "read_chunked"):
            reads.append(args.get("path"))
            return "CONTENTS OF index.html:\n" + big
        return "SUCCESS"

    written, reason = await CE._apply_file(
        _runner,
        {"path": "index.html", "content": "<html>small rewrite</html>"},
        {"index.html": truncated_snap},
        touched=set(), fresh=set())
    assert written is None, "a 20KB rewrite of a 300KB file was allowed"
    assert reason, "no reason given for refusing the overwrite"
    assert reads, "the truncated snapshot was trusted instead of re-read"


def test_the_truncation_marker_has_exactly_ONE_definition():
    """Producer and consumer drifting apart is the defect class this file
    already carries scars for."""
    from ghost_agent.core.coding_executor import _snap_truncated_mark
    from ghost_agent.core.project_advancer import SNAPSHOT_TRUNCATED_MARK
    assert _snap_truncated_mark() == SNAPSHOT_TRUNCATED_MARK
    src = inspect.getsource(sys.modules["ghost_agent.core.coding_executor"])
    assert "SNAPSHOT TRUNCATED" not in src, (
        "the marker literal was copied instead of imported")


def test_the_PRODUCER_actually_marks_a_truncated_snapshot_entry():
    """⚠ BOTH SIDES OR NEITHER. Mutation-testing showed the consumer guard was
    pinned but the PRODUCER was not: removing the marker from
    `_gather_project_files` reddened nothing, so the guard would silently
    never fire again. A one-sided pin on a producer/consumer contract is the
    same drift it exists to prevent."""
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace
    from ghost_agent.core.project_advancer import (
        _gather_project_files, SNAPSHOT_TRUNCATED_MARK)

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "projects" / "pid"
        root.mkdir(parents=True)
        # Exceed the char budget so truncation is forced.
        (root / "big.html").write_text("<html>" + "y" * 600_000 + "</html>")
        (root / "small.py").write_text("print(1)\n")
        store = SimpleNamespace(sandbox_root=str(Path(d)))
        snap = _gather_project_files(store, "pid")
        assert snap, "snapshot came back empty — the probe is broken"
        marked = [k for k, v in snap.items() if SNAPSHOT_TRUNCATED_MARK in v]
        assert marked, (
            "a truncated entry carries no marker — the consumer's guard can "
            f"never fire. entries: { {k: len(v) for k, v in snap.items()} }")
        # …and a file that fits is NOT marked (no false positives).
        assert SNAPSHOT_TRUNCATED_MARK not in snap.get("small.py", "")


# ── C: a crash must not downgrade to a weaker closer ────────────────────

def test_a_coding_executor_CRASH_leaves_the_leaf_open():
    """⚠ THE THIRD MEMBER OF THE FAMILY. A transient exception inside
    `coding_executor` left `cres = None` and fell through to the generic
    single-shell-command path, which marks the leaf DONE on ANY output that is
    not `ERROR:`-prefixed and not a non-zero exit — verify gate, smoke gate,
    files-written check and constraint gate all bypassed. `echo`-shaped output
    closed a build task.

    Same shape as the two coding-executor CRITICALs fixed the same day: a task
    reaching DONE on evidence that no work happened. A crash is a reason to
    STOP, not to accept a lower standard of proof."""
    src = inspect.getsource(PA)
    i = src.find('logger.warning("coding_executor crashed')
    assert i > 0, "the crash handler moved — re-anchor this pin"
    after = src[i:i + 1600]          # §4PG r3: the deferral branch sits between the log line and the crash result
    assert "AdvanceResult(" in after and '"blocked"' in after, (
        "a coding-executor crash still falls through to the weaker "
        "single-command path that can close the leaf DONE")
    # and it must NOT simply swallow and continue
    assert "return" in after.split("AdvanceResult(")[0] or "AdvanceResult(" in after


def test_a_frontier_run_records_WHICH_TEMPLATE_produced_it():
    """⚠ AUDITABILITY, not correctness. A run recorded only the cluster it was
    FILED under, never the template that produced it — so when attribution
    turned out to be 51% wrong, reconstructing what belonged where required
    re-rendering the whole template bank at four tiers and matching challenge
    hashes. The record has to say."""
    import tempfile
    from pathlib import Path
    from ghost_agent.memory.frontier import FrontierTracker

    with tempfile.TemporaryDirectory() as d:
        f = FrontierTracker(Path(d))          # takes a memory DIR
        f.record_run("bash", "count lines in logs", 1, True, 40,
                     template_key="bash")
        f.record_run("sql", "an LLM-generated challenge", 1, True, 40)
        import json, glob
        files = glob.glob(str(Path(d) / "*frontier*.json"))
        assert files, f"no frontier file written: {list(Path(d).iterdir())}"
        runs = json.load(open(files[0]))["runs"]
        assert len(runs) == 2
        assert runs[0].get("template_key") == "bash", (
            "the template that produced the run is not recorded")
        # …and an untemplated challenge records EMPTY, which is itself the
        # distinction the cluster field cannot carry.
        assert runs[1].get("template_key") == ""
