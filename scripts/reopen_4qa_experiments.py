"""§4QA-c: reopen the experiments of project 448d5afa86f0 (operator, 2026-10-18).

Dry run by default: no task, file, row or ledger line changes (opening the
store may still create its folders). ``--apply`` writes — the agent must be STOPPED
(the store's single-writer lock refuses otherwise) and a snapshot taken first:
    scripts/snapshot_stores.py --tag pre-4qa-c

No model was run for these tasks: their results are text the agent wrote
itself (§4QA). Now that ``probe_model`` exists they are re-run for real:
  1. the files written without a model move under ``simulated/`` (same
     relative path), with their artifact rows, manifest entries and research
     index re-pointed — kept, never deleted, and out of the way, so the
     deliverable path is free for the real transcript and no task can close
     on the old text;
  2. the experiment tasks (and the harness run that scored the invented
     text) go back to PENDING, saying what to do;
  3. one ledger line says so; the project is set ACTIVE.
Safe to run again: once applied, metadata ``reopen_4qa_experiments_applied``
is set and steps 2–3 never run again; a move already made is skipped.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PID = "448d5afa86f0"
DATA = Path(os.path.expanduser("~/Data/AI/Data"))
MEM = DATA / "system" / "memory"
SANDBOX = DATA / "sandbox"
MARKER = "reopen_4qa_experiments_applied"
ASIDE = "simulated/"
SIMULATED = [
    "results/layer0-base.md", "results/layer1-reflection.md", "results/layer2-metareflection.md",
    "results/layer3-convergence.md", "results/layer4-plus.md",
    "results/abstract-start.md", "results/emotional-start.md", "results/perceptual-start.md",
    "results/self-referential-start.md",
    "results/control-condition.md", "results/meta-experiment.md", "results/falsification-tests.md",
    "results/cross-model-validation.md", "results/cross-model-validation.json",
    "tools/cross_model_validation.py", "tools/harness_results.json",
]
HOW = ("run its prompts on a real model with probe_model (save_to a NEW results file), then write the "
       "deliverable FROM that transcript; never write a model response yourself")
REOPEN = {
    "5f8437285b82": "Layer 0", "85c7f8e50386": "Layer 1", "19724e2df2f4": "Layer 2",
    "97014c4f3624": "Layer 3", "2a1f392f265f": "Layer 4+",
    "fffda6122c4c": "the abstract start", "a4d78764fdff": "the emotional start",
    "51b78f69e8db": "the perceptual start", "6102d00e031e": "the self-referential start",
    "04a06cd0caae": "the meta-experiment",
    "fef34a33bcd0": "the control condition",
    "8e680ab69b9e": "cross-model validation (node=worker is the second model; temperatures are real "
                    "probe_model temperatures, not labels)",
    "92d11cf64423": "the falsification tests",
    "e6b2888bfee8": "the harness scores (re-run recursive_harness.py on the NEW transcripts once the "
                    "experiment tasks above are done)",
}
LEDGER = ("§4QA-c (2026-10-18): the experiments are reopened to run on a real model with probe_model. "
          "The responses the agent wrote itself moved to simulated/ (kept as illustrations, not data); "
          "analysis/, theory/, report/ and results/meta-reflection.md still rest on them until rewritten "
          "from the new transcripts; re-run the harness only after the experiment tasks are done.")


NOT_DATA = "written without running a model (§4QA) — not data; "


def _move(store, ws: Path, src: str, dst: str, apply: bool, pid: str = "", label: str = NOT_DATA) -> None:
    pid = pid or PID
    s, d = ws / src, ws / dst
    if not s.exists() and d.is_file() and not d.is_symlink():
        # moved by a run that stopped before re-pointing: re-point only
        # (the rows are matched by the OLD path, so this is idempotent)
        print(f"re-point only (already moved): {src} → {dst}")
    elif not s.is_file() or s.is_symlink():
        print(f"skip move (no file): {src}"); return
    elif d.exists():
        print(f"skip move (target exists): {dst}"); return
    else:
        print(f"move {src} → {dst}")
    if not apply:
        return
    if s.is_file():
        d.parent.mkdir(parents=True, exist_ok=True)
        os.replace(s, d)
    with store._connect() as c:
        c.execute("UPDATE task_artifacts SET payload=? WHERE project_id=? AND kind='file' AND payload=?",
                  (dst, pid, src))
    from ghost_agent.core import project_research as PRM

    def _repoint(meta):
        meta[PRM.INDEX_META_KEY] = [
            {**e, "path": dst} if isinstance(e, dict) and e.get("path") == src else e
            for e in meta.get(PRM.INDEX_META_KEY) or []]
        return meta
    store._atomic_metadata_update(pid, _repoint)
    mf = store.get_file_manifest(pid)
    if src in mf:
        store.describe_file(pid, dst, label + (mf[src].get("desc") or ""), mf[src].get("role") or "")

        def _drop(meta):
            m = dict(meta.get("file_manifest") or {})
            m.pop(src, None)
            meta["file_manifest"] = m
            return meta
        store._atomic_metadata_update(pid, _drop)


def main(apply: bool) -> None:
    from ghost_agent.memory.projects import ProjectStore
    if apply:
        from ghost_agent.memory.store_lock import assert_no_other_writer
        assert_no_other_writer(MEM, "reopen_4qa_experiments")
    store = ProjectStore(MEM, sandbox_root=SANDBOX)
    proj = store.get_project(PID)
    if not proj:
        raise SystemExit(f"no project {PID}")
    ws = Path(store.ensure_workspace(PID))
    print(f"project {PID} status={proj['status']} workspace={ws}")
    for rel in SIMULATED:
        _move(store, ws, rel, ASIDE + rel, apply)

    if (proj.get("metadata") or {}).get(MARKER):
        print(f"reopen + ledger: already applied (marker {MARKER}) — skipped")
    else:
        for tid, what in REOPEN.items():
            t = store.get_task(tid)
            if not t or t.get("project_id") != PID:
                print(f"skip reopen (no task): {tid}"); continue
            print(f"reopen {tid} [{t['status']}] {t['description'][:70]}")
            if apply:
                store.update_task(tid, status="PENDING", failure_reason="",
                                  result_summary=f"§4QA-c reopened: no model was run for {what}; {HOW}. "
                                                 f"The old text is under {ASIDE} (not data).")
        print(f"project status {proj['status']} → ACTIVE; ledger += {LEDGER[:80]}…")
        if apply:
            store.update_project(PID, status="ACTIVE")
            if LEDGER[:40] not in (store.get_ledger(PID) or ""):
                store.append_ledger(PID, LEDGER)
            store._atomic_metadata_update(PID, lambda m: {**m, MARKER: "2026-10-18"})
    if apply:
        store.render_project_map(PID)
        print("applied; now:", store.get_project(PID)["status"])
    else:
        print("dry run — nothing written (pass --apply with the agent stopped)")


if __name__ == "__main__":
    main("--apply" in sys.argv[1:])
