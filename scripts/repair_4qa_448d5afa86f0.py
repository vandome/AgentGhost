"""§4QA repair of project 448d5afa86f0 (2026-10-18).

Dry run by default (reads only). ``--apply`` writes — the agent must be STOPPED
(the store's single-writer lock refuses otherwise) and a snapshot taken first:
    scripts/snapshot_stores.py --tag pre-4qa

What it does:
  1. moves the two files written under the pre-project prefix
     ``recursive-thinking-experiment/`` to the paths their tasks declared, and
     re-points their artifact rows and manifest entries;
  2. reopens (PENDING) the five tasks that closed DONE without their work:
     7.2 (3-byte deliverable), 6.2 (declared file never written), and the three
     edit/design tasks closed on web summaries (lexicon → embeddings, the
     human-subject protocol, the final-report update);
  3. sets the project ACTIVE again and writes one ledger line saying which
     results files were written without running a model;
  4. for EVERY project: records the hash of each research brief written
     before brief hashes existed (``backfill_legacy_brief_shas``), so an
     unchanged old brief stays the writer's own and any later edit makes it
     someone's work.
Nothing is deleted. Safe to run again: once applied, metadata
``repair_4qa_applied`` is set and steps 2–3 never run again (a task closed
since would otherwise be reopened, and a finished project re-activated); a
move already made and a brief already hashed are skipped.
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
MOVES = {
    "recursive-thinking-experiment/research/recursive-self-reference.md": "research/recursive-self-reference.md",
    "recursive-thinking-experiment/results/falsification-tests.md": "results/falsification-tests.md",
}
REOPEN = {
    "fa70d8ce6590": "its deliverable results/meta-reflection.md holds 3 bytes ('...')",
    "5627d9de486d": "the declared theory/comparison-with-existing-frameworks.md was never written "
                    "(a web-search summary went to research/task-6-2-…md)",
    "43c19f8b7511": "closed on a web-search summary; tools/recursive_harness.py still uses lexicon metrics",
    "16459f2ff912": "closed on a web-search summary about 'humans'; no journal format was designed",
    "3a9e5b194cbd": "closed on a web-search summary; report/final-report.md is unchanged since 10:24 and "
                    "still says the hypothesis is confirmed",
}
LEDGER = ("§4QA (2026-10-18): no model was run for the experiments — results/layer0-4*.md, "
          "results/*-start.md, results/control-condition.md, results/meta-experiment.md and "
          "results/falsification-tests.md hold responses the agent wrote itself, and "
          "tools/cross_model_validation.py hard-codes simulated responses (its 'temperatures' are "
          "labels). Treat them as illustrations, not data; re-run them with probe_model.")


def main(apply: bool) -> None:
    from ghost_agent.memory.projects import ProjectStore
    if apply:
        from ghost_agent.memory.store_lock import assert_no_other_writer
        assert_no_other_writer(MEM, "repair_4qa")
    store = ProjectStore(MEM, sandbox_root=SANDBOX)
    proj = store.get_project(PID)
    if not proj:
        raise SystemExit(f"no project {PID}")
    ws = Path(store.ensure_workspace(PID))
    print(f"project {PID} status={proj['status']} workspace={ws}")

    for src, dst in MOVES.items():
        s, d = ws / src, ws / dst
        if not s.is_file() or s.is_symlink():
            print(f"skip move (no file): {src}"); continue
        if d.exists():
            print(f"skip move (target exists): {dst}"); continue
        print(f"move {src} → {dst}")
        if apply:
            d.parent.mkdir(parents=True, exist_ok=True)
            os.replace(s, d)
            with store._connect() as c:
                c.execute("UPDATE task_artifacts SET payload=? WHERE project_id=? AND kind='file' AND payload=?",
                          (dst, PID, src))
            from ghost_agent.core import project_research as PRM

            def _repoint(meta, _s=src, _d=dst):
                # the index named the old path (r8 m3): the briefing listed a file that was gone
                meta[PRM.INDEX_META_KEY] = [
                    {**e, "path": _d} if isinstance(e, dict) and e.get("path") == _s else e
                    for e in meta.get(PRM.INDEX_META_KEY) or []]
                return meta
            store._atomic_metadata_update(PID, _repoint)
            mf = store.get_file_manifest(PID)
            if src in mf:
                store.describe_file(PID, dst, mf[src].get("desc") or "moved from " + src, mf[src].get("role") or "")

                def _drop(meta, _k=src):
                    m = dict(meta.get("file_manifest") or {})
                    m.pop(_k, None)
                    meta["file_manifest"] = m
                    return meta
                store._atomic_metadata_update(PID, _drop)
    nested = ws / "recursive-thinking-experiment"
    for sub in (nested / "research", nested / "results", nested):
        if sub.is_dir() and not any(sub.iterdir()):
            print(f"rmdir empty {sub.relative_to(ws)}")
            if apply:
                sub.rmdir()

    # applied before = the marker, or the ledger line (the first --apply ran
    # before the marker existed; since then the autopilot closed reopened
    # tasks again — r10 MAJOR-3: a re-run would have reopened them)
    marked = bool((proj.get("metadata") or {}).get("repair_4qa_applied"))
    done_before = marked or LEDGER[:40] in (store.get_ledger(PID) or "")
    if apply and done_before and not marked:
        store._atomic_metadata_update(PID, lambda m: {**m, "repair_4qa_applied": "2026-10-18"})
    if done_before:
        print("reopen + status + ledger: already applied ("
              + ("marker repair_4qa_applied" if marked else "the ledger line is there; marker set on --apply")
              + ") — skipped")
    for tid, why in ({} if done_before else REOPEN).items():
        t = store.get_task(tid)
        if not t:
            print(f"skip reopen (no task): {tid}"); continue
        if str(t.get("result_summary") or "").startswith("§4QA reopened"):
            print(f"skip reopen (already reopened): {tid} [{t['status']}]"); continue
        print(f"reopen {tid} [{t['status']}] {t['description'][:70]} — {why}")
        if apply:
            store.update_task(tid, status="PENDING", failure_reason="",
                              result_summary=f"§4QA reopened: {why}")
    if not done_before:
        print(f"project status {proj['status']} → ACTIVE; ledger += {LEDGER[:80]}…")
    if apply and not done_before:
        store.update_project(PID, status="ACTIVE")
        if LEDGER[:40] not in store.get_ledger(PID):
            store.append_ledger(PID, LEDGER)
        store._atomic_metadata_update(PID, lambda m: {**m, "repair_4qa_applied": "2026-10-18"})
        store.render_project_map(PID)
        print("applied; now:", store.get_project(PID)["status"])
    from ghost_agent.core import project_research as PRM
    for p in store.list_projects():
        got = PRM.backfill_legacy_brief_shas(store, p["id"], dry_run=not apply)
        if got:
            print(f"brief hashes {p['id']}: {'recorded' if apply else 'candidates'} {got}")
    if not apply:
        print("dry run — nothing written (pass --apply with the agent stopped)")


if __name__ == "__main__":
    main("--apply" in sys.argv[1:])
