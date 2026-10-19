"""§4QA-e: reopen the 11 experiment tasks of project 448d5afa86f0 (and 7.2) for a full-
length rerun (operator, 2026-10-19: "do 1 and 2").

The §4QA-c experiments were run for real with probe_model, but at its old
default of 600 tokens every answer was cut (11 of 11 runs), and in the
chained cascades each later layer reflected on a cut answer. Task 4.4 also
reused the main cascade's prompts instead of its own starting point. This
script:
  1. moves that run's transcripts and write-ups to ``results/run1-600tok/``
     with their rows re-pointed (the same move as reopen_4qa_experiments.py:
     kept, never deleted);
  2. keeps each moved transcript's bytes for the §4QA-e quote check under
     its new path (r29: run 1 predates the check, so a quote of it was judged
     against the rerun's transcript of the same name);
  3. sets the 11 tasks — and 7.2, which reflected on 7.1 — PENDING again,
     saying why (4.4 with its own starting point), with their dependency
     edges kept (the analysis, theory and report tasks still wait for them);
     the control task declares the file it writes (→ results/control-
     condition.md) — without one, the §4QA-e quote check never read the
     write-up that held 5 of the 8 reworded quotes;
  4. writes one ledger line.
Dry run by default; ``--apply`` needs the agent STOPPED and a snapshot first:
    scripts/snapshot_stores.py --tag pre-4qa-e
Safe to run again: metadata ``rerun_4qa_experiments_applied`` stops every step
(r30: the moves too — a second apply moved the rerun's own files).
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("reopen_4qa_experiments",
                                               ROOT / "scripts" / "reopen_4qa_experiments.py")
EXP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EXP)

PID = EXP.PID
MEM, SANDBOX = EXP.MEM, EXP.SANDBOX
MARKER = "rerun_4qa_experiments_applied"
RUN1 = "results/run1-600tok/"
RUN1_FILES = [
    "results/layer-cascade-transcript.md", "results/layer0-base.md", "results/layer1-reflection.md",
    "results/layer2-metareflection.md", "results/layer3-convergence.md", "results/layer4-plus.md",
    "results/abstract-start-transcript.md", "results/abstract-start.md",
    "results/emotional-start-transcript.md", "results/emotional-start.md",
    "results/perceptual-start-transcript.md", "results/perceptual-start.md",
    "results/self-referential-start-transcript.md", "results/self-referential-start.md",
    "results/control-transcript.md", "results/control-condition.md",
    "results/meta-experiment-transcript.md", "results/meta-experiment.md",
    "results/meta-reflection.md",
    # r29: the harness scored the cut run
    "results/harness_9_results.json",
]
#: the system's own first lines of a probe_model transcript
TRANSCRIPT_HEAD = "# Model probe transcript\n\n_Recorded by `probe_model`: the system wrote this file."
#: r28: the first run WAS real model output — cut short, not invented
RUN1_LABEL = "first real run, every answer cut at max_tokens=600 (§4QA-e) — superseded by the rerun; "
TASKS = ["5f8437285b82", "85c7f8e50386", "19724e2df2f4", "97014c4f3624", "2a1f392f265f",
         "fffda6122c4c", "a4d78764fdff", "51b78f69e8db", "6102d00e031e", "fef34a33bcd0", "04a06cd0caae",
         # r28: 7.2 reflected on the cut 7.1 run; it waits on 7.1 already
         "fa70d8ce6590"]
#: task → the file it writes, for a task that declares none
DECLARE = {"fef34a33bcd0": "results/control-condition.md"}
WHY = ("§4QA-e rerun: the first real run (kept in results/run1-600tok/) cut every answer at "
       "max_tokens=600, so the chained layers reflected on cut answers; run it again at the new "
       "default (1500) and quote the transcript word for word")
#: r29: 4.4's own starting point — run 1 put the main cascade's paradox prompts to it
WHY_BY_TASK = {"6102d00e031e": WHY + "; START from a self-referential prompt of its own (the model asked "
               "to think about its own thinking right now), NOT the paradox-of-self-reference prompts of the "
               "main cascade, which run 1 reused"}
LEDGER = ("§4QA-e (2026-10-19): the first real experiment run cut every answer at 600 tokens; "
          "its files moved to results/run1-600tok/ and 12 tasks (the 11 experiments and 7.2) "
          "reopened for a full-length rerun.")


def main(apply: bool) -> None:
    from ghost_agent.memory.projects import ProjectStore
    from ghost_agent.core.task_deliverable import declared_deliverable
    if apply:
        from ghost_agent.memory.store_lock import assert_no_other_writer
        assert_no_other_writer(MEM, "rerun_4qa_experiments")
    store = ProjectStore(MEM, sandbox_root=SANDBOX)
    proj = store.get_project(PID)
    if not proj:
        raise SystemExit(f"no project {PID}")
    ws = Path(store.ensure_workspace(PID))
    print(f"project {PID} status={proj['status']} workspace={ws}")
    from ghost_agent.core.probe_transcripts import record
    done = bool((proj.get("metadata") or {}).get(MARKER))
    # r30: once applied, a file at a run-1 path is the RERUN's — never moved
    for rel in ([] if done else RUN1_FILES):
        dst = RUN1 + rel[len("results/"):]
        EXP._move(store, ws, rel, dst, apply, pid=PID, label=RUN1_LABEL)
        if apply and rel.endswith("-transcript.md") and (ws / dst).is_file():
            text = (ws / dst).read_text(encoding="utf-8")
            if text.startswith(TRANSCRIPT_HEAD):
                print(f"keep for the quote check: {dst}")
                record(MEM, PID, dst, text)

    if done:
        print(f"moves, reopen + ledger: already applied (marker {MARKER}) — skipped")
    else:
        for tid in TASKS:
            t = store.get_task(tid)
            if not t or t.get("project_id") != PID:
                print(f"skip reopen (no task): {tid}"); continue
            print(f"reopen {tid} [{t['status']}]: {t['description'][:70]}")
            fields = dict(status="PENDING", failure_reason="", result_summary=WHY_BY_TASK.get(tid, WHY) + ".")
            if tid in DECLARE and not declared_deliverable(t["description"]):
                fields["description"] = f"{t['description'].rstrip()} → {DECLARE[tid]}"
                print(f"  declares → {DECLARE[tid]}")
            if apply:
                store.update_task(tid, **fields)
        print(f"project status {proj['status']} → ACTIVE; ledger += {LEDGER[:80]}…")
        if apply:
            store.update_project(PID, status="ACTIVE")
            if LEDGER[:40] not in (store.get_ledger(PID) or ""):
                store.append_ledger(PID, LEDGER)
            store._atomic_metadata_update(PID, lambda m: {**m, MARKER: "2026-10-19"})
    if apply:
        store.render_project_map(PID)
        print("applied; now:", store.get_project(PID)["status"])
    else:
        print("dry run — nothing written (pass --apply with the agent stopped)")


if __name__ == "__main__":
    main("--apply" in sys.argv[1:])
