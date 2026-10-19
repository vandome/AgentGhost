"""§4QA-d: reopen the work of project 448d5afa86f0 that rests on the invented
experiments (operator, 2026-10-19: "fix all open items").

§4QA-c reopened the experiments; the analysis, theory, report and
meta-reflection written FROM their invented text stayed DONE, and the
autopilot ran the harness task before any experiment existed (it had no
dependency, only prose saying "once the experiments are done"). This script:
  1. moves those files (and tools/embedding_scores.json, scores with no run
     behind them) under ``simulated/`` with their rows re-pointed, the same
     way as scripts/reopen_4qa_experiments.py — kept, never deleted;
  2. sets the tasks PENDING with ``depends_on`` edges, so they are picked
     only in order: experiments → harness → analysis → theory → report;
  3. reopens 43c19f8b7511 (failed on the 12-file snapshot, fixed in §4QA-d);
  4. writes one ledger line.
Dry run by default; ``--apply`` needs the agent STOPPED and a snapshot first:
    scripts/snapshot_stores.py --tag pre-4qa-d
Safe to run again: metadata ``reopen_4qa_downstream_applied`` stops steps 2–4.
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
MEM, SANDBOX, ASIDE = EXP.MEM, EXP.SANDBOX, EXP.ASIDE
MARKER = "reopen_4qa_downstream_applied"
SIMULATED = [
    "analysis/cross-layer-comparison.md", "analysis/convergence-signature.md",
    "analysis/three-layer-hypothesis.md", "analysis/cascade-map.md",
    "theory/recursive-self-awareness-theory.md", "theory/comparison-with-existing-frameworks.md",
    "theory/testable-predictions.md", "report/final-report.md", "results/meta-reflection.md",
    "tools/embedding_scores.json",
]
LAYERS = ["5f8437285b82", "85c7f8e50386", "19724e2df2f4", "97014c4f3624", "2a1f392f265f",
          "fffda6122c4c", "a4d78764fdff", "51b78f69e8db", "6102d00e031e"]
HARNESS, CONTROL, META_EXP = "e6b2888bfee8", "fef34a33bcd0", "04a06cd0caae"
CROSS_MODEL, FALSIFICATION = "8e680ab69b9e", "92d11cf64423"
ANALYSIS = ["85895898d3fc", "57838625c368", "cd71caf700c9", "729320a6873b"]
THEORY = ["d94038512187", "5627d9de486d", "79b5f0c06175"]
REPORT, REPORT_DATA, META_REFL, EMBED = "5b88afd5f065", "3a9e5b194cbd", "fa70d8ce6590", "43c19f8b7511"
# task → (what it waits for, why it is reopened)
REOPEN = {
    # review r19: the harness also scores the control, with the embedding metrics
    HARNESS: (LAYERS + [CONTROL, EMBED], "the harness ran before any experiment had a transcript; "
                                        "score the NEW results files (and the control) once the "
                                        "experiment tasks and the embedding metrics are DONE"),
    **{t: ([HARNESS, CONTROL] + LAYERS, "written from the invented responses (old text under "
                                        "simulated/); rewrite it from the new transcripts and "
                                        "harness scores") for t in ANALYSIS},
    # review r19: every §4QA-c experiment must reach the report — cross-model
    # validation and the falsification tests had nothing waiting on them
    **{t: (ANALYSIS + [CROSS_MODEL, FALSIFICATION],
           "built on the invented analysis (old text under simulated/); rewrite it "
           "from the new analysis") for t in THEORY},
    REPORT: (THEORY + ANALYSIS + [CROSS_MODEL, FALSIFICATION, META_EXP, META_REFL],
             "built on the invented data (old text under simulated/); "
             "rewrite it from the new analysis and theory"),
    REPORT_DATA: ([REPORT, HARNESS], "add the real harness numbers to the rewritten report"),
    META_REFL: ([META_EXP], "reflected on an invented meta-experiment (old text under "
                           "simulated/); rewrite it from the real one"),
    EMBED: ([], "failed because the build saw only 12 project files (fixed in §4QA-d); "
                "tools/embedding_scoring.py exists — wire it into the harness"),
}
LEDGER = ("§4QA-d (2026-10-19): analysis/, theory/, report/, results/meta-reflection.md and "
          "tools/embedding_scores.json rested on the invented experiments: moved to simulated/ "
          "and their tasks reopened, in order (experiments → harness → analysis → theory → "
          "report) by task dependencies.")


def main(apply: bool) -> None:
    from ghost_agent.memory.projects import ProjectStore
    if apply:
        from ghost_agent.memory.store_lock import assert_no_other_writer
        assert_no_other_writer(MEM, "reopen_4qa_downstream")
    store = ProjectStore(MEM, sandbox_root=SANDBOX)
    proj = store.get_project(PID)
    if not proj:
        raise SystemExit(f"no project {PID}")
    ws = Path(store.ensure_workspace(PID))
    print(f"project {PID} status={proj['status']} workspace={ws}")
    for rel in SIMULATED:
        EXP._move(store, ws, rel, ASIDE + rel, apply, pid=PID)

    if (proj.get("metadata") or {}).get(MARKER):
        print(f"reopen + ledger: already applied (marker {MARKER}) — skipped")
    else:
        for tid, (deps, why) in REOPEN.items():
            t = store.get_task(tid)
            if not t or t.get("project_id") != PID:
                print(f"skip reopen (no task): {tid}"); continue
            deps = [d for d in deps if (store.get_task(d) or {}).get("project_id") == PID]
            deps = list(dict.fromkeys(list(t.get("depends_on") or []) + deps))   # keep existing edges
            print(f"reopen {tid} [{t['status']}] after {len(deps)} task(s): {t['description'][:60]}")
            if apply:
                store.update_task(tid, status="PENDING", failure_reason="", depends_on=deps,
                                  result_summary=f"§4QA-d reopened: {why}.")
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
