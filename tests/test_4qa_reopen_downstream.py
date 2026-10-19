"""§4QA-d: scripts/reopen_4qa_downstream.py on a scratch store (operator 2026-10-19:
"fix all open items" — the analysis/theory/report built on the invented
experiments, and the harness that ran before any experiment existed)."""
import importlib.util
from pathlib import Path

from ghost_agent.core.planning import ProjectPlan
from ghost_agent.memory.projects import ProjectStore

ROOT = Path(__file__).resolve().parents[1]


def _load(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("reopen_4qa_downstream",
                                                  ROOT / "scripts" / "reopen_4qa_downstream.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mem, sb = tmp_path / "mem", tmp_path / "sb"
    store = ProjectStore(mem, sandbox_root=sb)
    pid = store.create_project("Recursion", goal="run introspection loops on an LLM")
    layer = store.add_task(pid, "Task 3.1: Run Layer 0 → results/layer0-base.md")
    harness = store.add_task(pid, "Run recursive_harness.py on the layer outputs")
    analysis = store.add_task(pid, "Task 5.1: Cross-layer comparison → analysis/cross-layer-comparison.md")
    other = store.add_task(pid, "Task 1.1: Research → research/r.md")
    store.update_task(layer, status="NEEDS_USER", result_summary="model experiment? — in your turn")
    store.update_task(harness, status="NEEDS_USER", result_summary="built but no verification")
    for t in (analysis, other):
        store.update_task(t, status="DONE", result_summary="done")
    ws = Path(store.ensure_workspace(pid))
    (ws / "analysis").mkdir(parents=True)
    (ws / "analysis" / "cross-layer-comparison.md").write_text("Layer 2 shows 0.85 density (invented)")
    store.describe_file(pid, "analysis/cross-layer-comparison.md", "comparison", "")
    for k, v in (("MEM", mem), ("SANDBOX", sb), ("PID", pid),
                 ("SIMULATED", ["analysis/cross-layer-comparison.md", "tools/missing.json"]),
                 ("REOPEN", {harness: ([layer], "score the new results"),
                             analysis: ([harness, layer, "ffffffffffff"], "rewrite it")})):
        monkeypatch.setattr(mod, k, v)
    return mod, store, pid, dict(layer=layer, harness=harness, analysis=analysis, other=other), ws


def test_the_dry_run_writes_nothing(monkeypatch, tmp_path, capsys):
    mod, store, pid, t, ws = _load(monkeypatch, tmp_path)
    mod.main(apply=False)
    assert (ws / "analysis" / "cross-layer-comparison.md").is_file()
    assert store.get_task(t["analysis"])["status"] == "DONE"
    assert "dry run" in capsys.readouterr().out


def test_apply_moves_the_file_aside_and_reopens_with_dependencies(monkeypatch, tmp_path):
    mod, store, pid, t, ws = _load(monkeypatch, tmp_path)
    mod.main(apply=True)
    assert not (ws / "analysis" / "cross-layer-comparison.md").exists()
    assert (ws / "simulated" / "analysis" / "cross-layer-comparison.md").read_text().endswith("(invented)")
    a = store.get_task(t["analysis"])
    assert a["status"] == "PENDING" and "§4QA-d" in a["result_summary"]
    assert a["depends_on"] == [t["harness"], t["layer"]]      # an unknown id is dropped
    assert store.get_task(t["harness"])["depends_on"] == [t["layer"]]
    assert store.get_task(t["other"])["status"] == "DONE"
    assert "§4QA-d" in store.get_ledger(pid)
    assert store.get_project(pid)["status"] == "ACTIVE"


def test_the_autopilot_order_waits_for_the_experiment(monkeypatch, tmp_path):
    """The live defect: the harness ran while every experiment was still open.
    With the edges set, nothing downstream is ready until the layer is DONE."""
    mod, store, pid, t, ws = _load(monkeypatch, tmp_path)
    mod.main(apply=True)
    plan = ProjectPlan(store, pid)
    nxt = plan.next_ready_leaf()
    assert nxt is None or nxt.id not in (t["harness"], t["analysis"])
    store.update_task(t["layer"], status="DONE", result_summary="ran with probe_model")
    assert ProjectPlan(store, pid).next_ready_leaf().id == t["harness"]


def test_a_second_apply_never_reopens_a_task_closed_since(monkeypatch, tmp_path):
    mod, store, pid, t, ws = _load(monkeypatch, tmp_path)
    mod.main(apply=True)
    store.update_task(t["analysis"], status="DONE", result_summary="rewritten from the transcripts")
    mod.main(apply=True)
    assert store.get_task(t["analysis"])["status"] == "DONE"
    assert store.get_ledger(pid).count("§4QA-d") == 1


def test_the_real_graph_reaches_every_reopened_experiment_and_has_no_cycle():
    """review r19: the tests above monkeypatch REOPEN, so the real graph was
    never checked — and cross-model validation and the falsification tests
    had nothing waiting on them, so the report could close without them."""
    spec = importlib.util.spec_from_file_location("reopen_4qa_downstream",
                                                  ROOT / "scripts" / "reopen_4qa_downstream.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    deps = {t: list(d) for t, (d, _why) in mod.REOPEN.items()}

    def reach(t, seen=None):
        seen = set() if seen is None else seen
        for d in deps.get(t, []):
            if d not in seen:
                seen.add(d)
                reach(d, seen)
        return seen

    for t in deps:
        assert t not in reach(t), f"cycle through {t}"
    from_report = reach(mod.REPORT_DATA) | {mod.REPORT_DATA}
    experiments = set(mod.EXP.REOPEN)                   # everything §4QA-c reopened
    assert experiments <= from_report, experiments - from_report
    assert mod.EMBED in from_report and mod.META_REFL in from_report
    assert {mod.CONTROL, mod.EMBED} <= set(deps[mod.HARNESS])


def test_existing_dependencies_are_kept(monkeypatch, tmp_path):
    mod, store, pid, t, ws = _load(monkeypatch, tmp_path)
    store.update_task(t["analysis"], depends_on=[t["other"]])
    mod.main(apply=True)
    assert store.get_task(t["analysis"])["depends_on"] == [t["other"], t["harness"], t["layer"]]
