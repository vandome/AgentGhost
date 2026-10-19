"""§4QA-e: scripts/rerun_4qa_experiments.py on a scratch store (operator 2026-10-19:
"do 1 and 2" — the first real run cut every answer at 600 tokens)."""
import importlib.util
from pathlib import Path

from ghost_agent.memory.projects import ProjectStore

ROOT = Path(__file__).resolve().parents[1]


def _load(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("rerun_4qa_experiments",
                                                  ROOT / "scripts" / "rerun_4qa_experiments.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mem, sb = tmp_path / "mem", tmp_path / "sb"
    store = ProjectStore(mem, sandbox_root=sb)
    pid = store.create_project("Recursion", goal="run introspection loops on an LLM")
    layer = store.add_task(pid, "Task 3.1: Run Layer 0 → results/layer0-base.md")
    control = store.add_task(pid, "Create and run control condition: same prompts but non-recursive → compare outputs")
    analysis = store.add_task(pid, "Task 5.1: Cross-layer comparison → analysis/cmp.md")
    store.update_task(layer, status="DONE", result_summary="ran with probe_model")
    store.update_task(analysis, status="PENDING", depends_on=[layer])
    ws = Path(store.ensure_workspace(pid))
    (ws / "results").mkdir(parents=True)
    (ws / "results" / "layer0-base.md").write_text("# Layer 0\n\ncut at 600")
    (ws / "results" / "layer-cascade-transcript.md").write_text(mod.TRANSCRIPT_HEAD + "\n\n```text\nthe ladder of levels\n```\n")
    (ws / "results" / "control-transcript.md").write_text("# Model probe transcript\n\nhand-written\n")
    store.describe_file(pid, "results/layer0-base.md", "layer 0", "")
    self_ref = store.add_task(pid, "Task 4.4: self-referential start → results/self-referential-start.md")
    mod.WHY_BY_TASK = {self_ref: mod.WHY_BY_TASK["6102d00e031e"]}
    mod.SELF_REF_ID = self_ref
    for k, v in (("MEM", mem), ("SANDBOX", sb), ("PID", pid), ("TASKS", [layer, control, self_ref, "ffffffffffff"]),
                 ("DECLARE", {control: "results/control-condition.md", layer: "results/other.md"}),
                 ("RUN1_FILES", ["results/layer0-base.md", "results/layer-cascade-transcript.md",
                                 "results/control-transcript.md", "results/missing.md"])):
        monkeypatch.setattr(mod, k, v)
    mod.CONTROL_ID = control
    return mod, store, pid, layer, analysis, ws


def test_the_dry_run_writes_nothing(monkeypatch, tmp_path, capsys):
    mod, store, pid, layer, analysis, ws = _load(monkeypatch, tmp_path)
    mod.main(apply=False)
    assert (ws / "results" / "layer0-base.md").is_file()
    assert store.get_task(layer)["status"] == "DONE"
    assert "dry run" in capsys.readouterr().out


def test_apply_keeps_the_first_run_and_reopens_the_experiment(monkeypatch, tmp_path):
    mod, store, pid, layer, analysis, ws = _load(monkeypatch, tmp_path)
    mod.main(apply=True)
    assert not (ws / "results" / "layer0-base.md").exists()
    assert (ws / "results" / "run1-600tok" / "layer0-base.md").read_text().endswith("cut at 600")
    assert (ws / "results" / "run1-600tok" / "layer-cascade-transcript.md").is_file()
    # r28: labelled as the real (cut) run it was — never "not data"
    desc = store.get_file_manifest(pid)["results/run1-600tok/layer0-base.md"]["desc"]
    assert desc.startswith("first real run") and "not data" not in desc
    t = store.get_task(layer)
    assert t["status"] == "PENDING" and "§4QA-e rerun" in t["result_summary"]
    a = store.get_task(analysis)
    assert a["status"] == "PENDING" and a["depends_on"] == [layer]      # still waits for it
    assert "§4QA-e" in store.get_ledger(pid) and store.get_project(pid)["status"] == "ACTIVE"
    from ghost_agent.core.task_deliverable import declared_deliverable
    # the control now names its write-up, so the quote check reads it; a task
    # that already declares a file keeps it
    assert declared_deliverable(store.get_task(mod.CONTROL_ID)["description"]) == "results/control-condition.md"
    assert declared_deliverable(store.get_task(layer)["description"]) == "results/layer0-base.md"


def test_a_second_apply_never_reopens_a_rerun_closed_since(monkeypatch, tmp_path):
    mod, store, pid, layer, analysis, ws = _load(monkeypatch, tmp_path)
    mod.main(apply=True)
    store.update_task(layer, status="DONE", result_summary="rerun at 1500")
    mod.main(apply=True)
    assert store.get_task(layer)["status"] == "DONE"
    assert store.get_ledger(pid).count("§4QA-e") == 1


def test_apply_keeps_the_first_runs_system_written_transcripts_for_the_quote_check(monkeypatch, tmp_path):
    # r29: run 1 predates the check — its moved transcripts are kept under
    # their new path, so a write-up quoting run 1 is judged against run 1
    mod, store, pid, layer, analysis, ws = _load(monkeypatch, tmp_path)
    from ghost_agent.core.probe_transcripts import recorded
    mod.main(apply=False)
    assert recorded(tmp_path / "mem", pid) == []
    mod.main(apply=True)
    kept = recorded(tmp_path / "mem", pid)
    # only the file the SYSTEM wrote; a file without its header is not kept
    assert [r for r, _ in kept] == ["results/run1-600tok/layer-cascade-transcript.md"]
    assert "the ladder of levels" in kept[0][1]


def test_task_4_4_is_told_to_start_from_its_own_prompt(monkeypatch, tmp_path):
    # r29: run 1 put the main cascade's paradox prompts to 4.4
    mod, store, pid, layer, analysis, ws = _load(monkeypatch, tmp_path)
    mod.main(apply=True)
    why = store.get_task(mod.SELF_REF_ID)["result_summary"]
    assert "§4QA-e rerun" in why and "self-referential prompt of its own" in why
    assert "of its own" not in store.get_task(layer)["result_summary"]


def test_the_real_lists_cover_the_whole_first_run():
    spec = importlib.util.spec_from_file_location("rerun_real", ROOT / "scripts" / "rerun_4qa_experiments.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert len(mod.TASKS) == len(set(mod.TASKS)) == 12 and "fa70d8ce6590" in mod.TASKS
    assert set(mod.WHY_BY_TASK) == {"6102d00e031e"} <= set(mod.TASKS)
    assert mod.DECLARE == {"fef34a33bcd0": "results/control-condition.md"}
    assert all(r.startswith("results/") and "/" not in r[len("results/"):] for r in mod.RUN1_FILES)
    for need in ("results/meta-reflection.md", "results/harness_9_results.json",
                 "results/control-condition.md", "results/layer-cascade-transcript.md"):
        assert need in mod.RUN1_FILES
    assert sum(r.endswith("-transcript.md") for r in mod.RUN1_FILES) == 7


def test_a_second_apply_never_moves_a_file_the_rerun_wrote(monkeypatch, tmp_path):
    # r30 M8: a run-1 path missing at the first apply, written by the rerun,
    # was moved into run1-600tok/ by the second
    mod, store, pid, layer, analysis, ws = _load(monkeypatch, tmp_path)
    mod.main(apply=True)
    (ws / "results" / "missing.md").write_text("# rerun at 1500 tokens\n\nfull answers here")
    (ws / "results" / "layer-cascade-transcript.md").write_text(mod.TRANSCRIPT_HEAD + "\n\nrerun")
    mod.main(apply=True)
    assert (ws / "results" / "missing.md").is_file()
    assert (ws / "results" / "layer-cascade-transcript.md").read_text().endswith("rerun")
    assert not (ws / "results" / "run1-600tok" / "missing.md").exists()
