#!/usr/bin/env python3
"""§4PF battery — the §4PE open items closed before autopilot goes to production (2026-10-15).
Same harness as §4PE. — file_system / projects / knowledge_base / scratchpad / workspace / services review fixes (2026-10-15).
In place, backup + restore + byte compare; one mutant per run; py_compile first; the related test set;
no-op control must SURVIVE, a known-bad control must DIE.    python3 scripts/mutate_4pf.py [all|rerun M1 M2 …]
"""
import os, shutil, subprocess, sys, signal, filecmp, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/Users/vasilis/Data/AI/.agent.venv/bin/python"
FS = "src/ghost_agent/tools/file_system.py"; PJ = "src/ghost_agent/tools/projects.py"; MM = "src/ghost_agent/tools/memory.py"
PV = "src/ghost_agent/utils/provenance.py"; ST = "src/ghost_agent/core/strikes.py"; WS = "src/ghost_agent/tools/workspace.py"
WT = "src/ghost_agent/tools/workspace_track.py"; WM = "src/ghost_agent/workspace/model.py"
SV = "src/ghost_agent/tools/sandbox_services.py"; SS = "src/ghost_agent/sandbox/services.py"
def _tests():
    pat = r"project_digest\|project_advancer\|sandbox\.jobs\|jobs import\|workspace\.schema\|workspace\.state\|recognition\|request_origin_label\|file_system\|tools\.projects\|tools import projects\|tool_knowledge_base\|forget_execute\|forget_preview\|_resolve_plan\|user_confirms\|tool_scratchpad\|workspace_track\|tools\.workspace\|workspace\.model\|sandbox_services\|manage_services\|provenance\|strikes\|_declare_ingest\|tool_gain_knowledge\|render_document_outline\|_protected_dir_refusal\|_released_write_block\|tool_download_file\|_get_safe_path"
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py"))), capture_output=True, text=True).stdout.split()
    return sorted({os.path.relpath(t, ROOT) for t in out} | {"tests/test_4pe_files_projects_review.py", "tests/test_4pf_open_items.py"})
LG = "src/ghost_agent/utils/logging.py"; SC = "src/ghost_agent/workspace/schema.py"; ST = "src/ghost_agent/workspace/state.py"
RC = "src/ghost_agent/workspace/recognition.py"; JB = "src/ghost_agent/sandbox/jobs.py"; PA = "src/ghost_agent/core/project_advancer.py"
PS = "src/ghost_agent/memory/projects.py"; DG = "src/ghost_agent/core/project_digest.py"; AG = "src/ghost_agent/core/agent.py"; EX = "src/ghost_agent/tools/execute.py"
MUT = [
    ("CONTROL no-op (must SURVIVE)", LG, "def request_origin_label(rid=None) -> str:", "# ctl\ndef request_origin_label(rid=None) -> str:"),
    ("CONTROL known-bad (must DIE): every origin is owner", LG, '    pinned = str(event_origin_context.get() or "")\n    if pinned:\n        return pinned', '    return "owner"'),
    ("M1 pinned origin ignored", LG, '    pinned = str(event_origin_context.get() or "")\n    if pinned:\n        return pinned', '    pinned = ""'),
    ("M2 sched- reads as sub-agent", LG, '        return "scheduled" if r.startswith("sched-") else ("self-play" if r.startswith("sim-") else "sub-agent")', '        return "sub-agent"'),
    ("M3 origin dropped on read", SC, '            origin=str(d.get("origin") or ""),            # §4PF', ''),
    ("M4 ledger rows not stamped", WM, 'def _origin_now() -> str:\n    """§4PF: the origin label for an event recorded right now."""\n    try:', 'def _origin_now() -> str:\n    return ""\n    try:'),
    ("M5 prefix tag dropped", RC, '    return f" [{o}]" if o and o != "owner" else ""', '    return ""'),
    ("M6 advance tick pins owner", PA, '    _tok = _eoc.set(_tick_origin(owner_requested))', '    _tok = _eoc.set("owner")'),
    ("M7 project events not stamped", PS, '            payload.setdefault("origin", request_origin_label())', '            pass'),
    ("M8 digest counts the owner's own turns", DG, '(ev.get("payload") or {}).get("origin") in ("probe", "owner"):', '(ev.get("payload") or {}).get("origin") in ("probe",):'),
    ("M9 finished job never booked", WM, '            if origin == "probe":\n                return None\n            if state == STATE_EXPIRED_LABEL:', '            return None\n            if state == STATE_EXPIRED_LABEL:'),
    ("M10 promotion row never hidden", WS, '        done = _finished_job_ids(events)', '        done = set()'),
    ("M11 reaper hook not fired", JB, '                    self._cleanup_files(jid, entry=entry)\n                    _fire_finish(entry)                                  # §4PF (W7)', '                    self._cleanup_files(jid, entry=entry)'),
    ("M12 promotion row without job id", EX, '            job_id=str(job.get("id") or ""),                       # §4PF (W7)', ''),
    ("M13 missing file is newly tracked news", ST, '        return "newly tracked" if curr.exists else None       # §4PF (W6): a missing file is not a change', '        return "newly tracked"'),
    ("M14 research dedupe global again", WM, '            if self._research_recorded(url, _pid):\n                return None', '            if self.state.has_seen_url(url):\n                return None'),
    ("M15 listings unscoped", WS, '    return (filter_events_for_project(events, pid) if pid else list(events)), pid', '    return list(events), pid'),
    ("M16 nudge persisted again", WM, '            # §4PF (W13): a per-session nudge is returned to the caller, never\n            # persisted as a project event\n            return suggestion', '            self.note(suggestion, url=u, count=n, repeat=True)\n            return suggestion'),
    ("M17 a no does not retire previews", MM, '        with _PLANS_LOCK:\n            gone = [t for t, p in _FORGET_PLANS.items() if p.get("conv") == conv]\n            for t in gone:\n                _FORGET_PLANS.pop(t, None)', '        gone = []'),
    ("M18 narrowing retires previews too", MM, '        if not re.fullmatch(r"(?:no|nope|nah|cancel|don\'?t|οχι|ακυρο|ακυρωσε)[\\s.!,]*"\n                            r"(?:thanks|thank you|ευχαριστω)?[\\s.!]*", text):\n            return 0', '        pass'),
    ("M19 chat handler never retires", AG, '                    from ..tools.memory import retire_declined_previews as _rdp\n                    _rdp(last_user_content)', '                    pass'),
    ("M20 miss not logged", MM, '    if log:\n        pretty_log("Confirm Lookup", f"no preview found: {reason}", icon=Icons.BRAIN_PLAN, level="WARNING")', '    pass'),
    ("M21 ingest claim never refuses", MM, '            if _claim in _TOOL_INGEST_IN_FLIGHT:\n                return ToolOutcome.rejected(', '            if False:\n                return ToolOutcome.rejected('),
    ("M22 ingest claim never released", MM, '                _TOOL_INGEST_IN_FLIGHT.discard(_claim)', '                pass'),
    ("M23 index failure is SUCCESS again", MM, '        await asyncio.to_thread(_rollback_partial, memory_system, filename)\n        return (f"Embedding Error: the library index could not be updated ({e}) — nothing was kept; "\n                f"a retry ingests it from scratch.")', '        pass'),
    ("M24 update ARCHIVED bypasses archive", PJ, '                if str(status).upper() == "ARCHIVED":', '                if False:'),
    ("M25 outer root by parent name only", FS, '    if _fold(sd.parent.name) == "projects" and re.fullmatch(r"[0-9a-f]{12}", _fold(sd.name)):', '    if _fold(sd.parent.name) == "projects":'),
    ("M26 PDF page uncapped", FS, '                if len(text) > chunk_size:                     # §4PF (F11): a page is bounded like a text chunk', '                if False:'),
    ("M27 write-time swap check skipped", FS, '        def _atomic_write():\n            _swap0 = _parent_swapped(sandbox_dir, path)', '        def _atomic_write():\n            _swap0 = None'),
    ("M28 replace result untagged", FS, '        result = _with_text(result, _txt.rstrip() + f" (This call\'s old_string {_old_tag(old_text)}.)")', '        pass'),
    ("M29 portless unconfirmed reads RUNNING", SS, '        _unconfirmed = port is None and _pst is None               # §4PF (W9)', '        _unconfirmed = False'),
    # ── round 4 ──
    ("M30 job finish booked on the observer's project", WM, '                with pinned_event_project(str((entry or {}).get("project_id") or "")):', '                if True:'),
    ("M31 job finish booked with the observer's origin", WM, '            tok = event_origin_context.set(origin)', '            tok = event_origin_context.set(event_origin_context.get())'),
    ("M32 a probe's job leaves a row", WM, '            if origin == "probe":\n                return None', '            pass'),
    ("M33 expired jobs do not fire the hook", JB, '                    entry["expired_reason"] = "log_size_cap"\n                    self._cleanup_files(jid, entry=entry)\n                    _fire_finish(entry)                                  # §4PF r4', '                    entry["expired_reason"] = "log_size_cap"\n                    self._cleanup_files(jid, entry=entry)'),
    ("M34 reader ignores the project filter", "src/ghost_agent/workspace/activity.py", '            if _want is not None and _want and not filter_events_for_project([ev], _want):\n                continue', '            pass'),
    ("M35 listing scan unfiltered", WS, '    events = workspace_model.activity.recent(limit=max(limit * 4, limit), kind=kind, project_id=pid or None)', '    events = workspace_model.activity.recent(limit=max(limit * 4, limit), kind=kind)'),
    ("M36 probe advance pinned owner", PA, '        if request_kind() == "probe":\n            return "probe"', '        pass'),
    ("M37 bare stop retires previews", MM, 'r"(?:no|nope|nah|cancel|don\'?t|οχι|ακυρο|ακυρωσε)[\\s.!,]*"', 'r"(?:no|nope|nah|cancel|stop|abort|don\'?t|οχι|ακυρο|ακυρωσε)[\\s.!,]*"'),
    ("M38 archived update drops the other fields", PJ, '                    if _other:\n                        store.update_project(project_id, **_other)', '                    pass'),
    ("M39 write-time containment not re-checked", FS, '        if any(real == r or r in real.parents for r in roots):\n            return None\n    except Exception:  # noqa: BLE001\n        return None', '        return None\n    except Exception:  # noqa: BLE001\n        return None'),
    ("M40 listings use the shared project, not the pin", WS, '        return str(workspace_model._stamp_project_id() or "")', '        return str(getattr(workspace_model, "current_project_id", "") or "")'),
]
def run_tests(tests):
    env = dict(os.environ, PYTHONPATH="src", GHOST_API_KEY=open(os.path.expanduser("~/Data/AI/.ghost_api_key")).read().strip()); env.pop("FORCE_COLOR", None)
    try:
        r = subprocess.run([PY, "-m", "pytest", *tests, "-q", "--no-header", "-p", "no:cacheprovider", "-x", "-n", "8", "--dist", "loadfile"], capture_output=True, text=True, env=env, cwd=ROOT, timeout=1500)
    except subprocess.TimeoutExpired:
        return 1, "TIMEOUT (a mutant wedged the run — counted as killed)"
    return r.returncode, (r.stdout.strip().splitlines() or ["?"])[-1]
def main():
    args = sys.argv[1:] or ["all"]
    if args[0] == "rerun":
        want = set(args[1:]); batch = [m for m in MUT if m[0].split(" ")[0] in want or m[0].startswith("CONTROL")]
    else:
        batch = MUT
    tests = _tests(); files = sorted({m[1] for m in batch}); backups = {f: f + ".4pe.bak" for f in files}
    for f, b in backups.items(): shutil.copy2(os.path.join(ROOT, f), os.path.join(ROOT, b))
    def restore(*_):
        for f, b in backups.items(): shutil.copy2(os.path.join(ROOT, b), os.path.join(ROOT, f))
        subprocess.run(["find", os.path.join(ROOT, "src"), "-name", "__pycache__", "-exec", "rm", "-rf", "{}", "+"], capture_output=True)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP): signal.signal(sig, lambda *a: (restore(), sys.exit(130)))
    killed = survived = 0
    print(f"{len(tests)} test files; {len(batch)} mutants", flush=True)
    try:
        for label, f, old, new in batch:
            restore(); path = os.path.join(ROOT, f); src = open(path).read()
            if src.count(old) != 1: print(f"SKIP   {label}: anchor count {src.count(old)}", flush=True); continue
            open(path, "w").write(src.replace(old, new))
            if subprocess.run([PY, "-m", "py_compile", path], capture_output=True).returncode != 0: print(f"NOCOMP {label}", flush=True); continue
            rc, tail = run_tests(tests); status = "SURVIVED" if rc == 0 else "KILLED"; killed += rc != 0; survived += rc == 0
            print(f"{status:8} {label} — {tail}", flush=True)
    finally:
        restore(); ok = all(filecmp.cmp(os.path.join(ROOT, f), os.path.join(ROOT, b), shallow=False) for f, b in backups.items())
        for b in backups.values(): os.remove(os.path.join(ROOT, b))
        print(f"RESTORED identical={ok} killed={killed} survived={survived}", flush=True)
if __name__ == "__main__": main()
