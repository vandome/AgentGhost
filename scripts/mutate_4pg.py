#!/usr/bin/env python3
"""§4PG battery — the autopilot path review (2026-10-15). Same harness as §4PE/§4PF. — file_system / projects / knowledge_base / scratchpad / workspace / services review fixes (2026-10-15).
In place, backup + restore + byte compare; one mutant per run; py_compile first; the related test set;
no-op control must SURVIVE, a known-bad control must DIE.    python3 scripts/mutate_4pg.py [all|rerun M1 M2 …]
"""
import os, shutil, subprocess, sys, signal, filecmp, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/Users/vasilis/Data/AI/.agent.venv/bin/python"
FS = "src/ghost_agent/tools/file_system.py"; PJ = "src/ghost_agent/tools/projects.py"; MM = "src/ghost_agent/tools/memory.py"
PV = "src/ghost_agent/utils/provenance.py"; ST = "src/ghost_agent/core/strikes.py"; WS = "src/ghost_agent/tools/workspace.py"
WT = "src/ghost_agent/tools/workspace_track.py"; WM = "src/ghost_agent/workspace/model.py"
SV = "src/ghost_agent/tools/sandbox_services.py"; SS = "src/ghost_agent/sandbox/services.py"
def _tests():
    pat = r"coding_executor\|coding_loop\|project_research\|projects_routes\|project_digest\|project_advancer\|sandbox\.jobs\|jobs import\|workspace\.schema\|workspace\.state\|recognition\|request_origin_label\|file_system\|tools\.projects\|tools import projects\|tool_knowledge_base\|forget_execute\|forget_preview\|_resolve_plan\|user_confirms\|tool_scratchpad\|workspace_track\|tools\.workspace\|workspace\.model\|sandbox_services\|manage_services\|provenance\|strikes\|_declare_ingest\|tool_gain_knowledge\|render_document_outline\|_protected_dir_refusal\|_released_write_block\|tool_download_file\|_get_safe_path"
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py"))), capture_output=True, text=True).stdout.split()
    return sorted({os.path.relpath(t, ROOT) for t in out} | {"tests/test_4pe_files_projects_review.py", "tests/test_4pf_open_items.py", "tests/test_4pg_autopilot_review.py"})
LG = "src/ghost_agent/utils/logging.py"; SC = "src/ghost_agent/workspace/schema.py"; ST = "src/ghost_agent/workspace/state.py"
RC = "src/ghost_agent/workspace/recognition.py"; JB = "src/ghost_agent/sandbox/jobs.py"; PA = "src/ghost_agent/core/project_advancer.py"
PS = "src/ghost_agent/memory/projects.py"; DG = "src/ghost_agent/core/project_digest.py"; AG = "src/ghost_agent/core/agent.py"; EX = "src/ghost_agent/tools/execute.py"
CE = "src/ghost_agent/core/coding_executor.py"; CL = "src/ghost_agent/core/coding_loop.py"; PR = "src/ghost_agent/core/project_research.py"
RT = "src/ghost_agent/api/projects_routes.py"
MUT = [
    ("CONTROL no-op (must SURVIVE)", PA, "def _research_output_usable(output) -> bool:", "# ctl\ndef _research_output_usable(output) -> bool:"),
    ("CONTROL known-bad (must DIE): every unattended build closes DONE", PA, '    if unverified_build:\n        # §4PG (R2-1)', '    if False:\n        # §4PG (R2-1)'),
    ("M1 unverified build closes DONE", PA, '                                      unverified_build=not bool(getattr(cres, "verified", False)))', '                                      unverified_build=False)'),
    ("M2 contract quote is a verify command", CL, '    m = [x for x in _VERIFY_LINE_RE.findall(text or "")\n         if not _CONTRACT_PLACEHOLDER_RE.search(x)]', '    m = list(_VERIFY_LINE_RE.findall(text or ""))'),
    ("M3 VERIFY: true passes", CL, '        verify = "" if v.lower() in ("none", "n/a", "-", "", "true", "ok", "passed", "done") else v', '        verify = "" if v.lower() in ("none", "n/a", "-", "") else v'),
    ("M4 timed-out attempt files dropped", CL, '            carried = _carry_after_abort(ws, before, carried)        # §4PG (R2-2)\n            continue\n        except Exception as e:', '            continue\n        except Exception as e:'),
    ("M5 spec path never verified", CE, '                            verified=bool(_verify_ran_ok))   # §4PG r3: the verify RAN and passed', '                            verified=False)'),
    ("M6 postcondition demotion recorded as completed", PA, '        if _final == TaskStatus.FAILED and _st != TaskStatus.FAILED:', '        if False:'),
    ("M7 dotfiles skipped again", CL, '                if fn.startswith(".") and not _deliverable_dot(fn):', '                if fn.startswith("."):'),
    ("M8 big file silently skipped", CL, '                    h = hashlib.sha1(data[:1_000_000]).hexdigest()[:12] + f"+{len(data)}"\n                    out[str(p.relative_to(root)).replace(os.sep, "/")] = h\n                    continue', '                    continue'),
    ("M9 low-relevance research closes DONE", PA, '                         and (not research_path or not _research_output_usable(output))))   # §4PG (R3-4)', '                         and not research_path))'),
    ("M29 research judged by a snippet", PA, 'r"\\A\\s*(?:NOTE: low relevance|NOTE: every web search engine failed|ERROR: No search results|"', 'r"(?:NOTE: low relevance|NOTE: every web search engine failed|ERROR: No search results|No results found|"'),
    ("M30 deferral checked after the discard", CL, '            if is_background and "[ATTEMPT_ABORTED_UPSTREAM]" in str(reply or ""):', '            if False:'),
    ("M31 deferral scored as no progress", PA, '    if not (res.task_id or res.classification in ("timeout", "error")) or res.classification == "deferred":', '    if not (res.task_id or res.classification in ("timeout", "error")):'),
    ("M32 skipped verify counts as verified", CE, '        return None, False                 # §4PG r3: skipped is not verified', '        return None, True'),
    ("M33 cache dot-dirs are deliverables", CL, '            if any(part.startswith(".") and not _deliverable_dot(part) for part in _rel_parts):\n                continue', '            pass'),
    ("M34 probe id not prefixed on the route", RT, '        if _origin == "probe" and not _rid.startswith("probe-"):\n            _rid = f"probe-{_rid}"', '        pass'),
    ("M35 digest stops at 500", DG, '                if len(_chunk) < 500:\n                    break\n                _before = int(_chunk[-1].get("id") or 0)', '                break'),
    ("M10 planted source accepted", PR, 'for m in re.findall(r"^\\s*\\[Source:\\s*([^\\]\\n]+)\\]", search_output, flags=re.MULTILINE):', 'for m in re.findall(r"\\[Source:\\s*([^\\]]+)\\]", search_output):'),
    ("M11 owner batch beside a live step", PJ, '                if _live:\n                    return _err(f"an unattended autopilot step is running', '                if False:\n                    return _err(f"an unattended autopilot step is running'),
    ("M12 stale same-boot marker stays busy", PA, '            if time.time() - float(prev.get("ts") or 0) <= UNATTENDED_STEP_TIMEOUT_S + 600:\n                _claim["busy"] = True', '            if True:\n                _claim["busy"] = True'),
    ("M13 blocked project never stamped", PA, '        _stamp_autoadvanced(store, project_id)\n        try:\n            _last = store.list_events(project_id, limit=1, event_type="autopilot_blocked")', '        try:\n            _last = store.list_events(project_id, limit=1, event_type="autopilot_blocked")'),
    ("M14 blocked event every tick", PA, '            if not _last or (_last[0].get("payload") or {}).get("dependency") != _dep:', '            if True:'),
    ("M15 research write failure never pauses", PA, '    if score == "held" and "did not land" not in str(res.summary or ""):', '    if score == "held":'),
    ("M16 needs-user push every tick", PA, '        if _key in _NOTIFIED_HOLDS:\n            return', '        if False:\n            return'),
    ("M17 error after DONE reopens", PA, '    if tid and _task_status(store, project_id, tid) != "IN_PROGRESS":     # §4PG (R1-5): both kinds', '    if kind == "timeout" and tid and _task_status(store, project_id, tid) != "IN_PROGRESS":'),
    ("M18 failed build not pushed", PA, '    if not owner_requested:\n        _record_failed_activity(context, project_id, nxt.description, (cres.summary or "")[:160])   # §4PG (R1-8)', '    pass'),
    ("M19 deferral swallowed into FAILED", CE, '            if isinstance(e, _BgDeferred):\n                raise                       # §4PG (R2-3): the owner is back — the advancer reopens the leaf', '            if False:\n                raise'),
    ("M36 deferral is a crash in the advancer", PA, '            if isinstance(e, _BgDeferred):\n                # §4PG r3 (R2-3)', '            if False:\n                # §4PG r3 (R2-3)'),
    ("M20 routes accept autopilot metadata", RT, '    if bad:\n        raise HTTPException(400,', '    if False:\n        raise HTTPException(400,'),
    ("M21 routes: a probe is the owner", RT, '        if hdr == ORIGIN_PROBE or is_probe_request_id(rid):\n            return "probe"', '        if False:\n            return "probe"'),
    ("M22 route passes owner_requested=True", RT, '                owner_requested=(_origin != "probe"),   # §4PG (R3-1): a probe\'s advance is not the owner\'s', '                owner_requested=True,'),
    ("M37 probe_created stamp from an owner accepted", RT, '    if origin == "probe" and "probe_created" in md:\n        md = {k: v for k, v in md.items() if k != "probe_created"}', '    md = {k: v for k, v in md.items() if k != "probe_created"}'),
    ("M23 digest window again", DG, '                _chunk = store.list_events(pid, limit=500, since_id=int(last_event_id), before_id=_before)', '                _chunk = store.list_events(pid, limit=50, before_id=_before)'),
    ("M24 since_id ignored by the store", PS, '        _since = (" AND id > ?" if since_id is not None else "") + (" AND id < ?" if before_id is not None else "")', '        _since = (" AND id > ? - 999999999" if since_id is not None else "") + (" AND id < ?" if before_id is not None else "")'),
    ("M25 confinement never applied", FS, '            if kwargs.get("confine_to_sandbox"):\n                _cb = _confined_write_block(sandbox_dir, _target)', '            if False:\n                _cb = _confined_write_block(sandbox_dir, _target)'),
    ("M26 confinement accepts the outer root", FS, '        if t == sd or sd in t.parents:\n            return None\n    except Exception:  # noqa: BLE001\n        return None\n    return ToolOutcome.rejected(\n        f"SYSTEM BLOCK: this build runs unattended', '        return None\n    except Exception:  # noqa: BLE001\n        return None\n    return ToolOutcome.rejected(\n        f"SYSTEM BLOCK: this build runs unattended'),
    ("M27 leaf context not confined", CL, '    iso.confine_writes_to_project = True     # §4PG (R2-4): file writes stay inside the pinned workspace', '    pass'),
    ("M28 leaf journals as interactive", AG, '        if str(getattr(self.context, "turn_origin_label", "") or "") == "leaf":\n            return', '        pass'),
]
def run_tests(tests):
    env = dict(os.environ, PYTHONPATH="src", GHOST_API_KEY=open(os.path.expanduser("~/Data/AI/.ghost_api_key")).read().strip()); env.pop("FORCE_COLOR", None)
    try:
        r = subprocess.run([PY, "-m", "pytest", *tests, "-q", "--no-header", "-p", "no:cacheprovider", "-x", "-n", "8", "--dist", "loadfile"], capture_output=True, text=True, env=env, cwd=ROOT, timeout=1500)
    except subprocess.TimeoutExpired:
        return 1, "TIMEOUT (a mutant wedged the run — counted as killed)"
    return r.returncode, (r.stdout.strip().splitlines() or ["?"])[-1]
def _fast_tests(mutated_file, pins):
    """Tier 1: the round's pin files plus every test that imports the mutated
    module (by its module name or basename) — the tests that CAN kill a
    mutant in that file. A survivor is re-judged on the full selection."""
    mod = os.path.splitext(os.path.basename(mutated_file))[0]
    pat = mod + r"\b"
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py"))), capture_output=True, text=True).stdout.split()
    return sorted({os.path.relpath(x, ROOT) for x in out} | set(pins))

PINS = ["tests/test_4pg_autopilot_review.py", "tests/test_4pf_open_items.py", "tests/test_4pe_files_projects_review.py"]

def main():
    args = sys.argv[1:] or ["all"]
    if args[0] == "rerun":
        want = set(args[1:]); batch = [m for m in MUT if m[0].split(" ")[0] in want or m[0].startswith("CONTROL")]
    else:
        batch = MUT
    full = _tests(); files = sorted({m[1] for m in batch}); backups = {f: f + ".4pe.bak" for f in files}
    for f, b in backups.items(): shutil.copy2(os.path.join(ROOT, f), os.path.join(ROOT, b))
    def restore(*_):
        for f, b in backups.items(): shutil.copy2(os.path.join(ROOT, b), os.path.join(ROOT, f))
        subprocess.run(["find", os.path.join(ROOT, "src"), "-name", "__pycache__", "-exec", "rm", "-rf", "{}", "+"], capture_output=True)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP): signal.signal(sig, lambda *a: (restore(), sys.exit(130)))
    killed = survived = 0
    print(f"{len(full)} test files in the full selection; {len(batch)} mutants; two-tier (fast set, then the full selection for survivors)", flush=True)
    try:
        for label, f, old, new in batch:
            restore(); path = os.path.join(ROOT, f); src = open(path).read()
            if src.count(old) != 1: print(f"SKIP   {label}: anchor count {src.count(old)}", flush=True); continue
            open(path, "w").write(src.replace(old, new))
            if subprocess.run([PY, "-m", "py_compile", path], capture_output=True).returncode != 0: print(f"NOCOMP {label}", flush=True); continue
            fast = _fast_tests(f, PINS) if not label.startswith("CONTROL no-op") else full
            rc, tail = run_tests(fast); tier = f"fast {len(fast)} files"
            if rc == 0 and fast is not full:
                rc, tail = run_tests(full); tier += " → full"
            status = "SURVIVED" if rc == 0 else "KILLED"; killed += rc != 0; survived += rc == 0
            print(f"{status:8} {label} — [{tier}] {tail}", flush=True)
    finally:
        restore(); ok = all(filecmp.cmp(os.path.join(ROOT, f), os.path.join(ROOT, b), shallow=False) for f, b in backups.items())
        for b in backups.values(): os.remove(os.path.join(ROOT, b))
        print(f"RESTORED identical={ok} killed={killed} survived={survived}", flush=True)
if __name__ == "__main__": main()
