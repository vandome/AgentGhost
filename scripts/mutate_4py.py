"""§4PY battery — owner commands answered by the system; list_lessons legend (2026-10-17). Harness of §4PG.
In place, backup + restore + byte compare; one mutant per run; py_compile first; the related test set;
no-op control must SURVIVE, a known-bad control must DIE.    python3 scripts/mutate_4py.py [all|rerun M1 M2 …]
"""
import os, shutil, subprocess, sys, signal, filecmp, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/Users/vasilis/Data/AI/.agent.venv/bin/python"
FR = "src/ghost_agent/core/failure_replay.py"; AG = "src/ghost_agent/core/agent.py"
MM = "src/ghost_agent/tools/memory.py"; RG = "src/ghost_agent/tools/registry.py"
def _tests():
    pat = r"failure_census\|failure_replay\|owner_rule\|rule_command\|_owner_rule_note\|list_lessons\|owner_test\|system_command_reply\|is_trivial_reply"
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py"))), capture_output=True, text=True).stdout.split()
    return sorted({os.path.relpath(t, ROOT) for t in out} | set(PINS))
MUT = [
    ("CONTROL no-op (must SURVIVE)", AG, "def system_command_reply(staged: str) -> str:", "# ctl\ndef system_command_reply(staged: str) -> str:"),
    ("CONTROL known-bad (must DIE): the model answers again", AG, "                if _owner_rule_banner_ctx.get():\n                    _sys_reply = system_command_reply(", "                if False:\n                    _sys_reply = system_command_reply("),
    # ── r2: the r1 fixes first ──
    ("M14 census books a greeting", AG, '                    await self._census_fast(req_id, _sys_reply, kind="command")', '                    await self._census_fast(req_id, _sys_reply)'),
    ("M15 census kind dropped in the hop", AG, '_fc.record_fast, str(req_id or ""), str(reply or ""), kind=kind), 2.0)', '_fc.record_fast, str(req_id or ""), str(reply or "")), 2.0)'),
    ("M16 record_fast ignores the kind", "src/ghost_agent/core/failure_census.py", '_append({"kind": "command" if kind == "command" else "fast", "req_id"', '_append({"kind": "fast", "req_id"'),
    ("M17 report never counts commands", "src/ghost_agent/core/failure_census.py", '            commands += cutoff < str(r.get("at") or "")[:10] <= today.isoformat()', '            pass'),
    ("M18 report line missing", "src/ghost_agent/core/failure_census.py", '        lines.append(f"owner commands answered by the system (no trajectory): "\n                     f"{report.get(\'system_command_turns\', 0)} recorded")', '        pass'),
    ("M4 reply labelable", AG, "                    self._note_trivial_reply(req_id)          # no trajectory behind it", "                    pass          # no trajectory behind it"),
    # ── r3: the r2 fix ──
    ("M19 diagnosis joined with empty parts", FR, '_c(d.get("explanation"), 400).strip()) if x)', '_c(d.get("explanation"), 400).strip()))'),
    ("M20 always a closing stop", FR, '+ ("" if _diag[-1:] in ".?!" else ".") if _diag else "")', '+ "." if _diag else "")'),
    ("M21 cause keeps its stop", FR, '_c(d.get("cause"), 40).strip(" ."),', '_c(d.get("cause"), 40).strip(),'),
    ("M22 shown when cleaning left nothing", FR, '+ ("" if _diag[-1:] in ".?!" else ".") if _diag else "")', '+ ("" if _diag[-1:] in ".?!" else ".") if d else "")'),
    ("M10 show diagnosis not defused", FR, '(_c(d.get("cause"), 40).strip(" ."), _c(d.get("explanation"), 400).strip())', '(str(d.get("cause") or "").strip(" ."), str(d.get("explanation") or "").strip())'),
    ("M6 show without the diagnosis", FR, '             + (f" The automatic diagnosis: {_diag}"', '             + ("" and f" The automatic diagnosis: {_diag}"'),
    ("M1 separator kept", AG, '    return _TRAILING_SEPARATOR_RE.sub("", str(staged or "")).strip()', '    return str(staged or "").strip()'),
    ("M2 every --- removed", AG, '    return _TRAILING_SEPARATOR_RE.sub("", str(staged or "")).strip()', '    return str(staged or "").replace("---", "").strip()'),
    ("M3 correction dropped", AG, "                    _sys_reply = system_command_reply(self._take_active_correction())", "                    _sys_reply = system_command_reply(_owner_rule_banner_ctx.get())"),
    ("M7 show without the request", FR, "             + (f\"From a replay of your request “{_c(case['request'], 200)}”.\" if case.get(\"request\")\n                else \"Approved by the operator, not a replay.\")", "             + \"\""),
    ("M8 show without verdicts", FR, "             + (f\" Replays without the rule: {_v(vb)}; with it: {_v(vt)}.\" if vb or vt else \"\")", "             + \"\""),
    ("M9 verdicts always 'no verdict'", FR, '        _v = lambda vs: "/".join(vs) if any(x != "none" for x in vs) else "no verdict"', '        _v = lambda vs: "no verdict"'),
    ("M11 legend not listed", MM, "    lines.append(LESSON_LEGEND)", "    pass"),
    ("M12 legend silent about the dot", MM, '"(✓ = verified lesson, · = unverified lesson. These are LESSONS', '"(These are LESSONS'),
    ("M13 description not about rules", RG, " These are NOT the owner's rules (### OWNER RULES, candidate/proposed rules): no tool lists or changes those.", ""),
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

PINS = ["tests/test_4py_system_command_reply.py", "tests/test_4px_dismiss_rules.py", "tests/test_4mx_rule_commands.py"]

def main():
    args = sys.argv[1:] or ["all"]
    if args[0] == "rerun":
        want = set(args[1:]); batch = [m for m in MUT if m[0].split(" ")[0] in want or m[0].startswith("CONTROL")]
    else:
        batch = MUT
    full = _tests(); files = sorted({m[1] for m in batch}); backups = {f: f + ".4py.bak" for f in files}
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
