#!/usr/bin/env python3
"""§4PH battery — the interactive coding-turn review (2026-10-15). Same two-tier harness as §4PG:
in place, backup + restore + byte compare; one mutant per run; py_compile first; per mutant the round's PIN
files + the tests that import the mutated module (fast), survivors re-judged on the full selection;
no-op control must SURVIVE (full set), a known-bad control must DIE.    python3 scripts/mutate_4ph.py [all|rerun M1 M2 …]
"""
import os, shutil, subprocess, sys, signal, filecmp, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/Users/vasilis/Data/AI/.agent.venv/bin/python"
AG = "src/ghost_agent/core/agent.py"; CC = "src/ghost_agent/core/code_claims.py"; ST = "src/ghost_agent/core/strikes.py"
EX = "src/ghost_agent/tools/execute.py"; TF = "src/ghost_agent/tools/tool_failure.py"; DK = "src/ghost_agent/sandbox/docker.py"
PR = "src/ghost_agent/core/prompts.py"; RG = "src/ghost_agent/tools/registry.py"; FS = "src/ghost_agent/tools/file_system.py"
CS = "src/ghost_agent/utils/constraints.py"
def _tests():
    pat = r"code_claims\|_unverified_mutation\|error_line_fingerprint\|normalise_volatile\|note_run_outcome\|_is_removal_capable\|_keys_removable_after_write\|_normalise_exit\|_normalise_pipe_exit\|tool_execute\|classify_tool_failure\|exec_exit_code\|_syntax_feedback\|request_forbids_running\|_PIPESTATUS\|_exec_run_streamed\|SYSTEM_PROMPT\|TOOL_DEFINITIONS\|futility"
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py"))), capture_output=True, text=True).stdout.split()
    return sorted({os.path.relpath(t, ROOT) for t in out} | set(PINS))
MUT = [
    ("CONTROL no-op (must SURVIVE)", CC, "def claims_code_works(text: str) -> str:", "# ctl\ndef claims_code_works(text: str) -> str:"),
    ("CONTROL known-bad (must DIE): the claim arm never runs", AG, '    if not _is_unverified_mutation(last_tool):\n        if reply:', '    if not _is_unverified_mutation(last_tool):\n        if False:'),
    ("M2 a suite run backs a web page", CC, '                if k in keys or (suite and not _is_web(rec["path"])):', '                if k in keys or suite:'),
    ("M3 a failed run reads as backed", CC, '        elif rec["run_ok"] is False:\n            failed.append(rec["path"])', '        elif False:\n            failed.append(rec["path"])'),
    ("M4 heredoc writes invisible", CC, '    for m in _SHELL_WRITE_RE.finditer(cmd):\n        out.append(m.group(1))', '    for m in []:\n        out.append(m.group(1))'),
    ("M5 a heredoc target counts as a run", CC, '    cmd = _SHELL_WRITE_RE.sub(" ", str(args.get("command") or ""))', '    cmd = str(args.get("command") or "")'),
    ("M6 negation guard dropped", CC, '        if _NEGATION_RE.search(_clause):\n            continue', '        if False:\n            continue'),
    ("M7 a detached run counts", CC, '            if ok is None:\n                continue\n            keys, suite = _exec_run_keys(t, args)', '            keys, suite = _exec_run_keys(t, args)'),
    ("M8 don't-run-it labelled failed", CC, '    if forbids_running:\n        return (f"\\n\\n---\\n*Not run, as you asked', '    if False:\n        return (f"\\n\\n---\\n*Not run, as you asked'),
    ("M9 finalize passes no reply", AG, '                            reply=final_ai_content)          # §4PH: the claim arm reads the reply', '                            reply="")'),
    ("M10 stream passes no reply", AG, '                    reply=full_content)                      # §4PH: the claim arm reads the reply', '                    reply="")'),
    ("M11 streamed work_log completed again", AG, '                    verifier_backfill=(("failed", UNVERIFIED_MUTATION_REASON)\n                                       if _um_failed_stream else None),', '                    verifier_backfill=None,'),
    ("M12 async branch ignores the arm", AG, '                            _unverified = _is_unverified_mutation(_lt) or bool(_claim_directive)', '                            _unverified = _is_unverified_mutation(_lt)'),
    ("M13 sync branch ignores the arm", AG, '                                and _is_unverified_mutation(_lt)\n                            ) or bool(_claim_directive)', '                                and _is_unverified_mutation(_lt)\n                            )'),
    ("M14 generic directive", AG, '                            _directive = _claim_directive or (', '                            _directive = ('),
    ("M15 execute removal-capable by name", AG, '        if isinstance(args, dict) and args:\n            blob = " ".join(str(args.get(k) or "") for k in ("command", "content", "filename"))', '        if False:\n            blob = " ".join(str(args.get(k) or "") for k in ("command", "content", "filename"))'),
    ("M16 call site passes no args", AG, '            if _is_removal_capable(name, str(t.get("content", "")),\n                                   getattr(t.get("content"), "call_args", None)):', '            if _is_removal_capable(name, str(t.get("content", "")),\n                                   None):'),
    ("M17 no .sh parse check", FS, '        elif ext in ("sh", "bash"):', '        elif ext in ("shx",):'),
    ("M18 qualified object dropped", CS, '    r"|downloaded|untrusted|external|remote|unsafe|arbitrary|unknown)\\b"', '    r")\\b"'),
    ("M19 numbers kept in the fingerprint", ST, '    norm = _NUMERIC_RE.sub("n", norm)', '    norm = norm'),
    ("M39 counts collapsed too", ST, '    if not _COUNT_LINE_RE.search(norm):\n        norm = _NUMERIC_RE.sub("n", norm)', '    if True:\n        norm = _NUMERIC_RE.sub("n", norm)'),
    ("M20 modules collapse again", ST, '    _keep = bool(_DISTINCT_LITERAL_RE.search(line))', '    _keep = False'),
    ("M21 futility fails on whole output", AG, '                fp = _elf(text, tool="execute") or fp', '                fp = fp'),
    ("M22 list laundering kept", EX, '    code = _normalise_list_exit(command, code, output)', '    code = code'),
    ("M23 && treated as a launderer", EX, '        elif c == ";" or c == "\\n" or s.startswith("||", i):', '        elif c == ";" or c == "\\n" or s.startswith(("||", "&&"), i):'),
    ("M24 None exit is 0 again", EX, '    if exit_code is None:\n        return 1, str(output or "") + _NO_STATUS_NOTE', '    if False:\n        return 1, str(output or "") + _NO_STATUS_NOTE'),
    ("M25 own timeout blamed on the budget", EX, '        elif (exit_code == 124 and elapsed_s is not None\n              and float(elapsed_s) < _EXEC_TIMEOUT_S - _OOM_KILL_MARGIN_S):', '        elif False:'),
    ("M26 broken-pipe text forgives any code", EX, '    if code == 1 and _BROKEN_PIPE_RE.search(str(output or "")):', '    if _BROKEN_PIPE_RE.search(str(output or "")):'),
    ("M27 no background note", EX, '        if promoted_job is None and _bg_detached(command):', '        if False:'),
    ("M40 background note on raw text", EX, '    cmd = _QUOTED_RE.sub(" ", cmd)\n    if re.search(r"\\bkill\\s+%", cmd):\n        return False\n    return bool(_BG_AMP_RE.search(cmd))', '    if re.search(r"\\bkill\\s+%", cmd):\n        return False\n    return bool(_BG_AMP_RE.search(str(command or "")))'),
    ("M41 laundering without a runner", EX, '    if code != 0 or not _list_launder_possible(command) or not _list_has_runner(command):', '    if code != 0 or not _list_launder_possible(command):'),
    ("M28 job banner wins wherever it is", TF, '    if mj is not None and (m is None or mj.start() < m.start()):', '    if mj is not None:'),
    ("M42 module runs not credited", CC, '    for a, b in _MODULE_RUN_RE.findall(cmd):\n        mod = (a or b)', '    for a, b in []:\n        mod = (a or b)'),
    ("M43 mechanism prose is a claim", CC, '    r"(?!\\s+(?:by|like|as\\s+follows|because|via|through|with|on|in|at|when|once|if|only|after|before|until)\\b)"', '    r""'),
    ("M29 infra is a strike", TF, '    re.compile(r"\\[SANDBOX INFRA ERROR", re.IGNORECASE),', ''),
    ("M30 budget kill transient again", TF, '    # §4PH (R2-2): the execute tool\'s own budget kill is NOT transient', '    re.compile(r"timed out / killed after \\d+s", re.IGNORECASE),\n    # §4PH (R2-2): the execute tool\'s own budget kill is NOT transient'),
    ("M31 PIPESTATUS last non-zero", EX, '''    '__g_r=0; for __g_s in "${__g_ps[@]}"; do [ "$__g_s" -ne 0 ] && [ "$__g_r" -eq 0 ] && __g_r=$__g_s; done; exit $__g_r')''', '''    '__g_r=0; for __g_s in "${__g_ps[@]}"; do [ "$__g_s" -ne 0 ] && __g_r=$__g_s; done; exit $__g_r')'''),
    ("M32 docker None not polled", DK, '                while code is None and _tries < 20:', '                while False:'),
    ("M33 last resort again", RG, '"description": "Run Python, Node.js, or Shell code — scripts, tests, shell commands — and the way to CHECK code you wrote: run the file (or its tests) and read the result before you say it works.', '"description": "Run Python, Node.js, or Shell code. USE THIS ONLY AS A LAST RESORT for custom math, logic, or formatting.'),
    ("M34 a failed status reads ok", CC, '    if st in ("failed", "rejected", "partial"):\n        return False', '    if st in ("failed", "rejected", "partial"):\n        return True'),
    ("M35 a failed page load counts", CC, '    if m:\n        return m.group(1) == "OK"', '    if m:\n        return True'),
    ("M36 prose files in the ledger", CC, '    return [p for p in _written_paths_from_confirmation(raw) if _is_code(p)]', '    return list(_written_paths_from_confirmation(raw))'),
    ("M37 no run-before-claim rule", PR, 'RUN IT before you finish: execute the file or its tests', 'you may skip running it: execute the file or its tests'),
    ("M38 the note never fails the turn", CC, '    return (f"\\n\\n---\\n**⚠ Unverified claim:** this reply says \\"{claim}\\", but {what}. "\n            f"{verdict}"), True', '    return (f"\\n\\n---\\n**⚠ Unverified claim:** this reply says \\"{claim}\\", but {what}. "\n            f"{verdict}"), False'),
]
def run_tests(tests):
    env = dict(os.environ, PYTHONPATH="src", GHOST_API_KEY=open(os.path.expanduser("~/Data/AI/.ghost_api_key")).read().strip()); env.pop("FORCE_COLOR", None)
    try:
        r = subprocess.run([PY, "-m", "pytest", *tests, "-q", "--no-header", "-p", "no:cacheprovider", "-x", "-n", "8", "--dist", "loadfile"], capture_output=True, text=True, env=env, cwd=ROOT, timeout=1500)
    except subprocess.TimeoutExpired:
        return 1, "TIMEOUT (a mutant wedged the run — counted as killed)"
    return r.returncode, (r.stdout.strip().splitlines() or ["?"])[-1]
def _fast_tests(mutated_file, pins):
    mod = os.path.splitext(os.path.basename(mutated_file))[0]
    pat = mod + r"\b"
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py"))), capture_output=True, text=True).stdout.split()
    return sorted({os.path.relpath(x, ROOT) for x in out} | set(pins))

PINS = ["tests/test_4ph_coding_turn_review.py", "tests/test_execute_oom_vs_timeout.py", "tests/test_execute_0720_review_fixes.py",
        "tests/test_execute_self_reported_exit.py", "tests/test_prompts_and_registry.py"]

def main():
    args = sys.argv[1:] or ["all"]
    if args[0] == "rerun":
        want = set(args[1:]); batch = [m for m in MUT if m[0].split(" ")[0] in want or m[0].startswith("CONTROL")]
    else:
        batch = MUT
    full = _tests(); files = sorted({m[1] for m in batch}); backups = {f: f + ".4ph.bak" for f in files}
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
