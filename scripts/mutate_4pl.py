#!/usr/bin/env python3
"""§4PL battery — the silent failure census (core/failure_census.py), 2026-10-16.

Two-tier harness as §4PG–§4PI, but on a COPY of the tree (`BATTERY_ROOT`), not
in place: the live agent is serving the §4PK replays from the real tree, and
a restart mid-battery would load a mutant. One mutant per run; py_compile
first; per mutant the PIN file + the tests naming the hold (fast), survivors
re-judged on the full selection; the no-op control must SURVIVE (full set),
the known-bad control must DIE.

    BATTERY_ROOT=/path/to/copy python3 scripts/mutate_4pk.py [all|rerun M1 …]
"""
import filecmp, glob, os, shutil, signal, subprocess, sys

ROOT = os.environ.get("BATTERY_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/Users/vasilis/Data/AI/.agent.venv/bin/python"
FC = "src/ghost_agent/core/failure_census.py"
QB = "src/ghost_agent/core/quote_bound.py"
AG = "src/ghost_agent/core/agent.py"
RT = "src/ghost_agent/api/routes.py"
LG = "src/ghost_agent/utils/logging.py"
PINS = ["tests/test_4pl_failure_census.py", "tests/test_4oj_quote_bound.py"]
_FULL_PAT = r"failure_census\|quote_bound\|_finalize_and_return"

MUT = [
    ("CONTROL no-op (must SURVIVE)", FC, "def _norm(s: str) -> str:", "# ctl\ndef _norm(s: str) -> str:"),
    ("CONTROL known-bad (must DIE): no turn is the owner's", FC, "    return str(ex.get(\"origin\") or \"user\") in (\"user\", \"\")", "    return False"),
    ("M1 probes counted", FC, "    if str(ex.get(\"requester_role\") or \"\") == \"member\" or rid.startswith(_NOT_OWNER_PREFIXES):\n        return False\n    return str(ex.get", "    if str(ex.get(\"requester_role\") or \"\") == \"member\":\n        return False\n    return str(ex.get"),
    ("M2 probes written", FC, "        return bool(rid) and not requester_is_member() and not rid.startswith(_NOT_OWNER_PREFIXES)", "        return bool(rid) and not requester_is_member()"),
    ("M3 members written", FC, "        return bool(rid) and not requester_is_member() and not rid.startswith(_NOT_OWNER_PREFIXES)", "        return bool(rid) and not rid.startswith(_NOT_OWNER_PREFIXES)"),
    ("M4 missing cut to 3", FC, "            row[\"missing\"] = [_redact(m) for m in miss[:20]]", "            row[\"missing\"] = [_redact(m) for m in miss[:3]]"),
    ("M5 missing not redacted", FC, "            row[\"missing\"] = [_redact(m) for m in miss[:20]]", "            row[\"missing\"] = [str(m)[:200] for m in miss[:20]]"),
    ("M6 bools stored as counts", FC, "            if isinstance(stats.get(k), (int, float)) and not isinstance(stats.get(k), bool):", "            if isinstance(stats.get(k), (int, float)):"),
    ("M7 no rotation", FC, "        if p.exists() and p.stat().st_size > LEDGER_MAX_BYTES:", "        if False:"),
    ("M8 dict rows reach the sniffer unwrapped", FC, "            failed = tool_call_failed(ns) and not is_designed_stop(tc.get(\"result\") or \"\")", "            failed = tool_call_failed(tc)"),
    ("M9 designed stops are failures", FC, "            failed = tool_call_failed(ns) and not is_designed_stop(tc.get(\"result\") or \"\")", "            failed = tool_call_failed(ns)"),
    ("M10 judge refutes counted", FC, "        and (str(v.get(\"route\") or \"\") in _MECH_ROUTES or v.get(\"override\"))\n", ""),
    ("M11 the vision model counted", FC, "        and str(v.get(\"override\") or \"\") not in _MODEL_OVERRIDES\n", ""),
    ("M12 not-measured reads as quiet (quote)", FC, "        out[\"quote_unverified\"] = None", "        out[\"quote_unverified\"] = False"),
    ("M13 tool turns in the no-tool population", FC, "    if ran:\n        for k in _NO_TOOL:\n            out[k] = None", "    if False:\n        for k in _NO_TOOL:\n            out[k] = None"),
    ("M14 history ignores earlier tool results", FC, "                              str(traj.get(\"system_prompt\") or \"\") + \" \" + history_context, False, tool_names)", "                              str(traj.get(\"system_prompt\") or \"\"), False, tool_names)"),
    ("M15 live row ignored", FC, "    elif turn_row is not None:                       # LIVE", "    elif False:                       # LIVE"),
    ("M16 system notes read as the reply", FC, "            reply = strip_system_notes(reply)", "            pass"),
    ("M17 undelivered turns counted", FC, "            if not delivered(t):\n                undelivered += 1\n                continue", "            if False:\n                undelivered += 1\n                continue"),
    ("M18 the window off by one", FC, "    window = [(today - datetime.timedelta(days=k)).isoformat() for k in range(int(days))]", "    window = [(today - datetime.timedelta(days=k)).isoformat() for k in range(int(days) + 1)]"),
    ("M19 quiet over all turns", FC, "        quiet = [r for r in pop if not r[\"det\"][name]]", "        quiet = [r for r in rows if not r[\"det\"][name]]"),
    ("M20 rate over all turns", FC, "        entry = {\"fired\": len(fired), \"of\": len(pop), \"rate\": round(len(fired) / len(pop), 3) if pop else None,", "        entry = {\"fired\": len(fired), \"of\": len(pop), \"rate\": round(len(fired) / len(rows), 3) if rows else None,"),
    ("M21 judge labels shown as the owner's", FC, "    return \"judge\" if reactions.get(tid) in (True, False) else \"\"", "    return \"human\" if reactions.get(tid) in (True, False) else \"\""),
    ("M22 the late verifier counts as the owner", FC, "             if str(c.get(\"source\") or \"\").startswith((\"human_feedback\", \"user_correction\"))]", "             if str(c.get(\"source\") or \"\")]"),
    ("M23 oldest human label wins", FC, "        last = sorted(human, key=lambda c: str(c.get(\"timestamp\") or c.get(\"at\") or \"\"))[-1]", "        last = sorted(human, key=lambda c: str(c.get(\"timestamp\") or c.get(\"at\") or \"\"))[0]"),
    ("M29 the owner's 👎 is the machine label", FC, "    if corrections:\n        return None                     # only human labels", "    if False:\n        return None                     # only human labels"),
    ("M30 the machine label reads the overlay", FC, "        d[\"outcome_failed\"] = machine_failed(t, corrections.get(tid, []))", "        d[\"outcome_failed\"] = str(t.get(\"outcome\") or \"\").lower() == \"failed\""),
    ("M24 finalize writes nothing", AG, "        await self._census_record(req_id, final_ai_content, last_user_content, messages,\n                                  tools_run_this_turn, _fc_quote_stats, path=\"finalize\")", "        pass"),
    ("M25 the quote stats never reach the row", AG, "                _fc_quote_stats = _qb_stats", "                pass"),
    ("M26 the prompt stash ignored", AG, "            ctx_parts = [stash.get(rid, \"\")] if isinstance(stash, dict) else []", "            ctx_parts = []"),
    ("M27 the streamed filter result dropped", QB, "                        on_result({\"markers\": len(filt.markers), \"failed\": len(claims)})", "                        pass"),
    ("M28 rows keyed on a different id", AG, "            rid = str(req_id or request_id_context.get() or \"\")    # the trajectory's req_id = the join key", "            rid = \"x-\" + str(req_id or \"\")"),
]


def _grep(pat):
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py"))),
                         capture_output=True, text=True).stdout.split()
    return {os.path.relpath(t, ROOT) for t in out}


def run_tests(tests):
    env = dict(os.environ, PYTHONPATH="src", GHOST_API_KEY="x")
    env.pop("FORCE_COLOR", None)
    try:
        r = subprocess.run([PY, "-m", "pytest", *tests, "-q", "--no-header", "-p", "no:cacheprovider", "-x",
                            "-n", "8", "--dist", "loadfile"], capture_output=True, text=True, env=env, cwd=ROOT, timeout=1500)
    except subprocess.TimeoutExpired:
        return 1, "TIMEOUT (a mutant wedged the run — counted as killed)"
    return r.returncode, (r.stdout.strip().splitlines() or ["?"])[-1]


def main():
    args = sys.argv[1:] or ["all"]
    if args[0] == "rerun":
        want = set(args[1:])
        batch = [m for m in MUT if m[0].split(" ")[0] in want or m[0].startswith("CONTROL")]
    else:
        batch = MUT
    fast = sorted(_grep("failure_census") | set(PINS))
    full = sorted(_grep(_FULL_PAT) | set(PINS))
    files = sorted({m[1] for m in batch})
    backups = {f: f + ".4pk.bak" for f in files}
    for f, b in backups.items():
        shutil.copy2(os.path.join(ROOT, f), os.path.join(ROOT, b))

    def restore(*_):
        for f, b in backups.items():
            shutil.copy2(os.path.join(ROOT, b), os.path.join(ROOT, f))
        subprocess.run(["find", os.path.join(ROOT, "src"), "-name", "__pycache__", "-exec", "rm", "-rf", "{}", "+"],
                       capture_output=True)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *a: (restore(), sys.exit(130)))
    killed = survived = 0
    print(f"ROOT={ROOT}; fast {len(fast)} files, full {len(full)} files; {len(batch)} mutants", flush=True)
    try:
        for label, f, old, new in batch:
            restore()
            path = os.path.join(ROOT, f)
            src = open(path).read()
            if src.count(old) != 1:
                print(f"SKIP     {label}: anchor count {src.count(old)}", flush=True)
                continue
            open(path, "w").write(src.replace(old, new))
            if subprocess.run([PY, "-m", "py_compile", path], capture_output=True).returncode != 0:
                print(f"NOCOMP   {label}", flush=True)
                continue
            sel = full if label.startswith("CONTROL no-op") else fast
            rc, tail = run_tests(sel)
            tier = f"fast {len(sel)}"
            if rc == 0 and sel is not full:
                rc, tail = run_tests(full)
                tier += " → full"
            status = "SURVIVED" if rc == 0 else "KILLED"
            killed += rc != 0
            survived += rc == 0
            print(f"{status:8} {label} — [{tier}] {tail}", flush=True)
    finally:
        restore()
        ok = all(filecmp.cmp(os.path.join(ROOT, f), os.path.join(ROOT, b), shallow=False) for f, b in backups.items())
        for b in backups.values():
            os.remove(os.path.join(ROOT, b))
        print(f"RESTORED identical={ok} killed={killed} survived={survived}", flush=True)


if __name__ == "__main__":
    main()
