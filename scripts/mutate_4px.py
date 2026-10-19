"""§4PX battery — dismiss proposed rules + the no-command hint (2026-10-17). Harness of §4PG.
In place, backup + restore + byte compare; one mutant per run; py_compile first; the related test set;
no-op control must SURVIVE, a known-bad control must DIE.    python3 scripts/mutate_4px.py [all|rerun M1 M2 …]
"""
import os, shutil, subprocess, sys, signal, filecmp, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/Users/vasilis/Data/AI/.agent.venv/bin/python"
FR = "src/ghost_agent/core/failure_replay.py"; AG = "src/ghost_agent/core/agent.py"
def _tests():
    pat = r"failure_replay\|owner_rule\|rule_command\|_owner_rule_note"
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py"))), capture_output=True, text=True).stdout.split()
    return sorted({os.path.relpath(t, ROOT) for t in out} | set(PINS))
MUT = [
    # ── r5: the r4 restructure ──
    ("M39 repeat confirm reads as failure", FR, '        if not hit:\n            if done:', '        if not hit:\n            if False:'),
    ("M42 already-done ignores the current state", FR, '                and float(e.get("dismissed_batch") or 0) == batch and rule_state(e) == "dismissed"]', '                and float(e.get("dismissed_batch") or 0) == batch]'),
    ("M43 already-done ignores who dismissed", FR, '                and float(e.get("dismissed_batch") or 0) == batch and rule_state(e) == "dismissed"]', '                and rule_state(e) == "dismissed"]'),
    ("M46 no-hit line invites a tool", FR, '            return _rn("Done by the system; do not call any tool. " + _SAID.format(b=b), b)\n        now = time.time()', '            return _rn(_SAID.format(b=b), b)\n        now = time.time()'),
    ("M47 changed rules not said beside already-done", FR, '                     f"dismissed." + (f" Left as it is: {\'; \'.join(kept)}." if kept else ""))', '                     f"dismissed.")'),
    # ── r4: the r3 fixes first ──
    ("M41 no room floor for open cases", FR, "        _room = max(min(OPEN_MIN, _MAX_ENTRIES // 15), _MAX_ENTRIES - len(_keep))", "        _room = _MAX_ENTRIES - len(_keep)"),
    ("M44 batch never stamped", FR, '            e["dismissed_batch"] = batch', '            pass'),
    ("M45 dismissed_batch not an owner field", FR, '"dismiss_batch", "dismissed_batch")', '"dismiss_batch")'),
    # ── r3: the r2 fixes first (R3) ──
    ("M25 every dismissed row kept", FR, "        _dis_ids = {id(e) for e in _dis[-DISMISSED_KEPT:]} |", "        _dis_ids = {id(e) for e in _dis} |"),
    ("M26 the oldest dismissed kept", FR, "        _dis_ids = {id(e) for e in _dis[-DISMISSED_KEPT:]} |", "        _dis_ids = {id(e) for e in _dis[:DISMISSED_KEPT]} |"),
    ("M37 no replay-window hold", FR, "if float(e.get(\"dismissed_at\") or 0) > _recent}", "if False}"),
    ("M31 bare 'new rules' is a rule noun again", FR, 'r"\\brules?\\s*#?\\s*\\d+\\b|\\b(?:proposed|candidate|pending|waiting|adopted|owner|suggested)"', 'r"\\brules?\\s*#?\\s*\\d+\\b|\\b(?:proposed|candidate|pending|waiting|adopted|owner|suggested|new|your)"'),
    ("M35 show offers learn only", FR, '(f", or “dismiss rule {n}” to set it aside." if is_proposed(case) else ".")))', '"."))'),
    ("M38 show offers dismiss under test", FR, '(f", or “dismiss rule {n}” to set it aside." if is_proposed(case) else ".")))', '(f", or “dismiss rule {n}” to set it aside." if rule_state(case) == "proposed" else ".")))'),
    ("M40 numbers never capped", FR, '    return ", ".join(nums[:k]) + (f" and {len(nums) - k} more" if len(nums) > k else "")', '    return ", ".join(nums)'),
    # ── r2: the r1 fixes first (R3) ──
    ("M23 dismiss_batch not an owner field", FR, '"dismissed_at", "dismiss_preview_at", "dismiss_batch", "dismissed_batch")', '"dismissed_at", "dismiss_preview_at", "dismissed_batch")'),
    ("M24 dismiss_preview_at not an owner field", FR, '"dismissed_at", "dismiss_preview_at", "dismiss_batch", "dismissed_batch")', '"dismissed_at", "dismiss_batch", "dismissed_batch")'),
    ("M27 shown-then-changed not reported", FR, '    if kept:            # r1', '    if False:            # r1'),
    ("M28 confirm-all without decline", FR, '_CONFIRM_ALL_RE = re.compile(_LEAD + r"confirm\\s+" + _DISMISS_VERBS', '_CONFIRM_ALL_RE = re.compile(_LEAD + r"confirm\\s+(?:forget|dismiss|reject|discard|drop|delete|remove|clear)\\s+"'),
    ("M29 suggested not a bulk noun", FR, '(?:proposed|candidate|pending|waiting|suggested)\\s+rules"', '(?:proposed|candidate|pending|waiting)\\s+rules"'),
    ("M30 hint misses 'rules you proposed'", FR, 'r"\\s+rules?\\b|\\ball\\s+(?:the\\s+|my\\s+)?rules\\b|\\brules?\\s+(?:you|ghost)\\s+"', 'r"\\s+rules?\\b|\\ball\\s+(?:the\\s+|my\\s+)?rules\\b|\\bxxrules?\\s+(?:you|ghost)\\s+"'),
    ("M32 hint misses 'rule proposals'", FR, '|\\brule\\s+(?:proposals?|suggestions?)\\b", re.I)', '", re.I)'),
    ("M33 hint redirects a question", FR, '"do not use it for this. If the owner asked a QUESTION, answer it (pointing to “show rule N” or "\n             "“show all rules” for the details); if they asked for a change, tell them', '"do not use it for this. Always tell them'),
    ("M34 refusal says dismiss to a forget", FR, '        return _confirm_dismiss(n, home, "forget")', '        return _confirm_dismiss(n, home)'),
    ("M36 preview refusal ignores the verb", FR, '        return _preview_dismiss(entries, case, n, home, "forget" if verb == "forget" else "dismiss")', '        return _preview_dismiss(entries, case, n, home)'),
    ("CONTROL no-op (must SURVIVE)", FR, "def rule_state(case: dict) -> str:", "# ctl\ndef rule_state(case: dict) -> str:"),
    ("CONTROL known-bad (must DIE): bulk dismiss never parses", FR, '    if _DISMISS_ALL_RE.match(t):\n        return _preview_dismiss_all(entries, home)', '    if False:\n        return _preview_dismiss_all(entries, home)'),
    ("M1 dismissed_at not an owner field (stale writer undoes)", FR, '"dismissed_at", "dismiss_preview_at", "dismiss_batch", "dismissed_batch")', '"dismiss_preview_at", "dismiss_batch", "dismissed_batch")'),
    ("M2 trim drops a dismissed rule", FR, '                                                        or id(e) in _dis_ids)', '                                                        )'),
    ("M3 dismissed never a state", FR, '    if float(case.get("dismissed_at") or 0) > float(case.get("forgotten_at") or 0):\n        return "dismissed"', '    if False:\n        return "dismissed"'),
    ("M4 forget on proposed is the old refusal", FR, '    if verb in ("dismiss", "reject") or (verb == "forget" and not is_adopted(case)):', '    if verb in ("dismiss", "reject"):'),
    ("M5 adopted rule dismissable", FR, '    if st == "adopted":\n        return f"Rule {n} is adopted, not proposed', '    if False:\n        return f"Rule {n} is adopted, not proposed'),
    ("M6 rule under test dismissable", FR, '    if case.get("stage") != "done":\n        return f"Rule {n} is still being tested', '    if False:\n        return f"Rule {n} is still being tested'),
    ("M7 single confirm without preview", FR, '        if time.time() - float(case.get("dismiss_preview_at") or 0) > FORGET_CONFIRM_S:\n            b = f"Say “dismiss rule {n}” first', '        if False:\n            b = f"Say “dismiss rule {n}” first'),
    ("M8 single confirm checks the state before the lock", FR, '        why = _not_dismissable(case, n, verb)\n        if why:\n            return _rn(_SAID.format(b=why), why)\n        if time.time()', '        if time.time()'),
    ("M9 bulk preview lists non-proposed", FR, '    props = [e for e in entries if is_proposed(e)]', '    props = [e for e in entries if e.get("rule") and not is_adopted(e)]'),
    ("M10 bulk confirm without a bulk preview", FR, '        if not batch or time.time() - batch > FORGET_CONFIRM_S:', '        if False:'),
    ("M11 bulk confirm dismisses every proposed", FR, '        hit = [e for e in shown if is_proposed(e)]', '        hit = [e for e in entries if is_proposed(e)]'),
    ("M12 bulk confirm dismisses an adopted-since", FR, '        hit = [e for e in shown if is_proposed(e)]', '        hit = list(shown)'),
    ("M13 late rules not reported", FR, '    if later:\n        b += (', '    if False:\n        b += ('),
    ("M14 single preview counts as the bulk", FR, '    case["dismiss_preview_at"] = time.time()\n    save(home, entries)\n    b = (f"Dismiss rule', '    case["dismiss_preview_at"] = case["dismiss_batch"] = time.time()\n    save(home, entries)\n    b = (f"Dismiss rule'),
    ("M15 confirm forget on proposed refuses", FR, '    if case is not None and not is_adopted(case):\n        return _confirm_dismiss(n, home, "forget")', '    if False:\n        return _confirm_dismiss(n, home, "forget")'),
    ("M16 confirm-all regex accepts bare rules", FR, '_CONFIRM_ALL_RE = re.compile(_LEAD + r"confirm\\s+" + _DISMISS_VERBS + _PROPOSED_RULES', '_CONFIRM_ALL_RE = re.compile(_LEAD + r"confirm\\s+" + _DISMISS_VERBS + r"(?:all\\s+)?(?:the\\s+)?rules|" + _PROPOSED_RULES'),
    ("M17 list has no dismissed group", FR, '                             ("Dismissed", "dismissed", "say “learn rule N” to adopt one")):', '                             ):'),
    ("M18 hint never given", AG, '                hint = rule_command_hint(text)', '                hint = ""'),
    ("M19 hint ignores the length cap", FR, '    if is_rule_command(t) or len(t.split()) > RULE_HINT_MAX_WORDS:', '    if is_rule_command(t):'),
    ("M20 hint without the noun", FR, '    return RULE_HINT if (_RULE_VERB_RE.search(t) and _RULE_NOUN_RE.search(t)) else ""', '    return RULE_HINT if _RULE_VERB_RE.search(t) else ""'),
    ("M21 hint for a command too", FR, '    if is_rule_command(t) or len(t.split()) > RULE_HINT_MAX_WORDS:', '    if len(t.split()) > RULE_HINT_MAX_WORDS:'),
    ("M22 show says proposed for dismissed", FR, '        state = {"adopted": "ADOPTED", "forgotten": "FORGOTTEN", "dismissed": "DISMISSED"}.get(', '        state = {"adopted": "ADOPTED", "forgotten": "FORGOTTEN"}.get('),
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

PINS = ["tests/test_4px_dismiss_rules.py", "tests/test_4mx_rule_commands.py"]

def main():
    args = sys.argv[1:] or ["all"]
    if args[0] == "rerun":
        want = set(args[1:]); batch = [m for m in MUT if m[0].split(" ")[0] in want or m[0].startswith("CONTROL")]
    else:
        batch = MUT
    full = _tests(); files = sorted({m[1] for m in batch}); backups = {f: f + ".4px.bak" for f in files}
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
