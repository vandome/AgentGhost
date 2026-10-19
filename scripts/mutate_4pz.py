"""§4PZ battery — "show proposed rules" + the see-verbs hint (2026-10-18). Harness of §4PG.
In place, backup + restore + byte compare; one mutant per run; py_compile first; the related test set;
no-op control must SURVIVE, a known-bad control must DIE.    python3 scripts/mutate_4pz.py [all|rerun M1 M2 …]
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
    ("CONTROL no-op (must SURVIVE)", FR, "def _list_proposed(entries: list) -> \"RuleNote\":", "# ctl\ndef _list_proposed(entries: list) -> \"RuleNote\":"),
    ("CONTROL known-bad (must DIE): proposed list never dispatched", FR, '    if _LIST_PROPOSED_RE.match(t):\n        return _list_proposed(entries)', '    if False:\n        return _list_proposed(entries)'),
    ("M1 not a rule command (goes to the model)", FR, "    return bool(_LIST_RE.match(t) or _LIST_PROPOSED_RE.match(t) or", "    return bool(_LIST_RE.match(t) or"),
    ("M2 'proposed' not a list word", FR, '_PROPOSED_ADJ = r"(?:proposed|candidate|pending|waiting|suggested)"', '_PROPOSED_ADJ = r"(?:candidate|pending|waiting|suggested)"'),
    ("M3 list regex unanchored at the end", FR, '_LIST_PROPOSED_RE = re.compile(_LEAD + r"(?:show|list)\\s+(?:me\\s+)?" + _PROPOSED_RULES\n                               + r"[\\s\\"\'“”‘’«».!?]*$", re.I)', '_LIST_PROPOSED_RE = re.compile(_LEAD + r"(?:show|list)\\s+(?:me\\s+)?" + _PROPOSED_RULES\n                               , re.I)'),
    ("M4 list shows every non-adopted rule", FR, 'for e in entries if is_proposed(e)]\n    if not rows:', 'for e in entries if e.get("rule") and not is_adopted(e)]\n    if not rows:'),
    ("M5 rule cut at 160", FR, "{_dfz(e['rule'])[:400]}\" for e in entries if is_proposed(e)]", "{_dfz(e['rule'])[:160]}\" for e in entries if is_proposed(e)]"),
    ("M6 rule not capped", FR, "{_dfz(e['rule'])[:400]}\" for e in entries if is_proposed(e)]", "{_dfz(e['rule'])}\" for e in entries if is_proposed(e)]"),
    ("M7 rule not defused", FR, "{_dfz(e['rule'])[:400]}\" for e in entries if is_proposed(e)]", "{e['rule'][:400]}\" for e in entries if is_proposed(e)]"),
    ("M8 oldest twenty listed", FR, '"\\n".join(rows[-20:]))\n        if len(rows) > 20:', '"\\n".join(rows[:20]))\n        if len(rows) > 20:'),
    ("M9 older not counted", FR, '            b += f"\\n(and {len(rows) - 20} older — “show rule N” shows one)"', '            pass'),
    ("M10 empty list without the pointer", FR, '        b += " “show all rules” lists every rule."', '        pass'),
    ("M11 bare 'show rules' is the command again (r3)", FR, '_LIST_RE = re.compile(_LEAD + r"(?:show|list)\\s+(?:me\\s+)?" + _WHICH_RULES.format(w="adopted")', '_LIST_RE = re.compile(_LEAD + r"(?:show|list)\\s+(?:me\\s+)?(?:all\\s+)?(?:my\\s+|the\\s+)?"'),
    ("M12 hint verbs lack show/list/see", FR, '                           r"show|list|see)\\b", re.I)', '                           r"xx)\\b", re.I)'),
    ("M13 hint verbs gain what/which again", FR, '                           r"show|list|see)\\b", re.I)', '                           r"show|list|see|which|what)\\b", re.I)'),
    ("M14 hint lacks 'see'", FR, '                           r"show|list|see)\\b", re.I)', '                           r"show|list)\\b", re.I)'),
    ("M15 bulk dismiss loses 'proposed'", FR, '_PROPOSED_ADJ = r"(?:proposed|candidate|pending|waiting|suggested)"', '_PROPOSED_ADJ = r"(?:candidate|pending|waiting|suggested|proposed(?=\\s+rules\\s+xx))"'),
    ("M17 the list without its actions", FR, '        b = ("**Proposed, waiting for you** (say “learn rule N” to adopt one, “dismiss rule N” or "\n             "“dismiss all proposed rules” to drop them):\\n"', '        b = ("**Proposed, waiting for you**:\\n"'),
    ("M20 Ghost-only nouns not enough alone (r1)", FR, '    if _GHOST_RULE_NOUN_RE.search(t):\n        return RULE_HINT', '    if False:\n        return RULE_HINT'),
    ("M21 'rules did you suggest' not a Ghost noun", FR, 'r"(?:proposed|suggested|made|wrote)\\b|\\brules?\\s+did\\s+(?:you|ghost)\\s+"', 'r"(?:proposed|suggested|made|wrote)\\b|\\bxxrules?\\s+did\\s+(?:you|ghost)\\s+"'),
    ("M22 'my pending rules' not a Ghost noun", FR, '_GHOST_RULE_NOUN_RE = re.compile(r"\\b(?:my|your)\\s+" + _PROPOSED_ADJ', '_GHOST_RULE_NOUN_RE = re.compile(r"\\bxx(?:my|your)\\s+" + _PROPOSED_ADJ'),
    ("M23 hint text lacks the new command", FR, '"“show me all proposed rules”, “show rule N”,', '"“show rule N”,'),
    ("M24 rules under test not named", FR, '        if testing:\n            b += (', '        if False:\n            b += ('),
    ("M25 abandoned counted as under test", FR, 'if e.get("rule") and e.get("stage") not in ("done", "abandoned")]', 'if e.get("rule") and e.get("stage") != "done"]'),
    ("M26 singular always", FR, "            b += (f\" Rule{'s' if len(testing) > 1 else ''} {_few(testing)} \"", "            b += (f\" Rule {_few(testing)} \""),
    ("M16 'your' is the command again (r2)", FR, "(?:all\\s+(?:of\\s+)?(?:the\\s+|my\\s+)?|my\\s+|", "(?:all\\s+(?:of\\s+)?(?:the\\s+|my\\s+|your\\s+)?|my\\s+|your\\s+|"),
    ("M18 bare 'the' is the command again (r1/r3)", FR, '_WHICH_RULES = r"(?:all\\s+(?:of\\s+)?(?:the\\s+|my\\s+)?|my\\s+|(?:the\\s+)?(?={w}\\s))"', '_WHICH_RULES = r"(?:all\\s+(?:of\\s+)?(?:the\\s+|my\\s+)?|my\\s+|(?:the\\s+)?)"'),
    ("M19 bare Ghost word not the command (r2)", FR, '_WHICH_RULES = r"(?:all\\s+(?:of\\s+)?(?:the\\s+|my\\s+)?|my\\s+|(?:the\\s+)?(?={w}\\s))"', '_WHICH_RULES = r"(?:all\\s+(?:of\\s+)?(?:the\\s+|my\\s+)?|my\\s+|(?!){w})"'),
    ("M27 delete is a command again (r2)", FR, '(learn|adopt|show|forget|dismiss|reject)\\s+', '(learn|adopt|show|forget|dismiss|reject|delete|remove)\\s+'),
    ("M28 'rules have you proposed' not a Ghost noun (r2)", FR, '|\\brules?\\s+(?:have|has)\\s+(?:you|ghost)\\s+"', '|\\bxxrules?\\s+(?:have|has)\\s+(?:you|ghost)\\s+"'),
    ("M29 'rules waiting for me' not a Ghost noun (r2)", FR, '|\\brules?\\s+(?:are\\s+)?waiting\\s+for\\s+me\\b|', '|\\bxxrules?\\s+(?:are\\s+)?waiting\\s+for\\s+me\\b|'),
    ("M34 bare list wordings get no note (r3)", FR, 'r"^[\\s\\"\'“”‘’«»]*(?:show|list)\\s+(?:me\\s+)?(?:the\\s+)?rules[\\s\\"\'“”‘’«».!?]*$|\\brule', 'r"\\brule'),
    ("M35 the Ghost-word lookahead dropped for adopted (r3)", FR, '_LIST_RE = re.compile(_LEAD + r"(?:show|list)\\s+(?:me\\s+)?" + _WHICH_RULES.format(w="adopted")', '_LIST_RE = re.compile(_LEAD + r"(?:show|list)\\s+(?:me\\s+)?" + _WHICH_RULES.format(w="xx")'),
    ("M33 legend lacks the new command", "src/ghost_agent/tools/memory.py", '“show all rules”, “show me all proposed "\n                 "rules”, “learn rule N”', '“show all rules”, “learn rule N”'),
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

PINS = ["tests/test_4pz_list_proposed_rules.py", "tests/test_4py_system_command_reply.py", "tests/test_4px_dismiss_rules.py", "tests/test_4mx_rule_commands.py"]

def main():
    args = sys.argv[1:] or ["all"]
    if args[0] == "rerun":
        want = set(args[1:]); batch = [m for m in MUT if m[0].split(" ")[0] in want or m[0].startswith("CONTROL")]
    else:
        batch = MUT
    full = _tests(); files = sorted({m[1] for m in batch}); backups = {f: f + ".4pz.bak" for f in files}
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
