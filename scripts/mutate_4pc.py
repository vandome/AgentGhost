#!/usr/bin/env python3
"""§4PC battery — browser + search→page hand-off review fixes (2026-10-14).
In place, backup + restore + byte compare; one mutant per run; py_compile
first; the full browser-related test set; no-op control must SURVIVE, a
known-bad control must DIE.    python3 scripts/mutate_4pc.py [a|b|all]
"""
import os, shutil, subprocess, sys, signal, filecmp, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/Users/vasilis/Data/AI/.agent.venv/bin/python"
R = "src/ghost_agent/tools/browser_runner.py"; B = "src/ghost_agent/tools/browser.py"
BS = "src/ghost_agent/tools/browser_support.py"; HM = "src/ghost_agent/tools/host_memo.py"
FM = "src/ghost_agent/core/fetch_memo.py"; AG = "src/ghost_agent/core/agent.py"; PR = "src/ghost_agent/core/prompts.py"
def _tests():
    out = subprocess.run(["grep", "-l", r"browser\|host_memo\|fetch_memo\|browser_runner\|system3\|System 3"] + sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py"))),
                         capture_output=True, text=True).stdout.split()
    return [os.path.relpath(t, ROOT) for t in out] + ["tests/test_4pc_browser_review.py", "tests/test_system3_crisis_pivot.py"]
A = [
    ("CONTROL no-op (must SURVIVE)", BS, "def _links_block(p: dict) -> str:", "# ctl\ndef _links_block(p: dict) -> str:"),
    ("CONTROL known-bad (must DIE): parser accepts any OK line", BS,
     'ok_tag = f"[BROWSER_OK:{nonce}] " if nonce else "[BROWSER_OK] "', 'ok_tag = "[BROWSER_OK] "'),
    ("M1 error message keeps its newlines", R,
     '    flat = " | ".join(part for part in str(msg).replace("\\r", "").split("\\n") if part.strip())',
     '    flat = str(msg)'),
    ("M2 nonce not validated", R, '    _NONCE = _n if _n.isalnum() and 8 <= len(_n) <= 64 else ""', '    _NONCE = _n'),
    ("M3 service workers allowed", R, '            service_workers="block",', ''),
    ("M4 websocket guard connects everything", R,
     '        if _ssrf_should_block(hu, sandbox_root, anonymous):', '        if False:'),
    ("M5 redirect violation ignored", R, '            if _SSRF_VIOLATIONS:\n                raise RuntimeError(', '            if False:\n                raise RuntimeError('),
    ("M6 no op deadline", R,
     '                result = await asyncio.wait_for(op_fn(page), timeout=max(1.0, timeout_ms / 1000.0 + 5.0))',
     '                result = await op_fn(page)'),
    ("M7 implicit goto status dropped", R,
     '            **({"initial_goto": initial_goto} if initial_goto else {}),', ''),
    ("M8 click status dropped", R, '                  "status": _cstatus}', '                  }'),
    ("M9 offset ignored", R, '        if offset:\n            text = text[offset:]', '        if False:\n            text = text[offset:]'),
    ("M10 links not collected on navigate", R, '        result["links"] = await _page_links(page)\n', ''),
    ("M11 url-less fallback reads the global sidecar again", R,
     '    for name in dict.fromkeys((_LAST_URL_FILENAME, _LAST_URL_CONV_FILENAME)):\n        if not name:\n            continue\n        try:\n            p = os.path.join(profile_dir, name)',
     '    for name in dict.fromkeys((_LAST_URL_FILENAME, _LAST_URL_CONV_FILENAME, ".last_url")):\n        if not name:\n            continue\n        try:\n            p = os.path.join(profile_dir, name)'),
]
Bm = [
    ("M12 tool: implicit goto not consulted", B,
     '        if _last_goto is None and isinstance(parsed.get("initial_goto"), dict):', '        if False:'),
    ("M13 tool: infra banner not latched", B, '    if "[SANDBOX INFRA ERROR" in (output or ""):', '    if False:'),
    ("M14 tool: latch not checked before exec", B, '        if _sandbox_down_recently():\n            return _sandbox_down_result()', '        pass'),
    ("M15 tool: kill note dropped", B, '                      if exit_code in (124, 137) else "")', '                      if False else "")'),
    ("M16 tool: cap note dropped", B, '        if _asked_chars > _MAX_TEXT_CHARS:', '        if False:'),
    ("M17 tool: clean click no longer clears the memo", B,
     '    elif ok and operation in ("navigate", "extract_text", "click", "screenshot", "interact"):',
     '    elif ok and operation in ("navigate", "extract_text"):'),
    ("M18 support: latch is global, not per request", BS, '    return bool(k and t is not None and (time.monotonic() - t) < _SANDBOX_DOWN_TTL_S)',
     '    return bool(_SANDBOX_DOWN)'),
    ("M19 memo: a block needs two strikes again", HM, '_HOST_BLOCK_STRIKES = 1', '_HOST_BLOCK_STRIKES = 2'),
    ("M20 memo: probes write it", HM, '        if is_probe_request_id(request_id_context.get() or ""):\n            return', '        pass'),
    ("M21 fetch memo files every failure", FM, '    if not fk or memo is None or not page_level_failure(text):', '    if not fk or memo is None:'),
    ("M22 brake: given urls refused too", AG, '            if h and h in self.hosts_404 and not self.is_given(u):', '            if h and h in self.hosts_404:'),
    ("M23 brake: a given 404 marks the host", AG, '            if not self.is_given(u):\n                h = self.host(u)', '            if True:\n                h = self.host(u)'),
    ("M24 brake: never fires", AG, '                if _brake:\n                    pretty_log("Composed URL"', '                if False:\n                    pretty_log("Composed URL"'),
    ("M25 loop breaker: browser steer dropped", AG, '                            elif _fail_fname == "browser":', '                            elif False:'),
    ("M26 planner prompt: tool rule dropped", PR, 'Every step must use ONLY a tool from that list by its exact name (never invent a tool such as google_search); ', ''),
    ("M27 planner message: no tool list", AG, '            "### TOOLS THE AGENT HAS (use these exact names, nothing else): "\n            + (", ".join(names) if names else "(none listed — propose steps that need no tool)"))',
     '            "")'),
    ("M28 hints: infra route dropped", BS, '    if "SANDBOX INFRA ERROR" in e:', '    if False:'),
    ("M29 brake: result urls booked as given BEFORE the 404 (the round-1 defect)", AG,
     '                            if fname == "browser" and _res_is_error:\n                                _prov.note_failure(_recorded_args if isinstance(_recorded_args, dict) else {}, str_res)\n                            _prov.note_text(str_res)\n',
     '                            _prov.note_text(str_res)\n                            if fname == "browser" and _res_is_error:\n                                _prov.note_failure(_recorded_args if isinstance(_recorded_args, dict) else {}, str_res)\n'),
]
def run_tests(tests):
    env = dict(os.environ, PYTHONPATH="src", GHOST_API_KEY=open(os.path.expanduser("~/Data/AI/.ghost_api_key")).read().strip()); env.pop("FORCE_COLOR", None)
    r = subprocess.run([PY, "-m", "pytest", *tests, "-q", "--no-header", "-p", "no:cacheprovider", "-x", "-n", "8", "--dist", "loadfile"], capture_output=True, text=True, env=env, cwd=ROOT)
    return r.returncode, (r.stdout.strip().splitlines() or ["?"])[-1]
def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    batch = {"a": A, "b": Bm, "all": A + Bm, "rerun": [m for m in A + Bm if m[0].startswith(("CONTROL", "M9 ", "M10 ", "M11 "))], "m25": [m for m in A + Bm if m[0].startswith("M25 ")]}[which]
    tests = _tests(); files = sorted({m[1] for m in batch}); backups = {f: f + ".4pc.bak" for f in files}
    for f, b in backups.items(): shutil.copy2(os.path.join(ROOT, f), os.path.join(ROOT, b))
    def restore(*_):
        for f, b in backups.items(): shutil.copy2(os.path.join(ROOT, b), os.path.join(ROOT, f))
        subprocess.run(["find", os.path.join(ROOT, "src"), "-name", "__pycache__", "-exec", "rm", "-rf", "{}", "+"], capture_output=True)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP): signal.signal(sig, lambda *a: (restore(), sys.exit(130)))
    killed = survived = 0
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
