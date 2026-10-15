#!/usr/bin/env python3
"""§4PE battery — file_system / projects / knowledge_base / scratchpad / workspace / services review fixes (2026-10-15).
In place, backup + restore + byte compare; one mutant per run; py_compile first; the related test set;
no-op control must SURVIVE, a known-bad control must DIE.    python3 scripts/mutate_4pe.py [all|rerun M1 M2 …]
"""
import os, shutil, subprocess, sys, signal, filecmp, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/Users/vasilis/Data/AI/.agent.venv/bin/python"
FS = "src/ghost_agent/tools/file_system.py"; PJ = "src/ghost_agent/tools/projects.py"; MM = "src/ghost_agent/tools/memory.py"
PV = "src/ghost_agent/utils/provenance.py"; ST = "src/ghost_agent/core/strikes.py"; WS = "src/ghost_agent/tools/workspace.py"
WT = "src/ghost_agent/tools/workspace_track.py"; WM = "src/ghost_agent/workspace/model.py"
SV = "src/ghost_agent/tools/sandbox_services.py"; SS = "src/ghost_agent/sandbox/services.py"
def _tests():
    pat = r"file_system\|tools\.projects\|tools import projects\|tool_knowledge_base\|forget_execute\|forget_preview\|_resolve_plan\|user_confirms\|tool_scratchpad\|workspace_track\|tools\.workspace\|workspace\.model\|sandbox_services\|manage_services\|provenance\|strikes\|_declare_ingest\|tool_gain_knowledge\|render_document_outline\|_protected_dir_refusal\|_released_write_block\|tool_download_file\|_get_safe_path"
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py"))), capture_output=True, text=True).stdout.split()
    return sorted({os.path.relpath(t, ROOT) for t in out} | {"tests/test_4pe_files_projects_review.py"})
MUT = [
    ("CONTROL no-op (must SURVIVE)", MM, "def plan_miss_reason() -> str:", "# ctl\ndef plan_miss_reason() -> str:"),
    ("CONTROL known-bad (must DIE): yes lexicon accepts anything", MM, '    if "?" in text or ";" in text:\n        return False', '    return True'),
    # ── class A: confirm flow ──
    ("M1 fallback no longer skips this request's own preview", MM, 'cands = [(t, p) for t, p in live if p.get("rid") != rid_now and (match is None or match(p))]', 'cands = [(t, p) for t, p in live if (match is None or match(p))]'),
    ("M2 projects: bare call in the yes turn previews again", PJ, '        if _plan0 is not None:\n            pretty_log("Confirm Lookup", f"{act} {rid}', '        if False:\n            pretty_log("Confirm Lookup", f"{act} {rid}'),
    ("M3 kb forget: bare call in the yes turn previews again", MM, '        if _plan0 is not None:\n            pretty_log("Confirm Lookup", f"forget {subject!r}', '        if False:\n            pretty_log("Confirm Lookup", f"forget {subject!r}'),
    ("M4 reset_all: bare call in the yes turn previews again", MM, '            if _plan0 is None:\n                return await asyncio.to_thread(_reset_preview', '            if True:\n                return await asyncio.to_thread(_reset_preview'),
    ("M5 token not bound to its conversation", MM, '        if conv and plan.get("conv") and plan.get("conv") != conv:\n            _PLAN_MISS.set("that token belongs to another conversation\'s preview")\n            return token, None', '        if False:\n            return token, None'),
    ("M6 a question confirms", MM, '    if "?" in text or ";" in text:\n        return False', '    pass'),
    ("M7 a reservation confirms the whole list", MM, '            and (narrowed or not _RESERVATION_RE.search(text)))', '            and True)'),
    ("M8 miss reason dropped from the projects error", PJ, '        _why = (plan_miss_reason() if plan is None', '        _why = ("" if plan is None'),
    ("M9 list_docs reads as outside content again", PV, '_KB_NON_READS = frozenset({"insert_fact", "forget", "reset_all", "list_docs",', '_KB_NON_READS = frozenset({"insert_fact", "forget",'),
    # ── class B: knowledge base + scratchpad ──
    ("M10 bg scratchpad set overwrites the owner's key", MM, '            if _owned:\n                return ToolOutcome.rejected(', '            if False:\n                return ToolOutcome.rejected('),
    ("M11 runtime counters not reserved", MM, '"_session_", "_promotion_")   # §4PE', ')   # §4PE'),
    ("M12 ingest: Error-headed page refused again", MM, '            if isinstance(full_text, _FetchError): return str(full_text)', '            if str(full_text).startswith("Error"): return str(full_text)'),
    ("M13 forget_execute report undeclared", MM, '    if not _bad:\n        return ToolOutcome.ok(text, world_changed=True)', '    return text\n    if not _bad:\n        return ToolOutcome.ok(text, world_changed=True)'),
    ("M14 partial ingest heads read as OK", MM, "            head.upper().startswith(\"SUCCESS (PARTIAL)\")          # §4PE (K8): the audio route's lowercase head", "            head.startswith(\"SUCCESS (PARTIAL)\")"),
    ("M15 audio outline rows dropped again", MM, '            page = int(row[2]) if len(row) > 2 and row[2] is not None else None   # §4PE (K6): audio rows carry none', '            page = int(row[2])'),
    ("M16 passage text not neutralised", MM, "                     f\"---\\n{neutralise_kb_text(h['text'])}\")", "                     f\"---\\n{h['text']}\")"),
    ("M17 strike counter keys on the marker anywhere", ST, '    return any(ln.lstrip().startswith(("⚠ " + KB_NO_ANSWER_MARKER, KB_NO_ANSWER_MARKER))\n               for ln in str(text or "").splitlines())', '    return KB_NO_ANSWER_MARKER in text'),
    # ── class C: file_system ──
    ("M18 protected-dir guard judges against the scoped dir again", FS, 'rel = tuple(_fold(p) for p in path.resolve().relative_to(_outer_root(sandbox_dir).resolve()).parts)', 'rel = tuple(_fold(p) for p in path.resolve().relative_to(Path(sandbox_dir).resolve()).parts)'),
    ("M19 fold is lower()", FS, '    return unicodedata.normalize("NFKC", str(name)).casefold()', '    return str(name).lower()'),
    ("M20 released guard lowercases again", FS, '            low = [_fold(p) for p in parts]                 # §4PE: NFKC+casefold, not lower()', '            low = [str(p).lower() for p in parts]'),
    ("M21 inspect opens a FIFO", FS, '        _nr = _nonregular_refusal(path, filename)       # §4PE (F3): a FIFO wedged the loop here\n        if _nr:\n            return _nr', '        pass'),
    ("M22 read_chunked opens a FIFO", FS, '        _nr = _nonregular_refusal(path, filename)       # §4PE (F3): a FIFO hung the request\'s thread\n        if _nr:\n            return _nr', '        pass'),
    ("M23 download writes through the target", FS, '            try:\n                os.replace(tmp, target_path)\n            except OSError as oe:\n                raise _LocalSaveError(f"{oe.strerror or oe} — \'{target_path.name}\'") from oe\n            tmp = None', '            shutil.copyfile(tmp, target_path)'),
    ("M24 local save error reported as the site refusing", FS, '        except _LocalSaveError as le:', '        except () as le:'),
    ("M25 receipt without size/type", FS, '    return f"SUCCESS: Downloaded \'{url}\' to \'{filename}\' ({total:,} bytes, {ct})."', '    return f"SUCCESS: Downloaded \'{url}\' to \'{filename}\'."'),
    ("M26 download not stamped as written", FS, '    try:\n        _mark_written(target_path)\n    except Exception:  # noqa: BLE001\n        pass\n    ct = ', '    ct = '),
    ("M27 NUL byte passes to resolve()", FS, '    if "\\x00" in str(filename or ""):\n        raise ValueError("Security Error: the path contains a NUL byte', '    if False:\n        raise ValueError("Security Error: the path contains a NUL byte'),
    ("M28 held payload not re-held", FS, '                _hold_rejected_content(sandbox_dir, target_path, str(final_content))', '                pass'),
    ("M29 httpd.conf is a URL again", FS, '    if path and re.match(r"https?://", str(path), re.I):     # §4PE: `httpd.conf` is a name', '    if path and str(path).startswith("http"):'),
    ("M30 find cap note dropped", FS, '        if len(output.strip().splitlines()) >= 100:       # §4PE: 100 names read as "all"', '        if False:'),
    # ── class D: projects ──
    ("M31 budget by title falls back to the current project", PJ, '                         "budget", "constraint_retire", "verify_release", "unrelease"}', '                         "constraint_retire", "verify_release", "unrelease"}'),
    ("M32 probe rewrites the binding sentinel", PJ, '            elif _probe_turn_now():\n                # §4PE (P3)', '            elif False:\n                # §4PE (P3)'),
    ("M33 events spliced whole", PJ, '"payload": _event_payload_preview(e["type"], e["payload"])}', '"payload": e["payload"]}'),
    ("M34 deleted id not named", PJ, '                    _tomb = _deleted_note(store, project_id)             # §4PE (P7)', '                    _tomb = ""'),
    ("M35 artifact path not normalised", PJ, '                if not _rel or ".." in Path(_rel).parts:', '                if False:'),
    # ── class E: workspace + services ──
    ("M36 workspace results undeclared", WS, '    return _TO.ok(str(res), world_changed=False)', '    return str(res)'),
    ("M37 probe writes the ledger through the tool", WT, '        if _is_probe():\n            return _TO.rejected("Error: a probe request does not write the workspace ledger', '        if False:\n            return _TO.rejected("Error: a probe request does not write the workspace ledger'),
    ("M38 model.note ungated for probes", WM, '        if not self.enabled or self.activity is None or _probe_turn():     # §4PE (W2)\n            return None\n        summary = (summary or "").strip()', '        if not self.enabled or self.activity is None:\n            return None\n        summary = (summary or "").strip()'),
    ("M39 track watches host paths", WM, '        if cand_r != root_r and root_r not in cand_r.parents:\n            return None', '        pass'),
    ("M40 ~ not expanded before the root check", WM, '        p = os.path.expanduser(str(path or "").strip())        # `~/…` names the host\'s home: outside', '        p = str(path or "").strip()'),
    ("M41 narrative unheaded", WS, '            return ("WORKSPACE NARRATIVE (the agent\'s own running summary):\\n" + text) if text \\\n                else "No workspace narrative on file yet."', '            return text or "No workspace narrative on file yet."'),
    ("M42 kill survivor booked as a refusal", SV, '        if "survived TERM+KILL" in _t:', '        if False:'),
    ("M43 sandbox-down undeclared", SV, '        return ToolOutcome.failed("Error: the sandbox is not running, so no service can be managed — "', '        return str("Error: the sandbox is not running, so no service can be managed — "'),
    ("M44 missing name says invalid", SS, '        if not name:                                                # §4PE (W8): missing ≠ invalid', '        if False:'),
    # ── round 3 ──
    ("M45 selection reply runs the whole list", MM, '''        if (str(selection if selection is not None else "all").strip().lower() in ("all", "*", "yes", "")
                and re.fullmatch(r"[\\d\\s,\\-\\[\\]]+", _msg)):
            return None, None''', '        pass'),
    ("M46 bare forget ignores the call's items", MM, '        _sel0 = kwargs.get("items") or "all"', '        _sel0 = "all"'),
    ("M47 probe close deletes the sentinels", PJ, '                if not _probe_turn_now():       # §4PE r3', '                if True:       # §4PE r3'),
    ("M48 untrack unmapped", WM, '        host = self.workspace_host_path(path)                              # §4PE r3: the spelling `track` echoes\n        if host and self.state.untrack_file(host):\n            return True', '        pass'),
    ("M49 empty page is a plain string again", "src/ghost_agent/utils/helpers.py", '                        else FetchError("Error: No text content extracted from page."))   # §4PE r3: a failure is a TYPE', '                        else "Error: No text content extracted from page.")'),
    ("M50 HTML body under a binary name is SUCCESS", FS, '    if suffix in _BINARY_SUFFIXES and (re.search(rb"<!doctype\\s+html|<html[\\s>]", head[:512], re.I)\n                                       or ct == "text/html"):', '    if False:'),
    ("M51 relative track path against the outer root", WM, '        if not Path(p).is_absolute() and self.current_project_id:         # §4PE r3: the model\'s cwd is the', '        if False:'),
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
