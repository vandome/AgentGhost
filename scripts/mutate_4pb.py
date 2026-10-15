#!/usr/bin/env python3
"""§4PB battery — image generation / editing / multi-step review fixes (2026-10-14).

IN PLACE with backup + restore + byte compare after every mutant; one mutant per
run; py_compile first; the FULL set of files touching the change; a no-op
control that must SURVIVE and a known-bad control that must DIE.

    python3 scripts/mutate_4pb.py [slice: a|b|all]
"""
import os, shutil, subprocess, sys, signal, filecmp
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/Users/vasilis/Data/AI/.agent.venv/bin/python"
IG = "src/ghost_agent/tools/image_gen.py"
SP = "src/ghost_agent/tools/subject_photos.py"
NODE = "interface/externals/image_generation/img_gen_server.py"
AG = "src/ghost_agent/core/agent.py"
TESTS = ["tests/test_4pb_image_review.py", "tests/test_4oz_subjects_collage.py", "tests/test_4mg_image_subjects.py",
         "tests/test_4mg_review.py", "tests/test_image_gen_tool.py", "tests/test_image_gen.py",
         "tests/test_image_gen_model_swap.py", "tests/test_image_gen_node_retune.py",
         "tests/test_4kd_image_pipeline_review.py", "tests/test_llm_client_image_gen.py",
         "tests/test_4mo_turn_loop.py", "tests/test_image_gen_integration.py",
         "tests/test_4kd_clarify_first_and_slack_lessons.py"]
A = [
    ("CONTROL no-op (must SURVIVE)", IG, "def _plain(name: str) -> str:", "# ctl\ndef _plain(name: str) -> str:"),
    ("CONTROL known-bad (must DIE): verdict parser accepts anything", IG,
     'if ans not in ("YES", "NO"):\n            return None', 'if False:\n            return None'),
    ("M1 download failure back to 'missing'", SP,
     'raise PhotoUnavailable(f"the photo from {page} could not be downloaded", page, title)',
     'raise LookupError(f"the photo from {page} could not be downloaded")'),
    ("M2 truncated download back to 'missing'", SP,
     'raise PhotoUnavailable(f"what {page} served is not a complete photo", page, title)',
     'raise LookupError(f"what {page} served is not a complete photo")'),
    ("M3 PhotoUnavailable arm dropped in the tool", IG,
     "        undelivered = [(s, r) for s, r in zip(subj, results) if isinstance(r, PhotoUnavailable)]",
     "        undelivered = []"),
    ("M4 fuzzy token unanchored", SP,
     "    anchored = any(t in have and len(t) >= FUZZY_ANCHOR_LEN for t in want if t != missing[0])",
     "    anchored = True"),
    ("M5 fuzzy allows two edits", SP,
     "    if len(a) == len(b):\n        return sum(x != y for x, y in zip(a, b)) == 1",
     "    if len(a) == len(b):\n        return sum(x != y for x, y in zip(a, b)) <= 2"),
    ("M6 redirect not trusted (titles step accepts only exact)", SP,
     "            if src and (title_is_exact(subject, title) or title in redirected_to):",
     "            if src and title_is_exact(subject, title):"),
    ("M7 search no longer prefers the exact title", SP,
     "            hits.append((0 if title_is_exact(subject, title) else 1, title, src))",
     "            hits.append((1, title, src))"),
    ("M8 missing title page accepted", SP,
     '            if "missing" in page or "disambiguation" in (page.get("pageprops") or {}):\n                continue\n            src = str((page.get("thumbnail") or {}).get("source") or "")\n            if src and (title_is_exact',
     '            if "disambiguation" in (page.get("pageprops") or {}):\n                continue\n            src = str((page.get("thumbnail") or {}).get("source") or "")\n            if src and (title_is_exact'),
    ("M9 wide panel cropped again", SP,
     "            if w <= cap:\n                ims.append(ImageOps.fit(im, (w, COMBINED_HEIGHT), Image.Resampling.LANCZOS,\n                                        centering=(0.5, 0.3)))\n            else:",
     "            if True:\n                w = min(w, cap)\n                ims.append(ImageOps.fit(im, (w, COMBINED_HEIGHT), Image.Resampling.LANCZOS,\n                                        centering=(0.5, 0.3)))\n            else:"),
    ("M10 aspect band unchecked in the tool", IG,
     "        if aspect is not None and not (0.25 <= aspect <= 4.0):", "        if False:"),
    ("M11 root-sandbox fallback removed", IG,
     '    if root.parent.name == "projects" and root.parent.parent.is_dir():\n        roots.append(root.parent.parent.resolve())',
     '    pass'),
    ("M12 root fallback without containment", IG,
     "            try:\n                target.relative_to(base)\n            except ValueError:\n                raise ValueError(f\"reference image {name!r} is outside the sandbox\")\n            if target.is_file():\n                return target",
     "            if target.is_file():\n                return target"),
]
B = [
    ("M13 legs not timed (wrapper books nothing, impl neither)", IG,
     "    RECENT_RENDER_SECONDS[kind].append(time.monotonic() - t0)\n    return resp", "    return resp"),
    ("M14 transparent sent with a reference", IG,
     "        if transparent and (ref_b64 or subject_note):", "        if False:"),
    ("M15 alpha claim from the flag again", IG,
     "               if transparent and _png_has_alpha(image_bytes) else", "               if transparent else"),
    ("M16 `subject` beside a prompt ignored", IG,
     '        subjects = kwargs["subject"]          # §4PB: `subject` beside a prompt is the subjects list',
     '        pass'),
    ("M17 wrong-typed subjects silently dropped", IG,
     "    if subjects is not None and not isinstance(subjects, (list, tuple, str)):", "    if False:"),
    ("M18 non-dict node answer crashes", IG,
     '        used_seed = resp_data.get("seed") if isinstance(resp_data, dict) else None',
     '        used_seed = resp_data.get("seed")'),
    ("M19 negative prompt not capped", IG,
     "            if len(negative_prompt) > MAX_PROMPT_CHARS:      # §4PB: the node 400s above it",
     "            if False:"),
    ("M20 scene keeps the strip's 768x512", IG,
     "        if not size_asked and _scene:", "        if False:"),
    ("M21 articles note dropped", IG,
     '            + _articles_note + "\\n")', '            + "\\n")'),
    ("M22 mismatch note dropped", IG,
     "        if _mismatch:\n            _articles_note += (", "        if False:\n            _articles_note += ("),
    ("M23 EDITED FROM record dropped", IG,
     '            _edit_note += f"EDITED FROM: {_plain(Path(_ref_names[0].strip()).name)}\\n\\n"', "            pass"),
    ("M24 agent ignores EDITED FROM", AG,
     "        if not before:\n            edited_from = _turn_edit_reference(tools_run)", "        if False:\n            edited_from = _turn_edit_reference(tools_run)"),
    ("M25 agent: a user screenshot no longer wins", AG,
     "        if not before:\n            edited_from = _turn_edit_reference(tools_run)", "        if True:\n            edited_from = _turn_edit_reference(tools_run)"),
    ("M26 node stretches again", NODE,
     "                scale = min(width / im.width, height / im.height)", "                scale = 1.0; im = im.resize((width, height), Image.LANCZOS); scale = 0"),
    ("M27 node reads the key after the body", NODE,
     '        if API_KEY and scope.get("path") in ("/generate", "/v1/images/generations"):', "        if False:"),
    ("M28 node 500 while retrying", NODE,
     "    if _load_error and not _preflight_done:", "    if False:"),
    ("M29 node preflight text unscrubbed", NODE,
     'return HTTPException(status_code=500, detail=f"preflight failed: {_public(_load_error)}")',
     'return HTTPException(status_code=500, detail=f"preflight failed: {_load_error}")'),
    ("M30 node image_size decodes pixels again", NODE,
     "            w, h = im.size\n            try:\n                if int(im.getexif().get(0x0112, 1) or 1) in (5, 6, 7, 8):\n                    w, h = h, w\n            except Exception:  # noqa: BLE001\n                pass",
     "            w, h = (ImageOps.exif_transpose(im) or im).size"),
    ("M31 costly-turn helper ignores the image result", AG,
     '                       or str(t.get("content") or "").lstrip().startswith("SUCCESS: Image generated"))', "                       )"),
    ("M32 limit error no longer steers the add case", IG,
     '                       "`reference_images`. " if _is_gen else "Pick the ones that matter most. ")',
     '                       "`reference_images`. " if False else "Pick the ones that matter most. ")'),
]
C = [m for m in B if m[0][:3] in ("M31", "M32")] + [m for m in A if m[0][:2] in ("M5", "M7", "M8")]
D = [m for m in A if m[0][:3] == "M5 "] + [A[0]]

def run_tests():
    env = dict(os.environ, PYTHONPATH="src", GHOST_API_KEY=open(os.path.expanduser("~/Data/AI/.ghost_api_key")).read().strip())
    env.pop("FORCE_COLOR", None)
    r = subprocess.run([PY, "-m", "pytest", *TESTS, "-q", "--no-header", "-p", "no:cacheprovider", "-x", "-n", "4", "--dist", "loadfile"],
                       capture_output=True, text=True, env=env, cwd=ROOT)
    tail = (r.stdout.strip().splitlines() or ["?"])[-1]
    return r.returncode, tail

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    batch = {"a": A, "b": B, "c": C, "d": D, "all": A + B}[which]
    files = sorted({m[1] for m in batch})
    backups = {f: f + ".4pb.bak" for f in files}
    for f, b in backups.items():
        shutil.copy2(os.path.join(ROOT, f), os.path.join(ROOT, b))
    def restore(*_):
        for f, b in backups.items():
            shutil.copy2(os.path.join(ROOT, b), os.path.join(ROOT, f))
        subprocess.run(["find", os.path.join(ROOT, "src"), "-name", "__pycache__", "-exec", "rm", "-rf", "{}", "+"], capture_output=True)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *a: (restore(), sys.exit(130)))
    killed = survived = 0
    try:
        for label, f, old, new in batch:
            restore()
            path = os.path.join(ROOT, f); src = open(path).read()
            if src.count(old) != 1:
                print(f"SKIP   {label}: anchor count {src.count(old)}", flush=True); continue
            open(path, "w").write(src.replace(old, new))
            c = subprocess.run([PY, "-m", "py_compile", path], capture_output=True, text=True)
            if c.returncode != 0:
                print(f"NOCOMP {label}", flush=True); continue
            rc, tail = run_tests()
            status = "SURVIVED" if rc == 0 else "KILLED"
            killed += rc != 0; survived += rc == 0
            print(f"{status:8} {label} — {tail}", flush=True)
    finally:
        restore()
        ok = all(filecmp.cmp(os.path.join(ROOT, f), os.path.join(ROOT, b), shallow=False) for f, b in backups.items())
        for b in backups.values():
            os.remove(os.path.join(ROOT, b))
        print(f"RESTORED identical={ok} killed={killed} survived={survived}", flush=True)

if __name__ == "__main__":
    main()
