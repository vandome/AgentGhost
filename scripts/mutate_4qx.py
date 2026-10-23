"""§4QX battery — the system shows what the picture shows (2026-10-23). Reuses the §4QA harness.
    python3 scripts/mutate_4qx.py [all|check|rerun O1 O2 …]
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "mutate_4qa", os.path.join(os.path.dirname(os.path.abspath(__file__)), "mutate_4qa.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

IG = "src/ghost_agent/tools/image_gen.py"
AG = "src/ghost_agent/core/agent.py"
LG = "src/ghost_agent/utils/logging.py"
RM = "src/ghost_agent/core/receipt_marks.py"
m.PINS = ["tests/test_4qx_look_owned.py", "tests/test_4qp_render_look.py", "tests/test_4qt_look_critic.py",
          "tests/test_4qu_check_critic.py", "tests/test_agent_image_review_fixes.py"]


def _tests():
    import glob, subprocess
    pat = r"look_owned\|LOOK_OWNED\|_owned_look_note\|render_look_enabled\|receipt_marks\|PROMPT_ARMS\|_two_panel_note\|_unshown_image_note"
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(m.ROOT, "tests", "test_*.py"))),
                         capture_output=True, text=True).stdout.split()
    return sorted({os.path.relpath(t, m.ROOT) for t in out} | set(m.PINS))


m._tests = _tests
m.MUT = [
    ("CONTROL no-op (must SURVIVE)", AG, "def _owned_look_note(text, tools_run_this_turn) -> str:",
     "# ctl\ndef _owned_look_note(text, tools_run_this_turn) -> str:"),
    ("CONTROL known-bad (must DIE): the line under every picture", AG,
     "            if end < 0 or not c.startswith(\"\\n\" + LOOK_OWNED_NOTE, end + 1):\n                continue",
     "            if end < 0:\n                continue"),
    ("O1 the env switch is ignored", IG,
     '    if os.environ.get(LOOK_OWNED_ENV, "0").strip().lower() in ("1", "true", "on", "yes"):\n        return True\n    try:\n        from ..utils.logging import prompt_arm\n        return prompt_arm() == LOOK_OWNED_ARM',
     '    if False:\n        return True\n    try:\n        from ..utils.logging import prompt_arm\n        return prompt_arm() == LOOK_OWNED_ARM'),
    ("O2 any env value owns the line", IG,
     '    if os.environ.get(LOOK_OWNED_ENV, "0").strip().lower() in ("1", "true", "on", "yes"):\n        return True\n    try:\n        from ..utils.logging import prompt_arm\n        return prompt_arm() == LOOK_OWNED_ARM',
     '    if os.environ.get(LOOK_OWNED_ENV, "0").strip():\n        return True\n    try:\n        from ..utils.logging import prompt_arm\n        return prompt_arm() == LOOK_OWNED_ARM'),
    ("O3 the env switch does not imply the look", IG,
     '        return True                                             # §4QX: owning the line implies the look',
     '        return False                                            # §4QX: owning the line implies the look'),
    ("O4 the arm does not imply the look", IG,
     "        return prompt_arm() in (LOOK_ARM, LOOK_CRITIC_ARM, LOOK_OWNED_ARM)  # §4QT/§4QX: these arms imply the look",
     "        return prompt_arm() in (LOOK_ARM, LOOK_CRITIC_ARM)"),
    ("O5 the arm is not honoured", IG, "        return prompt_arm() == LOOK_OWNED_ARM", "        return False"),
    ("O6 the arm is not registered", LG, '"check_critic", "look_owned"})', '"check_critic"})'),
    ("O7 the tool keeps the describe sentence under the owned arm", IG,
     "               if _seen and look_owned_enabled() else", "               if _seen and False else"),
    ("O8 owned with no look", IG, "               if _seen and look_owned_enabled() else", "               if look_owned_enabled() else"),
    ("O9 the tool's note drifts from the shared literal", IG, '               f"{_LOOK_OWNED_NOTE} Do NOT "',
     '               "The SYSTEM shows the user that line, directly under the picture. Do NOT "'),
    ("O10 a synthetic row owns a line", AG,
     "            if not isinstance(t, dict) or t.get(\"_synthetic\"):\n                continue\n            c = str(t.get(\"content\") or \"\")\n            if not c.lstrip().startswith(IMAGE_SUCCESS_HEAD):\n                continue\n            i = c.find",
     "            if not isinstance(t, dict):\n                continue\n            c = str(t.get(\"content\") or \"\")\n            if not c.lstrip().startswith(IMAGE_SUCCESS_HEAD):\n                continue\n            i = c.find"),
    ("O11 any tool text with the head owns a line", AG,
     "            if not c.lstrip().startswith(IMAGE_SUCCESS_HEAD):\n                continue\n            i = c.find",
     "            if IMAGE_SUCCESS_HEAD not in c:\n                continue\n            i = c.find"),
    ("O12 the note may sit anywhere in the result", AG,
     "            if end < 0 or not c.startswith(\"\\n\" + LOOK_OWNED_NOTE, end + 1):",
     "            if end < 0 or LOOK_OWNED_NOTE not in c:"),
    ("O13 a picture the reply does not show gets a line", AG,
     "            if not any(n in shown for n in names):\n                continue\n            # already carried",
     "            if False:\n                continue\n            # already carried"),
    ("O14 an empty look gets a bare label", AG,
     "            if look and names:\n                out.append((names, LOOK_OWNED_LABEL + look))",
     "            if names:\n                out.append((names, LOOK_OWNED_LABEL + look))"),
    ("O15 a line already carried is repeated", AG,
     '            if line in out or _owned_line_re(line).search(body):',
     '            if line in out:'),
    ("O16 the same look twice is shown twice", AG,
     '            if line in out or _owned_line_re(line).search(body):',
     '            if _owned_line_re(line).search(body):'),
    ("O23 a model sentence starting with the label suppresses the line (r1 M2)", AG,
     '            if line in out or _owned_line_re(line).search(body):',
     '            if line in out or line in body:'),
    ("O24 the verdict judges the system's line (r1/r2 M1)", AG,
     "        final_ai_content = _without_owned_look_lines(final_ai_content, tools_run_this_turn)\n",
     ""),
    ("O25 the strip removes every copy (cuts the model's sentence)", AG,
     "                h = hits[-1]", "                h = hits[0]"),
    ("O26 the strip matches a sentence that merely starts with the line", AG,
     '    return re.compile(r"(?:^|\\n)" + re.escape(line) + r"[ \\t]*(?=\\n|$)")',
     '    return re.compile(r"(?:^|\\n)" + re.escape(line))'),
    ("O17 a query string hides the shown picture", AG,
     "        shown = {os.path.basename(m.group(2).split(\"?\", 1)[0]) for m in _DOWNLOAD_LINK_RE.finditer(body)}",
     "        shown = {os.path.basename(m.group(2)) for m in _DOWNLOAD_LINK_RE.finditer(body)}"),
    ("O18 the finalize path does not append the line", AG,
     "            final_ai_content = (final_ai_content or \"\").rstrip() + _ol_note",
     "            final_ai_content = (final_ai_content or \"\").rstrip()"),
    ("O19 the stream does not send the line", AG,
     "'choices': [{'index': 0, 'delta': {'content': _ol_note_s}, 'finish_reason': None}]})}\\n\\n\".encode('utf-8')\n                    full_content = (full_content or \"\") + _ol_note_s",
     "'choices': [{'index': 0, 'delta': {'content': ''}, 'finish_reason': None}]})}\\n\\n\".encode('utf-8')\n                    full_content = (full_content or \"\") + _ol_note_s"),
    ("O20 the census row misses the line", AG,
     "                    _shown = _shown + _owned_look_note(_shown, tools_run)          # §4QX",
     "                    _shown = _shown"),
    ("O21 the label drifts", RM, 'LOOK_OWNED_LABEL = "What the picture shows: "', 'LOOK_OWNED_LABEL = "The picture shows: "'),
    ("O22 the look is cut at the first space", AG,
     "            end = c.find(\"\\n\", start)", "            end = c.find(\" \", start)"),
]

if __name__ == "__main__":
    m.main()
