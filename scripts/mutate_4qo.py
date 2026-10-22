"""§4QO battery — the shared caveat literals (2026-10-22; the turn record was removed).
    python3 scripts/mutate_4qo.py [all|check|rerun Q1 Q2 …]
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "mutate_4qa", os.path.join(os.path.dirname(os.path.abspath(__file__)), "mutate_4qa.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

RM = "src/ghost_agent/core/receipt_marks.py"; AG = "src/ghost_agent/core/agent.py"; IG = "src/ghost_agent/tools/image_gen.py"
IN = "src/ghost_agent/tools/introspect.py"; ME = "src/ghost_agent/tools/memory.py"; SY = "src/ghost_agent/tools/system.py"
m.PINS = ["tests/test_4qo_receipt_marks.py", "tests/test_4pn_confirm_wait.py", "tests/test_census_open_classes_4qj.py"]


def _tests():
    import glob, subprocess
    pat = r"receipt_marks\|OVERVIEW_HEALTH_LINE\|_confirm_wait_reply\|document_link_check_note\|tool_check_health\|NOT seen this image"
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(m.ROOT, "tests", "test_*.py"))),
                         capture_output=True, text=True).stdout.split()
    return sorted({os.path.relpath(t, m.ROOT) for t in out} | set(m.PINS))


m._tests = _tests
m.MUT = [
    ("CONTROL no-op (must SURVIVE)", RM, "UNSEEN_IMAGE_NOTE = ", "# ctl\nUNSEEN_IMAGE_NOTE = "),
    ("CONTROL known-bad (must DIE): introspect keeps its own copy of the health line", IN,
     "from ..core.receipt_marks import OVERVIEW_HEALTH_LINE  # §4QO: one home — the turn record reads this literal  # noqa: E402",
     'OVERVIEW_HEALTH_LINE = "Health: NOT checked by this overview."'),
    ("Q1 the image tool rewords the note", IG, '               + " " + _UNSEEN_IMAGE_NOTE + " "',
     '               + " You have NOT seen this picture: do not claim any specific detail is in it. "'),
    ("Q2 the memory gate rewords its refusal", ME, '        return (_NAY + " — show them the list and wait for their reply; confirm in the "',
     '        return ("the owner has not answered yet — show them the list and wait for their reply; confirm in the "'),
    ("Q3 the health check rewords its head", SY, "    health_status = [\n        _HCH,", '    health_status = [\n        "--- HEALTH ---",'),
    ("Q4 the link check rewords its head", AG, "    return (f\"\\n{_LCH}{len(missing)} URL(s) in '{path}' were NOT fetched successfully this request: {shown}. \"",
     "    return (f\"\\nLINKS: {len(missing)} URL(s) in '{path}' were NOT fetched successfully this request: {shown}. \""),
    ("Q5 the confirm-wait reply rewords its head", AG, '    return f"{_CWH}\\n\\n{shown}\\n\\n{ask}"', '    return f"Nothing deleted yet.\\n\\n{shown}\\n\\n{ask}"'),
    ("Q6 the marks module grows an import", RM, "#: image_generation: the model has NOT looked at the picture it produced",
     "import os  # noqa\n#: image_generation: the model has NOT looked at the picture it produced"),
]

if __name__ == "__main__":
    m.main()
