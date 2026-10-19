#!/usr/bin/env python3
"""§4PN battery — the confirmation-wait reply (core/agent.py
`_confirm_wait_reply` + its dispatch branch), 2026-10-16. Same harness as
§4PM, on a tree COPY (`BATTERY_ROOT`).

    BATTERY_ROOT=/path/to/copy python3 scripts/mutate_4pn.py [all|rerun N1 …]
"""
import os, runpy
_H = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mutate_4pm.py"), run_name="harness")
AG = "src/ghost_agent/core/agent.py"
MUT = [
    ("CONTROL no-op (must SURVIVE)", AG, "def _quiet_kb_result(res) -> bool:", "# ctl\ndef _quiet_kb_result(res) -> bool:"),
    ("CONTROL known-bad (must DIE): never answers", AG, "    if not _CONFIRM_WAIT_RE.search(str(refusal or \"\")):\n        return \"\"\n    want_reset", "    return \"\"\n    want_reset"),
    ("P1 any refusal reason answered", AG, "    if not _CONFIRM_WAIT_RE.search(str(refusal or \"\")):\n        return \"\"\n    want_reset", "    want_reset"),
    ("P2 minted rows read as the preview", AG, "        if not isinstance(t, dict) or t.get(\"_synthetic\"):\n            continue\n        if str(t.get(\"name\")", "        if not isinstance(t, dict):\n            continue\n        if str(t.get(\"name\")"),
    ("P3 other tools in the turn ignored", AG, "        if str(t.get(\"name\") or \"\").strip().lower() != \"knowledge_base\" or not _quiet_kb_result(t.get(\"content\")):", "        if not _quiet_kb_result(t.get(\"content\")):"),
    ("P4 loud kb rows in the turn ignored", AG, "        if str(t.get(\"name\") or \"\").strip().lower() != \"knowledge_base\" or not _quiet_kb_result(t.get(\"content\")):", "        if str(t.get(\"name\") or \"\").strip().lower() != \"knowledge_base\":"),
    ("P6 batch other tools ignored", AG, "        if str(name or \"\").strip().lower() != \"knowledge_base\" or not _quiet_kb_result(res):", "        if not _quiet_kb_result(res):"),
    ("P8 kind not matched", AG, "                     if bool(re.match(r\"\\s*PREVIEW — nothing was deleted\\.\\s*reset_all\\b\", p)) == want_reset]", "                     ]"),
    ("P12 truncated answered", AG, "    if len(named) != 1 or \"[TRUNCATED\" in named[0]:", "    if len(named) != 1:"),
    ("P13 first trailer, not last", AG, "    shown = _PREVIEW_HEAD_RE.sub(\"\", preview[:cut[-1]], count=1).strip()", "    shown = _PREVIEW_HEAD_RE.sub(\"\", preview[:cut[0]], count=1).strip()"),
    ("P14 trailer kept", AG, "    shown = _PREVIEW_HEAD_RE.sub(\"\", preview[:cut[-1]], count=1).strip()", "    shown = _PREVIEW_HEAD_RE.sub(\"\", preview, count=1).strip()"),
    ("P15 advice kept", AG, "    shown = re.sub(r\"\\s*Re-issue with [^\\n]*\", \"\", shown)\n", ""),
    ("P16 no defaults promises yes", AG, "        if not defaults:                                 # \"yes\" alone selects nothing (r1 M1)", "        if False:                                 # \"yes\" alone selects nothing (r1 M1)"),
    ("P17 defaults read from the extras too", AG, "        main = shown.split(\"\\nAlso found\", 1)[0].split(\"\\nNotes:\", 1)[0]", "        main = shown"),
    ("P18 numbering gap accepted", AG, "            if int(d) == len(defaults) + 1:\n                defaults.append(d)", "            defaults.append(d)"),
    ("P19 either-group wording always", AG, "                   + (\", or give the numbers of exactly the ones to delete (any from either group).\" if extra", "                   + (\", or give the numbers of exactly the ones to delete (any from either group).\" if True"),
    ("P20 reset wording lost", AG, "        shown = re.sub(r\"^reset_all would erase\", \"Resetting the memory would erase\", shown, count=1)\n", ""),
    ("P21 group label kept", AG, "    shown = shown.replace(\"(kept unless the user picks them by number)\", \"(kept unless you pick them by number)\")\n", ""),
    ("P22 branch not wired", AG, "                    elif (_dead_code in _CONFIRM_WAIT_CODES and not _batch_short_circuit", "                    elif (False and _dead_code in _CONFIRM_WAIT_CODES and not _batch_short_circuit"),
    ("P23 no force_stop", AG, "                        final_ai_content = _cw_reply\n                        force_stop = True", "                        final_ai_content = _cw_reply\n                        force_stop = False"),
    ("P24 call args not passed", AG, "                              _recorded_args if isinstance(_recorded_args, dict) else {}, _dead_code,", "                              {}, _dead_code,"),
    ("P25 batch not passed", AG, "                              [(tool_call_metadata[_j][0], results[_j]) for _j in range(len(results))\n                               if _j != i]))):", "                              []))):"),
    # r4 EQUIVALENT survivors (kept as tripwires): P3/P6 — no tool but memory emits
    # forget_/reset_ codes, so the name check is defence in depth; P7b — no OK result
    # carries a forget_/reset_ code; P9 — the later `len(named) != 1` guards it twice.
    # equivalent, not listed: confirm_dead_end added to _CONFIRM_WAIT_CODES (a project
    # refusal carries no forget/reset PREVIEW, so the reply is "" and the old path runs)
    ("P5 batch not checked", AG, "    for name, res in list(batch or []):\n        # (r3 m1", "    for name, res in []:\n        # (r3 m1"),
    ("P7 any refusal is quiet", AG, "            and str(getattr(res, \"reason_code\", \"\") or \"\").startswith((\"forget_\", \"reset_\")))", "            )"),
    ("P7b a read is quiet", AG, "    return (getattr(getattr(res, \"status\", None), \"value\", None) == \"rejected\"", "    return (getattr(getattr(res, \"status\", None), \"value\", None) in (\"rejected\", \"ok\")"),
    ("P9 two previews answered", AG, "    if len(kind_previews) != 1:\n        return \"\"\n", ""),
    ("P10 another preview's token answered", AG, "    if _PLAN_TOKEN_RE.fullmatch(token) and f\"confirm='{token}'\" not in kind_previews[0]:\n        return \"\"", "    if False:\n        return \"\""),
    ("P11 quotes not stripped", AG, "    token = str((call_args or {}).get(\"confirm\") or \"\").strip().strip(\"'\\\"\").strip()", "    token = str((call_args or {}).get(\"confirm\") or \"\").strip()"),
    ("P26 a None result is quiet", AG, "        if str(name or \"\").strip().lower() != \"knowledge_base\" or not _quiet_kb_result(res):", "        if res is not None and (str(name or \"\").strip().lower() != \"knowledge_base\" or not _quiet_kb_result(res)):"),
]
_H["MUT"][:] = MUT
_H["PINS"][:] = ["tests/test_4pn_confirm_wait.py", "tests/test_4lw_open_items.py"]
g = _H["main"].__globals__
g["MUT"], g["PINS"] = MUT, _H["PINS"]
g["_FULL_PAT"] = r"_confirm_wait_reply\|forget_not_confirmed\|_dispatch_and_process_tool_batch"
if __name__ == "__main__":
    _H["main"]()
