#!/usr/bin/env python3
"""§4PO battery — the tool-free answer that ends on a promise (core/agent.py
`_ends_on_announced_work` + the §4IW continuation trigger), 2026-10-16. Same
harness as §4PM, on a tree COPY (`BATTERY_ROOT`).

    BATTERY_ROOT=/path/to/copy python3 scripts/mutate_4po.py [all|rerun Q1 …]
"""
import os, runpy
_H = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mutate_4pm.py"), run_name="harness")
AG = "src/ghost_agent/core/agent.py"
MUT = [
    ("CONTROL no-op (must SURVIVE)", AG, "def _announced_tail(this_turn_text: str, request: str = \"\") -> str:", "# ctl\ndef _announced_tail(this_turn_text: str, request: str = \"\") -> str:"),
    ("CONTROL known-bad (must DIE): never fires", AG, "    return tail if narration_only(tail, request=request) else \"\"", "    return \"\""),
    ("Q1 whole reply judged, not the tail", AG, "    return tail if narration_only(tail, request=request) else \"\"", "    return tail if narration_only(body, request=request) else \"\""),
    ("Q2 last line, not last sentence", AG, "    tail = (sents[-1] if sents else last_line).strip()", "    tail = last_line"),
    ("Q3 first line", AG, "    last_line = [ln for ln in body.splitlines() if ln.strip()][-1].strip()", "    last_line = [ln for ln in body.splitlines() if ln.strip()][0].strip()"),
    ("Q4 fires on whole narration too", AG, "    if not tail or tail == body or _LATER_RE.search(tail):", "    if not tail or _LATER_RE.search(tail):"),
    ("Q5 trigger not wired", AG, "                        and (_aw_whole or _aw_tail)):", "                        and _aw_whole):"),
    ("Q6 tool turns too", AG, "                _aw_tail = (not _aw_whole and not _aw_tools_ran and bool(clean_ui)", "                _aw_tail = (not _aw_whole and bool(clean_ui)"),
    ("Q7 system notes read as the reply", AG, "    from .reply_smoothing import narration_only, strip_system_notes\n    body = strip_system_notes(str(this_turn_text or \"\")).strip()\n    if not body or \"```\"", "    from .reply_smoothing import narration_only, strip_system_notes\n    body = str(this_turn_text or \"\").strip()\n    if not body or \"```\""),
    ("Q8 log names the wrong shape", AG, "                         if _aw_tail else \"the reply ends by announcing work it did not do — one \")", "                         if False else \"the reply ends by announcing work it did not do — one \")"),
    ("Q9 later promises fire", AG, "    if not tail or tail == body or _LATER_RE.search(tail):", "    if not tail or tail == body:"),
    ("Q10 list steps fire", AG, "    if _LIST_LINE_RE.match(last_line):\n        return \"\"", "    if False:\n        return \"\""),
    ("Q11 creative requests fire", AG, "        if _CREATIVE.search(str(request or \"\")):\n            return \"\"", "        if False:\n            return \"\""),
    ("Q12 code at the end fires", AG, "    if not body or \"```\" in body[-400:]:", "    if not body:"),
    # Q15 is EQUIVALENT: the tail is read from the LAST LINE only, so no newline reaches the splitter
    ("Q15 newline counts as a sentence split", AG, "_SENTENCE_SPLIT_RE = re.compile(r\"(?<=[.!?;])[ \\t]+(?=[A-ZΑ-Ω*\\\"'(«])\")", "_SENTENCE_SPLIT_RE = re.compile(r\"(?<=[.!?;])\\s+(?=[A-ZΑ-Ω*\\\"'(«])\")"),
    ("Q13 the answer is not kept", AG, "                        if _kept:\n                            if final_ai_content", "                        if False:\n                            if final_ai_content"),
    ("Q16 the promise kept too", AG, "    return t[:i].rstrip()", "    return t"),
    # Q17 is EQUIVALENT in practice: the tail is taken from the END of the same text
    ("Q17 a mid-text match cut", AG, "    if not tail or i < 0 or t[i + len(tail):].strip():", "    if not tail or i < 0:"),
    ("Q18 the whole-narration arm keeps it too", AG, "                    if _aw_tail:\n                        _tail_s", "                    if _aw_tail or _aw_whole:\n                        _tail_s"),
    ("Q19 carry not recorded", AG, "                            _po_note_carry(self.context, request_id_context.get(),\n                                           len(final_ai_content or \"\"), _kept)\n", ""),
    ("Q20 empty carry claims the user has it", AG, "                            messages.append({\"role\": \"user\", \"content\":\n                                             _TRAILING_ANNOUNCED_STANDALONE % repr(_tail_s[:160])})", "                            messages.append({\"role\": \"user\", \"content\":\n                                             _TRAILING_ANNOUNCED_DIRECTIVE % repr(_tail_s[:160])})"),
    ("Q21 carry record unbounded", AG, "        while len(d) > 16:\n            d.pop(next(iter(d)))", "        pass"),
    # KNOWN GAP (reported, not claimed covered): Q19, Q20, Q22, Q23 run only behind the
    # verifier's repair, the forced-final fallback or a dropped-write note — the loop
    # harness has none of them; they were checked by reading (fresh reader r3)
    ("Q22 rewind keeps the carry", AG, "                            if _pc and 0 <= int(_pc[0]) <= len(final_ai_content or \"\"):\n                                final_ai_content = (final_ai_content or \"\")[:int(_pc[0])].rstrip()", "                            if False:\n                                final_ai_content = (final_ai_content or \"\")[:int(_pc[0])].rstrip()"),
    ("Q23 fallback drops the carry", AG, "                    final_ai_content = (str(_pc[1]) if _pc else \"\")", "                    final_ai_content = \"\""),
    ("Q14 the §4IW directive for the tail arm", AG, "                            messages.append({\"role\": \"user\", \"content\":\n                                             _TRAILING_ANNOUNCED_DIRECTIVE % repr(_tail_s[:160])})", "                            messages.append({\"role\": \"user\", \"content\": _ANNOUNCED_WORK_DIRECTIVE})"),
]
_H["MUT"][:] = MUT
_H["PINS"][:] = ["tests/test_4po_trailing_promise.py", "tests/test_4kw_announced_work_and_outcome.py"]
g = _H["main"].__globals__
g["MUT"], g["PINS"] = MUT, _H["PINS"]
g["_FULL_PAT"] = r"_announced_work_without_acting\|_ends_on_announced_work\|_ANNOUNCED_WORK_DIRECTIVE\|_ends_with_action_promise"
if __name__ == "__main__":
    _H["main"]()
