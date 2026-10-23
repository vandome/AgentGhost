"""§4QP battery — the image tool looks at every render (2026-10-22). Reuses the §4QA harness.
    python3 scripts/mutate_4qp.py [all|check|rerun L1 L2 …]
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "mutate_4qa", os.path.join(os.path.dirname(os.path.abspath(__file__)), "mutate_4qa.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

IG = "src/ghost_agent/tools/image_gen.py"
m.PINS = ["tests/test_4qp_render_look.py", "tests/test_census_open_classes_4qj.py", "tests/test_4oz_subjects_collage.py",
          "tests/test_agent_image_review_fixes.py", "tests/test_4qo_receipt_marks.py"]


def _tests():
    import glob, subprocess
    pat = r"image_gen\|tool_generate_image\|expected_render_seconds\|declared_wait_s\|render_look\|PROMPT_ARMS"
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(m.ROOT, "tests", "test_*.py"))),
                         capture_output=True, text=True).stdout.split()
    return sorted({os.path.relpath(t, m.ROOT) for t in out} | set(m.PINS))


m._tests = _tests
m.MUT = [
    ("CONTROL no-op (must SURVIVE)", IG, "def render_look_enabled() -> bool:", "# ctl\ndef render_look_enabled() -> bool:"),
    ("CONTROL known-bad (must DIE): the look on every production render", IG,
     '    if os.environ.get(LOOK_OWNED_ENV, "0").strip().lower() in ("1", "true", "on", "yes"):\n        return True                                             # §4QX: owning the line implies the look\n    try:\n        from ..utils.logging import prompt_arm\n        return prompt_arm() in (LOOK_ARM, LOOK_CRITIC_ARM, LOOK_OWNED_ARM)  # §4QT/§4QX: these arms imply the look',
     '    if os.environ.get(LOOK_OWNED_ENV, "0").strip().lower() in ("1", "true", "on", "yes"):\n        return True                                             # §4QX: owning the line implies the look\n    try:\n        from ..utils.logging import prompt_arm\n        return True'),
    ("L1 the look runs though the check gave a scene", IG, "        if image_bytes and not _seen and render_look_enabled():", "        if image_bytes and render_look_enabled():"),
    ("L2 the look ignores the time left", IG, "            if _left is not None and _left < _look_seconds():", "            if False:"),
    ("L3 the look is not sent to the vision pool", IG,
     "            resp = await llm_client.chat_completion(payload, use_vision=True, timeout=LOOK_TIMEOUT_S)",
     "            resp = await llm_client.chat_completion(payload, use_vision=False, timeout=LOOK_TIMEOUT_S)"),
    ("L4 a failed look is not booked", IG,
     "                resp = await llm_client.chat_completion(payload, use_vision=True, timeout=LOOK_TIMEOUT_S)\n        finally:\n            RECENT_LOOK_SECONDS.append(time.monotonic() - _t0)",
     "                resp = await llm_client.chat_completion(payload, use_vision=True, timeout=LOOK_TIMEOUT_S)\n        finally:\n            pass"),
    ("L5 a non-string scene is used", IG, '    return " ".join(_sc.split())[:400] if isinstance(_sc, str) else ""', '    return " ".join(str(_sc).split())[:400] if _sc else ""'),
    ("L6 the stray-quote fallback is gone", IG,
     "        _sc = re.sub(r'[\"},\\s]+$', \"\", _m.group(1)) if _m else None             # r3 m2: an empty value, a trailing comma",
     "        _sc = None"),
    ("L7 the estimate forgets the look", IG,
     "    look = _look_seconds() if (render_look_enabled() and (panels < 2 or COLLAGE_RETRIES <= 0)) else 0.0", "    look = 0.0"),
    ("L8 the estimate adds the look to the ladder too", IG,
     "    look = _look_seconds() if (render_look_enabled() and (panels < 2 or COLLAGE_RETRIES <= 0)) else 0.0\n    if panels < 2 or COLLAGE_RETRIES <= 0:\n        return single + look\n    return single * 2 + _verdict_seconds()",
     "    look = _look_seconds() if render_look_enabled() else 0.0\n    if panels < 2 or COLLAGE_RETRIES <= 0:\n        return single + look\n    return single * 2 + _verdict_seconds() + look"),
    ("L9 the question lets the node name people", IG,
     "    '{\"scene\": \"one plain sentence: what the picture shows — the people or things in it, what they '\n    'are doing, the setting; only what is visible, and never guess who anyone is\"}')",
     "    '{\"scene\": \"one plain sentence: what the picture shows — the people or things in it, what they '\n    'are doing, the setting; only what is visible\"}')"),
    ("L10 the look's seconds default to zero", IG, "    return float(max(RECENT_LOOK_SECONDS)) if RECENT_LOOK_SECONDS else LOOK_DEFAULT_S", "    return float(max(RECENT_LOOK_SECONDS)) if RECENT_LOOK_SECONDS else 0.0"),
    ("L11 an exception in the look propagates", IG,
     '    except Exception as e:  # noqa: BLE001 — the look is advisory\n        pretty_log("Image Generation", f"look unavailable ({e.__class__.__name__}) — picture kept, not described",\n                   icon=Icons.IMAGE_GEN)\n        return ""',
     '    except ValueError as e:  # noqa: BLE001 — the look is advisory\n        pretty_log("Image Generation", f"look unavailable ({e.__class__.__name__}) — picture kept, not described",\n                   icon=Icons.IMAGE_GEN)\n        return ""'),
    ("L12 the arm is not honoured", IG, "        return prompt_arm() in (LOOK_ARM, LOOK_CRITIC_ARM, LOOK_OWNED_ARM)  # §4QT/§4QX: these arms imply the look", '        return prompt_arm() == "no_lessons"'),
    ("L13 the look runs before the save", IG, '        await asyncio.to_thread(file_path.write_bytes, image_bytes)\n        # §4QP: a render with no scene yet gets ONE look, time permitting —\n        # AFTER the save (r1 M1: a cancel during the look — the hard stop, the\n        # idle supervisor — must not lose a picture the node already paid for)\n        # the reply then describes what the vision model saw, or says where\n        # the picture differs from what was asked; without a usable look the\n        # §4QJ NOT-seen sentence stays exactly as before\n        if image_bytes and not _seen and render_look_enabled():\n            _left = _request_time_left()\n            if _left is not None and _left < _look_seconds():\n                pretty_log("Image Generation",\n                           f"look skipped (needs ~{_look_seconds():.0f}s, {max(0.0, _left):.0f}s left before the "\n                           "report reserve) — picture kept, not described", icon=Icons.IMAGE_GEN)\n            else:\n                try:\n                    _seen = await scene_look(llm_client, image_bytes)\n                except BaseException:                    # r2 m3: the render is saved; name the orphan, then unwind\n                    try:\n                        pretty_log("Image Generation", f"look cancelled — {filename} is saved but unrecorded",\n                                   level="WARNING", icon=Icons.WARN)\n                    except Exception:  # noqa: BLE001 — r3 m5: a raising log must not replace the cancel\n                        pass\n                    raise\n                pretty_log("Image Generation",\n                           (f"looked at the render ({RECENT_LOOK_SECONDS[-1]:.0f}s): {_seen[:120]}" if _seen else\n                            "look gave no usable scene — picture kept, not described"), icon=Icons.IMAGE_GEN)\n\n', '        # §4QP: a render with no scene yet gets ONE look, time permitting —\n        # AFTER the save (r1 M1: a cancel during the look — the hard stop, the\n        # idle supervisor — must not lose a picture the node already paid for)\n        # the reply then describes what the vision model saw, or says where\n        # the picture differs from what was asked; without a usable look the\n        # §4QJ NOT-seen sentence stays exactly as before\n        if image_bytes and not _seen and render_look_enabled():\n            _left = _request_time_left()\n            if _left is not None and _left < _look_seconds():\n                pretty_log("Image Generation",\n                           f"look skipped (needs ~{_look_seconds():.0f}s, {max(0.0, _left):.0f}s left before the "\n                           "report reserve) — picture kept, not described", icon=Icons.IMAGE_GEN)\n            else:\n                try:\n                    _seen = await scene_look(llm_client, image_bytes)\n                except BaseException:                    # r2 m3: the render is saved; name the orphan, then unwind\n                    try:\n                        pretty_log("Image Generation", f"look cancelled — {filename} is saved but unrecorded",\n                                   level="WARNING", icon=Icons.WARN)\n                    except Exception:  # noqa: BLE001 — r3 m5: a raising log must not replace the cancel\n                        pass\n                    raise\n                pretty_log("Image Generation",\n                           (f"looked at the render ({RECENT_LOOK_SECONDS[-1]:.0f}s): {_seen[:120]}" if _seen else\n                            "look gave no usable scene — picture kept, not described"), icon=Icons.IMAGE_GEN)\n\n        await asyncio.to_thread(file_path.write_bytes, image_bytes)\n'),
    ("L14 the blind guard ignores the look", "src/ghost_agent/core/agent.py",
     "    inspected = _images_inspected_this_turn(messages) | _turn_looked_images(tools_run_this_turn)",
     "    inspected = _images_inspected_this_turn(messages)"),
    ("L15 text after the object is swallowed", IG,
     "        _m = (re.search(r'\"scene\"\\s*:\\s*\"(.*?)\"\\s*(?:\\}|,\\s*\"[^\"]+\"\\s*:)', _obj, re.S)   # r2 M1: a REAL terminator\n              or re.search(r'\"scene\"\\s*:\\s*\"(.+)', _obj, re.S))",
     "        _m = re.search(r'\"scene\"\\s*:\\s*\"(.+)', _obj, re.S)"),
    ("L16 the r1 terminator is back (a quoted word before a comma cuts the scene)", IG,
     "        _m = (re.search(r'\"scene\"\\s*:\\s*\"(.*?)\"\\s*(?:\\}|,\\s*\"[^\"]+\"\\s*:)', _obj, re.S)   # r2 M1: a REAL terminator",
     "        _m = (re.search(r'\"scene\"\\s*:\\s*\"(.*?)\"\\s*[},]', _obj, re.S)   # r2 M1: a REAL terminator"),
    ("L17 the guard ignores the tool's hold", "src/ghost_agent/core/agent.py", "        if _hm:\n            names =", "        if False:\n            names ="),
    ("L21 the hold is read from the whole body (the scene, a filename)", "src/ghost_agent/core/agent.py",
     '        _hm = re.search(r"^" + re.escape(NO_RERENDER_MARK) + r" ([^\\]\\n]{1,80})\\]$", head_part, re.M)',
     '        _hm = re.search(re.escape(NO_RERENDER_MARK) + r" ([^\\]\\n]{1,80})\\]", body)'),
    ("L22 a filename keeps its brackets", IG, '    return re.sub(r"[()\\[\\]\\r\\n]", "_", str(name))', '    return re.sub(r"[()\\r\\n]", "_", str(name))'),
    ("L23 an unclosed think block keeps its draft", IG,
     '    stripped = re.sub(r"<think>.*?(?:</think>|$)", "", txt, flags=re.S)',
     '    stripped = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)'),
    ("L24 a repair re-entry re-rolls the ladder (the bypass above the hold)", "src/ghost_agent/core/agent.py",
     "    from .receipt_marks import NO_RERENDER_MARK\n    for t in reversed(tools_run_this_turn or []):",
     "    if repair_active:\n        return None\n    from .receipt_marks import NO_RERENDER_MARK\n    for t in reversed(tools_run_this_turn or []):"),
    ("L27 the head is cut at a name quoting the markdown line", "src/ghost_agent/core/agent.py",
     '        head_part = re.split(r"^!\\[generated image\\]\\(", body, 1, flags=re.M)[0]   # r4 m1: the markdown LINE, not a name quoting it',
     '        head_part = body.split("![generated image](", 1)[0]'),
    ("L28 the think strip runs before valid JSON is tried", IG, "    for cand in (txt, stripped):", "    for cand in (stripped,):"),
    ("L25 the failed re-render note has no hold", IG,
     '                        f"{_NO_RERENDER_MARK} another attempt could not be rendered]\\n"\n', ''),
    ("L26 a trailing comma stays in the scene", IG,
     "        _sc = re.sub(r'[\"},\\s]+$', \"\", _m.group(1)) if _m else None             # r3 m2: an empty value, a trailing comma",
     "        _sc = re.sub(r'[\"}\\s]+$', \"\", _m.group(1)) if _m else None             # r3 m2: an empty value, a trailing comma"),
    ("L18 an older un-looked picture blocks", "src/ghost_agent/core/agent.py",
     "    newest = os.path.basename(prior[-1])\n    if newest in inspected:\n        return None\n    first = newest",
     "    uninspected = [p for p in prior if os.path.basename(p) not in inspected]\n    if not uninspected:\n        return None\n    first = os.path.basename(uninspected[-1])"),
    ("L19 the reference name is printed raw", IG,
     '                f"The model saw ONLY {_plain(Path(_ref_names[0].strip()).name)}: nothing from any earlier "',
     '                f"The model saw ONLY {Path(_ref_names[0].strip()).name}: nothing from any earlier "'),
    ("L20 a think block's draft is the scene", IG,
     '    stripped = re.sub(r"<think>.*?(?:</think>|$)", "", txt, flags=re.S)',
     '    stripped = txt'),
]

if __name__ == "__main__":
    m.main()
