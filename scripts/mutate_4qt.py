"""§4QT battery — the render look on the critic pool (2026-10-22). Reuses the §4QA harness.
    python3 scripts/mutate_4qt.py [all|check|rerun C1 C2 …]
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "mutate_4qa", os.path.join(os.path.dirname(os.path.abspath(__file__)), "mutate_4qa.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

IG = "src/ghost_agent/tools/image_gen.py"
VF = "src/ghost_agent/core/verifier.py"
LG = "src/ghost_agent/utils/logging.py"
m.PINS = ["tests/test_4qt_look_critic.py", "tests/test_4qu_check_critic.py", "tests/test_4qp_render_look.py", "tests/test_4oz_subjects_collage.py",
          "tests/test_4lm_vision.py", "tests/test_4mg_image_subjects.py"]


def _tests():
    import glob, subprocess
    pat = r"image_gen\|tool_generate_image\|look_on_critic\|LOOK_POOL_ENV\|look_critic\|check_critic\|visual_check_on_critic\|PROMPT_ARMS\|_call_llm_vision\|verify_visual"
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(m.ROOT, "tests", "test_*.py"))),
                         capture_output=True, text=True).stdout.split()
    return sorted({os.path.relpath(t, m.ROOT) for t in out} | set(m.PINS))


m._tests = _tests
m.MUT = [
    ("CONTROL no-op (must SURVIVE)", IG, "def look_on_critic(llm_client) -> bool:", "# ctl\ndef look_on_critic(llm_client) -> bool:"),
    ("CONTROL known-bad (must DIE): the switch ignores the pool's absence", IG,
     "    if not isinstance(clients, (list, tuple)) or not clients:\n        return False\n    if os.environ.get(LOOK_POOL_ENV",
     "    if False:\n        return False\n    if os.environ.get(LOOK_POOL_ENV"),
    ("C1 any switch value moves the look", IG,
     '    if os.environ.get(LOOK_POOL_ENV, "").strip().lower() == "critic":',
     '    if os.environ.get(LOOK_POOL_ENV, "").strip():'),
    ("C2 the critic leg may land on the main model", IG,
     "                    resp = await llm_client.chat_completion(payload, use_critic=True, off_main_only=True,",
     "                    resp = await llm_client.chat_completion(payload, use_critic=True, off_main_only=False,"),
    ("C3 the critic leg waits the vision pool's full timeout", IG,
     "                                                            timeout=LOOK_CRITIC_TIMEOUT_S, task_label=\"render-look\",",
     "                                                            timeout=LOOK_TIMEOUT_S, task_label=\"render-look\","),
    ("C4 a failed critic look is not retried on the vision pool", IG,
     "                    resp = None\n            if resp is None:\n                resp = await llm_client.chat_completion(payload, use_vision=True, timeout=LOOK_TIMEOUT_S)",
     "                    return \"\"\n            if resp is None:\n                resp = await llm_client.chat_completion(payload, use_vision=True, timeout=LOOK_TIMEOUT_S)"),
    ("C5 a cancel during the critic look is swallowed into a fallback", IG,
     "                except Exception as _ce:  # noqa: BLE001 — a cancel is BaseException and unwinds",
     "                except BaseException as _ce:  # noqa: BLE001"),
    ("C6 the critic arm does not imply the look", IG,
     "        return prompt_arm() in (LOOK_ARM, LOOK_CRITIC_ARM)     # §4QT: the critic arm implies the look",
     "        return prompt_arm() == LOOK_ARM"),
    ("C7 the critic arm is not honoured by the pool choice", IG,
     "        return prompt_arm() == LOOK_CRITIC_ARM\n    except Exception:  # noqa: BLE001\n        return False\n\n\ndef visual_check_on_critic",
     "        return False\n    except Exception:  # noqa: BLE001\n        return False\n\n\ndef visual_check_on_critic"),
    ("C8 the arm is not registered", LG, '"render_look",\n                                    "look_critic", "check_critic"})', '"render_look", "check_critic"})'),
    ("C9 the collage verdict moves to the critic pool too", IG,
     "            resp = await llm_client.chat_completion(payload, use_vision=True, timeout=COLLAGE_CHECK_TIMEOUT_S)",
     "            resp = await llm_client.chat_completion(payload, use_critic=look_on_critic(llm_client), use_vision=not look_on_critic(llm_client), timeout=COLLAGE_CHECK_TIMEOUT_S)"),
    ("V1 the pixel check ignores the switch", VF,
     "            _on_critic = bool(_check_on_critic(self.llm_client))",
     "            _on_critic = False"),
    ("V2 the pixel check's critic leg may land on the main model", VF,
     "                        payload, use_critic=True, off_main_only=True, timeout=_VISUAL_CRITIC_TIMEOUT_S,\n                        slot_wait=(_DEFERRED_SLOT_WAIT_S if _deferred else _VERIFY_SLOT_WAIT_S),",
     "                        payload, use_critic=True, off_main_only=False, timeout=_VISUAL_CRITIC_TIMEOUT_S,\n                        slot_wait=(_DEFERRED_SLOT_WAIT_S if _deferred else _VERIFY_SLOT_WAIT_S),"),
    ("X1 the check's own switch is ignored (§4QU)", IG,
     '    if os.environ.get(VISUAL_CHECK_POOL_ENV, "").strip().lower() == "critic":\n        return True',
     '    if False:\n        return True'),
    ("X2 the look switch no longer implies the check", IG,
     "    if look_on_critic(llm_client):\n        return True\n    clients = getattr(llm_client, \"critic_clients\", None)",
     "    if False:\n        return True\n    clients = getattr(llm_client, \"critic_clients\", None)"),
    ("X3 the check arm is not honoured", IG,
     "        return prompt_arm() == CHECK_CRITIC_ARM", "        return False"),
    ("X4 the check arm is not registered", LG, '"look_critic", "check_critic"})', '"look_critic"})'),
    ("X5 the check switch ignores the pool's absence", IG,
     '    if not isinstance(clients, (list, tuple)) or not clients:\n        return False\n    if os.environ.get(VISUAL_CHECK_POOL_ENV',
     '    if False:\n        return False\n    if os.environ.get(VISUAL_CHECK_POOL_ENV'),
    ("X6 the check borrows the look's bound again", VF,
     "timeout=_VISUAL_CRITIC_TIMEOUT_S,\n                        slot_wait=(_DEFERRED_SLOT_WAIT_S",
     "timeout=30.0,\n                        slot_wait=(_DEFERRED_SLOT_WAIT_S"),
    ("X7 the check switch moves the look too", IG,
     "    if os.environ.get(LOOK_POOL_ENV, \"\").strip().lower() == \"critic\":\n        return True\n    try:\n        from ..utils.logging import prompt_arm\n        return prompt_arm() == LOOK_CRITIC_ARM",
     "    if os.environ.get(LOOK_POOL_ENV, \"\").strip().lower() == \"critic\" or os.environ.get(VISUAL_CHECK_POOL_ENV, \"\").strip().lower() == \"critic\":\n        return True\n    try:\n        from ..utils.logging import prompt_arm\n        return prompt_arm() == LOOK_CRITIC_ARM"),
    ("V4 a deferred pixel check that found Nova busy is re-run on the main slot (r1 MAJOR 1)", VF,
     '                if _deferred and getattr(exc, "saturated", False):\n                    if isinstance(_deferred, dict):\n                        _deferred["nova_busy"] = True',
     '                if False:\n                    if isinstance(_deferred, dict):\n                        _deferred["nova_busy"] = True'),
    ("V5 the pixel check ignores a verdict that already found Nova busy", VF,
     '            _deferred = deferred_verdict_context.get()\n            if isinstance(_deferred, dict) and _deferred.get("nova_busy"):\n                return {}\n            try:\n                _crit_kw',
     '            _deferred = deferred_verdict_context.get()\n            if False:\n                return {}\n            try:\n                _crit_kw'),
    ("V6 a deferred pixel check queues only the in-turn slot wait", VF,
     "                        slot_wait=(_DEFERRED_SLOT_WAIT_S if _deferred else _VERIFY_SLOT_WAIT_S),\n                        **_crit_kw)",
     "                        slot_wait=_VERIFY_SLOT_WAIT_S,\n                        **_crit_kw)"),
    ("V7 the pixel check's critic leg is foreground even when nobody is live", VF,
     "                _crit_kw = dict(_bounded_fallback_kwargs(self.llm_client))\n                _crit_kw.pop(\"timeout\", None)\n                # a deferred call keeps the helper's \"deferred-verdict\" label —",
     "                _crit_kw = {}\n                # a deferred call keeps the helper's \"deferred-verdict\" label —"),
    ("V8 a deferred pixel check skips the lane", VF,
     "                _lane = _critic_lane() if _deferred else contextlib.nullcontext()\n                async with _lane:\n                    if isinstance(_deferred, dict) and _deferred.get(\"nova_busy\"):\n                        raise LaneBusy",
     "                _lane = contextlib.nullcontext()\n                async with _lane:\n                    if isinstance(_deferred, dict) and _deferred.get(\"nova_busy\"):\n                        raise LaneBusy"),
    ("C11 the look's critic leg may queue the client's full 90 s (r1 MAJOR 2)", IG,
     "                                                            slot_wait=LOOK_CRITIC_SLOT_WAIT_S,\n                                                            total_budget=LOOK_CRITIC_TIMEOUT_S)",
     "                                                            )"),
    ("C12 the look's critic leg bounds the POST but not the queue", IG,
     "                                                            total_budget=LOOK_CRITIC_TIMEOUT_S)",
     "                                                            total_budget=None)"),
    ("V3 a failed critic pixel check is not asked of the vision pool", VF,
     '                logger.warning("Visual verifier call on the critic pool failed (%s) — asking the vision pool",\n                               exc.__class__.__name__)',
     '                logger.warning("Visual verifier call on the critic pool failed (%s) — asking the vision pool",\n                               exc.__class__.__name__)\n                return {}'),
    ("C10 the vision fallback runs even after a usable critic answer", IG,
     "            if resp is None:\n                resp = await llm_client.chat_completion(payload, use_vision=True, timeout=LOOK_TIMEOUT_S)",
     "            if resp is None or not _scene_of((resp[\"choices\"][0][\"message\"].get(\"content\") or \"\")):\n                resp = await llm_client.chat_completion(payload, use_vision=True, timeout=LOOK_TIMEOUT_S)"),
]

if __name__ == "__main__":
    m.main()
