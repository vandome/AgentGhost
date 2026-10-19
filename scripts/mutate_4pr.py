#!/usr/bin/env python3
"""§4PR battery — web client review fixes (app.js, console modules, sw.js,
server.py, webpush_notify.py) and the face fixes (matrix_graph.js),
2026-10-17. Same harness as §4PM, on a tree COPY.

    BATTERY_ROOT=/path/to/copy python3 scripts/mutate_4pr.py [all|rerun W1 …]
"""
import os, runpy
_H = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mutate_4pm.py"), run_name="harness")
APP = "interface/static/app.js"
NOT = "interface/static/notifications.js"
SES = "interface/static/sessions.js"
WSJ = "interface/static/workspace.js"
SW = "interface/static/sw.js"
CSS = "interface/static/style.css"
SRV = "interface/server.py"
WP = "interface/webpush_notify.py"
MUT = [
    ("CONTROL no-op (must SURVIVE)", APP, "function _sendComposed(text) {", "// ctl\nfunction _sendComposed(text) {"),
    ("CONTROL known-bad (must DIE): mermaid back in the page", APP, "    papaparse: { src: '/static/vendor/papaparse.min.js', global: 'Papa' },", "    papaparse: { src: '/static/vendor/papaparse.min.js', global: 'Papa' },\n    mermaid: { src: '/static/vendor/mermaid.min.js', global: 'mermaid' },"),
    ("W1 srcdoc source unescaped", APP, "    const src = JSON.stringify(String(codeText)).replace(/</g, '\\\\u003c');", "    const src = JSON.stringify(String(codeText));"),
    ("W2 composed send drops the draft", APP, "    if (draft.trim()) {\n        chatInput.value = draft;", "    if (false) {\n        chatInput.value = draft;"),
    ("W3 composed send clobbers during a turn", APP, "        chatInput.value = draft.trim() ? `${draft.replace(/\\s+$/, '')} ${text}` : text;", "        chatInput.value = text;"),
    ("W4 mic records after release", APP, "            if (!_micPressHeld) {", "            if (false) {"),
    ("W5 release not recorded", APP, "        _micPressHeld = false;\n        if (!isRecording || !mediaRecorder) return;", "        if (!isRecording || !mediaRecorder) return;"),
    ("W6 inflight fetch not shared", APP, "    if (_authedBlobInflight.has(rawSrc)) return _authedBlobInflight.get(rawSrc);", ""),
    ("W7 live turn does not stop the recovery", APP, "    return !!isProcessingRequest;", "    return false;"),
    ("W8 switch keeps the ticker", APP, "        if (!isProcessingRequest) stopTurnTicker();", "        if (false) stopTurnTicker();"),
    ("W8b switch kills a live turn's ticker", APP, "        if (!isProcessingRequest) stopTurnTicker();", "        stopTurnTicker();"),
    ("W24 send leaves the recovery pending", APP, "        _autoOpenedImgSrcs.clear();   // a new turn may auto-open its images\n        _endRecoveryForNewTurn();", "        _autoOpenedImgSrcs.clear();   // a new turn may auto-open its images"),
    ("W25 ending clears the wrong handle", APP, "    clearInflightHandle(_reconcileHandle);", "    clearInflightHandle();"),
    ("W28 a running poll is not ended by a send", APP, "    if (!reconcilePollTimer && !_reconcileRunning) return;", "    if (!reconcilePollTimer) return;"),
    ("W29 send does not cancel the running poll", APP, "    _reconcileGen += 1;\n", ""),
    ("W30 poll armed during a live turn", APP, "    if (isProcessingRequest) return;\n    _reconcileHandle = handle || null;", "    _reconcileHandle = handle || null;"),
    ("W31 generation ignored", APP, "    if (gen !== undefined && gen !== _reconcileGen) return true;   // ended by a send", ""),
    ("W32 resume fights the recovery", APP, "                    if (reconcilePollTimer || _reconcileRunning) return;", ""),
    ("W33 lost resume offers a Retry", APP, "                } else if (resuming) {", "                } else if (false) {"),
    ("W34 two runs at once", APP, "    if (_reconcileRunning) return;\n    const gen = _reconcileGen;", "    const gen = _reconcileGen;"),
    ("W35 edit leaves the recovery", APP, "    // the cut message back\n    _endRecoveryForNewTurn(true);", "    // the cut message back"),
    ("W36 switch keeps a spent budget", APP, "        // the recovery ends here: the next one starts with a full budget\n        _reconcileAnnounced = false;", "        // the recovery ends here: the next one starts with a full budget"),
    ("W37 turns fetch unbounded", APP, "            const tr = await fetch('/api/turns', { signal: _timeoutSignal(_RECONCILE_FETCH_MS) });", "            const tr = await fetch('/api/turns');"),
    ("W38 session fetch unbounded", APP, "                { signal: _timeoutSignal(_RECONCILE_FETCH_MS) });\n", "                {});\n"),
    ("W39 settled turn keeps its task id", APP, "        if (handle && handle.taskId && handle.taskId === currentTaskId) currentTaskId = null;", ""),
    ("W26 over-long line leaves a fragment", SRV, "                dropping = False          # the rest of the over-long line\n                continue", "                dropping = False          # the rest of the over-long line"),
    ("W27 long line not shortened", SRV, "            if len(line.rstrip(b\"\\r\\n\")) > _LOG_LINE_SHOWN:   # a console line, not a dump", "            if False:"),
    ("W9 silent drop of a POST that never left", APP, "            if (currentTaskId && (errMsg.includes('load failed')", "            if ((errMsg.includes('load failed')"),
    ("W10 render seq not bumped", APP, "    _renderSeq += 1;\n", ""),
    ("W11 closer closes on its openers", NOT, "        if (panel && !panel.classList.contains('hidden') && !opener", "        if (panel && !panel.classList.contains('hidden')"),
    ("W12 clear-all empties seen", NOT, "        records = [];\n", "        records = [];\n        seen = new Set();\n"),
    ("W13 prune deletes unlisted ids", SES, "            if (!keep.has(sid) && listedIds.has(sid)) Core.safeStorage.remove(k);", "            if (!keep.has(sid)) Core.safeStorage.remove(k);"),
    ("W14 closed rail not inert", WSJ, "        rail.inert = !open;", "        rail.inert = false;"),
    ("W15 precache from the HTTP cache", SW, "{ credentials: 'same-origin', cache: 'reload' }", "{ credentials: 'same-origin' }"),
    ("W16 toast under the modal", CSS, "    z-index: 155;", "    z-index: 140;"),
    ("W17 STT multipart 502", SRV, "        return _err_json(e.status_code, str(e.detail))", "        return _upstream_failure(\"STT proxy error\", e)"),
    ("W18 TTS non-string accepted", SRV, "        if not isinstance(body.get(\"text\", \"\"), str):   # §4PR: was a 502", "        if False:"),
    ("W19 over-long line kept whole", SRV, "            if len(line) > _LOG_LINE_LIMIT:\n                logger.warning", "            if False:\n                logger.warning"),
    ("W20 push keys unchecked", WP, "            or not isinstance(keys.get(\"p256dh\"), str) or not keys.get(\"p256dh\")", "            or not keys.get(\"p256dh\")"),
    ("W21 whole client dict stored", WP, "    subscription = {\"endpoint\": endpoint,", "    _unused = {\"endpoint\": endpoint,"),
    ("W22 workspace saved before rebind", APP, "            } catch (err) { /* workspace modules not loaded */ }\n            saveChatState();\n", "            } catch (err) { /* workspace modules not loaded */ }\n"),
    ("W23 image auto-opens every frame", APP, "    if (!restoringAtStart && !_autoOpenedImgSrcs.has(rawSrc)) {", "    if (!restoringAtStart) {"),
]
_H["MUT"][:] = MUT
_H["PINS"][:] = ["tests/test_interface_review_4pr.py", "tests/test_4ml_interfaces.py",
                 "tests/test_interface_batch2_2026_09_11.py", "tests/test_webui_console_review.py",
                 "tests/test_interface_upload_ask.py", "tests/test_interface_review_fixes.py"]
g = _H["main"].__globals__
g["MUT"], g["PINS"] = MUT, _H["PINS"]
# The harness compile-checks with py_compile; JS/CSS mutants need their own
# syntax check (node for JS, none for CSS).
import subprocess as _sp, types as _types
def _run(args, *a, **k):
    if len(args) >= 4 and args[1:3] == ["-m", "py_compile"] and not str(args[3]).endswith(".py"):
        path = str(args[3])
        if path.endswith(".js"):
            if path.endswith("sw.js"):
                return _sp.run(["node", "--check", path], capture_output=True)
            with open(path, "rb") as fh:
                return _sp.run(["node", "--input-type=module", "--check"], stdin=fh, capture_output=True)
        return _sp.CompletedProcess(args, 0)
    return _sp.run(args, *a, **k)
g["subprocess"] = _types.SimpleNamespace(**{k: getattr(_sp, k) for k in dir(_sp) if not k.startswith("__")})
g["subprocess"].run = _run
g["_FULL_PAT"] = r"interface/static\|interface.server\|webpush_notify"
_LOCATION_BOUND = {"tests/test_4pi_youtube_review.py", "tests/test_4kh_youtube_canary.py", "tests/test_ghost_cli.py"}
_orig_grep = g["_grep"]
g["_grep"] = lambda pat: _orig_grep(pat) - _LOCATION_BOUND
if __name__ == "__main__":
    _H["main"]()
