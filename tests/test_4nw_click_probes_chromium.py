"""§4NW: the two in-page probes behind "⚠ NO EFFECT", run in REAL Chromium
(the review found every false claim only in a real page: shadow DOM, a
styled checkbox's knob, an iframe). Skipped where Chromium is not installed."""
import asyncio

import pytest

from ghost_agent.tools import browser_runner as R

pw = pytest.importorskip("playwright.async_api")

PAGE = """<html><body style="margin:0">
<div id="icon" ondblclick="document.getElementById('w').hidden=false"
     style="position:absolute;left:10px;top:10px;width:60px;height:60px">calc</div>
<div id="win" style="position:absolute;left:0;top:0;width:300px;height:120px;background:#333">terminal</div>
<div id="free" ondblclick="1" style="position:absolute;left:400px;top:10px;width:60px;height:60px">free</div>
<label id="sw" style="position:absolute;left:400px;top:200px;display:block;width:60px;height:30px">
  <input id="cb" type="checkbox" style="position:absolute;opacity:0;left:0;top:0;width:60px;height:30px">
  <span style="position:absolute;left:0;top:0;width:60px;height:30px;background:#0a0"></span></label>
<div id="host" style="position:absolute;left:400px;top:300px"></div>
<div id="w" hidden>Calculator</div>
<input id="field" ondblclick="1" style="position:absolute;left:600px;top:10px">
<span id="both" onclick="1" ondblclick="1" style="position:absolute;left:600px;top:100px">both</span>
<div id="titlebar" ondblclick="1" style="position:absolute;left:600px;top:150px"><button id="close">x</button><span id="label">title</span></div>
<input id="cb2" type="checkbox" style="position:absolute;left:700px;top:300px;width:40px;height:20px;opacity:0">
<label for="cb2" style="position:absolute;left:700px;top:300px;width:40px;height:20px;background:#00a"></label>
<a id="lnk" href="#" style="position:absolute;left:800px;top:300px;display:block;width:40px;height:40px">
  <span id="pic" style="position:absolute;left:0;top:0;width:40px;height:40px">p</span>
  <span style="position:absolute;left:0;top:0;width:40px;height:40px;background:#a00"></span></a>
<script>
const s = document.getElementById('host').attachShadow({mode: 'open'});
s.innerHTML = '<button id="in" style="width:80px;height:30px">x</button>' +
              '<span id="sd" ondblclick="1" style="display:inline-block;width:40px;height:20px">y</span>';
</script></body></html>"""


def _run(coro):
    return asyncio.run(coro)


async def _probe():
    async with pw.async_playwright() as p:
        try:
            b = await p.chromium.launch()
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"no Chromium: {e}")
        pg = await b.new_page(viewport={"width": 1000, "height": 600})
        await pg.set_content(PAGE)
        out = {
            "covered_icon": await pg.eval_on_selector("#icon", R._COVERED_BY_JS),
            "free_icon": await pg.eval_on_selector("#free", R._COVERED_BY_JS),
            "checkbox_under_knob": await pg.eval_on_selector("#cb", R._COVERED_BY_JS),
            "shadow_button": await pg.eval_on_selector("#host >> #in", R._COVERED_BY_JS),
            "checkbox_under_for_label": await pg.eval_on_selector("#cb2", R._COVERED_BY_JS),
            "inside_link_overlay": await pg.eval_on_selector("#pic", R._COVERED_BY_JS),
            "dbl_icon": await pg.eval_on_selector("#icon", R._DBLCLICK_ONLY_JS),
            "dbl_field": await pg.eval_on_selector("#field", R._DBLCLICK_ONLY_JS),
            "dbl_and_click": await pg.eval_on_selector("#both", R._DBLCLICK_ONLY_JS),
            "close_in_titlebar": await pg.eval_on_selector("#close", R._DBLCLICK_ONLY_JS),
            "label_in_titlebar": await pg.eval_on_selector("#label", R._DBLCLICK_ONLY_JS),
            "dbl_in_shadow": await pg.eval_on_selector("#host >> #sd", R._DBLCLICK_ONLY_JS),
        }
        await b.close()
        return out


def test_the_probes_tell_a_covered_target_from_one_that_was_reached():
    o = _run(_probe())
    assert o["covered_icon"].startswith("div#win")          # the live case: under the Terminal
    assert o["free_icon"] == ""
    assert o["checkbox_under_knob"] == ""                   # the knob is inside its label
    assert o["shadow_button"] == ""                         # hit-tested in its own shadow root
    assert o["checkbox_under_for_label"] == ""              # its <label for=…> passes the click on
    assert o["inside_link_overlay"] == ""                   # the overlay is inside the same link
    assert o["dbl_icon"] == "this element"                   # the handler is on the clicked element
    assert o["dbl_field"] is False                          # a form control: property changes
    assert o["dbl_and_click"] is False                      # it handles the click too
    assert o["close_in_titlebar"] is False                  # a button is a control of its own (review)
    assert o["label_in_titlebar"] == "div#titlebar"         # a plain child reaches the bar's handler
    assert o["dbl_in_shadow"] is False                      # the page signature cannot see it
