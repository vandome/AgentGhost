"""§4QG (2026-10-21): fixes for the image-turn review findings in core/agent.py.

1 clarify-first blocked every single word ("watercolor", "bigger",
  "ακουαρέλα"), even the answer to its own question.
2 the blind-regeneration block never said the call did NOT run (fecd88b3);
  the Turn Outcome line listed the refused call as a tool run.
3 the render-time estimate ignored the `subject` alias (deadline check).
4 the tool KNEW the picture was two panels; the reply said "as requested"
  (493f8587) — the fixed note is appended.
5 the image-markdown guard ordered "image_generation NOW" + a strike for a
  re-shown earlier image / an uploaded photo.
6 a subject the model misspelled against the user's words got "no photo"
  and the reply blamed the user's spelling (7d6617ad).
Every test names the world it fails in.
"""
import ast
import inspect
import json
import time

import pytest

from ghost_agent.core import agent as A
from ghost_agent.core.agent import CLARIFY_FIRST_REPLY, _clarify_first_block, _is_low_information_message
from ghost_agent.core.strikes import StrikeLedger
from ghost_agent.tools.outcome import ToolOutcome
import tests.test_4jj_search_yield_steer as H

IMG_REPLY = "![generated image](/api/download/gen_1a2b3c4d.png)\n\nA warm bakery at dusk."


def _conv(prev, user):
    return [{"role": "user", "content": "draw a bakery"}, {"role": "assistant", "content": prev},
            {"role": "user", "content": user}]


# ── 1. clarify-first ───────────────────────────────────────────────────────

@pytest.mark.parametrize("word", ["watercolor", "bigger", "darker", "Stalin", "ακουαρέλα",
                                  "Watercolor!", "hyper-real"])
@pytest.mark.parametrize("record", [True, False])
def test_one_real_word_after_an_image_turn_runs(word, record):
    """Fails in the pre-§4QG world: any single token was 'low information'."""
    assert not _is_low_information_message(word)
    assert _clarify_first_block("image_generation", word, _conv(IMG_REPLY, word), record) is None


@pytest.mark.parametrize("junk", ["k", "?", "emp1", "...", "ok", "now", "thanks", "lol", "hmm", "👍", "go2"])
def test_junk_and_fillers_still_ask(junk):
    assert _is_low_information_message(junk)
    assert _clarify_first_block("image_generation", junk, _conv(IMG_REPLY, junk), True)


@pytest.mark.parametrize("record", [True, False])
@pytest.mark.parametrize("word", ["watercolor", "ακουαρέλα", "bigger"])
def test_a_one_word_answer_to_the_guards_own_question_runs(word, record):
    """The live loop: "watercolor" → our question → "watercolor" → blocked
    again (the fixed question counted as the costly turn)."""
    msgs = [{"role": "assistant", "content": IMG_REPLY}, {"role": "user", "content": word},
            {"role": "assistant", "content": CLARIFY_FIRST_REPLY}, {"role": "user", "content": word}]
    assert _clarify_first_block("image_generation", word, msgs, record) is None


def test_the_one_word_rule_is_linear():
    """A new lexical regex needs a linearity test (memory rule)."""
    for s in ("a-" * 50000 + "1", "ά'" * 50000 + "?", "x" * 200000 + "1"):
        t0 = time.perf_counter()
        A._is_one_real_word(s)
        assert time.perf_counter() - t0 < 0.5


# ── 2. the blind-regeneration block, the Turn Outcome line ─────────────────

def _img_row(name, extra=""):
    return {"role": "tool", "name": "image_generation",
            "content": f"SUCCESS: Image generated and saved to sandbox. {extra}"
                       f"![generated image](/api/download/{name})\n"}


def test_the_blind_block_says_the_call_did_not_run_and_names_the_existing_file():
    """Fails in the fecd88b3 world: the model thought the refused call made a second image."""
    out = A._blind_regeneration_block("image_generation", [_img_row("gen_a0828dc1.png")], [], "x")
    assert out.startswith("SYSTEM BLOCK — BLOCKED: this image_generation call did NOT run; "
                          "nothing new was generated.")
    assert "gen_a0828dc1.png" in out and "made no file" in out
    batch = A._blind_regeneration_block("image_generation", [], [], "x", batch_images=1)
    assert batch.startswith("SYSTEM BLOCK — BLOCKED: this image_generation call did NOT run")


def test_the_turn_outcome_line_labels_a_refused_call(monkeypatch):
    """Fails where the line read 'tools: image_generation, image_generation'."""
    from tests.helpers import make_agent
    agent = make_agent()
    lines = []
    monkeypatch.setattr(A, "pretty_log", lambda title, content=None, **kw: lines.append((title, str(content))))
    tools = [_img_row("gen_a.png"), {"role": "tool", "name": "image_generation",
                                     "content": "SYSTEM BLOCK — BLOCKED", "_synthetic": True}]
    agent._emit_turn_outcome_line(req_id="sim-4qg", trajectory_id=None, final_content="x", tools=tools,
                                  execution_failure_count=0, exec_terminal=False,
                                  unacked_total_failure=False, budget_exhausted=False, shape_failed=False)
    out = [c for t, c in lines if t == "Turn Outcome"][-1]
    assert "tools: image_generation, image_generation (not run)" in out, out


# ── 3. the render-time estimate reads the whole call ───────────────────────

def test_the_estimate_gets_the_subject_alias_as_subjects(monkeypatch):
    """Fails where `subject` beside a prompt (the tool's subjects alias) was
    timed as a plain generate."""
    import ghost_agent.tools.image_gen as IG
    seen = []
    monkeypatch.setattr(IG, "expected_render_seconds", lambda a: seen.append(dict(a)) or 1.0)
    A.declared_wait_s("image_generation", {"prompt": "two leaders shake hands",
                                           "subject": "Joe Biden, Xi Jinping"})
    assert seen[-1].get("subjects") == "Joe Biden, Xi Jinping"
    A.declared_wait_s("image_generation", {"subject": "a cat"})          # a lone subject IS the prompt
    assert "subjects" not in seen[-1]
    A.declared_wait_s("image_generation", {"prompt": "x", "subjects": ["A", "B"], "subject": "z"})
    assert seen[-1]["subjects"] == ["A", "B"]


# ── 4. the two-panel note ──────────────────────────────────────────────────

PANELS = ("NOTE: the picture came out as separate side-by-side PANELS 3 times (the model copied "
          "the reference layout). Tell the user plainly that it is a two-panel picture.\n\n")


def test_the_two_panel_note_is_appended_when_the_reply_hides_it():
    """Fails in the 493f8587 world: 'Generated a scene … as requested'."""
    rows = [_img_row("gen_5a.png", PANELS)]
    reply = "![generated image](/api/download/gen_5a.png)\n\nGenerated a scene of the two as requested."
    assert A._two_panel_note(reply, rows, "draw them hugging") == A.TWO_PANEL_NOTE
    assert A._two_panel_note(reply, rows, "ζωγράφισέ τους") == A.TWO_PANEL_NOTE_EL


@pytest.mark.parametrize("reply,rows", [
    ("![generated image](/api/download/gen_5a.png)\n\nIt came out as two side-by-side panels.",
     [_img_row("gen_5a.png", PANELS)]),                                   # the reply says so
    # §4QJ: only an unmistakable phrase counts ("Βγήκε δίπλα-δίπλα" is how people stand too)
    ("![generated image](/api/download/gen_5a.png)\n\nΒγήκε σε δύο πάνελ.", [_img_row("gen_5a.png", PANELS)]),
    ("![generated image](/api/download/gen_5a.png)\n\nA scene.", [_img_row("gen_5a.png")]),   # no NOTE
    ("No picture shown.", [_img_row("gen_5a.png", PANELS)]),                # the image is not shown
    ("![generated image](/api/download/gen_5a.png)",
     [{**_img_row("gen_5a.png", PANELS), "_synthetic": True}]),             # not a real run
])
def test_no_two_panel_note_otherwise(reply, rows):
    assert A._two_panel_note(reply, rows, "x") == ""


def test_a_vague_side_by_side_still_gets_the_note():
    """§4QJ: a miss only shows the note twice; a false match hid it."""
    rows = [_img_row("gen_5a.png", PANELS)]
    reply = "![generated image](/api/download/gen_5a.png)\n\nΒγήκε δίπλα-δίπλα."
    assert A._two_panel_note(reply, rows, "ζωγράφισέ τους") == A.TWO_PANEL_NOTE_EL


def test_both_reply_paths_append_the_two_panel_note():
    """A correct helper nobody calls is the §4KB lesson."""
    src = ast.parse(inspect.getsource(A))
    for fn in ("_finalize_and_return", "_stream_final_generation"):
        node = next(n for n in ast.walk(src) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and n.name == fn)
        assert any(isinstance(c, ast.Call) and getattr(c.func, "id", "") == "_two_panel_note"
                   for c in ast.walk(node)), fn


# ── 5. the image-markdown guard ────────────────────────────────────────────

def _long_history(first_rows):
    filler = [{"role": "user" if i % 2 else "assistant", "content": f"chat {i}"} for i in range(10)]
    return first_rows + filler


def test_an_earlier_image_far_back_in_the_conversation_is_backed():
    """Fails in the old 4-message window world: re-showing an older picture
    ordered a new render."""
    msgs = _long_history([{"role": "assistant", "content": IMG_REPLY}])
    assert A._image_markdown_alert("Here it is again: ![x](/api/download/gen_1a2b3c4d.png)", msgs) is None


def test_an_uploaded_photo_named_in_a_vision_call_is_backed():
    msgs = _long_history([{"role": "assistant", "content": "", "tool_calls": [
        {"id": "v1", "function": {"name": "vision_analysis",
                                  "arguments": json.dumps({"target": "upload_cat.jpg"})}}]}])
    assert A._image_markdown_alert("![your cat](/api/download/upload_cat.jpg)", msgs) is None


def test_a_file_in_the_workspace_is_backed(tmp_path):
    (tmp_path / "gen_99999999.png").write_bytes(b"x")
    assert A._image_markdown_alert("![x](/api/download/gen_99999999.png)", [], tmp_path) is None


def test_an_unbacked_link_draws_one_steer_that_never_orders_a_render(tmp_path):
    out = A._image_markdown_alert("![x](/api/download/gen_deadbeef.png)", [], tmp_path)
    assert out and out.startswith(A._UNBACKED_IMAGE_ALERT_HEAD) and "gen_deadbeef.png" in out
    assert "MUST" not in out and "NOW" not in out and "Only if the user's own message asks" in out
    # once per request
    again = [{"role": "user", "content": out}]
    assert A._image_markdown_alert("![x](/api/download/gen_deadbeef.png)", again, tmp_path) is None


def test_the_guard_no_longer_orders_image_generation_or_strikes():
    tree = ast.parse(inspect.getsource(A))
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and "for `image_generation` NOW" in n.value]
    for n in ast.walk(tree):
        if isinstance(n, ast.If) and any(isinstance(c, ast.Name) and c.id == "_img_alert"
                                         for c in ast.walk(n.test)):
            assert not any(isinstance(c, ast.AugAssign) and getattr(c.target, "id", "") == "execution_failure_count"
                           for c in ast.walk(n))
            break
    else:
        pytest.fail("the guard does not read _image_markdown_alert")


# ── 6. the subject spelling ────────────────────────────────────────────────

@pytest.mark.parametrize("name,user,fix", [
    ("Zoe Konstantinou", ["draw Zoe Konstantopoulou hugging Joseph Stalin"], "Zoe Konstantopoulou"),
    ("Konstantinou", ["Konstantopoulou with Stalin"], "Konstantopoulou"),
    ("Zoe Konstantopoulou", ["draw Zoe Konstantopoulou"], None),          # the user's own spelling
    ("Zoe Konstantinou", ["Konstantinou, not Konstantopoulou"], None),    # …even beside a variant
    ("Joe Biden", ["draw Bidens dog"], None),                              # an inflection
    ("Joe Biden", ["draw the president"], None),                           # no variant at all
    ("Ζωή Κωνσταντοπούλου", ["Zoe Konstantopoulou"], None),                # another script
    ("Al Bo", ["Alberto"], None),                                          # too short
])
def test_the_spelling_fix_table(name, user, fix):
    assert A._subject_spelling_fix(name, user) == fix


def _no_photo(missing, found):
    return ("ERROR: no usable photo found for " + "; ".join(f"{m} — LookupError" for m in missing)
            + ". Nothing was rendered. If the name may be misspelled, you may retry once with the "
              "corrected spelling.\n[subjects: "
            + json.dumps({"missing": missing, "found": found}) + "]")


def test_the_hint_keeps_the_machine_tail_and_lets_the_users_spelling_through():
    body = _no_photo(["Konstantinou"], ["Joseph Stalin"])
    out = A._missing_subject_spelling_hint(body, ["Konstantopoulou hugging Joseph Stalin"])
    assert "the user wrote 'Konstantopoulou'" in out and '["Joseph Stalin", "Konstantopoulou"]' in out
    assert out.rstrip().endswith("]") and out.rfind("\n[subjects: ") > out.find("SYSTEM NOTE")
    rows = [{"role": "tool", "name": "image_generation", "content": out}]
    # the user's spelling passes although it differs > 0.75 from the guess…
    assert A._missing_subject_block("image_generation",
                                    {"prompt": "x", "subjects": ["Joseph Stalin", "Konstantopoulou"]}, rows) is None
    # …and dropping the subject is still refused
    assert A._missing_subject_block("image_generation", {"prompt": "x", "subjects": ["Joseph Stalin"]}, rows)


def test_no_hint_when_the_model_used_the_users_spelling():
    body = _no_photo(["Zoe Konstantopoulou"], [])
    assert A._missing_subject_spelling_hint(body, ["Zoe Konstantopoulou please"]) == body


def test_the_hint_regex_is_linear():
    s = "SYSTEM NOTE — spelling: " + "subjects=[" * 3000 + "x" * 5000
    t0 = time.perf_counter()
    A._SPELLING_HINT_SUBJECTS_RE.search(s)
    assert time.perf_counter() - t0 < 1.0


@pytest.mark.asyncio
async def test_the_dispatch_puts_the_hint_on_the_tool_row():
    """End to end through the batch: fails in the 7d6617ad world."""
    agent = H._agent()

    async def img(**kw):
        return ToolOutcome.rejected(_no_photo(["Zoe Konstantinou"], ["Joseph Stalin"]),
                                    reason_code="subject_photo_missing")
    agent.available_tools = {"image_generation": img}
    ts = H._ts([("image_generation", {"prompt": "Zoe Konstantinou hugging Joseph Stalin",
                                      "subjects": ["Zoe Konstantinou", "Joseph Stalin"]})],
               StrikeLedger(), set())
    ts.last_user_content = "draw Zoe Konstantopoulou hugging Joseph Stalin"
    await agent._dispatch_and_process_tool_batch(ts)
    row = [t for t in ts.tools_run_this_turn if t.get("name") == "image_generation"][-1]
    assert "the user wrote 'Konstantopoulou'" in str(row["content"])
    assert '"Joseph Stalin", "Zoe Konstantopoulou"' in str(row["content"])


def test_the_missing_subject_block_reads_a_singular_subject_as_the_subjects_list():
    """Fails where only `subjects` was read: a retry passing the found subject
    as `subject="Joseph Stalin"` read as dropping it (and a same-name retry
    via `subject=` escaped the 'same name again' stop)."""
    rows = [{"role": "tool", "name": "image_generation",
             "content": _no_photo(["Zoe Konstantinou"], ["Joseph Stalin"])}]
    same = A._missing_subject_block("image_generation",
                                    {"prompt": "the two hug", "subject": "Zoe Konstantinou"}, rows)
    assert same and "same name will return nothing again" in same
    # a lone `subject` with no prompt IS the prompt, not a subjects list → still the ask-first block
    lone = A._missing_subject_block("image_generation", {"subject": "Zoe Konstantinou"}, rows)
    assert lone and "ask the user first" in lone


# ── §4QG-b: fresh-eye review of the §4QG fixes ─────────────────────────────

_THREE_USER = ["Alexis Tsipras with Zoe Konstantopoulou and Nikos Xylouris"]


def _three_rows():
    body = _no_photo(["Zoe Konstantinou", "Nikos Xilouris"], ["Alexis Tsipras"])
    return [{"role": "tool", "name": "image_generation",
             "content": A._missing_subject_spelling_hint(body, _THREE_USER)}]


def test_one_fixed_spelling_does_not_excuse_a_second_unfixed_subject():
    """Fails in the §4QG world: `_respelled` asked whether ANY retry name was
    a suggestion, so fixing Zoe dropped Xylouris and the picture rendered."""
    blk = A._missing_subject_block(
        "image_generation", {"prompt": "a cafe", "subjects": ["Alexis Tsipras", "Zoe Konstantopoulou"]},
        _three_rows())
    assert blk and "ask the user first" in blk and "Nikos Xilouris" in blk
    assert "Zoe Konstantinou" not in blk          # the fixed one is not asked about


def test_the_hint_lists_only_fixed_and_found_names_and_says_to_ask_about_the_rest():
    """Fails in the §4QG world: the hint repeated 'Nikos Xilouris' unchanged
    in its subjects=, and following it exactly drew 'the same name will
    return nothing again'."""
    out = _three_rows()[0]["content"]
    m = A._SPELLING_HINT_SUBJECTS_RE.search(out)
    assert m and json.loads(m.group(1)) == ["Alexis Tsipras", "Zoe Konstantopoulou"]
    assert "Ask the user NOW about 'Nikos Xilouris'" in out and "do NOT retry now" in out
    follow = A._missing_subject_block(
        "image_generation", {"prompt": "a cafe", "subjects": json.loads(m.group(1))}, _three_rows())
    assert follow and "same name will return nothing again" not in follow
    assert "Nikos Xilouris" in follow


def test_both_fixed_retry_passes_and_one_suggestion_answers_only_its_own_name():
    user = ["Zoe Konstantopoulou and Nikos Xylouris"]
    body = _no_photo(["Zoe Konstantinou", "Nikos Xylourakis"], [])
    rows = [{"role": "tool", "name": "image_generation",
             "content": A._missing_subject_spelling_hint(body, user)}]
    assert "Retry image_generation ONCE" in rows[0]["content"]
    both = ["Zoe Konstantopoulou", "Nikos Xylouris"]
    assert A._missing_subject_block("image_generation", {"prompt": "x", "subjects": both}, rows) is None
    # one suggestion in the retry cannot stand for two missing names
    assert A._missing_subject_block("image_generation",
                                    {"prompt": "x", "subjects": ["Zoe Konstantopoulou"]}, rows)


@pytest.mark.parametrize("word", [
    "yess", "yea", "okey", "okk", "nope", "alright", "amazing", "lovely", "beautiful", "bravo",
    "haha", "hahaha", "lolol", "hmmmmm", "merci", "gracias", "danke", "grazie", "vale", "d'accord",
    "d’accord", "ωραίο", "τέλειο", "μπράβο", "χαχα", "ναί", "ΝΑΙ", "Yesss!", "thank-you",
    "agaaain", "Moooore", "ξανάαα"])
def test_fillers_in_any_spelling_still_ask(word):
    """Fails in the §4QG world: each counted as a one-word instruction and
    ran the GPU after an image turn."""
    assert _is_low_information_message(word)
    assert _clarify_first_block("image_generation", word, _conv(IMG_REPLY, word), True)


@pytest.mark.parametrize("word", ["watercolor", "bigger", "ακουαρέλα", "Stalin", "नमस्ते",
                                  "sepia", "banana", "hyper-real", "Παπαδόπουλος"])
def test_real_words_in_any_script_still_run(word):
    """नमस्ते fails in the §4QG world: its vowel signs are combining marks."""
    assert not _is_low_information_message(word)
    assert _clarify_first_block("image_generation", word, _conv(IMG_REPLY, word), True) is None


@pytest.mark.parametrize("s", ["a" + "-" * 50000 + "a", "-" * 50000, "a" + "!" * 50000,
                               "ά'" * 25000 + "?", "ha" * 25000, "a" * 25000 + "-" * 25000 + "a"],
                         ids=["a-dashes-a", "dashes", "a-bangs", "accent-apos", "ha-run", "a-run-dashes-a"])
def test_the_edge_strip_is_linear(s):
    """Fails in the §4QG world: `^[\\W_]+|[\\W_]+$` took ~7 s on the first."""
    t0 = time.perf_counter()
    A._is_one_real_word(s)
    assert time.perf_counter() - t0 < 0.05
