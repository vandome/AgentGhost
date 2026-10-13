"""§4OJ pilot: quote-bound answers — probe-only arm, mechanical quote check."""
import re
from ghost_agent.core import quote_bound as QB
from ghost_agent.utils import logging as L
from tests.pin_ast import tree, calls, call_kw, U


EV = QB.evidence_text(['{"text": "PostgreSQL 18.0 was released on **September 25, 2025** with asynchronous I/O."}',
                       "Weather: Temperature 21°C, feels like 19°C"])


def test_found_quotes_are_stripped_and_missing_ones_mark_the_specific():
    reply = ('PostgreSQL 18 came out on September 25, 2025 [src: "released on September 25, 2025"]. '
             'It is 3x faster [src: "up to 3x faster reads"]. It is 21°C [src: “Temperature 21°C, feels like 19°C”].')
    out, st = QB.apply_quotes(reply, EV)
    assert "[src" not in out
    assert out.startswith("PostgreSQL 18 came out on September 25, 2025. It is 3x faster (unverified). It is 21°C.")
    assert st["quotes"] == 3 and st["found"] == 2 and st["missing"] == ["up to 3x faster reads"]


def test_quote_matching_rules():
    assert QB.quote_found("PostgreSQL 18.0 was released … with asynchronous I/O", EV)      # ellipsis pieces
    assert not QB.quote_found("the", EV)                                                  # too short to be a source
    assert not QB.quote_found("released on September 26, 2025", EV)
    assert QB.quote_found("RELEASED ON   september 25, 2025.", EV)                          # case/space/markdown/punct
    out, _ = QB.apply_quotes('X [src: "nope one"] [src: "nope two"].', EV)
    assert out == "X (unverified)."


def test_tool_contents_reads_this_turn_and_earlier_tool_messages_but_not_synthetic_rows():
    class T:
        content = "obj body"
    got = QB.tool_contents([{"content": "a"}, {"content": "b", "_synthetic": True}, T()],
                           [{"role": "tool", "content": "earlier"}, {"role": "user", "content": "u"}])
    assert got == ["a", "obj body", "earlier"]                 # r2 M4: the user's words are not evidence


def test_the_arm_is_probe_only_and_wired_on_both_sides():
    assert QB.ARM in L.PROMPT_ARMS and L.parse_prompt_arm("quote_bound") == "quote_bound"
    tok = L.prompt_arm_context.set("quote_bound"); rt = L.request_id_context.set("req-real-1234")
    try:
        assert L.prompt_arm() == ""                     # a production request id never carries an arm
    finally:
        L.prompt_arm_context.reset(tok); L.request_id_context.reset(rt)
    import ast
    from ghost_agent.core import agent as ag
    t = tree(ag)
    th = [c for c in calls(t, "to_thread") if c.args and "check_reply_append" in U(c.args[0])]
    assert th and "mode()" in U(th[0].args[0])
    # §4OJ-c: one per-request mode, set from the stream flag; the rule and the rewrite read it
    sets = [c for c in calls(t, "set") if U(c.func).endswith("quote_bound_context.set")]
    assert sets and all(k in U(sets[0].args[0]) for k in ("append", "inline", "stream_response", "_qb_on"))
    tests = [U(n.test) for n in ast.walk(t) if isinstance(n, ast.If)]
    assert sum(1 for x in tests if x.endswith(".active()")) >= 2


def test_every_ellipsis_piece_must_stand_and_no_gap_is_left_before_punctuation():
    assert not QB.quote_found("PostgreSQL 18.0 was released … with synchronous magic", EV)
    out, _ = QB.apply_quotes('It is 21°C  [src: "Temperature 21°C, feels like"], today.', EV)
    assert out == "It is 21°C, today."


# ── "working" on a load alone ────────────────────────────────────────────── #

from ghost_agent.core import task_state as TS


def _fs(op, path, content="", result=None, **kw):
    if result is None:                       # the real success shape the path ledger reads
        result = f"SUCCESS: Wrote 9 chars to '{path}'. Script-side path (from sandbox cwd): '{path}'."
    return {"name": "file_system", "args": dict(operation=op, path=path, content=content, **kw), "content": result}


def _br(op, result="--- BROWSER RESULT ---\nSTATUS: OK\nOP: click", actions=None, url="file:///workspace/c/index.html"):
    a = {"operation": op, "url": url}
    if op == "interact":
        a["actions"] = actions if actions is not None else [{"action": "click", "selector": "#eq"}]
    return {"name": "browser", "args": a, "content": result}


PAGE = '<button onclick="go()">=</button><script>function go(){}</script>'


def test_a_scripted_page_only_loaded_is_named_and_a_click_after_the_last_change_clears_it():
    runs = [_fs("write", "c/index.html", PAGE), _br("navigate")]
    assert TS.untested_pages(runs) == ["c/index.html"]
    assert "NOT exercised" in TS.render_task_state(runs) and "`c/index.html`" in TS.render_task_state(runs)
    assert TS.untested_pages(runs + [_br("interact")]) == []
    assert TS.untested_pages(runs + [_br("click")]) == []
    # a change AFTER the click needs a new one
    assert TS.untested_pages(runs + [_br("click"), _fs("edit", "c/index.html", new_string="x")]) == ["c/index.html"]
    # a failed click tested nothing
    assert TS.untested_pages(runs + [_br("click", "--- BROWSER RESULT ---\nSTATUS: ERROR\nno element matches #eq")]) == ["c/index.html"]


def test_static_pages_refused_writes_and_other_files_are_not_named():
    assert TS.untested_pages([_fs("write", "a.html", "<h1>hi</h1>")]) == []
    assert TS.untested_pages([_fs("write", "a.js", "function go(){}")]) == []
    assert TS.untested_pages([_fs("write", "a.html", PAGE, result="REFUSED: read it first")]) == []
    assert TS.untested_pages([_fs("write", "a.html", "<p>x</p>"),
                              _fs("edit", "a.html", new_string='<b onclick="f()">')]) == ["a.html"]
    assert TS.untested_pages([_fs("write", "a.html", "<p>x</p>"),
                              _fs("replace", "a.html", replace_with="<script>f()</script>")]) == ["a.html"]


def test_only_html_pages_are_named():
    assert TS.untested_pages([_fs("write", "build.js", 'el.innerHTML = "<script>x()</script><b onclick=\\"f()\\">"')]) == []


# ── reader round 1 ───────────────────────────────────────────────────────── #

def test_a_right_quote_does_not_clear_a_wrong_figure_beside_it():
    out, st = QB.apply_quotes('It was released on September 26, 2025 [src: "released on September 25, 2025"].', EV)
    assert out == "It was released on September 26, 2025 (unverified)." and st["figure_mismatch"] == 1
    ok, _ = QB.apply_quotes('PostgreSQL 18 came out on September 25, 2025 [src: "released on September 25, 2025"].', EV)
    assert ok == "PostgreSQL 18 came out on September 25, 2025."


def test_code_is_untouched_and_markers_with_brackets_or_unclosed_do_not_eat_text():
    code = "Run:\n```bash\ncp a.txt .\ndocker build -t app .\nout = buf[src: src + n]\n```\nDone ."
    assert QB.apply_quotes(code, EV)[0] == code                      # no marker: nothing changes, not even "Done ."
    out, _ = QB.apply_quotes('Use `x[src: y]` and it shipped [src: "with asynchronous i/o"] .', EV)
    assert out.startswith("Use `x[src: y]` and it shipped") and "[src" not in out.replace("`x[src: y]`", "")
    br = QB.evidence_sources(["version 18.0 [beta] shipped today"])
    assert QB.apply_quotes('It shipped [src: "version 18.0 [beta] shipped"].', br)[0] == "It shipped."
    out, st = QB.apply_quotes('Released Sept 25 [src: "released on September 25\n\nNext, see [the docs](https://x.org) for more.', EV)
    assert "Next, see [the docs](https://x.org) for more." in out and "(unverified)" in out


def test_quotes_must_be_three_words_in_order_in_one_source_and_normalised():
    srcs = QB.evidence_sources(["PostgreSQL 19 roadmap", "it was released on Monday"])
    assert not QB.quote_found("PostgreSQL 19 roadmap … released on Monday", srcs)          # spans two sources
    assert not QB.quote_found("it is", srcs) and not QB.quote_found("September", srcs)
    assert not QB.quote_found("released on … it was", srcs)                                   # out of order
    js = QB.evidence_sources(['{"t": "caf\\u00e9 opened \\"today\\" in 2020\\u20132024 with 1,234,567 visitors &amp; more"}'])
    assert QB.quote_found('café opened "today" in 2020-2024', js)
    assert QB.quote_found("with 1234567 visitors & more", js)


def test_the_arm_is_refused_on_a_streamed_request():
    import ast
    from ghost_agent.api import routes as R
    src = [n for n in ast.walk(tree(R)) if isinstance(n, ast.If) and "quote_bound" in U(n.test)]
    assert src and "stream" in U(src[0].test)


def test_only_real_page_actions_on_that_page_count_as_a_test():
    runs = [_fs("write", "c/index.html", PAGE), _br("navigate")]
    load_only = _br("interact", actions=[{"action": "goto", "url": "file:///workspace/c/index.html"},
                                         {"action": "screenshot"}])
    assert TS.untested_pages(runs + [load_only]) == ["c/index.html"]
    assert TS.untested_pages(runs + [_br("click", url="https://news.ycombinator.com")]) == ["c/index.html"]
    assert TS.untested_pages(runs + [_br("click", url="http://127.0.0.1:8101/")]) == []          # served locally
    partial = "--- BROWSER RESULT ---\nSTATUS: PARTIAL (1 of 5 actions failed)\nOP: interact"
    assert TS.untested_pages(runs + [_br("interact", result=partial)]) == []
    two = [_fs("write", "b.html", PAGE), _fs("write", "a.html", PAGE),
           _br("click", url="file:///workspace/a.html")]
    assert TS.untested_pages(two) == ["b.html"]
    # spelling: /workspace/x.html then x.html is one page
    assert TS.untested_pages([_fs("write", "/workspace/x.html", PAGE), _br("click", url="file:///workspace/x.html"),
                              _fs("edit", "x.html", new_string="y")]) == ["x.html"]
    # rewritten static, deleted, renamed
    assert TS.untested_pages([_fs("write", "s.html", PAGE), _fs("write", "s.html", "<h1>static</h1>")]) == []
    assert TS.untested_pages([_fs("write", "d.html", PAGE), _fs("delete", "d.html")]) == []
    assert TS.untested_pages([_fs("write", "o.html", PAGE), _fs("rename", "o.html", destination="n.html")]) == ["n.html"]


def test_the_download_links_keep_priority_over_the_untested_line():
    runs = [_fs("write", f"projects/my-dashboard-app/pages/page_number_{i:02d}_with_a_long_name.html", PAGE)
            for i in range(9)]
    runs.append({"name": "execute", "args": {"command": "x"}, "content": "EXIT CODE: 1\nTraceback: boom"})
    block = TS.render_task_state(runs)
    static = TS.render_task_state([dict(r, args=dict(r["args"], content="<p>x</p>")) if r["name"] == "file_system" else r
                                   for r in runs])
    assert block.count("/api/download/") == static.count("/api/download/")
    assert len(block) <= TS.TASK_STATE_MAX_CHARS + 4


def test_a_present_two_word_quote_is_still_too_short():
    srcs = QB.evidence_sources(["it was released on Monday"])
    assert QB.quote_found("released on Monday", srcs) and not QB.quote_found("released on", srcs)


def test_a_synthetic_row_copied_into_messages_is_not_a_source():
    syn = "SYSTEM ERROR: Tool 'x' is explicitly disabled in this context."
    got = QB.tool_contents([{"content": syn, "_synthetic": True}], [{"role": "tool", "content": syn}])
    assert got == []


def test_links_survive_when_the_cap_is_tight():
    long = "projects/a-really-long-project-folder-name/pages/subsection/" + "x" * 60
    runs = [_fs("write", f"{long}_{i:02d}.html", PAGE) for i in range(6)]
    runs.append({"name": "execute", "args": {"command": "x"}, "content": "EXIT CODE: 1\nTraceback: boom " + "e" * 200})
    block = TS.render_task_state(runs)
    static = TS.render_task_state([dict(r, args=dict(r["args"], content="<p>x</p>")) if r["name"] == "file_system" else r
                                   for r in runs])
    assert static.count("/api/download/") >= 1
    assert block.count("/api/download/") == static.count("/api/download/")


# ── pilot findings ───────────────────────────────────────────────────────── #

def test_several_quotes_in_one_marker_and_short_key_value_quotes():
    w = QB.evidence_sources(["Condition: Light Showers\nTemp: 19.0°C\nHumidity: 83%\nWind: 5.0 km/h"])
    assert QB.quote_found('"Temp: 19.0°C", "Humidity: 83%"', w)
    assert not QB.quote_found('"Temp: 19.0°C", "Humidity: 85%"', w)
    assert QB.quote_found("Temp: 19.0°C", w) and not QB.quote_found("Light Showers", w)


def test_parenthesised_or_own_line_markers_leave_no_debris():
    out, _ = QB.apply_quotes('Beta 4 is out ([src: "nope not there"]). It rains ([src: "Condition: Light Showers"]).',
                             QB.evidence_sources(["Condition: Light Showers"]))
    assert out == "Beta 4 is out (unverified). It rains."
    out, _ = QB.apply_quotes('Temp 19\n[src: "nothing like this"]\nNext', QB.evidence_sources(["x"]))
    assert out == "Temp 19 (unverified)\nNext"



# ── §4OJ-b: on for non-streamed replies; near-verbatim quotes ─────────────── #

def test_on_by_default_and_switchable_off(monkeypatch):
    monkeypatch.delenv("GHOST_QUOTE_BOUND", raising=False)
    assert QB.production_enabled()
    for v in ("0", "off", "false"):
        monkeypatch.setenv("GHOST_QUOTE_BOUND", v)
        assert not QB.production_enabled()
    tok = QB.quote_bound_context.set(True)
    try:
        assert QB.active()
    finally:
        QB.quote_bound_context.reset(tok)
    assert not QB.active()


def test_near_verbatim_quotes_pass_but_a_changed_figure_or_name_does_not():
    src = QB.evidence_sources(["Δύο εμπορικά πλοία χτυπήθηκαν από drone ανοιχτά των ακτών της Βουλγαρίας στη Μαύρη Θάλασσα.",
                               "| 11 | 11.22 | No | October 18, 2018 | November 9, 2023 |\n| 12 | 12.22 | No | Sept 2019 |",
                               "Kyriakos Pierrakakis said the budget for 2026 rises to 4.2 billion euros next year."])
    assert QB.quote_found("δύο εμπορικά πλοία χτυπήθηκαναπό drone ανοιχτά των ακτών της Βουλγαρίας", src)   # merged words
    assert QB.quote_found("Pierrakakis said the budget for 2026 rises to 4.2 billion euro", src)            # small slip
    assert not QB.quote_found("Pierrakakis said the budget for 2026 rises to 4.3 billion euros", src)       # figure
    assert not QB.quote_found("Papadopoulos said the budget for 2026 rises to 4.2 billion euros", src)      # name
    assert not QB.quote_found("12 11.22 No October 18, 2018", src)                                         # row number
    assert not QB.quote_found("budget for 2026 rises", QB.evidence_sources(["budget for the year rises 2026"]))


def test_leftover_markers_never_reach_a_client():
    assert QB.strip_leftover_markers('Hi [src: "x y z"] there ([src: "a b c"]).\n`k[src: v]`') == "Hi there.\n`k[src: v]`"
    assert QB.strip_leftover_markers("plain") == "plain"
    import ast
    from ghost_agent.api import routes as R
    assert any(U(c.func).endswith("strip_leftover_markers") for c in calls(tree(R), "strip_leftover_markers"))


def test_words_spread_across_a_page_are_not_a_quote():
    filler = " ".join(["lorem"] * 12)
    src = QB.evidence_sources([f"the budget {filler} for {filler} 2026 rises {filler} to the {filler} highest level"])
    assert not QB.quote_found("the budget for 2026 rises to the highest level", src)
    assert QB.quote_found("the budget for 2026 rises to the highest level",
                          QB.evidence_sources(["the budget for 2026 rises to the highest level yet"]))



# ── reader round 2 (§4OJ-b) ──────────────────────────────────────────────── #

def test_meaning_changes_and_other_rows_are_never_near():
    cases = [
        ("2023 revenue was 5 million euros; 2024 revenue was 7 million euros.", "2023 revenue was 7 million euros"),
        ("Q1 revenue rose 10 percent to 4.2 billion. Q2 revenue rose 12 percent to 5 billion.",
         "Q1 revenue rose 12 percent to 4.2 billion"),
        ("The patient was not diagnosed with diabetes in 2019.", "The patient was diagnosed with diabetes in 2019"),
        ("The bridge is 1,200 meters long and opened in 1990.", "The bridge is 1,200 feet long and opened"),
        ("The Pro plan costs $49.99 per month for teams.", "The Pro plan costs $49.99 per year"),
        ("Jane Doe was appointed CEO of Acme Corp in March.", "Jane Doe was appointed CFO of Acme Corp"),
    ]
    for src, q in cases:
        assert not QB.quote_found(q, QB.evidence_sources([src])), q


def test_the_users_question_does_not_certify_its_premise():
    msgs = [{"role": "user", "content": "Is the Pro plan $59.99 per month? check the site"},
            {"role": "tool", "content": "Pro plan: $49.99 per month"}]
    srcs = QB.evidence_sources(QB.tool_contents([], msgs))
    out, _ = QB.apply_quotes('It costs $59.99 per month [src: "Pro plan $59.99 per month"].', srcs)
    assert out == "It costs $59.99 per month (unverified)."


def test_only_the_api_non_streamed_branch_is_eligible():
    from ghost_agent.api import routes as R
    import ast
    assigns = [n for n in ast.walk(tree(R)) if isinstance(n, ast.Assign)
               and any(U(t) == "body[ELIGIBLE_KEY]" for t in n.targets)]
    assert assigns                                                # the API's non-streamed branch marks its body
    assert "NO marker" in QB.RULE and "user told you" in QB.RULE


def test_long_whitespace_and_unclosed_fences_are_cheap_and_safe():
    import time
    t0 = time.monotonic()
    out, _ = QB.apply_quotes(" " * 40_000 + 'x [src: "a b c"]', [""])
    QB.strip_leftover_markers(" " * 40_000 + "[src: y]")
    assert time.monotonic() - t0 < 1.0 and out.endswith("x (unverified)")
    code = "Run:\n```python\nx = 1 [src: \"a b c\"]\n"
    assert QB.apply_quotes(code, [""])[0] == code


def test_failed_check_and_cancelled_partial_never_carry_raw_markers():
    import ast
    from ghost_agent.core import agent as ag
    t = tree(ag)
    assert len([c for c in ast.walk(t) if isinstance(c, ast.Call) and U(c.func).endswith(("strip_leftover_markers", "_slm"))]) >= 2


def test_near_match_edges_numbers_short_words_plurals_and_big_sources():
    src = QB.evidence_sources(["The city has 1234567 residents and two new bridges over the river since 2019.",
                               "He noted the price was 5 dollars in May."])
    assert not QB.quote_found("The city has 1234568 residents and two", src)          # one digit off
    assert not QB.quote_found("He note the price was 5 dollars", src)                 # 4 vs 5 letters: exact only
    assert QB.quote_found("two new bridge over the river since 2019", src)            # plural mid-quote
    big = QB.evidence_sources(["x " * 210_000 + "the river bridges were rebuilt in 2019 by the city"])
    assert not QB.quote_found("the river bridge were rebuilt in 2019", big)           # no index over huge outputs
    assert QB.quote_found("the river bridges were rebuilt in 2019", big)              # exact still works


def test_a_short_word_plural_is_the_same_word():
    src = QB.evidence_sources(["Four cats sat on the mat in 2019 near the door."])
    assert QB.quote_found("Four cat sat on the mat in 2019", src)


# ── reader round 3 (§4OJ-b) ──────────────────────────────────────────────── #

def test_near_typos_never_flip_meaning_or_magnitude():
    for src, q in [("Revenue reached 4.2 billion dollars in 2024 overall.", "Revenue reached 4.2 million dollars in 2024"),
                   ("A rate cut is unlikely before the end of the year.", "A rate cut is likely before the end"),
                   ("The drug was unapproved for adults in the trial.", "The drug was approved for adults in"),
                   ("Exports to Austria rose sharply in March 2024.", "Exports to Australia rose sharply in March"),
                   ("The service was insecure by default since 2020.", "The service was secure by default since")]:
        assert not QB.quote_found(q, QB.evidence_sources([src])), q
    assert QB.quote_found("Exports to Austrai rose sharply in March",
                          QB.evidence_sources(["Exports to Austria rose sharply in March 2024."])) is False   # 2-letter swap
    assert QB.quote_found("the goverment announced the new budget today",
                          QB.evidence_sources(["the government announced the new budget today"]))          # real typo


def test_exact_matches_respect_word_edges():
    src = QB.evidence_sources(["Model X costs 799 dollars. Version 3.12 shipped; 999 dollars for Y."])
    for q in ("Model X costs 79", "99 dollars", "Version 3.1"):
        assert not QB.quote_found(q, src), q
    assert QB.quote_found("Model X costs 799", src)


def test_a_marker_after_the_full_stop_backs_the_sentence_before():
    src = QB.evidence_sources(["Revenue was 4.2 billion dollars last year."])
    out, st = QB.apply_quotes('Revenue was 5.1 billion. [src: "Revenue was 4.2 billion dollars"]', src)
    assert out == "Revenue was 5.1 billion. (unverified)" or out.endswith("(unverified)")
    assert st["figure_mismatch"] == 1


def test_a_backtick_inside_a_quote_does_not_break_the_marker():
    src = QB.evidence_sources(["To build, run `make install` and then restart the service."])
    out, st = QB.apply_quotes('Run make install [src: "To build, run `make install` and then"] then restart.', src)
    assert out == "Run make install then restart." and st["found"] == 1
    assert QB.strip_leftover_markers('Run it [src: "run `make install` and then"] then.') == "Run it then."
    code = "```\nx = [src: y]\n```"
    assert QB.apply_quotes("A [src: \"a b c\"].\n" + code, src)[0].endswith(code)


def test_near_work_per_reply_is_capped_and_runs_off_the_loop():
    import time
    big = QB.evidence_sources([("lorem ipsum dolor sit amet " * 4000) + str(i) for i in range(40)])
    reply = " ".join(f'Claim {i} [src: "lorem ipsum dolor sit amet consectetur {i}"].' for i in range(20))
    t0 = time.monotonic()
    QB.apply_quotes(reply, big)
    assert time.monotonic() - t0 < 3.0
    import ast
    from ghost_agent.core import agent as ag
    th = [c for c in calls(tree(ag), "to_thread") if c.args and "check_reply" in U(c.args[0])]
    assert th


def test_spawned_work_does_not_inherit_eligibility():
    # the mark is a key on THIS request's body; sub-agents, leaves, replays and the scheduler build their own
    assert QB.ELIGIBLE_KEY.startswith("_ghost") and not hasattr(QB, "quote_bound_eligible")


def test_a_verified_marker_on_its_own_line_leaves_no_blank_line():
    src = QB.evidence_sources(["Model X costs 799 dollars today."])
    out, _ = QB.apply_quotes('- a 799\n[src: "Model X costs 799 dollars"]\n- b', src)
    assert out == "- a 799\n- b"


def test_the_near_budget_and_the_prefilter_each_bound_the_work(monkeypatch):
    seen = []
    real = QB._near_found
    monkeypatch.setattr(QB, "_near_found", lambda q, s: (seen.append(q), real(q, s))[1])
    src = QB.evidence_sources(["alpha beta gamma delta epsilon zeta 2024 eta theta"])
    reply = " ".join(f'C{i} [src: "alpha beta gamma delta epsilom zeta 2024"].' for i in range(20))
    QB.apply_quotes(reply, src)
    assert 0 < len(seen) <= QB.NEAR_MAX_PER_REPLY                     # budget
    seen.clear()
    QB.apply_quotes('C [src: "alpha beta gamma delta epsilon zeta 2025 eta"].', src)
    assert seen == []                                                 # a figure the source lacks: never tokenised



# ── reader round 4 ───────────────────────────────────────────────────────── #

def test_a_wrong_figure_in_inline_code_beside_a_marker_is_caught():
    src = QB.evidence_sources(["It costs $59.99 per month. Python version 3.12.1 is the latest stable."])
    assert QB.apply_quotes('The price is `$69.99` [src: "It costs $59.99 per month"].', src)[0] == \
        "The price is `$69.99` (unverified)."
    assert QB.apply_quotes('Pro: $69.99 / month - see [src: "It costs $59.99 per month"]', src)[0].endswith("(unverified)")
    assert QB.apply_quotes('Pro costs $59.99 [src: "It costs $59.99 per month"].', src)[0] == "Pro costs $59.99."


def test_a_malformed_marker_never_swallows_the_text_after_it():
    src = QB.evidence_sources(["It costs $59.99 per month; it was released on September 25, 2025."])
    out, _ = QB.apply_quotes('Price 59.99 [src: "It costs $59.99 per month" and it shipped in 1999 '
                             '[src: "was released on September 25, 2025"].', src)
    assert "and it shipped in 1999" in out
    assert "B 1999" in QB.strip_leftover_markers('A [src: "x" and B 1999 [src: "y"]. C')


def test_a_digit_inside_a_name_is_not_a_figure():
    src = QB.evidence_sources(["Greek exports rose 4 percent in the first half of 2024."])
    out, _ = QB.apply_quotes('Greek exports rose in H1 2024 [src: "exports rose 4 percent in the first half of 2024"].', src)
    assert out == "Greek exports rose in H1 2024."


def test_iso_and_month_name_dates_carry_the_same_figures():
    src = QB.evidence_sources(["September 24, 2026: PostgreSQL 19 Beta 4 Released!"])
    out, _ = QB.apply_quotes('19 Beta 4, released on **2026-09-24** [src: "September 24, 2026: PostgreSQL 19 Beta 4 Released!"]', src)
    assert out == "19 Beta 4, released on **2026-09-24**"
    out, _ = QB.apply_quotes('released on **2026-09-25** [src: "September 24, 2026: PostgreSQL 19 Beta 4 Released!"]', src)
    assert out.endswith("(unverified)")
    assert QB._figures("Sept 4") >= {"9", "4"} and QB._figures("2026-09-04") == {"2026", "9", "4"}


def test_a_month_word_counts_only_beside_a_day():
    assert QB._figures("you may see 4 errors") == {"4"}
    assert QB._figures("on 24 September 2026") == {"24", "9", "2026"}
    assert QB._figures("May 3") == {"5", "3"}



# ── §4OJ-c: streamed replies ─────────────────────────────────────────────── #

STREAM_SRC = QB.evidence_sources(["September 24, 2026: PostgreSQL 19 Beta 4 Released!"])
STREAM_REPLY = ('PostgreSQL 18.6 was released on 2026-08-13 [src: "18.6 released 2026-08-13 today"]. '
                'Beta 4 came out ([src: "September 24, 2026: PostgreSQL 19 Beta 4 Released!"]) and `code [src: x]` stays.'
                '\n- item [src: "Released!" ] done\nEnd.')


def test_the_streamed_view_only_ever_grows_and_hides_every_marker():
    prev = ""
    for i in range(1, len(STREAM_REPLY) + 1):
        v = QB.strip_view(STREAM_REPLY[:i])
        assert v.startswith(prev), (i, prev[-30:], v[-30:])
        assert '[src: "' not in v                                # only the bare marker inside code may show
        prev = v
    final = QB.strip_view(STREAM_REPLY, final=True)
    assert final.startswith(prev)
    assert final == ("PostgreSQL 18.6 was released on 2026-08-13. Beta 4 came out and `code` stays."   # r7: only fences protect
                     "\n- item done\nEnd.")


def test_the_closing_line_names_only_failed_claims():
    claims = QB.failed_claims(STREAM_REPLY, STREAM_SRC)
    assert claims[0] == "PostgreSQL 18.6 was released on 2026-08-13" and not any("Beta 4" in c for c in claims)
    assert QB.closing_line(claims).startswith("\n\n_Could not verify against the sources read: “PostgreSQL 18.6")
    assert QB.closing_line([]) == ""
    out, st = QB.check_reply_append(STREAM_REPLY, [{"content": "September 24, 2026: PostgreSQL 19 Beta 4 Released!"}], [])
    assert out.startswith(QB.strip_view(STREAM_REPLY, final=True)) and "Could not verify" in out


def test_the_frame_filter_streams_the_view_and_closes_with_the_line():
    import asyncio, json as _j

    def frame(text):
        return ("data: " + _j.dumps({"id": "chatcmpl-x", "object": "chat.completion.chunk", "created": 1, "model": "m",
                                     "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]})
                + "\n\n").encode()

    async def gen():
        for i in range(0, len(STREAM_REPLY), 7):
            yield frame(STREAM_REPLY[i:i + 7])
        yield b"data: [DONE]\n\n"

    async def run():
        out = []
        async for c in QB.filter_sse(gen(), [{"content": "September 24, 2026: PostgreSQL 19 Beta 4 Released!"}], []):
            out.append(c)
        return out

    chunks = asyncio.run(run())
    text = ""
    for c in chunks:
        for ev in c.decode().split("\n\n"):
            if ev.startswith("data: ") and ev[6:].strip() != "[DONE]":
                text += (_j.loads(ev[6:])["choices"][0]["delta"].get("content") or "")
    assert text == QB.strip_view(STREAM_REPLY, final=True) + QB.closing_line(QB.failed_claims(STREAM_REPLY, STREAM_SRC))
    assert chunks[-1].decode().strip().endswith("[DONE]")


def test_the_reply_tap_hides_markers_live_and_commits():
    from ghost_agent.core.reply_tap import ReplyTap
    tap = ReplyTap("r1", "m", 1, hold_chars=10)
    tap.quote_view = True
    tap.begin_generation(unsafe=lambda t: False)
    for i in range(1, len(STREAM_REPLY) + 1):
        tap.feed(STREAM_REPLY[:i])
    assert "[src: \"" not in tap.sent
    final, _ = QB.check_reply_append(STREAM_REPLY, [{"content": "September 24, 2026: PostgreSQL 19 Beta 4 Released!"}], [])
    frames = tap.finish(final)
    assert tap.outcome == "committed"


def test_streamed_wiring_frames_records_and_both_route_branches():
    import ast
    from ghost_agent.core import agent as ag
    from ghost_agent.api import routes as R
    t = tree(ag)
    assert [c for c in calls(t, "filter_sse")]
    assert [c for c in calls(t, "strip_view") if call_kw(c, "final") is not None]
    rt = tree(R)
    assigns = [n for n in ast.walk(rt) if isinstance(n, ast.Assign) and any(U(x) == "body[ELIGIBLE_KEY]" for x in n.targets)]
    assert len(assigns) == 1                                      # once, before the stream/non-stream split



# ── reader round 5: the append-only machine ──────────────────────────────── #

R5_CASES = [
    'Rome had 2.8 million people [src: "population of 2,761,632[2]"], making it big.',
    'Use `cmd [src: "a b"] x` end of line and more.',
    'Run `ls` or ` x [src: "q r s"] then `y` end of text.',
    'Fact [src: "abandoned quote\nNext line.',
    'Fact [src: "abc\ndef"] and on.',
    'A [src:] B [src: ' + "x" * 500 + '] C',
    '```\nf([src: "x"])\n```\nafter',
    'x(( y ([src: "q r s"]) z [ ] [x] [link](http://a.b) [1, 2]',
    'It costs $5 [src: "a b c"] and ships in 3 days [src: "d e f"].',
    'Price "x" [src: "It costs" and more text\nnext',
]


def test_the_view_only_grows_and_equals_the_final_for_every_round_5_case():
    import random
    random.seed(5)
    for text in R5_CASES:
        final = QB.strip_view(text, final=True)
        f, shown = QB.MarkerFilter(), ""
        i = 0
        while i < len(text):
            n = random.randint(1, 5)
            shown += f.feed(text[i:i + n])
            i += n
        shown += f.finish()
        assert shown == final, (text, shown, final)
        prev = ""
        for k in range(1, len(text) + 1):
            v = QB.strip_view(text[:k])
            assert v.startswith(prev) and final.startswith(v), (text, k)
            prev = v
        outside_code = re.sub(r"```.*?```|`[^`\n]*`", "", final, flags=re.S)
        assert "[src:" not in outside_code.lower(), (text, final)


def test_no_marker_text_is_byte_identical():
    for t in ["| a | b |\n|---|---|\n| [x] | (y) |", "see [docs](http://x) and `f([1])`  ", "ends with (", "trail  \n",
              "```py\nprint([1, 2])\n```", "CRLF\r\nline", "np.array(([1, 2], [3]))"]:
        assert QB.strip_view(t, final=True) == t, t


def test_closing_clauses_stop_at_earlier_markers_and_keep_figures():
    srcs = QB.evidence_sources(["nothing relevant here at all"])
    claims = QB.failed_claims('It costs $5 [src: "a b c"] and ships in 3 days [src: "d e f"]. 5*3 = 15 [src: "g h i"]', srcs)
    assert claims == ["It costs $5", "and ships in 3 days", "5*3 = 15"]
    assert QB.failed_claims("| Widget | $5 [src: \"x y z\"] |", srcs) == ["$5"]


def test_the_machine_is_linear():
    import time
    text = ("word " * 9 + 'fact 12 [src: "a quoted source line 12"] ') * 400
    t0 = time.monotonic()
    f = QB.MarkerFilter()
    for i in range(0, len(text), 3):
        f.feed(text[i:i + 3])
    f.finish()
    assert time.monotonic() - t0 < 1.0



R5_FINAL = {
    0: "Rome had 2.8 million people, making it big.",
    1: "Use `cmd x` end of line and more.",                                 # r7: only fences protect
    2: "Run `ls` or ` x then `y` end of text.",
    3: "Fact\nNext line.",                                                  # abandoned at the newline
    6: '```\nf([src: "x"])\n```\nafter',                                   # fenced code untouched
    7: "x(( y z [ ] [x] [link](http://a.b) [1, 2]",
    8: "It costs $5 and ships in 3 days.",
    9: 'Price "x" and more text\nnext',                                       # closed quote, no "]": ends there
}


def test_round_5_cases_have_exact_finals():
    for i, want in R5_FINAL.items():
        assert QB.strip_view(R5_CASES[i], final=True) == want, (i, QB.strip_view(R5_CASES[i], final=True))
    long = "A " + "[src: " + "x" * 500 + "] C"
    assert QB.strip_view(long, final=True).startswith("A")                  # an over-long marker is cut, not kept
    assert "[src:" not in QB.strip_view(long, final=True)
    assert QB.strip_view("held [src", final=True) == "held [src"            # never became a marker: shown at the end
    assert QB.strip_view("tail  ", final=True) == "tail  "


def test_the_frame_filter_closes_before_a_finish_frame():
    import asyncio, json as _j

    async def gen():
        for t in ("Fact 1 ", '[src: "zz yy xx"]', " end."):
            yield ("data: " + _j.dumps({"id": "c", "choices": [{"index": 0, "delta": {"content": t}, "finish_reason": None}]})
                   + "\n\n").encode()
        yield ("data: " + _j.dumps({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}) + "\n\n").encode()
        yield b"data: [DONE]\n\n"

    async def run():
        return [c async for c in QB.filter_sse(gen(), [], [])]

    evs = [e for c in asyncio.run(run()) for e in c.decode().split("\n\n") if e.startswith("data: ")]
    finish_at = next(i for i, e in enumerate(evs) if '"finish_reason": "stop"' in e)
    closing_at = next(i for i, e in enumerate(evs) if "Could not verify" in e)
    assert closing_at < finish_at


def test_machine_edges_long_markers_replayed_brackets_and_finishing_content_frames():
    long = "A [src: " + "x" * 500 + "] C"
    out = QB.strip_view(long, final=True)
    assert out.startswith("A") and out.endswith("] C") and len(out) < len(long) - 400     # cut at MARKER_MAX
    assert QB.strip_view('a [s[src: "q r s"] b', final=True) == "a [s b"                  # held "[s" re-fed
    import asyncio, json as _j

    async def gen():
        yield ("data: " + _j.dumps({"id": "c", "choices": [{"index": 0, "delta": {"content": 'Fact 1 [src: "zz yy xx"].'},
                                                            "finish_reason": "stop"}]}) + "\n\n").encode()
        yield b"data: [DONE]\n\n"

    async def run():
        return [c async for c in QB.filter_sse(gen(), [], [])]

    evs = [e for c in asyncio.run(run()) for e in c.decode().split("\n\n") if e.startswith("data: ")]
    closing_at = next(i for i, e in enumerate(evs) if "Could not verify" in e)
    finish_at = next(i for i, e in enumerate(evs) if '"finish_reason": "stop"' in e)
    assert closing_at < finish_at


# ── reader round 6 ───────────────────────────────────────────────────────── #

def test_quotes_holding_quote_marks_and_brackets_stay_whole():
    assert QB.strip_view('Revenue grew [src: "the CEO said “we grew 5%” last year"] in 2025.', final=True) == "Revenue grew in 2025."
    assert QB.strip_view('He said hi [src: "He said "hello" to me"] ok.', final=True) == "He said hi ok."
    assert QB.strip_view("Costs [src: 'it's 5 dollars here'] today.", final=True) == "Costs today."
    assert QB.strip_view('It shipped [src: "version 18.0 [beta] shipped"].', final=True) == "It shipped."
    f = QB.run_filter('Revenue grew [src: "the CEO said “we grew 5%” last year"] in 2025.')
    assert f.markers[0][0] == "the CEO said “we grew 5%” last year" and f.markers[0][2] is False


def test_abandoned_markers_are_listed_and_a_kept_paren_keeps_its_space():
    f = QB.run_filter('A claim [src: "unclosed quote here\nnext line')
    assert f.text() == "A claim\nnext line" and f.markers[0][2] is True
    assert f.failed_claims(QB.evidence_sources(["unclosed quote here"])) == ["A claim"]   # abandoned: listed
    assert QB.strip_view('costs 5 dollars ([src: "a b c"]; see table) ok', final=True) == "costs 5 dollars (; see table) ok"
    assert QB.strip_view('[src: "x y z"] Fact at start.', final=True) == "Fact at start."


def test_reasoning_before_an_orphan_think_close_does_not_feed_the_answer():
    import asyncio, json as _j

    def fr(t, fin=None):
        return ("data: " + _j.dumps({"id": "c", "choices": [{"index": 0, "delta": {"content": t} if t is not None else {},
                                                             "finish_reason": fin}]}) + "\n\n").encode()

    async def gen():
        yield fr('Plan: use ```\n</think>\n\nThe widget ')
        yield fr('costs 5 dollars [src: "nope not there"].')
        yield fr(None, "stop")
        yield fr(" More.")
        yield b"data: [DONE]\n\n"

    async def run():
        return [c async for c in QB.filter_sse(gen(), [], [])]

    evs = [e for c in asyncio.run(run()) for e in c.decode().split("\n\n") if e.startswith("data: ")]
    txt = "".join((_j.loads(e[6:])["choices"][0]["delta"].get("content") or "") for e in evs if e[6:].strip() != "[DONE]")
    assert "[src:" not in txt and txt.index(" More.") < txt.index("Could not verify")
    assert "“The widget costs 5 dollars”" in txt
    kinds = ["stop" in e for e in evs]
    assert kinds.index(True) == len(evs) - 2                       # the finish frame right before [DONE]



def test_page_refs_nested_markers_figures_and_a_failing_closing_line():
    f = QB.run_filter('Ref [src: "the Title here" p. 3] done.')
    assert f.text() == "Ref done." and f.markers[0][0] == "the Title here"
    assert QB.strip_view('a [src: "x y z" and b [src: "p q r"] c', final=True) == "a and b c"
    g = QB.run_filter('It costs $7 [src: "It costs $5 per month"].')
    assert g.failed_claims(QB.evidence_sources(["It costs $5 per month"])) == ["It costs $7"]
    import asyncio

    async def gen():
        yield b'data: {"id": "c", "choices": [{"index": 0, "delta": {"content": "X [src: \\"a b c\\"]."}, "finish_reason": null}]}\n\n'
        yield b"data: [DONE]\n\n"

    def boom(self, sources):
        raise IndexError("x")

    orig = QB.MarkerFilter.failed_claims
    QB.MarkerFilter.failed_claims = boom
    try:
        async def run():
            return [c async for c in QB.filter_sse(gen(), [], [])]
        chunks = asyncio.run(run())
    finally:
        QB.MarkerFilter.failed_claims = orig
    assert chunks[-1].decode().strip().endswith("[DONE]")



# ── reader round 7 ───────────────────────────────────────────────────────── #

def test_a_verified_claim_after_a_comma_is_not_listed():
    srcs = QB.evidence_sources(["It costs $10 and weighs 2 kg. A grew 5% and B grew 7%."])
    assert QB.failed_claims('It costs $10 [src: "It costs $10"], weighs 2 kg [src: "weighs 2 kg"].', srcs) == []
    assert QB.failed_claims('A grew 5% [src: "A grew 5%"], B grew 9% [src: "B grew 7%"].', srcs) == ["B grew 9%"]


def test_a_lone_backtick_never_shows_a_raw_marker():
    assert QB.strip_view('Don`t panic: A grew 5% [src: "A grew 5%"].', final=True) == "Don`t panic: A grew 5%."


def test_think_close_inside_a_fence_and_closing_line_outside_an_open_fence():
    import asyncio, json as _j

    def fr(t):
        return ("data: " + _j.dumps({"id": "c", "choices": [{"index": 0, "delta": {"content": t}, "finish_reason": None}]})
                + "\n\n").encode()

    async def gen():
        for t in ['X is 5 [src: "zz 5"]. Tag:\n```\n', "</think>",
                  '\n```\nThen Q is 3 [src: "Q is 3 ok"] and C fell 9% [src: "nope"].\n```py\nx = 1']:
            yield fr(t)
        yield b'data: {"id": "c", "choices": [], "usage": {"total_tokens": 9}}\n\n'
        yield b"data: [DONE]\n\n"

    async def run():
        return [c async for c in QB.filter_sse(gen(), [], [])]

    evs = [e for c in asyncio.run(run()) for e in c.decode().split("\n\n") if e.startswith("data: ")]
    txt = "".join((_j.loads(e[6:])["choices"][0]["delta"].get("content") or "")
                  for e in evs if e[6:].strip() != "[DONE]" and _j.loads(e[6:]).get("choices"))
    assert "[src:" not in txt and "x = 1\n```\n\n_Could not verify" in txt
    assert "“X is 5”" in txt and "Q is 3" in txt
    assert '"usage"' in evs[-2] and evs[-1].endswith("[DONE]")


# ── reader round 8 ───────────────────────────────────────────────────────── #

def test_a_reply_ending_on_a_closing_fence_gets_the_line_outside_code():
    out, _ = QB.check_reply_append('Revenue rose 9% [src: "made up quote here ok"].\n\n```\nprint(1)\n```', [], [])
    assert out.startswith("Revenue rose 9%.\n\n```\nprint(1)\n```\n\n_Could not verify") and out.count("```") == 2


def test_an_apostrophe_never_closes_a_double_quoted_marker():
    srcs = QB.evidence_sources(["revenue rose 5%"])
    f = QB.run_filter('Revenue rose 5% [src: "revenue rose 5%" — and the company\'s margin widened.\nNext line.')
    assert f.text() == "Revenue rose 5% — and the company's margin widened.\nNext line."
    assert f.failed_claims(srcs) == []
    assert QB.strip_view('X [src: "it\'s [1] here"] y', final=True) == "X y"


def test_mid_line_backticks_are_not_a_fence_and_a_leading_newline_goes():
    assert QB.strip_view('Wrap code in ``` fences. Revenue rose [src: "revenue rose"].', final=True) == \
        "Wrap code in ``` fences. Revenue rose."
    assert QB.strip_view("[src:\n]\nText", final=True) == "]\nText"


def test_the_stream_ending_on_a_closing_fence_keeps_the_line_outside_code():
    import asyncio, json as _j

    def fr(t):
        return ("data: " + _j.dumps({"id": "c", "choices": [{"index": 0, "delta": {"content": t}, "finish_reason": None}]})
                + "\n\n").encode()

    async def gen():
        for t in ['Revenue rose 9% [src: "made up quote here ok"].\n\n', "```\nprint(1)\n", "```"]:
            yield fr(t)
        yield b"data: [DONE]\n\n"

    async def run():
        return [c async for c in QB.filter_sse(gen(), [], [])]

    evs = [e for c in asyncio.run(run()) for e in c.decode().split("\n\n") if e.startswith("data: ")]
    txt = "".join((_j.loads(e[6:])["choices"][0]["delta"].get("content") or "") for e in evs if e[6:].strip() != "[DONE]")
    assert txt.count("```") == 2 and txt.endswith("”._")
