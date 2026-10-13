"""§4OG-3 — the research pipeline, second pass: a memo keyed by operation and
selector that never memoises a truncated page and remembers failures; and a
link check on every document this request writes (the §4OF report cited a
URL the browser answered 404 twice)."""
from ghost_agent.core import fetch_memo as FM
from ghost_agent.core.agent import document_link_check_note
from ghost_agent.core.link_grounding import unfetched_links, _cite_norm
from ghost_agent.core import agent as ag
from tests.pin_ast import tree, U, calls, call_kw, assigns, ifs

NAV = ("--- BROWSER RESULT ---\nSTATUS: OK\nOP: navigate\nURL: https://ex.org/a\nHTTP_STATUS: 200\n"
       "LENGTH: 29704 (truncated)\n--- PAGE TEXT (capped preview) ---\nThe page says the thing " + "x " * 500)
NAV_FULL = NAV.replace("LENGTH: 29704 (truncated)", "LENGTH: 1200")
FULL = "--- BROWSER RESULT ---\nSTATUS: OK\nOP: extract_text\nURL: https://ex.org/a\nLENGTH: 29704\n--- TEXT ---\nAll of it " + "y " * 2000
DEAD = "--- BROWSER RESULT ---\nSTATUS: BLOCKED (HTTP 404)\nOP: navigate\nURL: https://ex.org/dead\nHINT: There is no page at this URL."


def test_navigate_and_extract_are_different_keys_and_a_truncated_preview_is_not_memoised():
    """Fails where (tool, url) keyed every op and a 2,000-char head of an
    8 KB preview answered the full read (§4OF defect 3)."""
    memo = {}
    nav = FM.fetch_key("browser", {"operation": "navigate", "url": "https://ex.org/a/"})
    ext = FM.fetch_key("browser", {"operation": "extract_text", "url": "https://ex.org/a", "selector": "main"})
    ext2 = FM.fetch_key("browser", {"operation": "extract_text", "url": "https://ex.org/a", "max_chars": 60000})
    assert nav[2] == "navigate" and ext[2] == "extract" and nav != ext and ext != ext2
    assert FM.remember(memo, nav, NAV) is False and FM.repeat_result(memo, nav) is None     # truncated: not kept
    assert FM.remember(memo, nav, NAV_FULL) is True
    assert FM.repeat_result(memo, nav).startswith("ALREADY READ") and FM.repeat_result(memo, ext) is None
    assert FM.remember(memo, ext, FULL) is True and "All of it" in FM.repeat_result(memo, ext)
    assert FM.repeat_result(memo, ext2) is None                                   # a different max_chars runs
    assert FM.is_truncated_result("LENGTH: 5 (truncated)") and not FM.is_truncated_result("LENGTH: 5\n--- TEXT")
    # the LENGTH line sits after the JS diagnostics (reader 2 M3): still found; a "(truncated)" in the PAGE TEXT is not the header
    diag = "--- BROWSER RESULT ---\nSTATUS: OK\nURL: https://ex.org/a\n" + "JS EXCEPTION: TypeError x\n  at line\n" * 60
    assert FM.is_truncated_result(diag + "LENGTH: 29704 (truncated)\n--- PAGE TEXT (capped preview) ---\nbody")
    assert not FM.is_truncated_result("--- BROWSER RESULT ---\nSTATUS: OK\nLENGTH: 900\n--- PAGE TEXT (capped preview) ---\n" + "x (truncated) " * 50)


def test_a_declared_failure_is_remembered_by_url_and_cleared_by_a_later_success():
    """Fails where only OK results were memoised: 4 of 4 probes re-fetched a
    dead URL (§4OF lens D MAJOR-5)."""
    memo = {}
    k = FM.fetch_key("browser", {"operation": "navigate", "url": "https://ex.org/dead"})
    assert FM.remember_failure(memo, k, DEAD) is True
    rep = FM.repeat_result(memo, k)
    assert rep.startswith("--- BROWSER RESULT ---\nSTATUS: BLOCKED (already failed this request)") and "ALREADY FAILED" in rep
    assert "BLOCKED (HTTP 404)" in rep and "do not fetch it again" in rep
    # …so the grounding treats the repeat as the failure it is: the dead URL vouches for nothing (reader 2 M4)
    from ghost_agent.core.link_grounding import _failed_browser_result, fetched_urls
    assert _failed_browser_result(rep)
    assert fetched_urls([{"name": "browser", "args": {"url": "https://ex.org/dead"}, "content": rep}]) == []
    # any operation on that URL is answered the same way
    k2 = FM.fetch_key("browser", {"operation": "extract_text", "url": "https://ex.org/dead", "selector": "p"})
    assert "ALREADY FAILED" in FM.repeat_result(memo, k2)
    # a later success clears it
    assert FM.remember(memo, k, NAV_FULL.replace("ex.org/a", "ex.org/dead")) is True
    assert FM.repeat_result(memo, k).startswith("ALREADY READ")
    assert FM.remember_failure(memo, None, DEAD) is False


def test_the_dispatcher_remembers_failures_and_marks_failed_repeats():
    t = tree(ag)
    assert any(calls(i, "_fremember_failure") for i in ifs(t, "_res_is_error")), "failures are not memoised"
    failed = [c for c in calls(t, "failed") if U(c.func) == "_TO.failed"
              and U(call_kw(c, "reason_code")) == "'repeat_failed_fetch'"]
    assert failed, "a failed repeat is not filed as a failure"
    # the link note is APPENDED to the write's result the model sees (battery mutant L5)
    assert calls(t, "document_link_check_note")
    assert any(U(a.value) == "str(str_res) + _lc_note" for a in assigns(t, "str_res"))


def test_unfetched_links_normalise_spellings_and_skip_code():
    fetched = ["https://www.Ex.org/docs/a/", "https://ex.org/b#top", "https://ex.org/c"]
    text = ("See [a](https://ex.org/docs/a) and https://ex.org/b and `https://ex.org/zzz` and "
            "https://other.net/x) plus https://ex.org/c/ and https://ex.org/d.")
    assert unfetched_links(text, fetched) == ["https://other.net/x", "https://ex.org/d"]
    assert _cite_norm("https://WWW.ex.org/p/#frag") == "ex.org/p" and _cite_norm("http://ex.org") == "ex.org"


def test_a_written_report_is_link_checked_against_the_pages_read():
    """Fails where a 12 KB report written as a file was grounded by nothing
    (reply-side grounding only) and cited a twice-404 URL (§4OF defect 4)."""
    rows = [
        {"name": "browser", "args": {"operation": "navigate", "url": "https://ex.org/a"},
         "content": "--- BROWSER RESULT ---\nSTATUS: OK\nOP: navigate\nURL: https://ex.org/a\nHTTP_STATUS: 200\nLENGTH: 10\n"},
        {"name": "browser", "args": {"operation": "navigate", "url": "https://ex.org/dead"},
         "content": "--- BROWSER RESULT ---\nSTATUS: BLOCKED (HTTP 404)\nOP: navigate\nURL: https://ex.org/dead\n"},
    ]
    body = "# Report\n\n## Sources\n- https://ex.org/a\n- https://ex.org/dead\n- https://never.example/p\n"
    note = document_link_check_note({"operation": "write", "path": "r/report.md", "content": body}, rows)
    assert note.startswith("\nLINK CHECK: 2 URL(s)") and "https://ex.org/dead" in note and "https://never.example/p" in note
    assert "https://ex.org/a" not in note.split("were NOT")[1].split(". A report")[0]
    assert document_link_check_note({"operation": "write", "path": "r/report.md", "content": "# no links\n"}, rows) == ""
    assert document_link_check_note({"operation": "write", "path": "app.py", "content": "u = 'https://never.example'"}, rows) == ""
    assert document_link_check_note({"operation": "edit", "path": "r/report.md", "content": body}, rows) == ""
    assert document_link_check_note({"operation": "append", "path": "r/notes.txt", "content": "https://ex.org/a"}, rows) == ""
    # the STREAMED report write is link-checked too, and recorded (reader 2 M6)
    h = tree(ag.GhostAgent._run_streamed_report_writes)
    lc = calls(h, "document_link_check_note")
    assert lc and [U(a) for a in lc[0].args] == ["args", "list(tools_snapshot or [])"]
    assert any(U(a.value) == "_with_text(out, str(out) + _lc)" for a in assigns(h, "out"))
    d = tree(ag.GhostAgent._stream_final_generation)
    ext = [c for c in calls(d, "extend") if U(c.func) == "messages.extend"]
    assert ext and "_ff_rows" in U(ext[0].args[0])
    app = [c for c in calls(d, "append") if U(c.func) == "messages.append" and "_ff_report_calls" in U(c)]
    assert app and "'tool_calls'" in U(app[0])
