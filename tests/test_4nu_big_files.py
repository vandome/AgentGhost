"""§4NU (2026-10-11, owner request 7140be61 "write a webos…"): two 3.5-minute
generations of a whole index.html inside ONE file_system write were cut off
by the 16k-token tool-turn cap; the JSON-form call got a generic "Invalid
JSON arguments", so the model regenerated the same full-size write. The cut
must be named as a LENGTH problem, and a file of any size must be writable
in parts. Every test names the world it fails in."""
import asyncio
import json
from pathlib import Path

import pytest

from ghost_agent.core import agent as A
from ghost_agent.core.agent import _json_args_cut_off, cut_off_call_message
from ghost_agent.core.strikes import StrikeLedger
from ghost_agent.tools.file_system import tool_file_system
import tests.test_4jj_search_yield_steer as H

# the 7140be61 shape: arguments end inside `content`
CUT = ('{"operation":"write","path":"webos/index.html","content":"<!DOCTYPE html>\\n<html lang=\\"en\\">'
       + '\\n<div class=\\"app\\">x</div>' * 400)


def _err(raw):
    try:
        json.loads(raw, strict=False)
    except Exception as e:  # noqa: BLE001
        return e
    raise AssertionError("parsed")


def test_the_7140be61_shape_is_a_cut_off():
    assert _json_args_cut_off(CUT, _err(CUT))


@pytest.mark.parametrize("raw", ['{"operation": "read", "path": "a.txt"',      # short slip
                                 '{"operation": "write", "path": "a" "content": "x"}'])
def test_a_short_malformed_call_is_not_a_cut_off(raw):
    assert not _json_args_cut_off(raw, _err(raw))


def test_a_long_call_with_a_syntax_slip_in_the_middle_is_not_a_cut_off():
    raw = '{"operation": "write", "path": "a" "content": "' + "x" * 600 + '"}'
    assert not _json_args_cut_off(raw, _err(raw))


@pytest.mark.asyncio
async def test_the_model_is_told_it_was_cut_off_and_how_to_split():
    """Fails in the world of 7140be61: the row said only 'Invalid JSON
    arguments - Unterminated string', and the same write was regenerated."""
    agent = H._agent()
    ran = []

    async def fs(**kw):
        ran.append(kw)
        return "SUCCESS"
    agent.available_tools = {"file_system": fs}
    ts = H._ts([("file_system", {})], StrikeLedger(), set())
    # replace the well-formed arguments with the cut-off raw string
    # r4: a cut WRITE is now salvaged (tests below); an unsalvageable cut
    # (an edit) still gets the LENGTH message and runs nothing
    _cut_edit = CUT.replace('"operation":"write"', '"operation":"edit"', 1)
    for tc in getattr(ts, "tool_calls", []) or []:
        tc["function"]["arguments"] = _cut_edit
    await agent._dispatch_and_process_tool_batch(ts)
    assert ran == []
    rows = [m for m in ts.messages if m.get("role") == "tool"]
    assert rows, ts.messages
    body = str(rows[-1]["content"])
    assert getattr(rows[-1]["content"], "reason_code", None) == "tool_call_cut_off"
    assert "CUT OFF" in body and "LENGTH" in body and "append=true" in body
    assert "Invalid JSON arguments" not in body


def test_the_message_names_the_size_and_the_way_out():
    m = cut_off_call_message("file_system", 44242)
    assert "44,242" in m and "append=true" in m and "NOTHING ran" in m


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


def test_a_file_is_written_in_parts_and_is_whole(tmp_path):
    """Fails in a world with no append: a file bigger than one reply."""
    parts = ["<html><body>\n", "<div>" + "a" * 30000 + "</div>\n", "<p>end</p></body></html>\n"]
    r = _run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="big/index.html", content=parts[0]))
    assert str(r).startswith("SUCCESS")
    for p in parts[1:]:
        r = _run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="big/index.html",
                                  content=p, append=True))
        assert str(r).startswith("SUCCESS: Appended"), r
    # `write` drops part 1's trailing newline; the append restores it
    assert (tmp_path / "big/index.html").read_text().replace("\n", "") == "".join(parts).replace("\n", "")


def test_parts_split_at_line_boundaries_keep_their_lines(tmp_path):
    """Fails where part 1's stripped newline glued 'def f():' to the body."""
    _run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="m.py", content="def f():\n"))
    _run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="m.py",
                          content="    return 1\n", append=True))
    _run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="m.py",
                          content="\nprint(f())\n", append=True))
    src = (tmp_path / "m.py").read_text()
    assert src.startswith("def f():\n    return 1\n")
    compile(src, "m.py", "exec")


def test_append_needs_an_existing_file(tmp_path):
    r = _run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="new.html",
                              content="x", append=True))
    assert getattr(r, "reason_code", None) == "append_target_missing"
    assert not (tmp_path / "new.html").exists()


@pytest.mark.parametrize("val", [True, "true", "1"])
def test_append_flag_spellings(tmp_path, val):
    (tmp_path / "f.txt").write_text("a\n")
    _run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="f.txt", content="b\n", append=val))
    assert (tmp_path / "f.txt").read_text() == "a\nb\n"


@pytest.mark.parametrize("val", [False, "false", None])
def test_without_append_a_write_replaces(tmp_path, val):
    (tmp_path / "f.txt").write_text("a")
    kw = {} if val is None else {"append": val}
    _run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="f.txt", content="b", **kw))
    assert (tmp_path / "f.txt").read_text() == "b"


def test_append_stays_inside_the_sandbox(tmp_path):
    outside = tmp_path.parent / "outside_4nu.txt"
    outside.write_text("keep")
    try:
        _run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="../outside_4nu.txt",
                              content="x", append=True))
        assert outside.read_text() == "keep"
    finally:
        outside.unlink()


@pytest.mark.parametrize("first", [True, False])
def test_an_unfinished_file_is_not_booked_as_a_failure(tmp_path, first):
    """r1 MAJOR: part 1 of a .py is never complete — with more_parts=true it
    (and any middle part) is not a 'written_but_syntax_failed' partial whose
    'fix this BEFORE any other step' sends the model fixing a half file."""
    if first:
        r = _run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="a.py",
                                  content="def f():\n    return (1,\n", more_parts=True))
        assert str(r).startswith("SUCCESS: Wrote")
    else:
        (tmp_path / "a.py").write_text("def f():\n")
        r = _run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="a.py",
                                  content="    return (1,\n", append=True, more_parts=True))
        assert str(r).startswith("SUCCESS: Appended")
    assert getattr(r, "reason_code", None) != "written_but_syntax_failed"
    assert "More parts follow" in str(r) and "BEFORE any other step" not in str(r)


def test_the_last_part_is_checked(tmp_path):
    (tmp_path / "a.py").write_text("def f():\n")
    r = _run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="a.py",
                              content="    return (1,\n", append=True))
    assert getattr(r, "reason_code", None) == "written_but_syntax_failed"


def test_a_resent_part_is_not_appended_twice(tmp_path):
    """r1 MAJOR: a re-send after a cut-off or a 'continue' landed twice."""
    part = "<section>" + "x" * 200 + "</section>\n"
    (tmp_path / "p.html").write_text("<html>\n")
    _run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="p.html", content=part, append=True,
                          more_parts=True))
    r = _run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="p.html", content=part, append=True,
                              more_parts=True))
    assert getattr(r, "reason_code", None) == "duplicate_append"
    assert (tmp_path / "p.html").read_text() == "<html>\n" + part


def test_a_short_repeated_part_is_allowed(tmp_path):
    (tmp_path / "p.html").write_text("<div>\n</div>\n")
    _run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="p.html", content="</div>\n", append=True))
    assert (tmp_path / "p.html").read_text().count("</div>") == 2


def test_a_crlf_file_keeps_crlf_at_the_boundary(tmp_path):
    (tmp_path / "w.txt").write_bytes(b"a\r\nb")
    _run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="w.txt", content="c\r\n", append=True))
    assert (tmp_path / "w.txt").read_bytes() == b"a\r\nb\r\nc\r\n"


def test_the_added_line_break_is_reported(tmp_path):
    (tmp_path / "m.py").write_text("x = 1")
    r = _run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="m.py", content="y = 2\n", append=True))
    assert "line break was added" in str(r)


@pytest.mark.parametrize("bad", ["None", "<<HELD>>"])
def test_append_refuses_placeholder_content(tmp_path, bad):
    (tmp_path / "f.txt").write_text("a\n")
    r = _run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="f.txt", content=bad, append=True))
    assert getattr(r, "reason_code", None) == "missing_content"
    assert (tmp_path / "f.txt").read_text() == "a\n"


def test_a_dict_part_is_json(tmp_path):
    (tmp_path / "d.json").write_text("[\n")
    _run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="d.json", content={"b": 2}, append=True,
                          more_parts=True))
    assert '"b": 2' in (tmp_path / "d.json").read_text()


def test_a_fenced_sample_inside_a_page_is_not_extracted(tmp_path):
    """r1 (pre-existing): an HTML page with a ```js sample inside <pre> was
    saved as just the sample — 'Wrote 10 chars'."""
    page = ("<!DOCTYPE html>\n<html><body>\n<h1>Docs</h1>\n" + "<p>intro</p>\n" * 30
            + "<pre>\n```js\nconst a=1;\n```\n</pre>\n</body></html>")
    _run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="d.html", content=page))
    assert (tmp_path / "d.html").read_text() == page.strip()


def test_a_fully_wrapped_payload_is_still_unwrapped(tmp_path):
    _run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="u.py",
                          content="```python\nprint(1)\n```"))
    assert (tmp_path / "u.py").read_text().strip() == "print(1)"


def test_an_append_is_an_unverified_mutation():
    """r1 BLOCKER: the 'untested file write' gate never saw an append — a
    page built in parts finished clean without ever being run."""
    from ghost_agent.core.agent import _FILE_MUTATION_MARKERS
    assert "appended" in _FILE_MUTATION_MARKERS


def test_a_cut_xml_write_is_not_run():
    """r1 MAJOR: the XML parser recovered a cut <parameter name="content">
    and RAN the write with a fragment."""
    from ghost_agent.core.agent import _cut_call_must_not_run
    w = {"function": {"name": "file_system", "arguments": json.dumps({"operation": "write", "path": "i.html",
                                                                       "content": "<html>"})}}
    r = {"function": {"name": "file_system", "arguments": json.dumps({"operation": "read", "path": "i.html"})}}
    x = {"function": {"name": "execute", "arguments": json.dumps({"content": "print(1"})}}
    assert _cut_call_must_not_run(w) and _cut_call_must_not_run(x) and not _cut_call_must_not_run(r)


def test_the_artifact_ledger_sees_an_appended_file():
    """The verifier learns produced files by parsing write confirmations:
    an append must register (else WEB-EXEC / FILE-ARTIFACT never check it)."""
    line = ("SUCCESS: Appended 120 chars to 'webos/index.html'. Script-side path (from sandbox cwd): "
            "'webos/index.html'. The file is now 41,420 bytes.")
    assert A._FS_WROTE_PAIR_RE.match(line)
    assert any(rx.match(line) for rx in A._FS_PRODUCE_RES)
    assert not A._FS_WROTE_SIZE_RE.match(line)       # not a whole small write


def test_the_schema_offers_append_and_says_why():
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    fs = next(d for d in TOOL_DEFINITIONS if d["function"]["name"] == "file_system")
    props = fs["function"]["parameters"]["properties"]
    assert props["append"]["type"] == "boolean"
    assert "CUT OFF" in fs["function"]["description"] and "append=true" in fs["function"]["description"]


@pytest.mark.parametrize("dialect", [
    '<tool_call>\n<function=file_system>\n<parameter=operation>write</parameter>\n'
    '<parameter=path>webos/index.html</parameter>\n<parameter=content>\n<!DOCTYPE html>\n<html>' + "<div>x</div>\n" * 200,
    '<tool_call>\n<function name="file_system">\n<parameter name="operation">write</parameter>\n'
    '<parameter name="path">webos/index.html</parameter>\n<parameter name="content"><!DOCTYPE html>\n<html>'
    + "<div>x</div>\n" * 200,
])
def test_the_parser_does_not_run_a_cut_xml_write(dialect):
    """Wiring, not the helper: the real parser on a response the cap cut
    mid-`content` returns the truncation error, not a runnable write."""
    from ghost_agent.core.agent import GhostAgent
    agent = GhostAgent.__new__(GhostAgent)
    agent.available_tools = {n: (lambda **kw: None) for n in ("file_system", "execute", "web_search")}
    tool_calls, _ui, reason = agent._parse_assistant_tool_calls(dialect, {"content": dialect})
    names = [(t.get("function") or {}).get("name") for t in tool_calls]
    assert reason == "truncated"
    assert "file_system" not in names, tool_calls
    assert names == ["system_parse_error"]


def test_a_file_too_big_to_grow_by_appends_is_refused(tmp_path, monkeypatch):
    from ghost_agent.tools import file_system as FS
    monkeypatch.setattr(FS, "APPEND_MAX_FILE_BYTES", 100)
    (tmp_path / "b.txt").write_text("x" * 90 + "\n")
    r = _run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="b.txt", content="y" * 20, append=True))
    assert getattr(r, "reason_code", None) == "append_too_big"
    assert (tmp_path / "b.txt").read_text() == "x" * 90 + "\n"


_WRITE = ('<function=file_system>\n<parameter=operation>write</parameter>\n<parameter=path>a.txt</parameter>\n'
          '<parameter=content>hello</parameter>\n</function>')


@pytest.mark.parametrize("content", [
    # r2 MAJOR (production shape, 08-29): reasoning quotes the tag — the
    # response-wide count reads "cut", but the real call is complete
    'The state update says "DO NOT emit any <tool_call> blocks…"\n</think>\n\n<tool_call>\n' + _WRITE + '\n</tool_call>',
    # healed dialects with no </tool_call>
    _WRITE,
    '<tool_call>\n' + _WRITE,
])
def test_a_complete_call_is_never_refused_as_cut(content):
    from ghost_agent.core.agent import GhostAgent
    agent = GhostAgent.__new__(GhostAgent)
    agent.available_tools = {n: (lambda **kw: None) for n in ("file_system", "execute")}
    tool_calls, _ui, _reason = agent._parse_assistant_tool_calls(content, {"content": content})
    names = [(t.get("function") or {}).get("name") for t in tool_calls]
    # (a quoted tag's fragment may add its own pre-existing parse error)
    assert names[-1] == "file_system" and names.count("file_system") == 1, tool_calls


def test_a_complete_write_before_a_cut_call_still_runs():
    from ghost_agent.core.agent import GhostAgent
    agent = GhostAgent.__new__(GhostAgent)
    agent.available_tools = {n: (lambda **kw: None) for n in ("file_system", "execute")}
    content = ("<tool_call>\n" + _WRITE + "\n</tool_call>\n<tool_call>\n<function=execute>\n"
               "<parameter=content>print('" + "x" * 300)
    tool_calls, _ui, reason = agent._parse_assistant_tool_calls(content, {"content": content})
    names = [(t.get("function") or {}).get("name") for t in tool_calls]
    assert reason == "truncated"
    assert names == ["file_system", "system_parse_error"], names


@pytest.mark.parametrize("raw", ['{"operation": "write", "path": "a", "content": "' + "x" * 300 + '",}',
                                 '{"operation": "write", "path": "a", "content": "' + "x" * 300 + '"}}'])
def test_a_complete_call_with_a_slip_at_its_end_is_not_a_cut(raw):
    assert not _json_args_cut_off(raw, _err(raw))


def test_a_cut_block_that_produced_no_call_spares_the_complete_one():
    """r2 (case 3): a malformed block, a COMPLETE write, then a cut opener whose
    error the truncation dedup skipped — the last CALL is the complete write
    and must run; only calls the cut block itself produced are refused."""
    from ghost_agent.core.agent import GhostAgent
    agent = GhostAgent.__new__(GhostAgent)
    agent.available_tools = {n: (lambda **kw: None) for n in ("file_system", "execute")}
    content = "<tool_call>\n<function=\n<tool_call>\n" + _WRITE + "\n</tool_call>\n<tool_call>\n<function="
    tool_calls, _ui, reason = agent._parse_assistant_tool_calls(content, {"content": content})
    names = [(t.get("function") or {}).get("name") for t in tool_calls]
    assert reason == "truncated"
    assert names == ["system_parse_error", "file_system"], names    # no extra cut-off error either


@pytest.mark.parametrize("body", [
    # r3 MAJOR: a prefix closer test matched `</FunctionList>` in a JSX body
    "return (<FunctionList>\n  <li/>\n</FunctionList>);\n" + "const x = 1;\n" * 40,
    "<functional/>\n</functional>\n" + "x\n" * 60,
    # r3: an unterminated CDATA body hides a literal closer
    '<![CDATA[x = "</function>"\n' + "y = 2\n" * 60,
])
@pytest.mark.parametrize("name", ["file_system", "filesystem"])   # r3: aliases dispatch as file_system
def test_a_cut_write_with_closer_like_text_still_does_not_run(body, name):
    from ghost_agent.core.agent import GhostAgent
    agent = GhostAgent.__new__(GhostAgent)
    agent.available_tools = {n: (lambda **kw: None) for n in ("file_system", "execute")}
    content = (f"<tool_call>\n<function={name}>\n<parameter=operation>write</parameter>\n"
               f"<parameter=path>App.jsx</parameter>\n<parameter=content>\n" + body)
    tool_calls, _ui, reason = agent._parse_assistant_tool_calls(content, {"content": content})
    names = [(t.get("function") or {}).get("name") for t in tool_calls]
    assert reason == "truncated"
    assert names == ["system_parse_error"], names


# ── §4NU r4: the derailed live probe (probe-4nu-story-1791751511) ────────────
# The model PLANNED parts of 2-3k words, then emitted 72,662 and 73,083 chars in
# one write — both cut and discarded (8.5 min) — then sent "Part 2" as a plain
# write that REPLACED Part 1, three times.
from ghost_agent.core.agent import _salvage_cut_write


def _story_cut(n_lines=400, head='{"operation":"write","path":"probe4nu/story.md","content":"'):
    body = "".join(f'Line {i} of the story, with a \\"quote\\" and an em dash \\u2014 here.\\n' for i in range(n_lines))
    return head + body + "Line cut mid-wor"


def test_a_cut_write_is_salvaged_up_to_its_last_complete_line():
    args = _salvage_cut_write(_story_cut())
    assert args["operation"] == "write" and args["path"] == "probe4nu/story.md"
    assert args["more_parts"] is True and "cut_off_saved" not in args    # r5: out of band
    assert args["content"].endswith("here.\n") and "cut mid-wor" not in args["content"]
    assert '"quote"' in args["content"] and "—" in args["content"]


@pytest.mark.parametrize("tail", ["\\", "\\u20", "\\u"])
def test_a_cut_inside_an_escape_still_salvages(tail):
    assert _salvage_cut_write(_story_cut() + tail)


@pytest.mark.parametrize("raw", [
    _story_cut(n_lines=5),                                                    # too short to be worth a part
    _story_cut(head='{"operation":"edit","path":"a.md","content":"'),          # not a write
    _story_cut(head='{"operation":"write","content":"'),                       # path after content: unknown
])
def test_unsalvageable_cuts_return_none(raw):
    assert _salvage_cut_write(raw) is None


def test_the_salvaged_part_is_written_and_the_model_told_to_continue(tmp_path):
    args = _salvage_cut_write(_story_cut())
    r = _run(tool_file_system(sandbox_dir=tmp_path, cut_off_saved=30000, **args))
    assert str(r).startswith("SUCCESS: Wrote")
    assert "CUT OFF" in str(r) and "append=true" in str(r) and "Line 399 of the story" in str(r)
    # r5: the excerpt is the file's END, not the start of its last long lines
    assert "dash \u2014 here.\n>>>" in str(r)
    assert (tmp_path / "probe4nu/story.md").read_text().endswith("here.\n")


def test_a_salvaged_cut_never_replaces_an_existing_file(tmp_path):
    (tmp_path / "probe4nu").mkdir()
    (tmp_path / "probe4nu/story.md").write_text("the old story\n")
    args = _salvage_cut_write(_story_cut())
    r = _run(tool_file_system(sandbox_dir=tmp_path, cut_off_saved=30000, **args))
    assert getattr(r, "reason_code", None) == "tool_call_cut_off"
    assert (tmp_path / "probe4nu/story.md").read_text() == "the old story\n"
    # r5: the cut call may have been an APPEND (flags come after content):
    # never advise a plain first write without saying "if adding, append"
    assert "If you were ADDING to it" in str(r)


def test_a_plain_write_cannot_replace_a_file_in_progress(tmp_path):
    """The live derail: 'Part 2' as write+more_parts replaced Part 1."""
    _run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="s.md", content="part one\n" * 50,
                          more_parts=True))
    r = _run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="s.md", content="part two\n" * 50,
                              more_parts=True))
    assert getattr(r, "reason_code", None) == "file_in_progress"
    assert (tmp_path / "s.md").read_text().startswith("part one")
    r = _run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="s.md", content="part two\n" * 50,
                              append=True))
    assert str(r).startswith("SUCCESS: Appended")
    # the last part (no more_parts) finishes the file: a plain write is allowed again
    r = _run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="s.md", content="fresh\n"))
    assert str(r).startswith("SUCCESS: Wrote")


@pytest.mark.asyncio
async def test_the_dispatcher_saves_a_cut_native_write(tmp_path):
    """Wiring: the real batch dispatcher turns the cut call into a saved part."""
    agent = H._agent()
    seen = []

    async def fs(**kw):
        seen.append(kw)
        return "SUCCESS: Wrote 1 chars to 'x'."
    agent.available_tools = {"file_system": fs}
    ts = H._ts([("file_system", {})], StrikeLedger(), set())
    for tc in getattr(ts, "tool_calls", []) or []:
        tc["function"]["arguments"] = _story_cut()
    await agent._dispatch_and_process_tool_batch(ts)
    assert seen and seen[0]["more_parts"] is True and seen[0]["cut_off_saved"] > 1000
    assert seen[0]["content"].endswith("here.\n")
    # r5: the marker never enters the call the model sees in its history
    assert "cut_off_saved" not in json.loads(ts.tool_calls[0]["function"]["arguments"])


def test_a_cut_part_of_a_file_in_progress_is_appended(tmp_path):
    """r5 MAJOR: the model writes append/more_parts AFTER content, so a cut
    APPEND arrives as a plain write — refused as 'file_in_progress' and the
    part thrown away. It is the file's next part: appended."""
    _run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="s.md", content="part one\n" * 50,
                          more_parts=True))
    r = _run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="s.md", content="part two\n" * 50,
                              more_parts=True, cut_off_saved=30000))
    assert str(r).startswith("SUCCESS: Appended"), r
    assert (tmp_path / "s.md").read_text() == "part one\n" * 50 + "part two\n" * 50


def test_a_cut_restart_of_a_file_in_progress_writes_nothing(tmp_path):
    _run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="s.md", content="# Title\n" + "a\n" * 300,
                          more_parts=True))
    before = (tmp_path / "s.md").read_text()
    r = _run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="s.md", content="# Title\n" + "a\n" * 300,
                              more_parts=True, cut_off_saved=30000))
    assert getattr(r, "reason_code", None) == "tool_call_cut_off"
    assert (tmp_path / "s.md").read_text() == before


@pytest.mark.asyncio
async def test_a_model_supplied_marker_is_stripped():
    """r5 MAJOR: the tool trusted `cut_off_saved` from anyone — a model that
    copied it got a false 'CUT OFF … in progress' note or a refused write."""
    agent = H._agent()
    seen = []

    async def fs(**kw):
        seen.append(kw)
        return "SUCCESS: Wrote 1 chars to 'x'."
    agent.available_tools = {"file_system": fs}
    ts = H._ts([("file_system", {"operation": "write", "path": "a.md", "content": "hi",
                                 "cut_off_saved": 99999})], StrikeLedger(), set())
    await agent._dispatch_and_process_tool_batch(ts)
    assert seen and "cut_off_saved" not in seen[0]


# ── §4NU r6: the second live probe (probe-4nu-story2) grew the story with
# `edit` (old_string = the closing line, new_string = it + the next chapter);
# story text repeats lines ("The end."), the anchor was never unique, and the
# model detoured into a second file and then a script.
def test_an_ambiguous_add_at_the_end_points_at_append(tmp_path):
    (tmp_path / "s.md").write_text("Chapter 1\nThe end.\n\nChapter 2\nThe end.\n")
    r = _run(tool_file_system(operation="edit", sandbox_dir=tmp_path, path="s.md",
                              old_string="The end.", new_string="The end.\n\nChapter 3\nMore."))
    assert getattr(r, "reason_code", None) == "old_string_ambiguous"
    assert "append=true" in str(r)


def test_an_ambiguous_edit_in_the_middle_gets_no_append_hint(tmp_path):
    (tmp_path / "s.md").write_text("a\nx\nb\nx\nc\n")
    r = _run(tool_file_system(operation="edit", sandbox_dir=tmp_path, path="s.md",
                              old_string="x", new_string="y"))
    assert getattr(r, "reason_code", None) == "old_string_ambiguous"
    assert "append=true" not in str(r)


def test_the_schema_says_add_at_the_end_with_append():
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    from ghost_agent.core.agent import BIG_FILE_GUIDANCE
    fs = next(d for d in TOOL_DEFINITIONS if d["function"]["name"] == "file_system")
    assert "END of a file always use append=true, never an 'edit'" in fs["function"]["description"]
    assert "never with an 'edit'" in BIG_FILE_GUIDANCE
