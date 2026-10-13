"""§4NZ (2026-10-12, operator: "verify that append in file system works, i
don't think it does … ensure that it works under different angles"). The
append path, driven through the real `tool_file_system` from the angles a
model actually takes. Every test names the world it fails in."""
import asyncio
from pathlib import Path

import pytest

from ghost_agent.tools.file_system import tool_file_system
from ghost_agent.utils.logging import request_id_context


@pytest.fixture(autouse=True)
def _rid(request):
    tok = request_id_context.set(f"req-ap-{request.node.name[:40]}")
    yield
    request_id_context.reset(tok)


def fs(sb, **kw):
    return asyncio.run(tool_file_system(sandbox_dir=sb, **kw))


def test_three_python_parts_reassemble_line_for_line(tmp_path):
    p = ["def a():\n    return 1\n", "def b():\n    return 2\n", "print(a() + b())\n"]
    rs = [fs(tmp_path, operation="write", path="m.py", content=p[0], more_parts=True),
          fs(tmp_path, operation="write", path="m.py", content=p[1], append=True, more_parts=True),
          fs(tmp_path, operation="write", path="m.py", content=p[2], append=True)]
    assert all(str(r).startswith("SUCCESS") for r in rs)
    assert (tmp_path / "m.py").read_text().splitlines() == "".join(p).splitlines()


@pytest.mark.parametrize("flag", [True, "true", "True", "1", "yes"])
def test_the_append_flag_as_the_model_spells_it(tmp_path, flag):
    fs(tmp_path, operation="write", path="x.txt", content="one\n", more_parts=True)
    fs(tmp_path, operation="write", path="x.txt", content="two\n", append=flag)
    assert (tmp_path / "x.txt").read_text().splitlines() == ["one", "two"]


@pytest.mark.parametrize("op", ["append", "APPEND", " append_file "])
def test_operation_append_is_write_with_append(tmp_path, op):
    """Fails where operation='append' — the name a model reaches for — got
    "Unknown operation" and cost a round trip."""
    fs(tmp_path, operation="write", path="q.txt", content="a\n", more_parts=True)
    r = fs(tmp_path, operation=op, path="q.txt", content="b\n")
    assert str(r).startswith("SUCCESS: Appended") and (tmp_path / "q.txt").read_text().splitlines() == ["a", "b"]


def test_a_big_page_in_four_parts_is_one_whole_document(tmp_path):
    head = "<!DOCTYPE html>\n<html>\n<head><title>T</title></head>\n<body>\n<script>\n"
    body = "".join(f"const v{i} = {i};\n" for i in range(3000))
    fs(tmp_path, operation="write", path="g/index.html", content=head, more_parts=True)
    fs(tmp_path, operation="write", path="g/index.html", content=body[:len(body) // 2], append=True, more_parts=True)
    fs(tmp_path, operation="write", path="g/index.html", content=body[len(body) // 2:], append=True, more_parts=True)
    r = fs(tmp_path, operation="write", path="g/index.html", content="</script>\n</body>\n</html>\n", append=True)
    t = (tmp_path / "g/index.html").read_text()
    assert str(r).startswith("SUCCESS") and t.count("<!DOCTYPE") == 1 and t.count("const v") == 3000
    assert t.rstrip().endswith("</html>")


def test_a_part_without_its_last_newline_still_starts_a_new_line(tmp_path):
    fs(tmp_path, operation="write", path="e.py", content="x = 1", more_parts=True)
    r = fs(tmp_path, operation="write", path="e.py", content="y = 2\n", append=True)
    assert (tmp_path / "e.py").read_text().splitlines() == ["x = 1", "y = 2"]
    assert "line break was added" in str(r)          # the model is told


def test_crlf_and_unicode_parts(tmp_path):
    (tmp_path / "w.txt").write_bytes(b"a\r\nb")
    fs(tmp_path, operation="write", path="w.txt", content="c\r\n", append=True)
    assert (tmp_path / "w.txt").read_bytes() == b"a\r\nb\r\nc\r\n"
    fs(tmp_path, operation="write", path="u.md", content="# Τίτλος 🚀\n", more_parts=True)
    fs(tmp_path, operation="write", path="u.md", content="Κείμενο — 中文\n", append=True)
    assert (tmp_path / "u.md").read_text(encoding="utf-8") == "# Τίτλος 🚀\nΚείμενο — 中文\n"


def test_a_short_repeated_part_is_appended(tmp_path):
    fs(tmp_path, operation="write", path="i.js", content="function a(){\n}\n", more_parts=True)
    fs(tmp_path, operation="write", path="i.js", content="}\n", append=True, more_parts=True)
    assert (tmp_path / "i.js").read_text().count("}") == 2


def test_append_in_a_project_workspace_and_to_an_earlier_requests_file(tmp_path):
    proj = tmp_path / "projects" / "abc123"
    proj.mkdir(parents=True)
    fs(proj, operation="write", path="app.py", content="x=1\n", more_parts=True)
    fs(proj, operation="write", path="app.py", content="y=2\n", append=True)
    assert (proj / "app.py").read_text().splitlines() == ["x=1", "y=2"]
    fs(tmp_path, operation="write", path="log.txt", content="day1\n")
    tok = request_id_context.set("req-ap-next-request")
    try:
        fs(tmp_path, operation="write", path="log.txt", content="day2\n", append=True)
    finally:
        request_id_context.reset(tok)
    assert (tmp_path / "log.txt").read_text().splitlines() == ["day1", "day2"]


def test_refusals_change_nothing(tmp_path):
    r = fs(tmp_path, operation="write", path="nope.txt", content="x\n", append=True)
    assert not (tmp_path / "nope.txt").exists() and "does not exist" in str(r)
    fs(tmp_path, operation="write", path="o.txt", content="a\n", more_parts=True)
    fs(tmp_path, operation="write", path="o.txt", content="", append=True)
    assert (tmp_path / "o.txt").read_text() == "a\n"
    fs(tmp_path, operation="write", path="../escape.txt", content="x", append=True)
    assert not (tmp_path.parent / "escape.txt").exists()


def test_only_the_last_part_is_syntax_checked(tmp_path):
    r1 = fs(tmp_path, operation="write", path="n.py", content="def f(:\n", more_parts=True)
    r2 = fs(tmp_path, operation="write", path="n.py", content="    pass\n", append=True)
    assert "SYNTAX CHECK FAILED" not in str(r1) and "syntax" in str(r2).lower()


STORY = ("## Chapter 1\nThey sailed.\nThe end of the day came.\n\n"
         "## Chapter 2\nThey slept.\nThe end of the day came.\n")


def test_an_add_at_the_end_edit_with_a_repeated_anchor_is_applied_at_the_end(tmp_path):
    """Fails where the model grew a story with `edit` (old_string = the
    closing line, new_string = it + the next chapter) and the closing line
    repeated — "old_string occurs 2 times", refused (probe-ap-d; the model
    chose edit over append 5 of 8 times in a live experiment)."""
    (tmp_path / "s.md").write_text(STORY)
    old = "The end of the day came."
    r = fs(tmp_path, operation="edit", path="s.md", old_string=old,
           new_string=old + "\n\n## Chapter 3\nThey reached Paros.")
    t = (tmp_path / "s.md").read_text()
    assert str(r).startswith("SUCCESS: edited") and "applied at the END" in str(r)
    assert t == STORY.rstrip("\n") + "\n\n## Chapter 3\nThey reached Paros.\n"
    assert t.count("## Chapter 1") == 1 and t.count("## Chapter 2") == 1


def test_an_ambiguous_edit_that_is_not_at_the_end_is_still_refused(tmp_path):
    (tmp_path / "s.md").write_text(STORY + "## Chapter 3\nNot the end.\n")
    old = "The end of the day came."
    r = fs(tmp_path, operation="edit", path="s.md", old_string=old, new_string=old + "\nMore.")
    assert getattr(r, "reason_code", None) == "old_string_ambiguous"


def test_an_ambiguous_edit_that_changes_the_anchor_is_still_refused(tmp_path):
    """Only a pure ADD qualifies — a rewrite of the last line is ambiguous."""
    (tmp_path / "s.md").write_text(STORY)
    r = fs(tmp_path, operation="edit", path="s.md", old_string="The end of the day came.",
           new_string="The night fell.")
    assert getattr(r, "reason_code", None) == "old_string_ambiguous"


def test_a_cut_add_at_the_end_edit_is_saved_as_an_append(tmp_path):
    """Fails where a long add-at-the-end edit cut by the output limit was
    lost whole (probe-ap-b: 79 KB, 222 s)."""
    import json as _json
    from ghost_agent.core.agent import _salvage_cut_write
    (tmp_path / "s.md").write_text(STORY)
    old = "The end of the day came."
    chapter = "".join(f"Line {i} of the new chapter.\n" for i in range(80))
    raw = _json.dumps({"operation": "edit", "path": "s.md", "old_string": old,
                       "new_string": old + "\n\n## Chapter 3\n" + chapter})[:-200]   # cut
    salv = _salvage_cut_write(raw)
    assert salv and salv["append"] is True and salv["_tail_anchor"] == old
    anchor = salv.pop("_tail_anchor")
    r = fs(tmp_path, cut_off_saved=len(raw), append_after=anchor, **salv)
    t = (tmp_path / "s.md").read_text()
    assert str(r).startswith("SUCCESS: Appended") and t.startswith(STORY) and "## Chapter 3" in t
    assert t.count("## Chapter 1") == 1


def test_a_cut_edit_whose_anchor_is_not_the_end_changes_nothing(tmp_path):
    from ghost_agent.core.agent import _salvage_cut_write
    import json as _json
    (tmp_path / "s.md").write_text(STORY + "## Chapter 3\nlater\n")
    old = "The end of the day came."
    raw = _json.dumps({"operation": "edit", "path": "s.md", "old_string": old,
                       "new_string": old + "\n" + "x line\n" * 400})[:-100]
    salv = _salvage_cut_write(raw)
    anchor = salv.pop("_tail_anchor")
    before = (tmp_path / "s.md").read_text()
    r = fs(tmp_path, cut_off_saved=len(raw), append_after=anchor, **salv)
    assert getattr(r, "reason_code", None) == "tool_call_cut_off" and (tmp_path / "s.md").read_text() == before


def test_a_cut_edit_that_rewrites_rather_than_adds_is_not_salvaged():
    import json as _json
    from ghost_agent.core.agent import _salvage_cut_write
    raw = _json.dumps({"operation": "edit", "path": "s.md", "old_string": "A",
                       "new_string": "B" + "y\n" * 2000})[:-50]
    assert _salvage_cut_write(raw) is None


def test_a_cut_restart_of_a_file_this_request_wrote_is_the_new_version(tmp_path):
    """Fails where the model wrote a skeleton, re-sent the whole page from
    <!DOCTYPE, was cut off — and the part was refused as "a partial
    replacement of an existing file" (probe-ap-b, 4 minutes lost, while the
    agent's own log said it was saved)."""
    head = '<!DOCTYPE html>\n<html>\n<head><title>Q</title></head>\n'
    fs(tmp_path, operation="write", path="p/q.html", content=head + "<body></body>\n</html>\n")   # skeleton
    big = head + "<body>\n<script>\nconst q = [\n" + "".join(f'"quote {i}",\n' for i in range(400))
    r = fs(tmp_path, operation="write", path="p/q.html", content=big, more_parts=True, cut_off_saved=80000)
    assert str(r).startswith("SUCCESS") and (tmp_path / "p/q.html").read_text() == big
    assert '"path": "p/q.html", "append": true' in str(r)      # the exact next call


def test_a_cut_write_over_an_earlier_requests_file_still_writes_nothing(tmp_path):
    head = '<!DOCTYPE html>\n<html>\n'
    (tmp_path / "owner.html").write_text(head + "<body>owner's page</body></html>\n")
    r = fs(tmp_path, operation="write", path="owner.html", content=head + "x\n" * 900,
           more_parts=True, cut_off_saved=80000)
    assert getattr(r, "reason_code", None) == "tool_call_cut_off"
    assert "owner's page" in (tmp_path / "owner.html").read_text()


@pytest.mark.asyncio
async def test_the_dispatcher_turns_a_cut_add_at_the_end_edit_into_an_append(tmp_path):
    """Wiring: the anchor travels out of band — a model-supplied
    append_after is dropped."""
    import json as _json
    import tests.test_4jj_search_yield_steer as H
    from ghost_agent.core.strikes import StrikeLedger
    agent = H._agent()
    seen = []

    async def fsx(**kw):
        seen.append(kw)
        return "SUCCESS: Appended 1 chars to 's.md'."
    agent.available_tools = {"file_system": fsx}
    old = "The end of the day came."
    raw = _json.dumps({"operation": "edit", "path": "s.md", "old_string": old,
                       "new_string": old + "\n" + "".join(f"Line {i}.\n" for i in range(300))})[:-80]
    ts = H._ts([("file_system", {})], StrikeLedger(), set())
    for tc in getattr(ts, "tool_calls", []) or []:
        tc["function"]["arguments"] = raw
    await agent._dispatch_and_process_tool_batch(ts)
    assert seen and seen[0]["append"] is True and seen[0]["append_after"] == old
    assert "_tail_anchor" not in seen[0]
    forged = H._ts([("file_system", {"operation": "write", "path": "s.md", "content": "x",
                                     "append": True, "append_after": "anything"})], StrikeLedger(), set())
    await agent._dispatch_and_process_tool_batch(forged)
    assert "append_after" not in seen[-1]


# ── review round: no duplicates, no bypass ──────────────────────────────── #

def test_a_re_sent_add_at_the_end_edit_is_refused(tmp_path):
    """Fails where an identical re-send landed at the end again (review)."""
    (tmp_path / "s.md").write_text(STORY)
    old = "The end of the day came."
    new = old + "\n\n## Chapter 3\nThey reached Paros.\nThe end of the day came."
    fs(tmp_path, operation="edit", path="s.md", old_string=old, new_string=new)
    once = (tmp_path / "s.md").read_text()
    r = fs(tmp_path, operation="edit", path="s.md", old_string=old, new_string=new)
    assert getattr(r, "reason_code", None) == "duplicate_append" and (tmp_path / "s.md").read_text() == once


def test_after_a_salvaged_edit_the_full_resend_is_refused_not_duplicated(tmp_path):
    """Fails where a cut edit was saved as a part and the complete re-send
    of the same edit then applied mid-file (review MAJOR)."""
    import json as _json
    from ghost_agent.core.agent import _salvage_cut_write
    (tmp_path / "u.md").write_text("# T\nunique anchor line\n")
    old = "unique anchor line"
    chapter = "".join(f"Ch2 line {i}\n" for i in range(150))
    raw = _json.dumps({"operation": "edit", "path": "u.md", "old_string": old, "new_string": old + "\n" + chapter})[:-300]
    salv = _salvage_cut_write(raw)
    anchor = salv.pop("_tail_anchor")
    fs(tmp_path, cut_off_saved=len(raw), append_after=anchor, **salv)
    saved = (tmp_path / "u.md").read_text()
    assert saved.count("Ch2 line 0") == 1 and "unique anchor line\nCh2 line 0" in saved   # no blank line
    r = fs(tmp_path, operation="edit", path="u.md", old_string=old, new_string=old + "\n" + chapter[:2000])
    assert getattr(r, "reason_code", None) == "file_in_progress"
    assert (tmp_path / "u.md").read_text() == saved


def test_a_cut_edit_that_continues_the_anchor_line_is_not_salvaged():
    import json as _json
    from ghost_agent.core.agent import _salvage_cut_write
    raw = _json.dumps({"operation": "edit", "path": "s.md", "old_string": "The last line",
                       "new_string": "The last line" + " goes on and on" * 300 + "\nmore\n" * 50})[:-40]
    assert _salvage_cut_write(raw) is None


def test_an_append_to_an_owners_file_does_not_make_it_this_requests(tmp_path):
    """Fails where a small append to the owner's page marked it "written",
    so a later cut restart replaced the owner's page with a fragment."""
    head = '<!DOCTYPE html>\n<html>\n'
    (tmp_path / "report.html").write_text(head + "<body>owner's report</body></html>\n")
    fs(tmp_path, operation="write", path="report.html", content="<!-- note -->\n", append=True)
    r = fs(tmp_path, operation="write", path="report.html", content=head + "x\n" * 900,
           more_parts=True, cut_off_saved=80000)
    assert getattr(r, "reason_code", None) == "tool_call_cut_off"
    assert "owner's report" in (tmp_path / "report.html").read_text()


@pytest.mark.parametrize("op", ["append", "Append", "append_file", "write_append"])
def test_a_cut_append_alias_call_never_runs(op):
    """Fails where a cut XML call with operation='append' was run on its
    fragment — the cut-call guard listed only write/edit/replace."""
    from ghost_agent.core.agent import _cut_call_must_not_run
    assert _cut_call_must_not_run({"function": {"name": "file_system",
                                                "arguments": {"operation": op, "path": "a"}}}) is True


@pytest.mark.asyncio
async def test_the_dispatcher_runs_the_alias_as_a_write_with_append():
    """Wiring: the agent's own mutation checks see 'write', not 'append'."""
    import tests.test_4jj_search_yield_steer as H
    from ghost_agent.core.strikes import StrikeLedger
    agent = H._agent()
    seen = []

    async def fsx(**kw):
        seen.append(kw)
        return "SUCCESS: Appended 2 chars to 'a.txt'."
    agent.available_tools = {"file_system": fsx}
    ts = H._ts([("file_system", {"operation": "append", "path": "a.txt", "content": "x\n"})], StrikeLedger(), set())
    await agent._dispatch_and_process_tool_batch(ts)
    assert seen and seen[0]["operation"] == "write" and seen[0]["append"] is True


def test_a_cut_edit_that_replaces_its_anchor_is_not_salvaged_as_an_append():
    """The anchor is REPLACED (new_string does not start with it), even
    though what follows it begins on a new line."""
    import json as _json
    from ghost_agent.core.agent import _salvage_cut_write
    raw = _json.dumps({"operation": "edit", "path": "s.md", "old_string": "OLDX",
                       "new_string": "NEWX\n" + "line\n" * 600})[:-40]
    assert _salvage_cut_write(raw) is None
