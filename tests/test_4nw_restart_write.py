"""§4NW-b (2026-10-12, req c7f06c37 — "many things went wrong"): building a
72 KB Minecraft page in parts, the model re-emitted the file from <!DOCTYPE
twice. The plain restart was refused (138 s of output thrown away); the cut
restart was APPENDED onto the first copy — two documents in one file, a
syntax error at line 490, a delete and a fourth start before the client
disconnected at 851 s. A write that opens the way the file opens is a
restart: it replaces the parts. Every test names the world it fails in."""
import asyncio
from pathlib import Path

import pytest

from ghost_agent.tools.file_system import restarts_file, tool_file_system
from ghost_agent.utils.logging import request_id_context

HEAD = '<!DOCTYPE html>\n<html lang="en">\n<head>\n<title>Minecraft Voxel Game</title>\n'


def _run(c):
    return asyncio.run(c)


@pytest.fixture(autouse=True)
def _rid():
    tok = request_id_context.set("req-c7f06c37")
    yield
    request_id_context.reset(tok)


@pytest.mark.parametrize("old,new,expect", [
    (HEAD, HEAD.replace("Voxel Game", "Clone"), True),               # same doc start, other title
    ('<!doctype html><html>', '<html lang="el">', True),
    ("#!/usr/bin/env python3\nimport os\n", "#!/bin/sh\necho", True),
    ('<?xml version="1.0"?><a/>', '<?xml version="1.0"?><b/>', True),
    (HEAD, "  let bestDist=Infinity;\n", False),                      # a continuation
    (HEAD, "</script>\n</body>\n</html>\n", False),
    # review: a part that repeats a header is NOT a restart
    ("id,name,value\na0,x,1\n", "id,name,value\na100,y,2\n", False),
    ("==========\nReport\n", "==========\nAppendix\n", False),
    ("import os\nimport sys\n", "import os\nimport json\n", False),
    ("<!-- generated -->\n<p>a</p>", "<!-- generated -->\n<p>b</p>", False),
    ("#!/bin/sh\necho", HEAD, False),                                 # different kinds
    ("", HEAD, False),
])
def test_what_counts_as_starting_the_file_over(old, new, expect):
    assert restarts_file(old, new) is expect


def _write(tmp, content, **kw):
    return _run(tool_file_system(operation="write", sandbox_dir=tmp, path="index.html", content=content, **kw))


def test_a_restart_replaces_the_parts_instead_of_being_refused(tmp_path):
    """Fails where the plain restart was refused and 22 KB of output was
    thrown away (call 2 of c7f06c37)."""
    _write(tmp_path, HEAD + "<script>\nconst A=1;\n", more_parts=True)
    r = _write(tmp_path, HEAD + "<script>\nconst B=2;\n", more_parts=True)
    assert str(r).startswith("SUCCESS") and "REPLACED the parts" in str(r)
    assert (tmp_path / "index.html").read_text() == HEAD + "<script>\nconst B=2;\n"


def test_a_cut_restart_is_saved_as_the_new_version_not_appended(tmp_path):
    """Fails where the cut restart (call 3) was appended onto the first copy:
    two <!DOCTYPE in one file."""
    _write(tmp_path, HEAD + "<script>\nconst A=1;\n", more_parts=True)
    new = HEAD.replace("Voxel Game", "Voxel") + "<script>\nconst C=3;\n"
    r = _write(tmp_path, new, more_parts=True, cut_off_saved=46549)
    text = (tmp_path / "index.html").read_text()
    assert text.count("<!DOCTYPE") == 1 and text == new and str(r).startswith("SUCCESS")
    # …and the continuation after it lands on the new version
    _write(tmp_path, "</script>\n</html>\n", append=True)
    assert (tmp_path / "index.html").read_text() == new + "</script>\n</html>\n"


def test_a_continuation_without_append_is_still_refused(tmp_path):
    """The §4NU r4 guard stands: "part 2" sent as a plain write must not
    replace part 1."""
    _write(tmp_path, HEAD + "<script>\n", more_parts=True)
    r = _write(tmp_path, "const B=2;\n</script>\n", more_parts=True)
    assert getattr(r, "reason_code", None) == "file_in_progress"
    assert (tmp_path / "index.html").read_text() == HEAD + "<script>\n"


def test_a_cut_continuation_is_still_appended(tmp_path):
    _write(tmp_path, HEAD + "<script>\n", more_parts=True)
    _write(tmp_path, "const B=2;\n", more_parts=True, cut_off_saved=30000)
    assert (tmp_path / "index.html").read_text() == HEAD + "<script>\nconst B=2;\n"


def test_a_file_from_another_request_is_never_restarted_over(tmp_path):
    """A file NOT being built in this request keeps the §4KB rules (a cut
    replacement of an existing file writes nothing)."""
    (tmp_path / "index.html").write_text(HEAD + "<body>owner's page</body>\n")
    r = _write(tmp_path, HEAD + "<script>\n", more_parts=True, cut_off_saved=30000)
    assert getattr(r, "reason_code", None) == "tool_call_cut_off"
    assert "owner's page" in (tmp_path / "index.html").read_text()


def test_a_csv_part_that_repeats_its_header_never_replaces_the_rows(tmp_path):
    """Fails where a cut part 2 that repeated the header row replaced part
    1's rows (review)."""
    p = lambda c, **k: _run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="d.csv", content=c, **k))
    p("id,name\n" + "".join(f"a{i},x\n" for i in range(100)), more_parts=True)
    p("id,name\n" + "".join(f"b{i},y\n" for i in range(100)), more_parts=True, cut_off_saved=30000)
    text = (tmp_path / "d.csv").read_text()
    assert "a0,x" in text and "b0,y" in text


def test_an_append_that_opens_a_second_document_is_refused(tmp_path):
    """Fails where the model put append=true BEFORE content and re-sent the
    whole page from <!DOCTYPE — two documents in one file (review)."""
    _write(tmp_path, HEAD + "<script>\n", more_parts=True)
    r = _write(tmp_path, HEAD + "<script>\nconst Z=1;\n", append=True, more_parts=True)
    assert getattr(r, "reason_code", None) == "file_in_progress"
    assert (tmp_path / "index.html").read_text().count("<!DOCTYPE") == 1


def test_an_append_that_opens_a_second_document_on_a_finished_file_is_refused(tmp_path):
    """Fails where, after the last part (no more_parts), an append starting a
    new <!DOCTYPE was accepted — two documents (verify, residual)."""
    _write(tmp_path, HEAD + "<body></body>\n</html>\n")             # finished
    r = _write(tmp_path, HEAD + "<body>again</body></html>\n", append=True)
    assert getattr(r, "reason_code", None) == "second_document"
    assert (tmp_path / "index.html").read_text().count("<!DOCTYPE") == 1
    ok = _write(tmp_path, "<!-- footer -->\n", append=True)        # an ordinary append still works
    assert str(ok).startswith("SUCCESS")


def test_a_non_markup_file_that_starts_like_html_takes_any_append(tmp_path):
    """Review: a notes file whose text starts with <!DOCTYPE is not a web
    page — an appended snippet that starts with <html is fine."""
    (tmp_path / "notes.md").write_text(HEAD + "notes\n")
    r = _run(tool_file_system(operation="write", sandbox_dir=tmp_path, path="notes.md",
                              content='<html lang="en"> snippet\n', append=True))
    assert str(r).startswith("SUCCESS")
