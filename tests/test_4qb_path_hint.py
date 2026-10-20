"""§4QB: the hint for a wrong path prefix in the 'not found' message.

Live 2026-10-19 (project 448d5afa86f0, analysis 5.1–5.2): the model read
'448d5afa86f0/results/harness-scores.md' and listed '448d5afa86f0/results'.
The hint searched only the 20-file listing, which filled with research/ before
results/, so neither call got it and the turn was forced to its report at 41 s.

The rule after three review rounds (each found a CONFIDENT hint pointing at a
different same-named file): confident only when the dropped part is exactly
this project's id; every other match is hedged ("if that is the one you mean"),
and a bare root FILE name counts only when the name is unique.
"""
import asyncio
import os

import pytest

from ghost_agent.tools import file_system as fs
from ghost_agent.tools.file_system import _missing_file_message

CONFIDENT = "ALREADY inside"
ID = "0123456789ab"
HEDGED = "if that is the one you mean"


def _big_project(tmp_path):
    """A project workspace whose first 20 walked files never reach results/."""
    proj = tmp_path / "projects" / "448d5afa86f0"
    for d in ("analysis", "design", "research"):
        (proj / d).mkdir(parents=True)
        for i in range(12):
            (proj / d / f"note-{i:02d}.md").write_text("x\n")
    for i in range(5):
        (proj / f"top-{i}.md").write_text("x\n")
    (proj / "results").mkdir()
    (proj / "results" / "harness-scores.md").write_text("scores\n")
    return proj


def _proj(tmp_path, name="abc"):
    p = tmp_path / "projects" / name
    p.mkdir(parents=True)
    return p


def _listed(msg):
    return msg.split("Files that DO exist here: ", 1)[1].split(".", 1)[0]


# ---- the live case: the project's own id as a directory — confident

def test_live_case_file_gets_the_confident_hint(tmp_path):
    proj = _big_project(tmp_path)
    msg = _missing_file_message("448d5afa86f0/results/harness-scores.md", proj)
    assert "results/harness-scores.md" not in _listed(msg)   # the listing alone misses it
    assert CONFIDENT in msg
    assert "'448d5afa86f0/' (the project's id) is not a directory here" in msg
    assert "Use the relative path 'results/harness-scores.md'" in msg


def test_live_case_directory_gets_the_confident_hint(tmp_path):
    proj = _big_project(tmp_path)
    msg = _missing_file_message("448d5afa86f0/results", proj)
    assert CONFIDENT in msg and "relative path 'results'" in msg


def test_the_id_prefix_is_confident_even_with_a_duplicate_name(tmp_path):
    proj = _big_project(tmp_path)
    (proj / "notes.md").write_text("")
    (proj / "design" / "notes.md").write_text("")
    msg = _missing_file_message("448d5afa86f0/notes.md", proj)
    assert CONFIDENT in msg and "relative path 'notes.md'" in msg


def test_backslashes_and_dot_parts_still_match_the_id(tmp_path):
    proj = _big_project(tmp_path)
    msg = _missing_file_message("448d5afa86f0\\./results\\harness-scores.md", proj)
    assert CONFIDENT in msg and "relative path 'results/harness-scores.md'" in msg


def test_the_longest_existing_tail_wins(tmp_path):
    proj = _big_project(tmp_path)
    (proj / "harness-scores.md").write_text("root copy\n")
    msg = _missing_file_message("448d5afa86f0/results/harness-scores.md", proj)
    assert "relative path 'results/harness-scores.md'" in msg


# ---- r3: the id must be ALL that was dropped

def test_the_id_plus_a_real_directory_is_not_confident(tmp_path):
    """r3 MAJOR: <id>/sub/__init__.py → the root __init__.py, said confidently."""
    proj = _proj(tmp_path, ID)
    (proj / "sub").mkdir()
    (proj / "sub" / "a.py").write_text("")
    (proj / "__init__.py").write_text("")
    msg = _missing_file_message(f"{ID}/sub/__init__.py", proj)
    assert CONFIDENT not in msg
    assert "relative path '__init__.py'" in msg and HEDGED in msg


def test_an_id_that_is_also_a_subdirectory_is_not_confident(tmp_path):
    proj = _proj(tmp_path, ID)
    (proj / ID).mkdir()
    (proj / ID / "readme.md").write_text("")
    (proj / "x").mkdir()
    (proj / "x" / "notes.md").write_text("")
    msg = _missing_file_message(f"{ID}/{ID}/x/notes.md", proj)
    assert CONFIDENT not in msg and "relative path 'x/notes.md'" in msg and HEDGED in msg


def test_the_id_only_counts_inside_a_project_workspace(tmp_path):
    root = tmp_path / "sandbox"
    (root / "results").mkdir(parents=True)
    (root / "results" / "r.md").write_text("")
    msg = _missing_file_message("sandbox/results/r.md", root)
    assert CONFIDENT not in msg and HEDGED in msg


# ---- every other tail is hedged (r3 MAJOR 2)

def test_a_deeper_tail_under_a_real_directory_is_hedged(tmp_path):
    proj = _proj(tmp_path)
    (proj / "pkg_b").mkdir()
    (proj / "pkg_b" / "a.py").write_text("")
    (proj / "tests").mkdir()
    (proj / "tests" / "__init__.py").write_text("")
    msg = _missing_file_message("pkg_b/tests/__init__.py", proj)
    assert CONFIDENT not in msg and "not where it is" not in msg
    assert "NOTE: 'tests/__init__.py' exists in this workspace — " + HEDGED in msg


def test_a_title_prefix_is_hedged(tmp_path):
    proj = _proj(tmp_path)
    (proj / "train.py").write_text("")
    msg = _missing_file_message("Mini AI v2/train.py", proj)
    assert CONFIDENT not in msg and HEDGED in msg and "relative path 'train.py'" in msg


def test_a_directory_behind_a_title_is_hedged(tmp_path):
    """r3 minor: a bare DIRECTORY tail must still get a hint."""
    proj = _proj(tmp_path)
    (proj / "src").mkdir()
    (proj / "src" / "m.py").write_text("")
    msg = _missing_file_message("Mini AI/src", proj)
    assert HEDGED in msg and "relative path 'src'" in msg


# ---- a bare root file is a NAME match (r2 MAJOR)

def test_a_bare_root_name_found_twice_gets_no_hint(tmp_path):
    proj = _proj(tmp_path)
    (proj / "pkg_a").mkdir()
    (proj / "__init__.py").write_text("")
    (proj / "pkg_a" / "__init__.py").write_text("")
    msg = _missing_file_message("pkg_b/__init__.py", proj)
    assert "NOTE:" not in msg


def test_a_bare_root_name_under_a_real_directory_is_hedged(tmp_path):
    proj = _proj(tmp_path)
    (proj / "src").mkdir()
    (proj / "config.py").write_text("")
    msg = _missing_file_message("src/config.py", proj)
    assert CONFIDENT not in msg and HEDGED in msg and "relative path 'config.py'" in msg


def test_a_name_beyond_the_listing_is_found_and_hedged(tmp_path):
    proj = _big_project(tmp_path)
    msg = _missing_file_message("old/stuff/harness-scores.md", proj)
    assert "NOTE: 'results/harness-scores.md' exists in this workspace — " + HEDGED in msg
    assert CONFIDENT not in msg


def test_a_name_found_twice_names_no_file(tmp_path):
    proj = _big_project(tmp_path)
    (proj / "results" / "__init__.py").write_text("")
    (proj / "design" / "__init__.py").write_text("")
    assert fs._basename_match("__init__.py", proj) is None
    assert "NOTE:" not in _missing_file_message("src/utils/__init__.py", proj)


def test_no_hint_when_nothing_matches(tmp_path):
    proj = _big_project(tmp_path)
    assert "NOTE:" not in _missing_file_message("448d5afa86f0/results/nothing.md", proj)


# ---- safety: never outside the sandbox, never hidden, never raises

def test_parent_segments_never_match(tmp_path):
    proj = _big_project(tmp_path)
    (tmp_path / "secret.txt").write_text("x\n")
    assert fs._suffix_match("x/../../../secret.txt", proj) is None
    assert fs._suffix_match("../secret.txt", proj) is None
    assert fs._suffix_match("a/b/..", proj) is None


def test_a_symlink_out_of_the_sandbox_never_matches(tmp_path):
    proj = _big_project(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "f.txt").write_text("x\n")
    os.symlink(outside, proj / "link")
    assert fs._suffix_match("448d5afa86f0/link/f.txt", proj) is None


def test_a_hidden_tail_or_name_never_matches(tmp_path):
    proj = _big_project(tmp_path)
    (proj / ".env").write_text("SECRET=1\n")
    assert fs._suffix_match("448d5afa86f0/.env", proj) is None
    assert fs._basename_match(".env", proj) is None
    assert "NOTE:" not in _missing_file_message("448d5afa86f0/.env", proj)


def test_a_symlinked_file_is_never_suggested(tmp_path):
    proj = _proj(tmp_path)
    (proj / "d").mkdir()
    outside = tmp_path / "secret.csv"
    outside.write_text("x")
    os.symlink(outside, proj / "d" / "data.csv")
    assert fs._basename_match("data.csv", proj) is None


def test_a_symlink_loop_does_not_hide_a_shorter_tail(tmp_path):
    proj = _proj(tmp_path)
    (proj / "sub").mkdir()
    (proj / "sub" / "foo.py").write_text("")
    os.symlink(proj / "a", proj / "a")
    assert fs._suffix_match("x/a/sub/foo.py", proj) == "sub/foo.py"


def test_an_unreadable_tail_never_raises(tmp_path, monkeypatch):
    proj = _proj(tmp_path)
    (proj / "train.py").write_text("")
    real = fs.Path.is_dir

    def boom(self):
        if self.name == "train.py":
            raise OSError(63, "File name too long")
        return real(self)
    monkeypatch.setattr(fs.Path, "is_dir", boom)
    msg = _missing_file_message("Title/train.py", proj)
    assert "Error:" in msg


# ---- the walk is bounded

def test_the_name_walk_is_bounded(tmp_path, monkeypatch):
    proj = _big_project(tmp_path)
    monkeypatch.setattr(fs, "_HINT_SCAN_MAX", 10)
    assert fs._basename_match("harness-scores.md", proj) is None
    monkeypatch.setattr(fs, "_HINT_SCAN_MAX", 5000)
    assert fs._basename_match("harness-scores.md", proj) == "results/harness-scores.md"


def test_empty_directories_count_toward_the_bound(tmp_path, monkeypatch):
    proj = _proj(tmp_path)
    for i in range(30):
        (proj / f"d{i:02d}").mkdir()
    (proj / "zz").mkdir()
    (proj / "zz" / "target.md").write_text("x\n")
    monkeypatch.setattr(fs, "_HINT_SCAN_MAX", 20)
    assert fs._basename_match("target.md", proj) is None
    monkeypatch.setattr(fs, "_HINT_SCAN_MAX", 5000)
    assert fs._basename_match("target.md", proj) == "zz/target.md"


def test_the_name_walk_skips_hidden_and_vendor_dirs(tmp_path):
    proj = _proj(tmp_path)
    for d in (".git", "node_modules", "__pycache__", "venv"):
        (proj / d).mkdir()
        (proj / d / "target.md").write_text("x\n")
    assert fs._basename_match("target.md", proj) is None
    (proj / "docs").mkdir()
    (proj / "docs" / "target.md").write_text("x\n")
    assert fs._basename_match("target.md", proj) == "docs/target.md"


# ---- end to end: the model reads the tool result, not the helper

@pytest.mark.parametrize("call", ["read", "list"])
def test_end_to_end_through_the_tool(tmp_path, call):
    proj = _big_project(tmp_path)
    if call == "read":
        out = asyncio.run(fs.tool_read_file("448d5afa86f0/results/harness-scores.md", proj))
    else:
        out = asyncio.run(fs.tool_list_files(proj, path="448d5afa86f0/results"))
    assert CONFIDENT in str(out)


# ---- r4 review

def test_a_real_id_directory_makes_the_hint_hedged(tmp_path):
    """r4 MAJOR: a write with the id prefix made a REAL <id>/ directory, so
    "'<id>/' is not a directory here" would be false."""
    proj = _big_project(tmp_path)
    (proj / "448d5afa86f0" / "results").mkdir(parents=True)
    (proj / "448d5afa86f0" / "results" / "a.md").write_text("")
    (proj / "results" / "b.md").write_text("")
    msg = _missing_file_message("448d5afa86f0/results/b.md", proj)
    assert CONFIDENT not in msg and "is not a directory" not in msg
    assert "NOTE: 'results/b.md' exists in this workspace — " + HEDGED in msg


def test_with_a_hint_the_next_step_points_at_the_note(tmp_path):
    proj = _big_project(tmp_path)
    msg = _missing_file_message("448d5afa86f0/results/harness-scores.md", proj)
    assert "To proceed: use the relative path 'results/harness-scores.md' (see the NOTE), or CREATE it" in msg
    plain = _missing_file_message("448d5afa86f0/results/nothing.md", proj)
    assert "To proceed: CREATE it" in plain


def test_the_hint_never_raises_even_on_a_bad_sandbox(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise TypeError("bad sandbox")
    monkeypatch.setattr(fs, "_suffix_match", boom)
    proj = _big_project(tmp_path)
    msg = _missing_file_message("448d5afa86f0/results/harness-scores.md", proj)
    assert msg.startswith("Error:") and "NOTE:" not in msg


def test_only_the_last_tails_are_tried(tmp_path, monkeypatch):
    proj = _big_project(tmp_path)
    calls = []
    real = fs.Path.resolve

    def counting(self, *a, **k):
        calls.append(str(self))
        return real(self, *a, **k)
    monkeypatch.setattr(fs.Path, "resolve", counting)
    deep = "/".join(["d"] * 200) + "/nothing.md"
    assert fs._suffix_match(deep, proj) is None
    assert len(calls) <= fs._MAX_TAIL_PARTS + 1      # + the sandbox root itself



# ---- r5 review: a suggested path must open the entry that matched

def _suggested(msg):
    import re
    m = re.search(r"NOTE: '([^']+)' (?:DOES exist|exists in this workspace)", msg)
    return m.group(1) if m else None


def _assert_round_trip(msg, proj):
    s = _suggested(msg)
    if s is not None:
        assert fs._get_safe_path(proj, s).resolve() == (proj / s).resolve(), msg
        assert (proj / s).exists()


def test_a_tail_the_resolver_rewrites_is_never_offered(tmp_path):
    """r5 MAJOR: 'workspace/notes.md' is opened as the ROOT notes.md."""
    proj = _big_project(tmp_path)
    (proj / "workspace").mkdir()
    (proj / "workspace" / "notes.md").write_text("inner")
    (proj / "notes.md").write_text("A DIFFERENT FILE")
    msg = _missing_file_message("448d5afa86f0/workspace/notes.md", proj)
    assert "'workspace/notes.md'" not in msg.split("Files that DO exist")[0]
    _assert_round_trip(msg, proj)


def test_a_projects_tail_is_never_offered(tmp_path):
    proj = _big_project(tmp_path)
    (proj / "projects" / "foo").mkdir(parents=True)
    (proj / "projects" / "foo" / "x.md").write_text("inner")
    (proj / "x.md").write_text("OTHER x")
    msg = _missing_file_message("448d5afa86f0/projects/foo/x.md", proj)
    assert "'projects/foo/x.md'" not in msg.split("Files that DO exist")[0]
    _assert_round_trip(msg, proj)


def test_a_projects_parent_without_a_hex_id_is_not_a_project(tmp_path):
    root = tmp_path / "projects" / "sandbox"
    (root / "results").mkdir(parents=True)
    (root / "results" / "a.md").write_text("")
    msg = _missing_file_message("sandbox/results/a.md", root)
    assert CONFIDENT not in msg and "(the project's id)" not in msg


@pytest.mark.parametrize("path", [
    "448d5afa86f0/results/harness-scores.md", "448d5afa86f0/results",
    "old/stuff/harness-scores.md", "Title/top-0.md", "x/analysis/note-03.md",
    "448d5afa86f0/workspace/notes.md", "448d5afa86f0/projects/foo/x.md",
])
def test_every_hint_round_trips(tmp_path, path):
    proj = _big_project(tmp_path)
    (proj / "workspace").mkdir()
    (proj / "workspace" / "notes.md").write_text("inner")
    (proj / "projects" / "foo").mkdir(parents=True)
    (proj / "projects" / "foo" / "x.md").write_text("inner")
    _assert_round_trip(_missing_file_message(path, proj), proj)


def test_end_to_end_through_edit(tmp_path):
    proj = _big_project(tmp_path)
    out = asyncio.run(fs.tool_edit_text("448d5afa86f0/results/harness-scores.md", "scores", "s2", proj))
    assert CONFIDENT in str(out) and "relative path 'results/harness-scores.md'" in str(out)



# ---- r6 review: "found twice" folds case and form, and counts symlinked copies

def test_a_case_folded_twin_counts_as_a_second_copy(tmp_path):
    proj = _proj(tmp_path)
    (proj / "README.md").write_text("root")
    (proj / "docs").mkdir()
    (proj / "docs" / "readme.md").write_text("docs")
    assert fs._basename_match("readme.md", proj) is None
    assert "NOTE:" not in _missing_file_message("x/readme.md", proj)


def test_a_unicode_form_twin_counts_as_a_second_copy(tmp_path):
    import unicodedata
    proj = _proj(tmp_path)
    (proj / "a").mkdir()
    (proj / "b").mkdir()
    (proj / "a" / unicodedata.normalize("NFC", "café.md")).write_text("")
    (proj / "b" / unicodedata.normalize("NFD", "Café.md")).write_text("")
    assert fs._basename_match(unicodedata.normalize("NFC", "café.md"), proj) is None


def test_a_symlinked_copy_counts_but_is_never_suggested(tmp_path):
    proj = _proj(tmp_path)
    (proj / "archive").mkdir()
    (proj / "archive" / "old.md").write_text("")
    os.symlink(proj / "archive" / "old.md", proj / "notes.md")
    (proj / "docs").mkdir()
    (proj / "docs" / "notes.md").write_text("")
    assert fs._basename_match("notes.md", proj) is None
    assert "NOTE:" not in _missing_file_message("x/notes.md", proj)


def test_a_differently_cased_single_file_is_not_suggested_under_another_name(tmp_path):
    proj = _proj(tmp_path)
    (proj / "docs").mkdir()
    (proj / "docs" / "README.md").write_text("")
    # one copy, but its real name differs: the walk offers only an exact name
    assert fs._basename_match("readme.md", proj) is None
