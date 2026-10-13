"""§4OE-5b — the download route resolves a BARE basename (as a live-streamed
reply carries it before the record-side repair) to the one file of that name
in the sandbox; ambiguous or absent stays 404; a member's path is unchanged."""
import pytest
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

from ghost_agent.api import routes as R


def _req(role=None, project_id=None):
    hdrs = {"X-Ghost-Requester": role} if role else {}
    return MagicMock(headers=hdrs, query_params={"project_id": project_id} if project_id else {})


def _agent(sandbox, member_files=()):
    return NS(context=NS(sandbox_dir=sandbox, current_project_id=None), _member_generated_files=set(member_files))


def test_unique_sandbox_file_rules(tmp_path):
    (tmp_path / "probe-oe1b").mkdir(); (tmp_path / "probe-oe1b" / "servers.md").write_text("x")
    assert R.unique_sandbox_file(tmp_path, "servers.md") == (tmp_path / "probe-oe1b" / "servers.md").resolve()
    assert R.unique_sandbox_file(tmp_path, "missing.md") is None
    (tmp_path / "other").mkdir(); (tmp_path / "other" / "servers.md").write_text("y")
    assert R.unique_sandbox_file(tmp_path, "servers.md") is None            # ambiguous: never guess
    (tmp_path / ".hidden").mkdir(); (tmp_path / ".hidden" / "secret.md").write_text("z")
    assert R.unique_sandbox_file(tmp_path, "secret.md") is None             # hidden dirs are not deliverables
    assert R.unique_sandbox_file(tmp_path, ".env") is None


@pytest.mark.asyncio
async def test_a_bare_link_from_a_streamed_reply_resolves_at_the_route(tmp_path):
    """Fails where the streamed probe's `/api/download/servers_st.md` answered
    404 while `probe-oe1b/servers_st.md` existed (2026-10-13)."""
    (tmp_path / "probe-oe1b").mkdir(); f = tmp_path / "probe-oe1b" / "servers_st.md"; f.write_text("# report")
    with patch.object(R, "get_agent", return_value=_agent(tmp_path)):
        resp = await R.download_file(_req(), "servers_st.md")
        assert resp.path == str(f.resolve())
        with pytest.raises(HTTPException):                                   # a folder-qualified wrong path is not guessed
            await R.download_file(_req(), "wrong/servers_st.md")
        with pytest.raises(HTTPException):
            await R.download_file(_req(), "nothere.md")
        with pytest.raises(HTTPException):                                   # a member never reaches the owner's files
            await R.download_file(_req(role="member"), "servers_st.md")
    # a member whose generated file set names it but whose file lives only in a
    # subfolder: the sandbox-wide search is the owner's (parity with the project
    # fallback, §4KJ R9) — still 404
    with patch.object(R, "get_agent", return_value=_agent(tmp_path, member_files={"servers_st.md"})):
        with pytest.raises(HTTPException):
            await R.download_file(_req(role="member"), "servers_st.md")


@pytest.mark.asyncio
async def test_symlink_out_is_refused_and_the_project_copy_wins(tmp_path):
    import os
    sandbox = tmp_path / "sb"; sandbox.mkdir()
    outside = tmp_path / "outside.md"; outside.write_text("secret")
    os.symlink(outside, sandbox / "leak.md")
    assert R.unique_sandbox_file(sandbox, "leak.md") is None                 # a symlink out of the sandbox is not served
    (sandbox / "projects" / "p1").mkdir(parents=True); (sandbox / "projects" / "p1" / "r.md").write_text("project copy")
    (sandbox / "elsewhere").mkdir(); (sandbox / "elsewhere" / "r.md").write_text("other copy")
    with patch.object(R, "get_agent", return_value=_agent(sandbox)):
        resp = await R.download_file(_req(project_id="p1"), "r.md")
        assert resp.path == str((sandbox / "projects" / "p1" / "r.md").resolve())   # the project fallback decides first
