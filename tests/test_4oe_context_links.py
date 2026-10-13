"""§4OE-3 / §4OE-5 — a written body leaves the prompt once its write landed;
a composed download link is repaired to the written path; the browser runner
is written inside the lock and only when it changed."""
import json

from ghost_agent.core.context_manager import ContextManager
from ghost_agent.core import link_grounding as LG
from ghost_agent.tools import browser as B


def _write_call(cid, path, body):
    return {"id": cid, "type": "function", "function": {"name": "file_system",
            "arguments": json.dumps({"operation": "write", "path": path, "content": body})}}


def test_a_successful_writes_body_is_compacted_from_history_once_its_result_is_in():
    """Fails where a 70 KB write argument stayed verbatim until the L4 prune:
    write-heavy turns compacted 3× in 11 minutes (§4OD)."""
    big = "x = [\n" + "".join(f"    {i},\n" for i in range(4000)) + "]\n"
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": "build it"},
            {"role": "assistant", "content": "", "tool_calls": [_write_call("c1", "a.py", big), _write_call("c2", "b.py", big)]},
            {"role": "tool", "tool_call_id": "c1", "content": "SUCCESS: Wrote 20000 chars to 'a.py'."},
            {"role": "tool", "tool_call_id": "c2", "content": "REJECTED: 'b.py' already exists and this request has not read it"},
            {"role": "assistant", "content": "next"},
            {"role": "user", "content": "go on"},
            {"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"},
            {"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}]
    cm = ContextManager(max_tokens=100)                      # tiny budget → level ≥ 1 immediately
    out = cm.compress_if_needed(msgs, max_level=3)
    a = next(m for m in out if m.get("role") == "assistant" and m.get("tool_calls"))
    c1 = json.loads(a["tool_calls"][0]["function"]["arguments"])
    c2 = json.loads(a["tool_calls"][1]["function"]["arguments"])
    assert "chars written to the file — compacted" in c1["content"] and len(c1["content"]) < 400
    assert c2["content"] == big                               # the refused write keeps its body: nothing landed
    assert json.loads(msgs[2]["tool_calls"][0]["function"]["arguments"])["content"] == big   # never mutated in place
    small = {"role": "assistant", "content": "", "tool_calls": [_write_call("c3", "s.py", "x = 1\n")]}
    assert cm._compact_written_args(small, {"c3"}) is small    # a small body is left alone
    # review 2: on the inline lane the model's own `<tool_call>` text is what reaches the wire — the
    # live dialect is `<function=…><parameter=…>`; matched by tool + path, never by position (a schema
    # echo ahead of the real call must not be rewritten, the real block must)
    decoy = "<tool_call>\n<function=web_search>\n<parameter=query>decoy</parameter>\n</function>\n</tool_call>"
    real = ("<tool_call>\n<function=file_system>\n<parameter=operation>write</parameter>\n<parameter=path>a.py</parameter>\n"
            "<parameter=content>" + big + "</parameter>\n</function>\n</tool_call>")
    inline = {"role": "assistant", "content": "Writing.\n" + decoy + "\n" + real,
              "tool_calls": [{"id": "s0", "type": "function", "function": {"name": "web_search", "arguments": "{\"query\": \"decoy\"}"}},
                             _write_call("c9", "a.py", big)]}
    out2 = cm._compact_written_args(inline, {"c9", "s0"})
    assert len(out2["content"]) < 1200 and "compacted from this history" in out2["content"]
    assert "<parameter=query>decoy</parameter>" in out2["content"]           # the decoy block is untouched
    assert len(inline["content"]) > 30000                                  # the original message is never mutated
    j = {"role": "assistant", "content": "<tool_call>\n" + json.dumps({"name": "file_system", "arguments": {"operation": "write", "path": "a.py", "content": big}}) + "\n</tool_call>",
         "tool_calls": [_write_call("c9", "a.py", big)]}
    assert len(cm._compact_written_args(j, {"c9"})["content"]) < 1000        # the JSON dialect too
    from ghost_agent.core.agent import _render_assistant_with_tool_calls
    rendered = _render_assistant_with_tool_calls(out2["content"], out2["tool_calls"])
    assert len(str(rendered.get("content") if isinstance(rendered, dict) else rendered)) < 1500   # the WIRE is compact


def test_a_download_link_missing_its_folder_is_repaired_to_the_one_real_file(tmp_path):
    """Fails where the on-disk guard KEPT `/api/download/spec_decoding.md`
    because a file of that name existed under `probe-lh/` — and the link
    answered 404 (§4OD probe). One real file: the link is rewritten to it."""
    from ghost_agent.core.agent import _drop_missing_download_links as guard
    (tmp_path / "probe-lh").mkdir(); (tmp_path / "probe-lh" / "spec_decoding.md").write_text("# r")
    (tmp_path / "one").mkdir(); (tmp_path / "two").mkdir()
    (tmp_path / "one" / "x.md").write_text("1"); (tmp_path / "two" / "x.md").write_text("2")
    out, dropped = guard("See [the report](/api/download/spec_decoding.md) and [x](/api/download/x.md) and [gone](/api/download/nope.md).", tmp_path)
    assert "[the report](/api/download/probe-lh/spec_decoding.md)" in out
    assert "[x](/api/download/x.md)" in out                           # two candidates: the model's link stands
    assert "no such file: nope.md" in out and dropped == ["nope.md"]
    out2, _ = guard("![img](/api/download/probe-lh/spec_decoding.md)", tmp_path)
    assert out2 == "![img](/api/download/probe-lh/spec_decoding.md)"  # a correct link is untouched


def test_the_browser_runner_is_rewritten_only_when_its_text_changed(tmp_path):
    p = tmp_path / ".browser_runner.py"
    assert B._ensure_runner_file(p, "print(1)\n") is True and p.read_text() == "print(1)\n"
    assert B._ensure_runner_file(p, "print(1)\n") is False     # unchanged: no write, no rename
    assert B._ensure_runner_file(p, "print(2)\n") is True and p.read_text() == "print(2)\n"
    assert not list(tmp_path.glob("*.tmp"))


async def test_the_live_reply_path_repairs_a_composed_download_link(monkeypatch, tmp_path):
    """Wiring pin: the repair runs on the final reply of a turn that wrote
    the file — the model composed `/api/download/report.md` for
    `probe/report.md`."""
    from unittest.mock import AsyncMock
    from ghost_agent.core.agent import GhostAgent
    from tests.helpers import FakeBgTasks, make_context
    from tests.test_4ji_request_echo_narration import _resp, _tc
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1"); monkeypatch.setenv("GHOST_CRITIC_REPAIR_BUDGET", "0"); monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    ctx = make_context(); ctx.sandbox_dir = tmp_path; agent = GhostAgent(ctx)
    (tmp_path / "probe").mkdir(); (tmp_path / "probe" / "report.md").write_text("# r\n")
    agent.available_tools = {"file_system": AsyncMock(return_value="SUCCESS: Wrote 40 chars to 'probe/report.md'. Script-side path (from sandbox cwd): 'probe/report.md'.")}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("writing", [_tc("w1", "file_system", {"operation": "write", "path": "probe/report.md", "content": "# r\n"})]),
        _resp("Done — [download it here](/api/download/report.md)."),
    ])
    out, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "write probe/report.md"}]}, FakeBgTasks())
    assert "(/api/download/probe/report.md)" in out and "(/api/download/report.md)" not in out


async def test_the_runner_file_is_current_at_exec_time_and_not_rewritten_when_unchanged(monkeypatch, tmp_path):
    """Wiring pin for §4OE-5: the runner the exec sees is the current script,
    written under the profile lock; a second call does not rewrite it."""
    from unittest.mock import AsyncMock, MagicMock
    seen = []

    def _exec(cmd, timeout=None, **kw):                          # the manager's execute is SYNC here
        p = tmp_path / ".browser_runner.py"
        seen.append((p.exists(), p.read_text(encoding="utf-8") == B._runner_script() if p.exists() else None, p.stat().st_mtime_ns if p.exists() else None))
        return ('[BROWSER_OK] {"url": "https://a.example/", "title": "A", "text": "hello", "status": 200}', 0)
    sm = MagicMock(); sm.execute = MagicMock(side_effect=_exec)
    monkeypatch.setattr(B, "_egress_refusal", lambda *_a, **_k: "", raising=False)
    for _ in range(2):
        await B.tool_browser(operation="navigate", url="https://a.example/", sandbox_dir=tmp_path, sandbox_manager=sm,
                             tor_proxy="socks5://127.0.0.1:9050")
    assert len(seen) == 2 and all(s[0] and s[1] for s in seen), seen
    assert seen[0][2] == seen[1][2]                             # unchanged script: no second write
