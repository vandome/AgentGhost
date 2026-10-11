"""§4JS — a cancellation lost in the stream reader hung the shutdown.

THE LIVE FAILURE (2026-09-22, 10:36). `kill -TERM` for a deploy: the
lifespan shutdown cancelled the biological watchdog and awaited it — for
15 minutes. The watchdog was inside an idle self-play turn that was
streaming an LLM response when the cancel landed, and the cancel vanished:
no aborted-turn record, the sim turn finished at +161 s, a SECOND sim ran,
"idle cycle: ran self-play" was logged AFTER "system shutdown", and the
SIGUSR2 task dump showed the watchdog alive at its 60 s sleep. Python
3.10's `asyncio.wait_for` returns the inner result and DROPS the caller's
cancellation when the inner future completed in the same loop iteration
(bpo-42130, fixed in 3.12); the reader takes every chunk through it, and a
chunk lands every few milliseconds.

Now: `utils/aio.wait_for` (built on `asyncio.wait`, whose cancel path has
no `fut.done()` shortcut) at the chunk reader; an enumeration that no
`asyncio.wait_for` wraps an iterator step anywhere; and the shutdown's
watchdog await is bounded and names the straggler's frame instead of
waiting forever.

Worlds where these pins fail: the reader goes back to `asyncio.wait_for`;
the helper regains a `done()` shortcut or stops cancelling the inner on
timeout; the shutdown await loses its bound or its diagnosis.
"""
import ast
import asyncio
import glob
import logging
import os
import sys

import pytest

from ghost_agent.utils import aio
from ghost_agent.core import llm as llm_mod
from ghost_agent.core.llm import LLMClient
from tests.test_stream_idle_timeout import _FakeResp, _FakeClient


# ── the premise: the 3.10 stdlib drops the cancel; ours does not ──────────

async def _race(wait_for):
    """Complete the inner future and cancel the outer task in the SAME loop
    iteration — the reader's every-few-ms condition, made deterministic."""
    loop = asyncio.get_running_loop()
    fut = loop.create_future()

    async def worker():
        try:
            await wait_for(fut, 5)
        except asyncio.CancelledError:
            return "cancelled"
        return "continued"

    t = asyncio.create_task(worker())
    await asyncio.sleep(0)            # the worker is now parked in wait_for
    fut.set_result(1)
    t.cancel()
    return await t


@pytest.mark.skipif(sys.version_info >= (3, 12), reason="the stdlib race is fixed in 3.12")
async def test_the_stdlib_wait_for_loses_the_cancellation_on_this_runtime():
    assert await _race(asyncio.wait_for) == "continued"


async def test_ours_propagates_the_cancellation():
    assert await _race(aio.wait_for) == "cancelled"


# ── the contract is asyncio.wait_for's ────────────────────────────────────

async def test_result_timeout_and_inner_cancel():
    async def quick():
        return 7
    assert await aio.wait_for(quick(), 1) == 7
    assert await aio.wait_for(quick(), None) == 7

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow():
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise
    with pytest.raises(asyncio.TimeoutError):
        await aio.wait_for(slow(), 0.05)
    assert started.is_set() and cancelled.is_set()   # the inner was cancelled, and had landed


async def test_an_inner_exception_is_re_raised():
    async def boom():
        raise StopAsyncIteration
    with pytest.raises(StopAsyncIteration):
        await aio.wait_for(boom(), 1)


# ── the reader: a cancel at a chunk boundary reaches the consumer ─────────

def _client(resp):
    c = LLMClient.__new__(LLMClient)
    c.http_client = _FakeClient(resp)
    c.coding_clients = None
    return c


class _CancellingResp(_FakeResp):
    """Cancels the CONSUMER task in the same step that yields a chunk."""

    def __init__(self, holder):
        super().__init__([])
        self._holder = holder

    def aiter_lines(self):
        holder = self._holder

        async def _gen():
            yield "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}"
            await asyncio.sleep(0.01)
            holder["task"].cancel()      # lands as this chunk completes
            yield "data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}"
            await asyncio.sleep(0.01)
            yield "data: [DONE]"
        return _gen()


async def test_a_cancel_landing_with_a_chunk_stops_the_stream(monkeypatch):
    monkeypatch.setattr(llm_mod, "_STREAM_FIRST_BYTE_TIMEOUT", 2.0)
    monkeypatch.setattr(llm_mod, "_STREAM_IDLE_TIMEOUT", 2.0)
    holder = {}
    client = _client(_CancellingResp(holder))

    async def consume():
        out = []
        async for b in client._do_stream_chat_completion({"model": "m", "messages": []}):
            out.append(b)
        return out

    holder["task"] = asyncio.create_task(consume())
    with pytest.raises(asyncio.CancelledError):
        await holder["task"]


async def test_control_an_uncancelled_stream_completes(monkeypatch):
    monkeypatch.setattr(llm_mod, "_STREAM_FIRST_BYTE_TIMEOUT", 2.0)
    monkeypatch.setattr(llm_mod, "_STREAM_IDLE_TIMEOUT", 2.0)
    client = _client(_FakeResp([(0.0, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}"),
                                (0.01, "data: [DONE]")]))
    out = []
    async for b in client._do_stream_chat_completion({"model": "m", "messages": []}):
        out.append(b.decode("utf-8") if isinstance(b, (bytes, bytearray)) else b)
    assert "hi" in "".join(out)


# ── the class: no stdlib wait_for around an iterator step, anywhere ──────

def _src_files():
    root = os.path.join(os.path.dirname(llm_mod.__file__), "..")
    return glob.glob(os.path.join(root, "**", "*.py"), recursive=True)


def _is_iter_step(node):
    """`x.__anext__()`, `anext(x)`, `q.get()`, `r.readline()` — an await
    that completes often inside a loop."""
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    if isinstance(f, ast.Attribute) and f.attr in ("__anext__", "get", "readline", "receive"):
        return True
    return isinstance(f, ast.Name) and f.id == "anext"


def test_no_stdlib_wait_for_wraps_an_iterator_step():
    offenders = []
    for path in _src_files():
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "wait_for"
                    and isinstance(n.func.value, ast.Name) and n.func.value.id == "asyncio"
                    and n.args and _is_iter_step(n.args[0])):
                offenders.append(f"{os.path.relpath(path)}:{n.lineno}")
    assert offenders == [], offenders


def test_the_chunk_reader_uses_the_safe_helper():
    tree = ast.parse(open(llm_mod.__file__, encoding="utf-8").read())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "_wait_for_cancel_safe"
             and n.args and _is_iter_step(n.args[0])]
    assert len(calls) == 1
    assert llm_mod._wait_for_cancel_safe is aio.wait_for


# ── the shutdown: bounded, and it names the straggler ────────────────────

async def test_a_watchdog_that_swallows_its_cancel_no_longer_pins_the_shutdown(caplog):
    from ghost_agent import main as main_mod

    swallowed = []

    async def stubborn():
        while True:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                if swallowed:
                    raise                  # the test's own teardown cancel lands
                swallowed.append(1)        # the ONE swallowed cancel of the live failure
    task = asyncio.create_task(stubborn(), name="biological_watchdog")
    await asyncio.sleep(0)
    caplog.set_level(logging.WARNING, logger="GhostAgent")
    stopped = await main_mod._stop_biological_watchdog(task, grace_s=0.2)
    assert stopped is False
    assert not task.done()
    msg = "\n".join(r.getMessage() for r in caplog.records)
    assert "did not stop within 0s of cancel" in msg and "parked at stubborn:" in msg
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_a_cooperative_watchdog_stops_at_once():
    from ghost_agent import main as main_mod

    async def cooperative():
        await asyncio.sleep(3600)
    task = asyncio.create_task(cooperative())
    await asyncio.sleep(0)
    assert await main_mod._stop_biological_watchdog(task, grace_s=5) is True
    assert task.cancelled()
    assert await main_mod._stop_biological_watchdog(None) is True


def test_the_lifespan_calls_the_bounded_stop_not_a_bare_await():
    from ghost_agent import main as main_mod
    tree = ast.parse(open(main_mod.__file__, encoding="utf-8").read())
    awaits = [n for n in ast.walk(tree) if isinstance(n, ast.Await)
              and isinstance(n.value, ast.Name) and n.value.id == "bio"]
    assert awaits == [], [a.lineno for a in awaits]
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "_stop_biological_watchdog"]
    assert len(calls) == 1
    assert main_mod._BIO_SHUTDOWN_GRACE_S >= 5
