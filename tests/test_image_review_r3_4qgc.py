"""§4QG-c — the r3 fresh-eye review of the §4QG-b fixes (2026-10-21).

  M1  the image permit wait (up to 690 s behind another render) was booked as
      render time, inflating every later deadline estimate → the client
      reports the node's own time and the tool books that
  M2  the permit wait was bounded by the whole remaining time, not the time
      before the report reserve → it uses the same budget as the ladder
  m1  the background re-preflight shared _RENDER unlocked: a disconnected
      render's leftover `cancelled` flag failed every attempt → it runs under
      the GPU lock, freshly armed
  m2  an "error 12" retry after a long first run outlasted the client timeout
      → retried only when the first run died early
  m3  verify() refused a PNG without IEND that load() decodes → not a 400
      before the lock any more
  m4  warm-up 503s ("preflight retrying") booked breaker faults → not faults
"""
import asyncio
from io import BytesIO

import pytest
from fastapi.testclient import TestClient

from tests.test_image_node_client_review_fixes import (H, _load_server, _node_client, _ready,
                                                       _status_error)


# ── M1 ────────────────────────────────────────────────────────────────
def test_the_client_reports_the_node_time_without_the_permit_wait(monkeypatch):
    from ghost_agent.core.llm import LLMClient
    from unittest.mock import MagicMock
    import ghost_agent.core.llm as L
    client = LLMClient("http://upstream:8080", image_gen_nodes=[{"url": "http://node1:8000", "model": "q"}])
    client.circuit_breaker = MagicMock()
    client.circuit_breaker.is_available.return_value = True
    clock = {"t": 1000.0}
    monkeypatch.setattr(L.time, "monotonic", lambda: clock["t"])

    class _Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"data": [{"b64_json": "x"}], "seed": 1}

    async def post(*a, **k):
        clock["t"] += 120.0                       # the node renders for 120 s
        return _Resp()
    client.image_gen_clients[0]["client"].post = post
    out = asyncio.run(client.generate_image({"prompt": "x"}))
    assert out["_node_seconds"] == pytest.approx(120.0)


def test_the_tool_books_the_node_time_not_the_wall(monkeypatch):
    import ghost_agent.tools.image_gen as IG
    clock = {"t": 0.0}
    monkeypatch.setattr(IG.time, "monotonic", lambda: clock["t"])

    class _Llm:
        async def generate_image(self, payload):
            clock["t"] += 450.0                   # 300 s queued for the permit + 150 s render
            return {"data": [], "_node_seconds": 150.0}
    IG.RECENT_RENDER_SECONDS["edit"].clear()
    resp = asyncio.run(IG._render_leg(_Llm(), {}, "edit"))
    assert list(IG.RECENT_RENDER_SECONDS["edit"]) == [150.0]
    assert "_node_seconds" not in resp            # private: never reaches the payload readers
    IG.RECENT_RENDER_SECONDS["edit"].clear()
    asyncio.run(IG._render_leg(type("X", (), {"generate_image": staticmethod(
        lambda p: asyncio.sleep(0, result={"data": []}))})(), {}, "edit"))
    assert len(IG.RECENT_RENDER_SECONDS["edit"]) == 1   # an old client: the wall, as before
    IG.RECENT_RENDER_SECONDS["edit"].clear()


# ── M2 ────────────────────────────────────────────────────────────────
def test_the_permit_wait_stops_before_the_report_reserve(monkeypatch):
    import ghost_agent.core.llm as L
    import ghost_agent.core.agent as A
    import ghost_agent.utils.logging as G
    monkeypatch.delenv("GHOST_IMAGE_SLOT_WAIT_S", raising=False)
    monkeypatch.setattr(G, "request_remaining_s", lambda rid: 700.0)
    monkeypatch.setattr(G, "request_deadline_s", lambda rid: 1800.0)
    monkeypatch.setattr(A, "effective_report_floor", lambda d, rid=None: 300.0)
    import ghost_agent.tools.image_gen as IG
    monkeypatch.setitem(IG.RECENT_RENDER_SECONDS, "generate", IG.collections.deque(maxlen=3))
    # 700 left − 300 reserve − this render's default (§4QG-d; 115 since §4QR) — not 690, not 700
    assert L.image_node_slot_wait_s() == pytest.approx(400.0 - IG.DEFAULT_RENDER_SECONDS["generate"])


# ── m1 ────────────────────────────────────────────────────────────────
def test_a_recheck_runs_under_the_lock_and_clears_a_stale_cancel(monkeypatch, tmp_path):
    mod = _load_server(monkeypatch)
    _ready(mod, tmp_path)
    mod.LOAD_RETRY_DELAY_S = 0.0
    mod._RENDER["cancelled"] = True               # a disconnected render's leftover
    seen = []

    def load():
        seen.append((mod._gpu_lock.locked(), mod._RENDER["cancelled"]))
    monkeypatch.setattr(mod, "_load_model_blocking", load)
    asyncio.run(mod._background_load_attempts())
    assert seen == [(True, False)] and mod._ready is True


# ── m2 ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("first_s,code,calls", [(5.0, 200, 2), (300.0, 507, 1)])
def test_nvmap_is_retried_only_after_an_early_failure(monkeypatch, tmp_path, first_s, code, calls):
    mod = _load_server(monkeypatch)
    _ready(mod, tmp_path)
    clock = {"t": 0.0}
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock["t"])
    seen = []

    def gen(*a, **k):
        seen.append(1)
        if len(seen) == 1:
            clock["t"] += first_s
            raise mod.RenderNvMapTransient("sd-cli exit 1: NvMapMemAllocInternalTagged: error 12")
        return b"ok"
    monkeypatch.setattr(mod, "_generate_png", gen)
    r = TestClient(mod.app).post("/generate", json={"prompt": "x"}, headers=H)
    assert r.status_code == code and len(seen) == calls


# ── m3 ────────────────────────────────────────────────────────────────
def test_a_png_without_iend_passes_the_cheap_check_and_garbage_does_not(monkeypatch):
    from PIL import Image
    mod = _load_server(monkeypatch)
    buf = BytesIO()
    Image.new("RGB", (32, 32), (200, 10, 10)).save(buf, "PNG")
    data = buf.getvalue()
    no_iend = data[:data.rindex(b"IEND") - 4]
    with Image.open(BytesIO(no_iend)) as im:
        im.load()                                 # Pillow decodes it
    assert mod._structure_ok(data) and mod._structure_ok(no_iend)
    assert not mod._structure_ok(b"\x89PNG\r\n\x1a\nnot a png at all")


# ── m4 ────────────────────────────────────────────────────────────────
def test_a_warmup_503_is_retried_but_not_a_breaker_fault(monkeypatch):
    client, posts, sleeps = _node_client(monkeypatch, _status_error(503, "preflight retrying: x"))
    with pytest.raises(Exception):
        asyncio.run(client.generate_image({"prompt": "x"}))
    assert posts["n"] == 3 and sleeps == [10.0, 35.0]
    assert client.circuit_breaker.record_failure.call_count == 0


# ── §4QG-d (r4 review): the wait leaves room for the render itself ─────
@pytest.mark.parametrize("payload,kind", [({"prompt": "x"}, "generate"),
                                          ({"prompt": "x", "reference_images": ["AA"]}, "edit")])
def test_the_permit_wait_leaves_room_for_this_render(monkeypatch, payload, kind):
    import ghost_agent.core.llm as L
    import ghost_agent.tools.image_gen as IG
    want = 400.0 - IG.DEFAULT_RENDER_SECONDS[kind]          # §4QR: the default, not a literal
    monkeypatch.delenv("GHOST_IMAGE_SLOT_WAIT_S", raising=False)
    monkeypatch.setattr(IG, "_request_time_left", lambda: 400.0)
    for k in ("edit", "generate"):
        monkeypatch.setitem(IG.RECENT_RENDER_SECONDS, k, IG.collections.deque(maxlen=3))
    assert L.image_node_slot_wait_s(payload) == pytest.approx(want)
