"""GHOST_POINTER_HYDRATION — ghost-ingest's project pointers still hydrate.

§4NM retired the vector tier from prompts. A deployment that imports past
conversations (ghostai's ghost-ingest) writes one short `project_pointer` row
per imported project and relies on it reaching the prompt, so the model
learns that the history exists. MemoryBus._fetch_pointer serves those rows
only, behind the env flag.

World where each pin fails:
  * unset (or any value but "1") serves anything;
  * a row of another type, or one past the off-topic floor, is served;
  * more than two rows are asked for or served;
  * the fetch books retrieval credit;
  * a failing store breaks hydration instead of returning nothing;
  * a served pointer does not reach hydrate_context's block.
"""
import asyncio

import pytest

from ghost_agent.core.bus import MemoryBus


class _Vector:
    """search_advanced as in memory/vector.py: raw distance in `score`."""

    def __init__(self, rows, boom=False):
        self.rows, self.boom, self.calls = rows, boom, []

    def search_advanced(self, query, limit=5, where=None, record_retrievals=True):
        self.calls.append({"query": query, "limit": limit, "where": where,
                           "record_retrievals": record_retrievals})
        if self.boom:
            raise RuntimeError("store down")
        rows = [r for r in self.rows if not where or r["metadata"].get("type") == where.get("type")]
        return rows[:limit]


def _row(i, text, dist, typ="project_pointer"):
    return {"id": f"id{i}", "text": text, "metadata": {"type": typ}, "score": dist}


PTR = "[p:proj-tradebot] Project tradebot (git/aigens/tradebot): grib decoding; tests."


def _fetch(vec, query="the trading bot"):
    return asyncio.run(MemoryBus(vector_memory=vec)._fetch_pointer(query))


@pytest.mark.parametrize("value", [None, "", "0", "true", "yes", " 2 "])
def test_off_unless_exactly_one(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("GHOST_POINTER_HYDRATION", raising=False)
    else:
        monkeypatch.setenv("GHOST_POINTER_HYDRATION", value)
    vec = _Vector([_row(1, PTR, 0.1)])
    assert _fetch(vec) == [] and vec.calls == []


def test_serves_close_pointers_only_without_credit(monkeypatch):
    monkeypatch.setenv("GHOST_POINTER_HYDRATION", " 1 ")
    vec = _Vector([_row(1, PTR, 0.30), _row(2, "[p:proj-far] Project far", 0.43),
                   _row(3, "a document chunk", 0.05, typ="document")])
    got = _fetch(vec)
    assert got == [{"source": "vector", "text": PTR, "mem_id": "id1"}]
    assert vec.calls == [{"query": "the trading bot", "limit": 2,
                          "where": {"type": "project_pointer"}, "record_retrievals": False}]


def test_the_floor_is_inclusive_and_bad_scores_are_skipped(monkeypatch):
    monkeypatch.setenv("GHOST_POINTER_HYDRATION", "1")
    vec = _Vector([_row(1, PTR, 0.42), {"id": "x", "text": "no score", "metadata": {"type": "project_pointer"}}])
    assert [g["mem_id"] for g in _fetch(vec)] == ["id1"]


def test_a_failing_store_returns_nothing(monkeypatch, caplog):
    monkeypatch.setenv("GHOST_POINTER_HYDRATION", "1")
    assert _fetch(_Vector([], boom=True)) == []
    assert "pointer fetch failed: RuntimeError" in caplog.text


def test_no_store_or_no_search_advanced_returns_nothing(monkeypatch):
    monkeypatch.setenv("GHOST_POINTER_HYDRATION", "1")
    assert asyncio.run(MemoryBus()._fetch_pointer("q")) == []
    assert asyncio.run(MemoryBus(vector_memory=object())._fetch_pointer("q")) == []


def test_a_served_pointer_reaches_the_hydrated_block(monkeypatch):
    monkeypatch.setenv("GHOST_POINTER_HYDRATION", "1")
    bus = MemoryBus(vector_memory=_Vector([_row(1, PTR, 0.2)]))
    block = asyncio.run(bus.hydrate_context("how did the trading bot decode grib?"))
    assert PTR in block and "MEMORY CONTEXT" in block
    monkeypatch.delenv("GHOST_POINTER_HYDRATION")
    assert PTR not in asyncio.run(bus.hydrate_context("how did the trading bot decode grib?"))


def test_the_tier_list_is_upstreams_when_off(monkeypatch):
    """Off, _fetch_all_tiers returns exactly upstream's four lists (pinned by
    tests/test_bus_rag_fusion.py); on, the pointer list comes last."""
    vec = _Vector([_row(1, PTR, 0.2)])
    bus = MemoryBus(vector_memory=vec)
    assert len(asyncio.run(bus._fetch_all_tiers("the trading bot"))) == 4 and vec.calls == []
    monkeypatch.setenv("GHOST_POINTER_HYDRATION", "1")
    got = asyncio.run(bus._fetch_all_tiers("the trading bot"))
    assert len(got) == 5 and got[-1][0]["text"] == PTR
