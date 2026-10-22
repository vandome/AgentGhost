#!/usr/bin/env python
"""§4QM — seed the owner's fact ledger from the stores that exist today, and
attribute each live fact to the owner's own statement in the trajectory corpus.

    GHOST_HOME=~/Data/AI/Data PYTHONPATH=src python scripts/fact_ledger_backfill_4qm.py          # report only
    ... --apply                                                                                  # write the rows

What is seeded (event `backfilled`, store/field/value as the stores hold them):
  * every current profile value, dated by its `as_of` stamp (undated values: today, marked);
  * every LIVE owner graph edge (`user …`), dated by `valid_from`;
  * every EXPIRED owner edge (event `expired`, `valid_until`).
Attribution (§4EM's admissibility rule): the owner's OWN line — a user_request
of an owner turn, never a document or the assistant's reply — that contains the
value as whole words (the value's distinctive words, ≥ 2 of them or one word of
≥ 5 letters), the nearest such turn at or before the fact's date (+1 day for the
drain lag). A fact nothing in the corpus states is listed UNATTRIBUTED and still
seeded (with no `said`) — nothing live is changed either way. Idempotent: a
field+value already in the ledger is not seeded twice.
"""
from __future__ import annotations

import argparse
import time
import datetime as dt
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _words(text: str) -> set:
    from ghost_agent.memory.profile import _fold
    return {w for w in re.findall(r"[^\W\d_]{3,}|\d{4}", _fold(str(text or ""))) if w not in _STOP}


_STOP = {"the", "and", "for", "with", "greece", "user", "has", "που", "και", "για", "την", "του", "της"}


def _contains(line_words: set, value: str) -> bool:
    """EVERY distinctive word of the value is in the line (full coverage —
    a shared word is a lexical proxy: "ai persistence" met "verify that data
    persistence works", §4EM's lesson)."""
    vw = _words(value)
    return bool(vw) and vw <= line_words


def _statement_clauses(text: str) -> list:
    """The owner's first-person declarative clauses of a request (questions,
    quotes and role-play excluded — the attribution module's rule): a fact
    is attributed to a STATEMENT, never to a question that mentions it."""
    from ghost_agent.memory.attribution import owner_statements
    try:
        return [c for c, negated in owner_statements("USER: " + str(text or "")) if not negated]
    except Exception:  # noqa: BLE001
        return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--home", default=os.environ.get("GHOST_HOME", ""))
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if not a.home:
        print("GHOST_HOME not set", file=sys.stderr); return 2
    sysd = Path(a.home).expanduser() / "system"; mdir = sysd / "memory"
    from ghost_agent.memory.fact_ledger import FactLedger, _epoch
    from ghost_agent.memory.profile import ProfileMemory, unwrap, stamp_of
    from ghost_agent.distill.collector import TrajectoryCollector
    from ghost_agent.core.failure_census import is_owner_turn, delivered, _traj_dict

    # the corpus: owner turns, oldest first
    col = TrajectoryCollector(root=sysd / "trajectories", enabled=False)
    turns = []
    for tr in col.iter_trajectories(include_archive=True):
        t = _traj_dict(tr)
        if is_owner_turn(t) and delivered(t) and (t.get("user_request") or "").strip():
            for clause in _statement_clauses(t.get("user_request")):
                turns.append((_epoch(t.get("timestamp")) or 0.0, str((t.get("extra") or {}).get("req_id") or ""),
                              clause, _words(clause), str(t.get("user_request") or "")))
    turns.sort()
    print(f"owner statement clauses in the corpus: {len(turns)}")

    from ghost_agent.memory.attribution import owner_clause_for, cue_for_field

    def attribute(value: str, at: float, cat: str = "", key: str = ""):
        """The LAST owner statement clause (folded, full coverage) at or before
        the fact's date; the ledger keeps the owner's RAW clause for it."""
        best = None
        for ts, rid, text, ws, raw_req in turns:
            if ts > at + 86400:
                break
            if _contains(ws, value):
                best = (ts, rid, owner_clause_for(value, "USER: " + raw_req, cue=cue_for_field(cat, key)) or text)
        return best

    facts = []   # (store, category, key, value, event, at, how, prev_value)
    pm = ProfileMemory(mdir)
    raw = pm.load_raw()
    mirror = set()   # (has_<key>, value) — the graph's mirror edges of profile fields (one fact, one row)
    twins = set()    # every profile value: a graph edge that carries the same value is the same fact under
                     # another predicate (owns / assets.vehicles), dated by the EDGE's creation — one record
    for cat, sub in raw.items():
        if not isinstance(sub, dict):
            continue
        for key, item in sub.items():
            items = item if isinstance(item, list) else [item]
            # a replaced singleton keeps ONE previous value: the live value was
            # stated REPLACING it, the previous value was stated earlier (review
            # r1: the previous value was seeded as a live "replaced" row)
            pv = str(unwrap(item["previous"])).strip() if isinstance(item, dict) and "previous" in item else ""
            for it in items:
                v = str(unwrap(it)).strip()
                if not v:
                    continue
                st = stamp_of(it)
                mirror.add((f"has_{str(key).lower()}", v.lower()))
                twins.add(v.lower())
                if pv and pv != v:
                    facts.append(("profile", cat, key, v, "replaced", _epoch(st) if st else None, "" if st else "undated", pv))
                else:
                    facts.append(("profile", cat, key, v, "backfilled", _epoch(st) if st else None, "" if st else "undated", ""))
            if pv:
                pst = stamp_of(item["previous"]) if isinstance(item.get("previous"), dict) else None
                facts.append(("profile", cat, key, pv, "backfilled", _epoch(pst) if pst else None,
                              "previous value" if pst else "previous value, undated", ""))
    from ghost_agent.memory.graph import GraphMemory
    _gm = GraphMemory(mdir)
    g = sqlite3.connect(str(mdir / "knowledge_graph.db"))
    chatter = 0
    for s, p, o, vf, vu in g.execute("SELECT subject, predicate, object, valid_from, valid_until FROM triplets"
                                     " WHERE lower(subject)='user' OR lower(object)='user'"):
        if not _gm._is_owner_fact(str(s).lower(), str(p), str(o).lower()):
            chatter += 1          # ASKS_ABOUT / INTERESTED_IN / WANTS …: the extractor's topical edges, not a life fact
            continue
        other = o if str(s).lower() == "user" else s
        if (str(p).lower(), str(other).lower()) in mirror or str(other).lower() in twins:
            continue              # the profile row is the record; its mirror edge is derived (review r1);
                                  # a same-valued edge would date one fact twice (§4QM replay: "told me on 2026-09-21")
        if vu is None:
            facts.append(("graph", "graph", str(p).lower(), str(other), "backfilled", float(vf or 0) or None, "", ""))
        else:
            facts.append(("graph", "graph", str(p).lower(), str(other), "expired", float(vu), "expired before the ledger", ""))

    if a.apply:
        # §4MN/§4MR: stores are single-writer — the ledger is written by the live
        # agent; an --apply runs only while this process is the directory's only writer
        from ghost_agent.memory.store_lock import assert_no_other_writer
        assert_no_other_writer(mdir, "fact_ledger_backfill_4qm.py")
    led = FactLedger(mdir)
    from ghost_agent.memory.fact_ledger import _plain
    with led._conn() as c:
        # compared the way the ledger STORES them (lower-cased field, folded value)
        # — a raw compare re-seeded every double-spaced value on each --apply (review r1)
        have = {(r[0], r[1], r[2], r[3]) for r in c.execute("SELECT store, category, key, value FROM ledger")}
    attributed = unattributed = seeded = skipped = 0
    now = time.time()
    for store, cat, key, v, ev, at, how, prev in facts:
        if at is None:
            how = how or "undated"
            at = now
        hit = attribute(v, at, cat, key) if ev != "expired" else None
        if ev != "expired":
            attributed += bool(hit); unattributed += not hit
        label = "attributed" if hit else ("expired" if ev == "expired" else "UNATTRIBUTED")
        when = dt.datetime.fromtimestamp(at).strftime("%Y-%m-%d")
        said = f'  ← {hit[2][:90]!r} ({dt.datetime.fromtimestamp(hit[0]).strftime("%Y-%m-%d")})' if hit else ""
        print(f"  [{label:>12}] {store}:{cat}.{key} = {v[:60]!r} ({ev}, {when}){said}")
        if (store, str(cat).strip().lower()[:80], str(key).strip().lower()[:120], _plain(v)) in have:
            skipped += 1
            continue
        if a.apply:
            led.record(store=store, category=cat, key=key, value=v, event=ev, prev_value=prev or None, at=at,
                       how=how or ("" if hit else "unattributed"),
                       source={"req_id": hit[1] if hit else "", "said": hit[2] if hit else "", "via": "backfill"})
            seeded += 1
    print(f"\n({chatter} topical owner edges skipped: not durable facts)")
    print(f"facts {len(facts)}: attributed {attributed}, unattributed {unattributed}; "
          f"{'seeded ' + str(seeded) if a.apply else 'DRY RUN (use --apply)'}; already in the ledger {skipped}; rows now {led.count()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
