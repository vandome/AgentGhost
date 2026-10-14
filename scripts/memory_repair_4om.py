#!/usr/bin/env python3
"""§4OM one-off (2026-10-13, operator: "A, and proceed with the cleanup"): the
noise the three-day memory check found after the §4NG repair, and the
duplicate dog fact (profile assets.dogs + its graph/vector mirrors; the kept
copy is pet.dog_name). Same method as scripts/memory_repair_4om.py.

(§4NG's original header, for the method:)

The plan (`memory_repair_4om.json`, beside this script) was built read-only
from the §4NG lenses and approved by the operator:
  - graph: the owner's requests/greetings logged as "facts", the agent
    describing its own process, owner "facts" written from Slack MEMBERS'
    messages or inferred from questions, stale project ports/statuses, the
    image "FEATURES tsipras/samaras" rows the owner contradicted, the
    car-wash riddle, and the IBKR rows (operator: "No IBKR account");
  - vector: three false episodes, four noise rows, the two IBKR facts.

Every row is re-checked to still match before it is touched. Graph rows go
through `GraphMemory.delete_edge` (archived first); vector rows are appended
with their document and metadata to `vector_repair_archive_4om.jsonl` before
deletion. A full copy of the memory directory is taken first.

Run ONLY with the agent stopped. Dry run by default; `--apply` writes.
  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \\
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/memory_repair_4om.py [--apply]
"""
import json
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
APPLY = "--apply" in sys.argv
MEM = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/")) / "system" / "memory"
PLAN = json.loads((Path(__file__).with_suffix(".json")).read_text(encoding="utf-8"))
STAMP = time.strftime("%Y%m%dT%H%M%S")


def live_graph_rows() -> set:
    with sqlite3.connect(f"file:{MEM / 'knowledge_graph.db'}?mode=ro", uri=True) as c:
        return {(s, p, o) for s, p, o in c.execute(
            "SELECT subject, predicate, object FROM triplets WHERE valid_until IS NULL")}


def vector_docs(ids: list) -> dict:
    """{embedding_id: (document, metadata)} read-only from the chroma sqlite."""
    out = {}
    with sqlite3.connect(f"file:{MEM / 'chroma.sqlite3'}?mode=ro", uri=True) as c:
        for eid in ids:
            row = c.execute("SELECT id FROM embeddings WHERE embedding_id=?", (eid,)).fetchone()
            if not row:
                continue
            meta = {k: (sv if sv is not None else iv if iv is not None else fv)
                    for k, sv, iv, fv in c.execute(
                        "SELECT key, string_value, int_value, float_value FROM embedding_metadata WHERE id=?",
                        (row[0],))}
            out[eid] = (str(meta.pop("chroma:document", "") or ""), meta)
    return out


def main() -> int:
    live = live_graph_rows()
    graph_todo = [r for r in PLAN["graph"] if (r["subject"], r["predicate"], r["object"]) in live]
    graph_gone = len(PLAN["graph"]) - len(graph_todo)
    docs = vector_docs([r["id"] for r in PLAN["vector"]])
    vec_todo = [r for r in PLAN["vector"]
                if r["id"] in docs and docs[r["id"]][0].startswith(r["doc_head"][:40])]
    _p = json.loads((MEM / "user_profile.json").read_text(encoding="utf-8"))
    _pr = [r for r in PLAN.get("profile_remove", []) if ((_p.get(r["category"]) or {}).get(r["key"]) or {}).get("v") == r["expect"]]
    print(f"profile: {len(_pr)} of {len(PLAN.get('profile_remove', []))} field(s) to remove")
    print(f"graph: {len(graph_todo)} to archive+delete ({graph_gone} already gone); "
          f"vector: {len(vec_todo)} of {len(PLAN['vector'])} to archive+delete"
          + ("" if APPLY else " (dry run)"))
    if not APPLY:
        for r in graph_todo[:5]:
            print("  e.g.", r["subject"], r["predicate"], r["object"], "—", r["why"][:60])
        return 0
    from ghost_agent.memory.store_lock import assert_no_other_writer
    assert_no_other_writer(MEM, Path(__file__).name)          # §4MN: never beside the agent
    shutil.copytree(MEM, MEM.parent / f"memory.pre-4om-{STAMP}.bak",
                    ignore=shutil.ignore_patterns("*.bak", "*.bak-*", "*.pre-*", "*.pre_*"))
    from ghost_agent.memory.graph import GraphMemory
    g = GraphMemory(MEM)
    removed = sum(g.delete_edge(r["subject"], r["predicate"], r["object"]) for r in graph_todo)
    vec_removed = 0
    if vec_todo:
        arch = MEM / "vector_repair_archive_4om.jsonl"
        with arch.open("a", encoding="utf-8") as fh:
            for r in vec_todo:
                doc, meta = docs[r["id"]]
                fh.write(json.dumps({"id": r["id"], "document": doc, "metadata": meta, "why": r["why"],
                                     "archived_at": STAMP}, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        from ghost_agent.memory.vector import VectorMemory
        vm = VectorMemory(MEM, upstream_url=os.environ.get("GHOST_UPSTREAM", "http://127.0.0.1:8088"))
        ids = [r["id"] for r in vec_todo]
        vm.collection.delete(ids=ids)
        vec_removed = len(ids) - len(vm.collection.get(ids=ids).get("ids") or [])
    prof_removed = 0
    prof_path = MEM / "user_profile.json"
    prof = json.loads(prof_path.read_text(encoding="utf-8"))
    for r in PLAN.get("profile_remove", []):
        cat = prof.get(r["category"]) or {}
        cur = cat.get(r["key"])
        val = cur.get("v") if isinstance(cur, dict) else cur
        if val == r["expect"]:
            del cat[r["key"]]
            prof_removed += 1
    if prof_removed:
        tmp = prof_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(prof, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, prof_path)
    after = live_graph_rows()
    still = [r for r in graph_todo if (r["subject"], r["predicate"], r["object"]) in after]
    print(json.dumps({"graph_removed": removed, "graph_still_live": len(still), "vector_removed": vec_removed,
                      "graph_live_after": len(after), "profile_removed": prof_removed, "backup": f"memory.pre-4om-{STAMP}.bak"}, indent=1))
    return 1 if still else 0


if __name__ == "__main__":
    sys.exit(main())
