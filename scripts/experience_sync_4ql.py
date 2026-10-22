#!/usr/bin/env python
"""§4QL — rebuild the human-graded experience index from the trajectory corpus
and the label files (the same `sync_experience_index` the idle maintenance
runs), and manage the experiment hold-out.

    GHOST_HOME=~/Data/AI/Data PYTHONPATH=src python scripts/experience_sync_4ql.py
    ... --exclude-tids cases.json     # hold these trajectory ids OUT (leave-one-out replay)
    ... --clear-exclusions            # put them back
    ... --show "what is the latest version of postgresql ?"   # what the tier would render

The agent's own copy of the store is re-read on its next hydration (rows are
cached per process until the next sync), so run this with the agent stopped,
or let its 6-hourly store maintenance pick the labels up by itself.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--home", default=os.environ.get("GHOST_HOME", ""))
    ap.add_argument("--exclude-tids", default="", help="JSON list (or {tid: …}) of trajectory ids to hold out")
    ap.add_argument("--note", default="4ql replay hold-out")
    ap.add_argument("--clear-exclusions", action="store_true")
    ap.add_argument("--show", default="", help="render what the tier would hydrate for this request")
    ap.add_argument("--no-sync", action="store_true")
    a = ap.parse_args()
    if not a.home:
        print("GHOST_HOME not set", file=sys.stderr)
        return 2
    sysd = Path(a.home).expanduser() / "system"
    mdir = sysd / "memory"
    # the production embedder (same model name as the vector store) WITHOUT
    # opening the chroma store: that has one writer process (store_lock.py)
    from chromadb.utils import embedding_functions
    from ghost_agent.memory.vector import EMBED_MODEL_NAME
    from ghost_agent.memory.experience import ExperienceMemory, sync_experience_index
    efn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_NAME)
    store = ExperienceMemory(mdir, embed_fn=efn)
    if a.clear_exclusions:
        print("exclusions cleared:", store.clear_excluded())
    if a.exclude_tids:
        raw = json.loads(Path(a.exclude_tids).read_text(encoding="utf-8"))
        tids = list(raw.keys()) if isinstance(raw, dict) else list(raw)
        print("held out:", store.set_excluded(tids, note=a.note))
    if not a.no_sync:
        counts = sync_experience_index(store=store, system_dir=sysd)
        print("sync:", json.dumps(counts))
    print("rows:", store.count(), "excluded:", len(store.excluded()))
    if a.show:
        for ln in store.lines_for(a.show):
            print(ln)
    return 0


if __name__ == "__main__":
    sys.exit(main())
