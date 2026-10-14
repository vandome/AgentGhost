#!/usr/bin/env python3
"""§4OR one-off (2026-10-13, operator: "protect the rules"): repair the 10-12
damage — rule 1 (health) lost its text to rule 4 (statistics) through a
vector-dedup merge, and rule 4 was reported "not adopted".

1. `check_rule_integrity` puts rule 1's adopted text back (the ledger keeps it).
2. "learn rule 4" runs through the owner's own adoption path (now safe: an
   owner rule never merges into another).

Run ONLY with the agent stopped. Dry run by default; `--apply` writes.
Backups of the playbook and the rule ledger are taken first.
  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \\
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/rule_repair_4or.py [--apply]
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
APPLY = "--apply" in sys.argv
HOME = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/")) / "system"
MEM = HOME / "memory"
STAMP = time.strftime("%Y%m%dT%H%M%S")


def main() -> int:
    from ghost_agent.core import failure_replay as FR
    from ghost_agent.memory.skills import SkillMemory
    entries = FR.load(HOME)
    rows = json.loads((MEM / "skills_playbook.json").read_text(encoding="utf-8"))
    texts = {FR._norm(r.get("solution")) for r in rows if r.get("origin") == "owner_rule"}
    for e in entries:
        if e.get("rule"):
            n = FR.case_n(entries, e)
            state = "adopted" if FR.is_adopted(e) else "forgotten" if e.get("forgotten_at") else "proposed"
            print(f"rule {n}: {state:9} | text in playbook: {FR._norm(e['rule']) in texts} | {e['rule'][:70]}")
    if not APPLY:
        print("(dry run) would: restore every adopted rule whose text is missing; then 'learn rule 4'")
        return 0
    from ghost_agent.memory.store_lock import assert_no_other_writer
    assert_no_other_writer(MEM, Path(__file__).name)
    shutil.copy2(MEM / "skills_playbook.json", MEM / f"skills_playbook.json.pre-4or-{STAMP}.bak")
    led = FR._path(HOME)
    shutil.copy2(led, led.with_name(led.name + f".pre-4or-{STAMP}.bak"))
    sm = SkillMemory(MEM)
    from ghost_agent.memory.vector import VectorMemory
    vm = VectorMemory(MEM, upstream_url=os.environ.get("GHOST_UPSTREAM", "http://127.0.0.1:8088"))
    print("integrity:", FR.check_rule_integrity(HOME, sm, vm))
    note = FR.owner_rule_command("learn rule 4", HOME, skill_memory=sm, memory_system=vm)
    print("learn rule 4:", getattr(note, "banner", "") or note)
    print("integrity after:", FR.check_rule_integrity(HOME, sm, vm))
    # each rule's vector twin now carries ITS text (rule 1's still held rule 4's)
    for r in sm.all_owner_rule_rows():
        sm._refresh_twin(vm, r)
    rows = json.loads((MEM / "skills_playbook.json").read_text(encoding="utf-8"))
    for r in rows:
        if r.get("origin") == "owner_rule":
            print("  OWNER RULE |", (r.get("trigger") or "")[:60], "→", (r.get("solution") or "")[:70])
    return 0


if __name__ == "__main__":
    sys.exit(main())
