"""§4DA round 4 — an outage and a corpus gap are not the same event.

Round 2 fixed the PRINTED line for exactly this confusion — "a corpus gap
(stable across both arms)" vs "an outage (the thing that invalidates the
pairing)" — and then armed its NEW `underpowered` gate on the merged set. So a
pool with missing recordings refused an honest, significant win with a remedy
("re-run when the upstream is stable") that is a no-op loop costing
`iterations * len(pub)` main-model calls per attempt.

Round 3 then reshaped the artifact so `recheck_gepa_incumbent.py` could print
the override warning, and left in place the `_load_signature` call that rejects
the file before `--artifact` is consulted. The test that "verified" the reshape
re-typed the reader's expressions inside itself instead of running it — the
§4DA failure mode, verbatim, in a test written to close the §4DA failure mode.

§4NM: the ship gate, the recheck reader and `optim/ab_eval` were removed;
their tests went with them. What remains pins the read-site attribution.
"""

import json
