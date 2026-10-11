"""§4DA round 11 — the un-stamping was one-armed, one layer out.

Round 10 made the loader stamp control turns so both arms carry an era. Round 3
had made the READ SITE prune a stamp when it refuses the artifact. Together
those produce **one-armed attrition**: a control turn is served the baseline and
returns at `if not tuned: return baseline`, *before* the per-tool validator and
before either unnote call, so control stamps were created and never removed
while treatment's were removed on every refusal.

The aggregate ceiling is per-TURN — the read site gets the `_intent_filter`ed
subset and each signature randomizes independently — so whether the set is
dropped does not depend on which arm the turn drew. Driven over 200 turns with
an artifact that is neutral BY CONSTRUCTION (outcome depends only on turn
shape, identically in both arms):

    ceiling never fires : treatment 51/97, control 49/103  -> KEEP,   p=0.8020
    ceiling fires       : treatment  9/55, control 49/103  -> REVERT, p=0.0001

`activation_stats` showed `applied: 55, fallback: 103, rejected: 42` — 42
treatment stamps pruned, 0 control. `gepa_live_check --revert` renames on that
verdict. It is the principle `live_check` states — "BOTH ARMS, OR THE
COMPARISON STOPS BEING RANDOMIZED" — surviving in the refusal path rather than
the sha path.

§4NM: `optim/live_check`, `scripts/run_gepa.py` and the recheck reader were
removed; their tests went with them. The read-site pins remain.
"""

import json

from ghost_agent.optim import loader as L
from ghost_agent.tools import registry as R
