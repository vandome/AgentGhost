"""§4DA round 15 — the RETIRE code was pinned by its own source text.

Round 13 gave `recheck_gepa_incumbent.py` a fourth exit code so a script could
tell "the incumbent still earns its place" from "this holdout cannot settle the
question". Its only guard was a **source-shape assertion** — `body.index("if
_unmeasurable:")` followed by `assert "return 2" in body[i:i+80]`. Inserting one
line above the branch:

    _unmeasurable = _unmeasurable and False

restores the exact pre-fix defect (exit 0, "still wins", from the branch that
just printed *"it is evidence that this holdout cannot settle the question"*)
and **survives the full 855-test battery**. The ten tests that DO reach the
branch all assert `rc in (0, 1, 2)` — which admits the pre-fix 0 — with a
comment three lines above naming the very fix they cannot see.

Round 8's own ⚠ recorded *"SIX of round 8's own first pins were SOURCE GREPS"*.
Round 13 did it again, in the branch a script acts on to retire a live artifact.

§4NM: the recheck reader, both ship gates and `scripts/run_gepa.py` were
removed; their tests went with them. The read-site pin remains.
"""

import json

from ghost_agent.optim import loader as L
from ghost_agent.tools import registry as R
