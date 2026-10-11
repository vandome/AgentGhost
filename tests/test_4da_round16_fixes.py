"""§4DA round 16 — the turn's attribution described a tool set the model
never saw.

Applying the tuned descriptions is not a pure read. It draws an experiment arm
per artifact, STAMPS the request's attribution, and prunes that stamp when the
assembled set busts the aggregate ceiling — all three properties of *the tool
list passed in*. So a second call with a DIFFERENT list silently overwrites the
first one's verdict, and the last call wins.

The planner's "available tools" line built a name list with `query=None`, i.e.
over the UN-ROUTED superset, while the prompt was built from the routed subset
and cached per request. Turn 1 was correct (the prompt build ran last); from
turn 2 the prompt build came from cache and the name-list call was the ONLY one:

    turn 1  rendered ['file_system', 'execute'], stamps treatment/treatment
    turn 2  stamps {}   ->  gepa_artifact_applied False

That is round 14's headline defect — renders the artifact, keeps no stamp, and
is therefore mined into the pool that ship-gates the next run — reached through
a call site round 14 did not consider. It also blinds `gepa_live_check`
permanently, so `--revert` is structurally unreachable and a losing artifact is
never retired.

§4NM: `optim/live_check`, both ship gates and the recheck reader were removed;
their tests went with them. The read-site pins remain.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

from ghost_agent.optim import loader as L
from ghost_agent.tools import registry as R
from ghost_agent.utils.logging import request_id_context



def _ctx():
    return SimpleNamespace(
        llm_client=SimpleNamespace(swarm_clients=None, image_gen_clients=None),
        args=SimpleNamespace(default_db=None))

