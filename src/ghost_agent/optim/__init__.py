"""Ghost Agent prompt optimization — the LOADER side (§4NM).

What remains after the autonomy / optimizer removal: the signature
registry (`signatures.py`, which renders tool descriptions through
`tools/registry.py` and `core/verifier.py`), the epoch-pinned artifact
loader (`loader.py`, §4DE) and the gate contract (`gate_contract.py`)
that the loader validates artifacts against. The optimizer, A/B evaluator,
trainset builder, fixture/ontology miners and the autonomous GEPA loop
were removed in §4NM; artifacts already on disk under
`$GHOST_HOME/system/optim` keep loading.
"""

from .signatures import (
    OptimizableSignature,
    PLANNING_SIGNATURE,
    TOOL_SELECTION_SIGNATURE,
    SIGNATURES,
)

__all__ = [
    "OptimizableSignature",
    "PLANNING_SIGNATURE",
    "TOOL_SELECTION_SIGNATURE",
    "SIGNATURES",
]
