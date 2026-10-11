"""Rare-token leak hashing for text generated from a real owner turn.

§4MT built this module to turn an owner failure into a structured practice
brief for synthetic self-play. Self-play and the brief/shape mapping were
removed (§4NM, tidied after §4NN); what remains is the gate `failure_replay`
uses: hash the source turn's rare tokens (``source_token_hashes``) so a
generated text can be checked for copying (``leaked_tokens``) without the
owner's words travelling with it.
"""
from __future__ import annotations

import re


#: a token rare enough that sharing it means copying: numbers of 3+ digits,
#: @mentions, URLs, file names, capitalised words that are not sentence-initial
_RARE_RE = re.compile(r"\d{3,}|@\w+|https?://\S+|\b[\w-]+\.(?:pdf|csv|txt|md|json|py|html?|png|jpe?g|docx?)\b"
                      r"|(?<=[a-z,;:] )[A-Z][a-zA-Z]{3,}")


def _token_hash(tok: str) -> str:
    import hashlib
    return hashlib.sha256(tok.lower().encode("utf-8")).hexdigest()[:16]


def source_token_hashes(traj) -> list:
    """Hashes of the source turn's rare tokens (request + failure reason) —
    what a seed carries so the generated challenge can be gated without the
    owner's text travelling with it."""
    src = " ".join([str(getattr(traj, "user_request", "") or ""), str(getattr(traj, "failure_reason", "") or "")])
    return sorted({_token_hash(m.group(0)) for m in _RARE_RE.finditer(src)})


def leaked_tokens(text: str, hashes) -> list:
    """Rare tokens of ``text`` whose hash is among ``hashes``. Non-empty =
    reject the generated challenge."""
    hs = set(hashes or ())
    if not hs:
        return []
    return sorted({m.group(0) for m in _RARE_RE.finditer(str(text or "")) if _token_hash(m.group(0)) in hs})
