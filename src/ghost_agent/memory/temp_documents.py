"""Temporary knowledge-base documents (§4PP, 2026-10-16).

Why. "summarise this video: <youtube link>" ingests the video's transcript
into the knowledge base — the transcript action is how the model reads it in
order — and the document then stayed there for good: 5 of the 6 documents in
the owner's library were transcripts of videos he had only asked to have
summarised (one of them was later recalled as an answer to "what is my
name?"). The owner (operator, 2026-10-16): "it's fine to get them
temporarily but they should be marked as temp and removed with a maintenance
job later on".

What. A small registry beside the other memory sidecars,
``<memory dir>/temporary_documents.json``: ``{filename: {"expires_at": ts,
"added_at": ts, "reason": str}}`` for a temporary document, ``{"kept_at":
ts}`` for one the owner asked to keep. The store-hygiene idle job calls
:func:`sweep`, which deletes every expired temporary document through the
store's own ``delete_document_by_name`` (rows, summary row and both sidecars
in one critical section) and also marks any YouTube transcript that has no
entry yet (the ones ingested before this existed) as temporary from that
moment. Fail-safe: every function degrades rather than raising into a turn.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("GhostAgent")

REGISTRY_FILENAME = "temporary_documents.json"
#: how long a temporary document stays (GHOST_TEMP_DOC_DAYS overrides)
DEFAULT_TTL_DAYS = 7.0


import re as _re

#: the OWNER asked, in this very message, to put the video IN THE STORE —
#: the one unambiguous shape, decided at ingest time. Everything else is
#: temporary, and the model keeps it on the owner's word later with
#: action='keep' (§4PP r2 M2: a word list that tried to read every keep
#: flipped both ways in two review rounds — "keep it short" kept, "yes
#: please" refused — so it now decides only the clear case).
_KEEP_RES = (
    _re.compile(r"\b(?:save|store|add|put|keep|ingest|import|file)\b[^.?!\n]{0,40}?\b(?:to|in|into)\s+"
                r"(?:my\s+|the\s+|your\s+)?(?:knowledge[\s_-]?base|kb|library)\b", _re.I),
    _re.compile(r"(?:αποθ[ηή]κευσ|πρ[οό]σθεσ|β[αά]λ|κρ[αά]τ)\w*\s+[^.;!\n]{0,30}?στη?ν?\s+β[αά]ση\s+γν[ωώ]σ", _re.I),
)
_NEGATED_RE = _re.compile(
    r"\b(?:don'?t|do\s+not|no\s+need\s+to|never|not)\s+(?:\w+\s+){0,3}?"
    r"(?:save|store|keep|add|put)\b[^.?!\n]{0,40}?\b(?:knowledge[\s_-]?base|kb|library)\b"
    r"|\bμην\s+(?:\w+\s+){0,3}?(?:το\s+)?(?:αποθ|κρατ|κρ[αά]τ|βάλ|βαλ|προσθ)", _re.I)


def owner_asked_to_keep(user_text: str) -> bool:
    """True when the owner's own message asks to put it IN the knowledge
    base ("add this video to my knowledge base", "βάλ' το στη βάση
    γνώσεων"); anything else is decided later, by the owner's word."""
    t = str(user_text or "")
    return bool(t) and any(rx.search(t) for rx in _KEEP_RES) and not _NEGATED_RE.search(t)


def ttl_days() -> float:
    try:
        v = float(os.environ.get("GHOST_TEMP_DOC_DAYS", DEFAULT_TTL_DAYS))
        return v if v > 0 else DEFAULT_TTL_DAYS
    except (TypeError, ValueError):
        return DEFAULT_TTL_DAYS


def _path(memory_system) -> Optional[Path]:
    f = getattr(memory_system, "outlines_file", None)
    return Path(f).parent / REGISTRY_FILENAME if f else None


class _NoLock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _lock(memory_system):
    g = getattr(memory_system, "_get_lock", None)
    try:
        return g() if callable(g) else _NoLock()
    except Exception:  # noqa: BLE001
        return _NoLock()


def _load(memory_system) -> Dict[str, dict]:
    p = _path(memory_system)
    if p is None or not p.exists():
        return {}
    try:
        from ..tools.file_system import read_text_nofollow
        d = json.loads(read_text_nofollow(p) or "{}")
        return {str(k): v for k, v in d.items() if isinstance(v, dict)} if isinstance(d, dict) else {}
    except Exception as e:  # noqa: BLE001 — an unreadable registry deletes NOTHING
        logger.warning("temporary-document registry unreadable (%s); nothing will be removed", e)
        raise


def _save(memory_system, data: Dict[str, dict]) -> bool:
    p = _path(memory_system)
    if p is None:
        return False
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, p)
        return True
    except OSError as e:
        logger.warning("temporary-document registry not saved: %s", e)
        return False


def get(memory_system, filename: str) -> Optional[dict]:
    try:
        return _load(memory_system).get(str(filename))
    except Exception:  # noqa: BLE001
        return None


def is_temporary(memory_system, filename: str) -> bool:
    e = get(memory_system, filename)
    return bool(e and e.get("expires_at"))


def mark(memory_system, filename: str, reason: str = "", now: Optional[float] = None,
         days: Optional[float] = None) -> Optional[float]:
    """Mark ``filename`` temporary (a fresh expiry each time it is used) and
    return the expiry, or None. A document the owner KEPT stays kept."""
    now = time.time() if now is None else float(now)
    exp = now + 86400.0 * (ttl_days() if days is None else float(days))
    try:
        with _lock(memory_system):
            data = _load(memory_system)
            cur = data.get(str(filename)) or {}
            if cur.get("kept_at"):
                return None
            data[str(filename)] = {"expires_at": exp, "added_at": cur.get("added_at") or now,
                                   "reason": str(reason or cur.get("reason") or "")[:120]}
            return exp if _save(memory_system, data) else None
    except Exception as e:  # noqa: BLE001
        logger.warning("could not mark %s temporary: %s", filename, e)
        return None


def keep(memory_system, filename: str, now: Optional[float] = None) -> bool:
    """The owner keeps it: no expiry from now on."""
    now = time.time() if now is None else float(now)
    try:
        with _lock(memory_system):
            data = _load(memory_system)
            data[str(filename)] = {"kept_at": now}
            return _save(memory_system, data)
    except Exception as e:  # noqa: BLE001
        logger.warning("could not keep %s: %s", filename, e)
        return False


def forget_entry(memory_system, filename: str) -> None:
    """The document is gone (a forget): its keep/temporary record goes too
    (§4PP r2 M1: a kept record outlived the forget, and re-summarising the
    same video later stored it permanently, unasked)."""
    try:
        with _lock(memory_system):
            data = _load(memory_system)
            if data.pop(str(filename), None) is not None:
                _save(memory_system, data)
    except Exception as e:  # noqa: BLE001
        logger.warning("temporary-document record for %s not dropped: %s", filename, e)


def clear(memory_system, held: bool = False) -> None:
    """reset_all: no document is left, so no record either. ``held``: the
    caller already holds the store lock (the wipe is ONE critical section,
    pinned in test_forget_disk_sweep_scope)."""
    try:
        p = _path(memory_system)
        if p is None or not p.exists():
            return
        with (_NoLock() if held else _lock(memory_system)):
            _save(memory_system, {})
    except Exception as e:  # noqa: BLE001
        logger.warning("temporary-document registry not cleared: %s", e)


def _is_video_transcript(memory_system, filename: str) -> bool:
    """A YouTube transcript by its OUTLINE, which only the YouTube ingest
    writes with a ``video_id`` (r1 M1: the filename heuristics marked an
    owner's `yt-notes.pdf` or a `.captions.` subtitle file temporary)."""
    try:
        rec = memory_system.get_document_outline(filename) or {}
    except Exception:  # noqa: BLE001
        return False
    return bool(str(rec.get("video_id") or "").strip())


def _library(memory_system) -> List[str]:
    """The catalogue, RAISING when it cannot be read (r1 M2: the forgiving
    `get_library` answers [] on an error, and the sweep then dropped every
    entry — kept ones included — so kept transcripts were re-marked)."""
    load = getattr(memory_system, "_load_library", None)
    lib = load() if callable(load) else memory_system.get_library()
    if not isinstance(lib, list):
        raise ValueError("library index is not a list")
    return [str(x) for x in lib]


def _library_is_authoritative(memory_system) -> bool:
    """True when the catalogue came from the RAISING reader, so an empty
    list means an empty library (a forget of everything, a reset)."""
    return callable(getattr(memory_system, "_load_library", None))


def sweep(memory_system, now: Optional[float] = None) -> Dict[str, List[str]]:
    """Remove expired temporary documents; mark untracked video transcripts
    temporary from now; drop entries whose document is gone. Returns
    ``{"removed": [...], "marked": [...], "dropped": [...]}``. Never raises."""
    out: Dict[str, List[str]] = {"removed": [], "marked": [], "dropped": []}
    if memory_system is None or _path(memory_system) is None:
        return out
    now = time.time() if now is None else float(now)
    try:
        with _lock(memory_system):
            data = _load(memory_system)
            library = _library(memory_system)
            if not library and data and not _library_is_authoritative(memory_system):
                # an empty catalogue beside a non-empty registry is more likely
                # a read problem than a library emptied by hand: change nothing
                logger.warning("temporary-document sweep skipped: the library reads empty "
                               "while %d entries are registered", len(data))
                return out
            changed = False
            for name in library:
                if name not in data and _is_video_transcript(memory_system, name):
                    data[name] = {"expires_at": now + 86400.0 * ttl_days(), "added_at": now,
                                  "reason": "video transcript (found by the sweep)"}
                    out["marked"].append(name)
                    changed = True
            for name, e in list(data.items()):
                if name not in library and not e.get("kept_at"):
                    # (r3 M2: a KEPT record is never dropped on absence — a
                    # missing or truncated catalogue reads [] as authoritative;
                    # a real forget drops it through `forget_entry`, a reset
                    # through `clear`)
                    data.pop(name)
                    out["dropped"].append(name)
                    changed = True
                    continue
                exp = e.get("expires_at")
                if exp and float(exp) <= now:
                    try:
                        memory_system.delete_document_by_name(name)
                    except Exception as ex:  # noqa: BLE001 — try again next sweep
                        logger.warning("temporary document %s not removed: %s", name, ex)
                        continue
                    data.pop(name)
                    out["removed"].append(name)
                    changed = True
            if changed:
                _save(memory_system, data)
    except Exception as e:  # noqa: BLE001 — an unreadable registry removes nothing
        logger.warning("temporary-document sweep skipped: %s", e)
    return out


def expiry_note(expires_at: float) -> str:
    import datetime as _dt
    return _dt.datetime.fromtimestamp(float(expires_at)).strftime("%Y-%m-%d")


__all__ = ["owner_asked_to_keep", "forget_entry", "clear", "mark", "keep", "get", "is_temporary", "sweep", "ttl_days", "expiry_note",
           "REGISTRY_FILENAME", "DEFAULT_TTL_DAYS"]
