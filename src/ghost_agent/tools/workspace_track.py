"""workspace_track tool — the agent's write path into workspace state.

Author the workspace's tracked-file list and free-form notes. The
companion read tool is ``workspace``.

Actions:
  * ``track`` — add a file path to the watchlist (optionally with label).
  * ``untrack`` — remove a file from the watchlist.
  * ``note`` — append a free-form workspace event (kind='note').
  * ``mark_seen`` — record a URL as already-pulled (manual dedup).

Everything routes through ``WorkspaceModel`` so bounded / atomic /
dedup guarantees from the state thread are preserved.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..utils.logging import Icons, pretty_log


def _TO_failed(text):
    """A tool-level exception, DECLARED failed (§4LZ B8: these plain
    strings were booked OK by the turn loop and the corpus)."""
    from .outcome import ToolOutcome
    return ToolOutcome.failed(text, reason_code="tool_exception")


logger = logging.getLogger("GhostAgent")


_VALID_ACTIONS = frozenset({"track", "untrack", "note", "mark_seen"})


async def tool_workspace_track(
    action: str = None,
    path: str = None,
    label: str = None,
    text: str = None,
    url: str = None,
    workspace_model=None,
    **kwargs,
) -> str:
    # str() so a non-string action can't raise AttributeError on .strip().
    raw_action = str(action or "").strip().lower()
    if raw_action not in _VALID_ACTIONS:
        # Read-intent steer (2026-07-26): this is the WRITE tool, but the
        # model reaches for it wanting to READ recent events / changes —
        # then, with no action or a read-verb, gets a bare "action is
        # mandatory" and REPEATS the same wrong call (live req 30: two
        # fatal strikes on "what's new today?"). When the call looks like a
        # read (no action, or a read verb), point at the `workspace` tool.
        _READ_VERBS = {"recent", "list", "summary", "changes", "read",
                       "view", "show", "get", "check", "status", "activity",
                       "tasks", "research", "commands", "files", "narrative",
                       "search"}
        if not raw_action or raw_action in _READ_VERBS:
            return (
                "SYSTEM ERROR: 'workspace_track' is the WRITE tool (it only "
                "AUTHORS the watchlist / notes / dedup markers; actions: "
                f"{sorted(_VALID_ACTIONS)}). To READ workspace state — recent "
                "events, changes, tasks, research, what's new — call the "
                "`workspace` tool instead (e.g. workspace(action='recent') or "
                "workspace(action='changes'))."
            )
        return (
            "SYSTEM ERROR: 'action' is mandatory and must be one of "
            f"{sorted(_VALID_ACTIONS)}."
        )

    if workspace_model is None or not getattr(workspace_model, "enabled", False):
        from .outcome import ToolOutcome as _TOr
        return _TOr.rejected(                                   # declared (§4LZ B8)
            "Workspace continuity is unavailable — the workspace module "
            "is disabled. Nothing was recorded.",
            world_changed=False, reason_code="tool_unavailable")

    from .outcome import ToolOutcome as _TO
    # §4PE (W2): a probe turn leaves no trace in the owner's ledger — the
    # four writers below were ungated (a probe's mark_seen hid the owner's
    # later real pull; its notes and watch entries became the owner's record)
    try:
        from ..workspace.model import _probe_turn as _is_probe
        if _is_probe():
            return _TO.rejected("Error: a probe request does not write the workspace ledger — nothing recorded.",
                                world_changed=False, reason_code="not_owner_write")
    except ImportError:
        pass
    try:
        if raw_action == "track":
            path = (path or "").strip()
            if not path:
                return "SYSTEM ERROR: 'path' is required for track."
            tf = workspace_model.track_file(path, label=(label or "").strip())
            if tf is None:
                return _TO.rejected(
                    f"Error: '{path}' is not a file inside the workspace — tracked paths are workspace "
                    f"paths (as the sandbox sees them, e.g. /workspace/projects/<id>/app.py). Nothing recorded.",
                    world_changed=False, reason_code="workspace_path_outside")
            pretty_log("Workspace", f"now tracking {path}", icon=Icons.IDEA)
            desc = f" labelled '{tf.label}'" if tf.label else ""
            return _TO.ok(f"Now tracking '{workspace_model.display_path(tf.path)}'{desc}.", world_changed=True)

        if raw_action == "untrack":
            path = (path or "").strip()
            if not path:
                return "SYSTEM ERROR: 'path' is required for untrack."
            ok = workspace_model.untrack_file(path)
            if not ok:
                return f"'{path}' is not on the watchlist."
            pretty_log("Workspace", f"untracked {path}", icon=Icons.OK)
            return f"No longer tracking '{path}'."

        if raw_action == "note":
            text = (text or "").strip()
            if not text:
                return "SYSTEM ERROR: 'text' is required for note."
            ev = workspace_model.note(text)
            if ev is None:
                return "Nothing recorded — the note was empty."
            pretty_log("Workspace", f"note recorded: {text[:60]}", icon=Icons.IDEA)
            return _TO.ok(f"Recorded workspace note: {text}", world_changed=True)   # §4PE (W1): declared

        if raw_action == "mark_seen":
            url = (url or "").strip()
            if not url:
                return "SYSTEM ERROR: 'url' is required for mark_seen."
            if workspace_model.state is None:
                return "Workspace state unavailable."
            added = workspace_model.state.mark_url_seen(url)
            if not added:
                return _TO.ok(f"URL already on the seen list: {url}", world_changed=False)
            pretty_log("Workspace", f"marked seen: {url}", icon=Icons.OK)
            return _TO.ok(f"Marked '{url}' as already-pulled.", world_changed=True)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "workspace_track tool failed: %s: %s", type(e).__name__, e,
        )
        return _TO_failed(f"Workspace track operation failed: {type(e).__name__}: {e}")   # declared (§4LZ B8)

    return "SYSTEM ERROR: unreachable workspace_track branch."
