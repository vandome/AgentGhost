"""Repeat-failure guards shared by the turn loop.

``RecentFailureGuard`` (the pre-flight guard), ``primary_target_from_args``,
``guard_key_target`` and ``looks_mutating_command`` decide when a call is the
same failing action against the same target.

§4PT (2026-10-17): the trigger bus, the event types, the runtime budget, the
repetition counter and the ReplanBridge were removed. Across the whole log
history the bridge made 2 revision attempts (both rejected) and 0 revisions,
and every host-resource event was log-only.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import deque
from typing import Deque, Optional, Tuple

logger = logging.getLogger("GhostAgent")


# Arg keys that name the thing a tool call OPERATED ON, in priority order.
# A file_system read re-reading one path, a browser fetch re-pulling one
# url are the same pathology: repeating an action against an unchanged
# target. Shared by the in-run no-progress loop-breaker (core/agent.py)
# and :class:`RecentFailureGuard` so both agree on "the same target".
_PRIMARY_ARG_KEYS = ("path", "file_path", "file", "filename", "url", "selector", "query", "target")


def primary_target_from_args(args) -> str:
    """The value of the first recognised primary-arg key in ``args``, or
    "" when none is present (or ``args`` isn't a dict). Lower-cased +
    stripped so trivially different spellings of the same target collapse
    together."""
    if not isinstance(args, dict):
        return ""
    for k in _PRIMARY_ARG_KEYS:
        if k in args and args[k] is not None:
            return str(args[k]).strip().lower()[:200]
    return ""


class RecentFailureGuard:
    """Rolling memory of recently-FAILED tool calls, consulted *before*
    dispatch.

    Before a tool is dispatched the agent asks
    "have I already failed this exact action the same way?", and if so the
    call is blocked and the model is handed the prior error instead of
    re-running a known failure.

    Keying: ``(tool, primary-target)`` where the target comes from
    ``primary_target_from_args`` (path / url / selector / query / …). The
    error text is normalised to a short lowercased prefix so two failures
    that differ only in a trailing variable (an offset, a pid) count as the
    same recurring failure.

    The window is bounded (``window``) and entries age out as new failures
    push them off the back, so a one-off failure that isn't repeated soon
    simply falls out of memory and never blocks anything — the guard only
    fires on a *sustained* identical loop.

    Lifecycle (2026-07-30, solar-sim postmortem): a blocked call never
    dispatches, so it can never demonstrate that an intervening fix worked
    — without an external reset, two stale failures block a verifiably
    remediated action forever (observed live: the port holder was killed
    and the port confirmed free, yet three consecutive requests were boxed
    in by failures recorded before the fix). The dispatch loop therefore
    (a) calls :meth:`note_world_changed` after any SUCCESSFUL state-
    mutating call, and (b) calls :meth:`reset` at the start of each
    request — cross-request pathology stays covered by the offline
    post-mortem fingerprint, which mines whole transcripts.
    """

    def __init__(self, *, window: int = 24, repeat_threshold: int = 2):
        self.window = int(max(2, window))
        # Number of PRIOR identical (same tool, target, normalised error)
        # failures that must already be on record before the next attempt
        # is blocked. Default 2 → the action is allowed to fail twice the
        # SAME way (proving it isn't transient and wasn't fixed between
        # attempts) before the third identical re-issue is intercepted.
        #
        # Why not 1: keying on (tool, target) alone would block a LEGITIMATE
        # retry — e.g. re-running `execute` on a script after editing it to
        # fix the bug (the same pathology the idempotency guard deliberately
        # exempts `execute`/`file_system.write` for). Requiring the identical
        # ERROR to recur means a post-fix re-run that now succeeds, or fails a
        # NEW way, never trips the guard — only a genuine stuck loop does.
        self.repeat_threshold = int(max(1, repeat_threshold))
        # Each entry: ((tool, target, op), normalised-error).
        self._history: Deque[Tuple[Tuple[str, str], str]] = deque(maxlen=self.window)

    @staticmethod
    def _norm_err(text: str) -> str:
        """Normalised prefix of an error string — same shape as the offline
        ``_error_key`` so the live and offline notions of "the same error"
        line up."""
        return (text or "").strip()[:80].lower()

    def record(self, tool: str, target: str, error_text: str,
               op: str = "") -> None:
        """Remember one FAILED call. No-op for an empty tool or error (a
        successful call has no error and must never seed the guard).

        ``op`` is the tool's operation/action discriminator: for multi-op
        tools (file_system, manage_projects, browser) two calls on the same
        target with DIFFERENT operations are different actions — observed
        live 2026-07-08: keying on (tool, target) alone blocked the model's
        correct replace→write recovery three times in a row."""
        err = self._norm_err(error_text)
        if not tool or not err:
            return
        self._history.append(((tool, target or "", op or ""), err))

    def would_repeat(self, tool: str, target: str, op: str = "") -> Optional[str]:
        """If dispatching ``tool`` against ``target`` would re-run an action
        that has already failed identically at least ``repeat_threshold``
        times in the window, return the offending error text (for the
        corrective message); otherwise ``None``.

        "Identically" = same ``(tool, target)`` AND same normalised error.
        The most recent error recorded for the key anchors the match, so a
        target that failed two *different* ways does not trip the guard —
        only a genuine repeat of one failure does.
        """
        if not tool:
            return None
        key = (tool, target or "", op or "")
        last_err = None
        for k, err in reversed(self._history):
            if k == key:
                last_err = err
                break
        if last_err is None:
            return None
        count = sum(1 for k, err in self._history if k == key and err == last_err)
        if count >= self.repeat_threshold:
            return last_err
        return None

    def note_world_changed(self) -> int:
        """Forget every recorded failure because a SUCCESSFUL state-mutating
        call just ran (a file write, a service start/stop, a process kill).

        The guard's premise is "re-running the recorded call UNCHANGED will
        fail the same way" — a successful mutation invalidates that premise
        for everything on record, since the failure's cause may be exactly
        what was just changed (the live pathology: `manage_services start`
        failed twice on an occupied port, the model killed the holder via
        shell, and the now-correct retry was blocked pre-dispatch, so the
        guard could never learn the world had moved). Clearing is deliberately
        global rather than per-key: the guard cannot know which keys the
        mutation affected, a false block costs far more than a false allow,
        and ``repeat_threshold`` fresh identical failures re-arm it quickly
        if the change fixed nothing.

        Returns the number of entries dropped so the call site can log only
        when the reset actually mattered."""
        n = len(self._history)
        self._history.clear()
        return n

    def reset(self) -> None:
        self._history.clear()


def guard_key_target(primary_target: str, call_hash: str) -> str:
    """The target half of a :class:`RecentFailureGuard` key.

    When the call has a recognised primary target (path / url / query / …
    via ``primary_target_from_args``) that IS the key. When it has none — e.g.
    ``manage_services``, whose identity lives in ``name``/``port``/
    ``command`` — fall back to a short signature of the FULL canonical call
    (``call_hash`` is the dispatch loop's ``fname:json(args)`` string), so
    a retry with ANY changed argument is a DIFFERENT action. This makes the
    guard's message ("re-running it UNCHANGED will fail the same way")
    literally true: before this fix every ``manage_services start`` — any
    service, any port, any command — collapsed into one key, and legitimately
    changed retries (new port, new command) were blocked (2026-07-30)."""
    if primary_target:
        return primary_target
    if not call_hash:
        return ""
    return "args#" + hashlib.sha1(
        call_hash.encode("utf-8", "replace")).hexdigest()[:12]


# World-mutation shell heuristic for the failure guard's world-changed
# reset. Deliberately narrower than execute.py's _SHELL_MUTATION_RE (whose
# bare `>>?` counts every `2>/dev/null` probe as a mutation — safe for a
# refusal gate, but here it would clear the guard on nearly every probe and
# nullify it) and wider on process mutations, which that regex lacks and
# which were the live remediation (`kill -9`, `fuser -k`). `kill -0` /
# `pkill -0` are liveness PROBES and are excluded.
#
# Two precision rails (2026-07-30 review — probe traffic dominates this
# agent's shell use, so frequent false positives would keep the guard
# permanently disarmed):
#   * mutation VERBS must sit at command position (start of string, after
#     ;|&|(|`|$( separators, or after a runner prefix like sudo/xargs/
#     timeout) — `grep -rn 'mkdir' src/` and `ls release.tar.gz` mention
#     verbs as DATA, not commands;
#   * quoted segments are stripped before matching (`awk '$3 > 100'` is
#     not a redirect), EXCEPT that a `sh -c '…'` payload is itself a
#     command and gets its own scan.
# Residual false positives merely allow one extra real attempt (the
# threshold re-arms on fresh failures); false negatives preserve the
# block — the failure mode being fixed.
_WM_CMD_POS = (
    r"(?:^|[;&|(`]\s*|\$\(\s*"
    r"|\b(?:sudo|nohup|setsid|xargs|exec|env|command|do|then)\s+"
    r"|-exec\s+"
    r"|\btimeout\s+(?:-\S+\s+)*\S+\s+)"
)
_WM_VERBS = (
    r"(?:kill(?:all)?\s+(?!-0\b)"                      # kill/killall, not `kill -0` probes
    r"|pkill\s+(?!-0\b)"                               # pkill, not `pkill -0` probes
    r"|fuser\s+-[A-Za-z]*k\b"                          # fuser -k / -sk … (bare fuser = probe)
    r"|(?:rm|mv|cp|tee|touch|mkdir|rmdir|chmod|chown|truncate|dd|ln|unzip|tar|patch)\b"
    r"|sed\s+-i\b|perl\s+-i\b"
    # service managers only with a mutating subcommand nearby (either order:
    # `systemctl restart nginx` / `service nginx restart`) — `systemctl
    # status` and prose mentions of "service" are observations.
    r"|(?:systemctl|launchctl|service|supervisorctl)\s+(?:[\w./@:-]+\s+){0,2}"
    r"(?:start|stop|restart|kill|reload|enable|disable|bootout|bootstrap|kickstart)\b)"
)
_WORLD_MUTATING_CMD_RE = re.compile(
    _WM_CMD_POS + _WM_VERBS
    + r"|(?<![\d&>])>(?!&)"                            # bare redirect; not 2>/dev/null, 2>&1, or >>'s 2nd char
)
_WM_QUOTED_RE = re.compile(r"'[^']*'|\"[^\"]*\"")
# A nested shell's -c payload IS a command string (the model often wraps:
# `bash -c 'kill -9 …'`). python/node -c payloads are code, not shell —
# only sh/bash/zsh/dash qualify.
_WM_SHELL_C_RE = re.compile(
    r"\b(?:ba|z|da)?sh\s+(?:-\S+\s+)*-c\s+(?:'([^']*)'|\"([^\"]*)\")"
)


def looks_mutating_command(command: str) -> bool:
    """True when a shell command plausibly mutates world state (kills a
    process, moves/removes files, drives a service manager, redirects into
    a file) rather than merely observing it. Consumed by the dispatch loop
    to decide whether a SUCCESSFUL ``execute`` call should clear the
    pre-flight failure guard via ``note_world_changed``."""
    if not command:
        return False
    cmd = str(command)
    for m in _WM_SHELL_C_RE.finditer(cmd):
        inner = m.group(1) or m.group(2) or ""
        if inner and _WORLD_MUTATING_CMD_RE.search(inner):
            return True
    return bool(_WORLD_MUTATING_CMD_RE.search(_WM_QUOTED_RE.sub(" ", cmd)))


__all__ = [
    "RecentFailureGuard",
    "guard_key_target",
    "looks_mutating_command",
    "primary_target_from_args",
]
