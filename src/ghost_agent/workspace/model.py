"""WorkspaceModel — facade the rest of the agent talks to.

Symmetric with ``selfhood.SelfModel``. One attribute on the context
(``context.workspace_model``) gives:
  * a wake-up prefix renderer for the prompt assembly path,
  * a post-event capture hook (task outcomes, research artifacts,
    file-watch results, command outcomes),
  * an idle-phase narrative consolidator.

Disabled mode: when ``enabled=False`` (e.g. ``--no-memory`` or
``--no-workspace-model``), every method is a no-op. Lifespan always
attaches a facade so call sites don't have to branch.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
from pathlib import Path
from typing import Awaitable, Callable, Iterator, List, Optional

from .activity import WorkspaceActivity
from .narrative import WorkspaceNarrative
from .reactions import check_changed_python_files
from .recognition import build_workspace_prefix
from .schema import (
    CommandOutcome,
    ResearchArtifact,
    TaskOutcome,
    WorkspaceEvent,
    _PROJECTS_PATH_RE,
)
from .state import WorkspaceStateThread

logger = logging.getLogger("GhostWorkspace")


CritiqueFn = Callable[[str], Awaitable[str]]

# Task-local override for the project id stamped onto recorded events.
# ``WorkspaceModel.current_project_id`` is ONE process-global attribute
# owned by the chat turn's prompt assembly (serialized by the agent
# semaphore) — but record_* callers also run OUTSIDE that serialization:
# the idle autoadvance tick and the explicit-project autoadvance batch
# both pinned ``context.current_project_id`` (the sandbox-scoping field)
# while events kept stamping from the shared attribute, so an idle
# build's command outcomes landed on whatever project the LAST chat turn
# had open. A ContextVar is task-local under asyncio (and propagates
# into ``asyncio.to_thread``), so each execution context carries its own
# stamp; ``None`` means "no override — fall back to the shared attribute",
# which keeps every existing single-context caller byte-identical.
_EVENT_PROJECT_OVERRIDE: contextvars.ContextVar = contextvars.ContextVar(
    "workspace_event_project", default=None,
)


def set_event_project(project_id: Optional[str]) -> None:
    """Bind the event-stamp project id to the CURRENT asyncio task (and
    everything it awaits). Used by the chat turn right where it syncs
    ``workspace_model.current_project_id``, so a concurrent writer
    mutating the shared attribute mid-turn can't mis-stamp this turn's
    events."""
    _EVENT_PROJECT_OVERRIDE.set(str(project_id or ""))


@contextlib.contextmanager
def pinned_event_project(project_id: Optional[str]) -> Iterator[None]:
    """Scoped variant of :func:`set_event_project` for bounded work
    (idle autoadvance tick, explicit-project autoadvance batch): events
    recorded inside the ``with`` block stamp ``project_id``; the previous
    binding is restored on exit."""
    token = _EVENT_PROJECT_OVERRIDE.set(str(project_id or ""))
    try:
        yield
    finally:
        _EVENT_PROJECT_OVERRIDE.reset(token)




STATE_EXPIRED_LABEL = "expired"


def _origin_now() -> str:
    """§4PF: the origin label for an event recorded right now."""
    try:
        from ..utils.logging import request_origin_label
        return request_origin_label()
    except Exception:  # noqa: BLE001
        return ""


def _probe_turn() -> bool:
    """§4LP: a diagnostic probe is not the owner's work — every other learning
    store skips probe turns, the workspace record did not, and a project's
    continuity text became probe noise (`raise ValueError('boom')`, probe
    pulls)."""
    try:
        from ..utils.logging import (request_id_context, is_probe_request_id,
                                     request_origin_context, ORIGIN_PROBE)
        return (is_probe_request_id(str(request_id_context.get() or ""))
                or str(request_origin_context.get() or "") == ORIGIN_PROBE)
    except Exception:  # noqa: BLE001
        return False

class WorkspaceModel:
    """Top-level workspace facade. Cheap to construct."""

    def __init__(
        self,
        root: Path,
        *,
        enabled: bool = True,
        narrative_critique_fn: Optional[CritiqueFn] = None,
        max_recent_events_for_narrative: int = 12,
    ):
        self.root = Path(root)
        self.enabled = bool(enabled)
        #: §4PE (W3): the sandbox root tracked paths must live under — set
        #: by the owner (main) after construction; None refuses every track
        self.sandbox_root: Optional[Path] = None
        # The project the agent is currently working in ("" = none/free
        # chat). Kept in sync by the agent each request so (a) recorded
        # events are stamped with their owning project and (b) the wake-up
        # prefix scopes events to THIS project — preventing a prior
        # project's file:// pulls from bleeding into a new one.
        self.current_project_id: str = ""
        # Per-session navigation counter (feature 2C): URL → visit count,
        # used to suggest caching / a strategy switch when the agent keeps
        # re-fetching the same page. In-memory, bounded, reset each boot.
        self._nav_counts: dict = {}
        if self.enabled:
            self.activity: Optional[WorkspaceActivity] = WorkspaceActivity(
                self.root, enabled=True,
            )
            self.state: Optional[WorkspaceStateThread] = WorkspaceStateThread(
                self.root, enabled=True,
            )
            self.narrative: Optional[WorkspaceNarrative] = WorkspaceNarrative(
                self.root,
                critique_fn=narrative_critique_fn,
                max_recent_events=max_recent_events_for_narrative,
                enabled=True,
            )
        else:
            self.activity = None
            self.state = None
            self.narrative = None

    # -----------------------------------------------------------------
    # Wake-up path
    # -----------------------------------------------------------------

    def build_wakeup_prefix(self, active_project_id: Optional[str] = None) -> str:
        """Compose the workspace prefix the prompt assembly path
        splices into the system prompt. Empty when there's nothing to
        say.

        ``active_project_id`` (defaults to ``self.current_project_id``)
        scopes the rendered events to the current project so a prior
        project's research artifacts / narrative don't leak in.

        We route through the model's own ``scan_tracked()`` (not the
        state thread's bare scan) so any detected changes are MIRRORED
        into the activity log. That keeps the on-disk record
        consistent regardless of who triggered the scan, and lets the
        ``workspace action=changes`` tool fall back to the recent
        file_changed events when the prefix scan already consumed
        them in this turn."""
        if not self.enabled:
            return ""
        active = active_project_id if active_project_id is not None else self.current_project_id
        narrative_text = self.narrative.latest() if self.narrative else ""
        file_changes: List[dict] = []
        if self.state is not None and self.state.tracked_files():
            try:
                file_changes = self.scan_tracked()
            except Exception as e:  # noqa: BLE001
                logger.debug("workspace scan_tracked failed: %s", e)
        # Scope tracked-file changes the same way: a change to a file under
        # a DIFFERENT project's directory isn't relevant to this turn.
        if active and file_changes:
            _a = str(active).strip().lower()
            scoped = []
            for ch in file_changes:
                m = _PROJECTS_PATH_RE.search(str(ch.get("path", "")))
                if not m or m.group(1).lower() == _a:
                    scoped.append(ch)
            file_changes = scoped
        # React to the (scoped) changes: flag any changed .py file that no
        # longer parses, so the wake-up prefix warns about it (feature 2A).
        file_warnings: List[dict] = []
        if file_changes:
            try:
                file_warnings = check_changed_python_files(file_changes)
            except Exception as e:  # noqa: BLE001
                logger.debug("react_to_changes failed: %s", e)
        return build_workspace_prefix(
            activity=self.activity,
            state=self.state,
            narrative=narrative_text,
            file_changes=file_changes,
            file_warnings=file_warnings,
            active_project_id=active or None,
        )

    # -----------------------------------------------------------------
    # Hot-path capture APIs
    # -----------------------------------------------------------------

    def _stamp_project_id(self) -> str:
        """Project id to stamp onto an event being recorded RIGHT NOW:
        the task-local override when one is bound, else the shared
        ``current_project_id`` attribute."""
        override = _EVENT_PROJECT_OVERRIDE.get()
        if override is not None:
            return override
        return self.current_project_id

    def record_task_outcome(
        self,
        *,
        job_id: str,
        task_name: str = "",
        outcome: str = "unknown",
        duration_seconds: float = 0.0,
        summary: str = "",
        error: str = "",
    ) -> Optional[TaskOutcome]:
        """Append a scheduled-task outcome to the activity log."""
        if not self.enabled or self.activity is None or _probe_turn():     # §4PE (W2)
            return None
        try:
            # record_task_outcome is called when the task FINISHES, so
            # "now" is the finish time. Stamp finished_at explicitly and
            # back-date fired_at by the duration — otherwise fired_at
            # (which defaults to record time) wrongly reads as the finish
            # time and finished_at stayed empty forever.
            import datetime as _dt
            dur = float(duration_seconds or 0.0)
            finished_dt = _dt.datetime.now(_dt.timezone.utc)
            fired_dt = finished_dt - _dt.timedelta(seconds=max(dur, 0.0))

            def _iso(dt):
                return dt.replace(tzinfo=None).isoformat() + "Z"

            t = TaskOutcome(
                job_id=str(job_id or ""),
                task_name=str(task_name or ""),
                outcome=str(outcome or "unknown"),
                duration_seconds=dur,
                fired_at=_iso(fired_dt),
                finished_at=_iso(finished_dt),
                summary=str(summary or "")[:600],
                error=str(error or "")[:400],
            )
            self.activity.append(WorkspaceEvent(
                kind="task_outcome",
                payload=t.to_dict(),
                summary=(
                    f"task {task_name or job_id}: {outcome}"
                    if not error else f"task {task_name or job_id}: failed ({error[:80]})"
                ),
                project_id=self._stamp_project_id(), origin=_origin_now(),
            ))
            return t
        except Exception as e:  # noqa: BLE001
            logger.debug("record_task_outcome skipped: %s", e)
            return None

    def record_research_artifact(
        self,
        *,
        url: str,
        title: str = "",
        source: str = "",
        note: str = "",
    ) -> Optional[ResearchArtifact]:
        """Record a URL we pulled. Idempotent — dedupes against the
        seen-URL set on the state thread so repeated researches don't
        flood the log."""
        if not self.enabled or self.activity is None or self.state is None:
            return None
        if _probe_turn():
            return None
        if not (url or "").strip():
            return None
        try:
            # Dedup gate first, but only COMMIT the URL to the seen-set
            # after the activity append succeeds. Marking seen before the
            # append (the old order) meant that if the append failed the
            # artifact was lost AND every retry was silently deduped until
            # FIFO eviction 5000 URLs later.
            _pid = self._stamp_project_id()
            if self._research_recorded(url, _pid):
                return None  # already on THIS project's record (§4PF W11: the seen-set is global)
            art = ResearchArtifact(
                url=str(url),
                title=str(title or "")[:200],
                source=str(source or ""),
                note=str(note or "")[:200],
            )
            written = self.activity.append(WorkspaceEvent(
                kind="research",
                payload=art.to_dict(),
                summary=f"pulled {url}"[:200],
                project_id=self._stamp_project_id(), origin=_origin_now(),
            ))
            if written is None:
                return None  # append failed → don't poison the dedup set
            self.state.mark_url_seen(url)
            return art
        except Exception as e:  # noqa: BLE001
            logger.debug("record_research_artifact skipped: %s", e)
            return None

    # Host-specific aliases where ONE page has two spellings. Each entry
    # is (host, path-prefix-to-strip); the counter would otherwise treat
    # the two as different pages and never reach its threshold. Kept as
    # an explicit table with ONE consumer (`canonical_nav_url`) — a
    # general rule cannot know that Telegram's `/s/` preview and the bare
    # post are the same content. `tests/test_workspace_nav_canonical.py`
    # walks this table and pins BOTH directions of every entry.
    _NAV_PATH_ALIASES: tuple = (
        # t.me/s/<channel>/<id> is the web PREVIEW of t.me/<channel>/<id>.
        # The 2026-09-15 Revolut turn fetched one post five times across
        # the two spellings (3 + 2) and the nudge never fired, because
        # neither key reached 3 on its own.
        ("t.me", "/s/"),
    )

    @classmethod
    def canonical_nav_url(cls, url: str) -> str:
        """Collapse spellings of the SAME page onto one counter key.

        Normalises scheme/host case, drops the fragment and a trailing
        slash, and applies the host alias table above. The query string
        is deliberately KEPT: `?q=` genuinely changes the page, and a
        counter that merged those would under-report real distinct work.
        """
        raw = (url or "").strip()
        if not raw:
            return ""
        try:
            from urllib.parse import urlsplit, urlunsplit
            parts = urlsplit(raw)
            host = (parts.hostname or "").lower()
            if not host:
                return raw.rstrip("/") or raw
            path = parts.path or ""
            for alias_host, prefix in cls._NAV_PATH_ALIASES:
                if (host == alias_host or host.endswith("." + alias_host)) \
                        and path.startswith(prefix):
                    path = "/" + path[len(prefix):]
                    break
            if len(path) > 1:
                path = path.rstrip("/")
            netloc = host
            if parts.port:
                netloc = f"{host}:{parts.port}"
            return urlunsplit(
                ((parts.scheme or "").lower(), netloc, path, parts.query, ""))
        except Exception:  # noqa: BLE001 — never break a browser turn
            return raw

    def record_navigation(self, url: str, *, threshold: int = 3) -> Optional[str]:
        """Count visits to ``url`` this session; on EXACTLY the
        ``threshold``-th identical visit, return a one-line suggestion to
        cache the result or switch strategy (feature 2C), and record it as a
        workspace note so it also surfaces in the wake-up prefix. Returns
        None otherwise. Fires once (at the threshold) to avoid spamming every
        subsequent re-fetch.
        """
        if _probe_turn():
            return None
        if not self.enabled:
            return None
        # Count the PAGE, not the spelling: `t.me/s/x/1` and `t.me/x/1`
        # share a key, so five fetches split 3/2 across them still trip
        # the nudge at the third.
        u = self.canonical_nav_url(url)
        if not u:
            return None
        # pop+reinsert moves u to the end (most-recently-touched), so the
        # bound below evicts by RECENCY. Plain `get`+set kept insertion
        # order, so a hot URL touched early could be evicted mid-crawl and
        # have its counter reset — breaking the fire-once-at-threshold
        # contract.
        n = self._nav_counts.pop(u, 0) + 1
        self._nav_counts[u] = n
        # Bound the map so a long crawl can't grow it without limit. Evict
        # the least-recently-touched entry (front); never the current one.
        if len(self._nav_counts) > 512:
            oldest = next(iter(self._nav_counts))
            if oldest != u:
                self._nav_counts.pop(oldest, None)
        if n == int(threshold):
            suggestion = (
                f"You've navigated to {u} {n} times this session. If you're "
                f"re-checking the same page for a change or an error, cache "
                f"the last result or switch strategy instead of re-fetching it."
            )
            # §4PF (W13): a per-session nudge is returned to the caller, never
            # persisted as a project event
            return suggestion
        return None

    def has_seen_url(self, url: str) -> bool:
        if not self.enabled or self.state is None:
            return False
        try:
            return self.state.has_seen_url(url)
        except Exception:  # noqa: BLE001
            return False

    def _research_recorded(self, url: str, project_id: str) -> bool:
        """§4PF (W11): has THIS project's record already got this URL? The
        seen-URL set is global, so a URL pulled in project A was silently not
        recorded when pulled in project B."""
        key = str(url or "").strip().rstrip("/").lower()
        try:
            # the reader filters by project while scanning (§4PF r4), so another
            # project's rows cannot push this project's out of the window
            for ev in self.activity.recent(limit=5000, kind="research", project_id=project_id or ""):
                if str((ev.payload or {}).get("url") or "").strip().rstrip("/").lower() == key:
                    return True
        except Exception:  # noqa: BLE001
            return False
        return False

    def record_job_finished(self, entry: dict) -> Optional[CommandOutcome]:
        """§4PF (W7): a promoted command's REAL outcome, booked when the job
        reaper lands it — the ledger used to keep "exit=0 — still running"
        forever. Rendered rows hide the superseded promotion line."""
        try:
            from ..utils.logging import event_origin_context, request_origin_label
            jid = str((entry or {}).get("id") or "")
            code = (entry or {}).get("exit_code")
            dur = float((entry or {}).get("finished_at") or 0) - float((entry or {}).get("started_at") or 0)
            state = str((entry or {}).get("state") or "")
            # §4PF r4: the row belongs to the JOB's project and starter, not to
            # whoever observed the landing (the 60 s sweeper, the owner's next
            # execute); a probe's job leaves no row at all
            origin = str((entry or {}).get("origin") or request_origin_label(str((entry or {}).get("started_by") or "")))
            if origin == "probe":
                return None
            if state == STATE_EXPIRED_LABEL:
                note = f"background job {jid} EXPIRED — killed at its time limit"
            elif state == "cancelled":
                note = f"background job {jid} cancelled on request"
            elif code is None:
                note = f"background job {jid} {state or 'lost'} (no exit code)"
            else:
                note = f"background job {jid} finished"
            tok = event_origin_context.set(origin)
            try:
                with pinned_event_project(str((entry or {}).get("project_id") or "")):
                    return self._record_command_outcome_unguarded(
                        command=str((entry or {}).get("command") or ""),
                        exit_code=int(code) if code is not None else 1,
                        duration_seconds=max(0.0, dur), note=note, job_id=jid)
            finally:
                event_origin_context.reset(tok)
        except Exception as e:  # noqa: BLE001
            logger.debug("record_job_finished skipped: %s", e)
            return None

    def record_command_outcome(
        self,
        *,
        command: str,
        exit_code: int = 0,
        duration_seconds: float = 0.0,
        note: str = "",
        job_id: str = "",
    ) -> Optional[CommandOutcome]:
        """Capture a significant command outcome (long, failed, or
        mutating). Caller decides what's significant — we just write."""
        if _probe_turn():
            return None
        return self._record_command_outcome_unguarded(command=command, exit_code=exit_code,
                                                      duration_seconds=duration_seconds, note=note, job_id=job_id)

    def _record_command_outcome_unguarded(self, *, command, exit_code=0, duration_seconds=0.0, note="", job_id=""):
        """The write itself — the probe gate is the caller's (a job landing is
        judged by the job's starter, not by the reaper's context)."""
        if not self.enabled or self.activity is None:
            return None
        try:
            c = CommandOutcome(
                command=str(command or "")[:400],
                exit_code=int(exit_code),
                duration_seconds=float(duration_seconds or 0.0),
                note=str(note or "")[:200],
            )
            # The NOTE rides in the summary. The readers of this ledger take
            # `summary` and nothing else — rendered by tools/workspace.py,
            # workspace/narrative.py and workspace/recognition.py, indexed by
            # workspace/activity.py, and scanned for a project id by
            # workspace/schema.py (which cannot match on a note). So a note
            # was write-only, and a command that
            # is still running as a detached job (exit_code 0 by convention)
            # rendered as a completed success. "ran X exit=0" is not the
            # whole truth whenever the caller took the trouble to add one.
            _summary = f"ran `{(command or '')[:80]}` exit={exit_code}"
            if c.note:
                _summary += f" — {c.note}"
            self.activity.append(WorkspaceEvent(
                kind="command",
                payload={**c.to_dict(), **({"job_id": str(job_id)} if job_id else {})},
                summary=_summary,
                project_id=self._stamp_project_id(), origin=_origin_now(),
            ))
            return c
        except Exception as e:  # noqa: BLE001
            logger.debug("record_command_outcome skipped: %s", e)
            return None

    def note(self, summary: str, **payload) -> Optional[WorkspaceEvent]:
        """Generic free-form workspace event. Used by the workspace_track
        tool when the agent wants to record something that isn't a
        file / task / research / command outcome."""
        if not self.enabled or self.activity is None or _probe_turn():     # §4PE (W2)
            return None
        summary = (summary or "").strip()
        if not summary:
            return None
        try:
            ev = WorkspaceEvent(
                kind="note", payload=dict(payload), summary=summary[:600],
                project_id=self._stamp_project_id(), origin=_origin_now(),
            )
            self.activity.append(ev)
            return ev
        except Exception as e:  # noqa: BLE001
            logger.debug("note skipped: %s", e)
            return None

    # -----------------------------------------------------------------
    # File watching
    # -----------------------------------------------------------------

    def track_file(self, path: str, *, label: str = ""):
        if not self.enabled or self.state is None or _probe_turn():        # §4PE (W2)
            return None
        host = self.workspace_host_path(path)                              # §4PE (W3)
        if host is None:
            return None
        return self.state.track_file(host, label=label)

    def workspace_host_path(self, path: str) -> Optional[str]:
        """§4PE (W3): the HOST path of a workspace file named the way the
        model names files — `/workspace/...` (the container's view), or a
        path relative to the sandbox root — and None for anything outside
        the sandbox. `track` watched host paths verbatim: the model's
        container paths never resolved, a relative one watched the server's
        CWD, and `~/.ssh` was trackable (an existence/mtime oracle, and a
        changed `.py` there was read and parsed)."""
        p = os.path.expanduser(str(path or "").strip())        # `~/…` names the host's home: outside
        if not p:
            return None
        root = getattr(self, "sandbox_root", None)
        if root is None:
            return None
        root = Path(root)
        p = p.replace("\\", "/")
        if p == "/workspace" or p.startswith("/workspace/"):
            p = p[len("/workspace"):].lstrip("/")
        elif p.startswith("workspace/"):
            p = p[len("workspace/"):]
        if not Path(p).is_absolute() and self.current_project_id:         # §4PE r3: the model's cwd is the
            cand = root / "projects" / self.current_project_id / p         # project workspace when scoped
        else:
            cand = Path(p) if Path(p).is_absolute() else root / p
        try:
            cand_r = cand.resolve()
            root_r = root.resolve()
        except OSError:
            return None
        if cand_r != root_r and root_r not in cand_r.parents:
            return None
        if cand_r.exists() and not cand_r.is_file():
            return None
        return str(cand_r)

    def display_path(self, host_path: str) -> str:
        """The workspace-relative spelling of a tracked host path."""
        root = getattr(self, "sandbox_root", None)
        try:
            if root is not None:
                return "/workspace/" + str(Path(host_path).resolve().relative_to(Path(root).resolve()))
        except Exception:  # noqa: BLE001
            pass
        return str(host_path)

    def untrack_file(self, path: str) -> bool:
        if not self.enabled or self.state is None:
            return False
        host = self.workspace_host_path(path)                              # §4PE r3: the spelling `track` echoes
        if host and self.state.untrack_file(host):
            return True
        return self.state.untrack_file(path)

    def scan_tracked(self) -> List[dict]:
        if not self.enabled or self.state is None:
            return []
        if _probe_turn():                                                   # §4PE (W2): a read-only probe
            try:                                                            # sees the diff, books nothing
                return self.state.scan_tracked()
            except Exception:  # noqa: BLE001
                return []
        try:
            changes = self.state.scan_tracked()
        except Exception as e:  # noqa: BLE001
            logger.debug("scan_tracked failed: %s", e)
            return []
        # Mirror each change into the activity log so the narrative
        # consolidation can reason over it.
        if changes and self.activity is not None:
            for ch in changes:
                try:
                    # Derive the owning project from the changed path when
                    # possible (a tracked file lives under projects/<id>/),
                    # else fall back to the active project.
                    _m = _PROJECTS_PATH_RE.search(str(ch.get("path", "")))
                    _pid = _m.group(1).lower() if _m else self._stamp_project_id()
                    self.activity.append(WorkspaceEvent(
                        kind="file_changed",
                        payload=ch,
                        summary=f"{ch.get('path')}: {ch.get('change')}",
                        project_id=_pid, origin=_origin_now(),
                    ))
                except Exception:  # noqa: BLE001
                    continue
        return changes

    def mark_session_boot(self) -> None:
        if not self.enabled or self.state is None:
            return
        try:
            self.state.touch_session()
        except Exception as e:  # noqa: BLE001
            logger.debug("mark_session_boot skipped: %s", e)

    # -----------------------------------------------------------------
    # Idle path
    # -----------------------------------------------------------------

    async def consolidate_narrative(self) -> str:
        if not self.enabled or self.activity is None or self.narrative is None:
            return ""
        # Roll recent activity into a dated CHANGELOG alongside the prose
        # narrative (feature 2B). Best-effort — never blocks the narrative.
        try:
            self.write_changelog()
        except Exception as e:  # noqa: BLE001
            logger.debug("workspace changelog write skipped: %s", e)
        try:
            return await self.narrative.regenerate(
                activity=self.activity, state=self.state,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("workspace narrative consolidation failed: %s", e)
            return ""

    def write_changelog(self, *, max_events: int = 200) -> Optional[Path]:
        """Render recent activity into a per-project CHANGELOG.md under the
        workspace root and return its path (feature 2B). No-op (None) when
        disabled, no activity, or nothing to report.
        """
        if not self.enabled or self.activity is None:
            return None
        from .narrative import render_changelog
        events = self.activity.recent(limit=max_events)
        active = self.current_project_id or ""
        body = render_changelog(events, active_project_id=active or None)
        if not body:
            return None
        # Scope the file per active project so projects don't share one log.
        name = f"CHANGELOG.{active}.md" if active else "CHANGELOG.md"
        path = self.root / name
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".md.tmp")
            tmp.write_text(body, encoding="utf-8")
            tmp.replace(path)
            return path
        except Exception as e:  # noqa: BLE001
            logger.debug("changelog persist failed: %s", e)
            return None

    # -----------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------

    def search_events(self, query: str, *, limit: int = 10) -> List["object"]:
        """Relevance-ranked keyword search over the activity log — the
        workspace counterpart to selfhood's ``search_my_past``. Empty on
        disabled / no activity / no match. Never raises."""
        if not self.enabled or self.activity is None:
            return []
        try:
            return self.activity.search(query, limit=limit)
        except Exception as e:  # noqa: BLE001
            logger.debug("search_events failed: %s", e)
            return []

    def stats(self) -> dict:
        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "root": str(self.root),
            "tracked_files": len(self.state.tracked_files()) if self.state else 0,
            "seen_urls": (
                len(self.state.state.seen_urls) if self.state else 0
            ),
            "event_count": self.activity.count() if self.activity else 0,
            "event_kinds": self.activity.kinds() if self.activity else {},
            "narrative_present": (
                bool(self.narrative.latest()) if self.narrative else False
            ),
            "last_session_at": (
                self.state.state.last_session_at if self.state else ""
            ),
            "prior_session_at": (
                self.state.state.prior_session_at if self.state else ""
            ),
        }
