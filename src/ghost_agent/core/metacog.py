"""Metacognition bundle — the single object every wire-point consults.

The uplift modules (host telemetry, validators, triggers, replan-bridge)
are designed to be INDEPENDENT. They don't import each other. The bundle is the glue
that lives on ``GhostContext.metacog`` so call sites elsewhere in the
codebase can ask "is the uplift on?" with a single attribute check
instead of poking at six different `None` checks.

Lifecycle: ``MetacogBundle.from_args(context, args)`` constructs the
bundle from the parsed CLI args. ``None`` is returned when
``--enable-metacog`` is off, so call sites read::

    if context.metacog:
        context.metacog.record_outcome(...)

If the attribute is absent or ``None``, nothing the uplift adds
fires — that's how this lands on the live agent without changing
any existing test's behaviour.

Why a bundle and not a thousand fields on ``GhostContext``? Because
the uplift is one feature: turning a single switch on or off should
not leak across the codebase's attribute namespace. The bundle also
keeps the lifespan code (which constructs *and* tears down all eight
modules) in one obvious place.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("GhostAgent")


@dataclass
class MetacogBundle:
    """Container for the uplift modules + their config.

    Every field except ``enabled`` is initialised lazily by
    ``from_args`` so the import surface stays tight.

    §4PJ (2026-10-16): the composite confidence score, its inputs (the
    per-domain competence profile, per-token logprob entropy), the
    calibration refit and the dual-solver arbiter were removed — the score
    did not rank outcomes (AUC 0.48) and nothing read it once the verifier
    judge went off.
    """

    enabled: bool = False

    # Lazily filled by ``from_args``
    telemetry: Optional[Any] = None          # utils.telemetry.HostTelemetry
    bus: Optional[Any] = None                # core.triggers.TriggerBus
    bridge: Optional[Any] = None             # core.triggers.ReplanBridge
    runtime_budget: Optional[Any] = None     # core.triggers.ToolRuntimeBudget
    repetition: Optional[Any] = None         # core.triggers.RepetitionCounter

    # ------------------------------------------------------------------

    @classmethod
    def from_args(cls, context: Any, args: Any) -> Optional["MetacogBundle"]:
        """Construct from parsed CLI args. Returns ``None`` when the
        uplift is disabled or when imports fail (graceful degradation
        — a missing optional dep should not crash the agent).
        """
        if not getattr(args, "enable_metacog", False):
            return None
        try:
            from ..utils.telemetry import HostTelemetry
            from ..core.triggers import (
                TriggerBus, ReplanBridge, ToolRuntimeBudget, RepetitionCounter,
            )
        except Exception as exc:
            logger.warning("Metacog import failed: %s — uplift disabled", exc)
            return None

        bundle = cls(enabled=True)
        bundle.bus = TriggerBus()
        # HostTelemetry thresholds are operator-tunable — see the
        # --metacog-cpu-high / --metacog-mem-high / etc. CLI flags.
        # We pass through whatever the args namespace carries; missing
        # attributes fall back to the HostTelemetry class defaults so
        # tests that construct a bare Namespace still work.
        bundle.telemetry = HostTelemetry(
            cpu_high=float(getattr(args, "metacog_cpu_high",
                                   HostTelemetry.DEFAULT_CPU_HIGH)),
            mem_high=float(getattr(args, "metacog_mem_high",
                                   HostTelemetry.DEFAULT_MEM_HIGH)),
            mem_floor_mb=float(getattr(args, "metacog_mem_floor_mb",
                                       HostTelemetry.DEFAULT_MEM_FLOOR_MB)),
            mem_warn_free_mb=float(getattr(args, "metacog_mem_warn_free_mb",
                                           HostTelemetry.DEFAULT_MEM_WARN_FREE_MB)),
            disk_high=float(getattr(args, "metacog_disk_high",
                                    HostTelemetry.DEFAULT_DISK_HIGH)),
            heartbeat_s=float(getattr(args, "metacog_host_heartbeat_s", 300.0)),
        )
        bundle.runtime_budget = ToolRuntimeBudget()
        bundle.repetition = RepetitionCounter()

        # ReplanBridge pulls plan + current-task IDs from getters so it
        # tolerates the (common) state where no project is active.
        def _plan_getter():
            store = getattr(context, "project_store", None)
            proj_id = getattr(context, "current_project_id", None)
            if store is None or not proj_id:
                return None
            try:
                from .planning import ProjectPlan
                return ProjectPlan(store, proj_id)
            except Exception:
                return None

        def _task_getter():
            # The agent doesn't expose a hot "active task id" attribute
            # globally; metacog stashes one on the bundle as turns flow
            # through. Falling back to None is fine — ReplanBridge
            # records a noop:no_plan event in that case.
            return bundle._active_task_id

        bundle.bridge = ReplanBridge(
            bundle.bus,
            plan_getter=_plan_getter,
            current_task_getter=_task_getter,
            # Feeds the shutdown-summary replans_tried/replans_ok
            # counters — without this hook the bridge revised tasks but
            # the summary always printed replans 0/0.
            counter_hook=bundle.count,
        )
        bundle.bridge.attach()

        return bundle

    # ------------------------------------------------------------------

    _active_task_id: Optional[str] = field(default=None, repr=False)

    # ── Lifetime counters (for the shutdown summary) ─────────────────
    # Operators ask "did the uplift do anything useful today?" — these
    # counters back a single summary line on shutdown so the answer is
    # one log entry, not a grep over the whole session.
    _ctr_host_signals: int = field(default=0, repr=False)
    _ctr_host_critical: int = field(default=0, repr=False)
    _ctr_replan_attempts: int = field(default=0, repr=False)
    _ctr_replan_succeeded: int = field(default=0, repr=False)
    _ctr_validator_blocks: int = field(default=0, repr=False)

    def set_active_task(self, task_id: Optional[str]) -> None:
        """Stashes the current task id so ReplanBridge can attribute
        a triggered replan to a specific task. Pass ``None`` at the
        end of a turn so a late-firing trigger doesn't replan a
        completed task."""
        self._active_task_id = task_id

    async def shutdown(self) -> None:
        """Stop background tasks (the telemetry poller). Idempotent.

        Emits a single ``metacog summary`` line so the operator sees
        the lifetime activity of the uplift in one place — no grep
        needed to know "did anything fire today".
        """
        # Compose summary BEFORE tearing things down so we still have
        # the counter state. Use _emit_summary so logging failures
        # don't block the shutdown.
        self._emit_summary()
        if self.telemetry is not None:
            try:
                await self.telemetry.stop()
            except Exception as exc:
                logger.debug("Metacog shutdown: telemetry.stop failed: %s", exc)
        if self.bridge is not None:
            try:
                self.bridge.detach()
            except Exception as exc:
                logger.debug("Metacog shutdown: bridge.detach failed: %s", exc)

    def _emit_summary(self) -> None:
        """One-line lifetime summary. Fields are ordered so the most
        operationally-interesting numbers come first."""
        try:
            from .metacog_log import emit, Subsystem, LEVEL_INFO
            emit(
                Subsystem.SUMMARY,
                level=LEVEL_INFO,
                validator_blocks=self._ctr_validator_blocks,
                host_signals=self._ctr_host_signals,
                host_critical=self._ctr_host_critical,
                replans_ok=self._ctr_replan_succeeded,
                replans_tried=self._ctr_replan_attempts,
            )
        except Exception as exc:
            logger.debug("Metacog summary emit failed: %s", exc)

    # ── Counter API (called by integration sites) ────────────────────
    # Public-but-cheap: a no-op when the bundle is disabled. Each call
    # site can use the same method without checking ``self.enabled``
    # first.

    def count(self, *, host_signal: bool = False,
              host_critical: bool = False,
              replan_attempt: bool = False,
              replan_succeeded: bool = False,
              validator_block: bool = False) -> None:
        if not self.enabled:
            return
        if host_signal: self._ctr_host_signals += 1
        if host_critical: self._ctr_host_critical += 1
        if replan_attempt: self._ctr_replan_attempts += 1
        if replan_succeeded: self._ctr_replan_succeeded += 1
        if validator_block: self._ctr_validator_blocks += 1

    def record_outcome(self, tool_name: str, success: bool,
                       duration_s: Optional[float] = None) -> None:
        """One-call hook for the tool-outcome site: feeds the runtime
        budget (the ExecutionAnomaly trigger's p95 window). Safe no-op when
        it is missing. ``success`` is kept for the call site's shape; the
        per-domain competence profile it used to feed was removed (§4PJ).
        """
        if duration_s is not None and self.runtime_budget is not None:
            try:
                self.runtime_budget.record(tool_name, float(duration_s))
            except Exception as exc:
                logger.debug("Metacog runtime_budget.record failed: %s", exc)


# ──────────────────────────────────────────────────────────────────────
# Tool-name → domain map.
# ──────────────────────────────────────────────────────────────────────

_TOOL_DOMAIN: dict = {
    # shell
    "execute": "shell",
    "shell": "shell",
    "command": "shell",
    # sql
    "postgres_admin": "sql",
    "database": "sql",
    "sql": "sql",
    # fetch / web
    "web_search": "fetch",
    "fast_url": "fetch",
    "url_fetch": "fetch",
    "tor_browse": "fetch",
    "browser": "fetch",
    "deep_research": "fetch",
    # filesystem
    "file_system": "fs",
    "list_files": "fs",
    "read_file": "fs",
    "write_file": "fs",
    "search_filesystem": "fs",
    # memory
    "skill_memory": "memory",
    "episodic_memory": "memory",
    "graph_memory": "memory",
    "vector_memory": "memory",
    "manage_projects": "memory",
    "smart_update": "memory",
    "update_profile": "memory",
    "scratchpad": "memory",
    # vision / image
    "vision": "vision",
    "image_generation": "vision",
    "image_gen": "vision",
}


def _domain_for_tool(tool_name: str) -> str:
    t = (tool_name or "").strip().lower()
    return _TOOL_DOMAIN.get(t, "other")


__all__ = ["MetacogBundle"]
