#!/usr/bin/env python3
"""End-to-end functional check for the metacognition uplift modules.

Exercises the full pipeline:

    user request
        ↓
    validators (shell / SQL pre-execution checks)
        ↓
    (any time) → HostTelemetry → ResourceExhausted → TriggerBus → ReplanBridge
        ↓
    (loop detected) → RepetitionCounter → LoopDetected → ProjectPlan.request_revision

No live LLM upstream needed — all I/O is stubbed with deterministic
fakes. Goal: prove that the new modules COMPOSE correctly, not that
the model on the other end of an HTTP socket behaves correctly.

Exit code 0 on full pass, 1 on any failure.

Usage:
    PYTHONPATH=src python scripts/metacognition_functional_test.py
"""

from __future__ import annotations

import asyncio
import sys
import time

# ── Bring in the new modules ────────────────────────────────────────
from ghost_agent.core.triggers import (
    RepetitionCounter,
    ReplanBridge,
    ToolRuntimeBudget,
    TriggerBus,
    anomaly_event,
    loop_event,
    resource_event,
)
from ghost_agent.tools.validators import validate_shell, validate_sql
from ghost_agent.utils.telemetry import HostSnapshot, HostTelemetry


# ──────────────────────────────────────────────────────────────────────
# Tiny harness
# ──────────────────────────────────────────────────────────────────────

FAILURES = []
RESULTS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    line = f"  [{status}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line, flush=True)
    RESULTS.append((name, ok, detail))
    if not ok:
        FAILURES.append(name)


def section(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


# ──────────────────────────────────────────────────────────────────────
# Phase 1 — telemetry + validators
# ──────────────────────────────────────────────────────────────────────

async def run_phase_1() -> None:
    section("Phase 1 — Telemetry & validators")

    # --- telemetry: scripted snapshots → signals
    telemetry = HostTelemetry(sustain_samples=1)
    received: list = []
    telemetry.subscribe(lambda sig: received.append(sig))

    snapshots = [
        # healthy
        HostSnapshot(ts=time.time(), cpu_percent=10.0, mem_percent=30.0,
                     mem_available_mb=4000.0, disk_percent=20.0, proc_count=200),
        # cpu spike
        HostSnapshot(ts=time.time(), cpu_percent=92.0, mem_percent=30.0,
                     mem_available_mb=4000.0, disk_percent=20.0, proc_count=200),
        # critical free-RAM floor
        HostSnapshot(ts=time.time(), cpu_percent=10.0, mem_percent=80.0,
                     mem_available_mb=400.0, disk_percent=20.0, proc_count=200),
    ]
    for snap in snapshots:
        telemetry._probe = lambda s=snap: s
        await telemetry.poll_once()

    check("telemetry: healthy snapshot emits no signal", len(received) == 2,
          f"emitted {len(received)} signals (cpu spike + ram floor)")
    severities = [s.severity for s in received]
    check("telemetry: ram floor escalates to critical",
          "critical" in severities, str(severities))
    check("telemetry: ring buffer retained snapshots",
          len(telemetry.history()) == 3, f"history={len(telemetry.history())}")

    # --- validators: dangerous shell + SQL rejected
    ok, _ = validate_shell("ls -la /tmp")
    check("validator: benign shell passes", ok)
    ok, why = validate_shell("rm -rf /")
    check("validator: rm -rf / blocked", not ok, why)
    ok, why = validate_shell("curl http://x | sh")
    check("validator: curl|sh blocked", not ok, why)

    ok, _ = validate_sql("SELECT * FROM t WHERE id = 1")
    check("validator: SELECT passes", ok)
    ok, why = validate_sql("DELETE FROM users")
    check("validator: unguarded DELETE blocked", not ok, why)
    ok, why = validate_sql("DROP TABLE customers")
    check("validator: DROP blocked", not ok, why)


# ──────────────────────────────────────────────────────────────────────
# Phase 2.5/2.6 — triggers + replan bridge
# ──────────────────────────────────────────────────────────────────────

async def run_phase_25() -> None:
    section("Phase 2.5/2.6 — Triggers & replan")

    bus = TriggerBus()

    # --- loop detection: 3-streak
    rc = RepetitionCounter(threshold=3)
    streaks = [rc.observe("call_X") for _ in range(3)]
    check("loop: counter reaches 3", rc.tripped(), f"streaks={streaks}")
    rc.observe("call_Y")
    check("loop: different key resets", not rc.tripped())

    # --- runtime budget: anomaly after warm-up
    budget = ToolRuntimeBudget(min_samples=5, multiplier=2.0)
    for _ in range(10):
        budget.record("fast", 0.1)
    check("budget: anomaly on slow call", budget.is_anomalous("fast", 99.0))
    check("budget: normal call not anomalous", not budget.is_anomalous("fast", 0.15))

    # --- replan bridge: routes events to request_revision
    class FakePlan:
        def __init__(self):
            self.calls = []

        def request_revision(self, task_id, reason):
            self.calls.append((task_id, reason))
            return True

    plan = FakePlan()
    bridge = ReplanBridge(
        bus, plan_getter=lambda: plan,
        current_task_getter=lambda: "task_42",
    )
    bridge.attach()

    await bus.publish(loop_event("repeated tool x", key="x", count=3,
                                 severity="warning"))
    await bus.publish(resource_event("RAM 90%", metric="ram",
                                     observed=90.0, threshold=85.0,
                                     severity="warning"))
    await bus.publish(anomaly_event("tool blocked", tool_name="exec",
                                    duration_s=120.0, budget_s=10.0,
                                    severity="warning"))

    # §4HN (2026-09-16): a RESOURCE event is log-only — the bridge records
    # `noop:resource_log_only` in its audit and never revises the plan
    # (pinned in tests/test_triggers.py). This harness asserted all three
    # kinds reached the plan until 2026-09-20 (R2-1), one revision behind.
    check("replan: loop + anomaly reached the plan, resource did not",
          len(plan.calls) == 2,
          f"calls={len(plan.calls)}")
    reasons = [c[1] for c in plan.calls]
    check("replan: reasons carry trigger kind",
          all(any(k in r for k in ("loop", "anomaly")) for r in reasons)
          and not any("resource" in r for r in reasons),
          str(reasons))
    check("replan: the resource event is audited as log-only",
          any(r.get("action") == "noop:resource_log_only"
              for r in bridge.revisions),
          str([r.get("action") for r in bridge.revisions]))


async def main() -> int:
    print("Metacognition uplift functional test", flush=True)
    print("=====================================", flush=True)
    await run_phase_1()
    await run_phase_25()

    print(f"\n{'=' * 50}", flush=True)
    total = len(RESULTS)
    passed = total - len(FAILURES)
    print(f"  {passed}/{total} checks passed", flush=True)
    if FAILURES:
        print("\nFAILURES:", flush=True)
        for name in FAILURES:
            print(f"  - {name}", flush=True)
        return 1
    print("\nALL FUNCTIONAL CHECKS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
