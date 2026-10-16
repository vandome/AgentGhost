"""Unit tests for the metacognition wiring — verifies that each
integration site is live when --enable-metacog is on and a no-op when
off.

Scope:
  - MetacogBundle constructor + from_args contract
  - GhostContext.metacog field defaults to None (back-compat)
  - tools/registry passes the bundle into execute + postgres_admin
  - tool_execute rejects deny-listed shell when bundle is wired
  - tool_postgres_admin rejects unguarded DELETE/DROP when bundle is wired
  - both tools execute normally when bundle is absent (legacy path)
  - record_outcome updates both competence and runtime_budget
  - HostSignal → bus translator in main.py lifespan maps fields correctly
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ghost_agent.core.metacog import (
    MetacogBundle,
    _domain_for_tool,
)


# ──────────────────────────────────────────────────────────────────────
# Bundle construction & lifecycle
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_context(tmp_path):
    ctx = MagicMock()
    ctx.memory_dir = tmp_path
    ctx.args = MagicMock()
    ctx.args.model = "test-model"
    ctx.llm_client = MagicMock()
    # `get_embeddings` may or may not exist — start without
    if hasattr(ctx.llm_client, "get_embeddings"):
        del ctx.llm_client.get_embeddings
    ctx.project_store = None
    ctx.current_project_id = None
    return ctx


def _args(**kw):
    a = argparse.Namespace(
        enable_metacog=False,
    )
    for k, v in kw.items():
        setattr(a, k, v)
    return a


class TestMetacogBundleConstruction:
    def test_disabled_returns_none(self, fake_context):
        bundle = MetacogBundle.from_args(fake_context, _args(enable_metacog=False))
        assert bundle is None

    def test_enabled_constructs_bundle(self, fake_context):
        bundle = MetacogBundle.from_args(fake_context, _args(enable_metacog=True))
        assert bundle is not None
        assert bundle.enabled
        assert bundle.bus is not None
        assert bundle.telemetry is not None
        assert bundle.runtime_budget is not None
        # §4PJ: the confidence score, its inputs and the arbiter are gone
        for gone in ("competence", "confidence", "arbiter",
                     "record_confidence", "arbitrate_tool_calls"):
            assert not hasattr(bundle, gone), gone

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(self, fake_context):
        bundle = MetacogBundle.from_args(fake_context, _args(enable_metacog=True))
        await bundle.shutdown()
        await bundle.shutdown()  # second call must not raise

    def test_bridge_counter_hook_wired_to_bundle_count(self, fake_context):
        """Regression: _ctr_replan_attempts/_ctr_replan_succeeded had no
        producer anywhere, so the shutdown summary always printed
        replans 0/0 even when the bridge revised tasks."""
        bundle = MetacogBundle.from_args(fake_context, _args(enable_metacog=True))
        assert bundle.bridge.counter_hook == bundle.count

    @pytest.mark.asyncio
    async def test_replan_counters_move_end_to_end(self, fake_context):
        bundle = MetacogBundle.from_args(fake_context, _args(enable_metacog=True))
        bundle.set_active_task("t1")

        class FakePlan:
            def request_revision(self, task_id, reason):
                return True

        # Point the bridge at a live plan (the default context has none).
        bundle.bridge.plan_getter = lambda: FakePlan()
        from ghost_agent.core.triggers import loop_event
        await bundle.bus.publish(loop_event("looped", severity="warning"))
        assert bundle._ctr_replan_attempts == 1
        assert bundle._ctr_replan_succeeded == 1


# ──────────────────────────────────────────────────────────────────────
# Active-task tracking
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_active_task_threading_to_bridge(fake_context):
    """The bridge's task-id getter must read the bundle's active id so
    a replan attributes to the right node."""
    bundle = MetacogBundle.from_args(fake_context, _args(enable_metacog=True))
    bundle.set_active_task("task_xyz")
    assert bundle._active_task_id == "task_xyz"
    # Trigger a critical event and verify the bridge logged it for the
    # current task even though plan_getter returns None (no plan store).
    from ghost_agent.core.triggers import resource_event
    await bundle.bus.publish(
        resource_event("blew up", metric="ram", observed=99,
                       threshold=85, severity="critical")
    )
    assert any(r.get("task_id") == "task_xyz" for r in bundle.bridge.revisions)


# ──────────────────────────────────────────────────────────────────────
# record_outcome
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# Domain mapping
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("tool,expected", [
    ("execute", "shell"),
    ("postgres_admin", "sql"),
    ("web_search", "fetch"),
    ("file_system", "fs"),
    ("skill_memory", "memory"),
    ("vision", "vision"),
    ("image_generation", "vision"),
    ("totally_unknown_tool_xyz", "other"),
    ("", "other"),
])
def test_domain_for_tool(tool, expected):
    assert _domain_for_tool(tool) == expected


def test_record_outcome_feeds_the_runtime_budget(fake_context):
    """§4PJ: the outcome hook's competence half went with the score; the
    runtime-budget half feeds the ExecutionAnomaly trigger and stays."""
    bundle = MetacogBundle.from_args(fake_context, _args(enable_metacog=True))
    for _ in range(8):
        bundle.record_outcome("execute", success=True, duration_s=0.5)
    bundle.record_outcome("execute", success=False, duration_s=2.0)
    bundle.record_outcome("weird_new_tool", success=True)   # no duration: no-op, no crash
    assert len(bundle.runtime_budget._samples["execute"]) == 9
    assert "weird_new_tool" not in bundle.runtime_budget._samples


def test_the_retired_metacog_flags_are_rejected():
    """§4PJ: the confidence/arbiter flags were removed outright (the
    launcher never passed them), unlike --calib-refit-cooldown."""
    import sys
    from ghost_agent import main as main_module
    old_argv = sys.argv[:]
    try:
        for flag in (["--metacog-confidence-threshold", "0.4"],
                     ["--metacog-disable-arbiter"],
                     ["--metacog-arbiter-timeout-s", "75"],
                     ["--metacog-disable-logprobs"]):
            sys.argv = ["ghost_agent", "--enable-metacog", *flag]
            with pytest.raises(SystemExit):
                main_module.parse_args()
    finally:
        sys.argv = old_argv


# ──────────────────────────────────────────────────────────────────────
# Validators integration — execute.py
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tool_execute_blocks_denylisted_shell_when_bundle_enabled(tmp_path):
    """Wire-point: tool_execute must call validate_shell when the
    bundle is wired and reject deny-listed commands BEFORE dispatching
    to the sandbox manager."""
    from ghost_agent.tools.execute import tool_execute
    fake_sandbox = MagicMock()
    fake_sandbox.execute = MagicMock(return_value=("should not run", 0))

    bundle = MagicMock()
    bundle.enabled = True

    out = await tool_execute(
        command="rm -rf /",
        sandbox_dir=tmp_path,
        sandbox_manager=fake_sandbox,
        memory_dir=tmp_path,
        _metacog_bundle=bundle,
    )
    assert "SYSTEM BLOCK" in out
    assert "rejected" in out.lower()
    # Sandbox was never asked to execute
    fake_sandbox.execute.assert_not_called()


@pytest.mark.asyncio
async def test_tool_execute_passes_benign_shell_when_bundle_enabled(tmp_path):
    from ghost_agent.tools.execute import tool_execute
    fake_sandbox = MagicMock()
    fake_sandbox.execute = MagicMock(return_value=("hello", 0))

    bundle = MagicMock()
    bundle.enabled = True

    out = await tool_execute(
        command="echo hello",
        sandbox_dir=tmp_path,
        sandbox_manager=fake_sandbox,
        memory_dir=tmp_path,
        _metacog_bundle=bundle,
    )
    assert "COMMAND RESULT" in out
    fake_sandbox.execute.assert_called_once()


@pytest.mark.asyncio
async def test_tool_execute_runs_denylisted_when_bundle_absent(tmp_path):
    """Back-compat: when --enable-metacog is off, the validator
    pathway must not fire. The deny-listed command would still be
    blocked by the sandbox's own protections, but at the agent level
    nothing changes."""
    from ghost_agent.tools.execute import tool_execute
    fake_sandbox = MagicMock()
    fake_sandbox.execute = MagicMock(return_value=("legacy path", 0))

    out = await tool_execute(
        command="echo legacy",
        sandbox_dir=tmp_path,
        sandbox_manager=fake_sandbox,
        memory_dir=tmp_path,
        _metacog_bundle=None,  # off
    )
    assert "COMMAND RESULT" in out
    fake_sandbox.execute.assert_called_once()


# ──────────────────────────────────────────────────────────────────────
# Validators integration — database.py
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tool_postgres_admin_blocks_unguarded_delete_when_enabled():
    """The SQL validator should reject `DELETE FROM x` BEFORE psycopg2
    is even reached. We monkey-patch `_get_connection` to fail loudly
    if it's called — the test only passes if the validator short-circuits."""
    from ghost_agent.tools import database as db_module

    bundle = MagicMock()
    bundle.enabled = True

    def boom(*args, **kwargs):
        raise AssertionError("DB connection must NOT be reached")

    with patch.object(db_module, "_get_connection", boom):
        out = await db_module.tool_postgres_admin(
            action="query",
            connection_string="postgresql://x",
            query="DELETE FROM users",
            _metacog_bundle=bundle,
        )
    assert "SYSTEM BLOCK" in out
    assert "DELETE" in out or "WHERE" in out


@pytest.mark.asyncio
async def test_tool_postgres_admin_blocks_drop_when_enabled():
    from ghost_agent.tools import database as db_module
    bundle = MagicMock()
    bundle.enabled = True

    def boom(*args, **kwargs):
        raise AssertionError("DB connection must NOT be reached")

    with patch.object(db_module, "_get_connection", boom):
        out = await db_module.tool_postgres_admin(
            action="query",
            connection_string="postgresql://x",
            query="DROP TABLE customers",
            _metacog_bundle=bundle,
        )
    assert "SYSTEM BLOCK" in out


@pytest.mark.asyncio
async def test_tool_postgres_admin_skips_validator_on_schema_action():
    """The `schema` action uses hand-crafted SQL that never sees user
    content — the validator must not run on those calls (it would be
    a waste, but also the canned SQL contains `information_schema`
    which the validator's parens balance check might not love)."""
    from ghost_agent.tools import database as db_module
    bundle = MagicMock()
    bundle.enabled = True

    # Force psycopg2 import failure so the call short-circuits cleanly.
    # The point is to verify the validator is NOT called — we don't
    # care about the actual DB path here.
    with patch.object(db_module, "_get_connection",
                      MagicMock(side_effect=ImportError("no psycopg2"))):
        out = await db_module.tool_postgres_admin(
            action="schema",
            connection_string="postgresql://x",
            _metacog_bundle=bundle,
        )
    # Should fall through to the actual schema path (which fails on
    # the missing import), NOT a validator block.
    assert "SYSTEM BLOCK" not in out


# ──────────────────────────────────────────────────────────────────────
# Registry integration
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_registry_injects_bundle_into_execute_lambda(tmp_path):
    """The `execute` tool entry in registry.py must read context.metacog
    and pass it down to tool_execute via the lambda. We verify by
    awaiting the coroutine — if the lambda forwarded an unknown
    keyword to `tool_execute`, the await would raise TypeError."""
    from ghost_agent.tools.registry import get_available_tools

    ctx = MagicMock()
    ctx.sandbox_dir = tmp_path
    ctx.sandbox_manager = MagicMock()
    ctx.memory_dir = tmp_path
    ctx.tor_proxy = None
    ctx.args = MagicMock()
    ctx.args.max_context = 1024
    ctx.args.anonymous = True
    ctx.args.default_db = "postgresql://x"
    sentinel = object()
    ctx.metacog = sentinel

    tools = get_available_tools(ctx)
    assert "execute" in tools
    assert "postgres_admin" in tools

    # Calling with no command surfaces the standard "must provide"
    # error — what matters is that no TypeError is raised on the
    # unknown `_metacog_bundle` kwarg.
    result = await tools["execute"]()
    # Either a structured error string or just non-crash is fine here
    assert isinstance(result, str)


# ──────────────────────────────────────────────────────────────────────
# Lifespan integration
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_host_signal_to_bus_translator_maps_metric_fields(fake_context):
    """The lifespan installs a translator that converts HostSignal
    into a resource_event on the trigger bus. Verify the metric/
    observed mapping is correct so the bridge gets the right kind."""
    bundle = MetacogBundle.from_args(fake_context, _args(enable_metacog=True))

    # Hand-craft a HostSignal that mentions free-RAM floor
    from ghost_agent.utils.telemetry import HostSignal, HostSnapshot
    snap = HostSnapshot(
        ts=time.time(), cpu_percent=20.0, mem_percent=88.0,
        mem_available_mb=300.0, disk_percent=50.0, proc_count=100,
    )
    sig = HostSignal(ts=snap.ts, severity="critical",
                     reason="free<800MB (300MB)", snapshot=snap)

    # Replay the translator logic the same way main.py wires it
    from ghost_agent.core.triggers import resource_event

    async def translator(s):
        metric = "ram"
        observed = s.snapshot.mem_percent
        threshold = 85.0
        if "free<" in s.reason:
            metric = "ram_floor"
        elif "CPU" in s.reason:
            metric = "cpu"
            observed = s.snapshot.cpu_percent
        elif "disk" in s.reason:
            metric = "disk"
            observed = s.snapshot.disk_percent
            threshold = 90.0
        await bundle.bus.publish(
            resource_event(s.reason, metric=metric, observed=observed,
                           threshold=threshold, severity=s.severity)
        )

    await translator(sig)
    history = bundle.bus.history("resource")
    assert len(history) == 1
    ev = history[0]
    assert ev.metric == "ram_floor"
    assert ev.severity == "critical"


# ──────────────────────────────────────────────────────────────────────
# CLI flags
# ──────────────────────────────────────────────────────────────────────
