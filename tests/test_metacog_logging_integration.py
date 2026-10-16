"""Logging-contract integration tests.

We don't just verify that lines are emitted — we verify the CONTRACT
that monitoring depends on:

  * Every uplift line starts with the ``Metacog <Subsystem>`` title
    so ``grep "metacog "`` reliably catches them all.
  * The validator / replan / host events carry their
    primary identifier (tool, severity, action) as a key=value pair.
  * Lifetime counters bump on every relevant event.
  * The shutdown summary fires once with the right field set.
  * The bundle counts validator blocks, host signals and replans so
    the summary is accurate. (§4PJ: arbitrations and confidence counts
    went with the arbiter and the confidence score.)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ghost_agent.core.metacog import MetacogBundle
from ghost_agent.core.metacog_log import Subsystem
from ghost_agent.core.triggers import (
    ReplanBridge, TriggerBus,
    loop_event, resource_event,
)
from ghost_agent.utils.telemetry import HostSnapshot


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

def _args(**overrides):
    a = argparse.Namespace(
        enable_metacog=True,
    )
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


@pytest.fixture
def bundle(tmp_path):
    ctx = MagicMock()
    ctx.memory_dir = tmp_path
    ctx.args = MagicMock()
    ctx.args.model = "test"
    ctx.llm_client = MagicMock()
    if hasattr(ctx.llm_client, "get_embeddings"):
        del ctx.llm_client.get_embeddings
    ctx.project_store = None
    ctx.current_project_id = None
    return MetacogBundle.from_args(ctx, _args())


# ──────────────────────────────────────────────────────────────────────
# Counter API
# ──────────────────────────────────────────────────────────────────────

class TestCounters:
    def test_count_noop_when_disabled(self):
        b = MetacogBundle()  # enabled=False
        b.count(host_signal=True, validator_block=True)
        assert b._ctr_host_signals == 0
        assert b._ctr_validator_blocks == 0

    def test_count_bumps_when_enabled(self, bundle):
        bundle.count(host_signal=True)
        bundle.count(host_signal=True, host_critical=True)
        bundle.count(validator_block=True)
        bundle.count(replan_attempt=True, replan_succeeded=True)
        assert bundle._ctr_host_signals == 2
        assert bundle._ctr_host_critical == 1
        assert bundle._ctr_validator_blocks == 1
        assert bundle._ctr_replan_attempts == 1
        assert bundle._ctr_replan_succeeded == 1

# ──────────────────────────────────────────────────────────────────────
# Shutdown summary
# ──────────────────────────────────────────────────────────────────────

class TestShutdownSummary:
    @pytest.mark.asyncio
    async def test_shutdown_emits_summary(self, bundle):
        # Pre-populate some counters
        bundle.count(validator_block=True)
        bundle.count(host_signal=True, host_critical=True)

        with patch("ghost_agent.utils.logging.pretty_log") as pl:
            await bundle.shutdown()
        # Find the summary call
        summary_calls = [c for c in pl.call_args_list
                         if c.args and c.args[0] == "Metacog Summary"]
        assert len(summary_calls) == 1
        content = summary_calls[0].args[1]
        assert "arbitrations=" not in content and "conf_" not in content   # §4PJ
        assert "validator_blocks=1" in content
        assert "host_signals=1" in content
        assert "host_critical=1" in content

    @pytest.mark.asyncio
    async def test_shutdown_summary_robust_to_zero_counters(self, bundle):
        # No activity — summary should still fire (it's the record of
        # an idle session)
        with patch("ghost_agent.utils.logging.pretty_log") as pl:
            await bundle.shutdown()
        summary_calls = [c for c in pl.call_args_list
                         if c.args and c.args[0] == "Metacog Summary"]
        assert len(summary_calls) == 1
        content = summary_calls[0].args[1]
        assert "validator_blocks=0" in content


# ──────────────────────────────────────────────────────────────────────
# ReplanBridge logging
# ──────────────────────────────────────────────────────────────────────

class TestReplanBridgeLogging:
    @pytest.mark.asyncio
    async def test_revised_emits_info_level_log(self):
        bus = TriggerBus()

        class Plan:
            def request_revision(self, task_id, reason):
                return True

        bridge = ReplanBridge(
            bus, plan_getter=lambda: Plan(),
            current_task_getter=lambda: "task_42",
        )
        bridge.attach()
        with patch("ghost_agent.utils.logging.pretty_log") as pl:
            await bus.publish(loop_event("looped", severity="warning"))
        replan_calls = [c for c in pl.call_args_list
                        if c.args and c.args[0] == "Metacog Replan"]
        assert len(replan_calls) == 1
        content = replan_calls[0].args[1]
        assert "action=revised" in content
        assert "trigger=loop/warning" in content
        assert "task=task_42" in content

    @pytest.mark.asyncio
    async def test_revision_rejected_emits_warning_level_log(self):
        bus = TriggerBus()

        class Plan:
            def request_revision(self, task_id, reason):
                return False  # max revisions

        bridge = ReplanBridge(
            bus, plan_getter=lambda: Plan(),
            current_task_getter=lambda: "t1",
        )
        bridge.attach()
        with patch("ghost_agent.utils.logging.pretty_log") as pl:
            await bus.publish(loop_event("looped", severity="warning"))
        replan_calls = [c for c in pl.call_args_list
                        if c.args and c.args[0] == "Metacog Replan"]
        assert len(replan_calls) == 1
        assert replan_calls[0].kwargs["level"] == "WARNING"

    @pytest.mark.asyncio
    async def test_info_severity_logs_at_debug_only(self):
        bus = TriggerBus()
        bridge = ReplanBridge(
            bus, plan_getter=lambda: None,
            current_task_getter=lambda: None,
        )
        bridge.attach()
        with patch("ghost_agent.utils.logging.pretty_log") as pl:
            await bus.publish(loop_event("looped", severity="info"))
        # An info-severity event SHOULD log (debug level) but should
        # not be at INFO/WARNING level.
        replan_calls = [c for c in pl.call_args_list
                        if c.args and c.args[0] == "Metacog Replan"]
        # We don't pin the count (debug logs may be filtered) but
        # if any landed, level must be DEBUG.
        for c in replan_calls:
            assert c.kwargs["level"] == "DEBUG"


# ──────────────────────────────────────────────────────────────────────
# Validator log contract (smoke — full validator tests live separately)
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_shell_validator_block_emits_structured_log(tmp_path):
    from ghost_agent.tools.execute import tool_execute
    bundle = MagicMock()
    bundle.enabled = True
    with patch("ghost_agent.utils.logging.pretty_log") as pl:
        await tool_execute(
            command="rm -rf /",
            sandbox_dir=tmp_path,
            sandbox_manager=MagicMock(),
            memory_dir=tmp_path,
            _metacog_bundle=bundle,
        )
    valid_calls = [c for c in pl.call_args_list
                   if c.args and c.args[0] == "Metacog Valid"]
    assert len(valid_calls) == 1
    content = valid_calls[0].args[1]
    assert "verdict=block" in content
    assert "tool=shell" in content
    # Reason should be present and not empty
    assert "reason=" in content


@pytest.mark.asyncio
async def test_sql_validator_block_emits_structured_log():
    from ghost_agent.tools import database as db_module
    bundle = MagicMock()
    bundle.enabled = True

    def boom(*args, **kwargs):
        raise AssertionError("must not connect")

    with patch.object(db_module, "_get_connection", boom):
        with patch("ghost_agent.utils.logging.pretty_log") as pl:
            await db_module.tool_postgres_admin(
                action="query",
                connection_string="postgresql://x",
                query="DELETE FROM users",
                _metacog_bundle=bundle,
            )
    valid_calls = [c for c in pl.call_args_list
                   if c.args and c.args[0] == "Metacog Valid"]
    assert len(valid_calls) == 1
    content = valid_calls[0].args[1]
    assert "verdict=block" in content
    assert "tool=sql" in content
    assert "action=query" in content
