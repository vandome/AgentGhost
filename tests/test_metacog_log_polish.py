"""metacog_log assigns a distinct icon per subsystem (not BRAIN_THINK).

§4PT: the threshold-plumbing, host-telemetry dedup and ReplanBridge noop
tests went with host telemetry and the bridge.
"""

from __future__ import annotations

from unittest.mock import patch

from ghost_agent.core.metacog_log import (
    _DEFAULT_ICON,
    _SUBSYSTEM_ICONS,
    Subsystem,
    emit,
)


# ──────────────────────────────────────────────────────────────────────
# Per-subsystem distinct icons
# ──────────────────────────────────────────────────────────────────────

class TestSubsystemIcons:
    def test_every_subsystem_has_distinct_icon(self):
        # Every subsystem is mapped (§4PJ: CONF/CALIB/ARBITER removed)
        subsystems = {Subsystem.BOOT, Subsystem.SUMMARY, Subsystem.VALID}
        for gone in ("CONF", "CALIB", "ARBITER", "HOST", "REPLAN", "GATE"):   # §4PJ, §4PT
            assert not hasattr(Subsystem, gone), gone
        assert set(_SUBSYSTEM_ICONS.keys()) == subsystems
        # Icons distinct from each other
        icons = list(_SUBSYSTEM_ICONS.values())
        assert len(set(icons)) == len(icons), (
            f"duplicate icons in subsystem map: {icons}"
        )

    def test_metacog_icons_distinct_from_brain_think(self):
        """The whole point of this pass — operators couldn't tell
        metacog lines apart from agent brain-thinking lines because
        both used 💭."""
        from ghost_agent.utils.logging import Icons
        for sub, icon in _SUBSYSTEM_ICONS.items():
            assert icon != Icons.BRAIN_THINK, (
                f"subsystem {sub} still uses BRAIN_THINK 💭"
            )
        assert _DEFAULT_ICON != Icons.BRAIN_THINK

    def test_emit_uses_subsystem_icon(self):
        with patch("ghost_agent.utils.logging.pretty_log") as pl:
            emit(Subsystem.VALID, verdict="block")
        assert pl.called
        kwargs = pl.call_args.kwargs
        assert kwargs["icon"] == _SUBSYSTEM_ICONS[Subsystem.VALID]

    def test_emit_caller_override_respected(self):
        """Caller explicitly passing icon= overrides the subsystem map."""
        with patch("ghost_agent.utils.logging.pretty_log") as pl:
            emit(Subsystem.VALID, icon="🚨", verdict="block")
        assert pl.call_args.kwargs["icon"] == "🚨"

    def test_unknown_subsystem_falls_back(self):
        with patch("ghost_agent.utils.logging.pretty_log") as pl:
            emit("brand_new_subsystem", foo="bar")
        assert pl.call_args.kwargs["icon"] == _DEFAULT_ICON
