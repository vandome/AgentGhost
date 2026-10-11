"""Perfect-It is skipped inside a read-only (simulated) turn.

§4NM: the validator self-test gate (`_instrument_validator_for_self_test`,
`_extract_selftest_dump`, `_looks_like_validator_crash`) and the widened
runtime crash detector left core/dream.py with synthetic self-play; their
tests were removed. What remains pins the `is_read_only` marker on
`ReadOnlySkillMemory` and the SIMULATION SHORT-CIRCUIT in agent.py that
reads it.
"""


class TestPerfectItSkipDuringSelfPlay:
    """The Perfect-It follow-up LLM call ran unconditionally at the
    end of every successful heavy-tool turn — including the inner
    turn of a self-play sub-agent. Its `learn_lesson` write landed
    in the ReadOnlySkillMemory wrapper (no-op), but:
      (a) the misleading 'Saved optimization strategy to playbook'
          log line fired on the sub-agent's request ID (production
          trace 16:36, request C6), and
      (b) a full ~15s follow-up LLM call burned per self-play cycle
          generating suggestions that were silently discarded.

    The fix marks ReadOnlySkillMemory with `is_read_only = True`
    and agent.py skips the whole Perfect-It block when the marker
    is present.
    """

    def test_readonly_skill_memory_has_is_read_only_marker(self):
        """The marker must live on the class (not an instance
        attribute) so `getattr(wrapper, 'is_read_only', False)`
        works on any wrapper instance built inside self-play."""
        # §4CL S1: the façade is a real module-level class now, so the
        # marker is checked where it has to live — in the CLASS dict, so
        # `getattr(wrapper, "is_read_only", False)` works on every
        # instance without each one setting it.
        from unittest.mock import MagicMock
        from ghost_agent.core.isolation import ReadOnlySkillMemory

        assert vars(ReadOnlySkillMemory).get("is_read_only") is True
        assert getattr(ReadOnlySkillMemory(MagicMock()),
                       "is_read_only", False) is True

    def test_perfect_it_block_checks_read_only_marker(self):
        """The entry condition for the Perfect-It block must gate on
        `is_read_only`, so self-play inner turns skip the whole block."""
        import inspect
        from ghost_agent.core.agent import GhostAgent
        src = inspect.getsource(GhostAgent)
        # The marker check must be near the Perfect-It trigger.
        assert "is_read_only" in src
        perfect_it_idx = src.index('"Perfect It Protocol"')
        marker_idx = src.rindex("is_read_only", 0, perfect_it_idx)
        # Marker check must appear within ~1 KB before the log
        # line — i.e., it guards the entry, not some unrelated path.
        assert perfect_it_idx - marker_idx < 1500


