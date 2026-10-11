"""Stale-process guard (core/staleness.py + the main.py / watchdog-tick
entry points).

This file used to pin the §4BN "--prm-online-update must not fail
silently" loudness surfaces (boot warnings, learning-health wiring rows,
the phase-2.7 skip log, the feedback-channel warning, the online-update
dispatch). The PRM package and every one of those surfaces were removed in
batch 3 (§4NM); only the staleness guard that §4BN R30–R34 built — a live,
PRM-independent mechanism — remains, and its tests are kept here.
"""


class TestStaleProcessIsAnnounced:
    """R30 MAJOR-2 — the deployment-level instance of §4BN's own class.

    R28 found the live agent had run PRE-§4BN code for a day, printing the
    retracted §4BM framing every ~3h while the corrected message had never
    executed. R30 found the same condition NINE MINUTES after the restart
    that fixed it, because this section kept editing its own source. In
    both cases nothing in the tree would have noticed — the operator's only
    evidence was a log line that silently disagreed with the code.
    """

    def test_a_source_file_newer_than_the_process_is_LOUD(self, monkeypatch):
        from ghost_agent import main as _m
        from ghost_agent.core import staleness as _st
        emitted = []
        monkeypatch.setattr(_m, "pretty_log",
                            lambda *a, **k: emitted.append((a, k)))
        _st._REPORTED.clear()
        import ghost_agent.core.agent  # ensure it is loaded
        monkeypatch.setitem(_st._DIGESTS_AT_LOAD, "core/agent.py", "0" * 64)
        msg = _m.audit_source_newer_than_process()
        assert msg, ("a source file edited after boot was not announced — "
                     "CPython does not reload, so the operator is reading "
                     "code the box is not running")
        assert "core/agent.py" in msg
        assert emitted and emitted[0][1].get("level") == "WARNING"

    def test_a_byte_identical_rewrite_does_NOT_fire(self, tmp_path,
                                                    monkeypatch):
        """R32 MAJOR-6: the mtime version could not distinguish "source
        changed" from "file rewritten identically", so a `cp`-based restore
        left the process permanently telling the operator to restart — and
        this section's own review protocol does exactly that every round.
        A mechanism built to satisfy "cannot distinguish the two
        hypotheses", failing that test."""
        import os
        import shutil
        from ghost_agent import main as _m
        from ghost_agent.core import staleness as _st
        monkeypatch.setattr(_m, "pretty_log", lambda *a, **k: None)
        _st._REPORTED.clear()
        here = os.path.dirname(os.path.abspath(_m.__file__))
        target = os.path.join(here, "core/feedback.py")
        backup = tmp_path / "feedback.py"
        shutil.copy2(target, backup)
        try:
            shutil.copy(backup, target)          # new mtime, same bytes
            assert _m.audit_source_newer_than_process() is None, (
                "a byte-identical restore was reported as staleness — this "
                "fires on every review round and trains the operator to "
                "ignore the one warning that matters")
        finally:
            shutil.copy2(backup, target)

    def test_silent_when_the_process_matches_its_source(self, monkeypatch):
        from ghost_agent import main as _m
        from ghost_agent.core import staleness as _st
        monkeypatch.setattr(_m, "pretty_log", lambda *a, **k: None)
        assert _m.audit_source_newer_than_process() is None, (
            "noise on a healthy box — every boot would cry wolf")

    def test_the_tick_reevaluates_staleness(self, monkeypatch):
        from ghost_agent import main as _m
        # R31 MAJOR-2: an AST walk for the call name is the retired proxy —
        # `if False:` around it was green, auditor never running. Drive the
        # WATCHDOG TICK instead, which is where R31 CRIT-1 moved it,
        # because that is the only place it can see a post-boot edit.
        import asyncio
        from unittest.mock import MagicMock
        from ghost_agent import main as _m
        from ghost_agent.core import staleness as _st
        from ghost_agent.core.agent import GhostAgent

        seen = []
        # R34: the tick logs via `utils.logging` directly — it must NOT
        # import `main`, because under `python -m src.ghost_agent.main`
        # that re-executes the whole module inside the live process.
        import ghost_agent.utils.logging as _glog
        monkeypatch.setattr(_glog, "pretty_log",
                            lambda *a, **k: seen.append((a, k)))
        _st._REPORTED.clear()
        import ghost_agent.tools.memory  # ensure it is loaded
        monkeypatch.setitem(_st._DIGESTS_AT_LOAD, "tools/memory.py", "0" * 64)
        agent = GhostAgent.__new__(GhostAgent)
        agent.context = MagicMock()
        try:
            asyncio.run(agent._biological_tick())
        except Exception:      # noqa: BLE001 — the tick does far more
            pass
        assert any(a and a[0] == "Stale Process" for a, _k in seen), (
            "the watchdog tick does not re-evaluate source staleness, so a "
            "post-boot edit — the ONLY case this auditor exists for — goes "
            "unnoticed for the life of the process")

    def test_a_raising_logger_does_not_silence_the_divergence(self,
                                                              monkeypatch):
        """R32 MINOR-1 / R33 MAJOR-5: the divergence used to be marked
        BEFORE emitting, and the tick swallows exceptions — so one raising
        `pretty_log` silenced the auditor for the life of the process.
        R32 fixed the order and pinned nothing; moving it back was green
        across 639 tests."""
        from ghost_agent import main as _m
        from ghost_agent.core import staleness as _st
        _st._REPORTED.clear()
        import ghost_agent.core.agent  # ensure it is loaded
        monkeypatch.setitem(_st._DIGESTS_AT_LOAD, "core/agent.py", "0" * 64)

        def _boom(*a, **k):
            raise RuntimeError("log sink down")

        monkeypatch.setattr(_m, "pretty_log", _boom)
        try:
            _m.audit_source_newer_than_process()
        except RuntimeError:
            pass
        assert not any(r == "core/agent.py" for r, _d in _st._REPORTED), (
            "the divergence was marked reported even though the log call "
            "raised — it can never be announced again this process")
        seen = []
        monkeypatch.setattr(_m, "pretty_log",
                            lambda *a, **k: seen.append(a))
        _m.audit_source_newer_than_process()
        assert seen, "the retry after a transient log failure said nothing"

    def test_the_guard_is_alive_under_the_PRODUCTION_import_shape(self):
        """R34 CRIT-1 — the guard was 100% dead on the live box.

        Production launches `python -m src.ghost_agent.main`, so every
        module is `src.ghost_agent.*`; the guard keyed `sys.modules` on a
        hardcoded `"ghost_agent."`, so its loaded set was always empty and
        it returned None on every tick of every production process — dead
        in the only environment it exists for, while its tests passed.
        `utils/component_guard.py` records the identical prefix bug
        leaving five subsystems inert on the live agent for weeks.

        The prefix is DERIVED from `__name__` now. This pin asserts the
        derivation, which is the property; asserting one literal was what
        failed."""
        # Asserting `_package_root() == __name__.rsplit(...)` is a
        # verification that CANNOT DISTINGUISH: under the test shape a
        # hardcoded "ghost_agent" satisfies it too (verified — the
        # mutation was green). The only discriminating check is to run
        # under the OTHER shape, so run it in a subprocess exactly as
        # production does.
        import subprocess
        import sys
        from pathlib import Path

        repo = Path(__file__).resolve().parent.parent
        probe = (
            "import sys; sys.path.insert(0, '.')\n"
            "from src.ghost_agent.core import staleness as s\n"
            "import src.ghost_agent.core.agent\n"
            "loaded = s.loaded_watched_files()\n"
            "print('ROOT=' + s._package_root())\n"
            "print('LOADED=' + ','.join(loaded))\n"
        )
        out = subprocess.run([sys.executable, "-c", probe], cwd=repo,
                             capture_output=True, text=True, timeout=120)
        assert "ROOT=src.ghost_agent" in out.stdout, (
            "under the PRODUCTION import shape the guard does not derive "
            f"its package root — it is dead there:\n{out.stdout}\n{out.stderr[-400:]}")
        assert "core/agent.py" in out.stdout, (
            "under the PRODUCTION import shape the guard sees NO watched "
            "module as loaded, so it returns None on every tick of every "
            f"production process:\n{out.stdout}\n{out.stderr[-400:]}")

    def test_a_SECOND_divergence_of_the_same_file_is_still_reported(self):
        """R34 MAJOR-3: the dedup key was the PATH, so once a file had
        been reported it was silent for the life of the process — and this
        section edits `core/agent.py` most rounds, so edit → warn →
        restore → edit again was the R28/R30 condition, silently. Key on
        the digest too."""
        from ghost_agent.core import staleness as _st
        import ghost_agent.core.agent  # noqa: F401
        _st._REPORTED.clear()
        real = _st.read_digests
        seen = []
        try:
            _st.read_digests = lambda only=None: {"core/agent.py": "a" * 64}
            _st._DIGESTS_AT_LOAD["core/agent.py"] = "b" * 64
            assert _st.audit_source_newer_than_process(seen.append)
            assert not _st.audit_source_newer_than_process(seen.append), \
                "the same divergence was announced twice"
            # a DIFFERENT divergence of the SAME file must still speak up
            _st.read_digests = lambda only=None: {"core/agent.py": "c" * 64}
            assert _st.audit_source_newer_than_process(seen.append), (
                "a second, genuinely different divergence of the same file "
                "was silent — path-keyed dedup hides exactly the "
                "edit/restore/edit cycle this section performs")
        finally:
            _st.read_digests = real
            _st._REPORTED.clear()
            _st._DIGESTS_AT_LOAD.pop("core/agent.py", None)

    def test_the_watch_list_covers_the_whole_blast_radius(self):
        """R31 MAJOR-3: shrinking the list to one entry was green, and it
        omitted the TWIN skip log — the original drift site. (§4NM: the
        list lives in core/staleness.py and `prm/scorer.py` left it with
        the PRM package.)"""
        from ghost_agent.core.staleness import PRM_STALENESS_WATCHED
        for required in ("core/agent.py", "core/learning_health.py",
                         "core/feedback.py", "tools/memory.py", "main.py"):
            assert required in PRM_STALENESS_WATCHED, (
                f"{required} is an operator-facing module and is not "
                "watched — an operator could read its output from a "
                "process that never loaded it")
