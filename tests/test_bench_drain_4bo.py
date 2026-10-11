"""DockerSandbox teardown + the boot sweep of orphaned per-solve containers.

Both were found by running a real 20-item bench drain (2026-08-15), not by
review: the per-solve sandbox never closed its docker client (EMFILE after
~20 items) and a SIGKILL mid-solve left containers running against a
TemporaryDirectory Python had already deleted.

§4NM: the operator-armed bench drain itself (POST /api/bench/drain, the bench
idle phase, the ledger regime split, the health readback) was removed with the
bench banks; only the sandbox pins below remain.
"""

import os
from unittest.mock import MagicMock


# ──────────────────────────────────────────────────────────────────────
# The FD leak — found by running a real 20-item drain, not by review
# ──────────────────────────────────────────────────────────────────────

class TestTheSolveDoesNotLeakDockerClients:
    """LIVE FINDING (2026-08-15). A 20-item drain ran 15 items and then
    the last 5 died on `OSError: [Errno 24] Too many open files`: they
    failed to solve AND failed to append their ledger rows, each leaving
    a container stuck in `Created`. Measured on the box: 228 unix FDs to
    /var/run/docker.sock after ~20 items.

    Cause: `DockerSandbox.__init__` opens a client via `from_env()`,
    dream.py builds a FRESH sandbox per solve, and nothing ever closed
    the connection pool. Four review rounds did not find this; one real
    drain did — the organic cadence (one item per 45-72 min, with
    restarts between) never accumulates enough to show it.
    """

    def _sandbox(self):
        from ghost_agent.sandbox import docker as _d
        sb = object.__new__(_d.DockerSandbox)
        sb.client = MagicMock()
        sb.container = None
        sb.container_name = "ghost-agent-sandbox-deadbeef"
        sb.NotFound = _d.docker.errors.NotFound if hasattr(
            _d, "docker") else Exception

        class _NF(Exception):
            pass
        sb.NotFound = _NF
        sb.APIError = Exception
        sb.client.containers.get.side_effect = _NF()
        return sb

    def test_discarding_a_sandbox_CLOSES_its_docker_client(self):
        sb = self._sandbox()
        sb.close(remove=True)
        assert sb.client.close.called, (
            "the connection pool survives the sandbox — every solve then "
            "leaks its docker sockets until EMFILE")

    def test_a_KEPT_sandbox_stays_usable(self):
        """remove=False means "stop but keep it warm for a fast resume".
        Closing the client there would break the long-lived sandbox the
        agent reuses for real turns."""
        sb = self._sandbox()
        sb.close(remove=False)
        assert not sb.client.close.called

    def test_close_NEVER_raises(self):
        """Its docstring promises this ("expected to run from signal
        handlers / shutdown hooks where exceptions are disruptive") and
        the promise was false — an exception in the container step
        propagated straight through. Both callers are teardown paths, so
        a throwing close() is how one bad container takes the idle loop
        with it."""
        from ghost_agent.sandbox import docker as _d
        # A RuntimeError from `containers.get` is caught INSIDE
        # `_close_container`, so it proves nothing about the outer
        # handler — the first version of this test made that mistake and
        # a mutation removing the outer `except` stayed green. Drive a
        # failure that genuinely ESCAPES: `except self.NotFound` raises
        # AttributeError while handling an exception when the attribute
        # is missing, and that propagates out of `_close_container`.
        sb = object.__new__(_d.DockerSandbox)
        sb.client = MagicMock()
        sb.container = None
        sb.container_name = "ghost-agent-sandbox-nonf"
        sb.APIError = Exception
        sb.client.containers.get.side_effect = RuntimeError("docker is gone")
        # deliberately NO `sb.NotFound`
        sb.close(remove=True)          # must not raise
        assert sb.client.close.called, \
            "the client was not closed on the failure path"

    def test_close_survives_a_half_built_instance(self):
        """`self.container` was read outside any guard, so an instance
        whose __init__ was bypassed raised AttributeError — inconsistent
        with this file's own getattr-default convention."""
        from ghost_agent.sandbox import docker as _d
        sb = object.__new__(_d.DockerSandbox)
        sb.client = MagicMock()
        sb.container_name = "ghost-agent-sandbox-halfbuilt"

        class _NF(Exception):
            pass
        sb.NotFound = _NF
        sb.APIError = Exception
        sb.client.containers.get.side_effect = _NF()
        sb.close(remove=True)          # no `container` attribute at all
        assert sb.client.close.called
        # THE discriminating assertion: with the getattr default the code
        # proceeds to the by-name lookup (which is what removes an
        # orphan); reading `self.container` directly raises before ever
        # getting there, and the outer handler would swallow it — leaving
        # this test green while orphan cleanup silently stopped.
        sb.client.containers.get.assert_called_once_with(sb.container_name)

    def test_teardown_removes_a_container_it_never_BOUND(self):
        """The old per-solve teardown required `.container` to be truthy,
        so a sandbox that failed midway through provisioning — exactly
        what EMFILE causes — orphaned its container forever. Five such
        containers were left on the live box, one per failed item."""
        sb = self._sandbox()
        orphan = MagicMock()
        orphan.status = "created"
        sb.client.containers.get.side_effect = None
        sb.client.containers.get.return_value = orphan
        sb.close(remove=True)
        sb.client.containers.get.assert_called_with(sb.container_name)
        assert orphan.remove.called

# ──────────────────────────────────────────────────────────────────────
# Boot sweep of containers orphaned by a kill mid-solve
# ──────────────────────────────────────────────────────────────────────

class TestTheBootSweepOnlyRemovesProvableOrphans:
    """A `finally` cannot run through SIGKILL, so a daemon killed
    mid-solve leaves a container running against a TemporaryDirectory
    Python already deleted. Two were found live (3 days and 43 min old).

    This code DELETES containers, so every test here is about what it
    must NOT touch."""

    def _mgr(self, tmp_path, containers):
        from ghost_agent.sandbox import docker as _d
        sb = object.__new__(_d.DockerSandbox)
        sb.client = MagicMock()
        sb.container = None
        sb.container_name = "ghost-agent-sandbox-self"
        sb.client.containers.list.return_value = containers
        return sb

    def _c(self, name, sources, status="running", age_s=99999):
        import datetime as _dt
        c = MagicMock()
        c.name = name
        c.status = status
        created = (_dt.datetime.now(_dt.timezone.utc)
                   - _dt.timedelta(seconds=age_s))
        c.attrs = {"Mounts": [{"Source": s} for s in sources],
                   "Created": created.isoformat().replace("+00:00", "Z")}
        return c

    def _tmpws(self, name="tmpabcdefgh"):
        """A path shaped like `tempfile.TemporaryDirectory()` — the ONLY
        shape the sweep treats as a per-solve workspace."""
        import tempfile
        return os.path.join(tempfile.gettempdir(), name)

    def test_it_removes_an_old_PER_SOLVE_sandbox(self, tmp_path):
        dead = self._c("ghost-agent-sandbox-dead", [self._tmpws()])
        sb = self._mgr(tmp_path, [dead])
        assert sb.sweep_orphaned_containers() == ["ghost-agent-sandbox-dead"]
        assert dead.remove.called

    def test_a_SURVIVING_workspace_does_not_save_an_orphan(self, tmp_path):
        """THE correction that made this feature real. The first version
        swept "containers whose mount no longer exists", and a dry run
        against the live box refuted it: SIGKILL is exactly what orphans
        a container AND exactly what prevents
        `TemporaryDirectory.cleanup()` from running, so the workspace
        survives. That criterion spared every orphan it targeted."""
        ws = self._tmpws("tmpstillhere")
        os.makedirs(ws, exist_ok=True)
        try:
            dead = self._c("ghost-agent-sandbox-dead", [ws])
            sb = self._mgr(tmp_path, [dead])
            assert sb.sweep_orphaned_containers() == \
                ["ghost-agent-sandbox-dead"]
        finally:
            os.rmdir(ws)

    def test_it_NEVER_removes_the_agents_OWN_sandbox(self, tmp_path):
        """The long-lived sandbox mounts $GHOST_HOME/sandbox — a stable
        path outside the temp root, so it can never match."""
        live = tmp_path / "sandbox"
        live.mkdir()
        alive = self._c("ghost-agent-sandbox-live", [str(live)])
        sb = self._mgr(tmp_path, [alive])
        assert sb.sweep_orphaned_containers() == []
        assert not alive.remove.called

    def test_it_NEVER_removes_a_DETACHED_JOB_container(self, tmp_path):
        """`ghostjobs-*` workspaces belong to commands deliberately
        detached to survive agent restarts (§4AX). One was live on the
        box while this was written; sweeping it would destroy running
        work. It sits under the temp root, so only the `tmp` prefix test
        separates it."""
        import tempfile
        job = os.path.join(tempfile.gettempdir(), "ghostjobs-bq1yp821")
        c = self._c("ghost-agent-sandbox-job", [job])
        sb = self._mgr(tmp_path, [c])
        assert sb.sweep_orphaned_containers() == []
        assert not c.remove.called

    def test_a_tmp_NAMED_path_outside_the_temp_root_is_spared(self,
                                                              tmp_path):
        """The `tmp` prefix alone is not the test — containment in the
        system temp root is. A project directory that happens to be
        called `tmpdata` must never be read as a throwaway workspace."""
        # NOT pytest's tmp_path — on macOS that lives INSIDE the system
        # temp root, so it would legitimately qualify. A path genuinely
        # outside it is the case under test.
        decoy = os.path.join(os.path.expanduser("~"), "projects", "tmpdata")
        c = self._c("ghost-agent-sandbox-decoy", [decoy])
        sb = self._mgr(tmp_path, [c])
        assert sb.sweep_orphaned_containers() == []
        assert not c.remove.called

    def test_a_YOUNG_per_solve_sandbox_is_spared(self, tmp_path):
        """It may belong to a solve genuinely in flight in another
        process. True orphans persist for hours."""
        young = self._c("ghost-agent-sandbox-young", [self._tmpws()],
                        age_s=60)
        sb = self._mgr(tmp_path, [young])
        assert sb.sweep_orphaned_containers() == []

    def test_an_UNDATED_container_is_spared(self, tmp_path):
        c = self._c("ghost-agent-sandbox-nodate", [self._tmpws()])
        c.attrs["Created"] = ""
        sb = self._mgr(tmp_path, [c])
        assert sb.sweep_orphaned_containers() == []

    def test_a_container_with_ANY_non_per_solve_mount_is_spared(self,
                                                                tmp_path):
        mixed = self._c("ghost-agent-sandbox-mixed",
                        [self._tmpws(), str(tmp_path / "project")])
        sb = self._mgr(tmp_path, [mixed])
        assert sb.sweep_orphaned_containers() == []

    def test_a_container_with_NO_mounts_is_spared(self, tmp_path):
        """'Cannot tell' must never read as 'safe to delete'."""
        unknown = self._c("ghost-agent-sandbox-unknown", [])
        sb = self._mgr(tmp_path, [unknown])
        assert sb.sweep_orphaned_containers() == []
        assert not unknown.remove.called

    def test_it_never_removes_ITSELF(self, tmp_path):
        me = self._c("ghost-agent-sandbox-self", [self._tmpws()])
        sb = self._mgr(tmp_path, [me])
        assert sb.sweep_orphaned_containers() == []

    def test_it_ignores_containers_that_are_not_ours(self, tmp_path):
        other = self._c("postgres-prod", [self._tmpws()])
        sb = self._mgr(tmp_path, [other])
        assert sb.sweep_orphaned_containers() == []
        assert not other.remove.called

    def test_the_kill_switch_disables_it(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GHOST_SANDBOX_SWEEP", "0")
        dead = self._c("ghost-agent-sandbox-dead", [self._tmpws()])
        sb = self._mgr(tmp_path, [dead])
        assert sb.sweep_orphaned_containers() == []
        assert not dead.remove.called

    def test_one_unremovable_container_does_not_stop_the_sweep(self,
                                                               tmp_path):
        bad = self._c("ghost-agent-sandbox-bad", [self._tmpws("tmpg1")])
        bad.remove.side_effect = RuntimeError("docker said no")
        good = self._c("ghost-agent-sandbox-good", [self._tmpws("tmpg2")])
        sb = self._mgr(tmp_path, [bad, good])
        assert sb.sweep_orphaned_containers() == ["ghost-agent-sandbox-good"]

    def test_a_listing_failure_is_survivable(self, tmp_path):
        sb = self._mgr(tmp_path, [])
        sb.client.containers.list.side_effect = RuntimeError("no docker")
        assert sb.sweep_orphaned_containers() == []      # must not raise

    def test_the_cap_bounds_one_sweep(self, tmp_path):
        many = [self._c(f"ghost-agent-sandbox-{i}",
                        [self._tmpws(f"tmpcap{i}")]) for i in range(40)]
        sb = self._mgr(tmp_path, many)
        assert len(sb.sweep_orphaned_containers()) == sb._SWEEP_CAP
