"""Tests for scripts/processing/_dispatch.py: the shared fan-out engine.

Covers the job-sizing helpers, the ``run_stage`` dispatch in both modes (fail-soft:
attempt all, return the failed set; fail-fast: any failure aborts with exit 1), the
per-job timeout kill, the failure sentinel, and the ``_run_one`` interrupt guards.
Trivial subprocess stubs -- no real testdata needed. ``launch_interval`` defaults to
0 here so the dispatch tests stay fast and free of thread-timing flakiness.
"""

import logging
import os
import signal
import sys
import time

import pytest

from scripts.processing import _dispatch as f

# scripts/CLI/tools-layer suite: excluded from `make test-fast`.
pytestmark = pytest.mark.cli

_OK = [sys.executable, "-c", "pass"]
_FAIL = [sys.executable, "-c", "import sys; sys.exit(1)"]
_SLEEP = [sys.executable, "-c", "import time; time.sleep(30)"]  # a wedged job


@pytest.fixture(autouse=True)
def _clear_interrupted():
    """Keep the module-global interrupt flag clean around each test."""
    f._interrupted.clear()
    yield
    f._interrupted.clear()


# ---------------------------------------------------------------------------
# job sizing
# ---------------------------------------------------------------------------


class TestDefaultScienceJobs:
    @pytest.mark.parametrize(
        "cpus,expected",
        [(None, 1), (1, 1), (8, 8), (64, 16), (100, 25)],
    )
    def test_cap(self, monkeypatch, cpus, expected):
        monkeypatch.setattr(f.os, "cpu_count", lambda: cpus)
        assert f._default_science_jobs() == expected


class TestDefaultMastersJobs:
    @staticmethod
    def _fake_sysconf(ram_gib):
        page = 4096  # SC_PHYS_PAGES * SC_PAGE_SIZE = total bytes; 4 KiB page.
        pages = int(ram_gib * 2**30) // page
        return lambda name: {"SC_PHYS_PAGES": pages, "SC_PAGE_SIZE": page}[name]

    def test_big_host_gets_fixed_cap(self, monkeypatch):
        monkeypatch.setattr(f.os, "cpu_count", lambda: 256)
        monkeypatch.setattr(f.os, "sysconf", self._fake_sysconf(2048))
        assert f._default_masters_jobs() == f._MASTERS_JOBS

    def test_ram_floors_below_fixed_cap(self, monkeypatch):
        monkeypatch.setattr(f.os, "cpu_count", lambda: 256)
        monkeypatch.setattr(f.os, "sysconf", self._fake_sysconf(24))
        assert f._default_masters_jobs() == 24 // f._MASTERS_JOB_GIB
        assert f._default_masters_jobs() < f._MASTERS_JOBS

    def test_cores_floor_below_fixed_cap(self, monkeypatch):
        monkeypatch.setattr(f.os, "cpu_count", lambda: 8)
        monkeypatch.setattr(f.os, "sysconf", self._fake_sysconf(256))
        assert f._default_masters_jobs() == f._default_science_jobs() == 8

    def test_unknown_ram_uses_cores_floor_only(self, monkeypatch):
        # 8 cores, not 256: at 256 the cores floor is 64 and never binds, so the
        # fixed cap answers and this test cannot see the branch it names.
        monkeypatch.setattr(f.os, "cpu_count", lambda: 8)

        def _raise(_name):
            raise ValueError("SC_PHYS_PAGES unavailable")

        monkeypatch.setattr(f.os, "sysconf", _raise)
        assert f._default_masters_jobs() == 8
        assert f._default_masters_jobs() < f._MASTERS_JOBS

    def test_never_below_one(self, monkeypatch):
        # Tiny RAM would floor the cap to 0; the helper clamps to 1 (max(1, ...)),
        # so the pool is never empty (an empty pool would silently do nothing).
        monkeypatch.setattr(f.os, "cpu_count", lambda: 8)
        monkeypatch.setattr(f.os, "sysconf", self._fake_sysconf(1))
        assert f._default_masters_jobs() == 1


# ---------------------------------------------------------------------------
# run_stage -- fail-soft (abort_on_failure=False)
# ---------------------------------------------------------------------------


class TestRunStageFailSoft:
    def _run(self, tasks, tmp_path):
        return f.run_stage("job", tasks, 2, str(tmp_path), abort_on_failure=False)

    def test_empty_returns_empty_set(self, tmp_path):
        assert self._run([], tmp_path) == set()

    def test_all_succeed_returns_empty_set(self, tmp_path):
        tasks = [("a", _OK), ("b", _OK), ("c", _OK)]
        assert self._run(tasks, tmp_path) == set()

    def test_failed_canary_still_fans_out_and_is_reported(self, tmp_path, caplog):
        # A bad canary does not stop the rest; it is collected. Narration flows
        # through the batch logger, so assert against caplog, not stdout.
        caplog.set_level(logging.INFO)
        tasks = [("a", _FAIL), ("b", _OK), ("c", _OK)]
        assert self._run(tasks, tmp_path) == {"a"}
        assert "continuing" in caplog.text

    def test_collects_all_failures(self, tmp_path):
        tasks = [("a", _OK), ("b", _FAIL), ("c", _FAIL)]
        assert self._run(tasks, tmp_path) == {"b", "c"}


# ---------------------------------------------------------------------------
# run_stage -- fail-fast (abort_on_failure=True)
# ---------------------------------------------------------------------------


class TestRunStageFailFast:
    def _run(self, tasks, tmp_path):
        return f.run_stage("job", tasks, 2, str(tmp_path), abort_on_failure=True)

    def test_all_succeed_returns_empty_set(self, tmp_path):
        tasks = [("a", _OK), ("b", _OK)]
        assert self._run(tasks, tmp_path) == set()

    def test_failed_canary_aborts(self, tmp_path):
        tasks = [("a", _FAIL), ("b", _OK)]
        with pytest.raises(SystemExit) as exc:
            self._run(tasks, tmp_path)
        assert exc.value.code == 1

    def test_pool_failure_aborts(self, tmp_path):
        tasks = [("a", _OK), ("b", _FAIL), ("c", _OK)]
        with pytest.raises(SystemExit) as exc:
            self._run(tasks, tmp_path)
        assert exc.value.code == 1


# ---------------------------------------------------------------------------
# run_stage -- per-job timeout kill
# ---------------------------------------------------------------------------


class TestRunStageTimeout:
    # The timeouts here are sub-second floats on purpose: _run_one hands `timeout`
    # straight to Popen.communicate(), which takes floats, so whole-second values
    # buy nothing but wall clock.

    def test_slow_fanout_job_is_killed_and_counts_as_failure(self, tmp_path):
        # A job that overruns job_timeout is killed and reported as a failure, so
        # one stuck unit can't hang the batch. The 30s sleeper is bounded to 0.3s.
        start = time.monotonic()
        failed = f.run_stage(
            "job",
            [("canary", _OK), ("slow", _SLEEP)],
            2,
            str(tmp_path),
            job_timeout=0.3,
            abort_on_failure=False,
        )
        assert failed == {"slow"}
        assert time.monotonic() - start < 5  # killed at ~0.3s, not the full 30s

    def test_timed_out_job_returns_the_124_sentinel(self, tmp_path):
        # `failed == {"slow"}` above cannot tell a timeout kill from a job that
        # crashed for any other reason; 124 and its note are what say "killed".
        rc, err = f._run_one(_SLEEP, timeout=0.3)
        assert rc == 124
        assert "timed out after" in err

    def test_canary_uses_canary_timeout_not_job_timeout(self, tmp_path):
        # job_timeout bounds only the fan-out; the canary keeps its own, larger
        # limit, else the cold-cache canary would die on every real run.
        slow_canary = [sys.executable, "-c", "import time; time.sleep(0.5)"]
        failed = f.run_stage(
            "job",
            [("canary", slow_canary), ("b", _OK)],
            2,
            str(tmp_path),
            job_timeout=0.2,
            canary_timeout=30,
            abort_on_failure=False,
        )
        assert failed == set()  # the 0.5s canary survived a 0.2s job_timeout


# ---------------------------------------------------------------------------
# _report_failures + _run_one guard
# ---------------------------------------------------------------------------


class TestReportFailures:
    def test_prints_header_hint_and_stderr_tail(self, tmp_path, caplog):
        # The sentinels flow through the batch logger, so assert on caplog.
        caplog.set_level(logging.INFO)
        failures = [("science", "KP.x", 1, "boom line 1\nboom line 2")]
        f._report_failures(failures, str(tmp_path), header="WARNING: 1 failed")
        text = caplog.text
        assert "WARNING: 1 failed" in text
        assert "FAILED [science] KP.x (exit 1)" in text
        assert "kpf_science_KP.x_" in text  # the log-path hint
        assert "boom line 2" in text  # the stderr tail


class TestRunOneInterrupt:
    def test_returns_130_without_launching_when_interrupted(self):
        # The pre-launch guard: once teardown has begun, _run_one never spawns.
        f._interrupted.set()
        assert f._run_one(_OK) == (130, "")

    def test_interrupt_in_launch_window_kills_child(self, monkeypatch):
        # The launch-vs-track race: the interrupt lands after Popen but before the
        # child is tracked, so the top-of-function guard is clear and the post-track
        # re-check must catch it -- else the child runs its full 30s sleep untracked.
        real_popen = f.subprocess.Popen
        launched = []

        def popen_then_interrupt(*a, **k):
            proc = real_popen(*a, **k)
            launched.append(proc)
            f._interrupted.set()  # interrupt arrives in the launch/track window
            return proc

        monkeypatch.setattr(f.subprocess, "Popen", popen_then_interrupt)
        rc, _ = f._run_one(_SLEEP, timeout=None)
        assert rc == 130
        assert launched and launched[0].poll() is not None  # child killed + reaped


# ---------------------------------------------------------------------------
# run_stage -- interrupt teardown
# ---------------------------------------------------------------------------


class TestRunStageInterrupt:
    """The module's stated reason for existing (`run_stage`'s docstring: an
    interrupt leaves no orphaned subprocesses). No real processes are spawned:
    _run_one is replaced by the interrupt itself, and the escalation test drives
    fakes, so nothing here can outlive the test."""

    def test_interrupt_tears_down_children_and_exits_130(self, monkeypatch, tmp_path):
        torn_down = []

        def _interrupt(*_a, **_k):
            raise KeyboardInterrupt

        monkeypatch.setattr(f, "_run_one", _interrupt)
        monkeypatch.setattr(f, "_terminate_all_children", lambda: torn_down.append(1))

        with pytest.raises(SystemExit) as exc:
            f.run_stage("job", [("a", _OK)], 2, str(tmp_path))

        assert exc.value.code == 130
        assert f._interrupted.is_set()  # stops the pool launching anything more
        assert torn_down == [1]

    def test_terminate_escalates_sigterm_then_sigkill(self, monkeypatch):
        # A child that dies on SIGTERM is left alone; one still alive after the
        # grace period gets SIGKILL. Dropping the escalation would leave a wedged
        # recipe running after the orchestrator exits.
        signalled = []

        class _FakeProc:
            def __init__(self, pid, survives):
                self.pid = pid
                self._survives = survives

            def wait(self, timeout=None):
                return 0

            def poll(self):
                return None if self._survives else 0

        dies, survives = _FakeProc(101, False), _FakeProc(102, True)
        monkeypatch.setattr(f, "_live_procs", {dies, survives})
        monkeypatch.setattr(
            f.os, "killpg", lambda pid, sig: signalled.append((pid, sig))
        )

        f._terminate_all_children(grace=0.0)

        # Both children are SIGTERMed first (in set order), then only the
        # survivor is SIGKILLed -- assert the sequence, not a sorted set, since
        # TERM-before-KILL is the property under test.
        assert set(signalled[:2]) == {(101, signal.SIGTERM), (102, signal.SIGTERM)}
        assert signalled[2:] == [(102, signal.SIGKILL)]


# ---------------------------------------------------------------------------
# configure_runtime
# ---------------------------------------------------------------------------


class TestConfigureRuntime:
    def test_installs_sigterm_handler_and_pins_blas_threads(self, monkeypatch):
        # `f.signal` IS the singleton signal module, so this patch is
        # process-wide for the test's duration -- containment comes from
        # monkeypatch's TEARDOWN, not from any lexical scoping. Do not simplify
        # it away, and do not let this test install a real handler: a leaked
        # SIGTERM handler would turn an xdist worker's own shutdown into a
        # KeyboardInterrupt.
        installed = []
        monkeypatch.setattr(
            f.signal, "signal", lambda sig, handler: installed.append((sig, handler))
        )
        monkeypatch.setenv("OMP_NUM_THREADS", "8")  # an explicit operator setting
        for var in (
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            monkeypatch.delenv(var, raising=False)

        f.configure_runtime()

        assert installed == [(signal.SIGTERM, f._handle_termination_signal)]
        assert os.environ["OMP_NUM_THREADS"] == "8"  # setdefault: the caller wins
        assert os.environ["MKL_NUM_THREADS"] == "1"
        assert os.environ["VECLIB_MAXIMUM_THREADS"] == "1"
