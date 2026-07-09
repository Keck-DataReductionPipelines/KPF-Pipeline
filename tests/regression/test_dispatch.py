"""Tests for scripts/processing/_dispatch.py: the shared fan-out engine.

Covers the job-sizing helpers (cores-based and the masters cap), the
``run_stage`` dispatch in both modes (fail-soft: attempt all, return the failed
set; fail-fast: any failure aborts the run with exit 1), the failure sentinel,
and the ``_run_one`` interrupt guard. Trivial subprocess stubs -- no real
testdata needed. The launch throttle (``launch_interval``) defaults to 0 here so
the dispatch tests stay fast and free of thread-timing flakiness.
"""

import logging
import sys

import pytest

from scripts.processing import _dispatch as f

_OK = [sys.executable, "-c", "pass"]
_FAIL = [sys.executable, "-c", "import sys; sys.exit(1)"]


@pytest.fixture(autouse=True)
def _clear_interrupted():
    """Keep the module-global interrupt flag clean around each test."""
    f._interrupted.clear()
    yield
    f._interrupted.clear()


# ---------------------------------------------------------------------------
# job sizing
# ---------------------------------------------------------------------------


class TestDefaultJobs:
    @pytest.mark.parametrize(
        "cpus,expected",
        [(None, 1), (1, 1), (8, 8), (64, 16), (100, 25)],
    )
    def test_cap(self, monkeypatch, cpus, expected):
        monkeypatch.setattr(f.os, "cpu_count", lambda: cpus)
        assert f._default_jobs() == expected


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
        assert f._default_masters_jobs() == f._default_jobs() == 8

    def test_unknown_ram_uses_cores_floor_only(self, monkeypatch):
        monkeypatch.setattr(f.os, "cpu_count", lambda: 256)

        def _raise(_name):
            raise ValueError("SC_PHYS_PAGES unavailable")

        monkeypatch.setattr(f.os, "sysconf", _raise)
        assert f._default_masters_jobs() == f._MASTERS_JOBS


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
        # A bad canary does not stop the rest; it is collected. The narration now
        # flows through the batch logger, so assert against caplog, not stdout.
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
# _report_failures + _run_one guard
# ---------------------------------------------------------------------------


class TestReportFailures:
    def test_prints_header_hint_and_stderr_tail(self, tmp_path, caplog):
        # The sentinels now flow through the batch logger, so assert on caplog.
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
