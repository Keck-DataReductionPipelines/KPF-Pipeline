"""Tests for the shared scripts/ helpers: _argparse.py, _dispatch.py, _scan.py.

The three modules the processing and analysis scripts both build on: the argparse
parent parsers each command composes via ``parents=[...]``, the subprocess fan-out
engine behind the batch orchestrators, and the up-front parallel-by-datecode L0
mini-db cache warming. Parsers are driven the way the commands drive them; dispatch
uses trivial subprocess stubs; scan writes synthetic FITS into temp trees. No
testdata needed.
"""

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from scripts import _argparse, _dispatch, _scan

from ._scripts import write_l0_tree

# scripts/CLI/tools-layer suite: excluded from `make test-fast`.
pytestmark = pytest.mark.cli

_OK = [sys.executable, "-c", "pass"]
_FAIL = [sys.executable, "-c", "import sys; sys.exit(1)"]
_SLEEP = [sys.executable, "-c", "import time; time.sleep(30)"]  # a wedged job


@pytest.fixture(autouse=True)
def _clear_interrupted():
    """Keep the module-global interrupt flag clean around each test.

    run_stage resets its own state now; the tests that set the flag and call
    `_run_one` directly still need this.
    """
    _dispatch._interrupted.clear()
    yield
    _dispatch._interrupted.clear()


# ===========================================================================
# _argparse.py -- shared argparse parent parsers
# ===========================================================================


def _parse(parents, argv):
    return argparse.ArgumentParser(parents=parents).parse_args(argv)


class TestRecipeParser:
    def test_adds_recipe_and_config(self):
        ns = _parse(
            [_argparse.recipe_and_config_parser()], ["-r", "/x.py", "-c", "/y.toml"]
        )
        assert ns.recipe == "/x.py"
        assert ns.config == "/y.toml"

    def test_long_options(self):
        ns = _parse(
            [_argparse.recipe_and_config_parser()],
            ["--recipe", "/x.py", "--config", "/y.toml"],
        )
        assert (ns.recipe, ns.config) == ("/x.py", "/y.toml")

    def test_default_none(self):
        ns = _parse([_argparse.recipe_and_config_parser()], [])
        assert ns.recipe is None and ns.config is None


class TestDataDirsParser:
    def test_includes_science_output_by_default(self):
        ns = _parse(
            [_argparse.data_dirs_parser()],
            ["--kpf_data_input", "/in", "--kpf_masters_output", "/m",
             "--kpf_science_output", "/s"],
        )  # fmt: skip
        assert ns.kpf_data_input == "/in"
        assert ns.kpf_masters_output == "/m"
        assert ns.kpf_science_output == "/s"

    def test_omits_science_output_when_disabled(self):
        parser = argparse.ArgumentParser(
            parents=[_argparse.data_dirs_parser(science_output=False)]
        )
        # Masters produces no science output, so the flag is neither set nor accepted.
        ns = parser.parse_args(["--kpf_data_input", "/in"])
        assert not hasattr(ns, "kpf_science_output")
        with pytest.raises(SystemExit):
            parser.parse_args(["--kpf_science_output", "/s"])

    def test_input_dir_aliases_data_input(self):
        ns = _parse([_argparse.data_dirs_parser()], ["--input_dir", "/in"])
        assert ns.kpf_data_input == "/in"

    def test_output_dir_parses_to_its_own_dest(self):
        ns = _parse([_argparse.data_dirs_parser()], ["--output_dir", "/out"])
        assert ns.output_dir == "/out"
        # Still a raw value here; the fan-out happens in resolve_dir_shortcuts.
        assert ns.kpf_masters_output is None and ns.kpf_science_output is None


class TestResolveDirShortcuts:
    def _parser(self, *, science_output=True, plot_dir=False):
        p = argparse.ArgumentParser(
            parents=[
                _argparse.data_dirs_parser(science_output=science_output),
                _argparse.logging_parser(),
            ]
        )
        if plot_dir:
            p.add_argument("--plot_dir", default=None)
        return p

    def test_fans_out_to_all_unset_slots(self):
        ns = _argparse.resolve_dir_shortcuts(
            self._parser(plot_dir=True).parse_args(["--output_dir", "/out"])
        )
        assert ns.kpf_masters_output == "/out"
        assert ns.kpf_science_output == "/out"
        assert ns.log_dir == "/out/logs"
        assert ns.plot_dir == "/out/QLP/timeseries"
        assert ns.kpf_data_input is None  # --output_dir never touches the input dir

    def test_explicit_flags_win(self):
        ns = _argparse.resolve_dir_shortcuts(
            self._parser(plot_dir=True).parse_args(
                [
                    "--output_dir",
                    "/out",
                    "--kpf_masters_output",
                    "/m",
                    "--plot_dir",
                    "/p",
                ]
            )
        )
        assert ns.kpf_masters_output == "/m"
        assert ns.plot_dir == "/p"
        assert ns.kpf_science_output == "/out" and ns.log_dir == "/out/logs"

    def test_skips_absent_slots(self):
        # A masters-style parser has no science output or plot dir: absent slots
        # must be skipped, not raise AttributeError.
        ns = _argparse.resolve_dir_shortcuts(
            self._parser(science_output=False).parse_args(["--output_dir", "/out"])
        )
        assert ns.kpf_masters_output == "/out" and ns.log_dir == "/out/logs"
        assert not hasattr(ns, "kpf_science_output")

    def test_noop_without_output_dir(self):
        ns = _argparse.resolve_dir_shortcuts(
            self._parser().parse_args(["--kpf_masters_output", "/m"])
        )
        assert ns.kpf_masters_output == "/m"
        assert ns.kpf_science_output is None and ns.log_dir is None


class TestLoggingParser:
    def test_adds_log_flags(self):
        ns = _parse(
            [_argparse.logging_parser()],
            ["--log_dir", "/l", "--log_level", "DEBUG", "--run_id", "masters_x"],
        )
        assert ns.log_dir == "/l" and ns.log_level == "DEBUG"
        assert ns.run_id == "masters_x"

    def test_run_id_defaults_to_none(self):
        assert _parse([_argparse.logging_parser()], []).run_id is None

    def test_output_dir_does_not_fill_run_id(self):
        # --output_dir names the parent; only a parent *script* sets the run id.
        ns = _argparse.resolve_dir_shortcuts(
            _parse(
                [_argparse.data_dirs_parser(), _argparse.logging_parser()],
                ["--output_dir", "/out"],
            )
        )
        assert ns.log_dir == "/out/logs" and ns.run_id is None


class TestResolveLogSettings:
    """CLI over [LOGGER], with the parent-relative path hazard closed."""

    def _args(self, log_dir=None, log_level=None):
        return _parse(
            [_argparse.logging_parser()],
            [a for pair in (("--log_dir", log_dir), ("--log_level", log_level))
             for a in pair if pair[1]],
        )  # fmt: skip

    def test_config_supplies_both(self):
        got = _argparse.resolve_log_settings(
            self._args(), {"log_dir": "/l", "log_level": "DEBUG"}
        )
        assert got == ("/l", "DEBUG")

    def test_cli_wins_over_config(self):
        got = _argparse.resolve_log_settings(
            self._args("/cli", "WARNING"), {"log_dir": "/l", "log_level": "DEBUG"}
        )
        assert got == ("/cli", "WARNING")

    def test_level_defaults_to_info(self):
        assert _argparse.resolve_log_settings(self._args(), {"log_dir": "/l"})[1] == (
            "INFO"
        )

    def test_relative_log_dir_is_made_absolute(self):
        # The orchestrators forward this to children running from REPO_ROOT, not
        # the operator's cwd -- a relative path would mean two different places.
        log_dir, _ = _argparse.resolve_log_settings(self._args("rel/logs"), {})
        assert log_dir == os.path.join(os.getcwd(), "rel/logs")

    def test_unset_log_dir_exits(self):
        with pytest.raises(SystemExit, match="no log directory configured"):
            _argparse.resolve_log_settings(self._args(), {})


class TestPoolParser:
    def test_adds_jobs_and_timeout(self):
        ns = _parse(
            [_argparse.pool_parser(jobs_help="how many")],
            ["--jobs", "4", "--job_timeout", "120"],
        )
        assert ns.jobs == 4 and ns.job_timeout == 120

    def test_defaults(self):
        ns = _parse([_argparse.pool_parser(jobs_help="how many")], [])
        assert ns.jobs is None and ns.job_timeout == 1200

    def test_jobs_help_is_passed_through(self):
        parser = argparse.ArgumentParser(
            parents=[_argparse.pool_parser(jobs_help="SENTINEL help text")]
        )
        assert "SENTINEL help text" in parser.format_help()


class TestCacheParser:
    """The --cache mode flag. Its parameterised default is what makes the leaf
    `reduce` read-only while the orchestrators own cache writing -- the
    one-writer-per-cache-file invariant documented at _scan.py:1-15."""

    def test_factory_default_is_read_only(self):
        assert _parse([_argparse.cache_parser()], []).cache == "r"

    def test_orchestrator_default_is_honoured(self):
        assert _parse([_argparse.cache_parser(default="rw")], []).cache == "rw"

    @pytest.mark.parametrize("mode", ["r", "w", "rw", "wr"])
    def test_every_choice_parses(self, mode):
        assert _parse([_argparse.cache_parser()], ["--cache", mode]).cache == mode

    def test_unknown_mode_rejected(self):
        with pytest.raises(SystemExit):
            _parse([_argparse.cache_parser()], ["--cache", "rr"])


# ===========================================================================
# _dispatch.py -- shared subprocess fan-out engine
# ===========================================================================

# ---------------------------------------------------------------------------
# job sizing
# ---------------------------------------------------------------------------


class TestDefaultScienceJobs:
    @pytest.mark.parametrize(
        "cpus,expected",
        [(None, 1), (1, 1), (8, 8), (64, 16), (100, 25)],
    )
    def test_cap(self, monkeypatch, cpus, expected):
        monkeypatch.setattr(_dispatch.os, "cpu_count", lambda: cpus)
        assert _dispatch._default_science_jobs() == expected


class TestDefaultMastersJobs:
    @staticmethod
    def _fake_sysconf(ram_gib):
        page = 4096  # SC_PHYS_PAGES * SC_PAGE_SIZE = total bytes; 4 KiB page.
        pages = int(ram_gib * 2**30) // page
        return lambda name: {"SC_PHYS_PAGES": pages, "SC_PAGE_SIZE": page}[name]

    def test_big_host_gets_fixed_cap(self, monkeypatch):
        monkeypatch.setattr(_dispatch.os, "cpu_count", lambda: 256)
        monkeypatch.setattr(_dispatch.os, "sysconf", self._fake_sysconf(2048))
        assert _dispatch._default_masters_jobs() == _dispatch._MASTERS_JOBS

    def test_ram_floors_below_fixed_cap(self, monkeypatch):
        monkeypatch.setattr(_dispatch.os, "cpu_count", lambda: 256)
        monkeypatch.setattr(_dispatch.os, "sysconf", self._fake_sysconf(24))
        assert _dispatch._default_masters_jobs() == 24 // _dispatch._MASTERS_JOB_GIB
        assert _dispatch._default_masters_jobs() < _dispatch._MASTERS_JOBS

    def test_cores_floor_below_fixed_cap(self, monkeypatch):
        monkeypatch.setattr(_dispatch.os, "cpu_count", lambda: 8)
        monkeypatch.setattr(_dispatch.os, "sysconf", self._fake_sysconf(256))
        assert (
            _dispatch._default_masters_jobs() == _dispatch._default_science_jobs() == 8
        )

    def test_unknown_ram_uses_cores_floor_only(self, monkeypatch):
        # 8 cores, not 256: at 256 the cores floor is 64 and never binds, so the
        # fixed cap answers and this test cannot see the branch it names.
        monkeypatch.setattr(_dispatch.os, "cpu_count", lambda: 8)

        def _raise(_name):
            raise ValueError("SC_PHYS_PAGES unavailable")

        monkeypatch.setattr(_dispatch.os, "sysconf", _raise)
        assert _dispatch._default_masters_jobs() == 8
        assert _dispatch._default_masters_jobs() < _dispatch._MASTERS_JOBS

    def test_never_below_one(self, monkeypatch):
        # Tiny RAM would floor the cap to 0; the helper clamps to 1 (max(1, ...)),
        # so the pool is never empty (an empty pool would silently do nothing).
        monkeypatch.setattr(_dispatch.os, "cpu_count", lambda: 8)
        monkeypatch.setattr(_dispatch.os, "sysconf", self._fake_sysconf(1))
        assert _dispatch._default_masters_jobs() == 1


# ---------------------------------------------------------------------------
# run_stage -- fail-soft (abort_on_failure=False)
# ---------------------------------------------------------------------------


class TestRunStageFailSoft:
    def _run(self, tasks, tmp_path):
        return _dispatch.run_stage(
            "job", tasks, 2, str(tmp_path), abort_on_failure=False
        )

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
        return _dispatch.run_stage(
            "job", tasks, 2, str(tmp_path), abort_on_failure=True
        )

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
        failed = _dispatch.run_stage(
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
        rc, err = _dispatch._run_one(_SLEEP, timeout=0.3)
        assert rc == 124
        assert "timed out after" in err

    def test_canary_uses_canary_timeout_not_job_timeout(self, tmp_path):
        # job_timeout bounds only the fan-out; the canary keeps its own, larger
        # limit, else the cold-cache canary would die on every real run.
        slow_canary = [sys.executable, "-c", "import time; time.sleep(0.5)"]
        failed = _dispatch.run_stage(
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
        _dispatch._report_failures(failures, str(tmp_path), header="WARNING: 1 failed")
        text = caplog.text
        assert "WARNING: 1 failed" in text
        assert "FAILED [science] KP.x (exit 1)" in text
        assert "kpf_science_KP.x_" in text  # the log-path hint
        assert "boom line 2" in text  # the stderr tail


class TestRunOneInterrupt:
    def test_returns_130_without_launching_when_interrupted(self):
        # The pre-launch guard: once teardown has begun, _run_one never spawns.
        _dispatch._interrupted.set()
        assert _dispatch._run_one(_OK) == (130, "")

    def test_interrupt_in_launch_window_kills_child(self, monkeypatch):
        # The launch-vs-track race: the interrupt lands after Popen but before the
        # child is tracked, so the top-of-function guard is clear and the post-track
        # re-check must catch it -- else the child runs its full 30s sleep untracked.
        real_popen = _dispatch.subprocess.Popen
        launched = []

        def popen_then_interrupt(*a, **k):
            proc = real_popen(*a, **k)
            launched.append(proc)
            _dispatch._interrupted.set()  # interrupt arrives in the launch/track window
            return proc

        monkeypatch.setattr(_dispatch.subprocess, "Popen", popen_then_interrupt)
        rc, _ = _dispatch._run_one(_SLEEP, timeout=None)
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

        monkeypatch.setattr(_dispatch, "_run_one", _interrupt)
        monkeypatch.setattr(
            _dispatch, "_terminate_all_children", lambda: torn_down.append(1)
        )

        with pytest.raises(SystemExit) as exc:
            _dispatch.run_stage("job", [("a", _OK)], 2, str(tmp_path))

        assert exc.value.code == 130
        assert _dispatch._interrupted.is_set()  # stops the pool launching anything more
        assert torn_down == [1]

    def test_stale_interrupt_flag_is_cleared_on_entry(self, tmp_path):
        # `_interrupted` is module state that outlives a run, so without the reset
        # a second run_stage in this process launches nothing at all: every
        # _run_one short-circuits to 130 and every task is reported failed.
        _dispatch._interrupted.set()
        failed = _dispatch.run_stage(
            "job", [("a", _OK), ("b", _OK)], 2, str(tmp_path), abort_on_failure=False
        )
        assert failed == set()

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
        monkeypatch.setattr(_dispatch, "_live_procs", {dies, survives})
        monkeypatch.setattr(
            _dispatch.os, "killpg", lambda pid, sig: signalled.append((pid, sig))
        )

        _dispatch._terminate_all_children(grace=0.0)

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
        # `_dispatch.signal` IS the singleton signal module, so this patch is
        # process-wide for the test's duration -- containment comes from
        # monkeypatch's TEARDOWN, not from any lexical scoping. Do not simplify
        # it away, and do not let this test install a real handler: a leaked
        # SIGTERM handler would turn an xdist worker's own shutdown into a
        # KeyboardInterrupt.
        installed = []
        monkeypatch.setattr(
            _dispatch.signal,
            "signal",
            lambda sig, handler: installed.append((sig, handler)),
        )
        monkeypatch.setenv("OMP_NUM_THREADS", "8")  # an explicit operator setting
        for var in (
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            monkeypatch.delenv(var, raising=False)

        _dispatch.configure_runtime()

        assert installed == [(signal.SIGTERM, _dispatch._handle_termination_signal)]
        assert os.environ["OMP_NUM_THREADS"] == "8"  # setdefault: the caller wins
        assert os.environ["MKL_NUM_THREADS"] == "1"
        assert os.environ["VECLIB_MAXIMUM_THREADS"] == "1"


# ===========================================================================
# _scan.py -- up-front L0 mini-db cache warming
# ===========================================================================


def _cache_path(data_input, datecode):
    return Path(data_input) / "vNext" / "mini_db" / f"{datecode}_L0.csv"


# ---------------------------------------------------------------------------
# scan_night_to_cache
# ---------------------------------------------------------------------------


class TestScanNightToCache:
    def test_scans_and_writes_cache(self, tmp_path):
        write_l0_tree(str(tmp_path), "20240101", 3600)
        write_l0_tree(str(tmp_path), "20240101", 3700)

        df = _scan.scan_night_to_cache(str(tmp_path), "20240101")

        assert df is not None
        assert len(df) == 2
        assert _cache_path(str(tmp_path), "20240101").is_file()  # default cache="rw"

    def test_read_only_mode_does_not_write(self, tmp_path):
        # Recipes read the cache; only the scripts layer writes it.
        write_l0_tree(str(tmp_path), "20240101", 3600)

        df = _scan.scan_night_to_cache(str(tmp_path), "20240101", cache="r")

        assert df is not None and len(df) == 1
        assert not _cache_path(str(tmp_path), "20240101").exists()

    def test_empty_night_returns_none(self, tmp_path):
        # An empty datecode dir raises ValueError inside build_mini_database; _scan
        # swallows it and returns None rather than aborting the batch.
        (Path(tmp_path) / "L0" / "20240101").mkdir(parents=True)
        assert _scan.scan_night_to_cache(str(tmp_path), "20240101") is None
        assert not _cache_path(str(tmp_path), "20240101").exists()

    def test_absent_night_returns_none(self, tmp_path):
        assert _scan.scan_night_to_cache(str(tmp_path), "20240101") is None


# ---------------------------------------------------------------------------
# scan_datecodes: generic parallel dispatcher
# ---------------------------------------------------------------------------


class TestScanDatecodes:
    def test_returns_per_night_results(self, tmp_path):
        datecodes = ["20240101", "20240102", "20240103"]
        results = _scan.scan_datecodes(datecodes, jobs=3, worker=lambda dc: (dc, ""))
        assert sorted(results) == datecodes  # order-agnostic: the pool is unordered

    def test_tolerates_jobs_exceeding_datecodes(self, tmp_path):
        results = _scan.scan_datecodes(["20240101"], jobs=8, worker=lambda dc: (dc, ""))
        assert results == ["20240101"]

    def test_empty_datecodes(self, tmp_path):
        assert _scan.scan_datecodes([], jobs=4, worker=lambda dc: (dc, "")) == []

    def test_threaded_scan_no_contamination(self, tmp_path):
        # Each night gets its own FileHandler inside scan_night_to_cache, so a pooled
        # scan never collapses nights via a shared self._mini_db.
        nights = [f"202401{d:02d}" for d in range(1, 7)]
        expected = {}
        for dc in nights:
            ids = {write_l0_tree(str(tmp_path), dc, 3600 + j * 100) for j in range(4)}
            expected[dc] = ids

        def _worker(dc):
            df = _scan.scan_night_to_cache(str(tmp_path), dc)
            obs_ids = {fn.split("/")[-1][:-5] for fn in df["FILENAME"]}
            return (frozenset(obs_ids), "")

        results = _scan.scan_datecodes(nights, jobs=8, worker=_worker)
        assert set(results) == {frozenset(v) for v in expected.values()}


# ---------------------------------------------------------------------------
# warm_mini_db_caches: side-effect entry point, fail-soft
# ---------------------------------------------------------------------------


class TestWarmMiniDbCaches:
    def test_writes_all_and_counts(self, tmp_path):
        nights = ["20240101", "20240102"]
        for dc in nights:
            write_l0_tree(str(tmp_path), dc, 3600)

        written, skipped = _scan.warm_mini_db_caches(str(tmp_path), nights, jobs=2)

        assert (written, skipped) == (2, 0)
        for dc in nights:
            assert _cache_path(str(tmp_path), dc).is_file()

    def test_read_only_mode_skips_prescan(self, tmp_path):
        # A read-only mode warms nothing: every night is reported skipped, unscanned.
        nights = ["20240101", "20240102"]
        for dc in nights:
            write_l0_tree(str(tmp_path), dc, 3600)

        written, skipped = _scan.warm_mini_db_caches(
            str(tmp_path), nights, jobs=2, cache="r"
        )

        assert (written, skipped) == (0, 2)
        for dc in nights:
            assert not _cache_path(str(tmp_path), dc).exists()

    def test_empty_night_counted_skipped(self, tmp_path):
        write_l0_tree(str(tmp_path), "20240101", 3600)  # good
        (Path(tmp_path) / "L0" / "20240102").mkdir(parents=True)  # empty

        written, skipped = _scan.warm_mini_db_caches(
            str(tmp_path), ["20240101", "20240102"], jobs=2
        )

        assert (written, skipped) == (1, 1)

    def test_fail_soft_on_pool_error(self, tmp_path, monkeypatch):
        # A pool-level failure must never abort the batch: warm reports every night
        # skipped so the reduces fall back to in-process scans.
        def _boom(datecodes, jobs, worker, *, label="scanning"):
            raise RuntimeError("pool exploded")

        monkeypatch.setattr(_scan, "scan_datecodes", _boom)
        written, skipped = _scan.warm_mini_db_caches(
            str(tmp_path), ["20240101", "20240102"], jobs=2
        )
        assert (written, skipped) == (0, 2)
