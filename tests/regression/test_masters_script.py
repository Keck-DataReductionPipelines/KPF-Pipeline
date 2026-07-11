"""Tests for scripts/processing/masters.py: the nightly-masters build driver.

Cover the driver's own surface: arg parsing and the two input forms, the
``_cli_task`` argv it fans out, datecode resolution, and the ``main`` exit-code
contract (nonzero iff at least one night failed). The shared fan-out engine
(``run_stage``, job sizing) lives in ``_dispatch`` and the ``datecode_dirs_in_range``
helper in ``kpfpipe.utils.io``; they are tested in test_dispatch.py / test_io.py.

Unit tests use synthetic dir trees in tmp_path -- no real testdata needed.
"""

import sys

import pytest

from scripts.processing import masters as _masters


class _FakeConfig:
    """Stand-in for ConfigHandler in the exit-code tests (no real file read)."""

    def __init__(self, path):
        pass

    def get_params(self, keys):
        return {"KPF_DATA_INPUT": "/in", "log_dir": "/l"}


class _NoLogDirConfig(_FakeConfig):
    """A config with a resolvable data input but no configured log_dir."""

    def get_params(self, keys):
        return {"KPF_DATA_INPUT": "/in"}  # no log_dir


@pytest.fixture(scope="module")
def m():
    """The masters driver module (a normal package import)."""
    return _masters


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_explicit_datecodes(self, m):
        ns = m.parse_args(["--dates", "20240712", "20240405"])
        assert ns.dates == ["20240405", "20240712"]  # sorted + deduped
        assert ns.date_range is None

    def test_dedupes_datecodes(self, m):
        ns = m.parse_args(["--dates", "20240405", "20240405"])
        assert ns.dates == ["20240405"]

    def test_dates_from_file(self, m, tmp_path):
        f = tmp_path / "nights.txt"
        f.write_text("20240405\n20250912\n\n20241011\n")  # blank line skipped
        ns = m.parse_args(["--dates", str(f)])
        assert ns.dates == ["20240405", "20241011", "20250912"]  # sorted

    def test_dates_mixes_inline_and_file(self, m, tmp_path):
        f = tmp_path / "nights.txt"
        f.write_text("20250912\n20240405\n")  # 20240405 also given inline
        ns = m.parse_args(["--dates", "20240405", "20240712", str(f)])
        assert ns.dates == ["20240405", "20240712", "20250912"]  # merged + deduped

    def test_dates_file_with_bad_datecode_errors(self, m, tmp_path):
        f = tmp_path / "nights.txt"
        f.write_text("20240405\nnotadate\n")
        with pytest.raises(SystemExit):
            m.parse_args(["--dates", str(f)])

    def test_dates_empty_file_errors(self, m, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("\n  \n")
        with pytest.raises(SystemExit):
            m.parse_args(["--dates", str(f)])

    def test_dates_entry_neither_datecode_nor_file_errors(self, m):
        with pytest.raises(SystemExit):
            m.parse_args(["--dates", "/no/such/file.txt"])

    def test_date_range(self, m):
        ns = m.parse_args(["--date_range", "20240101", "20240131"])
        assert ns.date_range == ["20240101", "20240131"]
        assert ns.dates is None

    def test_neither_form_errors(self, m):
        with pytest.raises(SystemExit):
            m.parse_args([])

    def test_both_forms_error(self, m):
        with pytest.raises(SystemExit):
            m.parse_args(
                ["--dates", "20240405", "--date_range", "20240101", "20240131"]
            )

    @pytest.mark.parametrize(
        "argv",
        [
            ["--dates", "2024"],  # malformed datecode (list form)
            ["--date_range", "2024", "20240131"],  # malformed datecode (range form)
            ["--date_range", "20240201", "20240101"],  # start > end
            ["--dates", "20240405", "--jobs", "0"],  # jobs below 1
            ["--dates", "20240405", "--job_timeout", "0"],  # timeout below 1
        ],
    )
    def test_invalid_args_exit(self, m, argv):
        with pytest.raises(SystemExit):
            m.parse_args(argv)

    def test_config_defaults_to_none(self, m):
        assert m.parse_args(["--dates", "20240405"]).config is None

    def test_recipe_defaults_to_none(self, m):
        assert m.parse_args(["--dates", "20240405"]).recipe is None

    def test_recipe_override_parses(self, m):
        ns = m.parse_args(["--dates", "20240405", "-r", "/x.py"])
        assert ns.recipe == "/x.py"

    def test_jobs_unset_resolves_to_masters_default(self, m):
        ns = m.parse_args(["--dates", "20240405"])
        assert ns.jobs == m._default_masters_jobs()

    def test_jobs_override(self, m):
        assert m.parse_args(["--dates", "20240405", "--jobs", "3"]).jobs == 3

    def test_job_timeout_default_and_override(self, m):
        assert m.parse_args(["--dates", "20240405"]).job_timeout == 600
        assert (
            m.parse_args(["--dates", "20240405", "--job_timeout", "120"]).job_timeout
            == 120
        )

    def test_input_dir_aliases_data_input(self, m):
        ns = m.parse_args(["--dates", "20240405", "--input_dir", "/in"])
        assert ns.kpf_data_input == "/in"

    def test_output_dir_fans_out(self, m):
        # masters has no science output or plot dir -- only masters output + log dir.
        ns = m.parse_args(["--dates", "20240405", "--output_dir", "/out"])
        assert ns.kpf_masters_output == "/out" and ns.log_dir == "/out/logs"
        assert not hasattr(ns, "kpf_science_output")


# ---------------------------------------------------------------------------
# _cli_task / resolve_datecodes
# ---------------------------------------------------------------------------


class TestCliTask:
    def test_builds_masters_argv_with_defaults(self, m):
        # No -r/-c override: the masters default recipe/config are passed explicitly.
        tag, argv = m._cli_task("20240405", ["--log_level", "DEBUG"])
        assert tag == "20240405"
        assert argv == [
            sys.executable, "-m", "scripts.processing.reduce",
            "-r", m.DEFAULT_MASTERS_RECIPE, "-c", m.DEFAULT_MASTERS_CONFIG,
            "-d", "20240405", "--log_level", "DEBUG",
        ]  # fmt: skip

    def test_recipe_and_config_overrides(self, m):
        _, argv = m._cli_task("20240405", [], config="/c.toml", recipe="/x.py")
        assert argv == [
            sys.executable, "-m", "scripts.processing.reduce",
            "-r", "/x.py", "-c", "/c.toml", "-d", "20240405",
        ]  # fmt: skip

    def test_recipe_override_keeps_default_config(self, m):
        _, argv = m._cli_task("20240405", [], recipe="/x.py")
        assert argv == [
            sys.executable, "-m", "scripts.processing.reduce",
            "-r", "/x.py", "-c", m.DEFAULT_MASTERS_CONFIG, "-d", "20240405",
        ]  # fmt: skip

    def test_config_override_keeps_default_recipe(self, m):
        _, argv = m._cli_task("20240405", [], config="/c.toml")
        assert argv == [
            sys.executable, "-m", "scripts.processing.reduce",
            "-r", m.DEFAULT_MASTERS_RECIPE, "-c", "/c.toml", "-d", "20240405",
        ]  # fmt: skip


class TestResolveDatecodes:
    def test_explicit_list_passes_through(self, m, tmp_path):
        args = m.parse_args(["--dates", "20240405", "20240712"])
        # data_input is ignored for the explicit-list form.
        assert m.resolve_datecodes(args, str(tmp_path)) == ["20240405", "20240712"]

    def test_range_scans_l0_tree(self, m, tmp_path):
        l0 = tmp_path / "L0"
        for name in ["20240101", "20240115", "20240201"]:
            (l0 / name).mkdir(parents=True)
        args = m.parse_args(["--date_range", "20240101", "20240131"])
        assert m.resolve_datecodes(args, str(tmp_path)) == ["20240101", "20240115"]

    def test_range_missing_l0_root_exits(self, m, tmp_path):
        args = m.parse_args(["--date_range", "20240101", "20240131"])
        with pytest.raises(SystemExit):
            m.resolve_datecodes(args, str(tmp_path))  # no L0/ dir

    def test_range_no_nights_in_range_exits(self, m, tmp_path):
        (tmp_path / "L0" / "20250101").mkdir(parents=True)
        args = m.parse_args(["--date_range", "20240101", "20240131"])
        with pytest.raises(SystemExit):
            m.resolve_datecodes(args, str(tmp_path))


# ---------------------------------------------------------------------------
# main exit-code contract
# ---------------------------------------------------------------------------


class TestMainExitCode:
    def _patch(self, m, monkeypatch, failed, calls=None):
        # Skip the runtime setup and the real dir/config resolution + subprocess
        # fan-out; assert only the exit-code contract from run_stage's failure set.
        # setup_batch_logging is stubbed so main() writes no real batch log file.
        # `calls`, if given, records run_stage's kwargs for wiring assertions.
        monkeypatch.setattr(m, "configure_runtime", lambda: None)
        monkeypatch.setattr(m, "ConfigHandler", _FakeConfig)
        monkeypatch.setattr(m, "setup_batch_logging", lambda *a, **k: "/l/x.log")
        monkeypatch.setattr(m, "resolve_datecodes", lambda args, di: ["20240405"])
        monkeypatch.setattr(m, "warm_mini_db_caches", lambda *a, **k: (0, 0))

        def _fake_run_stage(*a, **k):
            if calls is not None:
                calls.append(k)
            return set(failed)

        monkeypatch.setattr(m, "run_stage", _fake_run_stage)

    def test_exits_zero_when_all_built(self, m, monkeypatch):
        self._patch(m, monkeypatch, failed=[])
        m.main(["--dates", "20240405"])  # no SystemExit

    def test_exits_nonzero_when_any_failed(self, m, monkeypatch):
        self._patch(m, monkeypatch, failed=["20240405"])
        with pytest.raises(SystemExit) as exc:
            m.main(["--dates", "20240405"])
        assert exc.value.code == 1

    def test_fan_out_is_staggered(self, m, monkeypatch):
        # Masters must pass its stagger interval to run_stage so the lockstep
        # disk-read wave is desynchronized (see _LAUNCH_INTERVAL).
        calls = []
        self._patch(m, monkeypatch, failed=[], calls=calls)
        m.main(["--dates", "20240405"])
        assert calls[0]["launch_interval"] == m._LAUNCH_INTERVAL
        assert m._LAUNCH_INTERVAL > 0

    def test_prescans_before_fan_out(self, m, monkeypatch):
        # The mini-db caches are warmed up front (parallel per-datecode) before the
        # reduces fan out, with the resolved datecodes and the pool size.
        order = []
        warm_args = {}

        def _warm(data_input, datecodes, jobs):
            order.append("warm")
            warm_args.update(datecodes=datecodes, jobs=jobs)
            return (len(datecodes), 0)

        self._patch(m, monkeypatch, failed=[])
        monkeypatch.setattr(m, "warm_mini_db_caches", _warm)
        monkeypatch.setattr(
            m, "run_stage", lambda *a, **k: order.append("run_stage") or set()
        )
        m.main(["--dates", "20240405"])
        assert order == ["warm", "run_stage"]  # pre-scan precedes fan-out
        assert warm_args["datecodes"] == ["20240405"]
        assert warm_args["jobs"] == m._default_masters_jobs()

    def test_errors_when_log_dir_unset(self, m, monkeypatch):
        # A missing log_dir is fatal before any fan-out (DRP-RUN-07).
        monkeypatch.setattr(m, "configure_runtime", lambda: None)
        monkeypatch.setattr(m, "ConfigHandler", _NoLogDirConfig)
        with pytest.raises(SystemExit) as exc:
            m.main(["--dates", "20240405"])
        assert "log directory" in str(exc.value)
