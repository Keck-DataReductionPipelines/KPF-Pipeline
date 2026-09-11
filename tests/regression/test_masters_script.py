"""Tests for scripts/processing/masters.py: the nightly-masters build driver.

Covers the driver's own surface: arg parsing and the two input forms, the
``_cli_task`` argv it fans out, datecode resolution, and the ``main`` exit-code
contract (nonzero iff at least one night failed). The shared fan-out engine and
the ``datecode_dirs_in_range`` helper are tested in test_script_helpers.py and
test_io.py.

Unit tests use synthetic dir trees in tmp_path -- no real testdata needed.
"""

import os
import sys
import tempfile

import pytest

from kpfpipe.utils import run_record as rr
from scripts.processing import masters as _masters

from ._scripts import _FakeConfig, _NoLogDirConfig

# scripts/CLI/tools-layer suite: excluded from `make test-fast`.
pytestmark = pytest.mark.cli


@pytest.fixture(scope="module")
def m():
    return _masters


def _stub_batch_log(monkeypatch, mod):
    """Stub setup_batch_logging with a real-looking .log path in a temp dir.

    The batch run.json sidecar is written beside the log, so the stub must return
    a writable path ending in .log (never a placeholder like /l/x.log).
    """
    base = tempfile.mkdtemp()
    fake_log = os.path.join(base, "logs", "20240405", "kpf_batch_x_20240405T000000.log")
    os.makedirs(os.path.dirname(fake_log), exist_ok=True)
    label = mod.__name__.rsplit(".", 1)[-1]
    monkeypatch.setattr(
        mod, "setup_batch_logging", lambda *a, **k: (f"{label}_x", fake_log)
    )
    return fake_log


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
        "argv,expected",
        [
            (["--dates", "2024"], "2024"),  # malformed datecode (list form)
            (
                ["--date_range", "2024", "20240131"],  # malformed datecode (range form)
                "2024",
            ),
            (["--date_range", "20240201", "20240101"], "20240201"),  # start > end
            (["--dates", "20240405", "--jobs", "0"], "jobs"),  # jobs below 1
            (
                ["--dates", "20240405", "--job_timeout", "0"],  # timeout below 1
                "job_timeout",
            ),
        ],
    )
    def test_invalid_args_exit(self, m, argv, expected, capsys):
        with pytest.raises(SystemExit):
            m.parse_args(argv)
        assert expected in capsys.readouterr().err

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
        assert m.parse_args(["--dates", "20240405"]).job_timeout == 1200
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
        with pytest.raises(SystemExit, match="L0 input directory not found"):
            m.resolve_datecodes(args, str(tmp_path))  # no L0/ dir

    def test_range_no_nights_in_range_exits(self, m, tmp_path):
        (tmp_path / "L0" / "20250101").mkdir(parents=True)
        args = m.parse_args(["--date_range", "20240101", "20240131"])
        with pytest.raises(SystemExit, match="no datecode dirs"):
            m.resolve_datecodes(args, str(tmp_path))


# ---------------------------------------------------------------------------
# main exit-code contract
# ---------------------------------------------------------------------------


class TestMainExitCode:
    def _patch(self, m, monkeypatch, failed, calls=None):
        # Stub out runtime setup, dir/config resolution and the subprocess fan-out
        # (setup_batch_logging included, so main() writes no real batch log), and
        # assert only what run_stage's failure set does. `calls`, if given, records
        # run_stage's kwargs for wiring assertions.
        monkeypatch.setattr(m, "configure_runtime", lambda: None)
        monkeypatch.setattr(m, "ConfigHandler", _FakeConfig)
        _stub_batch_log(monkeypatch, m)
        monkeypatch.setattr(m, "resolve_datecodes", lambda args, di: ["20240405"])
        monkeypatch.setattr(m, "warm_mini_db_caches", lambda *a, **k: (0, 0))

        def _fake_run_stage(*a, **k):
            if calls is not None:
                calls.append({"args": a, **k})
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
        # Masters passes its stagger interval to run_stage so the lockstep
        # disk-read wave is desynchronized.
        calls = []
        self._patch(m, monkeypatch, failed=[], calls=calls)
        m.main(["--dates", "20240405"])
        assert calls[0]["launch_interval"] == m._LAUNCH_INTERVAL
        assert m._LAUNCH_INTERVAL > 0

    def test_prescans_before_fan_out(self, m, monkeypatch):
        # The mini-db caches are warmed up front, before the reduces fan out.
        order = []
        warm_args = {}

        def _warm(data_input, datecodes, jobs, cache="rw"):
            order.append("warm")
            warm_args.update(
                data_input=data_input, datecodes=datecodes, jobs=jobs, cache=cache
            )
            return (len(datecodes), 0)

        self._patch(m, monkeypatch, failed=[])
        monkeypatch.setattr(m, "warm_mini_db_caches", _warm)
        monkeypatch.setattr(
            m, "run_stage", lambda *a, **k: order.append("run_stage") or set()
        )
        m.main(["--dates", "20240405"])
        assert order == ["warm", "run_stage"]
        assert warm_args["data_input"] == "/in"  # resolved L0 input root
        assert warm_args["datecodes"] == ["20240405"]
        assert warm_args["jobs"] == m._default_masters_jobs()
        assert warm_args["cache"] == "rw"  # masters default: warm up front

    def test_forwards_dir_and_log_overrides_to_each_child(self, m, monkeypatch):
        calls = []
        self._patch(m, monkeypatch, failed=[], calls=calls)
        m.main(
            [
                "--dates",
                "20240405",
                "--input_dir",
                "/in",
                "--output_dir",
                "/out",
                "--log_level",
                "DEBUG",
            ]
        )
        tasks = calls[0]["args"][1]
        assert len(tasks) == 1
        _, argv = tasks[0]
        # Looked up by flag rather than by position: this tail has churned twice.
        fwd = {argv[i]: argv[i + 1] for i in range(len(argv) - 1)}
        assert fwd["--kpf_data_input"] == "/in"
        assert fwd["--kpf_masters_output"] == "/out"
        assert fwd["--log_level"] == "DEBUG"
        # Both halves of the run directory, so the child joins the same one.
        assert fwd["--log_dir"] == "/out/logs"
        assert fwd["--run_id"] == "masters_x"

    def _batch_args(self, m, monkeypatch, argv, calls):
        """Run main(); return the (log_dir, label, run_id) setup_batch_logging got."""
        seen = []
        self._patch(m, monkeypatch, failed=[], calls=calls)
        # The batch run.json is written beside the log, so the stubbed path must
        # be writable (never a placeholder like /l/x.log).
        fake_log = os.path.join(tempfile.mkdtemp(), "kpf_masters_batch_x.log")
        monkeypatch.setattr(
            m,
            "setup_batch_logging",
            lambda d, label, rid=None, **k: (
                seen.append((d, label, rid))
                or (rid or "masters_20240405T010203", fake_log)
            ),
        )
        m.main(argv)
        return seen[0]

    def test_passes_the_parent_log_dir_and_mints_no_run_id(self, m, monkeypatch):
        # One run, one directory: --log_dir is the parent, not the destination.
        # Minting belongs to logger.py now, so masters passes no run id of its own.
        calls = []
        assert self._batch_args(
            m, monkeypatch, ["--dates", "20240405", "--log_dir", "/logs"], calls
        ) == ("/logs", "masters", None)
        # run_stage's failure hints point at the run directory, not the parent.
        assert calls[0]["args"][3] == "/logs/masters_20240405T010203"

    def test_forwarded_run_id_is_used_verbatim_and_reforwarded(self, m, monkeypatch):
        # A parent script's run id: joined, never nested inside a new one.
        calls = []
        parent = "timeseries_20240405T010203"
        log_dir, _, run_id = self._batch_args(
            m,
            monkeypatch,
            ["--dates", "20240405", "--log_dir", "/logs", "--run_id", parent],
            calls,
        )
        assert (log_dir, run_id) == ("/logs", parent)
        _, argv = calls[0]["args"][1][0]
        assert argv[-4:] == ["--log_dir", "/logs", "--run_id", parent]

    def test_errors_when_log_dir_unset(self, m, monkeypatch):
        # A missing log_dir is fatal before any fan-out.
        monkeypatch.setattr(m, "configure_runtime", lambda: None)
        monkeypatch.setattr(m, "ConfigHandler", _NoLogDirConfig)
        with pytest.raises(SystemExit) as exc:
            m.main(["--dates", "20240405"])
        assert "log directory" in str(exc.value)


# ---------------------------------------------------------------------------
# batch run.json sidecar
# ---------------------------------------------------------------------------


class TestBatchRunRecord:
    def test_batch_record_written_with_counts_and_children(
        self, m, monkeypatch, tmp_path
    ):
        log_dir = tmp_path / "logs"
        fake_log = log_dir / "20240405" / "kpf_masters_batch_20240405T000000.log"
        fake_log.parent.mkdir(parents=True)
        monkeypatch.setattr(m, "configure_runtime", lambda: None)
        monkeypatch.setattr(m, "ConfigHandler", _FakeConfig)
        monkeypatch.setattr(
            m, "setup_batch_logging", lambda *a, **k: ("masters_x", str(fake_log))
        )
        monkeypatch.setattr(m, "warm_mini_db_caches", lambda *a, **k: (0, 0))
        monkeypatch.setattr(rr, "git_sha", lambda repo_root=None: None)
        monkeypatch.delenv(rr.PARENT_ENV, raising=False)

        def fake_run_stage(label, tasks, jobs, log_dir_arg, **kw):
            # Children log straight into the run directory (one run, one dir).
            child = os.path.join(log_dir_arg, "kpf_masters_20240406_x.run.json")
            rr.write_json_atomic(
                child,
                {
                    "schema": rr.SCHEMA,
                    "kind": "run",
                    "status": "succeeded",
                    "target": "20240406",
                    "exit_status": 0,
                    "parent": os.environ.get(rr.PARENT_ENV),
                },
            )
            return set()

        monkeypatch.setattr(m, "run_stage", fake_run_stage)
        m.main(["--dates", "20240405", "20240406", "--log_dir", str(log_dir)])

        data = rr.read_run_record(rr.run_json_path(str(fake_log)))
        assert data["kind"] == "batch"
        assert data["recipe"] == "masters"
        assert data["status"] == "succeeded"
        assert data["counts"] == {"done": 2, "failed": 0, "skipped": 0}
        assert [c["tag"] for c in data["children"]] == ["20240406"]
        assert os.environ.get(rr.PARENT_ENV) is None
