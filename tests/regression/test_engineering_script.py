"""Tests for scripts/process/engineering.py: the engineering-reduction driver.

Covers the driver's own surface: arg parsing, the ``_cli_task`` argv it fans out,
and the ``main`` exit-code contract (nonzero iff at least one frame failed). The
``--obs_ids`` expansion is tested in test_script_helpers.py (``resolve_obs_ids``),
the fan-out engine in ``_dispatch``. Stubs only -- no real testdata needed.
"""

import sys

import pytest

from scripts.process import engineering as _engineering

from ._scripts import _FakeConfig, _NoLogDirConfig, _stub_batch_log

# scripts/CLI/tools-layer suite: excluded from `make test-fast`.
pytestmark = pytest.mark.cli

_OID1 = "KP.20240405.40113.57"
_OID2 = "KP.20240405.40237.36"


@pytest.fixture(scope="module")
def e():
    return _engineering


# ------------------------------------------------------------------
# parse_args
# ------------------------------------------------------------------


class TestParseArgs:
    def test_obs_ids(self, e):
        assert e.parse_args(["--obs_ids", _OID2, _OID1]).obs_ids == [_OID1, _OID2]

    @pytest.mark.parametrize(
        "argv",
        [
            [],  # one of --obs_ids/--dates/--date_range required
            ["--obs_ids", _OID1, "--dates", "20240405"],  # more than one
            ["--obs_ids", "not-an-obs-id"],
            ["--obs_ids", _OID1, "--jobs", "0"],
            ["--obs_ids", _OID1, "--job_timeout", "0"],
            ["--obs_ids", _OID1, "-r", "/x.py"],  # fixed recipe; no -r
            ["--obs_ids", _OID1, "--stages", "astro_query"],  # a science stage
            ["--obs_ids", _OID1, "--kpf_science_output", "/s"],  # no science output
            ["--obs_ids", _OID1, "--cache", "r"],  # no mini-db pre-scan
        ],
    )
    def test_invalid_args_exit(self, e, argv):
        with pytest.raises(SystemExit):
            e.parse_args(argv)

    def test_dates_and_date_range(self, e):
        assert e.parse_args(["--dates", "20240405"]).dates == ["20240405"]
        ns = e.parse_args(["--date_range", "20240401", "20240430"])
        assert ns.date_range == ["20240401", "20240430"] and ns.obs_ids is None

    def test_jobs_unset_resolves_to_cores_default(self, e):
        assert e.parse_args(["--obs_ids", _OID1]).jobs == e._default_frame_jobs()

    def test_output_dir_fans_out(self, e):
        # engineering has masters + engineering output and a log dir.
        ns = e.parse_args(["--obs_ids", _OID1, "--output_dir", "/out"])
        assert ns.kpf_masters_output == "/out"
        assert ns.kpf_engineering_output == "/out"
        assert ns.log_dir == "/out/logs"


# ------------------------------------------------------------------
# _cli_task
# ------------------------------------------------------------------


class TestCliTask:
    def test_builds_engineering_argv_with_defaults(self, e):
        tag, argv = e._cli_task(_OID1, ["--log_level", "DEBUG"])
        assert tag == _OID1
        assert argv == [
            sys.executable, "-m", "scripts.process.reduce",
            "-r", e.DEFAULT_ENGINEERING_RECIPE, "-c", e.DEFAULT_ENGINEERING_CONFIG,
            "-o", _OID1, "--log_level", "DEBUG",
        ]  # fmt: skip

    def test_config_override_keeps_default_recipe(self, e):
        _, argv = e._cli_task(_OID1, [], config="/c.toml")
        assert argv[3:7] == ["-r", e.DEFAULT_ENGINEERING_RECIPE, "-c", "/c.toml"]


# ------------------------------------------------------------------
# main exit-code contract
# ------------------------------------------------------------------


class TestMain:
    def _patch(self, e, monkeypatch, tmp_path, failed, calls=None):
        monkeypatch.setattr(e, "configure_runtime", lambda: None)
        monkeypatch.setattr(e, "ConfigHandler", _FakeConfig)
        _stub_batch_log(monkeypatch, e, tmp_path)

        def fake_run_stage(*args, **kwargs):
            if calls is not None:
                calls.append((args, kwargs))
            return set(failed)

        monkeypatch.setattr(e, "run_stage", fake_run_stage)

    def test_every_frame_is_reduced(self, e, monkeypatch, tmp_path):
        # No junk/single-CCD gate: every requested frame reaches the fan-out.
        calls = []
        self._patch(e, monkeypatch, tmp_path, failed=[], calls=calls)
        e.main(["--obs_ids", _OID1, _OID2])
        assert [tag for tag, _ in calls[0][0][1]] == [_OID1, _OID2]

    def test_exits_zero_when_all_reduced(self, e, monkeypatch, tmp_path):
        self._patch(e, monkeypatch, tmp_path, failed=[])
        e.main(["--obs_ids", _OID1])  # no SystemExit

    def test_exits_nonzero_when_any_failed(self, e, monkeypatch, tmp_path):
        self._patch(e, monkeypatch, tmp_path, failed=[_OID1])
        with pytest.raises(SystemExit) as exc:
            e.main(["--obs_ids", _OID1])
        assert exc.value.code == 1

    def test_fan_out_is_fail_soft_and_paced(self, e, monkeypatch, tmp_path):
        calls = []
        self._patch(e, monkeypatch, tmp_path, failed=[], calls=calls)
        e.main(["--obs_ids", _OID1, "--job_timeout", "42"])
        args, kwargs = calls[0]
        assert args[0] == "engineering"
        assert kwargs["abort_on_failure"] is False
        assert kwargs["job_timeout"] == 42
        assert kwargs["launch_interval"] == e._LAUNCH_INTERVAL > 0

    def test_forwards_dir_log_and_stage_overrides_to_each_child(
        self, e, monkeypatch, tmp_path
    ):
        calls = []
        self._patch(e, monkeypatch, tmp_path, failed=[], calls=calls)
        e.main(
            [
                "--obs_ids", _OID1,
                "--input_dir", "/in",
                "--output_dir", "/out",
                "--log_level", "DEBUG",
                "--stages", "image_assembly",
            ]
        )  # fmt: skip
        _, argv = calls[0][0][1][0]
        fwd = {argv[i]: argv[i + 1] for i in range(len(argv) - 1)}
        assert fwd["--kpf_data_input"] == "/in"
        assert fwd["--kpf_masters_output"] == "/out"
        assert fwd["--kpf_engineering_output"] == "/out"
        assert fwd["--log_level"] == "DEBUG"
        assert fwd["--log_dir"] == "/out/logs"
        assert fwd["--run_id"] == "engineering_x"
        assert argv[-2:] == ["--stages", "image_assembly"]

    def test_no_stages_forwarded_by_default(self, e, monkeypatch, tmp_path):
        calls = []
        self._patch(e, monkeypatch, tmp_path, failed=[], calls=calls)
        e.main(["--obs_ids", _OID1])
        assert "--stages" not in calls[0][0][1][0][1]

    def test_dates_reduce_every_frame_of_the_night(self, e, monkeypatch, tmp_path):
        # No Target filter and no gate: every L0 frame of the night is dispatched.
        calls, seen = [], {}
        self._patch(e, monkeypatch, tmp_path, failed=[], calls=calls)

        def fake_nights(data_input, nights):
            seen.update(data_input=data_input, nights=nights)
            return [_OID1, _OID2]

        monkeypatch.setattr(e, "obs_ids_in_nights", fake_nights)
        e.main(["--dates", "20240405"])
        assert seen == {"data_input": "/in", "nights": ["20240405"]}
        assert [tag for tag, _ in calls[0][0][1]] == [_OID1, _OID2]

    def test_date_range_expands_to_the_nights_on_disk(self, e, monkeypatch, tmp_path):
        calls, seen = [], {}
        self._patch(e, monkeypatch, tmp_path, failed=[], calls=calls)
        monkeypatch.setattr(
            e, "datecodes_in_range", lambda di, a, b: seen.update(r=(a, b)) or [a]
        )
        monkeypatch.setattr(e, "obs_ids_in_nights", lambda di, nights: [_OID1])
        e.main(["--date_range", "20240405", "20240406"])
        assert seen["r"] == ("20240405", "20240406")
        assert [tag for tag, _ in calls[0][0][1]] == [_OID1]

    def test_errors_when_log_dir_unset(self, e, monkeypatch):
        monkeypatch.setattr(e, "configure_runtime", lambda: None)
        monkeypatch.setattr(e, "ConfigHandler", _NoLogDirConfig)
        with pytest.raises(SystemExit) as exc:
            e.main(["--obs_ids", _OID1])
        assert "log directory" in str(exc.value)
