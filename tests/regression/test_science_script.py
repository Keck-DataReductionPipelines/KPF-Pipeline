"""Tests for scripts/process/science.py: the science-reduction driver.

Covers the driver's own surface: arg parsing, the
``_cli_task`` argv it fans out, and the ``main`` exit-code contract (nonzero iff
at least one frame failed). The shared fan-out engine lives in ``_dispatch`` and
is tested in test_script_helpers.py. Stubs only -- no real testdata needed.
"""

import logging
import os
import sys
import tempfile

import pandas as pd
import pytest

from scripts.process import science as _science

from ._scripts import _FakeConfig, _NoLogDirConfig

# scripts/CLI/tools-layer suite: excluded from `make test-fast`.
pytestmark = pytest.mark.cli

_OID1 = "KP.20240405.40113.57"
_OID2 = "KP.20240405.40237.36"


@pytest.fixture(scope="module")
def s():
    return _science


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
    def test_obs_ids(self, s):
        # Expansion details are tested on resolve_obs_ids (test_script_helpers.py).
        ns = s.parse_args(["--obs_ids", _OID2, _OID1])
        assert ns.obs_ids == [_OID1, _OID2]  # sorted + deduped

    def test_dates_and_date_range(self, s):
        # Expansion details are tested on resolve_dates (test_script_helpers.py).
        assert s.parse_args(["--dates", "20240405"]).dates == ["20240405"]
        ns = s.parse_args(["--date_range", "20240401", "20240430"])
        assert ns.date_range == ["20240401", "20240430"] and ns.obs_ids is None

    @pytest.mark.parametrize(
        "argv",
        [[], ["--obs_ids", _OID1, "--dates", "20240405"]],
        ids=["neither", "both"],
    )
    def test_exactly_one_selection(self, s, argv):
        with pytest.raises(SystemExit):
            s.parse_args(argv)

    @pytest.mark.parametrize(
        "argv",
        [
            ["--obs_ids", "not-an-obs-id"],  # malformed obs_id
            ["--obs_ids", "20240405"],  # a datecode is not an obs_id
            ["--obs_ids", _OID1, "--jobs", "0"],  # jobs below 1
            ["--obs_ids", _OID1, "--job_timeout", "0"],  # timeout below 1
        ],
    )
    def test_invalid_args_exit(self, s, argv):
        with pytest.raises(SystemExit):
            s.parse_args(argv)

    def test_config_defaults_to_none(self, s):
        assert s.parse_args(["--obs_ids", _OID1]).config is None

    def test_rejects_recipe_override(self, s):
        # The orchestrator runs its fixed recipe; -r is not accepted.
        with pytest.raises(SystemExit):
            s.parse_args(["--obs_ids", _OID1, "-r", "/x.py"])

    def test_jobs_unset_resolves_to_cores_default(self, s):
        ns = s.parse_args(["--obs_ids", _OID1])
        assert ns.jobs == s._default_frame_jobs()

    def test_jobs_override(self, s):
        assert s.parse_args(["--obs_ids", _OID1, "--jobs", "3"]).jobs == 3

    def test_job_timeout_default_and_override(self, s):
        assert s.parse_args(["--obs_ids", _OID1]).job_timeout == 1200
        assert (
            s.parse_args(["--obs_ids", _OID1, "--job_timeout", "120"]).job_timeout
            == 120
        )

    def test_input_dir_aliases_data_input(self, s):
        ns = s.parse_args(["--obs_ids", _OID1, "--input_dir", "/in"])
        assert ns.kpf_data_input == "/in"

    def test_output_dir_fans_out(self, s):
        # science has masters + science output and a log dir (no plot dir).
        ns = s.parse_args(["--obs_ids", _OID1, "--output_dir", "/out"])
        assert ns.kpf_masters_output == "/out"
        assert ns.kpf_science_output == "/out"
        assert ns.log_dir == "/out/logs"


# ---------------------------------------------------------------------------
# _cli_task
# ---------------------------------------------------------------------------


class TestCliTask:
    def test_builds_science_argv_with_defaults(self, s):
        # No -c override: the science recipe and default config are passed explicitly.
        tag, argv = s._cli_task(_OID1, ["--log_level", "DEBUG"])
        assert tag == _OID1
        assert argv == [
            sys.executable, "-m", "scripts.process.reduce",
            "-r", s.DEFAULT_SCIENCE_RECIPE, "-c", s.DEFAULT_SCIENCE_CONFIG,
            "-o", _OID1, "--log_level", "DEBUG",
        ]  # fmt: skip

    def test_config_override_keeps_default_recipe(self, s):
        _, argv = s._cli_task(_OID1, [], config="/c.toml")
        assert argv == [
            sys.executable, "-m", "scripts.process.reduce",
            "-r", s.DEFAULT_SCIENCE_RECIPE, "-c", "/c.toml", "-o", _OID1,
        ]  # fmt: skip


# ---------------------------------------------------------------------------
# main exit-code contract
# ---------------------------------------------------------------------------


def _fake_file_handler(rows=()):
    """A FileHandler stand-in whose mini database holds `rows`, each an
    ``(obs_id, isjunk, green, red[, obstype])`` tuple (obstype default Target)."""
    mini_db = pd.DataFrame(
        [
            (f"/in/L0/20240405/{o}.fits", j, g, r, *(t or ["Target"]))
            for o, j, g, r, *t in rows
        ],
        columns=["FILENAME", "ISJUNK", "GREEN", "RED", "DRP_OBSTYPE"],
    )

    class _FakeFileHandler:
        def __init__(self, data_dirs):
            pass

        def build_mini_database(self, datecode, cache=False):
            return mini_db

    return _FakeFileHandler


class TestMainExitCode:
    def _patch(self, s, monkeypatch, failed, calls=None):
        # Skip the runtime setup, config resolution and subprocess fan-out (and
        # any real batch log file). ``calls`` collects run_stage's arguments for
        # the fan-out contract tests below; the rest assert only its failure set
        # -> exit code.
        monkeypatch.setattr(s, "configure_runtime", lambda: None)
        monkeypatch.setattr(s, "ConfigHandler", _FakeConfig)
        _stub_batch_log(monkeypatch, s)
        monkeypatch.setattr(s, "warm_mini_db_caches", lambda *a, **k: (0, 0))
        monkeypatch.setattr(s, "FileHandler", _fake_file_handler())

        def fake_run_stage(*args, **kwargs):
            if calls is not None:
                calls.append((args, kwargs))
            return set(failed)

        monkeypatch.setattr(s, "run_stage", fake_run_stage)

    def test_derives_datecodes_for_prescan(self, s, monkeypatch):
        # The pre-scan gets the unique-sorted nights the frames span, not the raw
        # obs_ids: two 20240405 frames collapse to one datecode.
        warm_args = {}
        self._patch(s, monkeypatch, failed=[])
        monkeypatch.setattr(
            s,
            "warm_mini_db_caches",
            lambda di, dcs, jobs, cache="rw": (
                warm_args.update(data_input=di, datecodes=dcs, cache=cache) or (0, 0)
            ),
        )
        s.main(["--obs_ids", _OID1, _OID2])
        assert warm_args["data_input"] == "/in"  # resolved from config DATA_DIRS
        assert warm_args["datecodes"] == ["20240405"]
        assert warm_args["cache"] == "rw"

    def test_junk_and_single_ccd_frames_are_skipped(self, s, monkeypatch, caplog):
        # Junk (INFO) and single-CCD (WARNING) frames never reach the fan-out; a
        # GREEN or RED of None counts as not read out.
        calls = []
        self._patch(s, monkeypatch, failed=[], calls=calls)
        oid3, oid4 = "KP.20240405.40300.00", "KP.20240405.40400.00"
        monkeypatch.setattr(
            s,
            "FileHandler",
            _fake_file_handler(
                [
                    (_OID1, True, True, True),
                    (_OID2, False, True, False),
                    (oid3, False, None, True),
                    (oid4, False, True, True),
                ]
            ),
        )
        with caplog.at_level(logging.INFO, logger=s.logger.name):
            s.main(["--obs_ids", _OID1, _OID2, oid3, oid4])
        tasks = calls[0][0][1]
        assert [tag for tag, _ in tasks] == [oid4]
        levels = {r.getMessage(): r.levelno for r in caplog.records}
        assert levels[f"skipping {_OID1}: junk frame"] == logging.INFO
        single = "DRP does not yet support single-CCD exposures"
        assert levels[f"skipping {_OID2}: {single}"] == logging.WARNING
        assert levels[f"skipping {oid3}: {single}"] == logging.WARNING
        assert "done: reduced 1/1 frame(s), 3 skipped" in levels

    def test_exits_zero_when_all_reduced(self, s, monkeypatch):
        self._patch(s, monkeypatch, failed=[])
        s.main(["--obs_ids", _OID1])  # no SystemExit

    def test_exits_nonzero_when_any_failed(self, s, monkeypatch):
        self._patch(s, monkeypatch, failed=[_OID1])
        with pytest.raises(SystemExit) as exc:
            s.main(["--obs_ids", _OID1])
        assert exc.value.code == 1

    def test_fan_out_is_fail_soft(self, s, monkeypatch):
        # This driver reduces a whole night: one bad frame must not abort the
        # rest. Its sibling drivers assert the same contract for themselves.
        calls = []
        self._patch(s, monkeypatch, failed=[], calls=calls)
        s.main(["--obs_ids", _OID1, _OID2])
        assert calls[0][1]["abort_on_failure"] is False

    def test_fan_out_paces_launches(self, s, monkeypatch):
        # Every reduction resolves catalog astrometry, so the launches are spaced
        # to stay inside the SIMBAD/Gaia rate limits.
        calls = []
        self._patch(s, monkeypatch, failed=[], calls=calls)
        s.main(["--obs_ids", _OID1, "--job_timeout", "42"])
        assert calls[0][1]["launch_interval"] == s._LAUNCH_INTERVAL > 0
        assert calls[0][1]["job_timeout"] == 42

    def test_output_dir_reaches_the_per_frame_argv(self, s, monkeypatch):
        # Parsing --kpf_science_output is not enough: dropping it from `forward`
        # sends every product to the config's data root with a green suite.
        calls = []
        self._patch(s, monkeypatch, failed=[], calls=calls)
        s.main(["--obs_ids", _OID1, "--kpf_science_output", "/out"])
        tasks = calls[0][0][1]
        _, argv = tasks[0]
        assert "--kpf_science_output" in argv
        assert argv[argv.index("--kpf_science_output") + 1] == "/out"

    def test_stages_reach_every_child(self, s, monkeypatch):
        calls = []
        self._patch(s, monkeypatch, failed=[], calls=calls)
        s.main(["--obs_ids", _OID1, "--stages", "astro_query", "image_assembly"])
        for _, argv in calls[0][0][1]:
            assert argv[-3:] == ["--stages", "astro_query", "image_assembly"]

    def test_no_stages_forwarded_by_default(self, s, monkeypatch):
        calls = []
        self._patch(s, monkeypatch, failed=[], calls=calls)
        s.main(["--obs_ids", _OID1])
        _, argv = calls[0][0][1][0]
        assert "--stages" not in argv

    def test_forwards_dir_and_log_overrides_to_each_child(self, s, monkeypatch):
        # The sibling of masters' forwarding test: every override the orchestrator
        # resolved must reach each fanned-out reduce, or a child silently falls
        # back to its own config.
        calls = []
        self._patch(s, monkeypatch, failed=[], calls=calls)
        s.main(
            [
                "--obs_ids",
                _OID1,
                "--input_dir",
                "/in",
                "--output_dir",
                "/out",
                "--log_level",
                "DEBUG",
            ]
        )
        _, argv = calls[0][0][1][0]
        fwd = {argv[i]: argv[i + 1] for i in range(len(argv) - 1)}
        assert fwd["--kpf_data_input"] == "/in"
        assert fwd["--kpf_science_output"] == "/out"
        assert fwd["--log_level"] == "DEBUG"
        # Both halves of the run directory, so the child joins the same one.
        assert fwd["--log_dir"] == "/out/logs"
        assert fwd["--run_id"] == "science_x"

    def test_dates_reduce_only_a_nights_target_frames(self, s, monkeypatch, caplog):
        # A night expands to its frames; calibrations drop silently, junk is still
        # gated and counted as skipped.
        calls, seen = [], {}
        self._patch(s, monkeypatch, failed=[], calls=calls)
        oid3 = "KP.20240405.40300.00"

        def fake_nights(data_input, nights):
            seen.update(data_input=data_input, nights=nights)
            return [_OID1, _OID2, oid3]

        monkeypatch.setattr(s, "obs_ids_in_nights", fake_nights)
        monkeypatch.setattr(
            s,
            "FileHandler",
            _fake_file_handler(
                [
                    (_OID1, False, True, True),
                    (_OID2, False, True, True, "Bias"),
                    (oid3, True, True, True),
                ]
            ),
        )
        with caplog.at_level(logging.INFO, logger=s.logger.name):
            s.main(["--dates", "20240405"])
        assert seen == {"data_input": "/in", "nights": ["20240405"]}
        assert [tag for tag, _ in calls[0][0][1]] == [_OID1]
        assert "done: reduced 1/1 frame(s), 1 skipped" in caplog.text

    def test_date_range_expands_to_the_nights_on_disk(self, s, monkeypatch):
        calls, seen = [], {}
        self._patch(s, monkeypatch, failed=[], calls=calls)
        monkeypatch.setattr(
            s, "datecodes_in_range", lambda di, a, b: seen.update(r=(a, b)) or [a]
        )
        monkeypatch.setattr(s, "obs_ids_in_nights", lambda di, nights: [_OID1])
        monkeypatch.setattr(
            s, "FileHandler", _fake_file_handler([(_OID1, False, True, True)])
        )
        s.main(["--date_range", "20240405", "20240406"])
        assert seen["r"] == ("20240405", "20240406")
        assert [tag for tag, _ in calls[0][0][1]] == [_OID1]

    def test_dates_without_target_frames_warn_and_exit(self, s, monkeypatch, caplog):
        self._patch(s, monkeypatch, failed=[])
        monkeypatch.setattr(s, "obs_ids_in_nights", lambda di, nights: [_OID1])
        monkeypatch.setattr(
            s, "FileHandler", _fake_file_handler([(_OID1, False, True, True, "Dark")])
        )
        with caplog.at_level(logging.WARNING, logger=s.logger.name):
            with pytest.raises(SystemExit) as exc:
                s.main(["--dates", "20240405"])
        assert "no Target frames" in str(exc.value.code)
        assert "skipping 20240405: no Target frames" in caplog.text

    def test_errors_when_log_dir_unset(self, s, monkeypatch):
        # A missing log_dir is fatal before any fan-out.
        monkeypatch.setattr(s, "configure_runtime", lambda: None)
        monkeypatch.setattr(s, "ConfigHandler", _NoLogDirConfig)
        with pytest.raises(SystemExit) as exc:
            s.main(["--obs_ids", _OID1])
        assert "log directory" in str(exc.value)
