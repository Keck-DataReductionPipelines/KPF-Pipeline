"""Tests for the pipeline logging setup (kpfpipe/utils/logger.py)."""

import logging
import re
import sys
import time
import warnings

import pytest

from kpfpipe.utils import logger as kpflog
from kpfpipe.utils.config import ConfigHandler
from scripts.processing.reduce import resolve_logging

_UT_FROZEN = time.struct_time((2026, 7, 2, 14, 3, 22, 2, 183, 0))


@pytest.fixture(autouse=True)
def _teardown():
    """Always tear down after each test.

    Critical for the captureWarnings bridge: pytest wraps every test in
    ``catch_warnings``, so ``captureWarnings(False)`` must run inside the
    same test context or ``logging._warnings_showwarning`` goes stale and a
    later ``captureWarnings(True)`` silently no-ops.
    """
    yield
    kpflog.teardown_logging()


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


class TestGetLevel:
    def test_maps_names_any_case(self):
        assert kpflog.get_level("debug") == logging.DEBUG
        assert kpflog.get_level("INFO") == logging.INFO
        assert kpflog.get_level("Warning") == logging.WARNING

    def test_unknown_level_raises(self):
        with pytest.raises(ValueError, match="unknown log level"):
            kpflog.get_level("chatty")


class TestBuildLogPath:
    def test_layout_and_ut_components(self, tmp_path):
        path = kpflog.build_log_path(
            str(tmp_path), "science", "KP.20240923.33129.48", start_time=_UT_FROZEN
        )
        fn = "kpf_science_KP.20240923.33129.48_20260702T140322.log"
        assert path == str(tmp_path / "20260702" / fn)

    def test_empty_log_dir_raises(self):
        with pytest.raises(ValueError, match="log_dir"):
            kpflog.build_log_path("", "science", "x")


class TestSetupLogging:
    def test_creates_file_and_returns_path(self, tmp_path):
        path = kpflog.setup_logging(str(tmp_path), "science", "KP.1.2.3")
        datecode = time.strftime("%Y%m%d", time.gmtime())
        assert path.startswith(str(tmp_path / datecode))
        assert re.search(r"kpf_science_KP\.1\.2\.3_\d{8}T\d{6}\.log$", path)
        assert _read(path) == ""  # created, empty until a record arrives

    def test_record_format(self, tmp_path):
        path = kpflog.setup_logging(str(tmp_path), "science", "t", console=False)
        logging.getLogger("kpfpipe.x").info("hello world")
        line = _read(path).splitlines()[0]
        assert re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z "
            r"INFO\s+kpfpipe\.x: hello world",
            line,
        )

    def test_ut_timestamps(self, tmp_path):
        # Assert the converter rather than comparing wall clocks (TZ-robust).
        kpflog.setup_logging(str(tmp_path), "science", "t", console=False)
        handlers = logging.getLogger().handlers
        (handler,) = [h for h in handlers if h.name == "kpfpipe_file"]
        assert handler.formatter.converter is time.gmtime

    def test_level_filtering(self, tmp_path):
        path = kpflog.setup_logging(
            str(tmp_path), "science", "t", level="INFO", console=False
        )
        logging.getLogger("kpfpipe.x").debug("too quiet")
        logging.getLogger("kpfpipe.x").info("loud enough")
        text = _read(path)
        assert "too quiet" not in text
        assert "loud enough" in text

    def test_unique_filename_on_collision(self, tmp_path, monkeypatch):
        monkeypatch.setattr(kpflog.time, "gmtime", lambda *a: _UT_FROZEN)
        first = kpflog.setup_logging(str(tmp_path), "science", "t", console=False)
        second = kpflog.setup_logging(str(tmp_path), "science", "t", console=False)
        assert second == f"{first}.1"

    def test_repeated_setup_no_duplicate_handlers(self, tmp_path):
        root = logging.getLogger()
        before = len(root.handlers)
        first = kpflog.setup_logging(str(tmp_path), "science", "t", console=False)
        kpflog.setup_logging(str(tmp_path), "masters", "20240923", console=False)
        assert len(root.handlers) == before + 1  # one file handler, never two
        (handler,) = [h for h in root.handlers if h.name == "kpfpipe_file"]
        assert handler.baseFilename != first  # the survivor is the second file

    def test_console_handler_optional(self, tmp_path):
        root = logging.getLogger()
        kpflog.setup_logging(str(tmp_path), "science", "t", console=True)
        assert any(h.name == "kpfpipe_console" for h in root.handlers)
        kpflog.setup_logging(str(tmp_path), "science", "t", console=False)
        assert not any(h.name == "kpfpipe_console" for h in root.handlers)

    def test_console_defaults_to_stderr(self, tmp_path):
        # The leaf runner's console echo stays on stderr (stream=None default).
        kpflog.setup_logging(str(tmp_path), "science", "t", console=True)
        (console,) = [
            h for h in logging.getLogger().handlers if h.name == "kpfpipe_console"
        ]
        assert console.stream is sys.stderr

    def test_third_party_pins(self, tmp_path):
        kpflog.setup_logging(str(tmp_path), "science", "t", console=False)
        assert logging.getLogger("matplotlib").level == logging.WARNING
        assert logging.getLogger("PIL").level == logging.WARNING

    def test_bad_level_raises_before_side_effects(self, tmp_path):
        with pytest.raises(ValueError, match="unknown log level"):
            kpflog.setup_logging(str(tmp_path), "science", "t", level="chatty")
        assert list(tmp_path.iterdir()) == []


class TestSetupBatchLogging:
    """The batch-orchestrator sibling: a per-invocation ``_batch_`` log echoed
    to stdout so an operator can watch fan-out progress live."""

    def test_creates_batch_file_and_returns_path(self, tmp_path):
        path = kpflog.setup_batch_logging(str(tmp_path), "masters")
        datecode = time.strftime("%Y%m%d", time.gmtime())
        assert path.startswith(str(tmp_path / datecode))
        assert re.search(r"kpf_masters_batch_\d{8}T\d{6}\.log$", path)
        assert _read(path) == ""  # created, empty until a record arrives

    def test_module_logger_record_lands_in_file(self, tmp_path):
        path = kpflog.setup_batch_logging(str(tmp_path), "masters", console=False)
        logging.getLogger("scripts.processing.masters").info("dispatching 3 job(s)")
        assert "dispatching 3 job(s)" in _read(path)

    def test_console_echo_is_stdout(self, tmp_path):
        # The live-echo contract: batch progress mirrors to stdout (not stderr).
        kpflog.setup_batch_logging(str(tmp_path), "science")
        (console,) = [
            h for h in logging.getLogger().handlers if h.name == "kpfpipe_console"
        ]
        assert console.stream is sys.stdout

    def test_filter_on_console_only_not_file(self, tmp_path):
        # The batch console carries _BatchConsoleFilter; the log file stays raw.
        kpflog.setup_batch_logging(str(tmp_path), "science")
        root = logging.getLogger()
        (console,) = [h for h in root.handlers if h.name == "kpfpipe_console"]
        (file_h,) = [h for h in root.handlers if h.name == "kpfpipe_file"]
        assert any(
            isinstance(flt, kpflog._BatchConsoleFilter) for flt in console.filters
        )
        assert not file_h.filters

    def test_library_info_off_console_but_kept_in_file(self, tmp_path, capsys):
        # The whole point: library INFO chatter is trimmed from the live terminal
        # echo, while the driver's own narration and any WARNING still show -- and
        # the batch log file keeps every record regardless.
        # A neutral third-party name (not astropy, whose logger sets
        # propagate=False and so never reaches the root handlers).
        path = kpflog.setup_batch_logging(str(tmp_path), "masters")
        logging.getLogger("scripts.processing.masters").info("driver narration")
        logging.getLogger("thirdparty.io").info("library chatter")
        logging.getLogger("thirdparty.io").warning("library warning")

        out = capsys.readouterr().out
        assert "driver narration" in out  # scripts.* INFO echoed
        assert "library chatter" not in out  # library INFO trimmed from terminal
        assert "library warning" in out  # WARNING always echoed
        text = _read(path)
        assert {"driver narration", "library chatter", "library warning"} <= {
            line.split(": ", 1)[-1].strip() for line in text.splitlines()
        }  # ...but the file is unfiltered


class TestBatchConsoleFilter:
    """The batch stdout echo trims sub-WARNING records to the driver's own
    ``scripts.*``/``__main__`` sources; WARNING and above always pass."""

    _flt = kpflog._BatchConsoleFilter()

    def _rec(self, name, level):
        return logging.LogRecord(name, level, __file__, 1, "m", None, None)

    def test_info_from_scripts_passes(self):
        assert self._flt.filter(self._rec("scripts.processing.masters", logging.INFO))
        assert self._flt.filter(
            self._rec("scripts.plots.plot_timeseries", logging.INFO)
        )

    def test_info_from_main_passes(self):
        # A driver launched as `python -m scripts.processing.<name>` logs as __main__.
        assert self._flt.filter(self._rec("__main__", logging.INFO))

    def test_info_from_library_dropped(self):
        assert not self._flt.filter(self._rec("astropy.io.fits", logging.INFO))
        assert not self._flt.filter(self._rec("kpfpipe.utils.io", logging.INFO))
        assert not self._flt.filter(self._rec("py.warnings", logging.INFO))

    def test_debug_is_source_filtered_too(self):
        # Below WARNING covers DEBUG as well, not just INFO.
        assert not self._flt.filter(self._rec("astropy", logging.DEBUG))
        assert self._flt.filter(self._rec("scripts.processing.science", logging.DEBUG))

    def test_warning_and_above_always_pass(self):
        # Per-unit failures and warnings from anywhere are never hidden.
        assert self._flt.filter(self._rec("astropy.io.fits", logging.WARNING))
        assert self._flt.filter(self._rec("py.warnings", logging.WARNING))
        assert self._flt.filter(self._rec("kpfpipe.x", logging.ERROR))


class TestTeardown:
    def test_teardown_removes_handlers_and_restores_showwarning(self, tmp_path):
        root = logging.getLogger()
        before_handlers = list(root.handlers)
        before_showwarning = warnings.showwarning
        kpflog.setup_logging(str(tmp_path), "science", "t")
        kpflog.teardown_logging()
        assert root.handlers == before_handlers
        assert warnings.showwarning is before_showwarning

    def test_teardown_restores_root_level(self, tmp_path):
        root = logging.getLogger()
        before = root.level
        kpflog.setup_logging(str(tmp_path), "science", "t", level="DEBUG")
        kpflog.teardown_logging()
        assert root.level == before

    def test_teardown_safe_when_nothing_installed(self):
        kpflog.teardown_logging()  # must not raise


class TestIOChokepoints:
    """Every FITS write/read emits one INFO record (DRP-RUN-08)."""

    def test_to_fits_and_from_fits_log_records(self, tmp_path, caplog):
        from kpfpipe.data_models.level4 import KPF4

        path = str(tmp_path / "rt_l4.fits")
        with caplog.at_level(logging.INFO, logger="kpfpipe"):
            KPF4().to_fits(path)
            KPF4.from_fits(path)
        assert f"wrote KPF4 to {path}" in caplog.text
        assert f"reading KPF4 from {path}" in caplog.text


def _config(tmp_path, body):
    path = tmp_path / "config.toml"
    path.write_text(body)
    return path


class TestResolveLogging:
    _LOGGER_TOML = '[LOGGER]\nlog_dir = "/logs/"\nlog_level = "DEBUG"\n'

    def test_resolves_config_and_tokens(self, tmp_path):
        config = ConfigHandler(_config(tmp_path, self._LOGGER_TOML))
        params = resolve_logging(
            config, "/repo/recipes/kpf_drp_science.py", "KP.1.2.3", None
        )
        assert params == {
            "log_dir": "/logs/",
            "recipe_name": "science",
            "target": "KP.1.2.3",
            "level": "DEBUG",
            "console": True,
        }

    def test_datecode_target_and_defaults(self, tmp_path):
        config = ConfigHandler(_config(tmp_path, '[LOGGER]\nlog_dir = "/l/"\n'))
        params = resolve_logging(config, "kpf_drp_masters.py", None, "20240923")
        assert params["recipe_name"] == "masters"
        assert params["target"] == "20240923"
        assert params["level"] == "INFO"  # default when config omits log_level

    def test_no_target_falls_back_to_run(self, tmp_path):
        config = ConfigHandler(_config(tmp_path, '[LOGGER]\nlog_dir = "/l/"\n'))
        assert resolve_logging(config, "custom.py", None, None)["target"] == "run"

    def test_missing_log_dir_raises(self, tmp_path):
        config = ConfigHandler(_config(tmp_path, "[LOGGER]\n"))
        with pytest.raises(ValueError, match="no log directory configured"):
            resolve_logging(config, "kpf_drp_science.py", "KP.1.2.3", None)

    def test_cli_override_wins(self, tmp_path):
        # The CLI passes --log_dir/--log_level through ConfigHandler overrides.
        config = ConfigHandler(
            _config(tmp_path, self._LOGGER_TOML),
            overrides={"LOGGER": {"log_dir": "/elsewhere/", "log_level": "INFO"}},
        )
        params = resolve_logging(config, "kpf_drp_science.py", "KP.1.2.3", None)
        assert params["log_dir"] == "/elsewhere/"
        assert params["level"] == "INFO"


class TestWarningsBridge:
    @pytest.mark.filterwarnings("always")
    def test_capturewarnings_lands_in_file_with_source(self, tmp_path):
        path = kpflog.setup_logging(str(tmp_path), "science", "t", console=False)
        warnings.warn("bridged boom", stacklevel=1)  # 1: attribute to THIS line
        text = _read(path)
        assert "WARNING  py.warnings:" in text
        assert "bridged boom" in text
        assert "test_logger.py" in text  # file:lineno source identification

    @pytest.mark.filterwarnings("always")
    def test_pytest_warns_coexistence(self, tmp_path):
        # Inside pytest.warns the recorder wins: the assertion passes and the
        # record does NOT reach the log (captureWarnings only swaps
        # showwarning; pytest.warns uses catch_warnings(record=True)).
        path = kpflog.setup_logging(str(tmp_path), "science", "t", console=False)
        with pytest.warns(UserWarning, match="recorded boom"):
            warnings.warn("recorded boom", stacklevel=2)
        assert "recorded boom" not in _read(path)
